#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from __future__ import division
import numpy as np
import sys
import argparse
import struct
from .tensor_compare import TensorCompare, TensorCompareStats
import multiprocessing
from tqdm import tqdm


def parse_args(args_list):
    parser = argparse.ArgumentParser(description='Compare two npz tensor files.')
    parser.add_argument("target_file", help="Comparing target file")
    parser.add_argument("ref_file", help="Comparing reference file")
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument("--tolerance",
                        type=str,
                        default='0.99,0.99,0.90,50',
                        help="tolerance for cos/cor/euclid similarity/SQNR")
    parser.add_argument('--op_info',
                        type=str,
                        help="A csv file op_info, including order and dequant threshold")
    parser.add_argument("--dequant",
                        action='store_true',
                        default=False,
                        help="Do dequantization flag, use threshold table provided in --op_info")
    parser.add_argument("--excepts", type=str, help="List of tensors except from comparing")
    parser.add_argument("--full-array",
                        action='store_true',
                        default=False,
                        help="Dump full array data when comparing failed")
    parser.add_argument("--stats_int8_tensor",
                        action='store_true',
                        default=False,
                        help="Do statistics on int8 tensor for saturate ratio and low ratio")
    parser.add_argument("--int8_tensor_close",
                        type=int,
                        default=1,
                        help="whether int8 tensor compare close")
    parser.add_argument("--save", type=str, help="Save result as a csv file")
    parser.add_argument("--per_axis_compare",
                        type=int,
                        default=-1,
                        help="Compare along axis, usually along axis 1 as per-channel")
    args = parser.parse_args(args_list)
    return args


def bf16_to_fp32(d_bf16):
    s = d_bf16.shape
    d_bf16 = d_bf16.flatten()
    assert d_bf16.dtype == np.uint16
    d_fp32 = np.empty_like(d_bf16, dtype=np.float32)
    for i in range(len(d_bf16)):
        d_fp32[i] = struct.unpack('<f', struct.pack('<HH', 0, d_bf16[i]))[0]
    return d_fp32.reshape(s)


def fp32_to_bf16(d_fp32):
    s = d_fp32.shape
    d_fp32 = d_fp32.flatten()
    assert d_fp32.dtype == np.float32
    d_bf16 = np.empty_like(d_fp32, dtype=np.uint16)
    for i in range(len(d_bf16)):
        bytes = struct.pack('f', d_fp32[i])
        d_bf16[i] = struct.unpack('<H', struct.pack('BB', bytes[2], bytes[3]))[0]
    return d_bf16.reshape(s)


def align_type_and_shape(d1, d2):
    try:
        d2 = d2.reshape(d1.shape)
    except:
        print("WARRING: Two narraies are not the same shape." + \
              " {} v.s. {}, check if continue:".format(d1.shape, d2.shape))
        # check if do-early-stride case
        if d2.shape[2] % d1.shape[2] == 0 and \
           d2.shape[3] % d1.shape[3] == 0:
            sh = int(d2.shape[2] / d1.shape[2])
            sw = int(d2.shape[3] / d1.shape[3])
            d2 = d2[:, :, ::sh, ::sw]
            print("Ignore this warning, continue")
        else:
            raise RuntimeError("Fatal, stop")

    t1 = d1.dtype
    t2 = d2.dtype
    # print(t1, d1.size, d1.shape)
    # print(t2, d2.size, d2.shape)
    if t1 == np.int8 or t2 == np.int8:
        t = np.int8
    else:
        t = np.float32
    if t1 == np.uint16:
        d1 = bf16_to_fp32(d1)
    if t2 == np.uint16:
        d2 = bf16_to_fp32(d2)

    d1 = d1.astype(t)
    d2 = d2.astype(t)
    return d1, d2


def dequantize(d1, threshold):
    scale = threshold / 127.0
    d1 = d1 * scale
    return d1


def compare_one_array(tc, npz1, npz2, name, verbose, lock, dic, int8_tensor_close,
                      per_axis_compare):
    lock.acquire()
    d1 = npz1[name]
    d2 = npz2[name]
    lock.release()
    try:
        d1, d2 = align_type_and_shape(d1, d2)
    except:
        print("Error: {} in two npz file is not same shape. {} v.s. {}".format(
            name, d1.shape, d2.shape))
        result = (False, tc.NOT_MATCH, {}, None)
        dic[name] = result
        return result
    result = tc.compare(d1, d2, verbose, int8_tensor_close, per_axis_compare)
    dic[name] = result
    return result


def print_result_one_array(tc, npz1, name, dic, verbose, per_axis_compare):
    d1 = npz1[name]
    tc.print_result(d1, name, dic[name], verbose, per_axis_compare)


def npz_compare(args_list):
    lock = multiprocessing.Lock()
    dic = multiprocessing.Manager().dict()
    args = parse_args(args_list)
    f1 = args.target_file
    f2 = args.ref_file

    np.set_printoptions(precision=6)
    np.set_printoptions(suppress=True)
    if args.full_array:
        np.set_printoptions(threshold=sys.maxsize)
    if args.tolerance:
        tolerance = [float(s) for s in args.tolerance.split(',')]
    excepts = []
    if args.excepts:
        excepts = [str(s) for s in args.excepts.split(',')]

    ordered_names = []
    operations = {}
    quant_types = {}

    int8_tensor_close = args.int8_tensor_close
    npz1 = np.load(f1)
    npz2 = np.load(f2)

    tc = TensorCompare(close_order_tol=3,
                       cosine_similarity_tol=tolerance[0],
                       euclidean_similarity_tol=tolerance[1],
                       signal_to_quantization_noise_tol=float('-inf'),
                       per_axis_compare=args.per_axis_compare)

    common = list()
    for name in npz1.files:
        if name in npz2.files and name not in excepts:
            common.append(name)
    if ordered_names:
        names = []
        for name in ordered_names:
            if name in common:
                names.append(name)
    else:
        names = common

    stats = TensorCompareStats()

    names_list = list(names)  # deep copy
    process_number = multiprocessing.cpu_count() // 2 + 1
    if args.per_axis_compare >= 0:
        process_number = 1

    pbar = tqdm(names, total=len(names), position=0, leave=True)
    while (len(names_list) > 0):
        compare_process_name_list = names_list[:process_number]
        names_list = names_list[process_number:]  # remove done name
        # take process number names
        # take name which will do compare
        processes = []
        for name in compare_process_name_list:
            pbar.set_description("compare {}".format(name))
            pbar.update(1)
            p = multiprocessing.Process(target=compare_one_array,
                                        args=(tc, npz1, npz2, name, args.verbose, lock, dic,
                                              int8_tensor_close, args.per_axis_compare))
            processes.append(p)
            p.start()

        for j in processes:
            j.join()

    for name in names:
        if dic.get(name) == None:
            continue
        stats.update(name, dic.get(name))
        print_result_one_array(tc, npz1, name, dic, args.verbose, args.per_axis_compare)

    stats.print_result()
    if (args.save):
        stats.save_result(args.save, operations, quant_types)
        print("Results save as {}".format(args.save))
    print("Target    {}".format(f1))
    print("Reference {}".format(f2))
    if (stats.failed == 0):
        print("npz compare PASSED.", flush=True)
        return stats
    else:
        print("npz compare FAILED.", flush=True)
        sys.exit(-1)


if __name__ == '__main__':
    npz_compare(sys.argv)
