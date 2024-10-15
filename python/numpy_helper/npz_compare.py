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
import gc


def parse_args(args_list):
    # yapf: disable
    parser = argparse.ArgumentParser(description='Compare two npz tensor files.')
    parser.add_argument("target_file", help="Comparing target file")
    parser.add_argument("ref_file", help="Comparing reference file")
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help="print full message")
    parser.add_argument("--tolerance", type=str, default='0.99,0.99',
                        help="tolerance for cos/euclid similarity")
    parser.add_argument("--excepts", type=str, help="List of tensors except from comparing")
    parser.add_argument("--full-array", action='store_true',
                        help="Dump full array data when comparing failed")
    parser.add_argument("--stats_int8_tensor", action='store_true',
                        help="Do statistics on int8 tensor for saturate ratio and low ratio")
    parser.add_argument("--int8_tensor_close", type=int, default=1,
                        help="whether int8 tensor compare close")
    parser.add_argument("--save", type=str, help="Save result as a csv file")
    parser.add_argument("--per_axis_compare", type=int, default=-1,
                        help="Compare along axis, usually along axis 1 as per-channel")
    parser.add_argument("--fuzzy_match", action='store_true',
                        help="fuzzy_match")
    parser.add_argument("--forall", '-a', action='store_true',
                        help="whether print all the arrays or not.")
    args = parser.parse_args(args_list)
    # yapf: enable
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


def crop_array(data, shape):
    slices = [slice(0, dim) for dim in shape]
    return data[tuple(slices)]


def align_type_and_shape(d1, d2):
    try:
        d2 = d2.reshape(d1.shape)
    except:
        print("WARRING: not the same shape " + " {} v.s. {}".format(d1.shape, d2.shape))
        s1 = np.array(d1.shape)
        s2 = np.array(d2.shape)
        if s1.size == s2.size and (np.all(s1 >= s2) or np.all(s1 <= s2)):
            if np.all(s1 >= s2):
                d1 = crop_array(d1, s2)
            else:
                d2 = crop_array(d2, s1)
        else:
            raise RuntimeError("Fatal, stop")

    t1 = d1.dtype
    t2 = d2.dtype
    # print(t1, d1.size, d1.shape)
    # print(t2, d2.size, d2.shape)
    if t1 == np.int8 or t2 == np.int8:
        t = np.int8
    elif t1 == np.int32 or t2 == np.int32:
        t = np.int32
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


def compare_one_array(tc : TensorCompare, npz1, npz2, name, verbose, lock, dic, int8_tensor_close,
                      per_axis_compare):
    lock.acquire()
    d1 = npz1.get(name)
    d2 = npz2.get(name)
    lock.release()
    try:
        # dirty hack for NonMaxSuppression
        # onnx and bmodel can get correct shape, but top/tpu always get largest shape
        # so we need to align shape
        if "NonMaxSuppression" in name:
            min_v = min(d1.shape[0], d2.shape[0])
            d2 = d2[:min_v, :]
            d1 = d1[:min_v, :]
        d1, d2 = align_type_and_shape(d1, d2)
    except:
        print("Error: {} in two npz file is not same shape. {} v.s. {}".format(
            name, d1.shape, d2.shape))
        result = (False, tc.NOT_MATCH, 0, {}, None)
        dic[name] = result
        return result
    result = tc.compare(d1, d2, verbose, int8_tensor_close, per_axis_compare)
    dic[name] = result
    return result


def print_result_one_array(tc, npz1, name, dic, verbose, per_axis_compare):
    d1 = npz1[name]
    tc.print_result(d1, name, dic[name], verbose, per_axis_compare)


def npz_compare(args_list, log_level="normal"):
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
    # excepts.append("Y_Index_TopK")
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
    if args.fuzzy_match:
        min_cos, min_euc = 1, 1
        for name in npz1.files:
            max_cos, max_euc = 0.0, 0.0
            for name2 in npz2.files:
                d1 = npz1.get(name)
                d2 = npz2.get(name2)
                if d1.shape == d2.shape:
                    d1, d2 = align_type_and_shape(d1, d2)
                    result = tc.compare(d1, d2, args.verbose, int8_tensor_close, args.per_axis_compare)
                    if result[1] == 'EQUAL':
                        if log_level == "normal":
                            print(f'find EQUAL for {name}')
                        break
                    elif result[3]['cosine'] > max_cos:
                        max_cos = result[3]['cosine']
                        max_euc = result[3]['euclid']
            if log_level == "normal":
                print(f'find max_cos:{max_cos}, max_euc:{max_euc} for {name}')
            if min_cos > max_cos:
                min_cos = max_cos
            if min_euc > max_euc:
                min_euc = max_euc
        if min_cos < tolerance[0] or min_euc < tolerance[1]:
            # print("npz compare FAILED.", flush=True)
            # sys.exit(-1)
            return
        else:
            if log_level == "normal":
                print("npz compare PASSED.", flush=True)
            return

    common = list()
    for name in npz2.files:
        if name in npz1.files and name not in excepts:
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
    if not args.forall and len(names_list) > 200:
        step = len(names_list) // 200
        if step > 1:
            names_list = names_list[::step]

        for i in range(1, 10):
            if names[-i] not in names_list:
                # last n value may be outputs, these comparison is very important
                names_list.append(names[-1])
    process_number = min(multiprocessing.cpu_count(), 8)
    if args.per_axis_compare >= 0:
        process_number = 1

    pbar = tqdm(names, total=len(names_list), position=0, leave=True, disable=(log_level != "normal"))
    while (len(names_list) > 0):
        compare_process_name_list = names_list[:process_number]
        names_list = names_list[process_number:]  # remove done name
        # take process number names
        # take name which will do compare
        processes = []
        for name in compare_process_name_list:
            if log_level == "normal":
                pbar.set_description("compare {}".format(name))
                pbar.update(1)
            p = multiprocessing.Process(target=compare_one_array,
                                        args=(tc, npz1, npz2, name, args.verbose, lock, dic,
                                              int8_tensor_close, args.per_axis_compare))
            processes.append(p)
            p.start()

        for j in processes:
            j.join()
        gc.collect()

    print('\n', flush=True)

    for name in names:
        if dic.get(name) is None:
            continue
        stats.update(name, dic.get(name))
        if log_level == "normal":
            print_result_one_array(tc, npz1, name, dic, args.verbose, args.per_axis_compare)

    if log_level == "normal":
        stats.print_result()

    if args.save:
        stats.save_result(args.save, operations, quant_types)
        print("Results saved as {}".format(args.save))

    if log_level != "quiet":
        print("Target    {}".format(f1))
        print("Reference {}".format(f2))
        if stats.failed == 0:
            print("npz compare PASSED.", flush=True)
            return stats
        else:
            print("npz compare FAILED.", flush=True)
            sys.exit(-1)



if __name__ == '__main__':
    npz_compare(sys.argv)
