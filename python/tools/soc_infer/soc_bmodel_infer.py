#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
# this script should only run on SOC

import argparse
import pickle
import os
import time
import numpy as np
import debugger
from debugger.target_1684x.memmap import *
from debugger.target_1684x.soc import BM1684XRunner
from debugger.target_common.op_support import *
from debugger.atomic_dialect import BModel2MLIR
from debugger.disassembler import BModel
from debugger.lowering import lowering
from debugger.final_mlir import Pickled_Value
from npz_file import IncNpzFile
from tqdm import tqdm
from collections import Counter
import warnings

# ignore all warning of"Duplicate name", this comes when args.using_memory_opt=True.
warnings.filterwarnings("ignore", category=UserWarning, message="Duplicate name")


npz_in_memory = {}
log_txt = ""


class soc_launch_struct:
    def __init__(self, tiu_num, dma_num, tiu_buf, dma_buf):
        self.tiu_num = tiu_num
        self.dma_num = dma_num
        self.tiu_buf = tiu_buf
        self.dma_buf = dma_buf
        self.tiu_buf_len = len(tiu_buf)
        self.dma_buf_len = len(dma_buf)


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "debugger.target_1684x.pcie" and name == "soc_launch_struct":
            return soc_launch_struct
        return super().find_class(module, name)


def set_inputs_dict(atomic_mlir, inputs, memory):
    args = atomic_mlir.functions[0].signature[0]
    for id, arg in enumerate(args):  # type: List[int, tensor_cls]
        input = lowering(
            inputs[arg.name],
            pdtype=arg.dtype.name,
            pshape=arg.shape[0],
            pzero_point=arg.zero_point,
            pscale=arg.scale,
        )
        mem = args[id].memref
        memory.set_data(mem, input)


def get_ref_data(operand: Pickled_Value, ref):
    if operand.name not in ref:
        return None
    reshape = operand.reshape
    ref_data = ref[operand.name]

    if reshape:
        reshape = eval(reshape[1:-1].replace("x", ","))  # '1,3,1,192,1024'
        ref_data = ref_data.reshape(reshape)

    _slice = operand.slice
    data = eval(f"ref_data{_slice}")  # type: np.ndarray
    # The data in HW has a transposed collapsed shape.
    # To align the Bmodel with TPU.mlir, we need to transpose the reference data.
    if operand.layout in (
        "continuous_group3d",
        "eu_align_group3d",
        "compact_group3d",
        "eu_align_xn_group3d",
        "compact_xn_group3d",
    ):
        n, c, d, h, w = 0, 1, 2, 3, 4
        data = data.transpose((d, n, c, h, w))
    return data


def finished_cur_layer(_slice, target_shape):
    if _slice == "[...]":
        return True

    slice_list = _slice[1:-1].split(",")
    for idx, slice in enumerate(slice_list):
        if int(slice.split(":")[1]) != target_shape[idx]:
            return False
    return True


def collect_with_mem_opt(operand, origin_shape, reshape, _slice, slices, actual, npz_in_disk):
    if operand.name not in npz_in_memory:
        tmp = np.zeros(reshape, dtype=actual.dtype)
    else:
        tmp = npz_in_memory[operand.name].astype(actual.dtype)
        tmp = tmp.reshape(reshape)

    tmp[tuple(slices)] = actual
    tmp = tmp.reshape(origin_shape)
    if finished_cur_layer(_slice, reshape):
        npz_in_disk[operand.name] = tmp
        if operand.name in npz_in_memory:
            npz_in_memory.pop(operand.name)
    else:
        npz_in_memory[operand.name] = tmp


def collect_without_mem_opt(operand, origin_shape, reshape, slices, actual, npz_in_disk):
    if operand.name not in npz_in_disk:
        tmp = np.zeros(reshape, dtype=actual.dtype)
    else:
        tmp = npz_in_disk[operand.name]
        tmp = tmp.reshape(reshape)
    tmp[tuple(slices)] = actual
    npz_in_disk[operand.name] = tmp.reshape(origin_shape)


def collect_infer_data_from_ref(operand: Pickled_Value, actual, desired, ref, npz_in_disk, use_memory_opt):
    if operand.name not in ref:
        return
    if use_memory_opt and operand.name in npz_in_disk:
        return

    _slice = operand.slice
    if _slice == "[...]":
        slice_list = operand.memory_type[1:-1].replace("x", ",").split(",")[:-1]
        sliced_shape = tuple(int(i) for i in slice_list)
        slices = [slice(None, None) for _ in slice_list]
    else:
        slice_list = _slice[1:-1].split(",")
        sliced_shape = tuple(
            [
                int(slice.split(":")[1]) - int(slice.split(":")[0])
                for slice in slice_list
            ]
        )
        slices = [
            slice(int(s.strip().split(":")[0]), int(s.strip().split(":")[1]))
            for s in slice_list
        ]
    actual = actual.reshape(desired.shape)

    if operand.layout in (
        "continuous_group3d",
        "eu_align_group3d",
        "compact_group3d",
        "eu_align_xn_group3d",
        "compact_xn_group3d",
    ):
        d, n, c, h, w = 0, 1, 2, 3, 4
        actual = actual.transpose((n, c, d, h, w))

    reshape = operand.reshape
    origin_shape = ref[operand.name].shape
    if reshape:
        reshape = eval(reshape[1:-1].replace("x", ","))
    else:
        reshape = sliced_shape

    if use_memory_opt:
        collect_with_mem_opt(operand, origin_shape, reshape, _slice, slices, actual, npz_in_disk)
    else:
        collect_without_mem_opt(operand, origin_shape, reshape, slices, actual, npz_in_disk)


def infer_combine(
    value,
    soc_runner,
    ref_data,
    infer_data,
    out_fixed,
    using_memory_opt,
    is_operand=True,
    enable_log=False,
):
    global log_txt
    memref = value.memref
    raw_data = soc_runner.memory.get_data(memref)
    if out_fixed == True:
        actual = raw_data.astype(np.float32)
    else:
        actual = (raw_data.astype(np.float32) - value.zero_point) * value.scale
    desired = get_ref_data(value, ref_data)
    collect_infer_data_from_ref(value, actual, desired, ref_data, infer_data, using_memory_opt)
    if enable_log:
        if is_operand:
            log_txt += f"gather operand slice: {value}\n"
        else:
            log_txt += f"gather result slice: {value}\n"


def main():
    parser = argparse.ArgumentParser(description="Soc bmodel infer combine")
    parser.add_argument(
        "--path",
        default="/tmp",
        help="The folder should contain the BModel, its input_data, and reference files.",
    )
    parser.add_argument(
        "--bmodel",
        required=True,
        help="Bmodel file path",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input file path",
    )
    parser.add_argument(
        "--ref",
        required=True,
        help="Ref file path, only support .npz",
    )
    parser.add_argument(
        "--tool_path",
        default="/tmp",
        help="The folder where place the soc_infer dir.",
    )
    parser.add_argument(
        "--out_fixed",
        action="store_true",
        help="Whether to output fixed number.",
    )
    parser.add_argument(
        "--enable_log", action="store_true", help="Whether to enable log."
    )
    parser.add_argument(
        "--using_memory_opt",
        action="store_true",
        help="Whether to enable memory opt, which decrease memory usage but increase time cost.",
    )
    parser.add_argument(
        "--run_by_atomic",
        action="store_true",
        help="Whether to run by atomic cmds, instead of running by ops as default.",
    )
    args = parser.parse_args()
    return args


def log_start(idx, run_by_atomic=False, enable_log=False):
    if not enable_log:
        return
    global log_txt
    if run_by_atomic:
        log_type = "Atomic CMD"
    else:
        log_type = "OP"
    log_txt += f"############################### {log_type} {idx} Begin ###############################\n"


def log_end(idx, run_by_atomic=False, enable_log=False):
    if not enable_log:
        return
    global log_txt
    if run_by_atomic:
        log_type = "Atomic CMD"
    else:
        log_type = "OP"
    log_txt += f"################################ {log_type} {idx} End ################################\n"


def collect_before_compute(
    values_in_pkl, in_value_idx, soc_runner, ref_data, infer_data, args
):
    in_values = values_in_pkl[in_value_idx]
    for value in in_values:
        infer_combine(
            value,
            soc_runner,
            ref_data,
            infer_data,
            args.out_fixed,
            args.using_memory_opt,
            is_operand=True,
            enable_log=args.enable_log,
        )


def collect_after_compute(
    values_out_pkl, out_value_idx, soc_runner, ref_data, infer_data, args
):
    out_value = values_out_pkl[out_value_idx]
    infer_combine(
        out_value,
        soc_runner,
        ref_data,
        infer_data,
        args.out_fixed,
        args.using_memory_opt,
        is_operand=False,
        enable_log=args.enable_log,
    )


if __name__ == "__main__":
    try:
        start_time = time.time()
        args = main()
        with open(os.path.join(args.path, "cmds.pkl"), "rb+") as pkl_cmds:
            cmds_pkl = CustomUnpickler(pkl_cmds).load()
        with open(os.path.join(args.path, "values_in.pkl"), "rb+") as pkl_values_in:
            values_in_pkl = CustomUnpickler(pkl_values_in).load()
        with open(os.path.join(args.path, "values_out.pkl"), "rb+") as pkl_values_out:
            values_out_pkl = CustomUnpickler(pkl_values_out).load()
        bmodel_file = os.path.join(args.path, args.bmodel)
        input_data_fn = os.path.join(args.path, args.input)
        reference_data_fn = os.path.join(args.path, args.ref)

        # init device
        bmodel = BModel(bmodel_file)
        atomic_mlir = BModel2MLIR(bmodel)
        soc_runner = BM1684XRunner(1)
        soc_runner.fast_checker = False
        coeff = atomic_mlir.functions[0].regions[0].data
        if coeff:
            address = coeff.address
            addr_offset_ddr = address - memmap[MType.G.value][0]
            # load constant data
            soc_runner.memory.set_data_to_address(
                coeff.address, np.frombuffer(coeff.data, dtype=np.uint8)
            )
        op_start_idx = 0
        in_op = False
        op_idx = 0

        # init input and reference
        if isinstance(input_data_fn, dict):
            set_inputs_dict(atomic_mlir, input_data_fn, soc_runner.memory)
        elif input_data_fn.endswith(".dat"):
            inputs = np.fromfile(input_data_fn, dtype=np.uint8)
            _offset = 0
            for arg in atomic_mlir.functions[0].signature[0]:
                mem = arg.memref
                size = int(np.prod(mem.shape) * mem.itemsize)
                soc_runner.memory.set_data(
                    mem, inputs[_offset : _offset + size].view(mem.np_dtype)
                )  #  load input tensor
                _offset += size
        elif input_data_fn.endswith(".npz"):
            inputs = np.load(input_data_fn)
            set_inputs_dict(atomic_mlir, inputs, soc_runner.memory)

        ref_data = np.load(reference_data_fn)
        dump_path = args.tool_path
        file_name = os.path.basename(reference_data_fn).split(".")[0]
        save_path = os.path.join(dump_path, f"soc_infer_{file_name}.npz")
        log_txt_path = os.path.join(dump_path, f"log.txt")
        if args.using_memory_opt:
            infer_data = IncNpzFile(save_path)
        else:
            infer_data = {}

        # compute and collect
        cmd_points = sorted(values_in_pkl.keys())
        tqdm_iter = tqdm(cmds_pkl)
        history = Counter({"tiu": 0, "dma": 0})
        total_compute_time = 0
        for idx, struct in enumerate(tqdm_iter):
            log_start(idx, args.run_by_atomic, args.enable_log)

            if not args.run_by_atomic:
                collect_before_compute(values_in_pkl, cmd_points[idx], soc_runner, ref_data, infer_data, args)
                soc_runner.checker_fast_compute(struct.tiu_num, struct.dma_num, struct.tiu_buf, struct.dma_buf)
                history.update({"tiu": struct.tiu_num, "dma": struct.dma_num})
                tqdm_iter.set_description(f"execute {history['tiu']} tiu {history['dma']} dma cmds.")
                collect_after_compute(values_out_pkl, idx, soc_runner, ref_data, infer_data, args)
            else:
                if not in_op:
                    op_start_idx = idx
                    collect_before_compute(values_in_pkl, cmd_points[op_idx], soc_runner, ref_data, infer_data, args)
                    in_op = True
                soc_runner.fast_compute(struct.tiu_num, struct.dma_num, struct.tiu_buf, struct.dma_buf)
                history.update({"tiu": struct.tiu_num, "dma": struct.dma_num})
                tqdm_iter.set_description(f"execute {history['tiu']} tiu {history['dma']} dma cmds.")
                if op_start_idx + values_out_pkl[op_idx].cmd_point-values_in_pkl[cmd_points[op_idx]][0].cmd_point == idx:
                    collect_after_compute(values_out_pkl, op_idx, soc_runner, ref_data, infer_data, args)
                    in_op = False
                    op_idx += 1

            log_end(idx, args.run_by_atomic, args.enable_log)

        os.makedirs(dump_path, exist_ok=True)
        if not args.using_memory_opt:
            np.savez(save_path, **infer_data)
        print(f"Inference combine file: {save_path} generated!")
        end_time = time.time()
        print(f"Soc_infer Time Cost on Device: {end_time-start_time} seconds")
    finally:
        with open(log_txt_path, "w+", encoding="utf-8") as log_file:
            log_file.write(log_txt)
