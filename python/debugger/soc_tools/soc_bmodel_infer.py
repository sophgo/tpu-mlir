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

os.environ["TDB_CACHE_MODE"] = "auto"

import debugger
from debugger.target_common.op_support import *
from debugger.atomic_dialect import BModel2MLIR
from debugger.disassembler import BModel
from debugger.lowering import lowering
from debugger.final_mlir import Pickled_Value
from debugger.target_common.op_support import ValueRef
from debugger.target_1684x.soc import BM1684XRunner as BM1684XSoc
from debugger.target_1684x.context import BM1684XContext
from debugger.target_1688.device_rt import BM1688Runner as BM1688Soc
from debugger.target_1688.context import BM1688Context
from npz_file import IncNpzFile
from tqdm import tqdm
from collections import Counter
import warnings

# ignore all warning of"Duplicate name", this comes when args.using_memory_opt=True.
warnings.filterwarnings("ignore", category=UserWarning, message="Duplicate name")

npz_in_memory = {}
log_txt = ""


class soc_launch_struct:

    def __init__(self, tiu_num, dma_num, tiu_buf, dma_buf, core_ids=set({0})):
        self.tiu_num = tiu_num
        self.dma_num = dma_num
        self.tiu_buf = tiu_buf
        self.dma_buf = dma_buf
        self.tiu_buf_len = len(tiu_buf)
        self.dma_buf_len = len(dma_buf)
        self.core_ids = core_ids


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
        assert memory.set_data(mem, input)


def get_ref_data(operand: Pickled_Value, ref):
    if operand.tlvalue.name not in ref:
        return None
    reshape = operand.tlvalue.reshape
    ref_data = ref[operand.tlvalue.name]

    if reshape:
        reshape = eval(reshape[1:-1].replace("x", ","))  # '1,3,1,192,1024'
        ref_data = ref_data.reshape(reshape)

    _slice = operand.tlvalue.slice
    data = eval(f"ref_data{_slice}")  # type: np.ndarray
    # The data in HW has a transposed collapsed shape.
    # To align the Bmodel with TPU.mlir, we need to transpose the reference data.
    if operand.tlvalue.layout in (
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
    if operand.tlvalue.name not in npz_in_memory:
        tmp = np.zeros(reshape, dtype=actual.dtype)
    else:
        tmp = npz_in_memory[operand.tlvalue.name].astype(actual.dtype)
        tmp = tmp.reshape(reshape)

    tmp[tuple(slices)] = actual
    tmp = tmp.reshape(origin_shape)
    if finished_cur_layer(_slice, reshape):
        npz_in_disk[operand.tlvalue.name] = tmp
        if operand.tlvalue.name in npz_in_memory:
            npz_in_memory.pop(operand.tlvalue.name)
    else:
        npz_in_memory[operand.tlvalue.name] = tmp


def collect_without_mem_opt(operand, origin_shape, reshape, slices, actual, npz_in_disk):
    if operand.tlvalue.name not in npz_in_disk:
        tmp = np.zeros(reshape, dtype=actual.dtype)
    else:
        tmp = npz_in_disk[operand.tlvalue.name]
        tmp = tmp.reshape(reshape)
    tmp[tuple(slices)] = actual
    npz_in_disk[operand.tlvalue.name] = tmp.reshape(origin_shape)


def collect_infer_data_from_ref(operand: Pickled_Value, actual, desired, ref, npz_in_disk,
                                use_memory_opt, desire_op):
    if desire_op and operand.tlvalue.name not in desire_op:
        return
    if operand.tlvalue.name not in ref:
        return
    if use_memory_opt and operand.tlvalue.name in npz_in_disk:
        return

    _slice = operand.tlvalue.slice
    if _slice == "[...]":
        slice_list = operand.tlvalue.memory_type[1:-1].replace("x", ",").split(",")[:-1]
        sliced_shape = tuple(int(i) for i in slice_list)
        slices = [slice(None, None) for _ in slice_list]
    else:
        slice_list = _slice[1:-1].split(",")
        sliced_shape = tuple(
            [int(slice.split(":")[1]) - int(slice.split(":")[0]) for slice in slice_list])
        slices = [
            slice(int(s.strip().split(":")[0]), int(s.strip().split(":")[1])) for s in slice_list
        ]
    actual = actual.reshape(desired.shape)

    if operand.tlvalue.layout in (
            "continuous_group3d",
            "eu_align_group3d",
            "compact_group3d",
            "eu_align_xn_group3d",
            "compact_xn_group3d",
    ):
        d, n, c, h, w = 0, 1, 2, 3, 4
        actual = actual.transpose((n, c, d, h, w))

    reshape = operand.tlvalue.reshape
    origin_shape = ref[operand.tlvalue.name].shape
    if reshape:
        reshape = eval(reshape[1:-1].replace("x", ","))
    else:
        reshape = sliced_shape

    if use_memory_opt:
        collect_with_mem_opt(operand, origin_shape, reshape, _slice, slices, actual, npz_in_disk)
    else:
        collect_without_mem_opt(operand, origin_shape, reshape, slices, actual, npz_in_disk)


def infer_combine(
    value,  # Pickled_value
    context,
    soc_runner,
    ref_data,
    infer_data,
    out_fixed,
    using_memory_opt,
    is_operand=True,
    enable_log=False,
    desire_op=[],
):
    global log_txt
    memref = value.tlvalue.get_memref(context)
    # memref=value.memref
    # raw_data = context.memory.get_data(ValueRef(memref, core_id=value.core_id))
    raw_data = soc_runner.memory.get_data(ValueRef(memref, core_id=value.core_id))
    if out_fixed == True:
        actual = raw_data.astype(np.float32)
    else:
        actual = (raw_data.astype(np.float32) - value.tlvalue.zero_point) * value.tlvalue.scale

    desired = get_ref_data(value, ref_data)
    collect_infer_data_from_ref(value, actual, desired, ref_data, infer_data, using_memory_opt,
                                desire_op)
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
    parser.add_argument("--enable_log", action="store_true", help="Whether to enable log.")
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
    parser.add_argument(
        "--desire_op",
        type=str,
        default="",
        help="Whether to only dump specific ops, dump all ops as default.",
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


def collect_before_compute(values_in_pkl, in_value_idx, soc_runner, ref_data, infer_data, args):
    in_values = values_in_pkl[in_value_idx]
    for value in in_values:
        infer_combine(
            value,
            context,
            soc_runner,
            ref_data,
            infer_data,
            args.out_fixed,
            args.using_memory_opt,
            is_operand=True,
            enable_log=args.enable_log,
            desire_op=[desire_op.strip()
                       for desire_op in args.desire_op.split(",")] if args.desire_op != "" else [],
        )


def collect_after_compute(values_out_pkl, out_value_idx, soc_runner, ref_data, infer_data, args):
    out_value = values_out_pkl[out_value_idx]
    infer_combine(
        out_value,
        context,
        soc_runner,
        ref_data,
        infer_data,
        args.out_fixed,
        args.using_memory_opt,
        is_operand=False,
        enable_log=args.enable_log,
        desire_op=[desire_op.strip()
                   for desire_op in args.desire_op.split(",")] if args.desire_op != "" else [],
    )


if __name__ == "__main__":
    try:
        start_time = time.time()
        args = main()
        with open(os.path.join(args.path, "cmds.pkl"), "rb+") as pkl_cmds:
            cmds_pkl = CustomUnpickler(pkl_cmds).load()
        with open(os.path.join(args.path, "values_in.pkl"), "rb+") as pkl_values_in:
            values_in_pkl = pickle.load(pkl_values_in)
        with open(os.path.join(args.path, "values_out.pkl"), "rb+") as pkl_values_out:
            values_out_pkl = pickle.load(pkl_values_out)
        bmodel_file = os.path.join(args.path, args.bmodel)
        input_data_fn = os.path.join(args.path, args.input)
        reference_data_fn = os.path.join(args.path, args.ref)

        # init device
        bmodel = BModel(bmodel_file)
        atomic_mlir = BModel2MLIR(bmodel)
        context = bmodel.context
        soc_runner = context.get_runner(1)
        soc_runner.fast_checker = False
        coeff = atomic_mlir.functions[0].regions[0].data
        context.memory.set_neuron_size(bmodel.neuron_size)
        if hasattr(coeff, "data"):
            coeff_data = np.frombuffer(getattr(coeff, "data"), dtype=np.uint8)
            context.memory.set_coeff_size(coeff_data.size)

        if coeff:
            address = coeff.address
            if isinstance(context, BM1688Context):
                address = context.fix_addr(address)
            soc_runner.memory.set_data_to_address(coeff.address, coeff_data)

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
                assert soc_runner.memory.set_data(mem, inputs[_offset:_offset + size].view(
                    mem.np_dtype))  #  load input tensor
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

            if isinstance(soc_runner, BM1684XSoc):
                if not args.run_by_atomic:
                    collect_before_compute(values_in_pkl, cmd_points[idx], soc_runner, ref_data,
                                           infer_data, args)
                    soc_runner.checker_fast_compute(struct.tiu_num, struct.dma_num, struct.tiu_buf,
                                                    struct.dma_buf)
                    history.update({"tiu": struct.tiu_num, "dma": struct.dma_num})
                    tqdm_iter.set_description(
                        f"execute {history['tiu']} tiu {history['dma']} dma cmds.")
                    collect_after_compute(values_out_pkl, idx, soc_runner, ref_data, infer_data,
                                          args)
                else:
                    if not in_op:
                        op_start_idx = idx
                        collect_before_compute(values_in_pkl, cmd_points[op_idx], soc_runner,
                                               ref_data, infer_data, args)
                        in_op = True
                    soc_runner.fast_compute(struct.tiu_num, struct.dma_num, struct.tiu_buf,
                                            struct.dma_buf)
                    history.update({"tiu": struct.tiu_num, "dma": struct.dma_num})
                    tqdm_iter.set_description(
                        f"execute {history['tiu']} tiu {history['dma']} dma cmds.")
                    if op_start_idx + values_out_pkl[op_idx].cmd_point - values_in_pkl[
                            cmd_points[op_idx]][0].cmd_point == idx:
                        collect_after_compute(values_out_pkl, op_idx, soc_runner, ref_data,
                                              infer_data, args)
                        in_op = False
                        op_idx += 1

            elif isinstance(soc_runner, BM1688Soc):
                if not in_op:
                    op_start_idx = idx
                    collect_before_compute(values_in_pkl, cmd_points[op_idx], soc_runner, ref_data,
                                           infer_data, args)
                    in_op = True
                soc_runner._cmds_soc_single_compute(struct.tiu_num, struct.dma_num, struct.tiu_buf,
                                                    struct.dma_buf, struct.core_ids)
                history.update({"tiu": struct.tiu_num, "dma": struct.dma_num})
                tqdm_iter.set_description(
                    f"execute {history['tiu']} tiu {history['dma']} dma cmds.")
                if op_start_idx + values_out_pkl[op_idx].cmd_point - values_in_pkl[
                        cmd_points[op_idx]][0].cmd_point == idx:
                    collect_after_compute(values_out_pkl, op_idx, soc_runner, ref_data, infer_data,
                                          args)
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
