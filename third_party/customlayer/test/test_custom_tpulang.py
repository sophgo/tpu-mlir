#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import json
import numpy as np
import os
import transform.TpuLang as tpul
from utils.auto_remove import file_mark, file_clean, clean_kmp_files
import my_tpulang_layer

def coeff_tensor(shape, dtype, scale=1.0):
    data = rand_data(shape, dtype)
    data = data * scale if dtype == 'float32' else data
    return tpul.Tensor(dtype=dtype, shape=shape, data=data)

def rand_data(shape, dtype):
    if dtype == 'float32':
        return np.random.random(shape).astype(np.float32)
    if dtype == 'int32' or 'uint32' or 'int16' or 'uint16' or 'int8' or 'uint8':
        return np.random.randint(0, 256, size=shape).astype(dtype)
    raise Exception("Not supported data type: {}!".format(dtype))

class CUSTOM_TPULANG_TESTER(object):
    # This class is built for testing single operator transform.
    def __init__(self,
                 chip: str = "bm1684x",
                 mode: str = "all",
                 dynamic: bool = False,
                 simple: bool = False,
                 disable_thread: bool = False,
                 num_core: int = 1,
                 op_type: str = "all"):
        Y, N = True, False
        # yapf: disable
        self.test_tpu_cases = {
            #########################################
            # CUSTOM TPULANG Test Case, Alphabetically
            #########################################
            # case: (test, bm1684x_support, bm1688_support)
            "AbsAdd":          (self.test_AbsAdd,       Y, Y),
            "CeilAdd":         (self.test_CeilAdd,      Y, Y),
            "Crop":            (self.test_Crop,         Y, Y),
            "SimpleBlock":     (self.test_SimpleBlock,  Y, Y),
            "SwapChannel":     (self.test_SwapChannel,  Y, Y),
        }

        self.test_cpu_cases = {
            "CpuTopk":         (self.test_CpuTopk,      Y, Y),
            "CpuTopk2":        (self.test_CpuTopk2,     Y, Y),
        }
        # yapf: enable
        if op_type == "all":
            self.test_cases = {**self.test_tpu_cases, **self.test_cpu_cases}
        elif op_type == "tpu":
            self.test_cases = self.test_tpu_cases
        else:
            self.test_cases = self.test_cpu_cases

        # no quantization when quant_mode == "f32"
        self.support_quant_modes = ["f32", "f16", "bf16"]
        self.support_asym = [False]
        self.model_file = ".bmodel"
        self.chip = chip.lower()
        self.dynamic = dynamic
        self.simple = simple
        self.multithread = not disable_thread
        self.num_core = num_core
        self.mode = mode.lower()
        if self.simple:
            self.support_quant_modes = ["f16"]
            self.support_asym = [False]
        if self.mode == "" or self.mode == "all":
            self.quant_modes = self.support_quant_modes
        else:
            if self.chip == "bm1688":
                self.support_quant_modes.append("int4")
            if self.mode not in self.support_quant_modes:
                raise RuntimeError("{} not support mode: {}".format(self.chip, self.mode))
            self.quant_modes = [self.mode]
        # initialize tpulang
        tpul.init(self.chip.upper())

    def test_single(self, case: str):
        np.random.seed(0)
        print("Test: {}".format(case))
        if case in self.test_cases:
            os.makedirs(case, exist_ok=True)
            os.chdir(case)
            func = self.test_cases[case][0]
            func()
            print("====== TEST {} Success ======".format(case))
        else:
            raise RuntimeError("case [{}] is not exist".format(case))

    def check_support(self, case):
        _, bm1684x_support, bm1688_support = self.test_cases[
            case]
        if self.chip == "bm1684x" and bm1684x_support:
            return True
        if self.chip == "bm1688" and bm1688_support:
            return True
        return False

    def make_test_calibration_table(self, tensors, table_name, qmode=None):
        # simple calibration table
        with open(table_name, 'w') as f:
            for name in tensors:
                flatten_tensor = tensors[name].flatten()
                max_val = max(flatten_tensor)
                min_val = min(flatten_tensor)
                if max_val == min_val:
                    max_val = max_val + 0.01
                t = 1.1 * max(abs(min_val), abs(max_val)) + 0.01
                f.write("{} {} {} {}\n".format(name, t, min_val, max_val))

    def compile_and_check(self, model_name, graph_ins, graph_outs, ref_data):
        # compile to Top mlir file, the input will be saved in {top_mlir}_in_f32.npz
        model_name = model_name + ("_dyn" if self.dynamic else "_static")
        for mode in self.quant_modes:
            tpul.compile_f32(model_name, graph_ins, graph_outs, mode=mode, refs=ref_data, dynamic=self.dynamic)

    ##################################
    # adding your operators here
    ##################################

    def test_AbsAdd(self):
        # 1. prepare the input
        dtype = "float32"
        input_shape = [2, 2, 2, 2]
        x_data = rand_data(input_shape, dtype)
        x = tpul.Tensor(name="in", dtype=dtype, shape=input_shape, data=x_data)
        ins = [x]
        # 2. build model
        b = 1.2
        outs = my_tpulang_layer.absAdd.tpulang(inputs=[x], b=float(b), dtype=dtype)
        # 3. save the origin output for comparison
        out_data = my_tpulang_layer.absAdd.native(x_data, b)
        # There are two outputs because in non-f32 quant mode, the result will be
        # dequant back to f32 with castOp so that the final result will be named with the suffix '_f32'

        ref_data = {outs[0].name: out_data, f"{outs[0].name}_f32": out_data}
        self.compile_and_check("absadd", ins, outs, ref_data)

    def test_CeilAdd(self):
        # 1. prepare the input
        dtype = "float32"
        input_shape = [10, 3, 14, 14]
        x_data = rand_data(input_shape, dtype)
        x = tpul.Tensor(name="in", dtype=dtype, shape=input_shape, data=x_data)
        ins = [x]
        # 2. build model
        b = 1.2
        outs = my_tpulang_layer.ceilAdd.tpulang(inputs=[x], b=float(b), dtype=dtype)
        # 3. save the origin output for comparison
        out_data = my_tpulang_layer.ceilAdd.native(x_data, b)
        # There are two outputs because in non-f32 quant mode, the result will be
        # dequant back to f32 with castOp so that the final result will be named with the suffix '_f32'
        ref_data = {outs[0].name: out_data, f"{outs[0].name}_f32": out_data}
        self.compile_and_check("ceiladd", ins, outs, ref_data)

    def test_SwapChannel(self):
        # 1. prepare the input
        dtype = "float32"
        input_shape = [10, 3, 14, 14]
        x_data = rand_data(input_shape, dtype)
        x = tpul.Tensor(name="in", dtype=dtype, shape=input_shape, data=x_data)
        ins = [x]
        # 2. build model
        outs = my_tpulang_layer.swapChannel.tpulang(inputs=[x], dtype=dtype)
        # 3. save the origin output for comparison
        out_data = my_tpulang_layer.swapChannel.native(x_data)
        # There are two outputs because in non-f32 quant mode, the result will be
        # dequant back to f32 with castOp so that the final result will be named with the suffix '_f32'
        ref_data = {outs[0].name: out_data, f"{outs[0].name}_f32": out_data}
        self.compile_and_check("swap_channel", ins, outs, ref_data)

    def test_Crop(self):
        # 1. prepare the input
        dtype = "float32"
        input_shape = [4, 32, 6, 7]
        x_data = rand_data(input_shape, dtype)
        x = tpul.Tensor(name="in", dtype=dtype, shape=input_shape, data=x_data)
        ins = [x]
        # 2. build model
        hoffset = 1; woffset = 2; hnew = 3; wnew = 4
        outs = my_tpulang_layer.crop.tpulang(ins, hoffset, woffset, hnew, wnew, dtype=dtype)
        # 3. save the origin output for comparison
        out_data = my_tpulang_layer.crop.native(x_data, hoffset, woffset, hnew, wnew)
        # There are two outputs because in non-f32 quant mode, the result will be
        # dequant back to f32 with castOp so that the final result will be named with the suffix '_f32'
        ref_data = {outs[0].name: out_data, f"{outs[0].name}_f32": out_data}
        self.compile_and_check("crop", ins, outs, ref_data)

    def test_CpuTopk(self):
        self.quant_modes = ["f32"]
        dtype = "float32"
        input_shape = [1, 1000]
        x_data = rand_data(input_shape, dtype)
        x = tpul.Tensor(name="in", dtype=dtype, shape=input_shape, data=x_data)
        ins = [x]
        axis, k = 1, 200
        outs = my_tpulang_layer.cpuTopk.tpulang(inputs=[x], axis=axis, k=k, dtype=dtype)
        out_data = my_tpulang_layer.cpuTopk.native(x_data, axis, k)
        ref_data = {outs[0].name: out_data, f"{outs[0].name}_f32": out_data}
        self.compile_and_check("cpuTopk", ins, outs, ref_data)

    def test_CpuTopk2(self):
        self.quant_modes = ["f32"]
        dtype = "float32"
        axis, k = 1, 200
        def relu_op(x):
            x = tpul.relu(x[0])
            return[x]

        def model_def(x, flag: int):
            if flag == 0:
                x = my_tpulang_layer.cpuTopk.tpulang(inputs=x, axis=axis, k=k, dtype=dtype)
                x = relu_op(x)
            elif flag == 1:
                x = relu_op(x)
                x = my_tpulang_layer.cpuTopk.tpulang(inputs=x, axis=axis, k=k, dtype=dtype)
            else:
                x = relu_op(x)
                x = my_tpulang_layer.cpuTopk.tpulang(inputs=x, axis=axis, k=k, dtype=dtype)
                x = relu_op(x)
            return x

        input_shape = [1, 1000]
        x_data = rand_data(input_shape, dtype)
        x = tpul.Tensor(name="in", dtype=dtype, shape=input_shape, data=x_data)
        x = [x]
        ins = x

        for flag in [0, 1, 2]:
            y = model_def(x, flag)
            out_data = my_tpulang_layer.cpuTopk.native(x_data, axis, k)
            out_data = np.maximum(0, out_data)
            ref_data = {y[0].name: out_data, f"{y[0].name}_f32": out_data}
            self.compile_and_check("cpuTopk2", ins, y, ref_data)

    # test case for a simple model with customlayers
    def test_SimpleBlock(self):
        def conv_op(x,
                    kshape,
                    stride,
                    pad=None,
                    group=1,
                    dilation=[1, 1],
                    zp=[None, None],
                    dtype="float32"):
            oc = kshape[0]
            weight = coeff_tensor(kshape, dtype)
            out_dtype = dtype if dtype == 'float32' else 'int32'
            bias = coeff_tensor(oc, out_dtype)
            conv = tpul.conv(x,
                            weight,
                            bias=bias,
                            stride=stride,
                            pad=pad,
                            dilation=dilation,
                            group=group,
                            out_dtype=out_dtype)
            return conv

        def abs_add_op(x, b, dtype="float32"):
            return my_tpulang_layer.absAdd.tpulang([x], b, dtype)[0]

        def model_def(x, flag: int):
            if flag & 0b1:
                x = abs_add_op(x, 1.2)
            conv1 = conv_op(x, [4, 32, 3, 3], [1, 1], [1, 1, 1, 1])
            relu1 = tpul.relu(conv1)
            conv2 = conv_op(relu1, [4, 4, 3, 3], [2, 2], [2, 2, 2, 2])
            if flag & 0b10:
                conv2 = abs_add_op(conv2, -1.5)
            relu2 = tpul.relu(conv2)
            conv3 = conv_op(relu2, [4, 4, 3, 3], [1, 1], [1, 1, 1, 1])
            relu3 = tpul.relu(conv3)
            if flag & 0b100:
                relu3 = abs_add_op(relu3, 2.1)
            return relu3

        def gen_name(flag: int):
            return "front" if flag == 0b1 else "middle" if flag == 0b10 else "back" if flag == 0b100 else "mix"

        def compile_model(flag: int, dynamic: bool):
            shape = [4, 32, 64, 48]
            x_data = np.random.random(shape).astype(np.float32)
            x = tpul.Tensor(dtype='float32', shape=shape, data=x_data)
            y = model_def(x, flag)
            postfix = "dyn" if dynamic else "static"
            tpul.compile_f32("model_def_{}_{}".format(gen_name(flag), postfix), [x], [y], dynamic=dynamic)

        for flag in (0b1, 0b10, 0b100, 0b111):
            # tpul.init(self.chip.upper())
            print("--- model_def_{} ---".format(gen_name(flag)))
            compile_model(flag, self.dynamic)
            # tpul.deinit()

def test_one_case_in_all(tester: CUSTOM_TPULANG_TESTER, case, error_cases, success_cases):
    try:
        tester.test_single(case)
    except:
        error_cases.append(case)
        return
    success_cases.append(case)

def test_all_base(tester: CUSTOM_TPULANG_TESTER):
    if tester.multithread:
        import multiprocessing
        from utils.misc import collect_process
        process_number = multiprocessing.cpu_count() // 2 + 1
        processes = []
        error_cases = multiprocessing.Manager().list()
        success_cases = multiprocessing.Manager().list()
        for case in tester.test_cases:
            if tester.check_support(case):
                print("====== test_custom_tpulang.py --case {} --chip {} TEST START PROCESSING ======".format(
                    case, tester.chip))
                p = multiprocessing.Process(target=test_one_case_in_all,
                                            name=case,
                                            args=(tester, case, error_cases, success_cases))
                processes.append(p)
            if len(processes) == process_number:
                collect_process(processes, error_cases)
                processes = []
        collect_process(processes, error_cases)
        processes = []
    else:
        error_cases = []
        success_cases = []
        for case in tester.test_cases:
            if tester.check_support(case):
                test_one_case_in_all(tester, case, error_cases, success_cases)
    print("Success: {}".format(success_cases))
    print("Failure: {}".format(error_cases))
    if error_cases:
        print("====== test_custom_tpulang.py --chip {} TEST Failed ======".format(tester.chip))
        # exit(1)
    else:
        print("====== test_custom_tpulang.py --chip {} TEST Success ======".format(tester.chip))
    clean_kmp_files()
    return error_cases, success_cases

def test_all(tester: CUSTOM_TPULANG_TESTER):
    f, s = test_all_base(tester)
    return f

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--chip", default="bm1684x", type=str,
                        choices=['bm1684x', 'bm1688'],
                        help="chip platform name")
    parser.add_argument("--case", default="all", type=str, help="test one case, if all, then test all cases")
    parser.add_argument("--mode", default="all", type=str, choices=['all', 'f32', 'f16', 'bf16'],
                        help="quantize modes")
    parser.add_argument("--op_type", default="all", type=str, choices=['all', 'tpu', 'cpu'], help='test tpu or cpu custom op')
    parser.add_argument("--dynamic", action="store_true", help='do dynamic compile')
    parser.add_argument("--debug", action="store_true", help='keep middle file if debug')
    parser.add_argument("--simple", action="store_true", help='do simple test for commit test')
    parser.add_argument("--num_core", default=1, type=int, help='The numer of TPU cores used for parallel computation')
    parser.add_argument("--disable_thread", action="store_true", help='do test without multi thread')
    parser.add_argument("--show_all", action="store_true", help='show all cases')
    parser.add_argument("--report", default="", type=str, help="report file name")

    # yapf: enable
    args = parser.parse_args()
    tester = CUSTOM_TPULANG_TESTER(args.chip, args.mode, args.dynamic, args.simple, args.disable_thread,
                                   args.num_core, args.op_type)
    if args.show_all:
        print("====== Show All Cases ============")
        for case in tester.test_cases:
            print(case)
        exit(0)
    dir = "custom_tpulang_test_{}".format(args.chip)
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)
    if args.case == "" or args.case.lower() == "all":
        if (args.report):
            f, s = test_all_base(tester)
        else:
            test_all(tester)
    else:
        tester.test_single(args.case)
    if not args.debug:
        file_clean()
    if args.report:
        result = {'succsess': list(s), 'failure': list(f)}
        with open(args.report, "w") as f:
            json.dump(result, f)
