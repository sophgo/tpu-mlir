#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import torch
from tools.train.FxGraphConverter import fx2mlir
from torch.fx._symbolic_trace import symbolic_trace
import argparse
import numpy as np
from utils.timer import Timer
from utils.auto_remove import file_mark, file_clean, clean_kmp_files
import sys
import os
import torch.nn as nn
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
import torch.nn.functional as F
from utils.regression_logger import run_in_log_wrapper

class FX_IR_TESTER(object):
    ID = 0
    CURRENT_CASE = ""
    def __init__(
        self, chip: str,
        concise_log: bool = False,
        disable_thread: bool = False,
        disable_cmp: bool = False):
        Y, N = True, False
        # yapf: disable
        self.test_cases = {
            #########################################
            # FX Test Case, Alphabetically
            #########################################
            # case: (test, bm1684x_support, bm1688_support, bm1690_support)
            "Convolution":              (self.test_Conv,                    N, N, Y),
            "Convbackward":             (self.test_Conv_backward,           N, N, Y),
            "batchnormbwd":             (self.test_batchnormbwd,            N, N, N),
            "batchnormfwd":             (self.test_batchnormfwd,            N, N, Y),
            "maxpoolwithmask":          (self.test_maxpoolwithmask,         N, N, N),
            "maxpoolwithmask_bwd":      (self.test_maxpoolwithmask_bwd,     N, N, N),
            "maxpoolwithmask_full":     (self.test_maxpoolwithmask_full,    N, N, Y),
            "where_batchnormbwd":       (self.test_where_batchnormbwd,      N, N, N),
        }
        # yapf: enable
        self.chip = chip
        self.concise_log = concise_log
        self.multithread = not disable_thread
        self.cmp = not disable_cmp

    def convert_module_fx(
        self,
        submodule_name: str,
        module: torch.fx.GraphModule,
    ):
        c = fx2mlir(
            submodule_name=submodule_name,
            chip=self.chip,
            model=submodule_name,
            bwd_graph=False,
            para_shape=[],
            cmp=self.cmp)
        return c.convert(module)

    def generate_random(self, shape, dtype='float32', min=-1, max=1):
        scale = max - min
        return (np.random.rand(*shape) * scale + min).astype(dtype)

    def create_random_input(self, shapes, descs):
        if len(descs) == 0:
            inputs = [self.generate_random(s) for s in shapes]
        else:
            inputs = list()
            for i in range(len(shapes)):
                inputs.append(
                    self.generate_random(shapes[i], descs[i].dtype, descs[i].min, descs[i].max))
        return [torch.from_numpy(inp) for inp in inputs]

    @run_in_log_wrapper
    def test_single(self, case: str):
        np.random.seed(0)
        torch.manual_seed(7)
        FX_IR_TESTER.ID = 0
        FX_IR_TESTER.CURRENT_CASE = case
        print("Test: {}".format(case))
        if case in self.test_cases:
            os.makedirs(case, exist_ok=True)
            os.chdir(case)
            func, _, _, _ = self.test_cases[case]
            func()
            print("====== TEST {} Success ======".format(case))
        else:
            raise RuntimeError("case [{}] is not exist".format(case))

    def check_support(self, case):
        _, bm1684x_support, bm1688_support, bm1690_support = self.test_cases[
            case]
        if self.chip == "bm1684x" and bm1684x_support:
            return True
        if self.chip == "bm1688" and bm1688_support:
            return True
        if self.chip == "bm1690" and bm1690_support:
            return True
        return False

    def trace_and_test(self,in_shapes, torch_model: nn.Module, descs = [], use_cos: bool = False, input_info = None):
        model_name = "{}_{}".format(self.CURRENT_CASE, FX_IR_TESTER.ID)
        FX_IR_TESTER.ID += 1
        inputs = self.create_random_input(in_shapes, descs)
        fx_module = symbolic_trace(torch_model)
        FakeTensorProp(fx_module).propagate(*inputs)
        res = fx_module(*inputs)
        input_ref = {}
        i=0
        for node in fx_module.graph.nodes:
            if node.op == "placeholder":
                name = node.name
                input_ref[name] = inputs[i].detach().numpy()
                i+=1
        np.savez('input_ref.npz', **input_ref)
        output_ref = {}
        for i in range(len(res)):
            output_ref[str(i)] = res[i].detach().numpy()
        np.savez('output_ref.npz', **output_ref)
        self.convert_module_fx(
            submodule_name=model_name,
            module=fx_module)

    def test_Conv_backward(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x,y,z):
                out0,out1,out2 = torch.ops.aten.convolution_backward(x, y, z, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
                out2 = None
                return [out0,out1]

        self.trace_and_test([[8, 256, 56, 56],[8, 64, 56, 56],[256, 64, 1, 1]], Model())

    def test_Conv(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x,y):
                res = torch.ops.aten.convolution.default(x, y, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
                return res

        self.trace_and_test([[1,3,16,16],[3,3,3,3]], Model())

    def test_maxpoolwithmask(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                out0,out1 = torch.ops.aten.max_pool2d_with_indices(x,[3,3],[1,1],[1,1],[1,1],False)
                #max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
                return [out0,out1]

        self.trace_and_test([[8,3,16,16]], Model())

    def test_batchnorm(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, inp,mean,bias,rm,rv):
                out = torch.ops.aten._native_batch_norm_legit_functional(inp,mean,bias,rm,rv,True,0.1, 1e-5)
                out0 = out[0]
                out1 = out[1]
                out2 = out[2]
                out3 = out[3]
                out4 = out[4]
                return [out0,out1,out2,out3,out4]

        self.trace_and_test([[8,3,16,16],[3],[3],[3],[3]], Model())


    def test_maxpoolwithmask_bwd(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x,y,z):
                x = x + 1
                out0 = torch.ops.aten.max_pool2d_with_indices_backward(x,y,[3,3],[2,2],[1,1],[1,1],False, z)
                return [out0]

        self.trace_and_test([[8,64,56,56],[8,64,112,112], [8,64,56,56]], Model())


    def test_maxpoolwithmask_full(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y, z):
                max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(x, [3, 3], [2, 2], [1, 1])
                getitem_5 = max_pool2d_with_indices[0]
                getitem_6 = max_pool2d_with_indices[1]
                y        += 1
                out0 = torch.ops.aten.max_pool2d_with_indices_backward(y,x,[3,3],[2,2],[1,1],[1,1],False, getitem_6)
                out0 += 1
                return [out0, getitem_6, getitem_5]

        self.trace_and_test([[8,1,112,112],[8,1,56,56], [8,1,56,56]], Model())


    def test_maxpoolwithmask_full(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y, z):
                x = torch.nn.functional.relu(x)
                y *= 0.5
                max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(x, [3, 3], [2, 2], [1, 1])
                getitem_5 = max_pool2d_with_indices[0]
                getitem_6 = max_pool2d_with_indices[1]
                out0 = torch.ops.aten.max_pool2d_with_indices_backward(y,x,[3,3],[2,2],[1,1],[1,1],False, getitem_6)
                out0 += 1
                return [out0, getitem_6, getitem_5]
        self.trace_and_test([[8,64,112,112],[8,64,56,56], [8,64,56,56]], Model())

    # batchnormbwd
    def test_batchnormbwd(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x,y,z,w,a,b,c):

                out0,out1,out2 = torch.ops.aten.native_batch_norm_backward(x, y, z, w,a,b,c, False, 1e-5, [True, True, True])
                return [out0,out1,out2]

        self.trace_and_test([[8,2048,7,7],[8,2048,7,7],[2048],[2048],[2048],[2048],[2048]], Model())
        c = 512
        self.trace_and_test([[8,c,7,7],[8,c,7,7],[c],[c],[c],[c],[c]], Model())
        h = 14
        w = 14
        self.trace_and_test([[8,c,h,w],[8,c,h,w],[c],[c],[c],[c],[c]], Model())


    def test_where_batchnormbwd(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x,y,z,w,a,b,c,cond,cha):

                threshold_backward = torch.ops.aten.threshold_backward.default(cond, cha+1, 0)
                out0,out1,out2 = torch.ops.aten.native_batch_norm_backward(threshold_backward, y, z, w,a,b,c, False, 1e-5, [True, True, True])
                return [out0,out1,out2]

        self.trace_and_test([[8,2048,7,7],[8,2048,7,7],[2048],[2048],[2048],[2048],[2048],[8,2048,7,7],[8,2048,7,7]], Model())
        c = 512
        self.trace_and_test([[8,c,7,7],[8,c,7,7],[c],[c],[c],[c],[c],[8,c,7,7],[8,c,7,7]], Model())
        h = 14
        w = 14
        self.trace_and_test([[8,c,h,w],[8,c,h,w],[c],[c],[c],[c],[c],[8,c,h,w],[8,c,h,w]], Model())


    def test_batchnormfwd(self):
        class Model(torch.nn.Module):

            def __init__(self, config:list|None = None):
                super(Model, self).__init__()
                self.config = config

            def forward(self, conv, running_mean, running_var, weight, bias):
                res = torch.ops.aten._native_batch_norm_legit_functional.default(conv, running_mean, running_var, weight, bias, True, 0.1, 1e-5)
                y = res[0]
                mean = res[1]
                invstd = res[2]
                running_mean_update = res[3]
                running_var_update = res[4]
                return [y, mean, invstd, running_mean_update, running_var_update]
        n = 8
        #for c, h, w in [(2048, 7, 7), (512, 14, 14)]:
        for c, h, w in [(2048, 7, 7)]:
            self.trace_and_test([[n, c, h, w], [c], [c], [c], [c]], Model())


def test_one_case_in_all(tester: FX_IR_TESTER, case, error_cases, success_cases):
    t = Timer()
    try:
        tester.test_single(case)
    except:
        error_cases.append("{}:{}s".format(case, int(t.elapsed_time())))
        return
    success_cases.append("{}:{}s".format(case, int(t.elapsed_time())))

def test_all(tester: FX_IR_TESTER):
    if tester.multithread:
        import multiprocessing
        from utils.misc import collect_process
        process_number = multiprocessing.cpu_count() // 2 + 1
        processes = []
        error_cases = multiprocessing.Manager().list()
        success_cases = multiprocessing.Manager().list()
        for case in tester.test_cases:
            if tester.check_support(case):
                print("====== test_fx.py --case {} --chip {} TEST START PROCESSING ======".format(
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
        print("====== test_fx.py --chip {} TEST Failed ======".format(tester.chip))
        # exit(1)
    else:
        print("====== test_fx.py --chip {} TEST Success ======".format(tester.chip))
        for k in error_cases:
            case_name = k.split(":")[0]
            print("{} --chip {} --case {} failed".format(sys.argv[0], tester.chip, case_name))
    clean_kmp_files()
    return error_cases

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_all", action="store_true", help='show all cases')
    parser.add_argument("--chip", default="bm1690", choices=['bm1684x', 'bm1688', 'bm1690'], help="chip name")
    parser.add_argument("--case", default="all", help="test case")
    parser.add_argument("--disable_thread", action="store_true", help='do test without multi thread')
    parser.add_argument("--disable_cmp", action="store_true", help='do data compare')
    parser.add_argument("--concise_log", action="store_true", help="use concise log")
    parser.add_argument("--debug", default="", help="debug")
    args = parser.parse_args()
    tester = FX_IR_TESTER(args.chip, args.concise_log, args.disable_thread, args.disable_cmp)
    if args.show_all:
        print("====== Show All Cases ============")
        for case in tester.test_cases:
            print(case)
        exit(0)
    dir = "fx_test_{}".format(args.chip)
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)
    if args.case == "" or args.case == "all":
        test_all(tester)
    else:
        tester.test_single(args.case)
    if args.debug == False:
        file_clean()
