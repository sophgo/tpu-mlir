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
from tools.train import config


class FX_IR_TESTER(object):
    ID = 0
    CURRENT_CASE = ""

    def __init__(self, chip: str = "bm1690", concise_log: bool = False, disable_cmp: bool = False):
        Y, N = True, False
        # yapf: disable
        self.test_cases = {
            #########################################
            # FX Test Case, Alphabetically
            #########################################
            # case: (test, bm1684x_support, bm1688_support, bm1690_support)
            "Convolution":              (self.test_Conv,                    N, N, Y),
            "Convbackward":             (self.test_Conv_backward,           N, Y, Y),
            "batchnormbwd":             (self.test_batchnormbwd,            N, N, Y),
            "batchnormfwd":             (self.test_batchnormfwd,            N, N, Y),
            "maxpoolwithmask":          (self.test_maxpoolwithmask,         N, N, Y),
            "maxpoolwithmask_bwd":      (self.test_maxpoolwithmask_bwd,     N, Y, Y),#indices need real data
            "maxpoolwithmask_fwd_bwd":  (self.test_maxpoolwithmask_fwd_bwd, N, N, Y),
            "where_batchnormbwd":       (self.test_where_batchnormbwd,      N, N, Y),
        }
        # yapf: enable
        self.chip = chip
        self.concise_log = concise_log
        self.cmp = not disable_cmp

    def convert_module_fx(
        self,
        submodule_name: str,
        module: torch.fx.GraphModule,
    ):
        c = fx2mlir(submodule_name=submodule_name, chip=self.chip, bwd_graph=False, cmp=self.cmp)
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

    class Desc():

        def __init__(self, dtype, min=-10, max=10) -> None:
            self.dtype = dtype
            self.min = min
            self.max = max

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
        _, bm1684x_support, bm1688_support, bm1690_support = self.test_cases[case]
        if self.chip == "bm1684x" and bm1684x_support:
            return True
        if self.chip == "bm1688" and bm1688_support:
            return True
        if self.chip == "bm1690" and bm1690_support:
            return True
        return False

    def trace_and_test(self,
                       in_shapes,
                       torch_model: nn.Module,
                       descs=[],
                       use_cos: bool = False,
                       input_info=None,
                       real_data=None):
        model_name = "{}_{}".format(self.CURRENT_CASE, FX_IR_TESTER.ID)
        FX_IR_TESTER.ID += 1
        inputs = self.create_random_input(in_shapes, descs)
        inputs = real_data if real_data else inputs  # use real data if provided
        fx_module = symbolic_trace(torch_model)
        FakeTensorProp(fx_module).propagate(*inputs)
        res = fx_module(*inputs)
        input_ref = {}
        i = 0
        for node in fx_module.graph.nodes:
            if node.op == "placeholder":
                name = node.name
                input_ref[name] = inputs[i].detach().numpy()
                i += 1
        np.savez('input_ref.npz', **input_ref)
        output_ref = {}
        if isinstance(res, (list, tuple)):
            for i in range(len(res)):
                tensor = res[i].detach().cpu().numpy()
                output_ref[str(i)] = tensor
        elif isinstance(res, torch.Tensor):
            tensor = res.detach().cpu().numpy()
            output_ref["0"] = tensor
        np.savez('output_ref.npz', **output_ref)
        config.unit_test = True
        self.convert_module_fx(submodule_name=model_name, module=fx_module)

    def test_Conv_backward(self):

        class Model(torch.nn.Module):

            def __init__(self, stride=[1, 1], padding=[0, 0], dilation=[1, 1]):
                super(Model, self).__init__()
                self.stride = stride
                self.padding = padding
                self.dilation = dilation

            def forward(self, x, y, z):
                #convolution_backward(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
                out0, out1, out2 = torch.ops.aten.convolution_backward(
                    x, y, z, [0], self.stride, self.padding, self.dilation, False, [0, 0], 1,
                    [True, True, False])
                out2 = None
                return [out0, out1]

        if self.chip == "bm1690":
            for batch, oc, ic, oh, ow, ih, iw, kh, kw in [(8, 128, 256, 56, 56, 56, 56, 1, 1),
                                                          (128, 128, 256, 56, 56, 56, 56, 1, 1)]:
                self.trace_and_test([[batch, oc, oh, ow], [batch, ic, ih, iw], [oc, ic, kh, kw]],
                                    Model())
        if self.chip == "bm1688":
            for batch, oc, ic, oh, ow, ih, iw, kh, kw in [(8, 64, 64, 224, 224, 224, 224, 3, 3)]:
                self.trace_and_test([[batch, oc, oh, ow], [batch, ic, ih, iw], [oc, ic, kh, kw]],
                                    Model(padding=[1, 1]))

    def test_Conv(self):

        class Model(torch.nn.Module):

            def __init__(self,
                         stride=[1, 1],
                         padding=[1, 1],
                         dilation=[1, 1],
                         output_padding=[0, 0],
                         groups=1):
                super(Model, self).__init__()
                self.stride = stride
                self.padding = padding
                self.dilation = dilation
                self.output_padding = output_padding
                self.groups = groups

            def forward(self, x, y):
                #convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups) -> Tensor
                res = torch.ops.aten.convolution.default(x, y, None, self.stride, self.padding,
                                                         self.dilation, False, self.output_padding,
                                                         self.groups)
                return res

        self.trace_and_test([[1, 3, 16, 16], [3, 3, 3, 3]], Model())

    def test_maxpoolwithmask(self):

        class Model(torch.nn.Module):

            def __init__(self, kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], dilation=[1, 1]):
                super(Model, self).__init__()
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding
                self.dilation = dilation

            def forward(self, x):
                #max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
                out0, out1 = torch.ops.aten.max_pool2d_with_indices(x, self.kernel_size,
                                                                    self.stride, self.padding,
                                                                    self.dilation, False)
                return [out0, out1]

        for batch, ch, h, w, kernel_size in [(8, 64, 64, 64, [3, 3]), (8, 64, 224, 224, [2, 2])]:
            self.trace_and_test([[batch, ch, h, w]], Model(kernel_size=kernel_size))

    def test_maxpoolwithmask_bwd(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y, z):
                #max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor
                out0 = torch.ops.aten.max_pool2d_with_indices_backward(
                    x, y, [2, 2], [2, 2], [0, 0], [1, 1], False, z)
                # out0 = torch.ops.aten.max_pool2d_with_indices_backward(x,y,[3,3],[2,2],[1,1],[1,1],False,z)
                return [out0]

        class Model_fwd(torch.nn.Module):

            def __init__(self):
                super(Model_fwd, self).__init__()

            def forward(self, x):
                #max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
                out0, out1 = torch.ops.aten.max_pool2d_with_indices(x, [2, 2], [2, 2], [0, 0],
                                                                    [1, 1], False)
                # out0,out1 = torch.ops.aten.max_pool2d_with_indices(x,[3,3],[2,2],[1,1],[1,1],False)

                return [out0, out1]

        # bm1688 f32 0.99,0.92
        # bm1684x f32 1.0 1.0
        ch = 64
        batch = 1

        for isize, osize in [(112, 56), (224, 112)]:
            x = torch.randn(batch, ch, isize, isize)
            fwd_out = Model_fwd()(torch.randn(batch, ch, isize, isize))
            grad = torch.randn(batch, ch, osize, osize)
            inputs = [grad, x, fwd_out[1]]
            self.trace_and_test(
                [[batch, ch, osize, osize], [batch, ch, isize, isize], [batch, ch, osize, osize]],
                Model(),
                real_data=inputs)

    def test_maxpoolwithmask_fwd_bwd(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y, z):
                x = torch.nn.functional.relu(x)
                y *= 0.5
                max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(
                    x, [3, 3], [2, 2], [1, 1])
                indices = max_pool2d_with_indices[1]
                out0 = torch.ops.aten.max_pool2d_with_indices_backward(
                    y, x, [3, 3], [2, 2], [1, 1], [1, 1], False, indices)
                out0 += 1
                return [out0, indices]

        self.trace_and_test([[8, 64, 112, 112], [8, 64, 56, 56], [8, 64, 56, 56]], Model())

    def test_batchnormbwd(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y, z, w, a, b, c):
                #native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
                out0, out1, out2 = torch.ops.aten.native_batch_norm_backward(
                    x, y, z, w, a, b, c, True, 1e-5, [True, True, True])
                return [out0, out1, out2]

        for batch, c, h, w in [(8, 64, 112, 112), (8, 512, 7, 7), (128, 512, 14, 14)]:
            self.trace_and_test([[batch, c, h, w], [batch, c, h, w], [c], [c], [c], [c], [c]],
                                Model())

    def test_where_batchnormbwd(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, conv, weight, bias, running_mean, running_var, grad, cond):
                '''WhereBnbwdFusePattern need BatchNormTrainOp and fuse relu'''
                fwd_out = torch.ops.aten._native_batch_norm_legit_functional.default(
                    conv, weight, bias, running_mean, running_var, True, 0.1, 1e-5)
                relu_out = torch.nn.functional.relu(fwd_out[0])
                threshold_backward = torch.ops.aten.threshold_backward.default(grad, relu_out, 0)
                # np.savez('threshold_backward.npz', threshold_backward=threshold_backward.detach().cpu().numpy())
                out0, out1, out2 = torch.ops.aten.native_batch_norm_backward(
                    threshold_backward, conv, weight, running_mean, running_var, fwd_out[1],
                    fwd_out[2], True, 1e-5, [True, True, True])
                return [relu_out, out0, out1, out2]

        batch = 8
        c = 32
        h = 7
        w = 7

        shapes = [[batch, c, h, w], [c], [c], [c], [c], [batch, c, h, w], [batch, c, h, w]]
        inputs = [torch.randn(s) for s in shapes]
        self.trace_and_test(shapes, Model(), real_data=inputs)
        # c = 512
        # self.trace_and_test([[8,c,7,7],[8,c,7,7],[c],[c],[c],[c],[c],[8,c,7,7],[8,c,7,7]], Model())
        # h = 14
        # w = 14
        # self.trace_and_test([[8,c,h,w],[8,c,h,w],[c],[c],[c],[c],[c],[8,c,h,w],[8,c,h,w]], Model())

    def test_batchnormfwd(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, conv, weight, bias, running_mean, running_var):
                #_native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
                res = torch.ops.aten._native_batch_norm_legit_functional.default(
                    conv, weight, bias, running_mean, running_var, True, 0.1, 1e-5)
                y = res[0]
                mean = res[1]
                invstd = res[2]
                running_mean_update = res[3]
                running_var_update = res[4]
                return [y, mean, invstd, running_mean_update, running_var_update]

        n = 8
        for c, h, w in [(2048, 7, 7), (512, 14, 14)]:
            self.trace_and_test([[n, c, h, w], [c], [c], [c], [c]], Model())


def test_one_case_in_all(tester: FX_IR_TESTER, case, error_cases, success_cases):
    t = Timer()
    original_dir = os.getcwd()  # fx_test_{chip_name}
    try:
        tester.test_single(case)
        os.chdir(original_dir)  # back to fx_test_{chip_name}
    except:
        os.chdir(original_dir)
        error_cases.append("{}:{}s".format(case, int(t.elapsed_time())))
        return
    success_cases.append("{}:{}s".format(case, int(t.elapsed_time())))


def test_all(tester: FX_IR_TESTER):
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
    parser.add_argument("--chip",
                        default="bm1690",
                        choices=['bm1684x', 'bm1688', 'bm1690'],
                        help="chip name")
    parser.add_argument("--case", default="all", help="test case")
    parser.add_argument("--disable_cmp", action="store_true", help='do data compare')
    parser.add_argument("--concise_log", action="store_true", help="use concise log")
    parser.add_argument("--debug", default="", help="debug")
    args = parser.parse_args()
    tester = FX_IR_TESTER(args.chip, args.concise_log, args.disable_cmp)
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
