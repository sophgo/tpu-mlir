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
import json


class FX_IR_TESTER(object):
    ID = 0
    CURRENT_CASE = ""

    def __init__(self,
                 chip: str = "bm1690",
                 concise_log: bool = False,
                 disable_cmp: bool = False,
                 json_file: str = "",
                 case_id: int = -1,
                 mode: str = "f16"):
        Y, N = True, False
        # yapf: disable
        self.test_cases = {
            #########################################
            # FX Test Case, Alphabetically
            #########################################
            # case: (test, bm1684x_support, bm1688_support, bm1690_support)
            "BatchNormTrain":           (self.test_bn_fwd,                      N, N, Y),
            "BatchNormBwd":             (self.test_bn_bwd,                      N, N, Y),
            "Conv":                     (self.test_conv_fwd,                    N, N, Y),
            "Convbwd":                  (self.test_conv_bwd,                    N, Y, Y),
            "MaxPoolWithMask":          (self.test_maxpool_fwd,                 N, N, Y),
            "MaxPoolingIndicesBwd":     (self.test_maxpool_bwd,                 N, Y, Y),#indices need real data
            "IndexPut":                 (self.test_index_put,                   N, Y, Y),
            # "Maxpool_fwd_bwd":  (self.test_maxpool_fwd_bwd,                 N, N, Y),
            # "Where_BNbwd":       (self.test_where_bnbwd,                    N, N, Y),
        }
        # yapf: enable
        self.chip = chip
        self.concise_log = concise_log
        self.cmp = not disable_cmp
        self.json_file = json_file
        self.json_cases = {}
        self.failed_cases = []
        self.case_id = case_id
        self.mode = mode
        if self.json_file:
            assert not self.json_file or os.path.exists(
                self.json_file), f"Path {self.json_file} does not exist"
            with open(self.json_file, 'r', encoding='utf-8') as file:
                self.json_cases = json.load(file)

    def convert_module_fx(
        self,
        submodule_name: str,
        module: torch.fx.GraphModule,
    ):
        c = fx2mlir(submodule_name=submodule_name,
                    chip=self.chip,
                    bwd_graph=False,
                    cmp=self.cmp,
                    mode=self.mode)
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
            if self.json_cases:
                assert self.json_cases["operator"] == case, \
                f"json_cases operator {self.json_cases['operator']} not match case {case}"
                for json_case in self.json_cases["cases"]:
                    if self.case_id >= 0 and json_case["id"] != self.case_id:
                        continue
                    try:
                        func(json_case=json_case)
                    except:
                        self.failed_cases.append(
                            [json_case["id"], json_case["input_shapes"], json_case["model_params"]])
                if self.failed_cases:
                    for failed_case in self.failed_cases:
                        print(failed_case)
                    print("====== TEST {} Failed ======".format(case))
                    return
            else:
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

    def test_conv_bwd(self, json_case=None):

        class Model(torch.nn.Module):

            def __init__(self,
                         stride=[1, 1],
                         padding=[0, 0],
                         dilations=[1, 1],
                         groups=1,
                         grad_input_enable=True,
                         grad_weight_enable=True,
                         grad_bias_enable=False):
                super(Model, self).__init__()
                self.stride = stride
                self.padding = padding if len(padding) <= 2 else padding[0:-1:2]
                if len(padding) >= 2:
                    for idx, pad in enumerate(padding[:int(len(padding) / 2)]):
                        assert pad == padding[-(idx + 1)], "Only support symmetric padding for now"
                self.dilations = dilations
                self.output_mask = [grad_input_enable, grad_weight_enable, grad_bias_enable]
                self.groups = groups

            def forward(self, x, y, z):
                outs = torch.ops.aten.convolution_backward(x, y, z, [0], self.stride, self.padding,
                                                           self.dilations, False, [0, 0],
                                                           self.groups, self.output_mask)
                grads = []
                if self.output_mask[0]:  # grad_input
                    grads.append(outs[0])
                if self.output_mask[1]:  # grad_weight
                    grads.append(outs[1])
                if self.output_mask[2]:  # grad_bias
                    grads.append(outs[2])
                return grads

        if json_case:
            self.trace_and_test(json_case['input_shapes'], Model(**json_case['model_params']))
        else:
            for batch, oc, ic, oh, ow, ih, iw, kh, kw, grad_input_enable in [(8, 128, 256, 56, 56, 56, 56, 1, 1, True),
                                                          (128, 128, 256, 56, 56, 56, 56, 1, 1, True)]:
                self.trace_and_test([[batch, oc, oh, ow], [batch, ic, ih, iw], [oc, ic, kh, kw]],
                                    Model(grad_input_enable=grad_input_enable))
            if self.chip == "bm1690":
                # bm1690 support grad_bias
                self.trace_and_test([[8, 255, 40, 40], [8, 256, 40, 40], [255, 256, 1, 1]],
                                    Model(grad_bias_enable=True))

    def test_conv_fwd(self, json_case=None):

        class Model(torch.nn.Module):

            def __init__(self, strides=[1, 1], pads=[1, 1], dilations=[1, 1], group=1):
                super(Model, self).__init__()
                self.strides = strides
                self.pads = pads if len(pads) <= 2 else pads[0:-1:2]
                if len(pads) >= 2:
                    for idx, pad in enumerate(pads[:int(len(pads) / 2)]):
                        assert pad == pads[-(idx + 1)], "Only support symmetric padding for now"
                self.dilations = dilations
                self.group = group

            def forward(self, x, y):
                #convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups) -> Tensor
                res = torch.ops.aten.convolution.default(x, y, None, self.strides, self.pads,
                                                         self.dilations, False, [0, 0], self.group)
                return res

        if json_case:
            self.trace_and_test(json_case['input_shapes'], Model(**json_case['model_params']))
        else:
            self.trace_and_test([[1, 3, 16, 16], [3, 3, 3, 3]], Model())

    def test_maxpool_fwd(self, json_case=None):

        class Model(torch.nn.Module):

            def __init__(self,
                         kernel_shape=[3, 3],
                         strides=[2, 2],
                         pads=[0, 0],
                         ceil_mode=False,
                         dilation=[1, 1],
                         do_relu=False):
                super(Model, self).__init__()
                self.kernel_shape = kernel_shape
                self.strides = strides
                self.pads = pads if len(pads) <= 2 else pads[0:-1:2]
                if len(pads) >= 2:
                    for idx, pad in enumerate(pads[:int(len(pads) / 2)]):
                        assert pad == pads[-(idx + 1)], "Only support symmetric padding for now"
                self.dilation = dilation
                self.ceil_mode = ceil_mode
                self.do_relu = do_relu

            def forward(self, x):
                #max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
                out0, out1 = torch.ops.aten.max_pool2d_with_indices(x, self.kernel_shape,
                                                                    self.strides, self.pads,
                                                                    self.dilation, self.ceil_mode)
                if self.do_relu:
                    out0 = torch.nn.functional.relu(out0)
                return [out0, out1]

        if json_case:
            self.trace_and_test(json_case['input_shapes'], Model(**json_case['model_params']))
        else:
            for batch, ch, h, w, kernel_shape in [(8, 64, 64, 64, [3, 3]), (8, 64, 224, 224, [2,
                                                                                              2])]:
                self.trace_and_test([[batch, ch, h, w]], Model(kernel_shape=kernel_shape))

    def test_maxpool_bwd(self, json_case=None):

        class Model(torch.nn.Module):

            def __init__(self,
                         kernel_shape=[2, 2],
                         strides=[2, 2],
                         dilations=[1, 1],
                         pads=[0, 0],
                         ceil_mode=False,
                         input_shape=None):
                super(Model, self).__init__()
                self.kernel_shape = kernel_shape
                self.strides = strides
                self.pads = pads if len(pads) <= 2 else pads[0:-1:2]
                if len(pads) >= 2:
                    for idx, pad in enumerate(pads[:int(len(pads) / 2)]):
                        assert pad == pads[-(idx + 1)], "Only support symmetric padding for now"
                self.dilations = dilations[0:-1:2]
                self.ceil_mode = ceil_mode

            def forward(self, x, y, z):
                #max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor
                out0 = torch.ops.aten.max_pool2d_with_indices_backward(
                    x, y, self.kernel_shape, self.strides, self.pads, self.dilations,
                    self.ceil_mode, z)
                # out0 = torch.ops.aten.max_pool2d_with_indices_backward(x,y,[3,3],[2,2],[1,1],[1,1],False,z)
                return [out0]

        class Model_fwd(torch.nn.Module):

            def __init__(self,
                         kernel_shape=[2, 2],
                         strides=[2, 2],
                         dilations=[1, 1],
                         pads=[0, 0],
                         ceil_mode=False,
                         input_shape=None):
                super(Model_fwd, self).__init__()
                self.kernel_shape = kernel_shape
                self.strides = strides
                self.pads = pads if len(pads) <= 2 else pads[0:-1:2]
                if len(pads) >= 2:
                    for idx, pad in enumerate(pads[:int(len(pads) / 2)]):
                        assert pad == pads[-(idx + 1)], "Only support symmetric padding for now"
                self.dilations = dilations[0:-1:2]
                self.ceil_mode = ceil_mode

            def forward(self, x):
                #max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
                out0, out1 = torch.ops.aten.max_pool2d_with_indices(x, self.kernel_shape,
                                                                    self.strides, self.pads,
                                                                    self.dilations, self.ceil_mode)
                # out0,out1 = torch.ops.aten.max_pool2d_with_indices(x,[3,3],[2,2],[1,1],[1,1],False)

                return [out0, out1]

        # bm1688 f32 0.99,0.92
        # bm1684x f32 1.0 1.0
        if json_case:
            batch = json_case['input_shapes'][0][0]
            ch = json_case['input_shapes'][0][1]
            oh = json_case['input_shapes'][0][2]
            ow = json_case['input_shapes'][0][3]
            ih = json_case['model_params']['input_shape'][2]
            iw = json_case['model_params']['input_shape'][3]
            x = torch.randn(batch, ch, ih, iw)
            fwd_out = Model_fwd(**json_case["model_params"])(torch.randn(batch, ch, ih, iw))
            grad = torch.randn(batch, ch, oh, ow)
            inputs = [grad, x, fwd_out[1]]
            self.trace_and_test([[batch, ch, oh, ow], [batch, ch, ih, iw], [batch, ch, oh, ow]],
                                Model(**json_case["model_params"]),
                                real_data=inputs)
        else:
            ch = 64
            batch = 1
            for isize, osize in [(112, 56)]:
                x = torch.randn(batch, ch, isize, isize)
                fwd_out = Model_fwd()(torch.randn(batch, ch, isize, isize))
                grad = torch.randn(batch, ch, osize, osize)
                inputs = [grad, x, fwd_out[1]]
                self.trace_and_test([[batch, ch, osize, osize], [batch, ch, isize, isize],
                                     [batch, ch, osize, osize]],
                                    Model(),
                                    real_data=inputs)

    def test_maxpool_fwd_bwd(self, json_case=None):

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

    def test_bn_bwd(self, json_case=None):

        class Model(torch.nn.Module):

            def __init__(self, epsilon=1e-5):
                super(Model, self).__init__()
                self.epsilon = epsilon

            def forward(self, x, y, z, w, a, b, c):
                out0, out1, out2 = torch.ops.aten.native_batch_norm_backward(
                    x, y, z, w, a, b, c, True, self.epsilon, [True, True, True])
                return [out0, out1, out2]

        if json_case:
            self.trace_and_test(json_case['input_shapes'][:2] + [json_case['input_shapes'][2]] * 5,
                                Model(**json_case['model_params']))
        else:
            for batch, c, h, w in [(8, 64, 112, 112), (8, 512, 7, 7), (128, 512, 14, 14)]:
                self.trace_and_test([[batch, c, h, w], [batch, c, h, w], [c], [c], [c], [c], [c]],
                                    Model())

    def test_where_bnbwd(self, json_case=None):

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

    def test_bn_fwd(self, json_case=None):

        class Model(torch.nn.Module):

            def __init__(self, do_relu=False, epsilon=1e-5, momentum=0.1):
                super(Model, self).__init__()
                self.do_relu = do_relu
                self.epsilon = epsilon
                self.momentum = momentum

            def forward(self, conv, weight, bias, running_mean, running_var):
                #_native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
                res = torch.ops.aten._native_batch_norm_legit_functional.default(
                    conv, weight, bias, running_mean, running_var, True, self.momentum,
                    self.epsilon)
                y = res[0]
                if self.do_relu:
                    y = torch.nn.functional.relu(y)
                mean = res[1]
                invstd = res[2]
                running_mean_update = res[3]
                running_var_update = res[4]
                return [y, mean, invstd, running_mean_update, running_var_update]

        if json_case:
            self.trace_and_test(json_case['input_shapes'], Model(**json_case['model_params']))
        else:
            n = 8
            for c, h, w in [(2048, 7, 7), (512, 14, 14)]:
                self.trace_and_test([[n, c, h, w], [c], [c], [c], [c]], Model())

    def test_index_put(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, updates, axis_h, axis_w, values):
                #index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
                res = torch.ops.aten._unsafe_index_put.default(
                    updates, [None, None, axis_h, axis_w], values, True)
                return res

        ##inplace_add case
        updates = torch.zeros((1, 1280, 32, 32))
        axis_h = (torch.arange(64) // 2).long().unsqueeze(-1)
        axis_w = (torch.arange(64) // 2).long()
        values = torch.randn(1, 1280, 64, 64)
        self.trace_and_test([[1, 1280, 32, 32], [64, 1], [64], [1, 1280, 64, 64]],
                            Model(),
                            real_data=[updates, axis_h, axis_w, values])


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
    parser.add_argument("--case", default="all", help="test case name")
    parser.add_argument("--disable_cmp", action="store_true", help='do data compare')
    parser.add_argument("--concise_log", action="store_true", help="use concise log")
    parser.add_argument("--debug", default="", help="debug mode to keep all files")
    parser.add_argument("--json_file",
                        default="",
                        help="load json file for test cases, only for single op test")
    parser.add_argument("--case_id",
                        default=-1,
                        type=int,
                        help="load the specific id case from json file, only for single op test")
    parser.add_argument("--mode",
                        default="f16",
                        choices=['f16', 'f32'],
                        type=str.lower,
                        help="quantize mode, f16 or f32")
    args = parser.parse_args()
    tester = FX_IR_TESTER(chip=args.chip,
                          concise_log=args.concise_log,
                          disable_cmp=args.disable_cmp,
                          json_file=args.json_file,
                          case_id=args.case_id,
                          mode=args.mode)
    if args.show_all:
        print("====== Show All Cases ============")
        for case in tester.test_cases:
            print(case)
        exit(0)
    dir = "fx_test_{}".format(args.chip)
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)
    if args.case == "" or args.case == "all":
        assert args.json_file == "", "json_cases should not be set when running full test"
        test_all(tester)
    else:
        tester.test_single(args.case)
    if args.debug == False:
        file_clean()
