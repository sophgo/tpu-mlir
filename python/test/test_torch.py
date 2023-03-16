#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from copy import deepcopy
from re import T
import numpy as np
from typing import List, Union

from tools.model_runner import mlir_inference, model_inference, torch_inference, show_fake_cmd
from tools.npz_tool import npz_compare
from tools.model_transform import *
from utils.mlir_shell import *
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
import traceback


class TORCH_IR_TESTER(object):
    ID = 0
    CURRENT_CASE = ""

    # This class is built for testing single operator transform.
    def __init__(self, chip: str = "bm1684x", mode: str = "all"):
        Y, N = True, False
        # yapf: disable
        self.test_cases = {
            ##################################
            # Torch Test Case, Alphabetically
            ##################################
            # case: (test, bm1684x_support, bm1686_support, cv183x_support)
            "Abs":              (self.test_Abs,         Y, N, N),
            "Activation":       (self.test_Activation,  Y, N, N),
            "Add":              (self.test_Add,         Y, N, N),
            "AvgPool1d":        (self.test_AvgPool1d,   Y, N, N),
            "AvgPool2d":        (self.test_AvgPool2d,   Y, N, N),
            "AvgPool3d":        (self.test_AvgPool3d,   Y, N, N),
            "Compare":          (self.test_Compare,     Y, N, N),
            "Concat":           (self.test_Concat,      Y, N, N),
            "Conv1d":           (self.test_Conv1d,      Y, N, N),
            "Conv2d":           (self.test_Conv2d,      Y, N, N),
            "Conv3d":           (self.test_Conv3d,      Y, N, N),
            "ConvTrans":        (self.test_ConvTrans,   Y, N, N),
            "Div":              (self.test_Div,         Y, N, N),
            "Dropout":          (self.test_Dropout,     Y, N, N),
            "Elu":              (self.test_Elu,         Y, N, N),
            "Gather":           (self.test_Gather,      N, N, N),
            "GRU":              (self.test_GRU,         Y, N, N),
            "IndexSelect":      (self.test_IndexSelect, Y, N, N),
            "LayerNorm":        (self.test_LayerNorm,   Y, N, N),
            "LeakyRelu":        (self.test_LeakyRelu,   Y, N, N),
            "LogSoftmax":       (self.test_LogSoftmax,  Y, N, N),
            "LSTM":             (self.test_LSTM,        Y, N, N),
            "MaxPool1d":        (self.test_MaxPool1d,   Y, N, N),
            "MaxPool2d":        (self.test_MaxPool2d,   Y, N, N),
            "MaxPool3d":        (self.test_MaxPool3d,   Y, N, N),
            "Mul":              (self.test_Mul,         Y, N, N),
            "PRelu":            (self.test_PRelu,       Y, N, N),
            "Permute":          (self.test_Permute,     Y, N, N),
            "Pad1d":            (self.test_Pad1d,       Y, N, N),
            "Pad2d":            (self.test_Pad2d,       Y, N, N),
            "Scatter":          (self.test_Scatter,     N, N, N),
            "Softmax":          (self.test_Softmax,     Y, N, N),
            "Softmin":          (self.test_Softmin,     Y, N, N),
            "Sub":              (self.test_Sub,         Y, N, N),
            "T":                (self.test_T,           Y, N, N),
            "Tile":             (self.test_Tile,        Y, N, N),
            "Transpose":        (self.test_Transpose,   Y, N, N),
            "View":             (self.test_View,        Y, N, N),
            "Where":            (self.test_Where,       N, N, N),
        }
        # yapf: enable
        self.support_quant_modes = ["f32", "f16", "bf16"]
        #self.support_quant_modes = ["f32", "f16", "bf16", "int8"]
        self.support_asym = [True, False]
        self.model_file = ".bmodel"
        self.is_cv18xx = False
        self.chip = chip.lower()
        # self.dynamic = dynamic
        if self.chip.startswith("cv18"):
            self.support_quant_modes = ["bf16", "int8"]
            self.support_asym = [False]
            self.model_file = ".cvimodel"
            self.is_cv18xx = True
        elif self.chip == "bm1684":
            self.support_quant_modes = ["f32", "int8"]
            self.support_asym = [False]
        self.mode = mode.lower()
        if self.mode == "" or self.mode == "all":
            self.quant_modes = self.support_quant_modes
        else:
            if self.mode not in self.support_quant_modes:
                raise RuntimeError("{} not support mode: {}".format(self.chip, self.mode))
            self.quant_modes = [self.mode]

    def test_single(self, case: str):
        np.random.seed(0)
        torch.manual_seed(7)
        TORCH_IR_TESTER.ID = 0
        TORCH_IR_TESTER.CURRENT_CASE = case
        print("Test: {}".format(case))
        if case in self.test_cases:
            func, _, _, _ = self.test_cases[case]
            func()
            print("====== TEST {} Success ======".format(case))
        else:
            raise RuntimeError("case [{}] is not exist".format(case))

    def check_support(self, case):
        _, bm1684x_support, bm1686_support, cv183x_support = self.test_cases[case]
        if self.is_cv18xx and cv183x_support:
            return True
        if self.chip == "bm1684x" and bm1684x_support:
            return True
        if self.chip == "bm1686" and bm1686_support:
            return True
        return False

    def square_rooted(self, x):
        return np.sqrt(sum([a * a for a in x]))

    def cosine_similarity(self, x, y):
        numerator = sum(a * b for a, b in zip(x, y))
        denominator = self.square_rooted(x) * self.square_rooted(y)
        return round(numerator / float(denominator), 3)

    def compare(self, ref_out, targe_out):
        if ref_out.dtype in [np.int64, np.int32, np.int16, np.int8]:
            cos = self.cosine_similarity(ref_out, targe_out)
            assert (cos > 0.999)
        else:
            np.testing.assert_allclose(ref_out, targe_out, rtol=1e-5, atol=1e-01)

    def make_test_calibration_table(self, tensors, table_name):
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

    def create_random_input(self, shapes):
        inputs = [np.clip(np.random.randn(*s).astype(np.float32), -10, 10) for s in shapes]
        return [torch.from_numpy(inp) for inp in inputs]

    def torch_convert(self, in_shapes, torch_model, model_name: str):
        # torch --> mlir conversion (origin and optimized mlir models will be generated and saved)
        fp32_mlir = "{}.mlir".format(model_name)

        tool = TorchTransformer(model_name, torch_model, input_shapes=in_shapes)
        tool.model_transform(fp32_mlir)

        input_npz = "{}_ref_in_fp32.npz".format(model_name)
        ref_npz = model_name + '_ref_outputs.npz'
        input_data = {}
        for idx, name in enumerate(tool.converter.input_names):
            input_data[name] = np.random.random(size=in_shapes[idx]).astype(np.float32)
        np.savez(input_npz, **input_data)
        file_mark(input_npz)
        # # top mlir outputs will be inferenced first in case the quant mode is int8
        show_fake_cmd(input_npz, torch_model, ref_npz)
        torch_outs = torch_inference(input_data, torch_model, True)
        np.savez(ref_npz, **torch_outs)
        file_mark(ref_npz)
        show_fake_cmd(input_npz, fp32_mlir, "top_out.npz")
        top_mlir_outs = mlir_inference(input_data, fp32_mlir, True)
        return (torch_outs, top_mlir_outs, input_npz)

    def bmodel_generate(self,
                        model_name: str,
                        top_mlir_outs: dict,
                        quant_mode: str,
                        isAsym: bool = False):
        table_name = None
        top_mlir = "{}.mlir".format(model_name)
        tpu_mlir = "{}_{}".format(model_name, quant_mode)
        if quant_mode == "int8":
            tpu_mlir += "_asym" if isAsym else "_sym"
            table_name = "{}_cali_table".format(model_name)
            self.make_test_calibration_table(top_mlir_outs, table_name)

        # lowering
        mlir_lowering(top_mlir,
                      tpu_mlir + ".mlir",
                      mode=quant_mode,
                      chip=self.chip,
                      cali_table=table_name,
                      asymmetric=isAsym)

        # transform
        tpu_final = tpu_mlir + "_final.mlir"
        bmodel = tpu_mlir + ".bmodel"
        mlir_to_model(tpu_mlir + ".mlir", bmodel, tpu_final)

        return (tpu_mlir + ".mlir", bmodel)

    def inference_and_compare(self,
                              torch_output: dict,
                              tpu_mlir: str,
                              bmodel: str,
                              input_npz: str,
                              quant_mode: str,
                              model_name: str,
                              isAsym: bool = False):
        ref_tpu_tolerance = "0.9,0.9"
        input_data = np.load(input_npz)
        # save ref
        ref_npz = "{}_ref_outputs.npz".format(model_name)
        # tpu mlir inference and compare
        tpu_npz = tpu_mlir.replace(".mlir", "_tpu_out.npz")
        show_fake_cmd(input_npz, tpu_mlir, tpu_npz)
        tpu_mlir_outs = mlir_inference(input_data, tpu_mlir, dump_all=True)
        np.savez(ref_npz, **torch_output)
        np.savez(tpu_npz, **tpu_mlir_outs)
        file_mark(ref_npz)
        file_mark(tpu_npz)
        npz_compare([ref_npz, tpu_npz, "--tolerance", ref_tpu_tolerance, "-v"])
        # bmodel inference and compare
        model_npz = bmodel.replace("." + bmodel.split(".")[-1], "_model_out.npz")
        show_fake_cmd(input_npz, bmodel, model_npz)
        model_outs = model_inference(input_data, bmodel)
        np.savez(model_npz, **model_outs)
        file_mark(model_npz)
        npz_compare([tpu_npz, model_npz, "--tolerance", "0.95,0.80", "-v"])

        msg = quant_mode.upper()
        if quant_mode == "int8":
            msg += ", Asymmetric: {}".format(isAsym)
        print("[Success] test {} {}".format(model_name, msg))

    def trace_and_test(
        self,
        in_shapes,
        torch_model,
    ):
        """Generic function to generate and compare torch and Tpu-Mlir output"""
        model_name = "{}_{}".format(self.CURRENT_CASE, TORCH_IR_TESTER.ID)
        TORCH_IR_TESTER.ID += 1
        model_def = model_name + ".pt"
        inputs = self.create_random_input(in_shapes)
        jit.trace(torch_model, inputs).save(model_def)
        torch_outs, top_mlir_outs, input_npz = self.torch_convert(in_shapes, model_def, model_name)
        # test onnx and mlir outputs
        counter = 0
        for name in torch_outs:
            if name in top_mlir_outs:
                print("Compare mlir and torch:{}\n".format(name))
                top_mlir_output = top_mlir_outs[name].flatten()
                onnx_output = torch_outs[name].flatten()
                self.compare(onnx_output, top_mlir_output)
                counter += 1
        if counter == 0:
            raise RuntimeError("No compare between torch outs and mlir outts")
        print("Success: Torch outs and Mlir outs are equal\n")
        for quant_mode in self.quant_modes:
            if quant_mode == "int8" or quant_mode == "int4":
                for isAsym in self.support_asym:
                    tpu_mlir, bmodel = self.bmodel_generate(model_name, top_mlir_outs, quant_mode,
                                                            isAsym)
                    self.inference_and_compare(top_mlir_outs, tpu_mlir, bmodel, input_npz,
                                               quant_mode, model_name, isAsym)
            else:
                tpu_mlir, bmodel = self.bmodel_generate(model_name, top_mlir_outs, quant_mode)
                self.inference_and_compare(top_mlir_outs, tpu_mlir, bmodel, input_npz, quant_mode,
                                           model_name)

    #######################################################################
    # Convolution
    # ------------
    def _test_Conv(self):

        def case1(conv_fun, input_shape):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.conv1 = conv_fun(8, 8, 3, 1, 1)
                    self.conv2 = conv_fun(8, 8, 3, 1, 1)

                def forward(self, x):
                    y = self.conv1(x)
                    z = self.conv2(x)
                    return y + z

            self.trace_and_test([input_shape], Model())

        def case2(conv_fun,
                  input_shape,
                  kernel_shape,
                  oc,
                  has_bias=False,
                  padding: Union[int, str, List[int]] = 0,
                  stride: Union[int, List[int]] = 1,
                  dilation: Union[int, List[int]] = 1,
                  group=1):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    filter_shape = (oc, input_shape[1] // group, *kernel_shape)
                    self.filter = torch.randn(filter_shape)
                    self.bias = torch.randn(oc) if has_bias else None

                def forward(self, x):
                    y = conv_fun(x,
                                 self.filter,
                                 bias=self.bias,
                                 padding=padding,
                                 stride=stride,
                                 dilation=dilation,
                                 groups=group)
                    return y

            self.trace_and_test([input_shape], Model())

        return dict(case1=case1, case2=case2)

    def test_Conv1d(self):
        """Conv 1D"""

        test = self._test_Conv()
        test["case1"](nn.Conv1d, (4, 8, 28))
        test["case2"](F.conv1d, (1, 3, 32), [3], 12, has_bias=True, group=1, padding="same")
        test["case2"](F.conv1d, (2, 32, 16), [5], 64, padding=2, stride=2, dilation=1)
        # Tpu/Interfaces/BM1684X/Conv1D.cpp::152 Not supported yet.
        # test["case2"](F.conv1d, (1, 3, 32), 3, 12, group=3, padding=1, stride=2)

    def test_Conv2d(self):
        """Conv 2D"""

        test = self._test_Conv()
        test["case1"](nn.Conv2d, (4, 8, 28, 28))
        test["case2"](F.conv2d, (1, 3, 32, 32), (3, 3), 12, has_bias=True, group=1, padding="same")
        test["case2"](F.conv2d, (2, 32, 16, 16), (5, 5), 64, padding=2, stride=2, dilation=1)
        test["case2"](F.conv2d, (1, 3, 32, 32), (3, 3), 12, group=3, padding=(1, 1), stride=(2, 1))

    def test_Conv3d(self):
        """Conv 3D"""
        # conv3d is a computationally intensive operation that uses a small kernel to reduce runtime.
        test = self._test_Conv()
        test["case1"](nn.Conv3d, (4, 8, 7, 10, 20))
        test["case2"](F.conv3d, (1, 3, 6, 10, 12), (3, 3, 2),
                      12,
                      has_bias=True,
                      group=1,
                      padding="same")
        test["case2"](F.conv3d, (2, 32, 8, 10, 10), (5, 5, 3), 64, padding=2, stride=2, dilation=1)
        # Tpu/Interfaces/BM1684X/Conv3D.cpp::94 Not supported yet.
        # test["case2"](F.conv3d, (1, 3, 32, 32, 32), (3, 3, 3), 12, group=3, padding=(1, 1, 2), stride=(2, 1, 1))

    #######################################################################
    # Transposed Convolution
    # ------------
    def test_ConvTrans(self):

        def test_deconv(
            input_shape,
            kernel_shape,
            stride=1,
            padding=0,
            output_padding=0,
            groups=1,
            dilation=1,
        ):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.deconv = nn.ConvTranspose2d(16, 32, 3, stride=2)
                    self.weight = torch.randn(kernel_shape)

                def forward(self, x):
                    y = self.deconv(x)
                    z = nn.functional.conv_transpose2d(
                        y,
                        self.weight,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                        groups=groups,
                        dilation=dilation,
                    )
                    return z

            self.trace_and_test([input_shape], Model())

        test_deconv((2, 16, 8, 8), (32, 8, 3, 5))
        test_deconv((2, 16, 8, 8), (32, 4, 3, 4), (2, 3), (1, 1), (0, 0), groups=2)

    #######################################################################
    # AvgPooling
    # ------------
    def _test_AvgPool(self):

        def test_case(pool_funs, input_shape, kernel_size, stride, padding, count_include_pad=True):

            fun1, fun2 = pool_funs

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.pooling = fun1(5, 2)

                def forward(self, x):
                    y = self.pooling(x)
                    z = fun2(y,
                             kernel_size,
                             stride=stride,
                             padding=padding,
                             count_include_pad=count_include_pad)
                    return z

            self.trace_and_test([input_shape], Model())

        return test_case

    def test_AvgPool1d(self):
        test = self._test_AvgPool()
        test((nn.AvgPool1d, F.avg_pool1d), (4, 8, 40), 4, 3, 2)
        test((nn.AvgPool1d, F.avg_pool1d), (1, 8, 40), 2, 1, 1, False)

    def test_AvgPool2d(self):
        test = self._test_AvgPool()
        test((nn.AvgPool2d, F.avg_pool2d), (4, 8, 40, 30), 4, 3, 2)
        test((nn.AvgPool2d, F.avg_pool2d), (1, 64, 32, 32), (3, 2), (1, 2), (0, 1))
        test((nn.AvgPool2d, F.avg_pool2d), (1, 64, 32, 32), (3, 2), (1, 2), (1, 1), False)

    def test_AvgPool3d(self):
        test = self._test_AvgPool()
        test((nn.AvgPool3d, F.avg_pool3d), (4, 8, 64, 64, 64), 4, 3, 2)
        test((nn.AvgPool3d, F.avg_pool3d), (1, 3, 20, 30, 40), (3, 3, 2), (1, 1, 1), (1, 0, 1))

    #######################################################################
    # MaxPooling
    # ------------
    def _test_MaxPool(self):

        def test_case(pool_funs, input_shape, kernel_size, stride, padding):
            fun1, fun2 = pool_funs

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.pooling = fun1(5, 2)

                def forward(self, x):
                    y = self.pooling(x)
                    z = fun2(y, kernel_size, stride=stride, padding=padding)
                    return z

            self.trace_and_test([input_shape], Model())

        return test_case

    def test_MaxPool1d(self):
        test = self._test_MaxPool()
        test((nn.MaxPool1d, F.max_pool1d), (4, 8, 40), 4, 3, 2)
        test((nn.MaxPool1d, F.max_pool1d), (1, 8, 40), 2, 1, 0)

    def test_MaxPool2d(self):
        test = self._test_MaxPool()
        test((nn.MaxPool2d, F.max_pool2d), (4, 8, 40, 30), 4, 3, 2)
        test((nn.MaxPool2d, F.max_pool2d), (1, 64, 32, 32), (3, 2), (1, 2), (0, 1))
        test((nn.MaxPool2d, F.max_pool2d), (1, 64, 32, 32), (3, 2), (1, 2), (1, 1))

    def test_MaxPool3d(self):
        test = self._test_MaxPool()
        test((nn.MaxPool3d, F.max_pool3d), (4, 8, 10, 64, 64), 2, 1, 1)
        test((nn.MaxPool3d, F.max_pool3d), (1, 3, 10, 30, 40), (3, 3, 2), (1, 2, 1), (1, 0, 1))

    #######################################################################
    # Binary Base
    # ------------
    def _test_binary(self, op_type, in0_shape, in1_shape, alpha=None):

        _alpha = {}
        if alpha:
            _alpha = dict(alpha=alpha)

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.weight = torch.randn(in1_shape)

            def forward(self, x):
                y0 = x + 3
                y1 = op_type(self.weight, y0, **_alpha)
                y2 = op_type(y0, y1, **_alpha)
                return y2

        self.trace_and_test([in0_shape], Model())

    #######################################################################
    # Add
    # ------------
    def test_Add(self):
        """Add"""

        self._test_binary(torch.add, (1, 3, 32, 32), (1, 3, 32, 32), 3)
        self._test_binary(torch.add, (2, 32, 16), (2, 1, 16), 3)
        self._test_binary(torch.add, (32, 32), (32))

    #######################################################################
    # Sub
    # ------------
    def test_Sub(self):
        """Sub"""

        self._test_binary(torch.sub, (1, 3, 32, 31), (1, 3, 32, 1), 3)
        self._test_binary(torch.sub, (2, 32, 16), (2, 1, 16), 3)
        self._test_binary(torch.sub, (32, 32), (32))

    #######################################################################
    # Mul
    # ------------
    def test_Mul(self):
        """Mul"""

        self._test_binary(torch.multiply, (1, 3, 32, 31), (1, 3, 32, 1))
        self._test_binary(torch.multiply, (2, 32, 16), (2, 1, 16))
        self._test_binary(torch.multiply, (32, 32), (32))

    #######################################################################
    # Div
    # ------------
    def test_Div(self):
        """Div"""

        self._test_binary(torch.div, (1, 3, 32, 31), (1, 3, 32, 1))
        self._test_binary(torch.div, (2, 32, 16), (2, 1, 16))
        self._test_binary(torch.div, (32, 32), (32))

    #######################################################################
    # Compare
    # ------------
    def test_Compare(self):
        """Compare
        greater, greater_equal, less, less_equal, equal

        """

        def test_cmp_const(cmp_fun, input_shape, const):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = cmp_fun(x, const)
                    return y

            self.trace_and_test([input_shape], Model())

        self._test_binary(torch.greater, (1, 3, 32, 31), (1, 3, 32, 1))
        self._test_binary(torch.greater, (2, 32, 16), (2, 1, 16))
        self._test_binary(torch.greater, (32, 32), (32))
        self._test_binary(torch.greater_equal, (1, 3, 32, 31), (1, 3, 32, 1))
        self._test_binary(torch.less, (1, 3, 32, 31), (1, 3, 32, 1))
        self._test_binary(torch.less_equal, (1, 3, 32, 31), (1, 3, 32, 1))
        self._test_binary(torch.eq, (1, 3, 32, 31), (1, 3, 32, 1))
        test_cmp_const(torch.greater, (1, 2, 3, 4), 0)
        test_cmp_const(lambda x, y: y > x, (1, 2, 3, 4), 0)
        test_cmp_const(torch.greater_equal, (1, 2, 3, 4), 0)
        test_cmp_const(lambda x, y: y >= x, (1, 2, 3, 4), 0)
        test_cmp_const(torch.less, (1, 2, 3, 4), 0)
        test_cmp_const(lambda x, y: y < x, (1, 2, 3, 4), 0)
        test_cmp_const(torch.less_equal, (1, 2, 3, 4), 0)
        test_cmp_const(lambda x, y: y <= x, (1, 2, 3, 4), 0)
        test_cmp_const(torch.eq, (1, 2, 3, 4), 0)
        test_cmp_const(lambda x, y: y == x, (1, 2, 3, 4), 0)

    #######################################################################
    # LayerNorm
    # ------------
    def test_LayerNorm(self):
        normalize_shape = [40, 80]

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.layer_norm = nn.LayerNorm(normalize_shape, elementwise_affine=True)
                self.prelu = nn.PReLU()

            def forward(self, x):
                x = self.layer_norm(x)
                y = self.prelu(x)
                return x, y

        input_shape = [14, 25] + normalize_shape
        self.trace_and_test([input_shape], Model())

    #######################################################################
    # PRelu
    # ------------
    def test_PRelu(self):
        """PRelu"""

        def _test_prelu(input_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    # self.weight0 = torch.randn(1)
                    self.weight1 = torch.randn(input_shape[1])

                def forward(self, x):
                    # y0 = F.prelu(x, self.weight0)
                    y1 = F.prelu(x, self.weight1)
                    return y1

            self.trace_and_test([input_shape], Model())

        _test_prelu((1, 3, 32, 32))
        _test_prelu((2, 32, 16))
        _test_prelu((32, 32))

    #######################################################################
    # Gather
    # ------------
    def test_Gather(self):
        """Gather"""

        def _test_gather(in0_shape, in1_shape, dim=None):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    # if dim is None:
                    #   y1 = torch.concat((x, self.weight))
                    # else:
                    #   y1 = torch.concat((x, self.weight), dim=dim)
                    return

            self.trace_and_test([in0_shape], Model())

        _test_gather((1, 3, 32, 32), (1, 6, 32, 32), 1)
        _test_gather((2, 32, 16), (3, 32, 16))
        _test_gather((32, 32), (1, 32), 0)

    #######################################################################
    # Permute
    # ------------
    def test_Permute(self):
        """Permute"""

        def _test_permute(in_shape, dims):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y1 = torch.permute(x, dims=dims)
                    return y1

            self.trace_and_test([in_shape], Model())

        _test_permute((1, 3, 32, 32), (0, 3, 1, 2))
        _test_permute((2, 32, 16), (2, 0, 1))
        _test_permute((32, 32), (1, 0))

    #######################################################################
    # T
    # ------------
    def test_T(self):
        """T"""

        def _test_t(in_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    x = torch.concat((x, x))
                    x = torch.t(x)
                    y1 = torch.concat((x, x))
                    return y1

            self.trace_and_test([in_shape], Model())

        _test_t((32, 32))
        _test_t((32, ))

    #######################################################################
    # Tile
    # ------------
    def test_Tile(self):
        """Tile"""

        def _test_tile(in_shape, repeats):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y1 = torch.tile(x, repeats)
                    return y1

            self.trace_and_test([in_shape], Model())

        _test_tile((1, 3, 32, 32), (1, 3, 1, 2))
        _test_tile((2, 32, 16), (2, 1))
        _test_tile((32, 16), (1, 2, 1))

    #######################################################################
    # Transpose
    # ------------
    def test_Transpose(self):
        """Transpose"""

        def _test_transpose(in_shape, dims):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y1 = torch.transpose(x, dim0=dims[0], dim1=dims[1])
                    return y1

            self.trace_and_test([in_shape], Model())

        _test_transpose((1, 3, 32, 32), (0, 3))
        _test_transpose((2, 32, 16), (2, 0))
        _test_transpose((32, 32), (1, 0))

    #######################################################################
    # View
    # ------------
    def test_View(self):
        """View"""

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                a = x.view(64, -1, 1024) # 64, 8, 1024
                b = a.transpose(0, 1) # 8, 64, 1024
                c = b.view(8, 64, 64, 16) # 64, 8, 64, 16
                d = c.transpose(1, 2) # 64, 64, 8, 16
                return d

        in_shape = (512,1024)
        self.trace_and_test([in_shape], Model())


    #######################################################################
    # Where
    # ------------
    def test_Where(self):
        """Where"""

        def _test_where(in0_shape, in1_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.weight = torch.randn(in1_shape)

                def forward(self, x):
                    y = torch.where(x > 0, 1, 0)
                    y1 = torch.where(self.weight > 0, y, x)
                    return y1

            self.trace_and_test([in0_shape], Model())

        _test_where((1, 3, 32, 32), (1, 3, 32, 32))
        _test_where((3, 32, 16), (3, 32, 16))
        # TODO: where backend do not support broadcast
        # _test_where((2, 32, 16), (32, 1))
        # _test_where((32, 32), (1, 32))

    #######################################################################
    # IndexSelect
    # ------------
    def test_IndexSelect(self):
        """IndexSelect"""

        def _test_index_select(in_shape, dim, index_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    high = in_shape[dim]
                    self.index = torch.randint(0, high, index_shape)

                def forward(self, x):
                    y1 = torch.index_select(x, dim, self.index)
                    return y1

            self.trace_and_test([in_shape], Model())

        _test_index_select((1, 3, 32, 32), 2, (5, ))
        _test_index_select((2, 32, 16), 0, (3, ))
        _test_index_select((32, 5), 1, (6, ))

    #######################################################################
    # Scatter
    # ------------
    def test_Scatter(self, case_name):
        """Scatter"""

        # in_shape.dims == index_shape.dims && in_shape[i] >=
        def _test_scatter(in_shape, dim, index_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    high = in_shape[dim]
                    self.index = torch.randint(0, high, index_shape)

                def forward(self, x, updates):
                    y1 = torch.scatter(x, dim, self.index, updates)
                    return y1

            self.trace_and_test([in_shape, index_shape], Model())

        _test_scatter((1, 3, 32, 32), 2, (1, 3, 24, 12))
        # _test_scatter((2, 32, 16), 1, (2, 12, 4))
        _test_scatter((32, 5), 0, (13, 5))

    #######################################################################
    # Concat
    # ------------
    def test_Concat(self):
        """Concat"""

        def _test_concat(in0_shape, in1_shape, dim=None):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.weight = torch.randn(in1_shape)

                def forward(self, x):
                    if dim is None:
                        y1 = torch.concat((x, self.weight))
                    else:
                        y1 = torch.concat((x, self.weight), dim=dim)
                    return y1

            self.trace_and_test([in0_shape], Model())

        _test_concat((1, 3, 32, 32), (1, 6, 32, 32), 1)
        _test_concat((2, 32, 16), (3, 32, 16))
        _test_concat((32, 32), (1, 32), 0)

    #######################################################################
    # Dropout
    # ------------
    def test_Dropout(self):
        """Dropout"""

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.dropout = nn.Dropout(0)

            def forward(self, x):
                a = x.view(64, -1, 1024) # 64, 8, 1024
                b = a.transpose(0, 1) # 8, 64, 1024
                c = self.dropout(b)
                d = c.transpose(1, 2) # 8, 1024, 64
                return d

        in_shape = (512,1024)
        self.trace_and_test([in_shape], Model())

    #######################################################################
    # Elu
    # ------------
    def test_Elu(self):
        """Elu"""

        def _test_elu(input_shape, alpha=1.0):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return F.elu(x)

            self.trace_and_test([input_shape], Model())

        _test_elu((1, 3, 32, 32), 2.0)
        _test_elu((3, 16, 32), 3.5)
        _test_elu((64, 32))

    #######################################################################
    # Activation
    # ------------
    def test_Activation(self):
        """Activation Functions Without Extra Arguments"""

        def _test_activation(op_type, in_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.activation = op_type()

                def forward(self, x):
                    return self.activation(x)

            self.trace_and_test([in_shape], Model())

        for op_type in [
                nn.GELU, nn.Hardsigmoid, nn.Hardswish, nn.LogSigmoid, nn.Mish, nn.ReLU, nn.ReLU6,
                nn.Sigmoid, nn.SiLU, nn.Softplus, nn.Softsign, nn.Tanh
        ]:
            _test_activation(op_type, (1, 3, 32, 32))
            _test_activation(op_type, (3, 16, 32))
            _test_activation(op_type, (64, 32))

    #######################################################################
    # LeakyRelu
    # ------------
    def test_LeakyRelu(self):
        """LeakyRelu"""

        def _test_leaky_relu(in_shape, alpha):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return F.leaky_relu(x, negative_slope=alpha)

            self.trace_and_test([in_shape], Model())

        for alpha in [0.67, -0.2]:
            _test_leaky_relu((1, 3, 32, 32), alpha)
            _test_leaky_relu((3, 16, 32), alpha)
            _test_leaky_relu((64, 32), alpha)

    #######################################################################
    # LSTM
    # ------------
    def test_LSTM(self):

        def _test_lstm0(batch, bidir):
            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.rnn = nn.LSTM(input_size=100, hidden_size=128, bidirectional=bidir)

                def forward(self, x, h_0, c_0):
                    Y, (Y_h, Y_c) = self.rnn(x, (h_0, c_0))
                    return Y, Y_h, Y_c
            num_dir = 2 if bidir else 1
            input_shapes = [(81, batch, 100), (num_dir, batch, 128), (num_dir, batch, 128)]
            self.trace_and_test(input_shapes, Model())

        def _test_lstm1(batch, bidir):
            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.rnn = nn.LSTM(input_size=100, hidden_size=128, bidirectional=bidir)

                def forward(self, x):
                    Y, _ = self.rnn(x)
                    return Y
            input_shapes = [(81, batch, 100)]
            self.trace_and_test(input_shapes, Model())

        for bidir in [True, False]:
            for batch in [1, 4]:
                _test_lstm0(batch, bidir)
                _test_lstm1(batch, bidir)

    #######################################################################
    # GRU
    # ------------
    def test_GRU(self):

        def _test_gru0(batch, bidir):
            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.rnn = nn.GRU(input_size=100, hidden_size=128, bidirectional=bidir)

                def forward(self, x, h_0):
                    Y, Y_h = self.rnn(x, h_0)
                    return Y, Y_h
            num_dir = 2 if bidir else 1
            input_shapes = [(81, batch, 100), (num_dir, batch, 128)]
            self.trace_and_test(input_shapes, Model())

        def _test_gru1(batch, bidir):
            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.rnn = nn.GRU(input_size=100, hidden_size=128, bidirectional=bidir)

                def forward(self, x):
                    Y, _ = self.rnn(x)
                    return Y
            input_shapes = [(81, batch, 100)]
            self.trace_and_test(input_shapes, Model())

        for bidir in [True, False]:
            for batch in [1, 4]:
                _test_gru0(batch, bidir)
                _test_gru1(batch, bidir)

    #######################################################################
    # Softmax
    # ------------
    def test_Softmax(self):
        """Softmax"""

        def _test_softmax(in_shape, dim):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return F.softmax(x, dim=dim)

            self.trace_and_test([in_shape], Model())

        for dim in [1, 2]:
            _test_softmax((3, 100, 10, 1), dim)
            _test_softmax((3, 100, 32), dim)
            _test_softmax((3, 100, 32, 1), dim)

    #######################################################################
    # LogSoftmax
    # ------------
    def test_LogSoftmax(self):
        """LogSoftmax"""

        def _test_log_softmax(in_shape, dim):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return F.log_softmax(x, dim=dim)

            self.trace_and_test([in_shape], Model())

        for dim in [1, 2]:
            _test_log_softmax((3, 100, 10, 1), dim)
            _test_log_softmax((3, 100, 32), dim)
            _test_log_softmax((3, 100, 32, 1), dim)

    #######################################################################
    # Softmin
    # ------------
    def test_Softmin(self):
        """Softmin"""

        def _test_softmin(in_shape, dim):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return F.softmin(x, dim=dim)

            self.trace_and_test([in_shape], Model())

        for dim in [1, 2]:
            _test_softmin((3, 100, 10, 1), dim)
            _test_softmin((3, 100, 32), dim)
            _test_softmin((3, 100, 32, 1), dim)

    #######################################################################
    # Pad1D
    # ------------
    def test_Pad1d(self):
        """Pad 1D (ReflectionPad1d, ReplicationPad1d, ConstantPad1d)
        Input: (N,C,W) or (C, W)
        Padding: int (pad_left == pad_right) or tuple (pad_left, pad_right)
      """

        def _test_pad1d(op_type, input_shape, padding: Union[int, tuple], value=0.):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    if op_type == nn.ConstantPad1d:
                        self.pad1d = op_type(padding, value)
                    else:
                        self.pad1d = op_type(padding)

                def forward(self, x):
                    return self.pad1d(x)

            self.trace_and_test([input_shape], Model())

        for op_type in [nn.ReflectionPad1d, nn.ReplicationPad1d, nn.ConstantPad1d]:
            _test_pad1d(op_type, (1, 3, 32), 5, 0.6)
            _test_pad1d(op_type, (2, 3, 32), (5, 3))
            _test_pad1d(op_type, (64, 32), (3, 6), 0.6)
            _test_pad1d(op_type, (64, 32), 7, 0.4)

    #######################################################################
    # Pad2D
    # ------------
    def test_Pad2d(self):
        """Pad 2D (ReflectionPad2d, ReplicationPad2d, ConstantPad2d, ZeroPad2d)
        Input: (C, H, W) or (N, C, H, W)
        Padding: int (pad_left == pad_right == pad_top == pad_bottom)
                  or tuple (pad_left, pad_right, pad_top, pad_bottom)
      """

        def _test_pad2d(op_type, input_shape, padding: Union[int, tuple], value=0.):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    if op_type == nn.ConstantPad2d:
                        self.pad2d = op_type(padding, value)
                    else:
                        self.pad2d = op_type(padding)

                def forward(self, x):
                    return self.pad2d(x)

            self.trace_and_test([input_shape], Model())

        for op_type in [nn.ReflectionPad2d, nn.ReplicationPad2d, nn.ZeroPad2d, nn.ConstantPad2d]:
            _test_pad2d(op_type, (1, 16, 32), 7, 0.6)
            _test_pad2d(op_type, (3, 16, 32), (4, 6, 7, 8))
            _test_pad2d(op_type, (1, 3, 16, 32), 3, 0.5)
            _test_pad2d(op_type, (2, 4, 16, 32), (3, 4, 5, 6), 0.4)

    def test_Abs(self):
        """Abs"""

        def _test_abs(input_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return torch.abs(x)

            self.trace_and_test([input_shape], Model())

        _test_abs((1, 16, 32, 32))


def test_one_case_in_all(tester: TORCH_IR_TESTER, case, error_cases, success_cases):
    try:
        tester.test_single(case)
    except:
        error_cases.append(case)
        traceback.print_exc()
        return
    success_cases.append(case)


def test_all(tester: TORCH_IR_TESTER):
    import multiprocessing
    process_number = multiprocessing.cpu_count() // 2 + 1
    processes = []
    error_cases = multiprocessing.Manager().list()
    success_cases = multiprocessing.Manager().list()
    for case in tester.test_cases:
        if tester.check_support(case):
            p = multiprocessing.Process(target=test_one_case_in_all,
                                        args=(tester, case, error_cases, success_cases))
            processes.append(p)
        if len(processes) == process_number:
            for p in processes:
                p.start()
            for j in processes:
                j.join()
            processes = []
    if processes:
        for p in processes:
            p.start()
        for j in processes:
            j.join()
    # error_cases = []
    # success_cases = []
    # for case in tester.test_cases:
    #     if tester.check_support(case):
    #         test_one_case_in_all(tester, case, error_cases, success_cases)
    print("Success: {}".format(success_cases))
    print("Failure: {}".format(error_cases))
    if error_cases:
        print("====== test_torch.py --chip {} TEST Failed ======".format(tester.chip))
        exit(1)
    else:
        print("====== test_torch.py --chip {} TEST Success ======".format(tester.chip))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--chip", default="bm1684x", type=str,
                        choices=['bm1684x', 'bm1686', 'cv183x'], help="chip platform name")
    parser.add_argument("--case", default="all", type=str, help="test one case, if all, then test all cases")
    parser.add_argument("--mode", default="all", type=str, choices=['all', 'f32', 'f16', 'bf16', 'int8'],
                        help="chip platform name")
    parser.add_argument("--debug", action="store_true", help='keep middle file if debug')
    parser.add_argument("--show_all", action="store_true", help='show all cases')
    # yapf: enable
    args = parser.parse_args()
    tester = TORCH_IR_TESTER(args.chip, args.mode)
    if args.show_all:
        print("====== Show All Cases ============")
        for case in tester.test_cases:
            print(case)
        exit(0)
    dir = "torch_test_{}".format(args.chip)
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)
    if args.case == "" or args.case.lower() == "all":
        test_all(tester)
    else:
        tester.test_single(args.case)
    if args.debug == False:
        file_clean()
