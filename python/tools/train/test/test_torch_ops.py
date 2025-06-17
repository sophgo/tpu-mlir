#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import numpy as np
from typing import List, Union

from tools.model_runner import mlir_inference, model_inference, torch_inference, show_fake_cmd
from tools.npz_tool import npz_compare
from tools.model_transform import *
from utils.mlir_shell import *
import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
import traceback

from torchvision.ops import DeformConv2d
from tools.train.tpu_mlir_jit import device, aot_backend, cosine_similarity
import tools.train.tpu_mlir_jit as tpu_mlir_jit


class TORCH_IR_TESTER(object):
    ID = 0
    CURRENT_CASE = ""

    # This class is built for testing single operator transform.
    def __init__(self, enable_thread: bool = False):
        # yapf: disable
        self.test_cases = {
            "Conv2d":           self.test_Conv2d,
            "Abs":              self.test_Abs,
            "Activation":       self.test_Activation,
            "AdaptiveAvgPool2d":self.test_AdaptiveAvgPool2d,
            "Add":              self.test_Add,
            "Add2":             self.test_Add2,
            "Add5d":            self.test_Add5d,
            "Addmm":            self.test_Addmm,
            "Arange":           self.test_Arange,
            "Attention":        self.test_Attention,
            "AttentionNew":     self.test_AttentionNew,
            "AvgPool1d":        self.test_AvgPool1d,
            "AvgPool2d":        self.test_AvgPool2d,
            "AvgPool3d":        self.test_AvgPool3d,
            "BatchNorm":        self.test_BatchNorm,
            "BMM":              self.test_BatchMatMul,
            "Ceil":             self.test_Ceil,
            "ChannelShuffle":   self.test_ChannelShuffle,
            "Chunk":            self.test_Chunk,
            "Clamp":            self.test_Clamp,
            "Compare":          self.test_Compare,
            "Concat":           self.test_Concat,
            "ConstantLike":     self.test_ConstantLike,
            "Conv1d":           self.test_Conv1d,
            "Conv3d":           self.test_Conv3d,
            "ConvMerge":        self.test_ConvMerge,
            "ConvGroup":        self.test_ConvGroup,
            "ConvTrans":        self.test_ConvTrans,
            "ConstantFill":     self.test_ConstantFill,
            "DeformConv2D":     self.test_DeformConv2D,
            "Div":              self.test_Div,
            "Dropout":          self.test_Dropout,
            "Elu":              self.test_Elu,
            "Embedding":        self.test_Embedding,
            "Flatten":          self.test_Flatten,
            "Floor":            self.test_Floor,
            "FloorDiv":         self.test_FloorDiv,
            "Gather":           self.test_Gather,
            "GridSampler":      self.test_GridSampler,
            "GroupNorm":        self.test_GroupNorm,
            "GRU":              self.test_GRU,
            "IndexSelect":      self.test_IndexSelect,
            "InstanceNorm":     self.test_InstanceNorm,
            "Interp":           self.test_Interp,
            "LayerNorm":        self.test_LayerNorm,
            "LeakyRelu":        self.test_LeakyRelu,
            "Linear":           self.test_Linear,
            "LogSoftmax":       self.test_LogSoftmax,
            "LSTM":             self.test_LSTM,
            "MaskedFill":       self.test_MaskedFill,
            "Math":             self.test_Math,
            "MatMul":           self.test_MatMul,
            "Max":              self.test_Max,
            "MaxPool1d":        self.test_MaxPool1d,
            "MaxPool2d":        self.test_MaxPool2d,
            "MaxPool3d":        self.test_MaxPool3d,
            "MeshGrid":         self.test_MeshGrid,
            "Min":              self.test_Min,
            "MM":               self.test_MM,
            "Mul":              self.test_Mul,
            "NewZeros":         self.test_NewZeros,
            "Reduce":           self.test_Reduce,
            "Remainder":        self.test_Remainder,
            "Repeat":           self.test_Repeat,
            "Reshape":          self.test_Reshape,
            "RMSNorm":          self.test_RMSNorm,
            "PixelShuffle":     self.test_PixelShuffle,
            "PixelUnshuffle":   self.test_PixelUnshuffle,
            "PRelu":            self.test_PRelu,
            "PRelu":            self.test_PRelu,
            "Permute":          self.test_Permute,
            "Permute2":         self.test_Permute2,
            "Pad1d":            self.test_Pad1d,
            "Pad2d":            self.test_Pad2d,
            "Pow":              self.test_Pow,
            "Scatter":          self.test_Scatter,
            "Select":           self.test_Select,
            "Slice":            self.test_Slice,
            "Softmax":          self.test_Softmax,
            "Softmin":          self.test_Softmin,
            "Split":            self.test_Split,
            "Squeeze":          self.test_Squeeze,
            "Stack":            self.test_Stack,
            "Sub":              self.test_Sub,
            "T":                self.test_T,
            "Tile":             self.test_Tile,
            "To":               self.test_To,
            "Type_as":          self.test_Type_as,
            "Transpose":        self.test_Transpose,
            "Upsample":         self.test_Upsample,
            "Unary":            self.test_Unary,
            "Unsqueeze":        self.test_Unsqueeze,
            "View":             self.test_View,
            "Where":            self.test_Where,
            ## Special Case
            "Connect":          self.test_Connect,
            "InfError":         self.test_InfError,
            "SplitReshape":     self.test_SplitReshape,
            "WeightMultiUse":   self.test_WeightMultiUse
        }
        # yapf: enable

        self.multithread = enable_thread

    class Desc():

        def __init__(self, dtype, min=-10, max=10) -> None:
            self.dtype = dtype
            self.min = min
            self.max = max

    def test_single(self, case: str):
        np.random.seed(0)
        torch.manual_seed(7)
        TORCH_IR_TESTER.ID = 0
        TORCH_IR_TESTER.CURRENT_CASE = case
        print("Test: {}".format(case))
        if case in self.test_cases:
            os.makedirs(case, exist_ok=True)
            os.chdir(case)
            self.test_cases[case]()
            print("====== TEST {} Success ======".format(case))
        else:
            raise RuntimeError("case [{}] is not exist".format(case))

    def generate_random(self, shape, dtype='float32', min=-10, max=10):
        scale = max - min
        return (np.random.rand(*shape) * scale + min).astype(dtype)

    def create_random_input(self, shapes, descs: List[Desc]):
        if len(descs) == 0:
            inputs = [self.generate_random(s) for s in shapes]
        else:
            inputs = list()
            for i in range(len(shapes)):
                inputs.append(
                    self.generate_random(shapes[i], descs[i].dtype, descs[i].min, descs[i].max))
        inputs = [torch.from_numpy(inp).to(device) for inp in inputs]
        for i in inputs:
            i.requires_grad = True
        return inputs

    def trace_and_test(self, in_shapes, torch_model: nn.Module, descs: List[Desc] = []):
        """Generic function to generate and compare torch and Tpu-Mlir output"""
        TORCH_IR_TESTER.ID += 1
        inputs = self.create_random_input(in_shapes, descs)
        mlir_model = torch.compile(torch_model.to(device), backend=aot_backend)
        # with torch.enable_grad():
        out = mlir_model(*inputs)
        # if isinstance(out, (tuple, list)):
        #     out = out[0]
        # out.sum().backward()

        res_oringal = torch_model(*inputs)
        # res_oringal.sum().backward()
        cosine_similarity(res_oringal, out)

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

        def case3(conv_fun,
                  input_shape,
                  kernel_shape_0,
                  oc_0,
                  kernel_shape_1,
                  oc_1,
                  has_bias_0=False,
                  padding_0: Union[int, str, List[int]] = 0,
                  stride_0: Union[int, List[int]] = 1,
                  dilation_0: Union[int, List[int]] = 1,
                  group_0=1,
                  has_bias_1=False,
                  padding_1: Union[int, str, List[int]] = 0,
                  stride_1: Union[int, List[int]] = 1,
                  dilation_1: Union[int, List[int]] = 1,
                  group_1=1):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    filter_shape_0 = (oc_0, input_shape[1] // group_0, *kernel_shape_0)
                    self.filter_0 = torch.randn(filter_shape_0)
                    self.bias_0 = torch.randn(oc_0) if has_bias_0 else None

                    filter_shape_1 = (oc_1, oc_0 // group_1, *kernel_shape_1)
                    self.filter_1 = torch.randn(filter_shape_1)
                    self.bias_1 = torch.randn(oc_1) if has_bias_1 else None

                def forward(self, x):
                    y = conv_fun(x,
                                 self.filter_0,
                                 bias=self.bias_0,
                                 padding=padding_0,
                                 stride=stride_0,
                                 dilation=dilation_0,
                                 groups=group_0)

                    z = conv_fun(y,
                                 self.filter_1,
                                 bias=self.bias_1,
                                 padding=padding_1,
                                 stride=stride_1,
                                 dilation=dilation_1,
                                 groups=group_1)
                    return z

            self.trace_and_test([input_shape], Model())

        return dict(case1=case1, case2=case2, case3=case3)

    def test_Conv1d(self):
        """Conv 1D"""

        test = self._test_Conv()
        test["case1"](nn.Conv1d, (4, 8, 28))
        test["case2"](F.conv1d, (1, 3, 32), [3], 12, has_bias=True, group=1, padding="same")
        test["case2"](F.conv1d, (2, 32, 16), [5], 64, padding=2, stride=2, dilation=1)
        # Tpu/Interfaces/BM1684X/Conv1D.cpp::152 Not supported yet.
        test["case2"](F.conv1d, (1, 3, 32), [3], 12, group=3, padding=1, stride=2)

    def test_Conv2d(self):
        """Conv 2D"""

        test = self._test_Conv()
        test["case1"](nn.Conv2d, (4, 8, 28, 28))
        test["case2"](F.conv2d, (1, 3, 32, 32), (3, 3), 12, has_bias=True, group=1, padding="same")
        test["case2"](F.conv2d, (2, 32, 16, 16), (5, 5), 64, padding=2, stride=2, dilation=1)

    def test_ConvGroup(self):
        test = self._test_Conv()
        test["case2"](F.conv2d, (1, 8, 32, 32), (3, 3), 24, group=4, padding=(1, 1))

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

    def test_ConvMerge(self):
        """Conv Merge"""
        test = self._test_Conv()
        test["case3"](F.conv2d, (1, 3, 4, 4), (1, 1),
                      2, (3, 3),
                      4,
                      has_bias_0=False,
                      padding_0=0,
                      has_bias_1=False,
                      padding_1=0)
        test["case3"](F.conv2d, (2, 32, 16, 16), (1, 1),
                      64, (3, 3),
                      16,
                      has_bias_0=True,
                      padding_0=0,
                      has_bias_1=True,
                      padding_1=0)
        test["case3"](F.conv2d, (2, 32, 32, 32), (1, 1),
                      12, (3, 3),
                      64,
                      has_bias_0=True,
                      padding_0=1,
                      has_bias_1=True,
                      padding_1=1)

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
        if not self.is_cv18xx:
            test((nn.AvgPool2d, F.avg_pool2d), (1, 64, 32, 32), (3, 2), (2, 3), (1, 1), False)

    def test_AvgPool3d(self):
        test = self._test_AvgPool()
        test((nn.AvgPool3d, F.avg_pool3d), (4, 8, 12, 20, 24), 4, 3, 2)
        test((nn.AvgPool3d, F.avg_pool3d), (1, 3, 12, 20, 24), (3, 3, 2), (1, 1, 1), (1, 0, 1))

    #######################################################################
    # Max
    # ------------
    def test_Max(self):

        def _test_max(shape, dim, ret_case):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    max_values, max_indices = torch.max(x, dim=dim)
                    if ret_case == 0:
                        return max_indices
                    elif ret_case == 1:
                        return max_values
                    else:
                        return max_values, max_indices

            self.trace_and_test([shape], Model())

        for ret_case in [0, 1, 2]:
            _test_max((4, 30), 1, ret_case)
            _test_max((1, 3, 64, 64), 3, ret_case)
            # _test_max((4, 384), 0, ret_case)

    #######################################################################
    # Min
    # ------------
    def test_Min(self):

        def _test_min(shape, dim, ret_case):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    min_values, min_indices = torch.min(x, dim=dim)
                    if ret_case == 0:
                        return min_indices
                    elif ret_case == 1:
                        return min_values
                    else:
                        return min_values, min_indices

            self.trace_and_test([shape], Model())

        for ret_case in [0, 1, 2]:
            _test_min((4, 30), 1, ret_case)
            _test_min((1, 3, 64, 64), 3, ret_case)
            # _test_min((4, 384), 0, ret_case)

    #######################################################################
    # MaxPooling
    # ------------
    def _test_MaxPool(self):

        def test_case(pool_funs, input_shape, kernel_size, stride, padding, ceil_mode=False):
            fun1, fun2 = pool_funs

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.pooling = fun1(5, 2)

                def forward(self, x):
                    y = self.pooling(x)
                    z = fun2(
                        y,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        ceil_mode=ceil_mode,
                    )
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
        test((nn.MaxPool2d, F.max_pool2d), (1, 64, 32, 32), (3, 5), (1, 2), (1, 1))
        test((nn.MaxPool2d, F.max_pool2d), (1, 64, 32, 32), (4, 3), (3, 2), (1, 1), True)

    def test_MaxPool3d(self):
        test = self._test_MaxPool()
        test((nn.MaxPool3d, F.max_pool3d), (4, 8, 10, 64, 64), 2, 1, 1)
        test((nn.MaxPool3d, F.max_pool3d), (1, 3, 10, 30, 40), (3, 3, 2), (1, 2, 1), (1, 0, 1))
        test((nn.MaxPool3d, F.max_pool3d), (1, 3, 10, 30, 40), (3, 3, 5), (1, 2, 2), (1, 0, 1),
             True)

    #######################################################################
    # Binary Base
    # ------------
    def _test_binary(self, op_type, in0_shape, in1_shape, alpha=None, is_reverse=False, min=-10):

        _alpha = {}
        if alpha:
            _alpha = dict(alpha=alpha)

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.weight = torch.randn(in1_shape)

            def forward(self, x):
                if is_reverse:
                    y0 = 3 - x
                else:
                    y0 = x + 3
                y1 = op_type(self.weight, y0, **_alpha)
                y2 = op_type(y0, y1, **_alpha)
                return y2

        self.trace_and_test([in0_shape], Model(), [self.Desc('float32', min)])

    #######################################################################
    # Add2
    # ------------
    def test_Add2(self):
        """Add2"""

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                y1 = torch.add(x, x)
                return y1

        self.trace_and_test([(1, 3, 32, 32)], Model(), [self.Desc('float32', -10)])

    #######################################################################
    # Add
    # ------------
    def test_Add(self):
        """Add"""

        self._test_binary(torch.add, (1, 3, 32, 32), (1, 3, 32, 32), 3)
        self._test_binary(torch.add, (2, 32, 16), (2, 1, 16), 3)
        self._test_binary(torch.add, (32, 32), (32))

    def test_Add5d(self):
        """AddError"""

        self._test_binary(torch.add, (1, 4, 12, 147, 147), (1, 4, 1, 147, 147))

    #######################################################################
    # Sub
    # ------------
    def test_Sub(self):
        """Sub"""

        self._test_binary(torch.sub, (1, 3, 32, 31), (1, 3, 32, 1), 3)
        self._test_binary(torch.sub, (2, 32, 16), (2, 1, 16), 3, is_reverse=True)
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
        if self.chip != "bm1688":
            self._test_binary(torch.div, (1, 3, 32, 31), (1, 3, 32, 1), min=0)
            self._test_binary(torch.div, (32, 32), (32), min=0)
        self._test_binary(torch.div, (2, 32, 16), (2, 1, 16), min=0)

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
        self._test_binary(torch.eq, (1, 3, 32, 31), (1, 3, 32, 1), min=0)
        self._test_binary(torch.ne, (1, 3, 32, 31), (1, 3, 32, 1), min=0)
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
        test_cmp_const(torch.ne, (1, 2, 3, 4), 0)
        test_cmp_const(lambda x, y: y != x, (1, 2, 3, 4), 0)

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
    # RMSNorm
    # ------------
    def test_RMSNorm(self):
        normalize_shape = [80]
        eps = 1e-5

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.weight = torch.nn.Parameter(torch.randn(normalize_shape))
                self.prelu = nn.PReLU()

            def forward(self, hidden_states: torch.Tensor):
                input_dtype = hidden_states.dtype
                variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + eps)
                rmsnorm_out = self.weight * hidden_states

                return rmsnorm_out, self.prelu(rmsnorm_out)

        input_shape = [14, 25, 40] + normalize_shape
        self.trace_and_test([input_shape], Model())

    #######################################################################
    # Chunk
    # ------------
    def test_Chunk(self):

        class Model0(torch.nn.Module):

            def __init__(self):
                super(Model0, self).__init__()

            def forward(self, x):
                a, b, c = torch.chunk(x, 3, -1)
                d = a * b + c
                return d

        class Model1(torch.nn.Module):

            def __init__(self):
                super(Model1, self).__init__()
                self.weight = torch.randn((4, 16, 30))

            def forward(self, x):
                a, b, c = torch.chunk(self.weight, 3, -1)
                d = a * b + c + x
                return d

        self.trace_and_test([(4, 16, 30)], Model0())
        #self.trace_and_test([(4, 16, 10)], Model1())

    #######################################################################
    # Clamp
    # ------------
    def test_Clamp(self):

        class Model0(torch.nn.Module):

            def __init__(self):
                super(Model0, self).__init__()

            def forward(self, x):
                y = x * 30
                return torch.clamp(y, -10, 20)

        self.trace_and_test([(4, 16, 30)], Model0())

    #######################################################################
    # SplitUesless
    # ------------
    def test_SplitReshape(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                a, b, c, d = torch.chunk(x, 4, 1)
                a = torch.reshape(a, (1, 1, -1, 3))
                b = torch.reshape(b, (1, 1, -1, 3))
                c = torch.reshape(c, (1, 1, -1, 3))
                d = torch.reshape(d, (1, 1, -1, 3))
                return a, b, c, d

        self.trace_and_test([(1, 4, 16, 30)], Model())

    #######################################################################
    # InfError
    # ------------
    def test_InfError(self):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, mask):
                x = torch.masked_fill(x, mask, float("-inf"))
                y = torch.softmax(x, -1)
                return y

        self.trace_and_test(
            [(1, 32, 128, 128), (1, 1, 128, 128)], Model(),
            [self.Desc('float', -10, 10), self.Desc('int', 0, 2)])

    #######################################################################
    # MatMul
    # ------------
    def test_MatMul(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y, z):
                out = torch.matmul(x, y) + z
                return out

        self.trace_and_test([(4, 8, 49, 32), (4, 8, 32, 49), (1, 1, 1, 49)], Model())

    #######################################################################
    # test Connect Pass
    # ------------
    def test_Connect(self):

        def test_connect_(x_shape: tuple, filter_shape: tuple, bias_shape: tuple):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.filter = torch.randn(*filter_shape)
                    self.bias = torch.randn(*bias_shape)

                def forward(self, x):
                    out = torch.matmul(x, self.filter) + self.bias
                    return out

            self.trace_and_test([x_shape], Model())

        test_connect_((2, 4096, 1024), (2, 1024, 4096), (1, 1, 4096))
        test_connect_((2, 1024, 4096), (2, 4096, 1024), (1, 1, 1024))

    #######################################################################
    # test Weight multiple use Pass
    # ------------
    def test_WeightMultiUse(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.filter = torch.randn(32, 64)
                self.bias = torch.randn(1, 64)

            def forward(self, x, y):
                a = torch.matmul(x, self.filter) + self.bias
                b = y + self.filter + self.bias
                return a + b

        self.trace_and_test([(32, 32), (32, 64)], Model())

    #######################################################################
    # ConstantFill
    # ------------
    def test_ConstantFill(self):

        def _test_constant_fill(func, shape, type=None):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = func(shape, dtype=type)
                    z = y + x
                    return z

            self.trace_and_test([shape], Model())

        _test_constant_fill(torch.zeros, (2, 3, 64, 64), torch.float32)
        # _test_constant_fill(torch.zeros, (3, 64, 64))
        # _test_constant_fill(torch.ones, (1, 3, 64, 64), torch.float32)
        _test_constant_fill(torch.ones, (3, 64, 64))

    #######################################################################
    # Embedding
    # ------------
    def test_Embedding(self):

        def _test_embedding(shape, n, d):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.embedding = nn.Embedding(n, d)

                def forward(self, x):
                    y = self.embedding(x)
                    return y

            self.trace_and_test([shape], Model(), [self.Desc('int32', 0, n)])

        _test_embedding((2, 3, 64), 512, 768)
        _test_embedding((2, 64), 20, 30)
        # _test_embedding((1, 384), 30522, 1024)

    #######################################################################
    # To
    # ------------
    def test_To(self):

        def _test_to(shape, dtype):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = x.to(dtype) + 1
                    return y

            self.trace_and_test([shape], Model())

        for dtype in [torch.long, torch.int64, torch.float16]:
            _test_to((2, 3, 64), dtype)

    #######################################################################
    # Type_as
    # ------------
    def test_Type_as(self):

        def _test_type_as(shape, dtype):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.other = torch.tensor(-10, dtype=dtype)

                def forward(self, x):
                    y = x.type_as(self.other) + 1
                    return y

            self.trace_and_test([shape], Model())

        for dtype in [torch.long, torch.int64, torch.float16]:
            _test_type_as((4, 3, 100, 100), dtype)

    #######################################################################
    # Reduce
    # ------------
    def test_Reduce(self):

        def _test_reduce(func, shape, dim=None, keepdim=False):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = func(x, dim, keepdim)
                    return y

            self.trace_and_test([shape], Model())

        # _test_reduce(torch.sum, (2, 3, 64, 64))
        _test_reduce(torch.sum, (1, 3, 64, 64), 1, True)
        _test_reduce(torch.sum, (2, 3, 64, 64), [0, 1, 2])
        # _test_reduce(torch.mean, (2, 3, 64, 64))
        _test_reduce(torch.mean, (1, 3, 64, 64), 1, True)
        _test_reduce(torch.mean, (2, 3, 64, 64), [1, 2])

    #######################################################################
    # Pow
    # ------------
    def test_Pow(self):

        def _test_pow(shape, exp, min=0):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = torch.pow(x, exponent=exp)
                    return y

            self.trace_and_test([shape], Model(), [self.Desc('float32', min)])

        _test_pow((2, 3, 64, 64), 2, -10)
        _test_pow((3, 64, 64), 3)
        _test_pow((64, 64), 0.5)

    #######################################################################
    # MeshGrid
    # ------------
    def test_MeshGrid(self):

        def _test_mesh_grid(shape, indexing=None):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    dim2 = x.size(dim=2)
                    dim3 = x.size(dim=3)
                    in0 = torch.arange(0, dim2, 1)
                    in1 = torch.arange(0, dim3 * 2, 2)
                    y0, y1 = torch.meshgrid(in0, in1, indexing=indexing)
                    if indexing == 'xy':
                        y0 = y0.transpose(0, 1)
                        y1 = y1.transpose(0, 1)
                    y = torch.stack([y0 + x, y1 - x], dim=0)
                    return y

            self.trace_and_test([shape], Model())

        _test_mesh_grid((1, 3, 64, 32), 'ij')
        _test_mesh_grid((1, 3, 32, 64))
        _test_mesh_grid((1, 3, 64, 4), 'xy')

    #######################################################################
    # NewZeros
    # ------------
    def test_NewZeros(self):

        def _test_newzeros(shape, dtype=None):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.coeff = torch.randn(1, 3)

                def forward(self, x):
                    y = self.coeff.new_zeros(shape, dtype=dtype)
                    y = y.to(torch.float32) + x
                    return y

            self.trace_and_test([shape], Model())

        _test_newzeros((2, 3, 64, 64), torch.float32)
        _test_newzeros((3, 64, 64), torch.int32)
        _test_newzeros((64, 64))

    #######################################################################
    # Reshape
    # ------------
    def test_Reshape(self):
        """Reshape"""

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                a = torch.reshape(x, (64, -1, 1024))  # 64, 8, 1024
                b = a.transpose(0, 1).contiguous()  # 8, 64, 1024
                c = torch.reshape(b, (8, 64, 64, 16))  # 64, 8, 64, 16
                d = c.to(torch.float32)
                e = d.transpose(1, 2)  # 64, 64, 8, 16
                return e

        in_shape = (512, 1024)
        self.trace_and_test([in_shape], Model())

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
    # BatchNorm
    # ------------
    def test_BatchNorm(self):
        """BatchNorm"""

        def _test_batchnorm(op_type, input_shape, eps=1e-5, affine=True):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    c = input_shape[1]
                    self.bm = op_type(c, affine=affine, eps=eps)
                    if self.bm.weight is not None:
                        torch.nn.init.uniform_(self.bm.weight.data)
                    if self.bm.bias is not None:
                        torch.nn.init.uniform_(self.bm.bias.data)

                def forward(self, x):
                    y = self.bm(x)

                    return y

            self.trace_and_test([input_shape], Model())

        _test_batchnorm(nn.BatchNorm2d, (3, 4, 32, 32), 3e-5)
        _test_batchnorm(nn.BatchNorm2d, (2, 32, 16, 16), affine=False)
        _test_batchnorm(nn.BatchNorm1d, (3, 32), 3e-5)
        _test_batchnorm(nn.BatchNorm1d, (3, 32, 32), affine=False)
        if not self.is_cv18xx:
            _test_batchnorm(nn.BatchNorm3d, (3, 4, 5, 32, 32), 3e-5)
            _test_batchnorm(nn.BatchNorm3d, (2, 32, 16, 16, 3), affine=False)

    #######################################################################
    # InstanceNorm
    # ------------
    def test_InstanceNorm(self):
        """InstanceNorm"""

        def _test_instancenorm(op_type, input_shape, feature, eps=1e-5, affine=True):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.im = op_type(feature, affine=affine, eps=eps)

                def forward(self, x):
                    y = self.im(x)

                    return y

            self.trace_and_test([input_shape], Model())

        _test_instancenorm(nn.InstanceNorm2d, (3, 4, 32, 32), 4, 3e-5)
        # TODO: support squeeze & unsqueeze
        # _test_instancenorm(nn.InstanceNorm2d, (32, 16, 16), 16, affine=False)
        if (not self.is_cv18xx):
            # _test_instancenorm(nn.InstanceNorm1d, (3, 32), 3, 3e-5)
            _test_instancenorm(nn.InstanceNorm1d, (3, 32, 32), 32, affine=False)
            # _test_instancenorm(nn.InstanceNorm3d, (4, 5, 32, 32), 4, 3e-5)
            _test_instancenorm(nn.InstanceNorm3d, (2, 32, 16, 16, 3), 32, affine=False)

    #######################################################################
    # Interp
    # ------------
    def test_Interp(self):
        """Interp"""

        def _test_interp(input_shape, scale):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = nn.functional.interpolate(x,
                                                  None,
                                                  scale,
                                                  mode='bilinear',
                                                  align_corners=False)
                    return y

            self.trace_and_test([input_shape], Model())

        _test_interp((1, 3, 100, 100), 4)
        _test_interp((1, 1, 32, 32), 10)
        _test_interp((2, 32, 16, 16), 16)

    #######################################################################
    # BMM
    # ------------
    def test_BatchMatMul(self):
        """BatchMatMul"""

        def _test_batchmatmul(input_shape, right_shape, is_cv18xx):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.weight0 = torch.randn(input_shape)
                    self.weight1 = torch.randn(right_shape)

                def forward(self, x, y):
                    z1 = torch.bmm(x, self.weight1)
                    if is_cv18xx:
                        z2 = torch.bmm(x, y)
                    else:
                        z2 = torch.bmm(self.weight0, y)
                    z3 = torch.transpose(z2, 1, 2)
                    z4 = torch.bmm(z1, z3)
                    return z4

            self.trace_and_test([input_shape, right_shape], Model())

        _test_batchmatmul((3, 32, 32), (3, 32, 64), self.is_cv18xx)
        _test_batchmatmul((2, 32, 16), (2, 16, 34), self.is_cv18xx)

    #######################################################################
    # MM
    # ------------
    def test_MM(self):
        """MM"""

        def _test_mm(input_shape, right_shape, is_cv18xx):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.weight0 = torch.randn(input_shape)
                    self.weight1 = torch.randn(right_shape)
                    self.dim = len(input_shape)

                def forward(self, x, y):
                    z1 = torch.mm(x, self.weight1)
                    if is_cv18xx:
                        z2 = torch.mm(x, y)
                    else:
                        z2 = torch.mm(self.weight0, y)
                    z3 = torch.transpose(z2, 1, 0)
                    z4 = torch.mm(z1, z3)
                    return z4

            self.trace_and_test([input_shape, right_shape], Model())

        _test_mm((32, 32), (32, 64), self.is_cv18xx)
        _test_mm((32, 16), (16, 34), self.is_cv18xx)

    #######################################################################
    # Addmm
    # ------------
    def test_Addmm(self):
        """Addmm"""

        def _test_addmm(beta, alpha):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x, y, z):
                    o = torch.addmm(beta, x, alpha, y, z)
                    return o

            self.trace_and_test([(24, 32), (24, 16), (16, 32)], Model())

        _test_addmm(1.0, 1.0)
        # _test_addmm(0.5, 0.3) # need to support add with coeff

    #######################################################################
    # Arange
    # ------------
    def test_Arange(self):
        """Arange"""

        def _test_arange(end, start=None, step=None):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    if start is None:
                        a = torch.arange(end)
                    elif step is None:
                        a = torch.arange(start, end)
                    else:
                        a = torch.arange(start, end, step)
                    b = x + a
                    return b

            sta = start if start is not None else 0
            ste = step if step is not None else 1
            out_size = (end - sta) // ste
            self.trace_and_test([(32, out_size)], Model())

        _test_arange(60, 0, 1)
        _test_arange(60, 1)
        _test_arange(60)
        _test_arange(60, 0, 2)

    #######################################################################
    # Unsqueeze
    # ------------
    def test_Unsqueeze(self):
        """Unsqueeze"""

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y):
                a = torch.unsqueeze(y, 1)
                b = x + a
                return b

        self.trace_and_test([(32, 16, 28), (32, 28)], Model())

    #######################################################################
    # Gather
    # ------------
    def test_Gather(self):
        """Gather"""

        def _test_gather(in0_shape, in1_shape):
            input_1 = self.Desc(np.float32)
            input_2 = self.Desc(np.int64, 0, 5538)

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x, y):
                    return torch.gather(x, 2, y)

            self.trace_and_test([in0_shape, in1_shape], Model(), [input_1, input_2])

        _test_gather((10, 349, 5538), (10, 349, 1))

    #######################################################################
    # GroupNorm
    # ------------
    def test_GroupNorm(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.group_norm = nn.GroupNorm(8, 64)
                nn.init.normal_(self.group_norm.weight, std=0.01)
                nn.init.normal_(self.group_norm.bias, std=0.01)

            def forward(self, x):
                return self.group_norm(x)

        self.trace_and_test([(4, 64, 16, 16)], Model())

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

        # cv183x regression falure
        #_test_permute((2, 3, 16, 16), (3, 2, 1, 0))

        _test_permute((1, 3, 32, 32), (0, 3, 1, 2))
        _test_permute((2, 32, 16), (2, 0, 1))
        _test_permute((32, 32), (1, 0))

    def test_Permute2(self):
        """Permute"""

        def _test_permute(in_shape, dims):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y1 = torch.permute(x, dims=dims)
                    return y1

            self.trace_and_test([in_shape], Model())

        _test_permute((2, 3, 16, 16), (3, 2, 1, 0))

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
        if not self.is_cv18xx:
            _test_t((32, ))

    #######################################################################
    # MaskedFill
    # ------------
    def test_MaskedFill(self):

        def _test_masked_fill(in_shape, mask_shape, is_local):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x, mask):
                    if is_local:
                        x += x
                        x -= 2
                        x *= 2
                        x += 1
                    x = torch.masked_fill(x, mask, 5)
                    if is_local:
                        x += 1
                    return x

            self.trace_and_test(
                [in_shape, mask_shape], Model(),
                [self.Desc('float', -10, 10), self.Desc('int', 0, 2)])

        dims = [3, 4, 5]
        shape = [1, 3, 128, 300, 2]
        for dim in dims:
            shapes = [shape[:dim], shape[:dim]]
            odd = True
            for i in range(dim):
                shapes[odd][i] = 1
                odd = not odd
            _test_masked_fill(tuple(shapes[0]), tuple(shapes[1]), False)
        _test_masked_fill(([1, 3, 1, 300]), ([1, 1, 128, 300]), True)

    #######################################################################
    # Math: cos/sin/tan/tanh
    # ------------
    def test_Math(self):
        """Tanh"""

        def _test_math(func):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = func(x)
                    return y

            self.trace_and_test([(4, 3, 16, 16)], Model())

        for f in [torch.cos, torch.cosh, torch.sin, torch.sinh, torch.tan, torch.tanh, torch.exp]:
            _test_math(f)

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
    # Repeat
    # ------------
    def test_Repeat(self):
        """Repeat"""

        def _test_repeat(in_shape, repeats):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y1 = x.repeat(repeats)
                    # y1 = torch.tile(x, repeats)
                    return y1

            self.trace_and_test([in_shape], Model())

        _test_repeat((1, 3, 32, 32), (1, 3, 1, 2))
        _test_repeat((2, 32, 16), (2, 1, 1))
        _test_repeat((32, 16), (1, 2, 1))

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
                a = x.view(64, -1, 1024)  # 64, 8, 1024
                b = a.transpose(0, 1)  # 8, 64, 1024
                c = b.view(8, 64, 64, 16)  # 64, 8, 64, 16
                d = c.transpose(1, 2)  # 64, 64, 8, 16
                return d

        in_shape = (512, 1024)
        self.trace_and_test([in_shape], Model())

    #######################################################################
    # ChannelShuffle
    # ------------
    def test_ChannelShuffle(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.channel_shuffle = nn.ChannelShuffle(2)

            def forward(self, x):
                x = self.channel_shuffle(x)
                return x

        self.trace_and_test([(1, 4, 100, 100)], Model())

    #######################################################################
    # PixelShuffle
    # ------------
    def test_PixelShuffle(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.pixel_shuffle = nn.PixelShuffle(2)

            def forward(self, x):
                x = self.pixel_shuffle(x)
                return x

        self.trace_and_test([(1, 16, 32, 32)], Model())

    #######################################################################
    # PixelUnshuffle
    # ------------
    def test_PixelUnshuffle(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.pixel_unshuffle = nn.PixelUnshuffle(2)

            def forward(self, x):
                x = self.pixel_unshuffle(x)
                return x

        self.trace_and_test([(1, 4, 64, 64)], Model())

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
                    y = torch.where(x > 0, 1.0, -1.0)
                    y1 = torch.where(self.weight > 0, y, x)
                    return y1

            desc = self.Desc('float32', -1, 1)
            self.trace_and_test([in0_shape], Model(), [desc])

        _test_where((1, 3, 32, 32), (1, 3, 32, 32))
        _test_where((3, 32, 16), (3, 32, 16))
        # TODO: where backend do not support broadcast
        # _test_where((2, 32, 16), (32, 1))
        # _test_where((32, 32), (1, 32))

    #######################################################################
    # Attention
    # ------------
    def test_Attention(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)

            def forward(self, x, y, z):
                out, out_w = self.attention(x, y, z)
                return out, out_w

        self.trace_and_test([(1, 4, 64), (1, 4, 64), (1, 4, 64)], Model())

    def test_AttentionNew(self):

        def _test_attention0(shape, d, head, has_mask=True):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.q_w = torch.randn((shape[2], d * head)) / np.sqrt(d)
                    self.q_b = torch.randn((1, 1, d * head))
                    self.k_w = torch.randn((shape[2], d * head)) / np.sqrt(d)
                    self.k_b = torch.randn((1, 1, d * head))
                    self.v_w = torch.randn((shape[2], d * head)) / np.sqrt(d)
                    self.v_b = torch.randn((1, 1, d * head))
                    self.o_w = torch.randn((d * head, shape[2])) / np.sqrt(d)
                    self.o_b = torch.randn((1, 1, shape[2]))
                    self.mask = -((torch.randn((shape[0], 1, 1, shape[1])) > 0) * 10000)

                def forward(self, x):
                    q = torch.matmul(x, self.q_w) + self.q_b
                    q = q.reshape((shape[0], shape[1], head, d))
                    q = q.transpose(1, 2)
                    k = torch.matmul(x, self.k_w) + self.k_b
                    k = k.reshape((shape[0], shape[1], head, d))
                    k = k.transpose(1, 2)
                    k = k.transpose(3, 2)
                    m0 = torch.matmul(q, k)
                    m0 = m0 / np.sqrt(d)
                    if has_mask:
                        m0 = m0 + self.mask
                    m0 = torch.softmax(m0, 3)
                    v = torch.matmul(x, self.v_w) + self.v_b
                    v = v.reshape((shape[0], shape[1], head, d))
                    v = v.transpose(1, 2)
                    m1 = torch.matmul(m0, v)
                    m1 = m1.transpose(1, 2)
                    m1 = m1.reshape(shape[0], shape[1], head * d)
                    y = torch.matmul(m1, self.o_w) + self.o_b
                    y = y + 1
                    return y

            self.trace_and_test([shape], Model(), [self.Desc('float32', -1, 1)])

        def _test_attention1(shape0, shape1, d, head, has_bias=True):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.q_w = torch.randn((shape0[2], d * head)) / np.sqrt(d)
                    self.q_b = torch.randn((1, 1, d * head)) * 0.1
                    self.k_w = torch.randn((shape1[2], d * head)) / np.sqrt(d)
                    self.k_b = torch.randn((1, 1, d * head)) * 0.1
                    self.v_w = torch.randn((shape1[2], d * head)) / np.sqrt(d)
                    self.v_b = torch.randn((1, 1, d * head)) * 0.1
                    self.o_w = torch.randn((d * head, shape0[2])) / np.sqrt(d)
                    self.o_b = torch.randn((1, 1, shape0[2])) * 0.1

                def forward(self, x, x1):
                    q = torch.matmul(x, self.q_w)
                    if has_bias:
                        q = q + self.q_b
                    q = q.reshape((shape0[0], shape0[1], head, d))
                    q = q.transpose(1, 2)
                    k = torch.matmul(x1, self.k_w)
                    if has_bias:
                        k = k + self.k_b
                    k = k.reshape((shape1[0], shape1[1], head, d))
                    k = k.transpose(1, 2)
                    k = k.transpose(3, 2)
                    m0 = torch.matmul(q, k)
                    m0 = m0 / np.sqrt(d)
                    m0 = torch.softmax(m0, 3)
                    v = torch.matmul(x1, self.v_w)
                    if has_bias:
                        v = v + self.v_b
                    v = v.reshape((shape1[0], shape1[1], head, d))
                    v = v.transpose(1, 2)
                    m1 = torch.matmul(m0, v)
                    m1 = m1.transpose(1, 2)
                    m1 = m1.reshape(shape0[0], shape0[1], head * d)
                    y = torch.matmul(m1, self.o_w) + self.o_b
                    y = y + 1
                    return y

            self.trace_and_test([shape0, shape1], Model(),
                                [self.Desc('float32', -1, 1),
                                 self.Desc('float32', -1, 1)])

        # _test_attention0((1, 4096, 128), 64, 2)
        _test_attention0((1, 384, 768), 64, 12)
        # _test_attention0((2, 384, 64), 32, 2)
        # _test_attention0((2, 1024, 640), 80, 2, False)
        # _test_attention0((2, 256, 1280), 160, 2, False)
        # _test_attention0((2, 4096, 320), 40, 2, False)
        # _test_attention1((2, 4096, 320), (2, 128, 768), 40, 8)
        # _test_attention1((2, 256, 1280), (2, 77, 768), 160, 2, False)
        # _test_attention1((2, 1024, 640), (2, 77, 768), 80, 2, False)
        _test_attention1((2, 4096, 320), (2, 77, 768), 40, 2, False)
        # _test_attention1((1, 384, 64), (1, 384, 64), 32, 2, False)

    #######################################################################
    # Select
    # ------------
    def test_Select(self):
        """Select"""

        def _test_select(in0_shape, dim, index):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.const = torch.randn(in0_shape, dtype=torch.float32)
                    self.shape_checker = torch.select(self.const, dim=dim, index=index)

                def forward(self, x):
                    y = torch.select(x, dim=dim, index=index)
                    y += self.shape_checker
                    return y

            self.trace_and_test([in0_shape], Model())

        _test_select((1, 3, 32, 32), 2, 13)
        _test_select((3, 32, 16), 0, 2)
        _test_select((32, 16), 1, 4)
        _test_select((10, 20, 30, 40), 2, 5)

    #######################################################################
    # Split
    # ------------
    def test_Split(self):
        """Split"""

        def _test_split(in0_shape, dim, num):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y1 = torch.split(x, dim=dim, split_size_or_sections=num)
                    return y1

            self.trace_and_test([in0_shape], Model())

        _test_split((1, 3, 32, 32), 2, 8)
        _test_split((3, 32, 16), 0, (1, 2))
        _test_split((32, 15), 1, 4)

    #######################################################################
    # Slice
    # ------------
    def test_Slice(self):
        """Slice"""

        class Model0(nn.Module):

            def __init__(self):
                super(Model0, self).__init__()

            def forward(self, x):
                y1 = x[:, 2::2]
                return y1

        class Model1(nn.Module):

            def __init__(self):
                super(Model1, self).__init__()
                self.weight = torch.randn((16, 32, 8))

            def forward(self, x):
                w = self.weight[:, 2:20:2]
                y = x + w
                return y

        self.trace_and_test([(16, 32, 8)], Model0())
        self.trace_and_test([(16, 9, 8)], Model1())

    #######################################################################
    # Squeeze
    # ------------
    def test_Squeeze(self):
        """Squeeze"""

        def _test_squeeze(in0_shape, dim=None):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    x = x + 1
                    if dim is None:
                        y1 = torch.squeeze(x)
                    else:
                        y1 = torch.squeeze(x, dim=dim)
                    return y1

            self.trace_and_test([in0_shape], Model())

        _test_squeeze((1, 3, 1, 32), 0)
        _test_squeeze((1, 3, 1, 16))
        _test_squeeze((32, 1))

    #######################################################################
    # Stack
    # ------------
    def test_Stack(self):
        """Stack"""

        def _test_stack(in0_shape, num, dim=None):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y1 = torch.stack([x] * num, dim)
                    return y1

            self.trace_and_test([in0_shape], Model())

        _test_stack((1, 3, 16, 32), 2, 0)
        _test_stack((2, 3, 16), 3, 1)
        _test_stack((32, 16), 2, 2)

    #######################################################################
    # Upsample
    # ------------
    def test_Upsample(self):
        """Upsample"""

        def _test_upsample(in0_shape, size=None, scale=None, mode='nearest'):

            # If given scale, torch implementation will not recompute scale by default.
            # But mlir and backend will do, so this param is needed.
            args = {}
            if size is None and not scale is None:
                args["recompute_scale_factor"] = True

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.layer = nn.Upsample(size=size, scale_factor=scale, mode=mode, **args)

                def forward(self, x):
                    y1 = self.layer(x)
                    return y1

            self.trace_and_test([in0_shape], Model())

        _test_upsample((1, 3, 34, 34), None, 2)
        _test_upsample((1, 3, 33, 33), None, 2.2)
        _test_upsample((1, 1, 32, 32), None, (2.5))
        _test_upsample((1, 3, 32, 32), None, (2.0, 2.2))
        _test_upsample((1, 3, 32, 32), 64, None)
        _test_upsample((1, 3, 32, 32), (81), None)
        _test_upsample((1, 3, 32, 32), (64, 65), None)

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

        if self.is_cv18xx:
            _test_index_select((1, 3, 32, 32), 2, (1, ))
        else:
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
        _test_concat((32, 32), (32, 16), -1)

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
                a = x.view(64, -1, 1024)  # 64, 8, 1024
                b = a.transpose(0, 1)  # 8, 64, 1024
                c = self.dropout(b)
                d = c.transpose(1, 2)  # 8, 1024, 64
                return d

        in_shape = (512, 1024)
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
    # Floor
    # ------------
    def test_Floor(self):
        """Floor"""

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                o = torch.floor(x)
                return o

        self.trace_and_test([(4, 3, 32, 32)], Model())
        self.trace_and_test([(1, 65, 4, 4)], Model())

    #######################################################################
    # FloorDiv
    # ------------
    def test_FloorDiv(self):
        """FloorDiv"""

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y):
                o = torch.floor_divide(x, y)
                return o

        class Model2(nn.Module):

            def __init__(self):
                super(Model2, self).__init__()

            def forward(self, x):
                o = torch.floor_divide(x, 0.8)
                return o

        self.trace_and_test([(4, 3, 32, 32), (4, 3, 32, 32)], Model())
        self.trace_and_test([(4, 3, 32, 32)], Model2())

    #######################################################################
    # Unary
    # ------------
    def test_Unary(self):
        """Unary Functions"""

        def _test_unary(op_type, in_shape, min=0):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return op_type(x)

            self.trace_and_test([in_shape], Model(), [self.Desc('float32', min)])

        for op_type in [torch.sqrt]:
            _test_unary(op_type, (1, 3, 32, 32))
            _test_unary(op_type, (3, 16, 32))
            _test_unary(op_type, (64, 32))

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
            # _test_activation(op_type, (3, 16, 32))
            # _test_activation(op_type, (64, 32))

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
            # _test_leaky_relu((3, 16, 32), alpha)
            # _test_leaky_relu((64, 32), alpha)

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
            # _test_softmax((3, 100, 32), dim)
            # _test_softmax((3, 100, 32, 1), dim)

    #######################################################################
    # Flatten
    # ------------
    def test_Flatten(self):
        """Flatten"""

        def _test_flatten(in_shape, start_dim=0, end_dim=-1):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.flatten = nn.Flatten(start_dim=start_dim, end_dim=end_dim)

                def forward(self, x):
                    return self.flatten(x)

            self.trace_and_test([in_shape], Model())

        _test_flatten((3, 16, 32, 64))
        _test_flatten((3, 16, 32, 64), end_dim=2)
        _test_flatten((3, 16, 32, 64), start_dim=1)

    #######################################################################
    # Adaptive AvgPool2d
    # ------------
    def test_AdaptiveAvgPool2d(self):
        """AdaptiveAvgPool2d"""

        def _test_adaptive_avgpool2d(in_shape, output_size):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return F.adaptive_avg_pool2d(x, output_size)

            self.trace_and_test([in_shape], Model())

        _test_adaptive_avgpool2d((3, 64, 15, 15), (1, 1))

    #######################################################################
    # Linear
    # ------------
    def test_Linear(self):

        def _test_linear(in_shape, has_bias):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.linear = nn.Linear(32, 72, bias=has_bias)

                def forward(self, x):
                    y = self.linear(x)
                    return y

            self.trace_and_test([in_shape], Model())

        _test_linear((3, 100, 32), True)
        _test_linear((64, 32), False)

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
            # _test_log_softmax((3, 100, 32), dim)
            # _test_log_softmax((3, 100, 32, 1), dim)

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
            # _test_softmin((3, 100, 32), dim)
            # _test_softmin((3, 100, 32, 1), dim)

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
            if self.is_cv18xx and op_type == nn.ReplicationPad2d:
                _test_pad2d(op_type, (1, 16, 32, 32), 7, 0.6)
                _test_pad2d(op_type, (3, 16, 32, 32), (4, 6, 7, 8))
            else:
                _test_pad2d(op_type, (1, 16, 32), 7, 0.6)
                _test_pad2d(op_type, (3, 16, 32), (4, 6, 7, 8))
            _test_pad2d(op_type, (1, 3, 16, 32), 3, 0.5)
            _test_pad2d(op_type, (2, 4, 16, 32), (3, 4, 5, 6), 0.4)

    #######################################################################
    # Abs
    # ------------
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

    #######################################################################
    # ConstantLike
    # ------------
    def test_ConstantLike(self):
        """ZerosLike and OnesLike"""

        def _test_constant_like(op_type, input_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = op_type(x)
                    return x + y

            self.trace_and_test([input_shape], Model())

        for op_type in [torch.zeros_like, torch.ones_like]:
            _test_constant_like(op_type, (1, 16, 32, 32))

    #######################################################################
    # GridSampler
    # ------------
    def test_GridSampler(self):
        """GridSampler"""

        def _test_grid_sampler(in_shape, grid_shape, mode, padding_mode, align_corners):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    input_tensor = torch.randn(in_shape, dtype=torch.float32)
                    grid = torch.rand(grid_shape, dtype=torch.float32)

                def forward(self, input_tensor, grid):
                    output_tensor = torch.grid_sampler(input_tensor,
                                                       grid,
                                                       interpolation_mode=mode,
                                                       padding_mode=padding_mode,
                                                       align_corners=align_corners)
                    return output_tensor

            self.trace_and_test([in_shape, grid_shape], Model(),
                                [self.Desc('float32', -10, 10),
                                 self.Desc('float32', -1, 1)])

        mode_list = [0, 1]
        padding_mode_list = [0, 1]
        align_corners_list = [False, True]

        for mode in mode_list:
            for padding_mode in padding_mode_list:
                for align_corners in align_corners_list:
                    _test_grid_sampler((1, 3, 100, 150), (1, 100, 150, 2), mode, padding_mode,
                                       align_corners)
                    _test_grid_sampler((1, 3, 100, 150), (1, 40, 80, 2), mode, padding_mode,
                                       align_corners)
                    _test_grid_sampler((1, 3, 50, 50), (1, 1, 1, 2), mode, padding_mode,
                                       align_corners)

    #######################################################################
    # Deformable Convolution
    # ------------
    def test_DeformConv2D(self):
        in_channel = 3
        kernel = [3, 3]
        deform_groups = 1
        out_channel = 64

        class Model0(torch.nn.Module):

            def __init__(self):
                super(Model0, self).__init__()
                num_filter = kernel[0] * kernel[1] * 2 * deform_groups
                self.m0 = nn.Conv2d(in_channel, num_filter, kernel[0], 2, 0, 1)
                self.m1 = DeformConv2d(in_channel, out_channel, kernel[0], 2, 0, 1)

            def forward(self, x):
                y0 = self.m0(x)
                y1 = self.m1(x, y0)
                return y1

        class Model1(torch.nn.Module):

            def __init__(self):
                super(Model1, self).__init__()
                num_filter = kernel[0] * kernel[1] * 2 * deform_groups
                self.m0 = nn.Conv2d(in_channel, num_filter, kernel[0], 2, 0, 1)
                self.m1 = nn.Conv2d(in_channel, num_filter // 2, kernel[0], 2, 0, 1)
                self.m2 = DeformConv2d(in_channel, out_channel, kernel[0], 2, 0, 1)

            def forward(self, x):
                y0 = self.m0(x)
                y1 = self.m1(x)
                y2 = self.m2(x, y0, y1)
                return y2

        self.trace_and_test([(1, 3, 28, 28)], Model0())
        self.trace_and_test([(1, 3, 28, 28)], Model1())

    #######################################################################
    # Ceil
    # ------------
    def test_Ceil(self):
        """Ceil"""

        def _test_ceil(input_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return torch.ceil(x)

            self.trace_and_test([input_shape], Model())

        _test_ceil((1, 16, 32, 32))

    #######################################################################
    # Remainder
    # ------------
    def test_Remainder(self):
        """Remainder"""

        def _test_remainder(op_type, in0_shape, in1_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.weight = torch.randn(in1_shape)

                def forward(self, x):
                    y2 = op_type(x, self.weight)
                    return y2

            self.trace_and_test([in0_shape], Model(), [self.Desc('float32')])

        _test_remainder(torch.remainder, (1, 16, 32, 32), (1, 16, 32, 32))


def test_one_case_in_all(tester: TORCH_IR_TESTER, case, error_cases, success_cases):
    try:
        tester.test_single(case)
    except:
        error_cases.append(case)
        traceback.print_exc()
        return
    success_cases.append(case)


def test_all(tester: TORCH_IR_TESTER):
    if tester.multithread:
        import multiprocessing
        from utils.misc import collect_process
        process_number = multiprocessing.cpu_count() // 2 + 1
        processes = []
        error_cases = multiprocessing.Manager().list()
        success_cases = multiprocessing.Manager().list()
        for case in tester.test_cases:
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
            test_one_case_in_all(tester, case, error_cases, success_cases)
    print("Success: {}".format(success_cases))
    print("Failure: {}".format(error_cases))
    if error_cases:
        print("====== test_torch.py --chip {} TEST Failed ======".format(tester.chip))
        # exit(1)
    else:
        print("====== test_torch.py --chip {} TEST Success ======".format(tester.chip))
    return error_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--case", default="all", type=str, help="test one case, if all, then test all cases")
    parser.add_argument("--enable_thread", action="store_true", help='do test without multi thread')
    parser.add_argument("--show_all", action="store_true", help='show all cases')
    parser.add_argument("--debug", default="my_model", help="debug")
    parser.add_argument("--cmp", action='store_true', help="enable cmp")

    # yapf: enable
    args = parser.parse_args()
    tpu_mlir_jit.args = args
    tester = TORCH_IR_TESTER(args.enable_thread)
    if args.show_all:
        print("====== Show All Cases ============")
        for case in tester.test_cases:
            print(case)
        exit(0)
    if args.case == "" or args.case.lower() == "all":
        test_all(tester)
    else:
        tester.test_single(args.case)
