#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import json
import numpy as np
import os, sys
import transform.TpuLang as tpul
from typing import List
import math
from utils.timer import Timer
import cv2
from typing import List, Union
import random

def is_int(dtype, width = None):
    if width == None:
        if is_int(dtype, 8) or is_int(dtype, 16) or is_int(dtype, 32) or is_int(dtype, 64):
            return True
    if width == 64 or width == 32 or width == 16 or width == 8:
        if dtype == 'int'+str(width) or dtype == 'uint'+str(width):
            return True
    return False

def is_fp(dtype, width = None):
    if width == None:
        if is_fp(dtype, 8) or is_fp(dtype, 16) or is_fp(dtype, 32) or is_fp(dtype, 64):
            return True
    if width == 64 or width == 32 or width == 20:
        if dtype == 'float'+str(width):
            return True
    if width == 8 and dtype in ['float8e5m2', 'float8e5m2fnuz', 'float8e4m3fn', 'float8e4m3fnuz']:
        return True
    if width == 16 and dtype in ['float16', 'bfloat16']:
        return True
    return False

def rand_data(shape, dtype, min=-10, max=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if dtype in ['float32', 'float16']:
        return np.clip(np.random.randn(*shape).astype(dtype), min, max)
    if dtype == 'int32' or 'uint32' or 'int16' or 'uint16' or 'int8' or 'uint8':
        return np.random.randint(0, 127, size=shape).astype(dtype)
    raise Exception("Not supported data type: {}!".format(dtype))


def rand_indices(shape, dtype):
    num_elem = 1
    for i in range(len(shape)):
        num_elem *= shape[i]
    data = np.arange(num_elem)
    np.random.shuffle(data)
    return data.reshape(shape).astype(dtype=dtype)

def tpulang(chip):
    def wrapper(func):
        def decorate(*args, **kwargs):
            tpul.init(chip)
            ret = func(*args, **kwargs)
            tpul.deinit()
            return ret
        return decorate
    return wrapper


Failed_Cases = []

class TPULANG_IR_TESTER(object):
    ID = 0

    # This class is built for testing single operator transform.
    def __init__(self, chip: str = "bm1684x", mode: str = "all", simple: bool = False, no_save: bool = False):
        Y, N = True, False
        self.test_function = {
            #############################
            # TpuLang Test Case, Alphabetically
            #############################
            # case:  (test,                    bm1684x_support, bm1688_support)
            "Abs": (self.test_Abs,                      Y, Y),
            "Add": (self.test_Add,                      Y, Y),
            "AddShift": (self.test_AddShift,            Y, Y),
            "Arccos": (self.test_Arccos,                Y, Y),
            "Arctanh": (self.test_Arctanh,              Y, Y),
            "Arg": (self.test_Arg,                      Y, Y),
            "ArgSort": (self.test_ArgSort,              Y, Y),
            "Avgpool": (self.test_Avgpool,              Y, Y),
            "Avgpool3d":(self.test_Avgpool3d,           Y, Y),
            "Broadcast": (self.test_Broadcast,          Y, Y),
            "BatchNorm": (self.test_BatchNorm,          Y, Y),
            # "Cast": (self.test_Cast,                  Y, Y),
            "Ceil": (self.test_Ceil,                    Y, Y),
            "Clamp": (self.test_Clamp,                  Y, Y),
            "Concat": (self.test_Concat,                Y, Y),
            "Concat2": (self.test_Concat2,              Y, Y),
            "CondSelect": (self.test_CondSelect,        Y, Y),
            "Conv2d": (self.test_Conv2d,                Y, Y),
            "Conv3d": (self.test_Conv3d,                Y, Y),
            "Copy": (self.test_Copy,                    Y, Y),
            "Cos": (self.test_Cos,                      Y, Y),
            "Cosh": (self.test_Cosh,                    Y, Y),
            "Deconv2d": (self.test_Deconv2d,            Y, Y),
            # "Deconv3d": (self.test_Deconv3d,            N, N), # not support
            # "Dequantint": (self.test_dequantint,        Y, Y),
            "Div": (self.test_Div,                      Y, Y),
            "Elu": (self.test_Elu,                      Y, Y),
            "Eq": (self.test_Eq,                        Y, Y),
            "Eqs": (self.test_Eqs,                      Y, Y),
            "Erf": (self.test_Erf,                      Y, Y),
            "Exp": (self.test_Exp,                      Y, Y),
            "Extract": (self.test_Extract,              Y, Y),
            "Floor": (self.test_Floor,                  Y, Y),
            "Gelu": (self.test_Gelu,                    Y, Y),
            "Ge": (self.test_Ge,                        Y, Y),
            "Ges": (self.test_Ges,                      Y, Y),
            "GroupNorm": (self.test_Group_norm,         Y, Y),
            "Gt": (self.test_Gt,                        Y, Y),
            "Gts": (self.test_Gts,                      Y, Y),
            "Hsigmoid": (self.test_Hsigmoid,            Y, Y),
            "Hswish": (self.test_Hswish,                Y, Y),
            "IndexSelect": (self.test_IndexSelect,      Y, Y),
            "Interp": (self.test_Interp,                Y, Y),
            "Le": (self.test_Le,                        Y, Y),
            "Les": (self.test_Les,                      Y, Y),
            "LeakyRelu": (self.test_LeakyRelu,          Y, Y),
            #"Lenet": (self.test_Lenet,                  N, N),
            "Ln": (self.test_Ln,                        Y, Y),
            "Lt": (self.test_Lt,                        Y, Y),
            "Lts": (self.test_Lts,                      Y, Y),
            "Lut": (self.test_Lut,                      Y, Y),
            "LayerNorm":(self.test_LayerNorm,           Y, Y),
            "MatMul": (self.test_MatMul,                Y, Y),
            "MatMulRQ_OP": (self.test_MatMulRQ_OP,      Y, N),
            "MatMulRQ_Int_Group":(self.test_MatMulRQ_Int_Group,Y, N),
            "Max": (self.test_Max,                      Y, Y),
            "Maxpool": (self.test_Maxpool,              Y, Y),
            "Maxpool3d": (self.test_Maxpool3d,          Y, Y),
            "Min": (self.test_Min,                      Y, Y),
            "Mish": (self.test_Mish,                    Y, Y),
            "Mul": (self.test_Mul,                      Y, Y),
            "MulShift": (self.test_MulShift,            Y, Y),
            "Ne": (self.test_Ne,                        Y, Y),
            "Nes": (self.test_Nes,                      Y, Y),
            "NMS": (self.test_NMS,                      Y, Y),
            "Nonzero": (self.test_Nonzero,              Y, Y),
            "Normalize": (self.test_normalize,          Y, Y),
            "NoSave": (self.test_NoSave,                Y, Y),
            "Pad": (self.test_Pad,                      Y, Y),
            "Permute": (self.test_Permute,              Y, Y),
            "Prelu": (self.test_PRelu,                  Y, Y),
            "Reduce": (self.test_Reduce,                Y, Y),
            "Relu": (self.test_Relu,                    Y, Y),
            "Repeat": (self.test_Repeat,                Y, Y),
            "Reshape": (self.test_Reshape,              Y, Y),
            "RMSNorm": (self.test_RMSNorm,              Y, Y),
            "Roll": (self.test_Roll,                    Y, Y),
            "Round": (self.test_Round,                  Y, Y),
            "Rsqrt": (self.test_Rsqrt,                  Y, Y),
            "Scatter": (self.test_ScatterElements,      Y, Y),
            "ScatterND": (self.test_ScatterND,          Y, Y),
            "ShapeFetch": (self.test_Shape_fetch,       Y, Y),
            "Sign": (self.test_Sign,                    Y, Y),
            "Sigmoid": (self.test_Sigmoid,              Y, Y),
            "Silu": (self.test_Silu,                    Y, Y),
            "Sin": (self.test_Sin,                      Y, Y),
            "Sinh": (self.test_Sinh,                    Y, Y),
            "Select": (self.test_Select,                Y, Y),
            "Softmax": (self.test_Softmax,              Y, Y),
            "Sort": (self.test_Sort,                    Y, Y),
            # "SortByKey": (self.test_SortByKey,          Y, Y),
            "Split": (self.test_Split,                  Y, Y),
            "Sqrt": (self.test_Sqrt,                    Y, Y),
            "Square": (self.test_Square,                Y, Y),
            "Squeeze": (self.test_Squeeze,              Y, Y),
            "Sub": (self.test_Sub,                      Y, Y),
            "SubShift": (self.test_SubShift,            Y, Y),
            "Swish": (self.test_Swish,                  Y, Y),
            "Tan": (self.test_Tan,                      Y, Y),
            "Tanh": (self.test_Tanh,                    Y, Y),
            "Tile": (self.test_Tile,                    Y, Y),
            "TopK": (self.test_TopK,                    Y, Y),
            "Upsample": (self.test_Upsample,            Y, Y),
            "Unsqueeze": (self.test_Unsqueeze,          Y, Y),
            "Yuv2rgb": (self.test_Yuv2rgb,              Y, Y),
            #### model ####
            "AttenQuantBlock": (self.test_AttenQuant,   Y, Y),
            "Bert": (self.test_Bert,                    Y, Y),
            "HModel": (self.test_Model,                 Y, Y),
            "Resnet50":(self.test_Resnet50,             N, Y), # temp disable
            "ResnetBlock": (self.test_ResnetBlock,      Y, Y),
            "ResnetQuant": (self.test_ResnetQuant,      Y, Y),
            "SelfAttnBlock": (self.test_SelfAttnBlock,  Y, Y),
            "SwinT": (self.test_SwinT,                  N, N),
            "MobilenetBlock": (self.test_MobilenetBlock,Y, Y),
            "VitL": (self.test_Vit_L,                   Y, Y),
            "VitL16": (self.test_Vit_L_f16,             Y, Y),
            "VitB": (self.test_Vit_B,                   Y, Y),
            "KeepOutputOrder": (self.test_KeepOutputOrder,   Y, Y),
            "MeanStdScale": (self.test_MeanStdScale,    Y, N),
            "MeanStdScaleConv": (self.test_MeanStdScale_Conv,    Y, N),
            "Rope": (self.test_Rope,                    Y, N),
            #### error case ####
            "ErrorCase": (self.test_ErrorCase,          Y, Y),
        }
        # currently tpulang only supports fp quant mode
        self.support_quant_modes = ["f32", "f16"] # no need "bf16" for now
        self.mode = mode.lower()
        self.simple = simple
        self.chip = chip.lower()
        self.no_save = no_save
        if self.simple:
            self.support_quant_modes = ["f16"]
        if self.mode == "" or self.mode == "all":
            self.quant_modes = self.support_quant_modes
        else:
            if self.mode not in self.support_quant_modes:
                raise RuntimeError("{} not support mode: {}".format(self.chip, self.mode))
            self.quant_modes = [self.mode]

    def unique_name(self, name):
        name = "{}_{}".format(name, TPULANG_IR_TESTER.ID)
        TPULANG_IR_TESTER.ID += 1
        return name

    def test_single(self, case: str):
        np.random.seed(0)
        TPULANG_IR_TESTER.ID = 0
        print("Test: {}".format(case))
        if case in self.test_function:
            os.makedirs(case, exist_ok=True)
            os.chdir(case)
            func, _, _ = self.test_function[case]
            func(case)
            print("====== TEST {} Success ======".format(case))
        else:
            raise RuntimeError("case [{}] is not exist".format(case))

    def check_support(self, case):
        _, bm1684x_support, bm1688_support = self.test_function[case]
        if self.chip == "bm1684x" and bm1684x_support:
            return True
        if self.chip == "bm1688" and bm1688_support:
            return True
        return False

    def test_all(self):
        for case in self.test_function:
            if case not in Failed_Cases:
                self.test_single(case)
        print("====== ALL TEST Success ======".format(case))

    def list(self):
        print("====== All Support Ops ======")
        for case in self.test_function:
            if case not in Failed_Cases:
                print(case)
        print("====== Error Ops ======")
        for case in self.test_function:
            if case in Failed_Cases:
                print(case)

    def coeff_tensor(self, shape, dtype, data = None, scale=None, zero_point=None):
        if data is None:
            data = rand_data(shape, dtype)
            data = data * scale if (dtype in ['float32', 'float16'] and scale is not None) else data
        if dtype in ["int8", "uint8"]:
            return tpul.Tensor(dtype=dtype, shape=shape, data=data, ttype="coeff", scale=scale, zero_point=zero_point)
        else:
            return tpul.Tensor(dtype=dtype, shape=shape, data=data, ttype="coeff")

    def compile_and_check(self, model_name, inputs, outputs, is_quantized=False, asymmetric=False, top_mlir_inference=True, tpu_mlir_inference=True,log_level='normal'):
        for input in inputs:
            assert input.ttype == "neuron", "coeff Tensor should not be input {}".format(input.name)

        if is_quantized == False:
            for mode in self.quant_modes:
                tpul.compile_f32(model_name, inputs, outputs, cmp=True, mode=mode, no_save=self.no_save, top_mlir_inference=top_mlir_inference, tpu_mlir_inference=tpu_mlir_inference,log_level=log_level)
        else:
            tpul.compile(model_name, inputs, outputs, cmp=True, dynamic=False, asymmetric=asymmetric, no_save=self.no_save, log_level=log_level)

    def test_base_binary_quant(self, case_name, func, shape_x: List[int], shape_y: List[int], scale=None, dtype="int8"):
        @tpulang(self.chip)
        def binary_coeff():
            x_data = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            y = self.coeff_tensor(shape_y, dtype, scale=4.0)
            out = func(y, x, scale=scale, out_dtype=dtype)
            self.compile_and_check(self.unique_name(case_name), [x], [out], is_quantized=True)
        @tpulang(self.chip)
        def binary():
            x_data = rand_data(shape_x, dtype)
            y_data = rand_data(shape_y, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            y = tpul.Tensor(dtype=dtype, shape=shape_y, data=y_data)
            out = func(x, y, scale=scale, out_dtype=dtype)
            self.compile_and_check(self.unique_name(case_name), [x, y], [out], is_quantized=True)
        @tpulang(self.chip)
        def binary_scalar():
            x_data = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            out = func(tpul.Scalar(2, dtype), x, scale=scale, out_dtype=dtype)
            self.compile_and_check(self.unique_name(case_name), [x], [out], is_quantized=True)

        binary_coeff()
        binary()
        binary_scalar()

    #######################################################################
    # Add
    # ------------
    def test_Add(self, case_name):
        """Add"""
        self.test_base_binary_quant(case_name, tpul.add, [1, 3, 32, 32], [1, 32, 32], [4.0, 2.0, 6.0], "int8")
        self.test_base_binary_quant(case_name, tpul.add, [1, 3, 32, 32], [1, 32, 32], dtype="float32")
        self.test_base_binary_quant(case_name, tpul.add, [1, 3, 32, 32], [1, 32, 32], dtype="float16")

    #######################################################################
    # Mul
    # ------------
    def test_Mul(self, case_name):
        """Mul"""
        self.test_base_binary_quant(case_name, tpul.mul, [1, 96, 56, 56], [1, 96, 1, 1], [4.0, 2.0, 6.0], "int8")
        self.test_base_binary_quant(case_name, tpul.mul, [1, 3, 32, 32], [1, 32, 32], dtype="float32")
        self.test_base_binary_quant(case_name, tpul.mul, [1, 3, 32, 32], [1, 32, 32], dtype="float16")

    #######################################################################
    # Sub
    # ------------
    def test_Sub(self, case_name):
        """Sub"""
        self.test_base_binary_quant(case_name, tpul.sub, [1, 3, 32, 32], [1, 32, 32], [4.0, 2.0, 6.0], "int8")
        self.test_base_binary_quant(case_name, tpul.sub, [1, 3, 32, 32], [1, 32, 32], dtype="float32")
        self.test_base_binary_quant(case_name, tpul.sub, [1, 3, 32, 32], [1, 32, 32], dtype="float16")

    #######################################################################
    # Div
    # ------------
    def div_op(self, input_0, input_1):
        div = tpul.div(input_0, input_1)
        return div

    def test_Div(self, case_name):
        """Div"""

        @tpulang(self.chip)
        def _test_div(shape_x: List[int], shape_y: List[int], dtype="float32"):
            x_data = rand_data(shape_x, dtype)
            y_data = rand_data(shape_y, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            y = tpul.Tensor(dtype=dtype, shape=shape_y, data=y_data, ttype="coeff")
            div = self.div_op(x, y)
            self.compile_and_check(self.unique_name(case_name), [x], [div], is_quantized=dtype!="float32")

        _test_div([1, 3, 28, 28], [1, 3, 28, 28])
        _test_div([1, 3, 32, 32], [1, 3, 32, 32])
        # _test_div([1, 3, 32, 32], [1, 1, 32, 32]) prob to be checked
        _test_div([1, 3, 32, 32], [1])
        _test_div([1], [1, 3, 32, 32])
        if self.chip == "bm1688":
            _test_div([1], [1, 3, 32, 32], dtype="float16")

    #######################################################################
    # Max
    # ------------
    def test_Max(self, case_name):
        """Max"""
        self.test_base_binary_quant(case_name, tpul.max, [1, 3, 32, 32], [1, 32, 32], [4.0, 2.0, 6.0], "int8")
        self.test_base_binary_quant(case_name, tpul.max, [1, 3, 32, 32], [1, 32, 32], dtype="int16")
        self.test_base_binary_quant(case_name, tpul.max, [1, 3, 32, 32], [1, 32, 32], dtype="uint32")
        self.test_base_binary_quant(case_name, tpul.max, [1, 3, 32, 32], [1, 32, 32], dtype="float16")

    #######################################################################
    # Min
    # ------------
    def test_Min(self, case_name):
        """Min"""
        self.test_base_binary_quant(case_name, tpul.min, [1, 3, 32, 32], [1, 32, 32], [4.0, 2.0, 6.0], "int8")
        self.test_base_binary_quant(case_name, tpul.max, [1, 3, 32, 32], [1, 32, 32], dtype="uint16")
        self.test_base_binary_quant(case_name, tpul.max, [1, 3, 32, 32], [1, 32, 32], dtype="int32")
        self.test_base_binary_quant(case_name, tpul.min, [1, 3, 32, 32], [1, 32, 32], dtype="float16")

    #######################################################################
    # AddShift
    # ------------
    def test_binary_shift(self, case_name, func, shape_x: List[int], shape_y: List[int], dtype_i0="int8", dtype_i1="int8", dtype_o="int8"):
        @tpulang(self.chip)
        def _test_binary():
            x_data = rand_data(shape_x, dtype_i0)
            y_data = rand_data(shape_y, dtype_i1)
            x = tpul.Tensor(dtype=dtype_i0, shape=shape_x, data=x_data, name="input")
            y = tpul.Tensor(dtype=dtype_i1, shape=shape_y, data=y_data)
            y_const = self.coeff_tensor(shape_y, dtype_i1, scale=4.0)

            out0 = func(y_const, x, shift=-1, out_dtype=dtype_o)
            binary_coeff = tpul.get_default_graph()

            tpul.reset_default_graph()
            out1 = func(y, x, shift=-1, out_dtype=dtype_o)
            binary = tpul.get_default_graph()

            tpul.reset_default_graph()
            out2 = func(tpul.Scalar(2, dtype_i1), x, shift=-1, out_dtype=dtype_o)
            binary_scalar = tpul.get_default_graph()

            tpul.reset_default_graph()
            y_s = tpul.add_shift(y, 0, out_dtype="int32", shift=2, out_name="shift")
            out3 = func(y_s, x, shift=-1, out_dtype=dtype_o)
            binary_quant = tpul.get_default_graph()

            tpul.reset_default_graph(binary_coeff)
            out0 = tpul.sub_shift(out0, 4, -1, "int8")
            self.compile_and_check(self.unique_name(case_name), [x], [out0], is_quantized=True)
            tpul.reset_default_graph(binary)
            self.compile_and_check(self.unique_name(case_name), [x, y], [out1], is_quantized=True)
            tpul.reset_default_graph(binary_scalar)
            self.compile_and_check(self.unique_name(case_name), [x], [out2], is_quantized=True)
            tpul.reset_default_graph(binary_quant)
            self.compile_and_check(self.unique_name(case_name), [x, y], [out3], is_quantized=True)

        _test_binary()

    def test_AddShift(self, case_name):
        """Add Shift"""
        self.test_binary_shift(case_name, tpul.add_shift, [1, 3, 32, 32], [1, 32, 32], "int8", "int8", "int8")
        self.test_binary_shift(case_name, tpul.add_shift, [1, 3, 1, 32], [1, 32, 32], "uint8", "int16", "int16")
        self.test_binary_shift(case_name, tpul.add_shift, [1, 3, 32, 32], [1, 32, 32], "uint32", "int8", "int32")

    def test_SubShift(self, case_name):
        """Sub Shift"""
        self.test_binary_shift(case_name, tpul.sub_shift, [1, 3, 32, 32], [1, 32, 32], "int8", "int8", "int8")
        self.test_binary_shift(case_name, tpul.sub_shift, [1, 3, 1, 32], [1, 32, 32], "uint8", "int16", "int16")
        self.test_binary_shift(case_name, tpul.sub_shift, [1, 3, 32, 32], [1, 32, 32], "uint32", "int8", "int32")

    def test_MulShift(self, case_name):
        """Mul Shift"""
        self.test_binary_shift(case_name, tpul.mul_shift, [1, 3, 32, 32], [1, 32, 32], "int8", "int8", "int8")
        self.test_binary_shift(case_name, tpul.mul_shift, [1, 3, 1, 32], [1, 32, 32], "uint8", "int16", "int16")
        self.test_binary_shift(case_name, tpul.mul_shift, [1, 3, 32, 32], [1, 32, 32], "uint32", "int8", "int32")

    #######################################################################
    # Convolution
    # ------------

    def conv3d_op(self,
                x,
                kshape,
                stride,
                pad=None,
                group=1,
                dilation=[1, 1, 1],
                bias=False,
                dtype="float32"):
        oc = kshape[0]
        weight = self.coeff_tensor(kshape, dtype)
        bias = self.coeff_tensor(oc, 'float32') if bias else None
        deconv = tpul.conv3d(x,
                            weight,
                            bias=bias,
                            stride=stride,
                            pad=pad,
                            dilation=dilation,
                            group=group,
                            out_dtype=dtype)
        return deconv

    def test_Conv3d(self, case_name):
        """Conv 3D"""

        @tpulang(self.chip)
        def _test_convolution(input_shape: List[int],
                              kernel_shape: List[int],
                              stride: List[int] = [1, 1, 1],
                              dilation: List[int] = [1, 1, 1],
                              pad: List[int] = None,
                              group=1,
                              dtype="float32"):
            x_data = rand_data(input_shape, dtype)
            x = tpul.Tensor(dtype=dtype, shape=input_shape, data=x_data)
            conv = self.conv3d_op(x,
                                kernel_shape,
                                stride,
                                pad,
                                group=group,
                                dilation=dilation,
                                dtype=dtype)
            self.compile_and_check(self.unique_name(case_name), [x], [conv], dtype!="float32")

        _test_convolution([1, 3, 28, 28, 28], [3, 1, 1, 1, 1], group=3)
        _test_convolution([1, 3, 28, 28, 28], [3, 1, 1, 1, 1], group=3, dtype="float16")

    #######################################################################
    # Deconvolution
    # ------------
    def deconv_op(self,
                x,
                kshape,
                stride,
                pad=None,
                output_padding=None,
                group=1,
                dilation=[1, 1],
                bias=False,
                dtype="float32"):
        oc = kshape[0]
        weight = self.coeff_tensor(kshape, dtype)
        bias = self.coeff_tensor(oc, 'flioat32') if bias else None
        deconv = tpul.deconv(x,
                            weight,
                            bias=bias,
                            stride=stride,
                            pad=pad,
                            output_padding=None,
                            dilation=dilation,
                            group=group,
                            out_dtype=dtype)
        return deconv

    def deconv3d_op(self,
                x,
                kshape,
                stride,
                pad=None,
                output_padding=None,
                group=1,
                dilation=[1, 1, 1],
                bias=False,
                dtype="float32"):
        oc = kshape[0]
        weight = self.coeff_tensor(kshape, dtype)
        out_dtype = dtype if dtype == 'float32' else 'int32'
        bias = self.coeff_tensor(oc, out_dtype) if bias else None
        deconv = tpul.deconv3d(x,
                            weight,
                            bias=bias,
                            stride=stride,
                            pad=pad,
                            output_padding=output_padding,
                            dilation=dilation,
                            group=group,
                            out_dtype=out_dtype)
        return deconv

    def test_Deconv2d(self, case_name):
        """Deconv 2D"""

        @tpulang(self.chip)
        def _test_deconvolution(input_shape: List[int],
                              kernel_shape: List[int],
                              stride: List[int] = [1, 1],
                              dilation: List[int] = [1, 1],
                              pad: List[int] = None,
                              group=1,
                              dtype="float32"):
            x_data = rand_data(input_shape, dtype)
            x = tpul.Tensor(dtype=dtype, shape=input_shape, data=x_data)
            deconv = self.deconv_op(x,
                                kernel_shape,
                                stride,
                                pad,
                                group=group,
                                dilation=dilation,
                                dtype=dtype)
            self.compile_and_check(self.unique_name(case_name), [x], [deconv], dtype!="float32")

        _test_deconvolution([1, 3, 28, 28], [12, 3, 1, 1], group=3)
        _test_deconvolution([1, 3, 32, 32], [12, 3, 3, 3], stride=[2, 2], pad=[1, 1, 1, 1])
        _test_deconvolution([1, 3, 28, 28], [12, 3, 1, 1], group=3, dtype="float16")
        _test_deconvolution([1, 3, 32, 32], [12, 3, 3, 3], stride=[2, 2], pad=[1, 1, 1, 1], dtype="float16")

        @tpulang(self.chip)
        def _test_deconvolution_int(input_shape: List[int],
                                    kernel_shape: List[int],
                                    stride: List[int] = [1, 1],
                                    dilation: List[int] = [1, 1],
                                    pad: List[int] = None,
                                    zp: List[int] = [0, 0],
                                    group=1,
                                    dtype="int8"
        ):
            x_data = rand_data(input_shape, dtype)
            x = tpul.Tensor(dtype=dtype, shape=input_shape, data=x_data)
            deconv = self.deconv_int_op(x,
                                kernel_shape,
                                stride,
                                pad,
                                group=group,
                                dilation=dilation,
                                zp=zp,
                                out_dtype="int32")
            rq1 = tpul.requant_int(deconv, 2030043136, -7, 0, 0, 'int8', round_mode='half_away_from_zero')
            self.compile_and_check(self.unique_name(case_name), [x], [rq1], is_quantized=True)

        _test_deconvolution_int([1, 3, 32, 32], [12, 3, 3, 3], stride=[2, 2], pad=[1, 1, 1, 1], zp=[1, 1])

    def deconv_int_op(self,
                x,
                kshape,
                stride,
                pad=None,
                group=1,
                dilation=[1, 1],
                bias=False,
                zp=[0,0],
                out_dtype="int32"):
        oc = kshape[0]
        weight = self.coeff_tensor(kshape, x.dtype, scale=1/(math.sqrt(kshape[1] * kshape[2] * kshape[3])), zero_point=zp[1])
        bias = self.coeff_tensor(oc, out_dtype) if bias else None
        conv = tpul.deconv_int(x,
                            weight,
                            bias=bias,
                            stride=stride,
                            pad=pad,
                            dilation=dilation,
                            group=group,
                            input_zp=zp[0],
                            weight_zp=zp[1],
                            out_dtype=out_dtype)
        return conv

    def test_Deconv3d(self, case_name):
        """Deconv 3D"""

        @tpulang(self.chip)
        def _test_deconvolution(input_shape: List[int],
                              kernel_shape: List[int],
                              stride: List[int] = [1, 1, 1],
                              dilation: List[int] = [1, 1, 1],
                              pad: List[int] = None,
                              group=1,
                              dtype="float32"):
            x_data = rand_data(input_shape, dtype)
            x = tpul.Tensor(dtype=dtype, shape=input_shape, data=x_data)
            conv = self.deconv3d_op(x,
                                kernel_shape,
                                stride,
                                pad,
                                group=group,
                                dilation=dilation,
                                dtype=dtype)
            self.compile_and_check(self.unique_name(case_name), [x], [conv], dtype!="float32")

        _test_deconvolution([1, 3, 28, 28, 28], [3, 3, 1, 1, 1], group=3)
        _test_deconvolution([1, 3, 28, 28, 28], [3, 3, 1, 1, 1], group=3, dtype="float16")

    #######################################################################
    # Dequantint
    # ------------
    # def dequant_int_op(self, tensor_i, out_dtype="int8", mul=0.875, shift=0, offset=None, lshift=0, requant_mode=0):
    #     print(f"dequant_int_op: tensor_i={tensor_i}, out_dtype={out_dtype}, mul={mul}, shift={shift}, offset={offset}, lshift={lshift}, requant_mode={requant_mode}")
    #     dequant = tpul.dequant_int(tensor_i=tensor_i, mul = mul, shift = shift, offset = offset , lshift= lshift, requant_mode = requant_mode, out_dtype= out_dtype)
    #     return dequant

    # def test_dequantint(self, case_name):
    #     """Dequant_int"""

    #     @tpulang(self.chip)
    #     def _test_dequantint(shape: List[int], dtype="int8", out_dtype='int32'):
    #         x_data = rand_data(shape, dtype)
    #         print(f"Generated random data: {x_data}")
    #         x = tpul.Tensor(dtype=dtype, shape=shape, data=x_data)
    #         print(f"Created tensor x: {x}")
    #         dq1= self.dequant_int_op(x, out_dtype, requant_mode=0)

    #         y_data = rand_data(shape, dtype)
    #         print(f"Generated random data: {x_data}")
    #         y = tpul.Tensor(dtype=dtype, shape=shape, data=y_data)
    #         print(f"Created tensor x: {x}")
    #         dq2 = self.dequant_int_op(y, out_dtype, requant_mode=0)

    #         r = tpul.add(dq1, dq2)
    #         print(f"Add operation result: {r}")
    #         self.compile_and_check(self.unique_name(case_name), [x], [r])

    #     # input int8, output int32
    #     _test_dequantint([1, 3, 32, 32])

    #######################################################################
    # HModel
    # ------------
    def test_Model(self, case_name):

        def conv_quant(x, kshape, has_bias=False, stride=None, pad=None, group=1,
                       dilation=[1,1], scale=[1, 1, 1], zp=[0, 0 , 0], dtype='int8'):
            oc = kshape[0]
            weight = self.coeff_tensor(kshape, dtype, scale=[scale[1]] * oc, zero_point=zp[1])
            out_dtype = dtype
            bias = self.coeff_tensor(oc, 'int32') if has_bias else None
            conv = tpul.conv_quant(x,
                                   weight,
                                   bias=bias,
                                   stride=stride,
                                   pad=pad,
                                   dilation=dilation,
                                   group=group,
                                   input_scale=scale[0],
                                   weight_scale=[scale[1]] * oc,
                                   output_scale=scale[2],
                                   input_zp=zp[0],
                                   weight_zp=zp[1],
                                   output_zp=zp[2],
                                   out_dtype=out_dtype)
            return conv

        def matmul_quant(left, right, has_bias=False, right_transpose=False, keep_dims=True,
                         scale=[1,1,1], zp=[0,0,0], dtype='int8'):
            bias = self.coeff_tensor(right.shape[-1], 'int32') if has_bias else None
            matm = tpul.matmul_quant(left,
                                     right=right,
                                     bias=bias,
                                     right_transpose=right_transpose,
                                     keep_dims=keep_dims,
                                     input_scale=scale[0],
                                     right_scale=scale[1],
                                     output_scale=scale[2],
                                     input_zp=zp[0],
                                     right_zp=zp[1],
                                     output_zp=zp[2],
                                     out_dtype=dtype)
            return matm

        def model_conv_int(x):
            rq0 = tpul.requant_fp_to_int(x, 0.078125, 0, 0, 'int8')
            kshape = [64, 3, 7, 7]
            data = rand_data(kshape, 'int8')
            weight0 = tpul.Tensor(dtype='int8', shape=kshape, data=data, ttype="coeff")
            data = rand_data(kshape[0], 'int32')
            bias1 = tpul.Tensor(dtype='int32', shape=kshape[0], data=data, ttype="coeff")

            conv1 = tpul.conv_int(rq0, weight0, bias=bias1, stride=[2,2], pad=None, dilation=None,
                        group=1, input_zp=0, weight_zp=0, out_dtype='int32')
            # mul, shift = quantization(input_scale * weight_scale / output_scale)
            # https://tpumlir.org/docs/developer_manual/06_quantization.html
            mul = [2030043136] * 64
            shift = [-38] * 64
            rq1 = tpul.requant_int(conv1, mul, shift, 0, 2, 'int8', round_mode='half_away_from_zero', out_name= 'conv1_name')
            rq1.quantization(0.875)
            relu1 = tpul.relu(rq1)
            # reshape1 = tpul.reshape(relu1, [64, 109, 109])
            # cast1 = tpul.dequant_int_to_fp(reshape1, scale=0.0625, offset=0)
            return relu1

        def model_conv_quant(x):
            rq0 = tpul.requant_fp_to_int(x, 0.078125, 0, 0, 'int8')
            # mul, shift = affine_quantization(input_scale * weight_scale / output_scale)
            # tensorflow/lite/kernels/internal/quantization_utils.cc:QuantizeMultiplier()
            conv1 = conv_quant(rq0, [64,3,7,7], True, stride=[2,2], pad=[3,3,3,3], dilation=None,
                        group=1, scale=[0.078125, 0.078125, 0.078125], zp=[0, 0, 0], dtype='int8')
            relu1 = tpul.relu(conv1)
            dq2 = tpul.dequant_int_to_fp(relu1, 0.078125, 0)
            reshape1 = tpul.reshape(dq2, [64, 12544])
            mat_w = self.coeff_tensor(shape=[12544, 1000], dtype="float32")
            matmul1 = self.matmul_op(reshape1, mat_w)
            soft1 = tpul.softmax(matmul1, 1)
            # pool1 = tpul.maxpool2d(relu1, [3,3], stride=[2,2], pad=[1,1,1,1])
            # conv2_1 = conv_quant(pool1, [64,64,1,1], True,
            #             scale=[0.078125, 0.078125, 0.078125], zp=[0, 0, 0], dtype='int8')
            # # relu2_1 = tpul.relu(conv2_1)
            # conv2_2 = conv_quant(conv2_1, [64,64,3,3], True, pad=[1,1,1,1],
            #             scale=[0.078125, 0.078125, 0.078125], zp=[0, 0, 0], dtype='int8')
            # # relu2_2 = tpul.relu(conv2_2)
            # conv2_3 = conv_quant(conv2_2, [256,64,1,1], True,
            #             scale=[0.078125, 0.078125, 0.078125], zp=[0, 0, 0], dtype='int8')
            # conv2_0 = conv_quant(pool1, [256,64,1,1], True,
            #             scale=[0.078125, 0.078125, 0.078125], zp=[0, 0, 0], dtype='int8')
            # add2 = tpul.add(conv2_3, conv2_0, scale=[0.078125, 0.078125, 0.0625], out_dtype='int8')
            # dq29 = tpul.dequant_int_to_fp(add2, 0.0625, 0)
            # reshape3 = tpul.reshape([])
            return soft1

        def model_mat_quant(x):
            rq0 = tpul.requant_fp_to_int(x, 0.078125, 0, 0, 'int8')
            weight0 = self.coeff_tensor([128, 512], 'int8', scale=0.078125, zero_point=0)
            mat1 = matmul_quant(rq0, weight0, True, scale=[0.078125, 0.078125, 0.078125], zp=[0, 0, 0], dtype='int8')
            return mat1

        def model_mat_int(x):
            rq0 = tpul.requant_fp_to_int(x, [0.078125], 0, 0, 'int8')
            weight0 = self.coeff_tensor([128, 512], 'int8')
            mat1 = tpul.matmul_int(rq0, weight0, None, input_zp=0, right_zp=0, out_dtype='int32')
            rq2 = tpul.requant_int(mat1, 2030043136, -13, 0, 0, 'int8', round_mode='half_away_from_zero', out_name= 'conv1_name')
            return rq2

        @tpulang(self.chip)
        def _test_model_def(in_shape, model):
            x_data = rand_data(in_shape, 'float32', -10, 10)
            x = tpul.Tensor(dtype='float32', shape=in_shape, data=x_data)
            out = model(x=x)
            self.compile_and_check(self.unique_name(case_name), [x], [out], is_quantized=True)

        _test_model_def([1, 3, 224, 224], model_conv_int)
        _test_model_def([1, 3, 224, 224], model_conv_quant)
        _test_model_def([12, 384, 128], model_mat_int)
        _test_model_def([12, 384, 128], model_mat_quant)

    def test_Resnet50(self, case_name):
        def conv_block(x, kshape, stride, pad):
            conv = self.conv_op(x, kshape, stride, pad, bias=True)
            norm = self.batch_norm_op(conv, kshape[0])
            relu = tpul.relu(norm)
            return relu

        def max_pool_block(x, oc, kshape, stride, pad):
            pool = tpul.maxpool2d(x, kernel=kshape, stride=stride, pad=pad)
            norm = self.batch_norm_op(pool, oc)
            relu = tpul.relu(norm)
            return relu

        def resnet_block0(x, oc, ic, kc, stride):
            conv0_0 = conv_block(x, [kc, ic, 1, 1], [1,1], [0,0,0,0])
            conv0_1 = conv_block(conv0_0, [kc, kc, 3, 3], [stride,stride], [1,1,1,1])
            conv0_2 = self.conv_op(conv0_1, [oc, kc, 1, 1], [1,1], [0,0,0,0], bias=True)
            conv1 = self.conv_op(x, [oc, ic, 1, 1], [stride,stride], [0,0,0,0], bias=True)
            return tpul.add(conv0_2, conv1)

        def resnet_block1(x, oc, kc):
            norm = self.batch_norm_op(x, oc)
            relu = tpul.relu(norm)
            conv0 = conv_block(relu, [kc, oc, 1, 1], [1,1], [0,0,0,0])
            conv1 = conv_block(conv0, [kc, kc, 3, 3], [1,1], [1,1,1,1])
            conv2 = self.conv_op(conv1, [oc, kc, 1, 1], [1,1], [0,0,0,0], bias=True)
            return tpul.add(conv2, x)

        def resnet50(x, dtype="float32"):
            # perm = tpul.permute(x, [0, 3, 1, 2])
            norm0 = self.batch_norm_op(x, 3)
            conv1 = conv_block(norm0, [64, 3, 7, 7], [2,2], [3,3,3,3])
            pool1 = max_pool_block(conv1, 64, [3,3], [2,2], [1,1,1,1])

            res2 = resnet_block0(pool1, 256, 64, 64, 1)
            res3 = resnet_block1(res2, 256, 64)
            res4 = resnet_block1(res3, 256, 64)
            norm4 = self.batch_norm_op(res4, 256)
            relu4 = tpul.relu(norm4)

            res5 = resnet_block0(relu4, 512, 256, 128, 2)
            res6 = resnet_block1(res5, 512, 128)
            res7 = resnet_block1(res6, 512, 128)
            res8 = resnet_block1(res7, 512, 128)
            norm8 = self.batch_norm_op(res8, 512)
            relu8 = tpul.relu(norm8)

            res9 = resnet_block0(relu8, 1024, 512, 256, 2)
            res10 = resnet_block1(res9, 1024, 256)
            res11 = resnet_block1(res10, 1024, 256)
            res12 = resnet_block1(res11, 1024, 256)
            res13 = resnet_block1(res12, 1024, 256)
            res14 = resnet_block1(res13, 1024, 256)
            norm14 = self.batch_norm_op(res14, 1024)
            relu14 = tpul.relu(norm14)

            res15 = resnet_block0(relu14, 2048, 1024, 512, 2)
            res16 = resnet_block1(res15, 2048, 512)
            res17 = resnet_block1(res16, 2048, 512)
            norm17 = self.batch_norm_op(res17, 2048)
            relu17 = tpul.relu(norm17)
            apool = tpul.avgpool2d(relu17, None, [1,1], [0,0,0,0])
            reshape = tpul.reshape(apool, [0, -1])
            mat_weight = self.coeff_tensor([2048, 1000], dtype, scale=0.05)
            mat_bias = self.coeff_tensor([1000], dtype=dtype if self.chip == "bm1684x" else "float32", scale=0.03)
            mat = self.matmul_op(reshape, mat_weight, mat_bias, dtype=dtype)
            return mat

        @tpulang(self.chip)
        def _test_model_def(in_shape, dtype='float32', is_quantized=True):
            x_data = rand_data(in_shape, dtype, -10, 10)
            x = tpul.Tensor(dtype=dtype, shape=in_shape, data=x_data)
            out = resnet50(x, dtype)
            if dtype == 'float32':
                x.preprocess(mean=[0, 0.3, 0])
            self.compile_and_check(self.unique_name(case_name), [x], [out], is_quantized=is_quantized)

        _test_model_def([1, 3, 224, 224], 'float16', is_quantized=True)

    def conv_int_op(self,
                x,
                kshape,
                stride,
                pad=None,
                group=1,
                dilation=[1, 1],
                bias=False,
                zp=[0,0],
                out_dtype="int32"):
        oc = kshape[0]
        weight = self.coeff_tensor(kshape, x.dtype, scale=1/(math.sqrt(kshape[1] * kshape[2] * kshape[3])), zero_point=0)
        bias = self.coeff_tensor(oc, out_dtype) if bias else None
        conv = tpul.conv_int(x,
                            weight,
                            bias=bias,
                            stride=stride,
                            pad=pad,
                            dilation=dilation,
                            group=group,
                            input_zp=zp[0],
                            weight_zp=zp[1],
                            out_dtype=out_dtype)
        return conv

    def test_ResnetQuant(self, case_name):

        def resnet_quant(x):
            rq0 = tpul.requant_fp_to_int(x, 1.0, 0, 0, 'int8')
            # conv1 = conv_block(rq0, [64, 3, 7, 7], [2, 2], [3,3,3,3], [2030043136, -13, 0])
            conv1 = self.conv_int_op(rq0, [64, 3, 7, 7], [2, 2], [3,3,3,3], zp=[0, 0], out_dtype='int32')
            rq1 = tpul.requant_int(conv1, 2030043136, -7, 0, 0, 'int8', round_mode='half_away_from_zero')
            # relu1 = tpul.relu(rq1)
            conv2 = self.conv_int_op(rq1, [96,64,3,3], [2,2], [1,1,1,1], zp=[0,0], out_dtype='int32')
            rq2 = tpul.requant_int(conv2, 1748893696, -10, 0, 0, 'int8', round_mode='half_away_from_zero')
            coeff3 = self.coeff_tensor([1,96,1,1], 'int8', scale=10.0)
            mul3 = tpul.mul(rq2, coeff3, scale=[0.25, 10.0, 2.5], out_dtype='int8')
            coeff4 = self.coeff_tensor([1,96,1,1], 'int8', scale=2.0)
            add4 = tpul.add(mul3, coeff4, scale=[2.5, 2.0, 4.0], out_dtype='int8')
            conv5 = self.conv_int_op(add4, [96,96,3,3], [1,1], [1,1,1,1], zp=[0,0], out_dtype='int32')
            rq5 = tpul.requant_int(conv5, 1623457792, -8, 0, 0, 'int8', round_mode='half_away_from_zero')
            conv6 = self.conv_int_op(rq5, [96,96,3,3], [1,1], [1,1,1,1], zp=[0,0], out_dtype='int32')
            rq6 = tpul.requant_int(conv6, 1623457792, -10, 0, 0, 'int8', round_mode='half_away_from_zero')
            add7 = tpul.add(rq2, rq6, [0.25, 0.0625, 0.4], out_dtype='int8')
            coeff7 = self.coeff_tensor([1,96,1,1], 'int8', scale=2.0)
            mul7 = tpul.mul(add7, coeff7, scale=[0.4, 2.0, 4.0], out_dtype='int8')
            coeff8 = self.coeff_tensor([1,96,1,1], 'int8', scale=-2)
            add8 = tpul.add(mul7, coeff8, scale=[4.0, 2.0, 8.0], out_dtype='int8')
            conv9 = self.conv_int_op(add8, [96,96,3,3], [1,1], [1,1,1,1], zp=[0,0], out_dtype='int32')
            rq9 = tpul.requant_int(conv9, 1712717824, -7, 0, 0, 'int8', round_mode='half_away_from_zero')
            dq10 = tpul.dequant_int_to_fp(rq9, 0.0625, 0)
            return add4, dq10

        def resnet_mix(x, fdtype="float32"):
            rq0 = tpul.requant_fp_to_int(x, 1.0, 0, 0, 'int8')
            # conv1 = conv_block(rq0, [64, 3, 7, 7], [2, 2], [3,3,3,3], [2030043136, -13, 0])
            conv1 = self.conv_int_op(rq0, [64, 3, 7, 7], [2, 2], [3,3,3,3], zp=[0, 0], out_dtype='int32')
            rq1 = tpul.requant_int(conv1, 2030043136, -7, 0, 0, 'int8', round_mode='half_away_from_zero')
            relu1 = tpul.relu(rq1)
            conv2 = self.conv_int_op(relu1, [96,64,3,3], [2,2], [1,1,1,1], zp=[0,0], out_dtype='int32')
            rq2 = tpul.requant_int(conv2, 1748893696, -10, 0, 0, 'int8', round_mode='half_away_from_zero')
            relu2 = tpul.relu(rq2)
            dq3 = tpul.dequant_int_to_fp(relu2, 0.25, 0, out_dtype=fdtype)
            coeff3 = self.coeff_tensor([1,96,1,1], fdtype, scale=10.0)
            mul3 = tpul.mul(dq3, coeff3)
            coeff4 = self.coeff_tensor([1,96,1,1], fdtype, scale=-2.0)
            add4 = tpul.add(mul3, coeff4)
            # add4 = self.batch_norm_op(dq3, 96)
            rq4 = tpul.requant_fp_to_int(add4, 4.0, 0, 0, "int8")
            conv5 = self.conv_int_op(rq4, [96,96,3,3], [1,1], [1,1,1,1], zp=[0,0], out_dtype='int32')
            rq5 = tpul.requant_int(conv5, 1623457792, -8, 0, 0, 'int8', round_mode='half_away_from_zero')
            relu5 = tpul.relu(rq5)
            conv6 = self.conv_int_op(relu5, [96,96,3,3], [1,1], [1,1,1,1], zp=[0,0], out_dtype='int32')
            rq6 = tpul.requant_int(conv6, 1623457792, -10, 0, 0, 'int8', round_mode='half_away_from_zero')
            relu6 = tpul.relu(rq6)
            dq7 = tpul.dequant_int_to_fp(relu6, 0.0625, 0, out_dtype=fdtype)
            add7 = tpul.add(dq3, dq7)
            coeff7 = self.coeff_tensor([1,96,1,1], fdtype, scale=2.0)
            mul7 = tpul.mul(add7, coeff7)
            coeff8 = self.coeff_tensor([1,96,1,1], fdtype, scale=-2)
            add8 = tpul.add(mul7, coeff8)
            rq8 = tpul.requant_fp_to_int(add8, 8.0, 0, 0, "int8")
            conv9 = self.conv_int_op(rq8, [96,96,3,3], [1,1], [1,1,1,1], zp=[0,0], out_dtype='int32')
            rq9 = tpul.requant_int(conv9, 1712717824, -7, 0, 0, 'int8', round_mode='half_away_from_zero')
            dq10 = tpul.dequant_int_to_fp(rq9, 0.0625, 0)
            return add4, dq10

        def resnet_mix_f16(x):
            return resnet_mix(x, fdtype="float16")

        def resnet_quant_fp(x):
            rq0 = tpul.requant_fp_to_int(x, 1.0, 0, 0, 'int8')
            # conv1 = conv_block(rq0, [64, 3, 7, 7], [2, 2], [3,3,3,3], [2030043136, -13, 0])
            conv1 = self.conv_int_op(rq0, [64, 3, 7, 7], [2, 2], [3,3,3,3], zp=[0, 0], out_dtype='int32')
            rq1 = tpul.requant_fp(conv1, 0.078125, 0.0, 'int8', round_mode='half_away_from_zero')
            # relu1 = tpul.relu(rq1)
            conv2 = self.conv_int_op(rq1, [96,64,3,3], [2,2], [1,1,1,1], zp=[0,0], out_dtype='int32')
            rq2 = tpul.requant_fp(conv2, 0.078125, 0.0, 'int8', round_mode='half_away_from_zero')
            return rq1, rq2

        @tpulang(self.chip)
        def _test_model_def(in_shape, model):
            x_data = rand_data(in_shape, 'float32')
            x = tpul.Tensor(dtype='float32', shape=in_shape, data=x_data)
            out0, out1 = model(x)
            self.compile_and_check(self.unique_name(case_name), [x], [out0, out1], is_quantized=True)

        _test_model_def([1, 3, 224, 224], resnet_quant)
        _test_model_def([1, 3, 224, 224], resnet_mix)
        _test_model_def([1, 3, 224, 224], resnet_mix_f16)
        _test_model_def([1, 3, 224, 224], resnet_quant_fp)

    def batch_norm_op(self, x, oc):
        mean = self.coeff_tensor([oc], x.dtype, scale=0.2)
        var = self.coeff_tensor([oc], x.dtype, data=np.clip(np.random.randn(oc), 0.5, 10).astype(x.dtype))
        gamma = self.coeff_tensor([oc], x.dtype, data=np.ones((oc)).astype(x.dtype))
        beta = self.coeff_tensor([oc], x.dtype, data=np.zeros((oc)).astype(x.dtype))
        return tpul.batch_norm(x, mean, var, epsilon=1e-5)

    def layer_norm_op(self, x, oc, axis):
        mean = self.coeff_tensor([oc], x.dtype, scale=0.2)
        var = self.coeff_tensor([oc], x.dtype, data=np.clip(np.random.randn(oc), 0.5, 10).astype(x.dtype))
        return tpul.layer_norm(x, mean, var, epsilon=1e-5, axis=axis)

    #######################################################################
    # bert
    # ------------
    def test_Bert(self, case_name):
        def matmul_weight(x, shape, dtype):
            weight = self.coeff_tensor(shape, dtype=dtype, scale=1/np.sqrt(shape[0]))
            bias = self.coeff_tensor(shape = [1, 1, shape[-1]], dtype="float32", scale=0.2)
            return tpul.matmul(x, weight, bias, out_dtype=dtype)
        def attention_block(x0, x1, x2, shape, d, head, musk, dtype="float32"):
            B = shape[0]
            S_q = shape[1]
            H_q = shape[2]
            S_k = shape[1]
            H_k = shape[2]
            S_v = shape[1]
            H_v = shape[2]
            q = matmul_weight(x0, [H_q, d*head], dtype)
            q = tpul.reshape(q, [B, S_q, head, d])
            q = tpul.permute(q, [0, 2, 1, 3])
            k = matmul_weight(x1, [H_k, d*head], dtype)
            k = tpul.reshape(k, [B, S_k, head, d])
            k = tpul.permute(k, [0, 2, 1, 3])
            k = tpul.permute(k, [0, 1, 3, 2])
            v = matmul_weight(x2, [H_v, d*head], dtype)
            v = tpul.reshape(v, [B, S_v, head, d])
            v = tpul.permute(v, [0, 2, 1, 3])
            m0 = tpul.matmul(q, k, out_dtype=dtype)
            if dtype == "float32":
                m0 = tpul.div(m0, np.sqrt(d))
            else:
                m0 = tpul.mul(m0, 1/np.sqrt(d))
            m0 = tpul.add(m0, musk) if musk is not None else m0
            m0 = tpul.softmax(m0, 3)
            m1 = tpul.matmul(m0, v, out_dtype=dtype)
            m1 = tpul.permute(m1, [0, 2, 1, 3])
            m1 = tpul.reshape(m1, [B, S_q, -1])
            out = matmul_weight(m1, [d*head, H_q], dtype=dtype)
            return out

        def mlp_block(x, shape, dtype):
            H = shape[2]
            norm = self.layer_norm_op(x, shape[2], 2)
            mat0 = matmul_weight(norm, [H, 4*H], dtype=dtype)
            gelu = tpul.gelu(mat0)
            mat1 = matmul_weight(gelu, [4*H, H], dtype=dtype)
            out = tpul.add(norm, mat1)
            return out

        def transformer_block(x, shape, d, head, musk, dtype="float32"):
            norm = self.layer_norm_op(x, shape[2], 2)
            self_atten = attention_block(norm, norm, norm, shape, d, head, musk, dtype=dtype)
            add = tpul.add(norm, self_atten)
            mlp = mlp_block(add, shape, dtype=dtype)
            return mlp

        def bert(x, mask, shape, d, head, num, dtype="float32"):
            reshape0 = tpul.reshape(mask, [1, 1, 1, -1])
            sub = tpul.sub(reshape0, 1)
            mul = tpul.mul(sub, -100000000)
            transformer = x
            for i in range(num):
                trans = transformer_block(transformer, shape, d, head, mul, dtype=dtype)
                transformer = trans
            norm2 = self.layer_norm_op(transformer, shape[2], 2)
            mat2 = matmul_weight(norm2, [shape[2], 2], dtype=dtype)
            slice = tpul.split(mat2, 2, 2)
            out0 = tpul.reshape(slice[0], [1, -1])
            out1 = tpul.reshape(slice[1], [1, -1])
            return out0, out1

        @tpulang(self.chip)
        def _test_model_def(in_shape, d, head, num, dtype='float32', is_quantized=True):
            x_data = rand_data(in_shape, dtype, -10, 10)
            x = tpul.Tensor(dtype=dtype, shape=in_shape, data=x_data)
            musk_num = np.random.randint(2, in_shape[1]+ 1)
            musk_data = np.hstack((np.array([1] * musk_num), np.array([0] * (in_shape[1] - musk_num)))).astype(dtype)
            musk = tpul.Tensor(dtype=dtype, shape=[in_shape[0], in_shape[1]], data=musk_data)
            out0, out1 = bert(x, musk, in_shape, d, head, num, dtype=dtype)
            case_unique_name = self.unique_name(case_name)
            self.compile_and_check(case_unique_name, [x, musk], [out0, out1], is_quantized=is_quantized)
            return case_unique_name

        names = []
        names.append(_test_model_def([1, 384, 1024], 64, 16, 1, 'float16', is_quantized=True))
        names.append(_test_model_def([1, 224, 768], 64, 12, 1, 'float16', is_quantized=True))
        names = [n+"_int8" for n in names]
        self.test_model_combine(names, case_name)

    #######################################################################
    # attention quant block
    # ------------
    def test_AttenQuant(self, case_name):
        def matmul_weight(x, shape, multi, shift, dtype='int8'):
            weight = self.coeff_tensor(shape, dtype)
            b_data = np.random.randint(-32768, 32767, size=[1, 1, shape[-1]]).astype('int32')
            bias = self.coeff_tensor(shape = [1, 1, shape[-1]], dtype="int32", data=b_data)
            mat = tpul.matmul_int(x, weight, bias, input_zp=0, right_zp=0, out_dtype='int32')
            return tpul.requant_int(mat, multi, shift, 0, 2, dtype, round_mode='half_away_from_zero')

        def attention_block(x0, x1, x2, shape, d, head, musk=None, dtype="int8"):
            B = shape[0]
            S_q = shape[1]
            H_q = shape[2]
            S_k = shape[1]
            H_k = shape[2]
            S_v = shape[1]
            H_v = shape[2]
            q = matmul_weight(x0, [H_q, d * head], 8158145, -30, dtype)
            q = tpul.reshape(q, [B, S_q, head, d])
            q = tpul.permute(q, [0, 2, 1, 3])
            k = matmul_weight(x1, [H_k, d * head], 8158145, -30, dtype)
            k = tpul.reshape(k, [B, S_k, head, d])
            k = tpul.permute(k, [0, 2, 1, 3])
            k = tpul.permute(k, [0, 1, 3, 2])
            v = matmul_weight(x2, [H_v, d * head], 8158145, -30, dtype)
            v = tpul.reshape(v, [B, S_v, head, d])
            v = tpul.permute(v, [0, 2, 1, 3])
            m0 = tpul.matmul_int(q, k, input_zp=0, right_zp=0, out_dtype='int32')
            m0 = tpul.requant_int(m0, 8158145, -30, 0, 2, dtype, round_mode='half_away_from_zero')
            dq0 = tpul.dequant_int_to_fp(m0, 0.875, 0)
            div0 = tpul.div(dq0, np.sqrt(d))
            m0 = tpul.add(div0, musk) if musk is not None else div0
            m0 = tpul.softmax(m0, 3)
            rq0 = tpul.requant_fp_to_int(m0, 0.0078125, 0, 2, dtype)
            m1 = tpul.matmul_int(rq0, v, input_zp=0, right_zp=0, out_dtype='int32')
            m1 = tpul.requant_int(m1, 8158145, -30, 0, 2, dtype, round_mode='half_away_from_zero')
            m1 = tpul.permute(m1, [0, 2, 1, 3])
            m1 = tpul.reshape(m1, [B, S_q, -1])
            out = matmul_weight(m1, [d*head, H_q], 8158145, -30, dtype=dtype)
            return out

        def mlp_block(x, shape, dtype):
            H = shape[2]
            mat0 = matmul_weight(x, [H, 4*H], 8158145, -30, dtype=dtype)
            ge = tpul.gelu(mat0, scale=[0.845, 0.845], zero_point=[0,0])
            mat1 = matmul_weight(ge, [4*H, H], 8158145, -30, dtype=dtype)
            return mat1

        def transformer_block(x, shape, d, head, dtype="int8"):
            norm = layer_norm(x, shape[2], -1, 1e-6, dtype="float32")
            rq0 = tpul.requant_fp_to_int(norm, 0.875, 0, 2, dtype)
            atten = attention_block(rq0, rq0, rq0, shape, d, head, dtype=dtype)
            dq0 = tpul.dequant_int_to_fp(atten, 0.875, 0)
            add0 = tpul.add(norm, dq0)
            norm1 = layer_norm(add0, shape[2], -1, 1e-6, dtype="float32")
            rq1 = tpul.requant_fp_to_int(norm1, 0.875, 0, 2, dtype)
            mlp = mlp_block(rq1, shape, dtype=dtype)
            dq1 = tpul.dequant_int_to_fp(mlp, 0.875, 0)
            add1 = tpul.add(add0, dq1)
            return add1

        def layer_norm(x, oc, axis, eps, dtype):
            gamma = self.coeff_tensor([oc], dtype=dtype, scale=1.0)
            beta = self.coeff_tensor(shape=[oc], dtype=dtype, scale=0.05)
            norm = tpul.layer_norm(x, gamma=gamma, beta=beta, epsilon=eps, axis=axis)
            return norm

        def vit(x, shape, d, head, num, dtype="float32", atten_dtype='int8'):
            H = d * head
            conv0 = self.conv_op(x, [H, 3, 16, 16], [16, 16], bias=True, dtype=dtype)
            reshape0 = tpul.reshape(conv0, [shape[0], shape[2], shape[1]-1])
            permut1 = tpul.permute(reshape0, [0, 2, 1])
            weight1 = self.coeff_tensor([shape[0], 1, H], dtype=dtype)
            concat1 = tpul.concat([weight1, permut1], axis=1)
            weight2 = self.coeff_tensor([1, shape[1], H], dtype=dtype)
            add2 = tpul.add(concat1, weight2)
            transformer = add2
            for i in range(num):
                trans = transformer_block(transformer, shape, d, head, dtype=atten_dtype)
                transformer = trans
            norm3 = layer_norm(transformer, shape[2], -1, 1e-6, dtype=dtype)
            slice3 = tpul.slice(norm3, [0, 0, 0], [np.iinfo(np.int64).max, 1, np.iinfo(np.int64).max])
            reshape3 = tpul.squeeze(slice3, [1])
            rq3 = tpul.requant_fp_to_int(reshape3, 0.875, 0, 2, atten_dtype)
            mat3 = matmul_weight(rq3, [shape[2], 1000], 8158145, -30, dtype=atten_dtype)
            dq3 = tpul.dequant_int_to_fp(mat3, 0.875, 0)
            return dq3

        @tpulang(self.chip)
        def _test_model_def(in_shape, d, head, num, dtype='float32', atten_dtype='int8', is_quantized=True):
            x_data = rand_data(in_shape, dtype, -2, 2)
            x = tpul.Tensor(dtype=dtype, shape=in_shape, data=x_data)
            shape = [in_shape[0], int(in_shape[2] * in_shape[3] / 16 / 16 + 1), head*d]
            out = vit(x, shape, d, head, num, dtype=dtype, atten_dtype=atten_dtype)
            self.compile_and_check(self.unique_name(case_name), [x], [out], is_quantized=is_quantized)

        def matmul_weight2(x, shape, multi, shift, dtype='int8'):
            weight = self.coeff_tensor(shape, dtype)
            b_data = np.random.randint(-32768, 32767, size=[1, 1, 1, shape[-1]]).astype('int32')
            bias = self.coeff_tensor(shape = [1, 1, 1, shape[-1]], dtype="int32", data=b_data)
            mat = tpul.matmul_int(x, weight, bias, input_zp=0, right_zp=0, out_dtype='int32')
            return tpul.requant_int(mat, multi, shift, 0, 2, dtype, round_mode='half_away_from_zero', rq_axis=-1, fuse_rq_to_matmul=True)

        def attention_block2(x0, x1, x2, shape, d, head, musk=None, dtype="int8"):
            B = shape[0]
            S_q = shape[2]
            H_q = shape[3]
            S_k = shape[2]
            H_k = shape[3]
            S_v = shape[2]
            H_v = shape[3]
            q = matmul_weight2(x0, [H_q, d * head], [8158145]*d*head, -31, dtype)
            q = tpul.reshape(q, [B, S_q, head, d])
            q = tpul.permute(q, [0, 2, 1, 3])
            k = matmul_weight2(x1, [H_k, d * head], [8158145]*d*head, -31, dtype)
            k = tpul.reshape(k, [B, S_k, head, d])
            k = tpul.permute(k, [0, 2, 1, 3])
            k = tpul.permute(k, [0, 1, 3, 2])
            v = matmul_weight2(x2, [H_v, d * head], [8158145]*d*head, -31, dtype)
            v = tpul.reshape(v, [B, S_v, head, d])
            v = tpul.permute(v, [0, 2, 1, 3])
            m0 = tpul.matmul_int(q, k, input_zp=0, right_zp=0, out_dtype='int32')
            m0 = tpul.requant_int(m0, 8158145, -32, 0, 2, dtype, round_mode='half_away_from_zero')
            dq0 = tpul.dequant_int_to_fp(m0, 0.875, 0)
            m0 = tpul.softmax(dq0, 3)
            rq0 = tpul.requant_fp_to_int(m0, 0.000178125, 0, 2, dtype)
            m1 = tpul.matmul_int(rq0, v, input_zp=0, right_zp=0, out_dtype='int32')
            m1 = tpul.requant_int(m1, 8158145, -32, 0, 2, dtype, round_mode='half_away_from_zero')
            m1 = tpul.permute(m1, [0, 2, 1, 3])
            m1 = tpul.reshape(m1, shape)
            out = matmul_weight2(m1, [d*head, H_q], [8158145]*d*head, -31, dtype=dtype)
            return out

        def transformer_block2(x, shape, d, head, dtype="int8"):
            cast = tpul.dequant_int_to_fp(x, 0.832, 0)
            norm = layer_norm(cast, shape[3], -1, 1e-6, dtype="float32")
            rq0 = tpul.requant_fp_to_int(norm, 0.875, 0, 2, dtype)
            atten = attention_block2(rq0, rq0, rq0, shape, d, head, dtype=dtype)
            dq0 = tpul.dequant_int_to_fp(atten, 0.875, 0)
            # add0 = tpul.add(norm, dq0)
            # norm1 = layer_norm(add0, shape[3], -1, 1e-6, dtype="float32")
            # rq1 = tpul.requant_fp_to_int(norm1, 0.875, 0, 2, dtype)
            # mlp = mlp_block(rq1, shape, dtype=dtype)
            # dq1 = tpul.dequant_int_to_fp(mlp, 0.875, 0)
            # add1 = tpul.add(add0, dq1)
            return dq0

        def vit2(x, shape, d, head, num, dtype='int8'):
            H = d * head
            # conv0 = self.conv_op(x, [H, 3, 14, 14], [14, 14], bias=True, dtype=dtype)
            conv0 = self.conv_int_op(x, [H, 3, 14, 14], [14, 14], [0,0,0,0], zp=[0, 0], out_dtype='int32')
            rq0 = tpul.requant_int(conv0, [5324]*H, [-27]*H, 0, 2, 'int8', round_mode='half_away_from_zero')
            reshape0 = tpul.reshape(rq0, [shape[0], shape[3], 1, shape[2]])
            permut1 = tpul.permute(reshape0, [0, 2, 3, 1])
            add1 = tpul.add_shift(permut1, 0, 3, out_dtype="int16")
            weight1 = self.coeff_tensor(shape, dtype=dtype)
            add2 = tpul.add_shift(add1, weight1, -2, out_dtype="int8")
            transformer = add2
            for i in range(num):
                trans = transformer_block2(transformer, shape, d, head, dtype=dtype)
                transformer = trans
            # norm3 = layer_norm(transformer, shape[2], -1, 1e-6, dtype=dtype)
            # slice3 = tpul.slice(norm3, [0, 0, 0], [np.iinfo(np.int64).max, 1, np.iinfo(np.int64).max])
            # reshape3 = tpul.squeeze(slice3, [1])
            # rq3 = tpul.requant_fp_to_int(reshape3, 0.875, 0, 2, dtype)
            # mat3 = matmul_weight(rq3, [shape[2], 1000], 8158145, -30, dtype=dtype)
            # dq3 = tpul.dequant_int_to_fp(mat3, 0.875, 0)
            return transformer

        @tpulang(self.chip)
        def _test_insert_shape(in_shape, d, head, num, dtype='int8'):
            x_data = rand_data(in_shape, dtype)
            x = tpul.Tensor(dtype=dtype, shape=in_shape, data=x_data)
            shape = [in_shape[0], 1, int(in_shape[2] * in_shape[3] / 14 / 14), head*d]
            out = vit2(x, shape, d, head, num, dtype=dtype)
            self.compile_and_check(self.unique_name(case_name), [x], [out], is_quantized=True)

        _test_model_def([1, 3, 224, 224], 64, 12, 2, 'float32', "int8", is_quantized=True)
        # _test_insert_shape([1, 3, 224, 224], 64, 8, 1)

    #######################################################################
    # vit
    # ------------
    def test_Vit(self, case_name, in_shape, d, head, num, dtype='float32', is_quantized=True):
        def matmul_weight(x, shape, dtype):
            weight = self.coeff_tensor(shape, dtype=dtype, scale=1/np.sqrt(shape[0]))
            bias = self.coeff_tensor(shape = [1, 1, shape[-1]], dtype="float32", scale=0.2)
            return tpul.matmul(x, weight, bias, out_dtype=dtype)
        def attention_block(x0, x1, x2, shape, d, head, musk=None, dtype="float32"):
            B = shape[0]
            S_q = shape[1]
            H_q = shape[2]
            S_k = shape[1]
            H_k = shape[2]
            S_v = shape[1]
            H_v = shape[2]
            q = matmul_weight(x0, [H_q, d*head], dtype)
            q = tpul.reshape(q, [B, S_q, head, d])
            q = tpul.permute(q, [0, 2, 1, 3])
            k = matmul_weight(x1, [H_k, d*head], dtype)
            k = tpul.reshape(k, [B, S_k, head, d])
            k = tpul.permute(k, [0, 2, 1, 3])
            k = tpul.permute(k, [0, 1, 3, 2])
            v = matmul_weight(x2, [H_v, d*head], dtype)
            v = tpul.reshape(v, [B, S_v, head, d])
            v = tpul.permute(v, [0, 2, 1, 3])
            m0 = tpul.matmul(q, k, out_dtype=dtype)
            if dtype == "float32":
                m0 = tpul.div(m0, np.sqrt(d))
            else:
                m0 = tpul.mul(m0, 1/np.sqrt(d))
            m0 = tpul.add(m0, musk) if musk is not None else m0
            m0 = tpul.softmax(m0, 3)
            m1 = tpul.matmul(m0, v, out_dtype=dtype)
            m1 = tpul.permute(m1, [0, 2, 1, 3])
            m1 = tpul.reshape(m1, [B, S_q, -1])
            out = matmul_weight(m1, [d*head, H_q], dtype=dtype)
            return out

        def mlp_block(x, shape, dtype):
            H = shape[2]
            norm = layer_norm(x, shape[2], -1, 1e-6, dtype=dtype)
            mat0 = matmul_weight(norm, [H, 4*H], dtype=dtype)
            ge = gelu(mat0)
            mat1 = matmul_weight(ge, [4*H, H], dtype=dtype)
            out = tpul.add(norm, mat1)
            return out

        def transformer_block(x, shape, d, head, dtype="float32"):
            norm = layer_norm(x, shape[2], -1, 1e-6, dtype=dtype)
            self_atten = attention_block(norm, norm, norm, shape, d, head, dtype=dtype)
            add = tpul.add(norm, self_atten)
            mlp = mlp_block(add, shape, dtype=dtype)
            return mlp

        def gelu(x):
            # return tpul.gelu(x)
            div = tpul.mul(x, 1/np.sqrt(2))
            erf = tpul.erf(div)
            add = tpul.add(erf, 1.0)
            mul = tpul.mul(x, add)
            mulc = tpul.mul(mul, 0.5)
            return mulc

        def layer_norm(x, oc, axis, eps, dtype):
            # gamma = self.coeff_tensor([oc], dtype=dtype, scale=1.0)
            # beta = self.coeff_tensor(shape=[oc], dtype=dtype, scale=0.05)
            # return tpul.layer_norm(x, gamma=gamma, beta=beta, epsilon=eps, axis=axis)
            mean = tpul.reduce(x, 'ReduceMean', axes=axis)
            sub = tpul.sub(x, mean)
            pow = tpul.square(sub)
            pow_mean = tpul.reduce(pow, 'ReduceMean', axes=axis)
            add = tpul.add(pow_mean, eps)
            if dtype != "float32":
                add = tpul.cast(add, "float32")
            rsqrt = tpul.rsqrt(add)
            if dtype != "float32":
                rsqrt = tpul.cast(rsqrt, dtype)
            mul = tpul.mul(sub, rsqrt)
            gamma = self.coeff_tensor([oc], dtype=dtype, scale=1.0)
            beta = self.coeff_tensor(shape=[oc], dtype=dtype, scale=0.05)
            gam = tpul.mul(mul, gamma)
            bet = tpul.add(gam, beta)
            return bet

        def vit(x, shape, d, head, num, dtype="float32"):
            H = d * head
            conv0 = self.conv_op(x, [H, 3, 16, 16], [16, 16], bias=True, dtype=dtype)
            # fetch_shape0 = tpul.shape_fetch(conv0)
            # slice0 = tpul.slice(fetch_shape0, 0, 2, axes=0)
            # neg_one = tpul.Tensor([1], ttype="coeff", data=np.array([-1]).astype("int32"), dtype="int32")
            # concat0 = tpul.concat([slice0, neg_one], 0)
            reshape0 = tpul.reshape(conv0, [shape[0], shape[2], shape[1]-1])
            permut1 = tpul.permute(reshape0, [0, 2, 1])
            weight1 = self.coeff_tensor([shape[0], 1, H], dtype=dtype)
            concat1 = tpul.concat([weight1, permut1], axis=1)
            weight2 = self.coeff_tensor([1, shape[1], H], dtype=dtype)
            add2 = tpul.add(concat1, weight2)
            transformer = add2
            for i in range(num):
                trans = transformer_block(transformer, shape, d, head, dtype=dtype)
                transformer = trans
            norm3 = layer_norm(transformer, shape[2], -1, 1e-6, dtype=dtype)
            slice3 = tpul.slice(norm3, [0, 0, 0], [np.iinfo(np.int64).max, 1, np.iinfo(np.int64).max])
            reshape3 = tpul.squeeze(slice3, [1])
            mat3 = matmul_weight(reshape3, [shape[2], 1000], dtype=dtype)
            return mat3

        @tpulang(self.chip)
        def _test_model_def(in_shape, d, head, num, dtype='float32', is_quantized=True):
            x_data = rand_data(in_shape, dtype, -2, 2)
            x = tpul.Tensor(dtype=dtype, shape=in_shape, data=x_data)
            shape = [in_shape[0], int(in_shape[2] * in_shape[3] / 16 / 16 + 1), head*d]
            out = vit(x, shape, d, head, num, dtype=dtype)
            self.compile_and_check(self.unique_name(case_name), [x], [out], is_quantized=is_quantized)

        _test_model_def(in_shape, d, head, num, dtype, is_quantized)
        # _test_model_def([2, 3, 384, 384], 64, 16, 2)

    def test_Vit_L(self, case_name):
        self.test_Vit(case_name, [1, 3, 384, 384], 64, 16, 1, 'float32', is_quantized=False)
    def test_Vit_L_f16(self, case_name):
        self.test_Vit(case_name, [1, 3, 224, 224], 64, 16, 1, 'float16', is_quantized=True)
    def test_Vit_B(self, case_name):
        self.test_Vit(case_name, [1, 3, 384, 384], 64, 12, 1, 'float32', is_quantized=False)
        self.test_Vit(case_name, [1, 3, 224, 224], 64, 12, 1, 'float16', is_quantized=True)

    #######################################################################
    # swin_t
    # ------------
    def test_SwinT(self, case_name):
        def matmul_weight(x, shape, dtype):
            weight = self.coeff_tensor(shape, dtype=dtype, scale=1/np.sqrt(shape[0]))
            bias = self.coeff_tensor(shape = [1, 1, shape[-1]], dtype="float32", scale=0.2)
            return tpul.matmul(x, weight, bias, out_dtype=dtype)
        def attention_block(x0, shape, d, head, musk=None, musk1=None, dtype="float32"):
            B = shape[0]
            S = shape[1]
            H = shape[2]
            m = matmul_weight(x0, [H, d*head*3], dtype)
            m = tpul.reshape(m, [B, S, 3, head, d])
            m = tpul.permute(m, [2, 3, 0, 1, 4])
            m = tpul.permute(m, [0, 2, 1, 3, 4])
            v, q, k = tpul.split(m, 0, 3)
            v = tpul.reshape(v, [B, head, S, d])
            q = tpul.reshape(q, [B, head, S, d])
            k = tpul.reshape(k, [B, head, S, d])
            q = tpul.mul(q, 1/np.sqrt(d))
            k = tpul.permute(k, [0, 1, 3, 2])
            m0 = tpul.matmul(q, k, out_dtype=dtype)
            m0 = tpul.add(m0, musk) if musk is not None else m0
            m0 = tpul.add(m0, musk1) if musk1 is not None else m0
            m0 = tpul.softmax(m0, 3)
            m1 = tpul.matmul(m0, v, out_dtype=dtype)
            m1 = tpul.permute(m1, [0, 2, 1, 3])
            m1 = tpul.reshape(m1, [B, S, -1])
            out = matmul_weight(m1, [d*head, H], dtype=dtype)
            return out

        def mlp_block(x, shape, dtype):
            H = shape[2]
            norm = layer_norm(x, shape[2], -1, 1e-6, dtype=dtype)
            mat0 = matmul_weight(norm, [H, 4*H], dtype=dtype)
            ge = gelu(mat0)
            mat1 = matmul_weight(ge, [4*H, H], dtype=dtype)
            out = tpul.add(norm, mat1)
            return out

        def slice_block(x, num, axes):
            slice0 = tpul.slice(x, num, 9223372036854775807, axes=axes)
            slice1 = tpul.slice(x, 0, num, axes=axes)
            concat = tpul.concat([slice0, slice1], axis=axes)
            return concat

        def depth2space_block(x, shape):
            reshape = tpul.reshape(x, shape)
            slice0 = tpul.slice(reshape, 0, 9223372036854775807, 2, axes=1)
            slice1 = tpul.slice(reshape, 1, 9223372036854775807, 2, axes=1)
            slice0_0 = tpul.slice(slice0, 0, 9223372036854775807, 2, axes=2)
            slice0_1 = tpul.slice(slice0, 1, 9223372036854775807, 2, axes=2)
            slice1_0 = tpul.slice(slice1, 0, 9223372036854775807, 2, axes=2)
            slice1_1 = tpul.slice(slice1, 1, 9223372036854775807, 2, axes=2)
            concat = tpul.concat([slice0_0, slice0_1, slice1_0, slice1_1], -1)
            return concat

        def transformer_block(x, shape, d, head, has_slice=False, dtype="float32"):
            norm = layer_norm(x, shape[2], -1, 1e-6, dtype=dtype)
            if has_slice:
                reshape = tpul.reshape(norm, [shape[0], 56, 56, shape[2]])
                slice0 = slice_block(reshape, 3, axes=1)
                norm = slice_block(slice0, 3, axes=2)
            reshape = tpul.reshape(norm, [shape[0], 8, 7, 8, 7, shape[2]])
            permute = tpul.permute(reshape, [0, 1, 3, 2, 4, 5])
            reshape2 = tpul.reshape(permute, [shape[0] * 64, 49, shape[2]])
            musk = self.coeff_tensor([1, head, 49, 49], dtype=dtype, scale=1.0)
            musk1 = None if has_slice else self.coeff_tensor([shape[0] * 64, 1, 49, 49], dtype=dtype, scale=1.0)
            self_atten = attention_block(reshape2, [shape[0]*64, 49, shape[2]], d, head, musk, musk1, dtype=dtype)
            reshape3 = tpul.reshape(self_atten, [shape[0], 8, 8, 7, 7, shape[2]])
            permute2 = tpul.permute(reshape3, [0, 1, 3, 2, 4, 5])
            if has_slice:
                reshape = tpul.reshape(permute2, [shape[0], 56, 56, shape[2]])
                slice0 = slice_block(reshape, -3, axes=1)
                norm = slice_block(slice0, -3, axes=2)
            reshape4 = tpul.reshape(permute2, shape)
            add = tpul.add(x, reshape4)
            mlp = mlp_block(add, shape, dtype=dtype)
            return mlp

        def gelu(x):
            div = tpul.mul(x, 1/np.sqrt(2))
            erf = tpul.erf(div)
            add = tpul.add(erf, 1.0)
            mul = tpul.mul(x, add)
            mulc = tpul.mul(mul, 0.5)
            return mulc

        def layer_norm(x, oc, axis, eps, dtype):
            mean = tpul.reduce(x, 'ReduceMean', axes=axis)
            sub = tpul.sub(x, mean)
            pow = tpul.square(sub)
            pow_mean = tpul.reduce(pow, 'ReduceMean', axes=axis)
            add = tpul.add(pow_mean, eps)
            if dtype != "float32":
                add = tpul.cast(add, "float32")
            rsqrt = tpul.rsqrt(add)
            if dtype != "float32":
                rsqrt = tpul.cast(rsqrt, dtype)
            mul = tpul.mul(sub, rsqrt)
            gamma = self.coeff_tensor([oc], dtype=dtype, scale=1.0)
            beta = self.coeff_tensor(shape=[oc], dtype=dtype, scale=0.05)
            gam = tpul.mul(mul, gamma)
            bet = tpul.add(gam, beta)
            return bet

        def swin_t(x, shape, d, head, dtype="float32"):
            H = d * head
            conv0 = self.conv_op(x, [H, 3, 4, 4], [4, 4], bias=True, dtype=dtype)
            # fetch_shape0 = tpul.shape_fetch(conv0)
            # slice0 = tpul.slice(fetch_shape0, 0, 2, axes=0)
            # neg_one = tpul.Tensor([1], ttype="coeff", data=np.array([-1]).astype("int32"), dtype="int32")
            # concat0 = tpul.concat([slice0, neg_one], 0)
            reshape0 = tpul.reshape(conv0, [shape[0], shape[2], shape[1]])
            permut1 = tpul.permute(reshape0, [0, 2, 1])
            norm1 = layer_norm(permut1, shape[2], -1, 1e-6, dtype=dtype)
            trans2 = transformer_block(norm1, [shape[0], shape[1], shape[2]], d, head, dtype=dtype)
            trans3 = transformer_block(trans2, [shape[0], shape[1], shape[2]], d, head, has_slice=True, dtype=dtype)
            reshape3_0 = tpul.reshape(trans3, [shape[0], 56, 56, 96])
            dep3 = depth2space_block(reshape3_0, [shape[0], 56, 56, 96])
            reshape3_1 = tpul.reshape(dep3, [shape[0], 784, 384])
            # norm3 = layer_norm(reshape3_1, 384, -1, 1e-6, dtype=dtype)
            # trans4 = transformer_block(norm3, [shape[0], 784, 384], d, head*2, dtype=dtype)
            # trans5 = transformer_block(trans4, [shape[0], 784, 192], d, head*2, has_slice=True, dtype=dtype)
            # reshape5_0 = tpul.reshape(trans5, [shape[0], 28, 28, 192])
            # dep5 = depth2space_block(reshape5_0, [shape[0], 28, 28, 192])
            # reshape5_1 = tpul.reshape(dep5, [shape[0], 196, 768])
            # norm5 = layer_norm(reshape5_1, 768, -1, 1e-6, dtype=dtype)
            # trans6 = transformer_block(norm5, [shape[0], 196, 768], d, head*4, dtype=dtype)
            # trans7 = transformer_block(trans6, [shape[0], 196, 384], d, head*4, has_slice=True, dtype=dtype)
            # trans8 = transformer_block(trans7, [shape[0], 196, 384], d, head*4, dtype=dtype)
            # trans9 = transformer_block(trans8, [shape[0], 196, 384], d, head*4, has_slice=True, dtype=dtype)
            return reshape3_1

        @tpulang(self.chip)
        def _test_model_def(in_shape, d, head, dtype='float32', is_quantized=True):
            x_data = rand_data(in_shape, dtype, -2, 2)
            x = tpul.Tensor(dtype=dtype, shape=in_shape, data=x_data)
            shape = [in_shape[0], 3136, head*d]
            out = swin_t(x, shape, d, head, dtype=dtype)
            self.compile_and_check(self.unique_name(case_name), [x], [out], is_quantized=is_quantized)

        _test_model_def([2, 3, 224, 224], 32, 3, is_quantized=False)
        _test_model_def([1, 3, 224, 224], 32, 3, 'float16', is_quantized=True)

    #######################################################################
    # Convolution
    # ------------
    def conv_op(self,
                x,
                kshape,
                stride,
                pad=None,
                group=1,
                dilation=[1, 1],
                bias=False,
                dtype=None):
        oc = kshape[0]
        weight = self.coeff_tensor(kshape, x.dtype, scale=1/(math.sqrt(kshape[1] * kshape[2] * kshape[3])))
        bias = self.coeff_tensor([oc], "float32") if bias else None
        conv = tpul.conv(x,
                            weight,
                            bias=bias,
                            stride=stride,
                            pad=pad,
                            dilation=dilation,
                            group=group,
                            out_dtype=dtype if dtype is not None else x.dtype)
        return conv

    def test_Conv2d(self, case_name):
        """Conv 2D"""

        @tpulang(self.chip)
        def _test_convolution(input_shape: List[int],
                              kernel_shape: List[int],
                              stride: List[int] = [1, 1],
                              dilation: List[int] = [1, 1],
                              pad: List[int] = None,
                              group=1,
                              dtype="float32",
                              is_quantized=False):
            x_data = rand_data(input_shape, dtype)
            x = tpul.Tensor(dtype=dtype, shape=input_shape, data=x_data)
            conv = self.conv_op(x,
                                kernel_shape,
                                stride,
                                pad,
                                group=group,
                                dilation=dilation,
                                dtype=dtype)
            self.compile_and_check(self.unique_name(case_name), [x], [conv], is_quantized=is_quantized)

        _test_convolution([1, 3, 28, 28], [12, 1, 1, 1], group=3)
        _test_convolution([1, 3, 32, 32], [12, 3, 3, 3], stride=[2, 2], pad=[0, 0, 1, 1])
        _test_convolution([1, 3, 28, 28], [12, 1, 1, 1], group=3, dtype="float32", is_quantized=True)
        _test_convolution([1, 3, 28, 28], [12, 1, 1, 1], group=3, dtype="float16", is_quantized=True)
        _test_convolution([1, 3, 32, 32], [12, 3, 3, 3], stride=[2, 2], pad=[0, 0, 1, 1], dtype="float16", is_quantized=True)

    #######################################################################
    # Lenet
    # ------------
    def test_Lenet(self, case_name):

        def model_lenet(x):
            conv0 = self.conv_op(x, kshape=[32, 1, 5, 5], stride=[1,1], pad=[2, 2, 2, 2], dtype='float32')
            relu1 = self.relu_op(conv0)
            maxpool2 = self.maxpool_op(relu1, kshape=[2, 2], stride=[2, 2], pad=[0, 0, 0, 0])
            conv3 = self.conv_op(maxpool2, kshape=[64, 32, 5, 5], stride=[1,1], pad=[2, 2, 2, 2], dtype='float32')
            relu4 =  self.relu_op(conv3)
            maxpool5 =self.maxpool_op(relu4, kshape=[2, 2], stride=[2, 2], pad=[0, 0, 0, 0])
            conv6 = self.conv_op(maxpool5, kshape=[1024, 64, 7, 7], stride=[1,1], dtype='float32')
            relu7 =  self.relu_op(conv6)
            conv9 = self.conv_op(relu7,  kshape=[10, 1024, 1, 1], stride=[1,1], dtype='float32')
            return conv9

        @tpulang(self.chip)
        def _test_lenet(in_shape):
            x_data = (rand_data(in_shape, 'float32') - 0.5) * 256
            x = tpul.Tensor(dtype='float32', shape=in_shape, data=x_data)
            out = model_lenet(x=x)
            self.compile_and_check(case_name, [x], [out])

        _test_lenet([1, 1, 28, 28])

    #######################################################################
    # Copy
    # ------------
    def copy_op(self, input):
        copy = tpul.copy(input)
        return copy

    def test_Copy(self, case_name):
        """Copy"""

        @tpulang(self.chip)
        def _test_copy(shape: List[int], dtype="float32", is_quantized=False):
            x_data = rand_data(shape, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape, data=x_data)
            copy = self.copy_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [copy], is_quantized=is_quantized)

        _test_copy([1, 3, 28, 28])
        _test_copy([1, 3, 32, 32])
        _test_copy([1, 3, 32, 32], dtype="float16", is_quantized=True)
        _test_copy([1, 3, 32, 32], dtype="int8", is_quantized=True)

    # #######################################################################
    # # Cast
    # # ------------
    # def cast_op(self, input, out_dtype):
    #     cast = tpul.cast(input, out_dtype)
    #     return cast

    # def test_Cast(self, case_name):
    #     """Cast"""

    #     @tpulang(self.chip)
    #     def _test_cast(shape: List[int], dtype="float32",out_dtype ='float32'):
    #         x_data = rand_data(shape, dtype)
    #         x = tpul.Tensor(dtype=dtype, shape=shape, data=x_data)
    #         cast = self.cast_op(x, out_dtype)
    #         self.compile_and_check(self.unique_name(case_name), [x], [cast])

    #     # input f32, output f32
    #     _test_cast([1, 3, 32, 32])
    #     # input f16, output f32
    #     _test_cast([1, 3, 28, 28], dtype="float16")
    #     # input int8, output
    #     _test_cast([1, 3, 28, 28], out_dtype="int8")

    #######################################################################
    # Clamp
    # ------------
    def clamp_op(self, input):
        clamp = tpul.clamp(input, -100., 100.)
        return clamp

    def test_Clamp(self, case_name):
        """Clamp"""

        @tpulang(self.chip)
        def _test_clamp(shape: List[int], dtype="float32", is_quantized=False):
            x_data = rand_data(shape, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape, data=x_data)
            clamp = self.clamp_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [clamp], is_quantized=is_quantized)

        _test_clamp([1, 3, 28, 28])
        _test_clamp([1, 3, 32, 32])
        _test_clamp([1, 3, 32, 32], dtype="float16", is_quantized=True)

    #######################################################################
    # Matmul
    # ------------
    def matmul_op(self, left, right, bias=None, dtype="float32"):
        matmul = tpul.matmul(left, right, bias, out_dtype=dtype)
        return matmul

    def test_MatMul(self, case_name):
        @tpulang(self.chip)
        def _test_matmul(shape_x: List[int], shape_y: List[int], bias_shape: List[int] = None, dtype="float32", is_quantized=False, is_fuse_rq=False, has_bias=False,bias_dtype="int32",requant_mode=2, input_transpose: bool = False):
            left = rand_data(shape_x, dtype)
            right = rand_data(shape_y, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=left)
            y = tpul.Tensor(dtype=dtype, shape=shape_y, data=right, ttype="coeff")
            matmul = self.matmul_op(x, y, dtype=dtype)
            self.compile_and_check(self.unique_name(case_name), [x], [matmul], is_quantized=is_quantized)

        _test_matmul([1, 3, 28, 10], [1, 3, 10, 8])
        _test_matmul([1, 3, 28, 10], [1, 3, 10, 8], dtype="float16", is_quantized=True)

    def matmul_int_op(self,input,
                        right,
                        bias = None,
                        input_transpose: bool = False,
                        right_transpose: bool = False,
                        output_transpose: bool = False,
                        keep_dims: bool = True,
                        out_dtype: str = "int8",
                        out_name: str = None,
                        multiplier: Union[int, List[int]] = None,
                        shift: Union[int, List[int]] = None,
                        offset: Union[int, List[int]] = None,
                        requant_mode: int = 2,  # Default to "MultiplierShift"
                        round_mode: str = 'half_away_from_zero'):
        assert len(input.shape) == len(right.shape) and len(input.shape) == len(bias.shape)
        rq_axis_ = -1
        matmul_output = tpul.matmul_int(input, right, bias, input_transpose=False, right_transpose=right_transpose, output_transpose=False, keep_dims=True, input_zp=0, right_zp=0, out_dtype="int32", out_name="matmul_int")
        requantized_output = tpul.requant_int(matmul_output, multiplier, shift, offset, requant_mode, out_dtype="int8", out_name='requant_pc', round_mode=round_mode, rq_axis=rq_axis_, fuse_rq_to_matmul=True)
        shape_right_tmp = [1 for i in range(len(input.shape)-2)] + [right.shape[-1],256]
        right_tmp = rand_data(shape_right_tmp, dtype="int8")

        weight  = tpul.Tensor(dtype="int8", shape=shape_right_tmp, data=right_tmp, ttype="coeff")
        matmul_output_ = tpul.matmul_int(requantized_output,  weight, None, input_transpose=False, right_transpose=False, output_transpose=False, keep_dims=True, input_zp=0, right_zp=0, out_dtype="int32", out_name="matmul_int_")
        return matmul_output_

    def test_MatMulRQ_Int_Group(self, case_name):

        @tpulang(self.chip)
        def _test_matmul(shape_x: List[int], shape_y: List[int], bias_shape: List[int] = None, dtype="float32", is_quantized=False, is_fuse_rq=False, has_bias=False,bias_dtype="int32",requant_mode=2, right_transpose: bool = False):
            left = rand_data(shape_x, dtype)
            right = rand_data(shape_y, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=left)
            y = tpul.Tensor(dtype=dtype, shape=shape_y, data=right, ttype="coeff")
            z = None
            if has_bias:
                bias = rand_data(bias_shape, bias_dtype)
                z  = tpul.Tensor(dtype=bias_dtype, shape=bias_shape, data=bias, ttype="coeff")
            if is_quantized and is_fuse_rq:
                multiplier = [random.randint(1000, 10000) for _ in range(shape_y[-1])]
                rshift = rshift = [23 for _ in range(shape_y[-1])]
                offset = 0
                out_dtype_ = "int8"
                if has_bias:
                    out_dtype_ = bias_dtype
                matmul = self.matmul_int_op(x, y, z, out_dtype=out_dtype_, multiplier=multiplier, shift=rshift, offset=offset,requant_mode=requant_mode, right_transpose=right_transpose)
            else:
                matmul = self.matmul_op(x, y, z, dtype=dtype)
            self.compile_and_check(self.unique_name(case_name), [x], [matmul], is_quantized=is_quantized)

        _test_matmul([1, 3, 28, 10], [1, 3, 10, 8])
        _test_matmul([1, 3, 28, 10], [1, 3, 10, 8], dtype="float16", is_quantized=True)
        _test_matmul([1, 1, 256, 1024], [1, 1, 1024, 1024], [1, 1, 1, 1024], dtype="int8", is_quantized=True, is_fuse_rq=True, has_bias=True, requant_mode=2)
        _test_matmul([197, 768], [768, 768], [1, 768], dtype="int8", is_quantized=True, is_fuse_rq=True, has_bias=True, requant_mode=2, right_transpose = True)

    def test_MatMulRQ_OP(self, case_name):
        @tpulang(self.chip)
        def _test_matmul(shape_x: List[int], shape_y: List[int], bias_shape: List[int] = None, idtype="float32", is_quantized=False, is_fuse_rq=False, has_bias=False,bias_dtype="int32", odtype = "int8", requant_mode=2, round_mode='half_away_from_zero'):
            left = rand_data(shape_x, idtype)
            right = rand_data(shape_y, idtype)
            x = tpul.Tensor(dtype=idtype, shape=shape_x, data=left)
            y = tpul.Tensor(dtype=idtype, shape=shape_y, data=right, ttype="coeff")
            z = None
            if has_bias:
                bias = rand_data(bias_shape, bias_dtype)
                z  = tpul.Tensor(dtype=bias_dtype, shape=bias_shape, data=bias, ttype="coeff")

            if is_quantized and is_fuse_rq:
                multiplier = [random.randint(1000, 10000) for _ in range(1024)]
                rshift = [-23]
                offset = 0
                matmul = tpul.matmulrq_int_op(x, y, z, out_dtype=odtype, multiplier=multiplier, shift=rshift, offset=offset,requant_mode=requant_mode, round_mode=round_mode)
                self.compile_and_check(self.unique_name(case_name), [x], [matmul], is_quantized=is_quantized)

        _test_matmul([2 ,197, 768], [2, 768, 768], [2, 1, 768], idtype="int8", odtype="int8", is_quantized=True, is_fuse_rq=True, has_bias=True, requant_mode=2,round_mode='half_up')

    #######################################################################
    # Maxpool
    # ------------
    def maxpool_op(self,
                input_0,
                kshape,
                stride,
                pad=None,
                ceil_mode=False,
                scale=None,
                zero_point=None):
        scale = [scale, scale] if scale != None else scale
        zero_point = [zero_point, zero_point] if zero_point != None else zero_point
        if len(input_0.shape) == 5:
            maxpool = tpul.maxpool3d(input_0, kshape, stride, pad, ceil_mode, scale=scale, zero_point=zero_point)
        else:
            maxpool = tpul.maxpool2d(input_0, kshape, stride, pad, ceil_mode, scale=scale, zero_point=zero_point)
        return maxpool

    def test_Maxpool(self, case_name):
        """Maxpool"""

        @tpulang(self.chip)
        def _test_maxpool(shape_x: List[int],
                                kshape: List[int] = [1,1],
                                stride: List[int] = [1, 1],
                                pad: List[int] = None,
                                scale=None,
                                zero_point=None,
                                dtype="float32",
                                is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input, scale=scale, zero_point=zero_point)
            maxpool = self.maxpool_op(x, kshape, stride, pad, scale=scale, zero_point=zero_point)
            self.compile_and_check(self.unique_name(case_name), [x], [maxpool], is_quantized=is_quantized)

        @tpulang(self.chip)
        def _test_maxpool_mask(shape_x: List[int],
                                kshape: List[int] = [1, 1],
                                stride: List[int] = [1, 1],
                                pad: List[int] = None,
                                dtype="float32",
                                is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            maxpool, mask = tpul.maxpool2d_with_mask(x, kshape, stride, pad)
            self.compile_and_check(self.unique_name(case_name), [x], [maxpool, mask], is_quantized=is_quantized)
        _test_maxpool([1, 32, 28, 28], kshape = [2, 2], stride = [2, 2], pad=[0, 0, 0, 0])
        _test_maxpool([1, 32, 28, 28], kshape = [2, 2], stride = [2, 2], pad=[0, 0, 0, 0], dtype="float16", is_quantized=True)
        _test_maxpool([1, 32, 28, 28], kshape = [2, 2], stride = [2, 2], pad=[0, 0, 0, 0], scale=10.0, dtype="int8", is_quantized=True)
    # _test_maxpool_mask([1, 32, 28, 28], kshape = [2, 2], stride = [2, 2], pad=[0, 0, 0, 0])

    def test_Maxpool3d(self, case_name):
        """Maxpool3d"""
        @tpulang(self.chip)
        def _test_maxpool3d(shape_x: List[int],
                                kshape: List[int] = [1,1,1],
                                stride: List[int] = [1, 1, 1],
                                pad: List[int] = None,
                                scale=None,
                                zero_point=None,
                                dtype="float32",
                                is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input, scale=scale, zero_point=zero_point)
            maxpool = self.maxpool_op(x, kshape, stride, pad, scale=scale, zero_point=zero_point)
            self.compile_and_check(self.unique_name(case_name), [x], [maxpool], is_quantized=is_quantized)

        _test_maxpool3d([1, 32, 28, 28, 28], kshape = [2, 2, 2], stride = [2, 2, 2], pad=[0, 0, 0, 0, 0, 0])
        _test_maxpool3d([1, 32, 28, 28, 28], kshape = [2, 2, 2], stride = [2, 2, 2], pad=[0, 0, 0, 0, 0, 0], dtype="float16", is_quantized=True)
        _test_maxpool3d([1, 32, 28, 28, 28], kshape = [2, 2, 2], stride = [2, 2, 2], pad=[0, 0, 0, 0, 0, 0], scale=10.0, dtype="int8", is_quantized=True)

    #######################################################################
    # Avgpool
    # ------------
    def avgpool_op(self,
                input_0,
                kshape,
                stride,
                pad=None,
                ceil_mode=False,
                scale=None,
                zero_point=None):
        scale = [scale, scale] if scale != None else scale
        zero_point = [zero_point, zero_point] if zero_point != None else zero_point
        print(scale,zero_point)
        if len(input_0.shape) == 5:
            maxpool = tpul.avgpool3d(input_0, kshape, stride, pad, ceil_mode, scale=scale, zero_point=zero_point)
        else:
            maxpool = tpul.avgpool2d(input_0, kshape, stride, pad, ceil_mode, scale=scale, zero_point=zero_point)
        return maxpool

    def test_Avgpool(self, case_name):
        """Avgpool"""

        @tpulang(self.chip)
        def _test_avgpool(shape_x: List[int],
                                kshape: List[int] = None,
                                stride: List[int] = None,
                                pad: List[int] = None,
                                scale=None,
                                zero_point=None,
                                dtype="float32",
                                is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input, scale=scale, zero_point=zero_point)
            avgpool = self.avgpool_op(x, kshape, stride, pad, scale=scale, zero_point=zero_point)
            self.compile_and_check(self.unique_name(case_name), [x], [avgpool], is_quantized=is_quantized)

        _test_avgpool([1, 32, 28, 28], kshape = [2, 2], stride = [2, 2], pad=[0, 0, 0, 0])
        _test_avgpool([1, 32, 28, 28], kshape = [2, 2], stride = [2, 2], pad=[0, 0, 0, 0], dtype="float16", is_quantized=True)
        _test_avgpool([1, 32, 28, 28], kshape = [2, 2], stride = [2, 2], pad=[0, 0, 0, 0], scale=10.0, dtype="int8", is_quantized=True)

    def test_Avgpool3d(self, case_name):
        """Avgpool3d"""

        @tpulang(self.chip)
        def _test_avgpool3d(shape_x: List[int],
                                kshape: List[int] = None,
                                stride: List[int] = None,
                                pad: List[int] = None,
                                scale=None,
                                zero_point=None,
                                dtype="float32",
                                is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input, scale=scale, zero_point=zero_point)
            avgpool = self.avgpool_op(x, kshape, stride, pad, scale=scale, zero_point=zero_point)
            self.compile_and_check(self.unique_name(case_name), [x], [avgpool], is_quantized=is_quantized)
       # _test_avgpool3d([1, 32, 28, 28, 28], kshape = [2, 2, 2], stride = [2, 2, 2], pad=[0, 0, 0, 0, 0, 0])
       # _test_avgpool3d([1, 32, 28, 28, 28], kshape = [2, 2, 2], stride = [2, 2, 2], pad=[0, 0, 0, 0, 0, 0], dtype="float16", is_quantized=True)
        _test_avgpool3d([1, 32, 28, 28, 28])
        _test_avgpool3d([1, 32, 28, 28, 28], kshape = 2)
        _test_avgpool3d([4, 8, 12, 20, 24], kshape = 2, stride = 2, pad = 0)
        _test_avgpool3d([4, 8, 12, 20, 24], kshape = [2, 2, 2], stride = [2, 2, 2], pad=[0, 0, 0, 0, 0, 0])
        _test_avgpool3d([4, 8, 12, 20, 24], kshape = [2, 2, 2], stride = [2, 2, 2], pad=[0, 0, 0, 0, 0, 0], scale=10.0, dtype="int8", is_quantized=True)

    #######################################################################
    # Relu
    # ------------
    def relu_op(self, input):
        relu = tpul.relu(input)
        return relu

    def test_Relu(self, case_name):
        """Relu"""

        @tpulang(self.chip)
        def _test_relu(shape_x: List[int], scale=None, zero_point=None, dtype="float32", is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input, scale=scale, zero_point=zero_point)
            relu = self.relu_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [relu], is_quantized=is_quantized)

        _test_relu([1, 32, 28, 28])
        _test_relu([1, 32, 28, 28], dtype="float16", is_quantized=True)
        _test_relu([1, 32, 28, 28], scale=10.0, zero_point=0, dtype="int8", is_quantized=True)

    #######################################################################
    # LeakyRelu
    # ------------
    def leaky_relu_op(self, input):
        leaky_relu = tpul.leaky_relu(input)
        return leaky_relu

    def test_LeakyRelu(self, case_name):
        """LeakyRelu"""

        @tpulang(self.chip)
        def _test_leaky_relu(shape_x: List[int], scale=None, zero_point=None, dtype="float32", is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input, scale=scale, zero_point=zero_point)
            leaky_relu = self.leaky_relu_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [leaky_relu], is_quantized=is_quantized)

        _test_leaky_relu([1, 32, 28, 28])
        _test_leaky_relu([1, 32, 28, 28], dtype="float16", is_quantized=True)
        _test_leaky_relu([1, 32, 28, 28], scale=10.0, dtype="int8", is_quantized=True)

    #######################################################################
    # PRelu
    # ------------
    def test_PRelu(self, case_name):
        """PRelu"""

        @tpulang(self.chip)
        def _test_prelu(shape_x: List[int], shape_s: List[int], dtype="float32", is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            slope_coeff = rand_data(shape_s, dtype, 0.0001, 0.2)
            slope = tpul.Tensor(shape_s, ttype="coeff", data=slope_coeff, dtype=dtype)
            prelu = tpul.prelu(x, slope)
            self.compile_and_check(self.unique_name(case_name), [x], [prelu], is_quantized=is_quantized)

        _test_prelu([1, 32, 28, 28], [1])
        _test_prelu([1, 32, 28, 28], [1, 32, 1, 1], dtype="float16", is_quantized=True)

    #######################################################################
    # Abs
    # ------------
    def test_Abs(self, case_name):
        """Abs"""

        @tpulang(self.chip)
        def _test_abs(shape_x: List[int], scale=None, zero_point=None, dtype="float32", is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input, scale=scale, zero_point=zero_point)
            y_data = np.ones(shape_x, dtype=dtype)
            y = tpul.Tensor(shape_x, ttype="coeff", data=y_data, dtype=dtype)
            abs = tpul.add(tpul.abs(x), y)
            self.compile_and_check(self.unique_name(case_name), [x], [abs], is_quantized=is_quantized)

        _test_abs([3, 2])
        # _test_abs([1, 32, 28, 28], dtype="float16", is_quantized=True)
        # _test_abs([1, 32, 28, 28], scale=3.0, dtype="int8", is_quantized=True)

    #######################################################################
    # No Save
    # ------------
    def test_NoSave(self, case_name):

        @tpulang(self.chip)
        def _test_nosave(shape_x: List[int], scale=None, zero_point=None, dtype="float32", is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input, scale=scale, zero_point=zero_point)
            y_data = np.ones(shape_x, dtype=dtype)
            y = tpul.Tensor(shape_x, ttype="coeff", data=y_data, dtype=dtype)
            abs = tpul.add(tpul.abs(x), y)
            self.compile_and_check(self.unique_name(case_name), [x], [abs], is_quantized=is_quantized)

        _test_nosave([3, 2])

    #######################################################################
    # Ceil
    # ------------
    def ceil_op(self, input, scale=None, zero_point=None):
        scale = scale if scale == None else [scale, scale]
        zero_point = zero_point if zero_point == None else [zero_point, zero_point]
        ceil = tpul.ceil(input, scale=scale, zero_point=zero_point)
        return ceil

    def test_Ceil(self, case_name):
        """Ceil"""

        @tpulang(self.chip)
        def _test_ceil(shape_x: List[int], scale=None, zero_point=None, dtype="float32", is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input, scale=scale, zero_point=zero_point)
            ceil = self.ceil_op(x, scale=scale, zero_point=zero_point)
            self.compile_and_check(self.unique_name(case_name), [x], [ceil], is_quantized=is_quantized)

        _test_ceil([1, 32, 28, 28])
        _test_ceil([1, 32, 28, 28], dtype="float16", is_quantized=True)
        _test_ceil([1, 32, 28, 28], scale=3.0, dtype="int8", is_quantized=True)

    #######################################################################
    # Floor
    # ------------
    def floor_op(self, input, scale=None, zero_point=None):
        scale = scale if scale == None else [scale, scale]
        zero_point = zero_point if zero_point == None else [zero_point, zero_point]
        floor = tpul.floor(input, scale=scale, zero_point=zero_point)
        return floor

    def test_Floor(self, case_name):
        """Floor"""

        @tpulang(self.chip)
        def _test_floor(shape_x: List[int], scale=None, zero_point=None, dtype="float32", is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input, scale=scale, zero_point=zero_point)
            floor = self.floor_op(x, scale=scale, zero_point=zero_point)
            self.compile_and_check(self.unique_name(case_name), [x], [floor], is_quantized=is_quantized)

        _test_floor([1, 32, 28, 28])
        _test_floor([1, 32, 28, 28], dtype="float16", is_quantized=True)
        _test_floor([1, 32, 28, 28], scale=3.0, dtype="int8", is_quantized=True)

    #######################################################################
    # Round
    # ------------
    def round_op(self, input):
        round = tpul.round(input)
        return round

    def test_Round(self, case_name):
        """Round"""

        @tpulang(self.chip)
        def _test_round(shape_x: List[int], dtype="float32", is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            round = self.round_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [round], is_quantized=is_quantized)

        _test_round([1, 32, 28, 28])
        _test_round([1, 32, 28, 28], dtype="float16", is_quantized=True)

    #######################################################################
    # Sin
    # ------------
    def sin_op(self, input, scale=None, zero_point=None):
        scale = scale if scale == None else [scale, scale]
        zero_point = zero_point if zero_point == None else [zero_point, zero_point]
        sin = tpul.sin(input, scale=scale, zero_point=zero_point)
        return sin

    def test_Sin(self, case_name):
        """sin"""

        @tpulang(self.chip)
        def _test_sin(shape_x: List[int], scale=None, zero_point=None, dtype="float32", is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input, scale=scale, zero_point=zero_point)
            sin = self.sin_op(x, scale=scale, zero_point=zero_point)
            self.compile_and_check(self.unique_name(case_name), [x], [sin], is_quantized=is_quantized)

        _test_sin([1, 32, 28, 28])
        _test_sin([1, 32, 28, 28], scale=3.0, dtype="int8", is_quantized=True)

    def _test_quantized_active_op(self, shape, func, case_name, scale=None, zero_point=None, dtype="float32"):
        def active_op(input, scale=None, zero_point=None):
            scale = scale if scale == None else [scale, scale*2]
            zero_point = zero_point if zero_point == None else [zero_point, zero_point]
            out = func(input, scale=scale, zero_point=zero_point)
            return out

        @tpulang(self.chip)
        def quantized_op(shape):
            import copy
            input = rand_data(shape, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape, data=input, scale=scale, zero_point=zero_point)
            new_shape = copy.copy(shape)
            new_shape[-1] = 1
            new_shape[-2] = -1
            reshape = tpul.reshape(x, new_shape)
            out = active_op(reshape, scale=scale, zero_point=zero_point)
            self.compile_and_check(self.unique_name(case_name), [x], [out], is_quantized=True)
        quantized_op(shape)

    #######################################################################
    # Cos
    # ------------
    def cos_op(self, input):
        cos = tpul.cos(input)
        return cos

    def test_Cos(self, case_name):
        """cos"""

        @tpulang(self.chip)
        def _test_cos(shape_x: List[int], dtype="float32"):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            cos = self.cos_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [cos])

        _test_cos([1, 32, 28, 28])
        self._test_quantized_active_op([1, 32, 28, 28], tpul.cos, case_name, scale=3.0, dtype="int8")

    #######################################################################
    # Exp
    # ------------
    def exp_op(self, input):
        exp = tpul.exp(input)
        return exp

    def test_Exp(self, case_name):
        """exp"""

        @tpulang(self.chip)
        def _test_exp(shape_x: List[int], dtype="float32"):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            exp = self.exp_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [exp])

        _test_exp([1, 32, 28, 28])
        self._test_quantized_active_op([1, 32, 28, 28], tpul.exp, case_name, scale=3.0, dtype="int8")
        self._test_quantized_active_op([1, 32, 28, 28], tpul.exp, case_name, dtype="float16")

    #######################################################################
    # Tanh
    # ------------
    def tanh_op(self, input):
        tanh = tpul.tanh(input)
        return tanh

    def test_Tanh(self, case_name):
        """tanh"""

        @tpulang(self.chip)
        def _test_tanh(shape_x: List[int], dtype="float32"):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            tanh = self.tanh_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [tanh])

        _test_tanh([1, 32, 28, 28])
        self._test_quantized_active_op([1, 32, 28, 28], tpul.tanh, case_name, scale=0.0078125, dtype="int8")
        self._test_quantized_active_op([1, 32, 28, 28], tpul.tanh, case_name, dtype="float16")

    #######################################################################
    # Sigmoid
    # ------------
    def sigmoid_op(self, input):
        sigmoid = tpul.sigmoid(input)
        return sigmoid

    def test_Sigmoid(self, case_name):
        """sigmoid"""

        @tpulang(self.chip)
        def _test_sigmoid(shape_x: List[int], dtype="float32"):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            sigmoid = self.sigmoid_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [sigmoid])

        _test_sigmoid([1, 32, 28, 28])
        self._test_quantized_active_op([1, 32, 28, 28], tpul.sigmoid, case_name, scale=3.0, dtype="int8")

    #######################################################################
    # Elu
    # ------------
    def elu_op(self, input):
        elu = tpul.elu(input)
        return elu

    def test_Elu(self, case_name):
        """elu"""

        @tpulang(self.chip)
        def _test_elu(shape_x: List[int], dtype="float32"):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            elu = self.elu_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [elu])

        _test_elu([1, 32, 28, 28])
        self._test_quantized_active_op([1, 32, 28, 28], tpul.elu, case_name, scale=3.0, dtype="int8")

    #######################################################################
    # Sqrt
    # ------------
    def sqrt_op(self, input):
        sqrt = tpul.sqrt(input)
        return sqrt

    def test_Sqrt(self, case_name):
        """sqrt"""

        @tpulang(self.chip)
        def _test_sqrt(shape_x: List[int], dtype="float32"):
            input = np.abs(rand_data(shape_x, dtype))
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            sqrt = self.sqrt_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [sqrt])

        _test_sqrt([1, 32, 28, 28])
        self._test_quantized_active_op([1, 32, 28, 28], tpul.sqrt, case_name, scale=3.0, dtype="int8")

    # #######################################################################
    # Rsqrt
    # ------------
    def rsqrt_op(self, input):
        rsqrt = tpul.rsqrt(input)
        return rsqrt

    def test_Rsqrt(self, case_name):
        """rsqrt"""

        @tpulang(self.chip)
        def _test_rsqrt(shape_x: List[int], dtype="float32"):
            input = np.abs(rand_data(shape_x, dtype))
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            rsqrt = self.rsqrt_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [rsqrt])

        _test_rsqrt([1, 32, 28, 28])
        self._test_quantized_active_op([1, 32, 28, 28], tpul.rsqrt, case_name, scale=3.0, dtype="int8")

    # #######################################################################
    # silu
    # ------------
    def silu_op(self, input):
        silu = tpul.silu(input)
        return silu

    def test_Silu(self, case_name):
        """silu"""

        @tpulang(self.chip)
        def _test_silu(shape_x: List[int], dtype="float32"):
            input = np.abs(rand_data(shape_x, dtype))
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            silu = self.silu_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [silu])

        _test_silu([1, 32, 28, 28])
        self._test_quantized_active_op([1, 32, 28, 28], tpul.silu, case_name, scale=3.0, dtype="int8")

    # #######################################################################
    # swish
    # ------------
    def test_Swish(self, case_name):
        """swish"""

        @tpulang(self.chip)
        def _test_swish(shape_x: List[int], beta: float, dtype="float32", scale=None, zp=None):
            input = np.abs(rand_data(shape_x, dtype))
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            silu = tpul.swish(x, beta=0.5, scale=scale, zero_point=zp)
            self.compile_and_check(self.unique_name(case_name), [x], [silu], is_quantized=dtype!="float32")

        _test_swish([1, 32, 28, 28], 0.3, "float16")
        _test_swish([1, 32, 28, 28], 0.5, "int8", scale=[3.0, 6.0], zp=[0, 0])

    #######################################################################
    # Erf
    # ------------
    def erf_op(self, input):
        erf = tpul.erf(input)
        return erf

    def test_Erf(self, case_name):
        """erf"""

        @tpulang(self.chip)
        def _test_erf(shape_x: List[int], dtype="float32"):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            erf = self.erf_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [erf])

        _test_erf([1, 32, 28, 28])
        self._test_quantized_active_op([1, 32, 28, 28], tpul.erf, case_name, scale=3.0, dtype="int8")
        self._test_quantized_active_op([1, 32, 28, 28], tpul.erf, case_name, dtype="float16")

    #######################################################################
    # Tan
    # ------------
    def tan_op(self, input):
        tan = tpul.tan(input)
        return tan

    def test_Tan(self, case_name):
        """tan"""

        @tpulang(self.chip)
        def _test_tan(shape_x: List[int], dtype="float32"):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            tan = self.tan_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [tan])

        _test_tan([1, 32, 28, 28])

    #######################################################################
    # Softmax
    # ------------
    def softmax_op(self, input, axis=0):
        softmax = tpul.softmax(input, axis=axis)
        return softmax

    def test_Softmax(self, case_name):
        """softmax"""

        @tpulang(self.chip)
        def _test_softmax(shape_x: List[int], axis: int, dtype="float32"):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            softmax = self.softmax_op(x, axis)
            self.compile_and_check(self.unique_name(case_name), [x], [softmax], is_quantized=dtype!="float32")

        @tpulang(self.chip)
        def _test_softmax_int(shape_x: List[int], axis: int, dtype="int8"):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input, scale=0.25)
            reshape = tpul.reshape(x, [1, -1])
            softmax = tpul.softmax_int(reshape, axis, scale=[0.25, 128.0])
            self.compile_and_check(self.unique_name(case_name), [x], [softmax], is_quantized=True)

        _test_softmax([1, 32, 28, 28], axis=1)
        _test_softmax([1, 32, 28, 28], axis=1, dtype="float16")
        _test_softmax_int([1, 32, 1, 1], axis=1)

    #######################################################################
    # Mish
    # ------------
    def mish_op(self, input):
        mish = tpul.mish(input)
        return mish

    def test_Mish(self, case_name):
        """mish"""

        @tpulang(self.chip)
        def _test_mish(shape_x: List[int], dtype="float32"):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            mish = self.mish_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [mish])

        _test_mish([1, 32, 28, 28])
        self._test_quantized_active_op([1, 32, 28, 28], tpul.mish, case_name, scale=3.0, dtype="int8")

    #######################################################################
    # Hswish
    # ------------
    def hswish_op(self, input):
        hswish = tpul.hswish(input)
        return hswish

    def test_Hswish(self, case_name):
        """hswish"""

        @tpulang(self.chip)
        def _test_hswish(shape_x: List[int], dtype="float32"):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            hswish = self.hswish_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [hswish])

        _test_hswish([1, 32, 28, 28])
        self._test_quantized_active_op([1, 32, 28, 28], tpul.hswish, case_name, scale=3.0, dtype="int8")
        self._test_quantized_active_op([1, 32, 28, 28], tpul.hswish, case_name, dtype="float16")

    #######################################################################
    # Arccos
    # ------------
    def arccos_op(self, input):
        arccos = tpul.arccos(input)
        return arccos

    def test_Arccos(self, case_name):
        """arccos"""

        @tpulang(self.chip)
        def _test_arccos(shape_x: List[int], dtype="float32"):
            input = np.clip(rand_data(shape_x, dtype), -0.99, 0.99)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            arccos = self.arccos_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [arccos])

        _test_arccos([1, 32, 28, 28])

    #######################################################################
    # Arctanh
    # ------------
    def arctanh_op(self, input):
        arctanh = tpul.arctanh(input)
        return arctanh

    def test_Arctanh(self, case_name):
        """arctanh"""

        @tpulang(self.chip)
        def _test_arctanh(shape_x: List[int], dtype="float32"):
            input = np.clip(rand_data(shape_x, dtype), -0.99, 0.99)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            arctanh = self.arctanh_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [arctanh])

        _test_arctanh([1, 32, 28, 28])

    #######################################################################
    # Sinh
    # ------------
    def sinh_op(self, input):
        sinh = tpul.sinh(input)
        return sinh

    def test_Sinh(self, case_name):
        """sinh"""

        @tpulang(self.chip)
        def _test_sinh(shape_x: List[int], dtype="float32"):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            sinh = self.sinh_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [sinh])

        _test_sinh([1, 32, 28, 28])
        self._test_quantized_active_op([1, 32, 28, 28], tpul.sinh, case_name, scale=3.0, dtype="int8")

    #######################################################################
    # Cosh
    # ------------
    def cosh_op(self, input):
        cosh = tpul.cosh(input)
        return cosh

    def test_Cosh(self, case_name):
        """cosh"""

        @tpulang(self.chip)
        def _test_cosh(shape_x: List[int], dtype="float32"):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            cosh = self.cosh_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [cosh])

        _test_cosh([1, 32, 28, 28])
        self._test_quantized_active_op([1, 32, 28, 28], tpul.cosh, case_name, scale=3.0, dtype="int8")

    #######################################################################
    # Sign
    # ------------
    def sign_op(self, input):
        sign = tpul.sign(input)
        return sign

    def test_Sign(self, case_name):
        """sign"""

        @tpulang(self.chip)
        def _test_sign(shape_x: List[int], dtype="float32"):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            sign = self.sign_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [sign])

        _test_sign([1, 32, 28, 28])
        self._test_quantized_active_op([1, 32, 28, 28], tpul.sign, case_name, scale=3.0, dtype="int8")
        self._test_quantized_active_op([1, 32, 28, 28], tpul.sign, case_name, dtype="float16")

    #######################################################################
    # Gelu
    # ------------
    def gelu_op(self, input):
        gelu = tpul.gelu(input)
        return gelu

    def test_Gelu(self, case_name):
        """gelu"""

        @tpulang(self.chip)
        def _test_gelu(shape_x: List[int], dtype="float32"):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            gelu = self.gelu_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [gelu])

        _test_gelu([1, 32, 28, 28])
        self._test_quantized_active_op([1, 32, 28, 28], tpul.gelu, case_name, scale=3.0, dtype="int8")
        self._test_quantized_active_op([1, 32, 28, 28], tpul.gelu, case_name, dtype="float16")

    #######################################################################
    # Ln
    # ------------
    def ln_op(self, input, scale=None, zero_point=None):
        scale = scale if scale == None else [scale, scale]
        zero_point = zero_point if zero_point == None else [zero_point, zero_point]
        ln = tpul.ln(input, scale=scale, zero_point=zero_point)
        return ln

    def test_Ln(self, case_name):
        """ln"""

        @tpulang(self.chip)
        def _test_ln(shape_x: List[int], scale=None, zero_point=None, dtype="float32", is_quantized=False):
            # input = np.clip(np.random.randn(*shape_x).astype(dtype) * 10.0, 0.5, 8)
            input = rand_data(shape=shape_x, dtype=dtype, min=0.01, max=8)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input, scale=scale, zero_point=zero_point)
            ln = self.ln_op(x, scale=scale, zero_point=zero_point)
            self.compile_and_check(self.unique_name(case_name), [x], [ln], is_quantized=is_quantized)

        _test_ln([1, 3, 32, 32])
        _test_ln([1, 32, 28, 28], scale=3.0, dtype="int8", is_quantized=True)
        _test_ln([1, 32, 28, 28], dtype="float16", is_quantized=True)

    #######################################################################
    # square
    # ------------
    def square_op(self, input):
        square = tpul.square(input)
        return square

    def test_Square(self, case_name):
        """square"""

        @tpulang(self.chip)
        def _test_square(shape_x: List[int], dtype="float32"):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            square = self.square_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [square])

        _test_square([1, 32, 28, 28])
        self._test_quantized_active_op([1, 32, 28, 28], tpul.square, case_name, scale=3.0, dtype="int8")
        self._test_quantized_active_op([1, 32, 28, 28], tpul.square, case_name, dtype="float16")

    #######################################################################
    # Hsigmoid
    # ------------
    def hsigmoid_op(self, input):
        hsigmoid = tpul.hsigmoid(input)
        return hsigmoid

    def test_Hsigmoid(self, case_name):
        """hsigmoid"""

        @tpulang(self.chip)
        def _test_hsigmoid(shape_x: List[int], dtype="float32"):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            hsigmoid = self.hsigmoid_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [hsigmoid])

        _test_hsigmoid([1, 32, 28, 28])
        self._test_quantized_active_op([1, 32, 28, 28], tpul.hsigmoid, case_name, scale=3.0, dtype="int8")

    #######################################################################
    # Arg
    # ------------

    def test_Arg(self, case_name):
        """arg"""

        @tpulang(self.chip)
        def _test_arg(shape_x: List[int], dtype="float32"):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input)
            arg1, arg2 = tpul.arg(x)
            self.compile_and_check(self.unique_name(case_name), [x], [arg1, arg2])

        _test_arg([1, 32, 28, 28])

    #######################################################################
    # Permute
    # ------------
    def permute_op(self, input):
        permute = tpul.permute(input, [0, 2, 3, 1])
        return permute

    def test_Permute(self, case_name):
        """permute"""

        @tpulang(self.chip)
        def _test_permute(shape_x: List[int], scale=None, zero_point=None, dtype="float32", is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input, scale=scale, zero_point=zero_point)
            permute = self.permute_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [permute], is_quantized=is_quantized)

        _test_permute([1, 1024, 10, 128])
        _test_permute([1, 1024, 10, 128], scale=3.0, dtype="int8", is_quantized=True)
        _test_permute([1, 1024, 10, 128], dtype="float16", is_quantized=True)
        _test_permute([1, 1024, 10, 128], dtype="int16")

    #######################################################################
    # Reshape
    # ------------
    def reshape_op(self, input):

        reshape = tpul.reshape(input, [32, 28, 1, 28])
        return reshape
        # reshape = tpul.reshape(input, [1, 28, 28, 192])
        # add = tpul.add(reshape, 1)
        # reshape1 = tpul.reshape(add, [1, 4, 7, 4, 7, 192])

        # add1 = tpul.add(input, reshape1)
        # reshape2 = tpul.reshape(add1, [1, 784, 192])
        # norm = tpul.layer_norm(reshape2)
        # return norm

    def test_Reshape(self, case_name):
        """reshape"""

        @tpulang(self.chip)
        def _test_reshape(shape_x: List[int], scale=None, zero_point=None, dtype="float32", is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input, scale=scale, zero_point=zero_point)
            reshape = self.reshape_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [reshape], is_quantized=is_quantized)

        _test_reshape([1, 32, 28, 28])
        _test_reshape([1, 32, 28, 28], scale=3.0, dtype="int8", is_quantized=True)
        _test_reshape([1, 32, 28, 28], dtype="float16", is_quantized=True)
        # _test_reshape([1, 4, 7, 4, 7, 192], dtype="float32", is_quantized=True)

    #######################################################################
    # Shape_fetch
    # ------------
    def shape_fetch_op(self, input):
        shape_fetch = tpul.shape_fetch(input)
        return shape_fetch

    def test_Shape_fetch(self, case_name):
        """Shape_fetch"""

        @tpulang(self.chip)
        def _test_shape_fetch(shape_x: List[int], scale=None, zero_point=None, dtype="float32", is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input, scale=scale, zero_point=zero_point)
            shape_fetch = self.shape_fetch_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [shape_fetch], is_quantized=is_quantized)

        _test_shape_fetch([1, 32, 28, 28])
        _test_shape_fetch([1, 32, 28, 28], scale=3.0, dtype="int8", is_quantized=True)
        _test_shape_fetch([1, 32, 28, 28], dtype="float16", is_quantized=True)

    #######################################################################
    # Squeeze
    # ------------
    def squeeze_op(self, input):
        squeeze = tpul.squeeze(input, [0])
        return squeeze

    def test_Squeeze(self, case_name):
        """squeeze"""

        @tpulang(self.chip)
        def _test_squeeze(shape_x: List[int], scale=None, zero_point=None, dtype="float32", is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input, scale=scale, zero_point=zero_point)
            squeeze = self.squeeze_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [squeeze], is_quantized=is_quantized)

        _test_squeeze([1, 32, 28, 28])
        _test_squeeze([1, 32, 28, 28], scale=3.0, dtype="int8", is_quantized=True)
        _test_squeeze([1, 32, 28, 28], dtype="float16", is_quantized=True)

    #######################################################################
    # Pad
    # ------------
    def pad_op(self, input, pad_val=0.0, pad=None):
        pad_val = tpul.Scalar(pad_val)
        pad = tpul.pad(input, value=pad_val, padding=pad)
        return pad

    def test_Pad(self, case_name):
        """pad"""

        @tpulang(self.chip)
        def _test_pad(shape_x: List[int], scale=None, zero_point=None, dtype="float32", is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input, scale=scale, zero_point=zero_point)
            pad = self.pad_op(x, pad=[0, 0, 1, 2, 0, 0, 3, 4])
            self.compile_and_check(self.unique_name(case_name), [x], [pad], is_quantized=is_quantized)

        _test_pad([1, 32, 28, 28])
        _test_pad([1, 32, 28, 28], scale=3.0, dtype="int8", is_quantized=True)
        _test_pad([1, 32, 28, 28], dtype="float16", is_quantized=True)
        _test_pad([1, 32, 28, 28], dtype="int16")

    #######################################################################
    # Tile
    # ------------
    def tile_op(self, input):
        tile = tpul.tile(input, [1,3,2,4])
        return tile

    def test_Tile(self, case_name):
        """tile"""

        @tpulang(self.chip)
        def _test_tile(shape_x: List[int], scale=None, zero_point=None, dtype="float32", is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input, scale=scale, zero_point=zero_point)
            tile = self.tile_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [tile], is_quantized=is_quantized)

        _test_tile([1, 32, 28, 28])
        _test_tile([1, 32, 28, 28], scale=3.0, dtype="int8", is_quantized=True)
        _test_tile([1, 32, 28, 28], dtype="float16", is_quantized=True)
        _test_tile([1, 32, 28, 28], dtype="int16")

    #######################################################################
    # Concat
    # ------------
    def test_Concat(self, case_name):
        """concat"""

        @tpulang(self.chip)
        def _test_concat(shapes: List[List[int]], scale=None, zero_point=None, dtype="float32", is_quantized=False):
            input0 = rand_data(shapes[0], dtype)
            input1 = rand_data(shapes[1], dtype)
            x = tpul.Tensor(dtype=dtype, shape=shapes[0], data=input0, scale=scale, zero_point=zero_point)
            y = tpul.Tensor(dtype=dtype, shape=shapes[1], data=input1, scale=scale, zero_point=zero_point)
            scale_values = scale if scale is None else [scale, scale, scale]
            zero_point_values = zero_point if zero_point is None else [zero_point, zero_point, zero_point]
            output = tpul.concat([x, y], scale_values, zero_point_values, axis=1, out_name=None, dtype=dtype)
            self.compile_and_check(self.unique_name(case_name), [x, y], [output], is_quantized=is_quantized)

        _test_concat([[1, 32, 28, 28], [1, 2, 28, 28]])
        _test_concat([[1, 32, 28, 28], [1, 2, 28, 28]], scale=3.0,zero_point=0, dtype="int8", is_quantized=True)
        _test_concat([[1, 32, 28, 28], [1, 2, 28, 28]], dtype="float16", is_quantized=True)
        _test_concat([[1, 32, 28, 28], [1, 2, 28, 28]], dtype="int16")

    def test_Concat2(self, case_name):
        """test concat for int8 with different scales and zerp points"""
        @tpulang(self.chip)
        def _test_concat(shapes: List[List[int]], scales, zero_points, asymmetric, dtype):
            dtype="int8"
            input0 = rand_data(shapes[0], dtype)
            input1 = rand_data(shapes[1], dtype)
            x = tpul.Tensor(dtype=dtype, shape=shapes[0], data=input0, scale=scales[0], zero_point=zero_points[0] if asymmetric else None)
            y = tpul.Tensor(dtype=dtype, shape=shapes[1], data=input1, scale=scales[1], zero_point=zero_points[1] if asymmetric else None)
            output = tpul.concat([x, y], scales, zero_points, axis=1, out_name=None, dtype=dtype)
            self.compile_and_check(self.unique_name(case_name), [x, y], [output], is_quantized=True, asymmetric=asymmetric)

        _test_concat([[1, 12, 28, 28], [1, 3, 28, 28]], [2.0, 3.0, 2.4], [-1, 1, 2], False, dtype="int8")
        _test_concat([[1, 12, 28, 28], [1, 3, 28, 28]], [2.0, 3.0, 2.4], [-1, 1, 2], True, dtype="int8")

    #######################################################################
    # broadcast
    # ------------
    def broadcast_op(self, input_0, input_1):
        broadcast = tpul.broadcast(input_0, input_1)
        return broadcast

    def test_Broadcast(self, case_name):
        """broadcast"""

        @tpulang(self.chip)
        def _test_broadcast(shape_x: List[int], shape_y: List[int], scale=None, zero_point=None, dtype="float32", is_quantized=False):
            input_x = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input_x, scale=scale, zero_point=zero_point)
            broadcast = self.broadcast_op(x, shape_y)
            self.compile_and_check(self.unique_name(case_name), [x], [broadcast], is_quantized=is_quantized)

        _test_broadcast([3, 1], [1, 4])
        _test_broadcast([1, 2, 1], [2, 1, 6])
        _test_broadcast([1, 2, 1], [2, 1, 6], scale=3.0, dtype="int8", is_quantized=True)
        _test_broadcast([1, 2, 1], [2, 1, 6], dtype="float16", is_quantized=True)
        _test_broadcast([1, 2, 1], [2, 1, 6], dtype="int16")

    #######################################################################
    # nonzero
    # ------------
    def nonzero_op(self, input, dtype="float32"):
        nonzero = tpul.nonzero(input, dtype = dtype)
        return nonzero

    def test_Nonzero(self, case_name):
        """nonzero"""

        @tpulang(self.chip)
        def _test_nonzero(shape: List[int], dtype="float32", is_quantized=False):
            input_x = rand_data(shape, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape, data=input_x)
            nonzero = self.nonzero_op(x, dtype=dtype)
            self.compile_and_check(self.unique_name(case_name), [x], [nonzero], is_quantized=is_quantized)

        _test_nonzero([10, 40, 224])
        _test_nonzero([10, 40, 224], dtype="float16", is_quantized=True)

    #######################################################################
    # upsample
    # ------------
    def upsample_op(self, input):
        upsample = tpul.upsample(input)
        return upsample

    def test_Upsample(self, case_name):
        """upsample"""

        @tpulang(self.chip)
        def _test_upsample(shape: List[int], dtype="float32", is_quantized=False):
            input_x = rand_data(shape, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape, data=input_x)
            upsample = self.upsample_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [upsample], is_quantized=is_quantized)

        _test_upsample([1, 3, 28, 28])
        _test_upsample([1, 3, 28, 28], dtype="float16", is_quantized=True)
        _test_upsample([1, 3, 28, 28], dtype="int8", is_quantized=True)

    #######################################################################
    # reduce
    # ------------
    def reduce_op(self, input):
        reduce = tpul.reduce(input)
        return reduce

    def test_Reduce(self, case_name):
        """reduce"""

        @tpulang(self.chip)
        def _test_reduce(shape: List[int], dtype="float32", is_quantized=False):
            input_x = rand_data(shape, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape, data=input_x)
            reduce = self.reduce_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [reduce], is_quantized=is_quantized)

        _test_reduce([1, 3, 28, 28])
        _test_reduce([1, 3, 28, 28], dtype="float16", is_quantized=True)

    #######################################################################
    # unsqueeze
    # ------------
    def unsqueeze_op(self, input):
        unsqueeze = tpul.unsqueeze(input)
        return unsqueeze

    def test_Unsqueeze(self, case_name):
        """unsqueeze"""

        @tpulang(self.chip)
        def _test_unsqueeze(shape: List[int], scale=None, zero_point=None, dtype="float32", is_quantized=False):
            input_x = rand_data(shape, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape, data=input_x, scale=scale, zero_point=zero_point)
            unsqueeze = self.unsqueeze_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [unsqueeze], is_quantized=is_quantized)

        _test_unsqueeze([1, 3, 28, 28])
        _test_unsqueeze([1, 3, 28, 28], dtype="float16", is_quantized=True)
        _test_unsqueeze([1, 3, 28, 28], scale=3.0, dtype="int8", is_quantized=True)

    #######################################################################
    # yuv2rgb
    # ------------
    def yuv2rgb_op(
        self,
        inputs,
        src_format: int,
        dst_format: int,
        ImageOutFormatAttr: str,
        formula_mode: str,
        round_mode: str,
    ):
        yuv2rgb = tpul.yuv2rgb(
            inputs,
            src_format,
            dst_format,
            ImageOutFormatAttr,
            formula_mode,
            round_mode,
        )
        return yuv2rgb

    def test_Yuv2rgb(self, case_name):
        """yuv2rgb"""

        def YCrCb2RGB_601_limited_f32(y, u, v):
            Y = y - 16
            U = u - 128
            V = v - 128

            r_fp = 1.16438 * Y + 1.59603 * V
            g_fp = 1.16438 * Y - 0.39176 * U - 0.81297 * V
            b_fp = 1.16438 * Y + 2.01723 * U

            r_fp = np.clip(r_fp, 0.0, 255.0)
            g_fp = np.clip(g_fp, 0.0, 255.0)
            b_fp = np.clip(b_fp, 0.0, 255.0)

            return r_fp, g_fp, b_fp

        def YCrCb2RGB_601_full_f32(y, u, v):
            Y = y
            U = u - 128
            V = v - 128

            r_fp = Y + 1.40188 * V
            g_fp = Y - 0.34581 * U - 0.71490 * V
            b_fp = Y + 1.77098 * U

            r_fp = np.clip(r_fp, 0.0, 255.0)
            g_fp = np.clip(g_fp, 0.0, 255.0)
            b_fp = np.clip(b_fp, 0.0, 255.0)

            return r_fp, g_fp, b_fp

        def YCrCb2RGB_601_limited_u8(y, u, v):
            r_fp, g_fp, b_fp = YCrCb2RGB_601_limited_f32(y, u, v)
            r = np.uint8(r_fp)
            g = np.uint8(g_fp)
            b = np.uint8(b_fp)
            return r, g, b

        def YCrCb2RGB_601_full_u8(y, u, v):
            r_fp, g_fp, b_fp = YCrCb2RGB_601_full_f32(y, u, v)
            r = np.uint8(r_fp)
            g = np.uint8(g_fp)
            b = np.uint8(b_fp)
            return r, g, b

        def np_yuv2rgb(YUV, height, width, mode="yu12", out_mode="bgr"):
            bgr_data = np.zeros((3, height, width), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    if mode == "yu12":
                        y = YUV[i, j]
                        u = YUV[
                            (int((i // 2) * width / 2) + j // 2) // width + height,
                            (int((i // 2) * width / 2) + j // 2) % width,
                        ]
                        v = YUV[
                            (int((i // 2) * width / 2) + j // 2) // width
                            + int(height * 5 / 4),
                            (int((i // 2) * width / 2) + j // 2) % width,
                        ]
                    elif mode == "yv12":
                        y = YUV[i, j]
                        v = YUV[
                            (int((i // 2) * width / 2) + j // 2) // width + height,
                            (int((i // 2) * width / 2) + j // 2) % width,
                        ]
                        u = YUV[
                            (int((i // 2) * width / 2) + j // 2) // width
                            + int(height * 5 / 4),
                            (int((i // 2) * width / 2) + j // 2) % width,
                        ]
                    elif mode == "nv12":
                        y = YUV[i, j]
                        u = YUV[
                            i // 2 + height,
                            (j // 2) * 2,
                        ]
                        v = YUV[
                            i // 2 + height,
                            (j // 2) * 2 + 1,
                        ]
                    elif mode == "nv21":
                        y = YUV[i, j]
                        v = YUV[
                            i // 2 + height,
                            (j // 2) * 2,
                        ]
                        u = YUV[
                            i // 2 + height,
                            (j // 2) * 2 + 1,
                        ]
                    r, g, b = YCrCb2RGB_601_full_u8(y, u, v)
                    if out_mode == "bgr":
                        bgr_data[0, i, j] = b
                        bgr_data[1, i, j] = g
                        bgr_data[2, i, j] = r
                    elif out_mode == "rgb":
                        bgr_data[0, i, j] = r
                        bgr_data[1, i, j] = g
                        bgr_data[2, i, j] = b
            return bgr_data

        @tpulang(self.chip)
        def _test_yuv2rgb(shape_yuv, scale=None, zero_point=None, dtype="uint8"):
            input_yuv = rand_data(shape_yuv, dtype)

            # generate python calculation
            yuv2rgb_python = {}
            in_mode = "nv12"
            out_mode = "bgr"
            for n in range(shape_yuv[0]):
                bgr_data = np_yuv2rgb(
                    input_yuv[n, :, :],
                    int(shape_yuv[-2] / 3 * 2),
                    shape_yuv[-1],
                    mode=in_mode,
                    out_mode=out_mode,
                )  # numpy
                yuv2rgb_python[f"Yuv2rgb_{n}"] = bgr_data
            np.savez(
                os.path.join(os.getcwd(), f"yuv2rgb_{in_mode}_{out_mode}"),
                **yuv2rgb_python,
            )

            yuv = tpul.Tensor(
                dtype=dtype,
                shape=shape_yuv,
                data=input_yuv,
                scale=scale,
                zero_point=zero_point,
            )

            yuv2rgb = self.yuv2rgb_op(
                yuv,
                2,  # nv12
                5,  # bgr
                "UINT8",  # output as fixed num
                "_601_full",
                "HalfAwayFromZero",
            )
            self.compile_and_check(
                self.unique_name(case_name), [yuv], [yuv2rgb], is_quantized=True
            )

        _test_yuv2rgb([3, 90, 60])

    #######################################################################
    # groupnorm
    # ------------
    def group_norm_op(self, input):
        group_norm = tpul.group_norm(input)
        return group_norm

    def test_Group_norm(self, case_name):
        """group_norm"""

        @tpulang(self.chip)
        def _test_model_def(in_shape, dtype='float32',is_quantized=False):
            x_data = rand_data(in_shape, dtype, -10, 10)
            x = tpul.Tensor(dtype=dtype, shape=in_shape, data=x_data)
            out = self.group_norm_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [out], is_quantized=is_quantized)

        _test_model_def([1, 3, 224, 224], dtype='float16',is_quantized=True)
        _test_model_def([1, 3, 224, 224], dtype='float32')
    #######################################################################
    # LayerNorm
    # ------------
    def test_LayerNorm(self, case_name):
        @tpulang(self.chip)
        def _test_model_def(in_shape, dtype='float32', axis=-1,is_quantized=False, reshape_after_layernorm=False, test_reshape_down=False):
            x_data = rand_data(in_shape, dtype, -10, 10)
            norm_shape = [1]
            for tmp_shape in in_shape[axis:]:
                norm_shape[0] *= tmp_shape
            x = tpul.Tensor(dtype=dtype, shape=in_shape, data=x_data)
            gamma = self.coeff_tensor(shape=norm_shape, dtype=dtype)
            beta = self.coeff_tensor(shape=norm_shape, dtype=dtype)
            if test_reshape_down:
                in_tensor = tpul.Tensor(dtype="int16", shape=in_shape, data=x_data)
                if reshape_after_layernorm:
                    #refs
                    cast_f16 = tpul.cast(in_tensor, "float16")
                    out = tpul.layer_norm(cast_f16, gamma, beta, axis=axis)
                    out_F32 = tpul.cast(out, "float32")
                    out = tpul.reshape(out_F32, [1, 8, 7, 7, 768])
                else:
                    #can check if reshape_down pattern work
                    #reshape -> cast -> layernorm -> cast ==> cast -> layernorm -> cast -> reshape
                    reshape_out = tpul.reshape(x, [1, 8, 7, 7, 768])
                    cast_f16 = tpul.cast(reshape_out, "float16")
                    out = tpul.layer_norm(cast_f16, gamma, beta, axis=4)
                    out = tpul.cast(out, "float32")
                    cast_f16 = tpul.cast(x, "float16")
            else:
                out = tpul.layer_norm(x, gamma, beta, axis=axis)
            self.compile_and_check(self.unique_name(case_name), [x], [out], is_quantized=is_quantized)

        _test_model_def([20, 5, 10, 10, 10], dtype='float16', axis=2, is_quantized=True)
        _test_model_def([20, 5, 10, 10, 10], dtype='float32', axis=2)
        _test_model_def([1, 392, 768], dtype='float16', axis=2, is_quantized=True, reshape_after_layernorm=False)
        _test_model_def([1, 392, 768], dtype='float16', axis=2, is_quantized=True, reshape_after_layernorm=True)

    #######################################################################
    # BatchNorm
    # ------------
    def test_BatchNorm(self, case_name):
        @tpulang(self.chip)
        def _test_model_def(in_shape, dtype=None,axis =-1,is_quantized=False):
            x_data = rand_data(in_shape, dtype, -10, 10)
            x = tpul.Tensor(dtype=dtype, shape=in_shape, data=x_data)
            if len(in_shape) == 1:
                oc = in_shape[0]
            else:
                oc = in_shape[1]
            mean = self.coeff_tensor([oc], x.dtype, scale=0.2)
            var = self.coeff_tensor([oc], x.dtype, data=np.clip(np.random.randn(oc), 0.5, 10).astype(x.dtype))
            gamma = self.coeff_tensor([oc], x.dtype, data=np.ones((oc)).astype(x.dtype))
            beta = self.coeff_tensor([oc], x.dtype, data=np.zeros((oc)).astype(x.dtype))
            y = tpul.batch_norm(x, mean, var, epsilon=1e-5)
            self.compile_and_check(self.unique_name(case_name), [x], [y],is_quantized=is_quantized)
        _test_model_def([1, 3, 224, 224],dtype="float32")
        _test_model_def([1, 3, 224, 224],dtype="float16",is_quantized=True)

    #######################################################################
    # rmsnorm
    # ------------
    def test_RMSNorm(self, case_name):
        """rms_norm"""

        @tpulang(self.chip)
        def _test_model_def(in_shape, dtype='float32',is_quantized=False):
            x_data = rand_data(in_shape, dtype, -10, 10)
            norm_shape = [in_shape[-1]]
            x = tpul.Tensor(dtype=dtype, shape=in_shape, data=x_data)
            gamma = self.coeff_tensor(shape=norm_shape, dtype=dtype)

            out = tpul.rms_norm(x, gamma)
            self.compile_and_check(self.unique_name(case_name), [x], [out], is_quantized=is_quantized)

        _test_model_def([1, 3, 224, 224])
        _test_model_def([1, 3, 224, 224],dtype="float16")
        _test_model_def([1, 3, 224, 224],dtype="float16",is_quantized=True)

    #######################################################################
    # normalize
    # ------------
    def test_normalize(self, case_name):
        """normalize"""

        @tpulang(self.chip)
        def _test_model_def(in_shape, p = 2.0, axes = 1, dtype='float32', is_quantized=False):
            x_data = rand_data(in_shape, dtype, -10, 10, seed=10)
            x = tpul.Tensor(dtype=dtype, shape=in_shape, data=x_data)

            out = tpul.normalize(x, p, axes)
            self.compile_and_check(self.unique_name(case_name), [x], [out], is_quantized=is_quantized)

        # float16
        _test_model_def([1, 3, 224, 224], dtype='float16')
        _test_model_def([1, 3, 224, 224],p=3.0, axes=[1,2], dtype='float16')

    ######################################################################
    # Split
    # ------------
    def split_op(self, input, axis=0, num=1, size=None):
        split = tpul.split(input, axis=axis, num=num, size=size)
        return split

    def test_Split(self, case_name):
        """split"""

        @tpulang(self.chip)
        def _test_split(shape_x: List[int], axis=0, num=1, size=None, scale=None, zero_point=None, dtype="float32", is_quantized=False):
            input = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=input, scale=scale, zero_point=zero_point)
            splits = self.split_op(x, axis, num, size)
            self.compile_and_check(self.unique_name(case_name), [x], splits, is_quantized=is_quantized)

        _test_split([1, 32, 28, 28], axis=1, num=2, size=[10, 22])
        _test_split([1, 32, 28, 28], axis=1, num=2, size=[10, 22], scale=3.0, dtype="int8", is_quantized=True)
        _test_split([1, 32, 28, 28], axis=1, num=2, size=[10, 22], dtype="float16", is_quantized=True)
        _test_split([1, 32, 28, 28], axis=1, num=2, size=[10, 22], dtype="int16")

    #######################################################################
    # Repeat
    # ------------
    def repeat_op(self, input):
        reps = self.coeff_tensor(len(input.shape), "float32", data=np.random.randint(low=1, high=3,size=len(input.shape)).astype(np.float32))
        repeat = tpul.repeat(input, reps)
        return repeat

    def test_Repeat(self, case_name):
        """repeat"""

        @tpulang(self.chip)
        def _test_repeat(shape_x: List[int], scale=None, zero_point=None, dtype="float32", is_quantized=False):
            x_data = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data, scale=scale, zero_point=zero_point)
            repeat = self.repeat_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [repeat], is_quantized=is_quantized)

        _test_repeat([1, 3, 28, 28])
        _test_repeat([1, 3, 28, 28], dtype="float16", is_quantized=True)
        _test_repeat([1, 3, 28, 28], scale=3.0, dtype="int8", is_quantized=True)
        _test_repeat([1, 3, 28, 28], dtype="int16")

    #######################################################################
    # Gt
    # ------------
    def test_base_compare_quant(self, case_name, func, shape_x: List[int], shape_y: List[int], scale=None, dtype="int8"):
        @tpulang(self.chip)
        def binary_coeff():
            x_data = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            y = self.coeff_tensor(shape_y, dtype, scale=4.0)
            out = func(y, x, scale=scale)
            self.compile_and_check(self.unique_name(case_name), [x], [out], is_quantized=True)
        @tpulang(self.chip)
        def binary():
            x_data = rand_data(shape_x, dtype)
            y_data = rand_data(shape_y, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            y = tpul.Tensor(dtype=dtype, shape=shape_y, data=y_data)
            out = func(x, y, scale=scale)
            self.compile_and_check(self.unique_name(case_name), [x, y], [out], is_quantized=True)

        binary_coeff()
        binary()

    def gt_op(self, input_0, input_1):
        gt = tpul.gt(input_0, input_1)
        return gt

    def test_Gt(self, case_name):
        """Gt"""

        @tpulang(self.chip)
        def _test_gt(shape_x: List[int], shape_y: List[int], dtype="float32"):
            x_data = rand_data(shape_x, dtype)
            y_data = rand_data(shape_y, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            y = tpul.Tensor(dtype=dtype, shape=shape_y, data=y_data)
            gt = self.gt_op(x, y)
            self.compile_and_check(self.unique_name(case_name), [x, y], [gt])

        _test_gt([1, 3, 28, 28], [1, 3, 28, 28])
        _test_gt([1, 3, 32, 32], [1, 3, 32, 32])
        self.test_base_compare_quant(case_name, tpul.gt, [1, 3, 32, 32], [1, 3, 32, 32], [4.0, 4.0, 4.0], "int8")
        self.test_base_compare_quant(case_name, tpul.gt, [1, 3, 32, 32], [1, 3, 32, 32], dtype="float16")

    #######################################################################
    # Lt
    # ------------
    def lt_op(self, input_0, input_1):
        lt = tpul.lt(input_0, input_1)
        return lt

    def test_Lt(self, case_name):
        """Lt"""

        @tpulang(self.chip)
        def _test_lt(shape_x: List[int], shape_y: List[int], dtype="float32"):
            x_data = rand_data(shape_x, dtype)
            y_data = rand_data(shape_y, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            y = tpul.Tensor(dtype=dtype, shape=shape_y, data=y_data)
            lt = self.lt_op(x, y)
            self.compile_and_check(self.unique_name(case_name), [x, y], [lt])

        _test_lt([1, 3, 28, 28], [1, 3, 28, 28])
        _test_lt([1, 3, 32, 32], [1, 3, 32, 32])
        self.test_base_compare_quant(case_name, tpul.lt, [1, 3, 32, 32], [1, 3, 32, 32], [4.0, 4.0, 4.0], "int8")
        self.test_base_compare_quant(case_name, tpul.lt, [1, 3, 32, 32], [1, 3, 32, 32], dtype="float16")

    #######################################################################
    # Ge
    # ------------
    def ge_op(self, input_0, input_1):
        ge = tpul.ge(input_0, input_1)
        return ge

    def test_Ge(self, case_name):
        """Ge"""

        @tpulang(self.chip)
        def _test_ge(shape_x: List[int], shape_y: List[int], dtype="float32"):
            x_data = rand_data(shape_x, dtype)
            y_data = rand_data(shape_y, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            y = tpul.Tensor(dtype=dtype, shape=shape_y, data=y_data)
            ge = self.ge_op(x, y)
            self.compile_and_check(self.unique_name(case_name), [x, y], [ge])

        _test_ge([1, 3, 28, 28], [1, 3, 28, 28])
        _test_ge([1, 3, 32, 32], [1, 3, 32, 32])
        self.test_base_compare_quant(case_name, tpul.ge, [1, 3, 32, 32], [1, 3, 32, 32], [4.0, 4.0, 4.0], "int8")
        self.test_base_compare_quant(case_name, tpul.ge, [1, 3, 32, 32], [1, 3, 32, 32], dtype="float16")

    #######################################################################
    # Le
    # ------------
    def le_op(self, input_0, input_1):
        le = tpul.le(input_0, input_1)
        return le

    def test_Le(self, case_name):
        """Le"""

        @tpulang(self.chip)
        def _test_le(shape_x: List[int], shape_y: List[int], dtype="float32"):
            x_data = rand_data(shape_x, dtype)
            y_data = rand_data(shape_y, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            y = tpul.Tensor(dtype=dtype, shape=shape_y, data=y_data)
            le = self.le_op(x, y)
            self.compile_and_check(self.unique_name(case_name), [x, y], [le])

        _test_le([1, 3, 28, 28], [1, 3, 28, 28])
        _test_le([1, 3, 32, 32], [1, 3, 32, 32])
        self.test_base_compare_quant(case_name, tpul.lt, [1, 3, 32, 32], [1, 3, 32, 32], [4.0, 4.0, 4.0], "int8")
        self.test_base_compare_quant(case_name, tpul.lt, [1, 3, 32, 32], [1, 3, 32, 32], dtype="float16")

    #######################################################################
    # Eq
    # ------------
    def eq_op(self, input_0, input_1):
        eq = tpul.eq(input_0, input_1)
        return eq

    def test_Eq(self, case_name):
        """Eq"""

        @tpulang(self.chip)
        def _test_eq(shape_x: List[int], shape_y: List[int], dtype="float32"):
            x_data = np.random.randint(0, 256, size=shape_x).astype(np.float32)
            y_data = np.random.randint(0, 256, size=shape_x).astype(np.float32)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            y = tpul.Tensor(dtype=dtype, shape=shape_y, data=y_data)
            eq = self.eq_op(x, y)
            self.compile_and_check(self.unique_name(case_name), [x, y], [eq])

        _test_eq([1, 3, 28, 28], [1, 3, 28, 28])
        _test_eq([1, 3, 32, 32], [1, 3, 32, 32])
        self.test_base_compare_quant(case_name, tpul.eq, [1, 3, 32, 32], [1, 3, 32, 32], [4.0, 4.0, 4.0], "int8")
        self.test_base_compare_quant(case_name, tpul.eq, [1, 3, 32, 32], [1, 3, 32, 32], dtype="float16")

    #######################################################################
    # Ne
    # ------------
    def ne_op(self, input_0, input_1):
        ne = tpul.ne(input_0, input_1)
        return ne

    def test_Ne(self, case_name):
        """Ne"""

        @tpulang(self.chip)
        def _test_ne(shape_x: List[int], shape_y: List[int], dtype="float32"):
            x_data = rand_data(shape_x, dtype)
            y_data = rand_data(shape_y, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            y = tpul.Tensor(dtype=dtype, shape=shape_y, data=y_data)
            ne = self.ne_op(x, y)
            self.compile_and_check(self.unique_name(case_name), [x, y], [ne])

        _test_ne([1, 3, 28, 28], [1, 3, 28, 28])
        _test_ne([1, 3, 32, 32], [1, 3, 32, 32])
        self.test_base_compare_quant(case_name, tpul.ne, [1, 3, 32, 32], [1, 3, 32, 32], [4.0, 4.0, 4.0], "int8")
        self.test_base_compare_quant(case_name, tpul.ne, [1, 3, 32, 32], [1, 3, 32, 32], dtype="float16")

    #######################################################################
    # Gts
    # ------------
    def test_compare_scalar_quant(self, case_name, func, shape_x: List[int], scale=None, dtype="int8"):
        @tpulang(self.chip)
        def binary_scalar():
            x_data = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            out = func(x, tpul.Scalar(2, "float32"), scale=scale)
            self.compile_and_check(self.unique_name(case_name), [x], [out], is_quantized=True)

        binary_scalar()

    def gts_op(self, input_0):
        gts = tpul.gts(input_0, 0)
        return gts

    def test_Gts(self, case_name):
        """Gts"""

        @tpulang(self.chip)
        def _test_gts(shape_x: List[int], dtype="float32"):
            x_data = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            gts = self.gts_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [gts])

        _test_gts([1, 3, 32, 32])
        self.test_compare_scalar_quant(case_name, tpul.gts, [1, 3, 32, 32], [4.0, 4.0], "int8")
        self.test_compare_scalar_quant(case_name, tpul.gts, [1, 3, 32, 32], dtype="float16")

    #######################################################################
    # Lts
    # ------------
    def lts_op(self, input_0):
        lts = tpul.lts(input_0, 0)
        return lts

    def test_Lts(self, case_name):
        """Lts"""

        @tpulang(self.chip)
        def _test_lts(shape_x: List[int], dtype="float32"):
            x_data = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            lts = self.lts_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [lts])

        _test_lts([1, 3, 28, 28])
        _test_lts([1, 3, 32, 32])
        self.test_compare_scalar_quant(case_name, tpul.lts, [1, 3, 32, 32], [4.0, 4.0], "int8")
        self.test_compare_scalar_quant(case_name, tpul.lts, [1, 3, 32, 32], dtype="float16")

    #######################################################################
    # Ges
    # ------------
    def ges_op(self, input_0):
        ges = tpul.ges(input_0, 0)
        return ges

    def test_Ges(self, case_name):
        """Ges"""

        @tpulang(self.chip)
        def _test_ges(shape_x: List[int], dtype="float32"):
            x_data = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            ges = self.ges_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [ges])

        _test_ges([1, 3, 28, 28])
        _test_ges([1, 3, 32, 32])
        self.test_compare_scalar_quant(case_name, tpul.ges, [1, 3, 32, 32], [4.0, 4.0], "int8")
        self.test_compare_scalar_quant(case_name, tpul.ges, [1, 3, 32, 32], dtype="float16")

    #######################################################################
    # Les
    # ------------
    def les_op(self, input_0):
        les = tpul.les(input_0, 0)
        return les

    def test_Les(self, case_name):
        """Les"""

        @tpulang(self.chip)
        def _test_les(shape_x: List[int], dtype="float32"):
            x_data = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            les = self.les_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [les])

        _test_les([1, 3, 28, 28])
        _test_les([1, 3, 32, 32])
        self.test_compare_scalar_quant(case_name, tpul.les, [1, 3, 32, 32], [4.0, 4.0], "int8")
        self.test_compare_scalar_quant(case_name, tpul.les, [1, 3, 32, 32], dtype="float16")

    #######################################################################
    # Eqs
    # ------------
    def eqs_op(self, input_0):
        eqs = tpul.eqs(input_0, 0)
        return eqs

    def test_Eqs(self, case_name):
        """Eqs"""

        @tpulang(self.chip)
        def _test_eqs(shape_x: List[int], dtype="float32"):
            x_data = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            eqs = self.eqs_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [eqs])

        _test_eqs([1, 3, 28, 28])
        _test_eqs([1, 3, 32, 32])
        self.test_compare_scalar_quant(case_name, tpul.eqs, [1, 3, 32, 32], [4.0, 4.0], "int8")
        self.test_compare_scalar_quant(case_name, tpul.eqs, [1, 3, 32, 32], dtype="float16")

    #######################################################################
    # Nes
    # ------------
    def nes_op(self, input_0):
        nes = tpul.nes(input_0, 0)
        return nes

    def test_Nes(self, case_name):
        """Nes"""

        @tpulang(self.chip)
        def _test_nes(shape_x: List[int], dtype="float32"):
            x_data = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            nes = self.nes_op(x)
            self.compile_and_check(self.unique_name(case_name), [x], [nes])

        _test_nes([1, 3, 28, 28])
        _test_nes([1, 3, 32, 32])
        self.test_compare_scalar_quant(case_name, tpul.nes, [1, 3, 32, 32], [4.0, 4.0], "int8")
        self.test_compare_scalar_quant(case_name, tpul.nes, [1, 3, 32, 32], dtype="float16")

    #######################################################################
    # Resnet50 like model case
    # ------------
    def test_ResnetBlock(self, case_name):
        def conv_op(x,
                    kshape,
                    stride,
                    pad=None,
                    group=1,
                    dilation=[1, 1],
                    bias=False,
                    dtype="float32"):
            oc = kshape[0]
            weight = self.coeff_tensor(kshape, dtype, scale=1/(math.sqrt(kshape[1] * kshape[2] * kshape[3])))
            bias = self.coeff_tensor(oc, dtype) if bias else None
            conv = tpul.conv(x,
                            weight,
                            bias=bias,
                            stride=stride,
                            pad=pad,
                            dilation=dilation,
                            group=group)
            return conv

        # 1. define model
        def model_def(x):
            maxpool1 = tpul.maxpool2d(x, kernel=[3, 3], stride=[2, 2], pad=[1, 1, 1, 1])
            conv2 = conv_op(maxpool1, kshape=[64, 64, 1, 1], stride=[1, 1], dtype='float32')
            relu3 = tpul.relu(conv2)
            conv4 = conv_op(relu3, kshape=[64, 64, 3, 3], stride=[1, 1], pad=[1, 1, 1, 1], dtype='float32')
            relu5 =  tpul.relu(conv4)
            conv6 = conv_op(relu5, kshape=[256, 64, 1, 1], stride=[1, 1], dtype='float32')
            conv7 =  conv_op(maxpool1, kshape=[256, 64, 1, 1], stride=[1, 1], dtype='float32')
            add8 = tpul.add(conv6, conv7)
            relu9 = tpul.relu(add8)
            return relu9

        def _test_resnet_block(in_shape):
            # 2. prepare input
            x_data = rand_data(in_shape, 'float32')
            x = tpul.Tensor(dtype='float32', shape=in_shape, data=x_data)
            # 3. init and compile tpulang model to top.mlir
            tpul.init(device=self.chip.upper())
            out = model_def(x)
            tpul.compile_f32(case_name, [x], [out])
            tpul.deinit()
            # tpul.compile will do top mlir inference with random input
            in_f32_npz = case_name + '_in_f32.npz'
            top_out = case_name + '_top_outputs.npz'
            # 4. deploy to bmodel
            deploy_cmd_base = f"model_deploy.py --mlir {case_name}.mlir "
            deploy_cmd_base += "--chip {} ".format(self.chip)
            deploy_cmd_base += "--test_input {} ".format(in_f32_npz)
            deploy_cmd_base += "--test_reference {} ".format(top_out)
            # deploy to [f32, f16, bf16] quant mode
            for mode in self.quant_modes:
                bmodel_name = "{}.bmodel".format(case_name + "_" + self.chip + "_" + mode)
                deploy_cmd = deploy_cmd_base
                deploy_cmd += "--model {} ".format(bmodel_name)
                deploy_cmd += "--quantize {} " .format(mode.upper())
                assert(os.system(deploy_cmd) == 0)

        _test_resnet_block([1, 64, 112, 112])

    #######################################################################
    # Mobilenet like model case
    # ------------
    def test_MobilenetBlock(self, case_name):
        def conv_op(x,
                    kshape,
                    stride,
                    pad=None,
                    group=1,
                    dilation=[1, 1],
                    bias=False,
                    dtype="float32"):
            oc = kshape[0]
            weight = self.coeff_tensor(kshape, dtype, scale=1/(math.sqrt(kshape[1] * kshape[2] * kshape[3])))
            bias = self.coeff_tensor(oc, dtype) if bias else None
            conv = tpul.conv(x,
                            weight,
                            bias=bias,
                            stride=stride,
                            pad=pad,
                            dilation=dilation,
                            group=group)
            return conv

        # 1. model define
        class Model2():
            def __init__(self):
                super(Model2, self).__init__()
            def forward(self, input):
                conv0 = conv_op(input, kshape=[32, 1, 5, 5], stride=[1,1], pad=[2, 2, 2, 2], dtype='float32')
                relu1 = tpul.relu(conv0)
                maxpool2 = tpul.maxpool2d(relu1, kernel=[2, 2], stride=[2, 2], pad=[0, 0, 0, 0])
                conv3 = conv_op(maxpool2, kshape=[64, 32, 5, 5], stride=[1,1], pad=[2, 2, 2, 2], dtype='float32')
                relu4 =  tpul.relu(conv3)
                maxpool5 = tpul.maxpool2d(relu4, kernel=[2, 2], stride=[2, 2], pad=[0, 0, 0, 0])
                conv6 = conv_op(maxpool5, kshape=[1024, 64, 7, 7], stride=[1,1], dtype='float32')
                relu7 =  tpul.relu(conv6)
                softmax8 = tpul.softmax(relu7, axis=1)
                tpul.compile_f32(case_name, [input], [softmax8])

        def _test_mobilenet_block(in_shape):
            # 2. prepare input
            x_data = rand_data(in_shape, 'float32')
            x = tpul.Tensor(dtype='float32', shape=in_shape, data=x_data)
            # 3. init and compile model to top.mlir
            tpul.init(device=self.chip.upper())
            model2 = Model2()
            model2.forward(x)
            # tpul.compile will do top mlir inference with random input
            in_f32_npz = case_name + '_in_f32.npz'
            top_out = case_name + '_top_outputs.npz'
            # 4. deploy to bmodel
            deploy_cmd_base = f"model_deploy.py --mlir {case_name}.mlir "
            deploy_cmd_base += "--chip {} ".format(self.chip)
            deploy_cmd_base += "--test_input {} ".format(in_f32_npz)
            deploy_cmd_base += "--test_reference {} ".format(top_out)
            for mode in self.quant_modes:
                bmodel_name = "{}.bmodel".format(case_name + "_" + self.chip + "_" + mode)
                deploy_cmd = deploy_cmd_base
                deploy_cmd += "--model {} ".format(bmodel_name)
                deploy_cmd += "--quantize {} " .format(mode.upper())
                assert(os.system(deploy_cmd) == 0)

        _test_mobilenet_block([1, 1, 28, 28])

    #######################################################################
    # TopK
    # ------------
    def test_TopK(self, case_name):
        """TopK"""

        @tpulang(self.chip)
        def _test_TopK(shape_x: List[int], k, dtype="float32"):
            x_data = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            y_ind, y_val = tpul.topk(x, 1, k)
            self.compile_and_check(self.unique_name(case_name), [x], [y_ind, y_val])

        _test_TopK([2, 16], 2)

    #######################################################################
    # NMS
    # ------------
    def gen_rand_boxes(self, num_batch, num_box, H, W, box_format):
        num = num_batch * num_box
        roi_xl = np.random.rand(num).astype(np.float32) * (W - 1)
        roi_xh = np.random.rand(num).astype(np.float32) * (W - 1)
        roi_yl = np.random.rand(num).astype(np.float32) * (H - 1)
        roi_yh = np.random.rand(num).astype(np.float32) * (H - 1)
        for i in range(num):
            if roi_xl[i] > roi_xh[i]:
                roi_xl[i], roi_xh[i] = roi_xh[i], roi_xl[i]
            elif roi_xl[i] == roi_xh[i]:
                roi_xl[i] = 0
                roi_xh[i] = W - 1
                roi_yl[i] = 0
                roi_yh[i] = H - 1
            if roi_yl[i] > roi_yh[i]:
                roi_yl[i], roi_yh[i] = roi_yh[i], roi_yl[i]
        roi_xl = np.reshape(roi_xl, [num_batch, num_box, 1])
        roi_xh = np.reshape(roi_xh, [num_batch, num_box, 1])
        roi_yl = np.reshape(roi_yl, [num_batch, num_box, 1])
        roi_yh = np.reshape(roi_yh, [num_batch, num_box, 1])
        if box_format == 'TENSORFLOW':
            boxes = np.concatenate((roi_yl, roi_xl, roi_yh, roi_xh), 2)
        else:
            center_x = (roi_xl + roi_xh) / 2
            center_y = (roi_yl + roi_yh) / 2
            width = roi_xh - roi_xl
            height = roi_yh - roi_yl
            boxes = np.concatenate((center_x, center_y, width, height), 2)
        return boxes

    def test_NMS(self, case_name):
        """NMS"""

        @tpulang(self.chip)
        def _test_NMS(num_batch, num_boxes, num_classes, box_format, max_box_num, dtype="float32"):
            box_data = self.gen_rand_boxes(num_batch, num_boxes, 256, 367, box_format)
            score_data = np.arange(num_batch * num_classes * num_boxes).reshape([num_batch, num_classes, num_boxes]).astype(np.float32)
            boxes = tpul.Tensor(dtype=dtype, shape=list(box_data.shape), data=box_data)
            scores = tpul.Tensor(dtype=dtype, shape=list(score_data.shape), data=score_data)
            y = tpul.nms(boxes, scores, box_format=box_format, max_box_num_per_class=max_box_num)
            self.compile_and_check(self.unique_name(case_name), [boxes, scores], [y])

        _test_NMS(2, 6, 4, 'PYTORCH', 4)
        _test_NMS(2, 6, 4, 'TENSORFLOW', 4)

    #######################################################################
    # Interp
    # ------------
    def test_Interp(self, case_name):
        """interp"""

        @tpulang(self.chip)
        def _test_interp(shape_x: List[int], method, coord_mode, dtype="float32", is_quantized=False):
            x_data = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data)
            y = tpul.interpolate(x, 2., 3., method=method, coord_mode=coord_mode)
            # y = tpul.requant_int(y, 3000, -12, 0, 2, out_dtype="int8")
            # y = tpul.add_shift(y, 1, 0, out_dtype="int8")
            # y = tpul.dequant_int_to_fp(y, 0.34, 0)
            self.compile_and_check(self.unique_name(case_name), [x], [y], is_quantized=is_quantized)

        _test_interp([2, 3, 24, 28], 'nearest', 'pytorch_half_pixel')
        _test_interp([2, 3, 24, 28], 'nearest', 'align_corners')
        _test_interp([2, 3, 24, 28], 'linear', 'pytorch_half_pixel')
        _test_interp([2, 3, 24, 28], 'linear', 'align_corners')
        _test_interp([2, 3, 24, 28], 'linear', 'align_corners', dtype="float16", is_quantized=True)
        _test_interp([2, 3, 24, 28], 'nearest', 'pytorch_half_pixel', dtype="float16", is_quantized=True)
        _test_interp([2, 3, 24, 28], 'nearest', 'pytorch_half_pixel', dtype="int8", is_quantized=True)

    #######################################################################
    # Lut
    # ------------
    def test_Lut(self, case_name):
        """lut"""

        @tpulang(self.chip)
        def _test_lut(shape_x: List[int]):
            x_dtype = 'int8'
            t_dtype = 'uint8'
            x_data = rand_data(shape_x, x_dtype)
            t_data = rand_data([256], t_dtype)
            x = tpul.Tensor(dtype=x_dtype, shape=shape_x, data=x_data)
            t = tpul.Tensor(dtype=t_dtype, shape=list(t_data.shape), data=t_data, ttype="coeff")
            y = tpul.lut(x, t)
            self.compile_and_check(self.unique_name(case_name), [x], [y], is_quantized=True)

        _test_lut([2, 3, 28, 28])

    #######################################################################
    # Extract
    # ------------
    def test_Extract(self, case_name):
        """extract"""

        @tpulang(self.chip)
        def _test_extract(shape_x: List[int], start, end, stride, scale=None, zero_point=None, dtype="float32", is_quantized=False):
            x_data = rand_data(shape_x, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape_x, data=x_data, scale=scale, zero_point=zero_point)
            y = tpul.extract(x, start, end, stride)
            self.compile_and_check(self.unique_name(case_name), [x], [y], is_quantized=is_quantized)

        _test_extract([2, 3, 24, 28], None, [1, 2, 12, 20], [1, 1, 2, 1])
        _test_extract([2, 3, 24, 28], [0, 0, 9, 0], None, [1, 1, 2, 1])
        _test_extract([2, 3, 24, 28], [0, 0, 9, 0], [1, 2, 12, 20], None)
        _test_extract([2, 3, 24, 28], [0, 0, 9, 0], [1, 2, 12, 20], [1, 1, 2, 1])
        _test_extract([2, 3, 24, 28], [0, 0, 9, 0], [1, 2, 12, 20], [1, 1, 2, 1], dtype="float16", is_quantized=True)
        _test_extract([2, 3, 24, 28], [0, 0, 9, 0], None, [1, 1, 2, 1], scale=3.0, dtype="int8", is_quantized=True)
        _test_extract([2, 3, 24, 28], [0, 0, 9, 0], [1, 2, 12, 20], [1, 1, 2, 1], dtype="int16")

    #######################################################################
    # CondSelect
    # ------------
    def test_CondSelect(self, case_name):
        """Cond Select"""

        @tpulang(self.chip)
        def _test_condselect(shape: List[int], flag: int, dtype="float32"):
            inputs = []
            cond_data = rand_data(shape, dtype)
            cond = tpul.Tensor(dtype=dtype, shape=shape, data=cond_data)
            inputs.append(cond)
            if flag & 0x1:
                tbrn_data = rand_data(shape, dtype)
                tbrn = tpul.Tensor(dtype=dtype, shape=shape, data=tbrn_data)
                inputs.append(tbrn)
            else:
                tbrn = tpul.Scalar(dtype=dtype, value=-1.2)
            if flag & 0x10:
                fbrn_data = rand_data(shape, dtype)
                fbrn = tpul.Tensor(dtype=dtype, shape=shape, data=fbrn_data)
                inputs.append(fbrn)
            else:
                fbrn = tpul.Scalar(dtype=dtype, value=+1.2)
            y = tpul.cond_select(cond, tbrn, fbrn)
            self.compile_and_check(self.unique_name(case_name), inputs, [y])

        for flag in (0x00, 0x01, 0x10, 0x11):
            _test_condselect([2, 3, 24, 28], flag)

    #######################################################################
    # Select
    # ------------
    def test_Select(self, case_name):
        """Select"""

        @tpulang(self.chip)
        def _test_select(shape: List[int], type: str, dtype="float32",out_name=None):
            lhs_data = rand_data(shape, dtype)
            lhs = tpul.Tensor(dtype=dtype, shape=shape, data=lhs_data)
            rhs_data = rand_data(shape, dtype)
            rhs = tpul.Tensor(dtype=dtype, shape=shape, data=rhs_data)
            tbrn_data = rand_data(shape, dtype)
            tbrn = tpul.Tensor(dtype=dtype, shape=shape, data=tbrn_data)
            fbrn_data = rand_data(shape, dtype)
            fbrn = tpul.Tensor(dtype=dtype, shape=shape, data=fbrn_data)
            y = tpul.select(lhs, rhs, tbrn, fbrn, type)
            self.compile_and_check(self.unique_name(case_name), [lhs, rhs, tbrn, fbrn], [y], dtype!="float32")

        _test_select([2, 3, 24, 28], "Greater")
        _test_select([2, 3, 24, 28], "Less")

    #######################################################################
    # Sort
    # ------------
    def test_Sort(self, case_name):
        """Sort"""

        @tpulang(self.chip)
        def _test_sort(shape: List[int], axis: int, dtype="float32"):
            x_data = rand_indices(shape, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape, data=x_data)
            val, ind = tpul.sort(x, axis)
            self.compile_and_check(self.unique_name(case_name), [x], [val, ind])

        _test_sort([4, 3, 4, 28], 3)

    #######################################################################
    # ArgSort
    # ------------
    def test_ArgSort(self, case_name):
        """ArgSort"""

        @tpulang(self.chip)
        def _test_argsort(shape: List[int], axis: int, dtype="float32"):
            x_data = rand_indices(shape, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape, data=x_data)
            y = tpul.argsort(x, axis)
            self.compile_and_check(self.unique_name(case_name), [x], [y])

        _test_argsort([4, 3, 4, 28], 3)

    #######################################################################
    # IndexSelect
    # ------------
    def test_IndexSelect(self, case_name):
        """IndexSelect"""

        @tpulang(self.chip)
        def _test_index_select(shape: List[int], axis: int, dtype="float32"):
            ind_data = rand_indices((shape[axis],), dtype)
            ind = tpul.Tensor(dtype=dtype, shape=list(ind_data.shape), data=ind_data)
            x_data = rand_data(shape, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape, data=x_data)
            y= tpul.index_select(x, ind, axis=axis)
            self.compile_and_check(self.unique_name(case_name), [x, ind], [y])

        # _test_index_select([28, 4], 0)
        _test_index_select([4, 11], 1)

    #######################################################################
    # SortByKey
    # ------------
    def test_SortByKey(self, case_name):
        """SortByKey"""

        @tpulang(self.chip)
        def _test_sort_by_key(shape: List[int], axis: int, dtype="float32"):
            key_data = rand_indices((shape[axis],), dtype)
            key = tpul.Tensor(dtype=dtype, shape=list(key_data.shape), data=key_data)
            x_data = rand_data(shape, dtype)
            x = tpul.Tensor(dtype=dtype, shape=shape, data=x_data)
            y, sorted_key = tpul.sort_by_key(x, key, axis=axis)
            self.compile_and_check(self.unique_name(case_name), [x, key], [y, sorted_key])

        # _test_sort_by_key([28, 4], 0)
        _test_sort_by_key([4, 11], 1)

    def test_SelfAttnBlock(self, case_name):
        class SelfAttnBlock():
            def __init__(self, batch, d, head, mq, mk):
                super(SelfAttnBlock, self).__init__()
                self.batch = batch
                self.d = d
                self.head = head
                self.mq = mq
                self.mk = mk

            def forward(self, q, k, v):
                permute_q = tpul.permute(q, [0, 2, 1, 3]) # 1, 10, 1024, 128
                permute_k = tpul.permute(k, [0, 2, 3, 1]) # 1, 10, 128, 1024
                mm0 = tpul.matmul(permute_q, permute_k)
                rsqrt0 = tpul.mul(mm0, tpul.Scalar(1 / np.sqrt(self.d), dtype=mm0.dtype))
                softmax0 = tpul.softmax(rsqrt0, axis=3)
                permute_v = tpul.permute(v, [0, 2, 1, 3])
                mm1 = tpul.matmul(softmax0, permute_v)
                reshape = tpul.reshape(mm1, [self.batch, self.head, self.mq, self.d])
                permute_mm1 = tpul.permute(reshape, [0, 2, 1, 3])
                reshape_mm1 = tpul.reshape(permute_mm1, [self.batch, self.mq, self.head * self.d])

                tpul.compile_f32(case_name, [q, k, v], [reshape_mm1])

        def _test_self_attn_block(batch, d, head, mq, mk):
            # 2. prepare input
            q_data = rand_data([batch, mq, head, d], 'float32')
            q = tpul.Tensor(dtype='float32', shape=[batch, mq, head, d], data=q_data) # 1, 1024, 10, 128
            k_data = rand_data([batch, mk, head, d], 'float32')
            k = tpul.Tensor(dtype='float32', shape=[batch, mq, head, d], data=k_data) # 1, 1024, 10, 128
            v_data = rand_data([batch, mk, head, d], 'float32')
            v = tpul.Tensor(dtype='float32', shape=[batch, mq, head, d], data=v_data) # 1, 1024, 10, 128
            # 3. init and compile tpulang model to top.mlir
            tpul.init(device=self.chip.upper())
            swint_block = SelfAttnBlock(batch, d, head, mq, mk)
            swint_block.forward(q, k, v)
            tpul.deinit()
            # tpul.compile will do top mlir inference with random input
            in_f32_npz = case_name + '_in_f32.npz'
            top_out = case_name + '_top_outputs.npz'
            # 4. deploy to bmodel
            deploy_cmd_base = f"model_deploy.py --mlir {case_name}.mlir "
            deploy_cmd_base += "--chip {} ".format(self.chip)
            deploy_cmd_base += "--test_input {} ".format(in_f32_npz)
            deploy_cmd_base += "--test_reference {} ".format(top_out)
            # deploy to [f32, f16, bf16] quant mode
            for mode in self.quant_modes:
                bmodel_name = "{}.bmodel".format(case_name + "_" + self.chip + "_" + mode)
                deploy_cmd = deploy_cmd_base
                deploy_cmd += "--model {} ".format(bmodel_name)
                deploy_cmd += "--quantize {} " .format(mode.upper())
                assert(os.system(deploy_cmd) == 0)

        _test_self_attn_block(1, 128, 10, 1024, 1024)

    def test_KeepOutputOrder(self, case_name):

        @tpulang(self.chip)
        def _test_keep_output_order(shape: List[int], dtype="float32"):
            shape = [2, 3, 10, 12]
            x_dtype = 'int8'
            t_dtype = 'uint8'
            x_data = rand_data(shape, x_dtype)
            t_data = rand_data([256], t_dtype)
            x = tpul.Tensor(dtype=x_dtype, shape=shape, data=x_data)
            t = tpul.Tensor(dtype=t_dtype, shape=list(t_data.shape), data=t_data, ttype="coeff")
            y = tpul.lut(x, t)
            z = tpul.lut(y, t)
            u = tpul.lut(z, t)
            self.compile_and_check(self.unique_name(case_name), [x], [z, u, y], is_quantized=True)

        _test_keep_output_order([4, 11])

    def test_MeanStdScale(self, case_name):
        @tpulang(self.chip)
        def _test_mean_std_scale(idtype="uint8", odtype="int8", shape=[4, 3, 500, 400]):
            mean = [128.0, 128.0, 128.0]
            std = [1/0.017, 1/0.017, 1/0.017]
            scale= [1.0, 1 / 2**6]
            is_quantized = True

            if idtype == "uint8":
                # customer_layer_path = os.getenv("CUSTOM_LAYER_PATH")
                # path = os.path.join(customer_layer_path, "test_if/unittest/data/picture/n02769748_5957.JPEG")
                # path = os.path.abspath(path)
                # img_rgb = cv2.imread(path)
                # img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).reshape((1,1,img_rgb.shape[0], img_rgb.shape[1]))
                # data_in = np.array(img_bgr)
                data_in = np.random.randint(0, 256, size=shape).astype(idtype)
                input = tpul.Tensor(name="in", dtype=idtype, shape=shape, data=data_in)
            else:
                np.random.seed(123)
                data_in = np.random.random(shape).astype(idtype)
                # np.savez("random_input.npz", data_in)
                input = tpul.Tensor(name="in", dtype=idtype, shape=shape, data=data_in)

            if odtype == "float16":
                is_quantized = False
                self.quant_modes = ["f16"]
                odtype = "float32"

            output = tpul.mean_std_scale(input, std, mean, scale, zero_points=[0,0], odtype=odtype, round_mode="half_up")
            self.compile_and_check(self.unique_name(case_name), [input], [output], is_quantized=is_quantized, top_mlir_inference=False)

        # output i8
        def _test_int8_to_int8_():
            _test_mean_std_scale("int8", "int8")
            _test_mean_std_scale("int8", "int8", [4, 3, 16, 224,224])

        def _test_uint8_to_int8_():
            _test_mean_std_scale("uint8", "int8")
            _test_mean_std_scale("uint8", "int8", [4, 3, 16, 224,224])

        def _test_f32_to_int8_():
            _test_mean_std_scale("float32", "int8")
            _test_mean_std_scale("float32", "int8", [4, 3, 16, 224,224])

        # output f16
        def _test_int8_to_f16_():
            _test_mean_std_scale("int8", "float16")
            _test_mean_std_scale("int8", "float16", [4, 3, 16, 224,224])

        def _test_uint8_to_f16_():
            _test_mean_std_scale("uint8", "float16")
            _test_mean_std_scale("uint8", "float16", [4, 3, 16, 224,224])

        def _test_f32_to_f16_():
            _test_mean_std_scale("float32", "float16")
            _test_mean_std_scale("float32", "float16", [4, 3, 16, 224,224])

        _test_int8_to_int8_()
        _test_uint8_to_int8_()
        _test_f32_to_int8_()

        _test_int8_to_f16_()
        _test_uint8_to_f16_()
        _test_f32_to_f16_()

    def test_MeanStdScale_Conv(self, case_name):
        @tpulang(self.chip)
        def _test_mean_std_scale(idtype="uint8", odtype="int8", shape=[4, 3, 500, 400]):
            mean = [128.0, 128.0, 128.0]
            std = [1 / 0.017, 1 / 0.017, 1 / 0.017]
            scale = [1.0, 1 / 2**6]
            is_quantized = True

            if idtype == "uint8":
                # customer_layer_path = os.getenv("CUSTOM_LAYER_PATH")
                # path = os.path.join(customer_layer_path, "test_if/unittest/data/picture/n02769748_5957.JPEG")
                # path = os.path.abspath(path)
                # img_rgb = cv2.imread(path)
                # img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).reshape((1,1,img_rgb.shape[0], img_rgb.shape[1]))
                # data_in = np.array(img_bgr)
                data_in = np.random.randint(0, 256, size=shape).astype(idtype)
                input = tpul.Tensor(name="in", dtype=idtype, shape=shape, data=data_in)
            else:
                np.random.seed(123)
                data_in = np.random.random(shape).astype(idtype)
                # np.savez("random_input.npz", data_in)
                input = tpul.Tensor(name="in", dtype=idtype, shape=shape, data=data_in)

            # if odtype == "float16":
            #     is_quantized = False
            #     self.quant_modes = ["f16"]
            #     odtype = "float32"

            output = tpul.mean_std_scale(
                input,
                std,
                mean,
                scale,
                zero_points=[0, 0],
                out_name="meanstdscale",
                odtype=odtype,
                round_mode="half_up",
            )

            if odtype in ["int8", "uint8"]:
                output = tpul.cast(output, "float16")

            kernel_shape = [4, 3, 3, 3]
            stride = [1, 1]
            conv = self.conv_op(output, kernel_shape, stride)
            self.compile_and_check(
                self.unique_name(case_name),
                [input],
                [conv],
                is_quantized=is_quantized,
                top_mlir_inference=False,
            )

        # output i8
        def _test_int8_to_int8_():
            _test_mean_std_scale("int8", "int8")

        def _test_uint8_to_int8_():
            _test_mean_std_scale("uint8", "int8")

        def _test_f32_to_int8_():
            _test_mean_std_scale("float32", "int8")

        # output f16
        def _test_int8_to_f16_():
            _test_mean_std_scale("int8", "float16")

        def _test_uint8_to_f16_():
            _test_mean_std_scale("uint8", "float16")

        def _test_f32_to_f16_():
            _test_mean_std_scale("float32", "float16")

        _test_int8_to_int8_()
        _test_uint8_to_int8_()
        _test_f32_to_int8_()

        _test_int8_to_f16_()
        _test_uint8_to_f16_()
        _test_f32_to_f16_()

    ########################################################################
    # ScatterElements case
    # ------------
    def test_ScatterElements(self,case_name):
        """ScatterElements"""

        @tpulang(self.chip)
        def _test_scatter(inputs:List[int] = None,
                          indices:List[int] = None,
                          updates:List[int] = None,
                          axis:int = 0,
                          dtype="float32",
                          is_quantized=False):
            inputs = np.array(inputs, dtype=dtype)
            indices = np.array(indices, dtype=np.int32)
            updates = np.array(updates, dtype=dtype)
            print(type(inputs))
            print(dtype)
            inputs_shapes = list(inputs.shape)
            indices_shapes = list(indices.shape)
            updates_shapes = list(updates.shape)
            x = tpul.Tensor(dtype=dtype, shape=inputs_shapes, data=inputs)
            y = tpul.Tensor(dtype='int32', shape=indices_shapes, data=indices)
            z = tpul.Tensor(dtype=dtype, shape=updates_shapes, data=updates)
            outputs = tpul.scatter(x, y, z, axis)
            self.compile_and_check(self.unique_name(case_name), [x, y, z], [outputs], is_quantized=is_quantized)
        _test_scatter([[0, 0, 0],[0, 0, 0],[0, 0, 0]], [[1, 0, 2],[0, 2, 1]], [[1.0, 1.1, 1.2],[2.0, 2.1, 2.2]], 0, dtype="float16", is_quantized=True)
        _test_scatter([[0, 0, 0],[0, 0, 0],[0, 0, 0]], [[1, 0, 2],[0, 2, 1]], [[1.0, 1.1, 1.2],[2.0, 2.1, 2.2]], 0, dtype="int8", is_quantized=True)
        _test_scatter([[1.0, 2.0, 3.0, 4.0, 5.0]],[[1.0, 3.0]],[[1.1, 2.1]], 1)
        _test_scatter([[0, 0, 0],[0, 0, 0],[0, 0, 0]], [[1, 0, 2],[0, 2, 1]], [[1.0, 1.1, 1.2],[2.0, 2.1, 2.2]], 0)

    ########################################################################
    # Roll case
    # ------------
    def test_Roll(self, case_name):
        """Roll"""
        @tpulang(self.chip)
        def _test_roll(input_shape,
                       shifts:List[int],
                       dim:List[int] = None,
                       dtype="float32",
                       is_quantized=False):
            inputs = rand_data(input_shape, dtype)
            x = tpul.Tensor(dtype=dtype, shape=list(input_shape), data=inputs)
            y = tpul.roll(x, shifts, dim)
            self.compile_and_check(self.unique_name(case_name), [x], [y], is_quantized=is_quantized)

        _test_roll((6,8,4,5,4),[1,2],[0,4], dtype="int8", is_quantized=True)
        _test_roll((4,2), 1, dtype="float16", is_quantized=True)
        _test_roll((4,2), (123,-133), (1,0))
        _test_roll((6,8,4,5,4),[1,2],[0,4], dtype="int16")

    ########################################################################
    # ScatterND case
    # ------------
    def test_ScatterND(self,case_name):
        """ScatterND"""

        @tpulang(self.chip)
        def _test_scatterND(inputs:List[int] = None,
                          indices:List[int] = None,
                          updates:List[int] = None,
                          dtype="float32",
                          is_quantized=False):
            inputs = np.array(inputs, dtype=dtype)
            indices = np.array(indices, dtype=np.uint32)
            updates = np.array(updates, dtype=dtype)
            inputs_shapes = list(inputs.shape)
            indices_shapes = list(indices.shape)
            updates_shapes = list(updates.shape)
            x = tpul.Tensor(dtype=dtype, shape=inputs_shapes, data=inputs)
            y = tpul.Tensor(dtype='uint32', shape=indices_shapes, data=indices)
            z = tpul.Tensor(dtype=dtype, shape=updates_shapes, data=updates)

            outputs = tpul.scatterND(x, y, z)
            self.compile_and_check(self.unique_name(case_name), [x, y, z], [outputs], is_quantized=is_quantized)

        _test_scatterND(
            inputs = [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]] ,
            indices = [[0], [2]],
            updates = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]] ,
            dtype="float16")

    def test_Rope(self, case_name):
        """Rope"""
        @tpulang(self.chip)
        def _test_rope( input_shape,
                        weight_shape,
                        dtype="float32",
                        is_quantized=False,
                        is_permute_optimize: bool = False,
                        mul1_round_mode : str ='half_up',
                        mul2_round_mode : str ='half_up',
                        add_round_mode : str ='half_up',
                        mul1_shift: int = 0,
                        mul2_shift: int = 0,
                        add_shift: int = 0,
                        mul1_saturation: bool = True,
                        mul2_saturation: bool = True,
                        add_saturation: bool = True,
                        out_name: str = None):

            input = rand_data(input_shape, dtype)
            weight0 = self.coeff_tensor(list(weight_shape), dtype)
            weight1 = self.coeff_tensor(list(weight_shape), dtype)
            x = tpul.Tensor(dtype=dtype, shape=list(input_shape), data=input)
            outputs = tpul.rope(x, weight0, weight1,
                        is_permute_optimize=is_permute_optimize,
                        mul1_round_mode = mul1_round_mode,
                        mul2_round_mode = mul2_round_mode,
                        add_round_mode = add_round_mode,
                        mul1_shift = mul1_shift,
                        mul2_shift = mul2_shift,
                        add_shift = add_shift,
                        mul1_saturation = mul1_saturation,
                        mul2_saturation = mul2_saturation,
                        add_saturation = add_saturation,
                        out_name=out_name
                        )
            self.compile_and_check(self.unique_name(case_name), [x], [outputs], is_quantized=is_quantized)

        ###############  test for 4 dims  ##################

        ###############  no permute_optimize ###############
        _test_rope((390, 4, 256, 64),(256, 64),
                        dtype="float32",
                        is_quantized=False,
                        is_permute_optimize= False,
                        mul1_round_mode = 'half_up',
                        mul2_round_mode = 'half_up',
                        add_round_mode = 'half_up',
                        mul1_shift = 0,
                        mul2_shift = 0,
                        add_shift= 0,
                        mul1_saturation = True,
                        mul2_saturation = True,
                        add_saturation = True
                        )

        _test_rope((390, 4, 256, 64),(256, 64),
                        dtype="int8",
                        is_quantized=True,
                        is_permute_optimize= False,
                        mul1_round_mode = 'half_up',
                        mul2_round_mode = 'half_up',
                        add_round_mode = 'half_up',
                        mul1_shift = -1,
                        mul2_shift = -6,
                        add_shift= 2,
                        mul1_saturation = True,
                        mul2_saturation = True,
                        add_saturation = True
                        )

        ################  permute_optimize ###############
        _test_rope((390, 256, 4, 64),(256, 64),
                        dtype="int8",
                        is_quantized=True,
                        is_permute_optimize= True,
                        mul1_round_mode = 'half_up',
                        mul2_round_mode = 'half_up',
                        add_round_mode = 'half_up',
                        mul1_shift = -1,
                        mul2_shift =-6,
                        add_shift=2,
                        mul1_saturation = True,
                        mul2_saturation = True,
                        add_saturation = True
                        )

        ###############  test for 3 dims  ##################

        _test_rope((390,256,64),(256,64),
                        dtype="int8",
                        is_quantized=True,
                        is_permute_optimize= False,
                        mul1_round_mode ='half_up',
                        mul2_round_mode ='half_up',
                        add_round_mode ='half_up',
                        mul1_shift= 0,
                        mul2_shift = 0,
                        add_shift= 0,
                        mul1_saturation = True,
                        mul2_saturation = True,
                        add_saturation = True
                        )

        ###############  test for 5 dims  ##################

        _test_rope((256,8,4,256,64),(256,64),
                        dtype="int8",
                        is_quantized=True,
                        is_permute_optimize= False,
                        mul1_round_mode ='half_up',
                        mul2_round_mode ='half_up',
                        add_round_mode ='half_up',
                        mul1_shift= 0,
                        mul2_shift = 0,
                        add_shift= 0,
                        mul1_saturation = True,
                        mul2_saturation = True,
                        add_saturation = True
                        )

    #######################################################################
    # Error Case: some error case
    # ------------
    def test_ErrorCase(self, case_name):

        @tpulang(self.chip)
        def _test_concat_conv():
            xshape = [1, 64, 68, 120]
            yshape = [1, 64, 68, 120]
            x_data = rand_data(xshape, "int16")
            y_data = rand_data(yshape, "int16")
            x = tpul.Tensor(dtype="int16", shape=xshape, data=x_data)
            y = tpul.Tensor(dtype="int16", shape=yshape, data=y_data)
            concat = tpul.concat([x,y])
            add = tpul.add_shift(concat, 0, 0, "int8")
            conv = self.conv_int_op(add, [64, 128, 1, 1], [1,1])
            requant = tpul.requant_int(conv, 6853, 21, 0, 2)
            self.compile_and_check(self.unique_name(case_name), [x, y], [requant], is_quantized=True)

        @tpulang(self.chip)
        def _test_conv_requant_axis():
            xshape = [1, 1, 64, 64]
            x_data = rand_data(xshape, "uint8")
            x = tpul.Tensor(dtype="uint8", shape=xshape, data=x_data)
            cast = tpul.cast(x, "int8")
            conv = self.conv_int_op(cast, [16, 1, 5, 5], [2,2], [2,2,2,2], bias=True)
            mul = [2198, 12522, 2630, 601, 1, 2371, 2058, 6401, 3131, 4390, 20056, 1375, 327, 1, 23562, 21906]
            requant = tpul.requant_int(conv, mul, [-22]*16, 0, 2, round_mode="half_up")
            relu = tpul.relu(requant)
            dequant = tpul.dequant_int_to_fp(relu, 0.125, 0)
            self.compile_and_check(self.unique_name(case_name), [x], [dequant], is_quantized=True)

        @tpulang(self.chip)
        def _test_conv_requant_axis2():
            xshape = [1, 3, 960, 1664]
            x_data = rand_data(xshape, "uint8")
            x = tpul.Tensor(dtype="uint8", shape=xshape, data=x_data)
            cast = tpul.mean_std_scale(x, [58.395015716552734, 57.120010375976563, 57.375015258789063],
                                       [123.67500305175781, 116.27999877929688, 103.52999877929688],
                                       [1.000000e+00, 3.125000e-02], zero_points=[0,0], odtype="int8")
            conv = self.conv_int_op(cast, [256, 3, 4, 4], [4,4], [0,0,0,0], bias=True)
            mul = [8381, 11835, 9388, 11401, 10891, 8457, 2442, 7073, 9480, 6737, 9899, 5729, 9247, 6445, 11883, 5498, 8389, 8254, 9035, 4795, 5431, 4774, 7962, 2001, 6811, 9148, 7447, 8088, 8148, 8124, 4268, 7164, 6662, 7889, 9799, 6141, 6454, 9638, 6885, 7053, 8776, 4893, 5639, 3476, 7384, 7073, 6067, 8070, 7949, 6663, 10634, 6589, 7549, 4053, 6298, 8084, 8207, 9423, 5541, 11113, 10640, 6017, 5042, 6613, 5482, 3437, 7466, 6489, 5872, 8635, 10686, 8576, 8542, 5182, 6499, 7351, 8841, 10095, 5223, 7515, 6151, 8082, 9586, 10747, 7551, 4053, 3260, 6773, 4580, 7966, 6628, 4689, 1234, 6854, 11772, 5365, 9933, 8418, 9137, 8345, 5945, 6779, 8608, 7204, 8424, 7990, 5571, 5631, 9834, 5945, 8715, 8189, 5594, 6131, 7666, 4383, 4096, 8637, 8867, 10363, 7863, 9184, 7302, 8977, 6935, 5398, 10467, 11291, 6763, 7402, 5761, 3598, 5450, 8365, 6781, 8239, 7628, 9388, 8061, 9350, 7290, 6931, 5790, 3514, 2276, 7604, 5675, 7100, 7216, 6476, 6666, 8814, 7559, 6071, 9114, 4527, 6095, 5111, 11796, 8120, 16662, 7833, 7184, 8716, 7383, 7989, 8743, 8577, 8757, 9068, 10968, 8238, 7177, 7762, 7374, 5857, 5857, 8226, 8738, 7535, 7490, 7143, 6851, 5107, 5607, 7955, 4956, 7717, 7417, 9053, 8342, 12142, 6698, 9804, 3725, 9466, 8122, 6621, 4760, 9267, 6098, 6435, 10698, 5545, 1250, 11823, 7338, 3970, 5438, 6714, 10597, 9560, 10605, 6050, 4358, 7538, 7324, 5266, 7816, 6659, 4947, 5419, 8608, 9861, 8430, 2764, 7393, 8697, 6340, 5242, 8964, 3939, 8123, 5877, 6811, 6013, 6209, 6775, 9088, 6125, 8780, 7812, 5923, 9768, 5002, 8820, 10018, 7739, 9847, 3970, 7242, 10376, 6602, 7098, 9144, 7874]
            requant = tpul.requant_int(conv, mul, [-15]*256, 0, 2, round_mode="half_up", out_dtype="int16")
            perm = tpul.permute(requant, [0,2,3,1])
            dequant = tpul.dequant_int_to_fp(perm, 1.0, 0)
            self.compile_and_check(self.unique_name(case_name), [x], [dequant], is_quantized=True)

        _test_concat_conv()
        _test_conv_requant_axis()
        _test_conv_requant_axis2()

    def test_model_combine(self, inputs, output="bm_combine"):
        from tools.bmodel_combine import combine
        # mode 0
        input_bmodles = [inp + "/compilation.bmodel" for inp in inputs]
        combine(input_bmodles, output=output+".bmodel", mode = 0)
        # mode 1
        combine(inputs, output=output, mode = 1)
        # model 2
        input_datas = []
        sizes = []
        for inp in inputs:
            file = open(inp + "/compilation.bmodel", 'rb')
            data = file.read()
            input_datas.append(data)
            sizes.append(len(data))
            file.close()
        output_data = combine(input_datas, sizes, mode=2)
        file = open(output+"Data.bmodel", 'wb')
        file.write(output_data)
        file.close()


def test_one_case_in_all(tester: TPULANG_IR_TESTER, case, error_cases, success_cases):
    t = Timer()
    try:
        tester.test_single(case)
    except:
        import traceback
        error_cases.append("{}:{}s".format(case, int(t.elapsed_time())))
        traceback.print_exc()
        return
    success_cases.append("{}:{}s".format(case, int(t.elapsed_time())))

def test_all_base(tester: TPULANG_IR_TESTER):
    import multiprocessing
    from utils.misc import collect_process
    process_number = multiprocessing.cpu_count() // 2 + 1
    processes = []
    error_cases = multiprocessing.Manager().list()
    success_cases = multiprocessing.Manager().list()
    for case in tester.test_function:
        if tester.check_support(case):
            p = multiprocessing.Process(target=test_one_case_in_all,
                                        name=case,
                                        args=(tester, case, error_cases, success_cases))
            processes.append(p)
        if len(processes) == process_number:
            collect_process(processes, error_cases)
            processes = []
    collect_process(processes, error_cases)
    processes = []
    print("Success: {}".format(success_cases))
    print("Failure: {}".format(error_cases))
    print("====== test_tpulang.py --chip {} TEST {} ======".format(
        tester.chip, 'Failed' if error_cases else 'Success'))
    return error_cases, success_cases

def test_all(tester: TPULANG_IR_TESTER):
    tester.no_save = False
    f, s = test_all_base(tester)
    if f:
      return f
    tester.no_save = True
    f, s = test_all_base(tester)
    return f

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--chip", default="bm1684x", type=str,
                        choices=['bm1684x', 'bm1688'], help="chip platform name")
    parser.add_argument("--mode", default="all", type=str, choices=['all', 'f32', 'f16', 'bf16'], help="quantize modes, only supports fp for now")
    parser.add_argument("--simple", action="store_true", help='do simple test for commit test')
    parser.add_argument("--case", default="all", type=str, help="test one case, if all, then test all cases")
    parser.add_argument("--show_all", action="store_true", help='show all cases')
    parser.add_argument("--report", default="", type=str, help="report file name")
    parser.add_argument("--no_save", action="store_true", help="whether to save mlir/weight in memory instead of hard disk.")
    parser.add_argument("--path", default="", type=str, help="the path to store intermediate file, accept "" or absolute path.")
    # yapf: enable
    args = parser.parse_args()
    tester = TPULANG_IR_TESTER(args.chip, args.mode, args.simple, args.no_save)
    if args.show_all:
        print("====== Show All Cases ============")
        for case in tester.test_function:
            print(case)
        exit(0)
    if args.path:
        dir = args.path
    else:
        dir = "tpulang_test_{}{}".format(args.chip, "_no_save" if args.no_save else "")
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)
    if args.case == "" or args.case.lower() == "all":
        if args.report:
            f, s = test_all_base(tester)
        else:
            test_all(tester)
    else:
        tester.test_single(args.case)
    if args.report:
        result = {'succsess': list(s), 'failure': list(f)}
        with open(args.report, "w") as f:
            json.dump(result, f)
