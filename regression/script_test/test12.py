#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import transform.TpuLang as tpul
from transform.TpuLangConverter import TpuLangConverter
from transform.TpuLang import TpuLang
import numpy as np
import math


def rand_data(shape, dtype, min=-10, max=10):
    if dtype in ['float32', 'float16']:
        return np.clip(np.random.randn(*shape).astype(dtype), min, max)
    if dtype == 'int32' or 'uint32' or 'int16' or 'uint16' or 'int8' or 'uint8':
        return np.random.randint(0, 127, size=shape).astype(dtype)
    raise Exception("Not supported data type: {}!".format(dtype))


def coeff_tensor(shape, dtype, data=None, scale=None, zero_point=None):
    if data is None:
        data = rand_data(shape, dtype)
        data = data * scale if (dtype in ['float32', 'float16'] and scale is not None) else data
    if dtype in ["int8", "uint8"]:
        return tpul.Tensor(dtype=dtype,
                           shape=shape,
                           data=data,
                           ttype="coeff",
                           scale=scale,
                           zero_point=zero_point)
    else:
        return tpul.Tensor(dtype=dtype, shape=shape, data=data, ttype="coeff")


def conv_int_op(x,
                kshape,
                stride,
                pad=None,
                group=1,
                dilation=[1, 1],
                bias=False,
                zp=[0, 0],
                out_dtype="int32"):
    oc = kshape[0]
    weight = coeff_tensor(kshape,
                          x.dtype,
                          scale=1 / (math.sqrt(kshape[1] * kshape[2] * kshape[3])),
                          zero_point=0)
    bias = coeff_tensor(oc, out_dtype) if bias else None
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


def resnet_quant(x):
    rq0 = tpul.requant_fp_to_int(x, 1.0, 0, 0, 'int8')
    # conv1 = conv_block(rq0, [64, 3, 7, 7], [2, 2], [3,3,3,3], [2030043136, -13, 0])
    conv1 = conv_int_op(rq0, [64, 3, 7, 7], [2, 2], [3, 3, 3, 3], zp=[0, 0], out_dtype='int32')
    rq1 = tpul.requant_int(conv1, 2030043136, -7, 0, 0, 'int8', round_mode='half_away_from_zero')
    # relu1 = tpul.relu(rq1)
    conv2 = conv_int_op(rq1, [96, 64, 3, 3], [2, 2], [1, 1, 1, 1], zp=[0, 0], out_dtype='int32')
    rq2 = tpul.requant_int(conv2, 1748893696, -10, 0, 0, 'int8', round_mode='half_away_from_zero')
    coeff3 = coeff_tensor([1, 96, 1, 1], 'int8', scale=10.0)
    mul3 = tpul.mul(rq2, coeff3, scale=[0.25, 10.0, 2.5], out_dtype='int8')
    coeff4 = coeff_tensor([1, 96, 1, 1], 'int8', scale=2.0)
    add4 = tpul.add(mul3, coeff4, scale=[2.5, 2.0, 4.0], out_dtype='int8')
    return add4


if __name__ == "__main__":
    tpul.init("bm1684x")
    in_shape = [1, 3, 224, 224]
    x_data = rand_data(in_shape, 'float32')
    x = tpul.Tensor(dtype='float32', shape=in_shape, data=x_data)
    out = resnet_quant(x)
    TpuLang.graph.inputs = [x]
    TpuLang.graph.outputs = [out]
    TpuLang.graph.quantized_type_inference()
    # convert to mlir
    model_name = "resnetquant"
    converter = TpuLangConverter(name=model_name, graph=TpuLang.graph, mode="quantized")
    mlir_origin = model_name + '_origin.mlir'
    converter.generate_mlir(mlir_origin)
    tpul.deinit()
