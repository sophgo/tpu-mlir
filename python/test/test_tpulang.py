#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import numpy as np
import transform.TpuLang as bml

def rand_data(shape, dtype):
    if dtype == 'float32':
        return  np.random.random(shape).astype(np.float32)
    if dtype == 'int32' or 'uint32' or 'int16' or 'uint16' or 'int8' or 'uint8':
        return np.random.randint(0, 256, size=shape).astype(dtype)
    raise Exception("Not supported data type: {}!".format(dtype))

def coeff_tensor(shape, dtype, scale=1.0):
    data = rand_data(shape, dtype)
    data = data * scale if dtype == 'float32' else data
    return bml.Tensor(dtype=dtype, shape=shape, data=data, is_const=True)

def conv_op(x, kshape, stride, pad, zp, dtype):
    oc = kshape[0]
    weight = coeff_tensor(kshape, dtype)
    out_dtype =  dtype if dtype == 'float32' else 'int32'
    bias = coeff_tensor(oc, out_dtype)
    conv = bml.conv_v2(x, weight, bias=bias, stride=stride, pad=pad, \
                       input_zp=zp[0], weight_zp=zp[1], out_dtype=out_dtype)
    return conv

def model_def(x):
    rq0 = bml.requant_fp_to_int(x, 1.0, 0, 0, 'int8')
    conv1 = conv_op(rq0, [64,1,7,7],[2,2], None, [0,0], 'int8')
    rq2 = bml.requant_int(conv1, 2030043136, -13, 0, 0, 'int8', round_mode='half_away_from_zero')
    relu3 = bml.relu(rq2)
    conv4 = conv_op(relu3, [96,64,3,3], [2,2], None, [0,0], 'int8')
    rq5 = bml.requant_int(conv4, 1748893696, -10, 0, 0, 'int8', round_mode='half_away_from_zero')
    relu6 = bml.relu(rq5)
    dq7 = bml.dequant_int_to_fp(relu6, 0.25, 0)
    coeff8 = coeff_tensor([1,96,1,1], 'float32', 10.0)
    bml.constdata(coeff8)
    mul9 = bml.mul(dq7, coeff8)
    coeff10 = coeff_tensor([1,96,1,1], 'float32', -2.0)
    bml.constdata(coeff10)
    add11 = bml.add(mul9, coeff10)
    relu12 = bml.relu(add11)
    rq13 = bml.requant_fp_to_int(relu12, 4.0, 0, 0, 'int8')
    conv14 = conv_op(rq13, [96,96,3,3], [1,1], [1,1,1,1], [0,0], 'int8')
    rq15 = bml.requant_int(conv14, 1623457792, -8, 0, 0, 'int8', round_mode='half_away_from_zero')
    relu16 = bml.relu(rq15)
    conv17 = conv_op(relu16, [96,96,3,3], [1,1], [1,1,1,1], [0,0], 'int8')
    rq18 = bml.requant_int(conv17, 1623457792, -10, 0, 0, 'int8', round_mode='half_away_from_zero')
    dq19 = bml.dequant_int_to_fp(rq18, 0.0625, 0)
    add20 = bml.add(dq19, dq7)
    coeff21 = coeff_tensor([1,96,1,1], 'float32', 2.0)
    bml.constdata(coeff21)
    mul22 = bml.mul(add20, coeff21)
    coeff23 = coeff_tensor([1,96,1,1], 'float32', -2.0)
    bml.constdata(coeff23)
    add24 = bml.add(mul22, coeff23)
    relu25 = bml.relu(add24)
    rq26 = bml.requant_fp_to_int(relu25, 8.0, 0, 0, 'int8')
    conv27 = conv_op(rq26, [96,96,3,3], [1,1], [1,1,1,1], [0,0], 'int8')
    rq28 = bml.requant_int(conv27, 1712717824, -7, 0, 0, 'int8', round_mode='half_away_from_zero')
    dq29 = bml.dequant_int_to_fp(rq28, 0.0625, 0)
    return dq29

def model_def1(x):
    conv1 = conv_op(x, [64,1,7,7],[2,2], None, [None,None], 'float32')
    return conv1

def hik_demo():
    in_shape = [1,3,173,141]
    x_data = (rand_data(in_shape, 'float32') - 0.5) * 256
    x = bml.Tensor(dtype='float32', shape=in_shape, data=x_data)
    out = model_def1(x)
    bml.compile("model_def", [x], [out], False, 2)

if __name__ == '__main__':
    bml.init("BM1684X", True)
    hik_demo()
    bml.deinit()
