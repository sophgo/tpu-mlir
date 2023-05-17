#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import numpy as np
import os, sys
import transform.TpuLang as tpul
from typing import List

# initialize tpulang
tpul.init("BM1684X", True)

def rand_data(shape, dtype):
    if dtype == 'float32':
        return np.random.random(shape).astype(np.float32)
    if dtype == 'int32' or 'uint32' or 'int16' or 'uint16' or 'int8' or 'uint8':
        return np.random.randint(0, 256, size=shape).astype(dtype)
    raise Exception("Not supported data type: {}!".format(dtype))


# define the swapChannel op by tpul.custom
def swapChannel(inputs, dtype="float32"):

    def shape_func(tensors_in):
        # the shape inference function
        # return the list of output shapes
        return [tensors_in[0].shape]

    out_names = ["out"]
    params = {"order": [2, 1, 0]}
    outs = tpul.custom(
        tensors_in=inputs,
        shape_func=shape_func,
        # op_name should be consistent with the backend, case insensitive
        op_name="SwapChannel",
        params=params,
        out_dtypes=[dtype],
        out_names=out_names)
    return outs


if __name__ == "__main__":
    os.makedirs("tmp", exist_ok=True)
    os.chdir("tmp")
    # 1. prepare the input
    dtype = "float32"
    input_shape = [1, 3, 14, 14]
    x_data = rand_data(input_shape, dtype)
    x = tpul.Tensor(dtype=dtype, shape=input_shape, data=x_data)

    # 2. inference
    outs = swapChannel(inputs=[x], dtype=dtype)

    # 3. compile to Top mlir file, the input will be saved in {top_mlir}_in_f32.npz
    top_mlir = "test_case"
    tpul.compile(top_mlir, [x], outs, False, 2, has_custom=True)

    # 4. deploy the model to F32 bmodel
    deploy_cmd = f"model_deploy.py --mlir {top_mlir}.mlir --quantize F32 --chip bm1684x --model test_case.bmodel"
    os.system(deploy_cmd)

    # 5. do bmodel inference and compare with the origin output
    infer_cmd = f"model_runner.py --input {top_mlir}_in_f32.npz --model test_case.bmodel --output ref_data.npz"
    os.system(infer_cmd)
    # save the origin output for comparison
    origin_out = x_data[:, [2, 1, 0], :, :]
    # There are two outputs because in non-f32 quant mode, the result will be
    # dequant back to f32 with castOp so that the final result will be named with the suffix '_f32'
    out = {outs[0].name: origin_out, f"{outs[0].name}_f32": origin_out}
    np.savez("target_data", **out)

    cmp_cmd = "npz_tool.py compare target_data.npz ref_data.npz"
    os.system(cmp_cmd)
