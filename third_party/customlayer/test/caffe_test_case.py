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
import os
import caffe
from my_converter import MyCaffeConverter
import my_caffe_layer


def rand_data(shape, dtype):
    if dtype == 'float32':
        return np.random.random(shape).astype(np.float32)
    if dtype == 'int32' or 'uint32' or 'int16' or 'uint16' or 'int8' or 'uint8':
        return np.random.randint(0, 256, size=shape).astype(dtype)
    raise Exception("Not supported data type: {}!".format(dtype))


if __name__ == "__main__":
    model_name = "caffe_test_net"
    prototxt = "./../my_model.prototxt"
    caffemodel = "./../my_model.caffemodel"

    os.makedirs("tmp", exist_ok=True)
    os.chdir("tmp")

    # 1. prepare the input and do origin model inference
    dtype = "float32"
    input_shape0 = [1, 3, 14, 14]
    input_shape1 = [1, 3, 24, 26]
    input_shapes = [input_shape0, input_shape1]
    input0 = rand_data(input_shape0, dtype)
    input1 = rand_data(input_shape1, dtype)
    inputs = {"input0": input0, "input1": input1}
    np.savez(f"{model_name}_in_f32", **inputs)

    origin_output = f"{model_name}_ref_data.npz"
    caffe_infer_cmd = f"model_runner.py --input {model_name}_in_f32.npz --model {prototxt} --weight {caffemodel} --output {origin_output}"
    os.system(caffe_infer_cmd)

    # 2. convert caffe model to top mlir
    mlir_file = f"{model_name}.mlir"

    converter = MyCaffeConverter(model_name=model_name,
                                 prototxt=prototxt,
                                 caffemodel=caffemodel,
                                 input_shapes=input_shapes,
                                 output_names=[])
    converter.model_convert()

    # 3. deploy to F32 bmodel
    deploy_cmd = f"model_deploy.py --mlir {model_name}.mlir --quantize F32 --chip bm1688 --model {model_name}.bmodel"
    os.system(deploy_cmd)

    # 4. do bmodel inference and compare with the origin output
    bmodel_output = f"{model_name}_target_data.npz"
    bmodel_infer_cmd = f"model_runner.py --input {model_name}_in_f32.npz --model {model_name}.bmodel --output {bmodel_output}"
    os.system(bmodel_infer_cmd)

    # 5. compare results
    cmp_cmd = f"npz_tool.py compare {bmodel_output} {origin_output}"
    os.system(cmp_cmd)
