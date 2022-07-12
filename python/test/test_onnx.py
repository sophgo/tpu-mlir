#!/usr/bin/env python3
# Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
#
# Licensed under the Apache License v2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ==============================================================================

import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto
from tools.model_runner import mlir_inference
from tools.model_runner import onnx_inference
from tools.model_transform import *
from utils.mlir_shell import *
import os
import re

TEST_ONNX_IR = ["Conv2d"]
NOT_SUPPORT_INT8_TEST_IR = []
NOT_SUPPORT_BF16_TEST_IR = []


class ONNX_IR_TESTER(object):
    '''
        This class is built for testing single operator transform. Currently only onnx operator is supoorted.
    '''

    def __init__(self):
        self.converter = None
        self.test_function = {
            # Todo: add more operators
            "Conv2d": self.test_Conv2d,
        }
        # self.set_quant_mode()

    def set_quant_mode(self, ):
        # Todo: add bf16 and int8 quantization
        pass

    def onnx_convert_and_inference(self, input_data: dict, model_def, model_name: str,
                                   input_shapes: list):
        # onnx --> mlir conversion (origin and optimized mlir models will be generated and saved)
        fp32_mlir = "{}.mlir".format(model_name)
        tool = OnnxModelTransformTool(model_name, model_def, input_shapes, preprocessor=None)
        tool.model_transform(fp32_mlir)

        # onnx and top mlir model inference
        model_file = "{}_opt.onnx".format(model_name)
        onnx_outs = onnx_inference(input_data, model_file, False)
        num_outputs = len(onnx_outs)
        input_npz = "{}_in_fp32.npz".format(model_name)
        for name in input_data:
            input_data[name] = input_data[name].astype(np.float32)
        np.savez(input_npz, **input_data)
        mlir_outs = mlir_inference(input_data, fp32_mlir, False)

        # test the output
        if num_outputs > 1:
            pattern = re.compile(r"_[A-Z]\w+?$")
            for name in mlir_outs:
                onnx_name = pattern.sub("", name)
                print("Compare Mlir [{}] : Onnx [{}]".format(name, onnx_name))
                np.testing.assert_allclose(mlir_outs[name].flatten(),
                                           onnx_outs[onnx_name].flatten(),
                                           rtol=1e-5,
                                           atol=1e-01)
        else:
            mlir_outs = list(mlir_outs.values())[0]
            onnx_out = onnx_outs.popitem()[1]
            np.testing.assert_allclose(mlir_outs.flatten(),
                                       onnx_out.flatten(),
                                       rtol=1e-5,
                                       atol=1e-01)

    def test_Conv2d(self):
        test_case = 'Conv2d'
        batch_size = 4
        input_size = 10
        ic = 3
        oc = 5
        kernel_size = 3
        input_data = np.random.randn(batch_size, ic, input_size, input_size).astype(np.float32)
        weight_data = np.random.randn(oc, ic, kernel_size, kernel_size).astype(np.float32)
        bias_data = np.random.randn(oc).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, list(input_data.shape))

        output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                               list([batch_size, oc, 10, 10]))

        weight = helper.make_tensor('weight', TensorProto.FLOAT, list(weight_data.shape),
                                    weight_data)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, list(bias_data.shape), bias_data)

        node_def = helper.make_node(
            "Conv",
            inputs=['input', 'weight', 'bias'],
            outputs=['output'],
            kernel_shape=[kernel_size, kernel_size],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
            dilations=[1, 1],
            group=1,
        )

        graph_def = helper.make_graph([node_def],
                                      test_case, [input], [output],
                                      initializer=[weight, bias])

        model_def = helper.make_model(graph_def, producer_name=test_case)
        onnx.checker.check_model(model_def)

        self.onnx_convert_and_inference({'input': input_data}, model_def, test_case,
                                        [[batch_size, ic, input_size, input_size]])


if __name__ == "__main__":
    os.makedirs("onnx_test", exist_ok=True)
    os.chdir("onnx_test")
    tester = ONNX_IR_TESTER()

    for test_item in TEST_ONNX_IR:
        if test_item not in NOT_SUPPORT_INT8_TEST_IR:
            tester.test_function[test_item]()
            print("TEST {} success".format(test_item))
