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
from tools.model_runner import mlir_inference, model_inference
from tools.model_runner import onnx_inference
from tools.npz_tool import npz_compare
from tools.model_transform import *
from utils.mlir_shell import *
import os
'''
    There are 3 places to add new operator:
      1. TEST_ONNX_IR: list
      2. ONNX_IR_TESTER.test_function: dict
      3. for function of building new operator plz add at the tail of the class
'''
TEST_ONNX_IR = ["Conv2d"]


def make_test_calibration_table(tensors, table_name):
    # simple calibration table
    with open(table_name, 'w') as f:
        for name in tensors:
            flatten_tensor = tensors[name].flatten()
            max_val = max(flatten_tensor)
            min_val = min(flatten_tensor)
            t = 1.1 * max(abs(min_val), abs(max_val)) + 0.01
            f.write("{} {} {} {}\n".format(name, t, min_val, max_val))


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
        self.quant_modes = ["fp32", "int8"]  # no quantization when quant_mode == "fp32"

    def onnx_convert(self, input_data: dict, model_def, model_name: str, input_shapes: list):
        # onnx --> mlir conversion (origin and optimized mlir models will be generated and saved)
        fp32_mlir = "{}.mlir".format(model_name)
        tool = OnnxModelTransformTool(model_name, model_def, input_shapes, preprocessor=None)
        tool.model_transform(fp32_mlir)

        onnx_model = "{}_opt.onnx".format(model_name)
        input_npz = "{}_in_fp32.npz".format(model_name)
        for name in input_data:
            input_data[name] = input_data[name].astype(np.float32)
        np.savez(input_npz, **input_data)
        # top mlir outputs will be inferenced first in case the quant mode is int8
        onnx_outs = onnx_inference(input_data, onnx_model, False)
        top_mlir_outs = mlir_inference(input_data, fp32_mlir, True)

        return (onnx_outs, top_mlir_outs, input_npz)

    def bmodel_generate(self,
                        model_name: str,
                        top_mlir_outs: dict,
                        quant_mode: str,
                        isAsym: bool = False):
        table_name = None
        top_mlir = "{}.mlir".format(model_name)
        tpu_mlir = "{}_tpu_{}".format(model_name, quant_mode)
        if quant_mode == "int8":
            table_name = "{}_cali_table".format(model_name)
            make_test_calibration_table(top_mlir_outs, table_name)
            tpu_mlir += "_asym" if isAsym else "_sym"
        # lowering
        mlir_lowering(top_mlir,
                      tpu_mlir + ".mlir",
                      mode="F32",
                      chip="bm1684x",
                      cali_table=table_name,
                      asymmetric=isAsym)
        # transform
        bmodel = tpu_mlir + "_1684x.bmodel"
        tpu_final = tpu_mlir + "_final.mlir"
        mlir_to_model(tpu_mlir + ".mlir", bmodel, tpu_final)

        return (tpu_mlir + ".mlir", bmodel)

    def inference_and_compare(self,
                              top_mlir_output: dict,
                              tpu_mlir: str,
                              bmodel: str,
                              input_npz: str,
                              quant_mode: str,
                              model_name: str,
                              isAsym: bool = False):
        input_data = np.load(input_npz)

        # tpu mlir inference
        tpu_mlir_outs = mlir_inference(input_data, tpu_mlir, dump_all=True)
        # bmodel inference
        bmodel_outs = model_inference(input_data, bmodel)

        ref_npz = "{}_top_mlir_out.npz".format(model_name)
        np.savez(ref_npz, **top_mlir_output)
        tpu_mlir_out = tpu_mlir.replace(".mlir", "_out.npz")
        np.savez(tpu_mlir_out, **tpu_mlir_outs)
        bmodel_out = bmodel.replace(".bmodel", "_out.npz")
        np.savez(bmodel_out, **bmodel_outs)

        # compare
        npz_compare([ref_npz, tpu_mlir_out, "--tolerance", "0.6,0.6,0.6", "-v"])
        npz_compare([tpu_mlir_out, bmodel_out, "--tolerance", "0.99,0.99,0.9", "-v"])

        msg = quant_mode.upper()
        if quant_mode == "int8":
            msg += ", Asymmetric: {}".format(isAsym)

        print("* {}, test success *".format(msg))

    def convert_and_test(self, input_data: dict, model_def, model_name: str, input_shapes: list):
        onnx_outs, top_mlir_outs, input_npz = self.onnx_convert(input_data, model_def, model_name,
                                                                input_shapes)

        # test onnx and mlir outputs
        top_mlir_output = list(top_mlir_outs.values())[1].flatten()
        onnx_output = list(onnx_outs.values())[0].flatten()
        np.testing.assert_allclose(top_mlir_output, onnx_output, rtol=1e-5, atol=1e-1)

        for quant_mode in self.quant_modes:
            if quant_mode == "int8":
                for isAsym in [True, False]:
                    tpu_mlir, bmodel = self.bmodel_generate(model_name, top_mlir_outs, quant_mode,
                                                            isAsym)
                    self.inference_and_compare(top_mlir_outs, tpu_mlir, bmodel, input_npz,
                                               quant_mode, model_name, isAsym)
            else:
                tpu_mlir, bmodel = self.bmodel_generate(model_name, top_mlir_outs, quant_mode)
                self.inference_and_compare(top_mlir_outs, tpu_mlir, bmodel, input_npz, quant_mode,
                                           model_name)

    # adding operator starts from here
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

        self.convert_and_test({'input': input_data}, model_def, test_case,
                              [[batch_size, ic, input_size, input_size]])


if __name__ == "__main__":
    os.makedirs("onnx_test", exist_ok=True)
    os.chdir("onnx_test")
    tester = ONNX_IR_TESTER()

    for test_item in TEST_ONNX_IR:
        tester.test_function[test_item]()
        print("****** TEST {} success ******".format(test_item))
