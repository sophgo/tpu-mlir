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
import gc
'''
    There are 3 places to add new operator:
      1. TEST_ONNX_IR: list
      2. ONNX_IR_TESTER.test_function: dict
      3. for function of building new operator plz add at the tail of the class
'''
TEST_ONNX_IR = [
    "Conv2d",
    "Pool",
    "SiLU",
]


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
        self.test_function = {
            # Todo: add more operators
            "Conv2d": self.test_Conv2d,
            "Pool": self.test_Pool,
            "SiLU": self.test_SiLU,
        }
        self.quant_modes = ["f32", "int8"]  # no quantization when quant_mode == "f32"

    def onnx_convert(self, input_data: dict, graph_def, model_name: str):
        # onnx --> mlir conversion (origin and optimized mlir models will be generated and saved)
        fp32_mlir = "{}.mlir".format(model_name)
        model_def = helper.make_model(graph_def, producer_name=model_name)
        onnx.checker.check_model(model_def)
        tool = OnnxModelTransformTool(model_name, model_def)
        tool.model_transform(fp32_mlir)

        onnx_model = "{}_opt.onnx".format(model_name)
        input_npz = "{}_in_fp32.npz".format(model_name)
        for name in input_data:
            input_data[name] = input_data[name].astype(np.float32)
        np.savez(input_npz, **input_data)
        # top mlir outputs will be inferenced first in case the quant mode is int8
        onnx_outs = onnx_inference(input_data, onnx_model, True)
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
        tpu_mlir_out = tpu_mlir.replace(".mlir", "_tpu_out.npz")
        np.savez(tpu_mlir_out, **tpu_mlir_outs)
        bmodel_out = bmodel.replace(".bmodel", "_model_out.npz")
        np.savez(bmodel_out, **bmodel_outs)

        # compare
        npz_compare([ref_npz, tpu_mlir_out, "--tolerance", "0.6,0.6,0.6", "-v"])
        npz_compare([tpu_mlir_out, bmodel_out, "--tolerance", "0.99,0.99,0.9", "-v"])

        msg = quant_mode.upper()
        if quant_mode == "int8":
            msg += ", Asymmetric: {}".format(isAsym)

        print("[Success] test {} {}".format(model_name, msg))

    def convert_and_test(self, input_data: dict, graph_def, model_name: str):
        onnx_outs, top_mlir_outs, input_npz = self.onnx_convert(input_data, graph_def, model_name)

        # test onnx and mlir outputs
        counter = 0
        for name in onnx_outs:
            # top_mlir_name = name + "_{}".format(model_name)
            if name in top_mlir_outs:
                top_mlir_output = top_mlir_outs[name].flatten()
                onnx_output = onnx_outs[name].flatten()
                np.testing.assert_allclose(top_mlir_output, onnx_output, rtol=1e-5, atol=1e-1)
                print("* Onnx and TOP result compared *")
                counter += 1
        assert (counter > 0)

        for quant_mode in self.quant_modes:
            if quant_mode == "int8":
                for isAsym in [False, True]:
                    tpu_mlir, bmodel = self.bmodel_generate(model_name, top_mlir_outs, quant_mode,
                                                            isAsym)
                    self.inference_and_compare(top_mlir_outs, tpu_mlir, bmodel, input_npz,
                                               quant_mode, model_name, isAsym)
            else:
                tpu_mlir, bmodel = self.bmodel_generate(model_name, top_mlir_outs, quant_mode)
                self.inference_and_compare(top_mlir_outs, tpu_mlir, bmodel, input_npz, quant_mode,
                                           model_name)

    ##################################
    # adding operators from here
    ##################################

    def test_Pool(self):
        # For average and max pooling
        for test_case in ["AveragePool", "MaxPool"]:
            batch_size = 4
            ic = 3
            oc = 5
            input_size = 28
            output_size = 14
            conv_kernel = 3
            pool_kernel = 2
            input_data = np.random.randn(batch_size, ic, input_size, input_size).astype(np.float32)

            # Conv
            weight_data = np.random.randn(oc, ic, conv_kernel, conv_kernel).astype(np.float32)
            bias_data = np.random.randn(oc).astype(np.float32)
            conv_input = helper.make_tensor_value_info('input', TensorProto.FLOAT,
                                                       list(input_data.shape))
            # conv_output = helper.make_tensor_value_info('conv_output', TensorProto.FLOAT, [batch_size, oc, 28, 28])
            weight = helper.make_tensor('weight', TensorProto.FLOAT, list(weight_data.shape),
                                        weight_data)
            bias = helper.make_tensor('bias', TensorProto.FLOAT, list(bias_data.shape), bias_data)
            conv_node_def = helper.make_node(
                "Conv",
                inputs=['input', 'weight', 'bias'],
                outputs=['conv_output'],
                kernel_shape=[conv_kernel, conv_kernel],
                pads=[1, 1, 1, 1],
                strides=[1, 1],
                dilations=[1, 1],
                group=1,
            )

            # Pool
            # pool_input = helper.make_tensor_value_info('conv_output', TensorProto.FLOAT, [batch_size, oc, input_size, input_size])
            pool_output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                                        [batch_size, oc, 14, 14])

            pool_node_def = onnx.helper.make_node(
                test_case,
                inputs=['conv_output'],
                outputs=['output'],
                kernel_shape=[pool_kernel, pool_kernel],
                strides=[2, 2],
            )
            graph_def = helper.make_graph([conv_node_def, pool_node_def],
                                          test_case, [conv_input], [pool_output],
                                          initializer=[weight, bias])
            self.convert_and_test({"input": input_data}, graph_def, test_case)

    def test_Conv2d(self):
        test_case = 'Conv2d'
        oc = 32
        input_shape = [1, 16, 100, 100]
        filter_shape = [oc, 16, 3, 3]
        output_shape = [1, oc, 100, 100]
        input_data = np.random.randn(*input_shape).astype(np.float32)
        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        bias_data = np.random.randn(oc).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, list(bias_data.shape), bias_data)

        node_def = helper.make_node(
            "Conv",
            inputs=['input', 'weight', 'bias'],
            outputs=['output'],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
            dilations=[1, 1],
            group=1,
        )

        graph_def = helper.make_graph([node_def],
                                      test_case, [input], [output],
                                      initializer=[weight, bias])
        self.convert_and_test({'input': input_data}, graph_def, test_case)

    def test_SiLU(self):
        test_case = 'SiLU'
        input_shape = [1, 16, 100, 100]
        filter_shape = [16, 16, 3, 3]
        input_data = np.random.randn(*input_shape).astype(np.float32)
        weight_data = np.random.randn(*filter_shape).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)

        sigmoid_def = helper.make_node(
            "Sigmoid",
            inputs=['input'],
            outputs=['x1'],
        )
        mul_def = helper.make_node(
            "Mul",
            inputs=['input', 'x1'],
            outputs=['x2'],
        )
        conv_def = helper.make_node(
            "Conv",
            inputs=['x2', 'weight'],
            outputs=['output'],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
            dilations=[1, 1],
            group=1,
        )

        graph_def = helper.make_graph([sigmoid_def, mul_def, conv_def],
                                      test_case, [input], [output],
                                      initializer=[weight])

        self.convert_and_test({'input': input_data}, graph_def, test_case)


if __name__ == "__main__":
    os.makedirs("onnx_test", exist_ok=True)
    os.chdir("onnx_test")
    tester = ONNX_IR_TESTER()

    if len(sys.argv) == 2:
        tester.test_function[sys.argv[1]]()
    else:
        for test_item in TEST_ONNX_IR:
            tester.test_function[test_item]()
            print("====== TEST {} Success ======".format(test_item))
