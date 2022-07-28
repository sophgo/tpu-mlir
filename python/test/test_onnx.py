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
TEST_ONNX_IR = [
    "AveragePool",
    "Conv2d",
    "MaxPool",
    "SiLU",
    "Concat",
    "Transpose",
    "LeakyRelu",
    "Mul",
    "Resize",
    "Softmax",
    "Log",
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
            "AveragePool": self.test_AveragePool,
            "MaxPool": self.test_MaxPool,
            "SiLU": self.test_SiLU,
            "Concat": self.test_Concat,
            "Transpose": self.test_Transpose,
            "LeakyRelu": self.test_LeakyRelu,
            "Mul": self.test_Mul,
            "Resize": self.test_Resize,
            "Softmax": self.test_Softmax,
            "Log": self.test_Log,
        }
        self.quant_modes = ["f32", "int8"]  # no quantization when quant_mode == "f32"

    def pytorch_transform_onnx(self, model, inputs, test_name):
        in_names = []
        if isinstance(inputs, tuple):
            for i in range(len(inputs)):
                in_names.append("in_{}".format(i))
        else:
            in_names = ["in_0"]
        torch.onnx.export(model,
                          inputs,
                          test_name + ".onnx",
                          export_params=True,
                          opset_version=11,
                          verbose=True,
                          input_names=in_names)

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
        tpu_mlir = "{}_{}".format(model_name, quant_mode)
        if quant_mode == "int8":
            table_name = "{}_cali_table".format(model_name)
            make_test_calibration_table(top_mlir_outs, table_name)
            tpu_mlir += "_asym" if isAsym else "_sym"

        # lowering
        mlir_lowering(top_mlir,
                      tpu_mlir + ".mlir",
                      mode=quant_mode,
                      chip="bm1684x",
                      cali_table=table_name,
                      asymmetric=isAsym)
        # transform
        bmodel = tpu_mlir + ".bmodel"
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

        ref_npz = "{}_top_out.npz".format(model_name)
        np.savez(ref_npz, **top_mlir_output)
        tpu_npz = tpu_mlir.replace(".mlir", "_tpu_out.npz")
        np.savez(tpu_npz, **tpu_mlir_outs)
        model_npz = bmodel.replace(".bmodel", "_model_out.npz")
        np.savez(model_npz, **bmodel_outs)

        # compare
        ref_tpu_tolerance = "0.9,0.9"
        if quant_mode == "int8":
            ref_tpu_tolerance = "0.95,0.70" if not isAsym else "0.90,0.54"
        npz_compare([ref_npz, tpu_npz, "--tolerance", ref_tpu_tolerance, "-v"])
        npz_compare([tpu_npz, model_npz, "--tolerance", "0.99,0.9", "-v"])

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
    def test_AveragePool(self):
        test_case = 'AveragePool'
        input_shape = [1, 32, 128, 128]
        output_shape = [1, 32, 64, 64]
        input_data = np.random.randn(*input_shape).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        pool_def = onnx.helper.make_node(
            'AveragePool',
            inputs=['input'],
            outputs=['output'],
            kernel_shape=[2, 2],
            strides=[2, 2],
        )
        graph_def = helper.make_graph([pool_def], test_case, [input], [output])
        self.convert_and_test({"input": input_data}, graph_def, test_case)

    def test_MaxPool(self):
        test_case = 'MaxPool'
        input_shape = [1, 32, 128, 128]
        output_shape = [1, 32, 64, 64]
        input_data = np.random.randn(*input_shape).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        pool_def = onnx.helper.make_node(
            'MaxPool',
            inputs=['input'],
            outputs=['output'],
            kernel_shape=[2, 2],
            strides=[2, 2],
        )
        graph_def = helper.make_graph([pool_def], test_case, [input], [output])
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
        input_shape = [1, 16, 64, 64]
        output_shape = [1, 16, 64, 64]
        input_data = np.random.randn(*input_shape).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        sigmoid_def = helper.make_node(
            "Sigmoid",
            inputs=['input'],
            outputs=['x1'],
        )
        mul_def = helper.make_node(
            "Mul",
            inputs=['input', 'x1'],
            outputs=['output'],
        )
        graph_def = helper.make_graph([sigmoid_def, mul_def], test_case, [input], [output])
        self.convert_and_test({'input': input_data}, graph_def, test_case)

    def test_Concat(self):
        test_case = "Concat"
        input_shape = {"input1": [1, 4, 32], "input2": [1, 4, 64], "input3": [1, 4, 64]}
        output_shape = [1, 4, 32 + 64 + 64]
        input_data = {k: np.random.randn(*x).astype(np.float32) for k, x in input_shape.items()}

        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, x) for k, x in input_shape.items()
        ]
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        concat_def = helper.make_node("Concat",
                                      inputs=list(input_shape.keys()),
                                      outputs=["output"],
                                      axis=2)

        graph_def = helper.make_graph([concat_def], test_case, inputs, [output])
        self.convert_and_test(
            input_data,
            graph_def,
            test_case,
        )

    def test_Transpose(self):
        test_case = 'Transpose'
        input_shapes = [[1, 16, 32, 32], [4, 3, 85, 20, 20]]
        transpose_orders = {4: [0, 2, 1, 3], 5: [0, 1, 3, 4, 2]}
        for input_shape in input_shapes:
            input_data = np.random.randn(*input_shape).astype(np.float32)
            input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
            order = transpose_orders[len(input_shape)]
            output_shape = [input_shape[order[i]] for i in range(len(order))]
            output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
            transpose_def = helper.make_node(test_case,
                                             inputs=['input'],
                                             outputs=['output'],
                                             perm=order)
            graph_def = helper.make_graph([transpose_def], test_case, [input], [output])
            self.convert_and_test({'input': input_data}, graph_def, test_case)
            print("[Success] {}D data test".format(len(input_shape)))

    def test_LeakyRelu(self):
        test_case = "LeakyRelu"
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

        conv_def = helper.make_node(
            "Conv",
            inputs=['input', 'weight', 'bias'],
            outputs=['conv_output'],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
            dilations=[1, 1],
            group=1,
        )

        leakyrelu_def = helper.make_node("LeakyRelu",
                                         inputs=['conv_output'],
                                         outputs=['output'],
                                         alpha=0.67)

        graph_def = helper.make_graph([conv_def, leakyrelu_def],
                                      test_case, [input], [output],
                                      initializer=[weight, bias])
        self.convert_and_test({'input': input_data}, graph_def, test_case)

    def test_Mul(self):
        test_case = 'Mul'
        input_shape = {"input1": [1, 3, 27, 27], "input2": [1, 3, 27, 27]}
        output_shape = [1, 3, 27, 27]
        input_data = {k: np.random.randn(*x).astype(np.float32) for k, x in input_shape.items()}

        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, x) for k, x in input_shape.items()
        ]
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        mul_def = helper.make_node("Mul", inputs=list(input_shape.keys()), outputs=["output"])

        graph_def = helper.make_graph([mul_def], test_case, inputs, [output])
        self.convert_and_test(
            input_data,
            graph_def,
            test_case,
        )

    def test_Resize(self):
        test_case = "Resize"
        input_shape = [1, 16, 32, 32]
        output_shape = [1, 16, 64, 64]
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        roi_data = np.array([], dtype=np.float32)
        scales_data = np.array([1, 1, 2, 2], dtype=np.float32)
        roi = helper.make_tensor('roi', TensorProto.FLOAT, [0], roi_data)
        scales = helper.make_tensor('scales', TensorProto.FLOAT, [4], scales_data)
        resize_def = helper.make_node('Resize',
                                      inputs=['input', 'roi', 'scales'],
                                      outputs=['output'],
                                      mode='nearest',
                                      nearest_mode='floor',
                                      coordinate_transformation_mode='asymmetric')
        graph_def = helper.make_graph([resize_def],
                                      test_case, [input], [output],
                                      initializer=[roi, scales])
        input_data = np.random.randn(*input_shape).astype(np.float32)
        self.convert_and_test({"input": input_data}, graph_def, test_case)

    def test_Softmax(self):
        test_case = 'Softmax'
        input_shape = [1, 1000, 1, 1]
        axis = 1
        input_data = np.random.randn(*input_shape).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
        softmax_def = helper.make_node(test_case, inputs=['input'], outputs=['output'], axis=axis)
        graph_def = helper.make_graph([softmax_def], test_case, [input], [output])
        self.convert_and_test({'input': input_data}, graph_def, test_case)

    def test_Log(self):
        test_case = 'Log'
        input_shape = [1, 3, 32, 32]
        input_data = np.clip(np.random.randn(*input_shape).astype(np.float32) * 10.0, 0.5, 8)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
        log_def = helper.make_node(test_case, inputs=['input'], outputs=['output'])
        graph_def = helper.make_graph([log_def], test_case, [input], [output])
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
