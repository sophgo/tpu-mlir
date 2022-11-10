#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from re import T
import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto
from tools.model_runner import mlir_inference, model_inference, onnx_inference
from tools.npz_tool import npz_compare
from tools.model_transform import *
from utils.mlir_shell import *
import os
import torch
import torch.nn as nn
import onnxruntime

Failed_Cases = [
    "GroupFC", "GRU", "GRU2", "LRN", "LSTM", "LSTM2", "Neg", "Reduce", "Reduce2", "ReduceL2",
    "Reciprocal", "Sub", "Sub2", "Sum", "Where", "TorchLayerNorm", "TorchLogSoftmax",
    "TorchMaskedFill", "TorchWhere"
]


class ONNX_IR_TESTER(object):
    # This class is built for testing single operator transform.
    def __init__(self):
        self.test_function = {
            #############################
            # ONNX Test Case, Alphabetically
            #############################
            "Abs": self.test_Abs,
            "Add": self.test_Add,
            "AddConst": self.test_AddConst,
            "AvgPool1D": self.test_AvgPool1D,
            "AvgPool2D": self.test_AvgPool2D,
            "AvgPool3D": self.test_AvgPool3D,
            "BatchMatMul": self.test_BatchMatMul,
            "BroadcastAdd": self.test_BroadcastAdd,
            "BroadcastMul": self.test_BroadcastMul,
            "BroadcastMulConst": self.test_BroadcastMulConst,
            "Concat": self.test_Concat,
            "Conv1d": self.test_Conv1d,
            "Conv2d": self.test_Conv2d,
            "Conv3d": self.test_Conv3d,
            "ConvTranspose2D": self.test_ConvTranspose,
            "Clip": self.test_Clip,
            "DepthToSpace": self.test_DepthToSpace,
            "Div": self.test_Div,
            "Expand": self.test_Expand,
            "Expand2": self.test_Expand2,
            "Gather": self.test_Gather,
            "GatherToSlice": self.test_GatherToSlice,
            "Gemm": self.test_Gemm,
            "GroupFC": self.test_GroupFC,
            "GRU": self.test_GRU,
            "GRU2": self.test_GRU2,  # test gru output Y_h
            "LeakyRelu": self.test_LeakyRelu,
            "Log": self.test_Log,
            "LRN": self.test_LRN,
            "LSTM": self.test_LSTM,
            "LSTM2": self.test_LSTM2,
            "MaxPool1D": self.test_MaxPool1D,
            "MaxPool2D": self.test_MaxPool2D,
            "MaxPool3D": self.test_MaxPool3D,
            "MatMul": self.test_MatMul,
            "Max": self.test_Max,
            "Mul": self.test_Mul,
            "Min": self.test_Min,
            "MulConst": self.test_MulConst,
            "Neg": self.test_Neg,
            "Pad0": self.test_Pad0,  # zero pad
            "Pad1": self.test_Pad1,  # pad val
            # "PadEdge": self.test_PadEdge, # nntc edge pad to be implemented
            "PadReflect": self.test_PadReflect,
            "PRelu": self.test_PRelu,
            "Resize": self.test_Resize,
            "Resize2": self.test_Resize2,
            "Reshape": self.test_Reshape,
            "Reduce": self.test_Reduce,
            "Reduce2": self.test_Reduce2,
            "ReduceL2": self.test_ReduceL2,
            "ReduceMean": self.test_ReduceMean,
            "Reciprocal": self.test_Reciprocal,
            "SiLU": self.test_SiLU,
            "Softmax": self.test_Softmax,
            "Squeeze": self.test_Squeeze,
            "Sigmoid": self.test_Sigmoid,
            "Slice": self.test_Slice,
            "Split": self.test_Split,
            "Scale": self.test_Scale,
            "Sub": self.test_Sub,
            "Sub2": self.test_Sub2,
            "Sum": self.test_Sum,
            "Tile": self.test_Tile,
            "Transpose": self.test_Transpose,
            "Where": self.test_Where,
            #############################
            # Torch Test Case, Alphabetically
            #############################
            "TorchLayerGroup": self.test_TorchLayerGroup,
            "TorchLayerNorm": self.test_TorchLayerNorm,
            "TorchLogSoftmax": self.test_TorchLogSoftmax,
            "TorchMaskedFill": self.test_TorchMaskedFill,
            "TorchWhere": self.test_TorchWhere,
        }
        # no quantization when quant_mode == "f32"
        self.quant_modes = ["f32", "int8", "f16", "bf16"]
        self.chip = self.get_chip_name()
        if self.chip.find("cv18") >= 0:
            self.quant_modes = ["int8", "bf16"]

    def get_chip_name(self):
        runchip = os.environ.get('SET_CHIP_NAME', None)
        if not runchip:
            print("no found SET_CHIP_NAME environment value, set bm1684x as default")
            runchip = "bm1684x"
        return runchip.lower()

    def test_single(self, case: str):
        print("Test: {}".format(case))
        if case in self.test_function:
            self.test_function[case](case)
            print("====== TEST {} Success ======".format(case))
        else:
            raise RuntimeError("case [{}] is not exist".format(case))

    def test_all(self):
        for case in self.test_function:
            if case not in Failed_Cases:
                self.test_single(case)
        print("====== ALL TEST Success ======".format(case))

    def onnx_convert(self, input_data: dict, graph_def, model_name: str):
        # onnx --> mlir conversion (origin and optimized mlir models will be generated and saved)
        fp32_mlir = "{}.mlir".format(model_name)
        model_def = helper.make_model(graph_def, producer_name=model_name)
        model_def.opset_import[0].version = 13
        onnx.checker.check_model(model_def)
        tool = OnnxTransformer(model_name, model_def)
        tool.model_transform(fp32_mlir)

        onnx_model = "{}_opt.onnx".format(model_name)
        input_npz = "{}_in_fp32.npz".format(model_name)
        for name in input_data:
            if input_data[name].dtype in [np.int64, np.int32]:
                input_data[name] = input_data[name].astype(np.int32)
            else:
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
            self.make_test_calibration_table(top_mlir_outs, table_name)
            tpu_mlir += "_asym" if isAsym else "_sym"

        # lowering
        mlir_lowering(top_mlir,
                      tpu_mlir + ".mlir",
                      mode=quant_mode,
                      chip=self.chip,
                      cali_table=table_name,
                      asymmetric=isAsym)

        # transform
        tpu_final = tpu_mlir + "_final.mlir"
        if self.chip.find("cv18") >= 0:
            # for cv183x cv182x cv181 cv180
            bmodel = tpu_mlir + ".cvimodel"
            mlir_to_cvi_model(tpu_mlir + ".mlir", bmodel, tpu_final)
        else:
            bmodel = tpu_mlir + ".bmodel"
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
        ref_tpu_tolerance = "0.9,0.9"
        input_data = np.load(input_npz)
        # save ref
        ref_npz = "{}_top_out.npz".format(model_name)
        np.savez(ref_npz, **top_mlir_output)
        # tpu mlir inference and compare
        if quant_mode == "int8":
            ref_tpu_tolerance = "0.95,0.70" if not isAsym else "0.90,0.54"
        tpu_mlir_outs = mlir_inference(input_data, tpu_mlir, dump_all=True)
        tpu_npz = tpu_mlir.replace(".mlir", "_tpu_out.npz")
        np.savez(tpu_npz, **tpu_mlir_outs)
        npz_compare([ref_npz, tpu_npz, "--tolerance", ref_tpu_tolerance, "-v"])
        # bmodel / cvimodel inference and compare
        model_outs = model_inference(input_data, bmodel)
        model_npz = bmodel.replace("." + bmodel.split(".")[-1], "_model_out.npz")
        np.savez(model_npz, **model_outs)
        npz_compare([tpu_npz, model_npz, "--tolerance", "0.95,0.80", "-v"])

        msg = quant_mode.upper()
        if quant_mode == "int8":
            msg += ", Asymmetric: {}".format(isAsym)

        print("[Success] test {} {}".format(model_name, msg))

    def make_test_calibration_table(self, tensors, table_name):
        # simple calibration table
        with open(table_name, 'w') as f:
            for name in tensors:
                flatten_tensor = tensors[name].flatten()
                max_val = max(flatten_tensor)
                min_val = min(flatten_tensor)
                t = 1.1 * max(abs(min_val), abs(max_val)) + 0.01
                f.write("{} {} {} {}\n".format(name, t, min_val, max_val))

    def simple_onnx_inference(self, input_data, onnx_file):
        ort_session = onnxruntime.InferenceSession(onnx_file)
        return ort_session.run(None, input_data)

    def torch_and_onnx_compare(self, input_data: dict, onnx_file: str, origin_output):
        onnx_outs = self.simple_onnx_inference(input_data, onnx_file)
        num_outputs = len(onnx_outs)
        if isinstance(origin_output, tuple):
            assert (len(origin_output) == num_outputs)
            for i in range(num_outputs):
                np.testing.assert_allclose(origin_output[i].data.numpy().flatten(),
                                           onnx_outs[i].flatten(),
                                           rtol=1e-5,
                                           atol=1e-01)
        else:
            np.testing.assert_allclose(origin_output.data.numpy().flatten(),
                                       onnx_outs[0].flatten(),
                                       rtol=1e-5,
                                       atol=1e-01)
        print("* Torch and Onnx result compared *")

    def torch_and_test(self, inputs, torch_model: nn.Module, model_name: str):
        origin_output = torch_model(inputs)
        onnx_file = model_name + ".onnx"
        in_names = []
        in_data = {}
        if isinstance(inputs, tuple):
            for idx, input in enumerate(inputs):
                name = "in_{}".format(idx)
                in_names.append(name)
                in_data[name] = input.data.numpy().astype(np.float32)
        else:
            in_names.append('in_0')
            in_data['in_0'] = inputs.data.numpy().astype(np.float32)

        torch.onnx.export(torch_model,
                          inputs,
                          onnx_file,
                          export_params=True,
                          verbose=True,
                          input_names=in_names)
        onnx_model = onnx.load(onnx_file)
        self.torch_and_onnx_compare(in_data, onnx_file, origin_output)
        self.onnx_and_test(in_data, onnx_model.graph, model_name)

    def onnx_and_test(self, input_data: dict, graph_def, name: str = ""):
        model_name = name if name else graph_def.name
        onnx_outs, top_mlir_outs, input_npz = self.onnx_convert(input_data, graph_def, model_name)
        # test onnx and mlir outputs
        counter = 0
        for name in onnx_outs:
            if name in top_mlir_outs:
                top_mlir_output = top_mlir_outs[name].flatten()
                onnx_output = onnx_outs[name].flatten()
                np.testing.assert_allclose(top_mlir_output, onnx_output, rtol=1e-5, atol=1e-1)
                counter += 1
        if counter > 0:
            print("* Onnx and TOP result compared *")
        else:
            print("* No comparison between Onnx and TOP result *")

        for quant_mode in self.quant_modes:
            if quant_mode == "int8":
                for isAsym in [False, True]:
                    if self.chip.find("cv18") >= 0 and isAsym:
                        continue
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
    def AvgPoolBase(self, case_name, input_shape, output_shape, kernel, strides):
        input_data = np.random.randn(*input_shape).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        pool_def = helper.make_node(
            'AveragePool',
            inputs=['input'],
            outputs=['output'],
            kernel_shape=kernel,
            strides=strides,
        )
        graph_def = helper.make_graph([pool_def], case_name, [input], [output])
        self.onnx_and_test({"input": input_data}, graph_def)

    def test_AvgPool1D(self, case_name):
        input_shape = [1, 32, 128]
        output_shape = [1, 32, 64]
        input_data = np.random.randn(*input_shape).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        pool_def = helper.make_node(
            'AveragePool',
            inputs=['input'],
            outputs=['pool_output'],
            kernel_shape=[2],
            strides=[2],
        )

        mul_const = helper.make_tensor(name='const_mul',
                                       data_type=TensorProto.FLOAT,
                                       dims=[],
                                       vals=[2.0])
        initializer = list()
        initializer.append(mul_const)
        mul_def = helper.make_node('Mul', ['pool_output', 'const_mul'], ['output'])

        graph_def = helper.make_graph([pool_def, mul_def], case_name, [input], [output],
                                      initializer)
        self.onnx_and_test({"input": input_data}, graph_def)

    def test_AvgPool2D(self, case_name):
        self.AvgPoolBase(case_name, [1, 32, 128, 128], [1, 32, 64, 64], [2, 2], [2, 2])

    def test_AvgPool3D(self, case_name):
        self.AvgPoolBase(case_name, [2, 32, 16, 32, 64], [2, 32, 8, 16, 32], [2, 2, 2], [2, 2, 2])

    def test_BatchMatMul(self, case_name):
        # matmul(1x16x40x64, 1x16x64x40) => 1x16x40x40
        input1_shape = [4, 16, 40, 64]
        input2_shape = [4, 16, 64, 40]
        output_shape = [4, 16, 40, 40]

        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, input1_shape)
        input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, input2_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        matmul_node = helper.make_node(
            'MatMul',  # node name
            ['input1', 'input2'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [matmul_node],
            case_name,
            [input1, input2],
            [output],
        )
        input1_data = np.random.randn(*input1_shape).astype(np.float32)
        input2_data = np.random.randn(*input2_shape).astype(np.float32)
        self.onnx_and_test({"input1": input1_data, "input2": input2_data}, graph_def)

    def MaxPoolBase(self, case_name, input_shape, output_shape, kernel, strides):
        input_data = np.random.randn(*input_shape).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        pool_def = helper.make_node(
            'MaxPool',
            inputs=['input'],
            outputs=['output'],
            kernel_shape=kernel,
            strides=strides,
        )
        graph_def = helper.make_graph([pool_def], case_name, [input], [output])
        self.onnx_and_test({"input": input_data}, graph_def)

    def test_GroupFC(self, case_name):
        input_shape = [16, 40, 43]
        filter_shape = [16, 43, 48]
        bias_shape = [16, 1, 48]
        output_shape = [16, 40, 48]

        input_data = np.random.rand(*input_shape).astype(np.float32)
        filter_data = np.random.rand(*filter_shape).astype(np.float32)
        bias_data = np.random.rand(*bias_shape).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        filter_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['filter'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=filter_data.shape,
                vals=filter_data.flatten(),
            ),
        )
        bias_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['bias'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=bias_data.shape,
                vals=bias_data.flatten(),
            ),
        )

        fc_node = helper.make_node(
            'MatMul',  # node name
            ['input', 'filter'],  # inputs
            ['fc'],  # outputs
        )
        add_node = helper.make_node(
            'Add',
            ['fc', 'bias'],
            ['output'],
        )

        graph_def = helper.make_graph([filter_def, bias_def, fc_node, add_node], case_name, [input],
                                      [output])
        self.onnx_and_test({"input": input_data}, graph_def)

    def test_GRU(self, case_name):
        seq_length = 75
        batch_size = 2
        num_dir = 2
        input_size = 64
        hidden_size = 32
        direction = 'forward' if num_dir == 1 else 'bidirectional'
        input_data = np.random.rand(seq_length, batch_size, input_size).astype(np.float32)
        h_data = np.random.rand(num_dir, batch_size, hidden_size).astype(np.float32)
        w_data = np.random.rand(num_dir, 3 * hidden_size, input_size).astype(np.float32)
        r_data = np.random.rand(num_dir, 3 * hidden_size, hidden_size).astype(np.float32)
        b_data = np.random.rand(num_dir, 6 * hidden_size).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, list(input_data.shape))

        output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                               [seq_length, num_dir, batch_size, hidden_size])

        w_value = helper.make_tensor(
            name='w',
            data_type=onnx.TensorProto.FLOAT,
            dims=w_data.shape,
            vals=w_data.flatten(),
        )
        r_value = helper.make_tensor(
            name='r',
            data_type=onnx.TensorProto.FLOAT,
            dims=r_data.shape,
            vals=r_data.flatten(),
        )
        b_value = helper.make_tensor(
            name='b',
            data_type=onnx.TensorProto.FLOAT,
            dims=b_data.shape,
            vals=b_data.flatten(),
        )
        h_value = helper.make_tensor(
            name='h',
            data_type=onnx.TensorProto.FLOAT,
            dims=h_data.shape,
            vals=h_data.flatten(),
        )
        gru_def = helper.make_node(
            "GRU",
            inputs=['input', 'w', 'r', 'b', '', 'h'],
            outputs=['output', ''],
            direction=direction,
            hidden_size=hidden_size,
            linear_before_reset=1,
        )
        graph_def = helper.make_graph([gru_def],
                                      case_name, [input], [output],
                                      initializer=[w_value, r_value, b_value, h_value])
        self.onnx_and_test({"input": input_data}, graph_def)

    def test_GRU2(self, case_name):
        seq_length = 75
        batch_size = 2
        num_dir = 2
        input_size = 128
        hidden_size = 64
        direction = 'forward' if num_dir == 1 else 'bidirectional'
        input_data = np.random.rand(seq_length, batch_size, input_size).astype(np.float32)
        h_data = np.random.rand(num_dir, batch_size, hidden_size).astype(np.float32)
        w_data = np.random.rand(num_dir, 3 * hidden_size, input_size).astype(np.float32)
        r_data = np.random.rand(num_dir, 3 * hidden_size, hidden_size).astype(np.float32)
        b_data = np.random.rand(num_dir, 6 * hidden_size).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, list(input_data.shape))

        output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                               [num_dir, batch_size, hidden_size])

        w_value = helper.make_tensor(
            name='w',
            data_type=onnx.TensorProto.FLOAT,
            dims=w_data.shape,
            vals=w_data.flatten(),
        )
        r_value = helper.make_tensor(
            name='const_tensor',
            data_type=onnx.TensorProto.FLOAT,
            dims=r_data.shape,
            vals=r_data.flatten(),
        )
        b_value = helper.make_tensor(name='const_tensor',
                                     data_type=onnx.TensorProto.FLOAT,
                                     dims=b_data.shape,
                                     vals=b_data.flatten())
        h_value = helper.make_tensor(
            name='const_tensor',
            data_type=onnx.TensorProto.FLOAT,
            dims=h_data.shape,
            vals=h_data.flatten(),
        )
        gru_def = helper.make_node(
            "GRU",
            inputs=['input', 'w', 'r', 'b', '', 'h'],
            outputs=['', 'output'],
            direction=direction,
            hidden_size=hidden_size,
            linear_before_reset=1,
        )
        graph_def = helper.make_graph([gru_def],
                                      case_name, [input], [output],
                                      initializer=[w_value, r_value, b_value, h_value])
        self.onnx_and_test({"input": input_data}, graph_def)

    def test_LSTM2(self, case_name):
        seq_length = 75
        batch_size = 2
        num_dir = 2
        input_size = 128
        hidden_size = 64
        direction = 'forward' if num_dir == 1 else 'bidirectional'
        input_data = np.random.rand(seq_length, batch_size, input_size).astype(np.float32)
        h0_data = np.random.rand(num_dir, batch_size, hidden_size).astype(np.float32)
        c0_data = np.random.rand(num_dir, batch_size, hidden_size).astype(np.float32)
        w_data = np.random.rand(num_dir, 4 * hidden_size, input_size).astype(np.float32)
        r_data = np.random.rand(num_dir, 4 * hidden_size, hidden_size).astype(np.float32)
        b_data = np.random.rand(num_dir, 8 * hidden_size).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, list(input_data.shape))
        h0 = helper.make_tensor_value_info('h0', TensorProto.FLOAT, list(h0_data.shape))
        c0 = helper.make_tensor_value_info('c0', TensorProto.FLOAT, list(c0_data.shape))
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                               [seq_length, num_dir, batch_size, hidden_size])
        Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT,
                                            [num_dir, batch_size, hidden_size])
        Y_c = helper.make_tensor_value_info('Y_c', TensorProto.FLOAT,
                                            [num_dir, batch_size, hidden_size])
        w_value = helper.make_tensor(
            name='w',
            data_type=onnx.TensorProto.FLOAT,
            dims=w_data.shape,
            vals=w_data.flatten(),
        )
        r_value = helper.make_tensor(
            name='r',
            data_type=onnx.TensorProto.FLOAT,
            dims=r_data.shape,
            vals=r_data.flatten(),
        )
        b_value = helper.make_tensor(
            name='b',
            data_type=onnx.TensorProto.FLOAT,
            dims=b_data.shape,
            vals=b_data.flatten(),
        ),
        lstm_def = helper.make_node(
            "LSTM",
            inputs=['input', 'w', 'r', 'b', '', 'h0', 'c0'],
            outputs=['output', 'Y_h', 'Y_c'],
            direction=direction,
            hidden_size=hidden_size,
        )
        graph_def = helper.make_graph([lstm_def],
                                      case_name, [input, h0, c0], [output, Y_h, Y_c],
                                      initializer=[w_value, r_value, b_value])
        inputs = {
            "input": input_data,
            "h0": h0_data,
            "c0": c0_data,
        }
        self.onnx_and_test(inputs, graph_def)

    def test_MaxPool1D(self, case_name):
        self.MaxPoolBase(case_name, [1, 32, 128], [1, 32, 64], [2], [2])

    def test_MaxPool2D(self, case_name):
        self.MaxPoolBase(case_name, [1, 32, 128, 128], [1, 32, 64, 64], [2, 2], [2, 2])

    def test_MaxPool3D(self, case_name):
        self.MaxPoolBase(case_name, [1, 32, 16, 32, 64], [1, 32, 8, 16, 63], [2, 1, 2], [2, 2, 1])

    def ConvBase(self, case_name, input_shape, filter_shape, output_shape, kernel, padding, stride,
                 dilation, groups):
        input_data = np.random.randn(*input_shape).astype(np.float32)
        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        bias_data = np.random.randn(output_shape[1]).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, list(bias_data.shape), bias_data)

        conv_def = helper.make_node(
            "Conv",
            inputs=['input', 'weight', 'bias'],
            outputs=['output'],
            kernel_shape=kernel,
            pads=padding,
            strides=stride,
            dilations=dilation,
            group=groups,
        )

        graph_def = helper.make_graph([conv_def],
                                      case_name, [input], [output],
                                      initializer=[weight, bias])
        self.onnx_and_test({'input': input_data}, graph_def)

    def test_Conv1d(self, case_name):
        oc = 32
        self.ConvBase(case_name, [1, 16, 100], [oc, 16, 3], [1, oc, 100], [3], [1, 1], [1], [1], 1)

    def test_Conv2d(self, case_name):
        oc = 32
        input_shape = [1, 16, 100, 100]
        filter_shape = [oc, 16, 3, 3]
        output_shape = [1, oc, 100, 100]
        self.ConvBase(case_name,
                      input_shape,
                      filter_shape,
                      output_shape,
                      kernel=[3, 3],
                      padding=[1, 1, 1, 1],
                      stride=[1, 1],
                      dilation=[1, 1],
                      groups=1)

    # def test_Conv2d(self, case_name):
    #     input_shape = [4, 3, 10, 10]
    #     filter_shape = [8, 3, 3, 3]
    #     output_shape = [4, 8, 10, 10]
    #     self.ConvBase(case_name,
    #                   input_shape,
    #                   filter_shape,
    #                   output_shape,
    #                   kernel=[3, 3],
    #                   padding=[1, 1, 1, 1],
    #                   stride=[1, 1],
    #                   dilation=[1, 1],
    #                   groups=1)

    def test_Conv3d(self, case_name):
        oc = 32
        input_shape = [1, 16, 10, 30, 50]
        filter_shape = [oc, 16, 3, 3, 3]
        output_shape = [1, oc, 10, 30, 50]
        self.ConvBase(case_name,
                      input_shape,
                      filter_shape,
                      output_shape,
                      kernel=[3, 3, 3],
                      padding=[1, 1, 1, 1, 1, 1],
                      stride=[1, 1, 1],
                      dilation=[1, 1, 1],
                      groups=1)

    def test_SiLU(self, case_name):
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
        graph_def = helper.make_graph([sigmoid_def, mul_def], case_name, [input], [output])
        self.onnx_and_test({'input': input_data}, graph_def)

    def test_Concat(self, case_name):
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

        graph_def = helper.make_graph([concat_def], case_name, inputs, [output])
        self.onnx_and_test(input_data, graph_def)

    def test_Transpose(self, case_name):
        input_shapes = [[1, 16, 32, 32], [4, 3, 85, 20, 20], [1, 4, 2, 16, 20, 40]]
        transpose_orders = {4: [0, 2, 1, 3], 5: [0, 1, 3, 4, 2], 6: [0, 1, 2, 5, 3, 4]}
        for input_shape in input_shapes:
            input_data = np.random.randn(*input_shape).astype(np.float32)
            input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
            order = transpose_orders[len(input_shape)]
            output_shape = [input_shape[order[i]] for i in range(len(order))]
            output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
            transpose_def = helper.make_node("Transpose",
                                             inputs=['input'],
                                             outputs=['output'],
                                             perm=order)
            graph_def = helper.make_graph([transpose_def], case_name, [input], [output])
            self.onnx_and_test({'input': input_data}, graph_def)
            print("[Success] {}D data test".format(len(input_shape)))

    def test_Where(self, case_name):
        # real case
        input_shape = [1, 40, 224]
        output_shape = input_shape
        condition_shape = [1, 40, 224]
        condition_data = np.zeros(condition_shape)
        # select top half
        #array([[[1., 1., 0., 0.],
        #        [1., 1., 0., 0.],
        #        [1., 1., 0., 0.],
        #        [1., 1., 0., 0.]],
        condition_data[:, :, :100] = 1
        const_data = np.array([0.5]).astype(np.float32)

        output_shape = input_shape
        condition_data = condition_data.astype(np.bool)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        condition_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['condition'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.BOOL,
                dims=condition_shape,
                vals=condition_data.flatten(),
            ),
        )
        masked_fill_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['fill'],
            value=onnx.helper.make_tensor(
                name='const_tensor_1',
                data_type=onnx.TensorProto.FLOAT,
                dims=const_data.shape,
                vals=const_data,
            ),
        )
        where_node = helper.make_node(
            'Where',
            ['condition', 'fill', 'input'],
            ['output'],
        )
        graph_def = helper.make_graph([condition_node_def, masked_fill_node_def, where_node],
                                      case_name, [input], [output])
        input_data = np.random.rand(*input_shape).astype(np.float32)
        self.onnx_and_test({"input": input_data}, graph_def)

    def test_LeakyRelu(self, case_name):
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
                                      case_name, [input], [output],
                                      initializer=[weight, bias])
        self.onnx_and_test({'input': input_data}, graph_def)

    def test_Mul(self, case_name):
        input_shape = {"input1": [1, 3, 27, 27], "input2": [1, 3, 27, 27]}
        output_shape = [1, 3, 27, 27]
        input_data = {k: np.random.randn(*x).astype(np.float32) for k, x in input_shape.items()}

        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, x) for k, x in input_shape.items()
        ]
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        mul_def = helper.make_node("Mul", inputs=list(input_shape.keys()), outputs=["output"])

        graph_def = helper.make_graph([mul_def], case_name, inputs, [output])
        self.onnx_and_test(input_data, graph_def)

    def test_MulConst(self, case_name):
        input_shape = [1, 3, 27, 27]
        output_shape = [1, 3, 27, 27]
        input_data = np.random.randn(*input_shape).astype(np.float32)

        input = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)

        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        mul_const = helper.make_tensor(name='const_mul',
                                       data_type=TensorProto.FLOAT,
                                       dims=[],
                                       vals=[2.0])

        const_mul_def = helper.make_node("Mul", inputs=["input", "const_mul"], outputs=["output"])

        graph_def = helper.make_graph([const_mul_def],
                                      case_name, [input], [output],
                                      initializer=[mul_const])
        self.onnx_and_test({"input": input_data}, graph_def)

    def test_Gemm(self, case_name):
        M = 50
        K = 100
        N = 25
        input_shape = [M, K]
        weight_shape = [K, N]
        bias_shape = [N]
        output_shape = [M, N]
        input_data = np.random.randn(*input_shape).astype(np.float32)
        weight_data = np.random.randn(*weight_shape).astype(np.float32)
        bias_data = np.random.randn(*bias_shape).astype(np.float32)
        input = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, weight_shape, weight_data)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, bias_shape, bias_data)
        gemm_def = helper.make_node("Gemm", inputs=["input", "weight", "bias"], outputs=["output"])
        graph_def = helper.make_graph([gemm_def],
                                      case_name, [input], [output],
                                      initializer=[weight, bias])
        self.onnx_and_test({"input": input_data}, graph_def)

    def test_MatMul(self, case_name):
        M = 50
        K = 100
        N = 25
        input_shape = [10, M, K]
        weight_shape = [K, N]
        output_shape = [10, M, N]
        input_data = np.random.randn(*input_shape).astype(np.float32)
        weight_data = np.random.randn(*weight_shape).astype(np.float32)
        input = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, weight_shape, weight_data)
        gemm_def = helper.make_node("MatMul", inputs=["input", "weight"], outputs=["output"])
        graph_def = helper.make_graph([gemm_def],
                                      case_name, [input], [output],
                                      initializer=[weight])
        self.onnx_and_test({"input": input_data}, graph_def)

    def test_Scale(self, case_name):
        input_shape = [1, 32, 100, 100]
        output_shape = [1, 32, 100, 100]
        weight_shape = [1, 32, 1, 1]
        offset_shape = [1, 32, 1, 1]
        input_data = np.random.randn(*input_shape).astype(np.float32)
        weight_data = np.random.randn(*weight_shape).astype(np.float32)
        offset_data = np.random.randn(*offset_shape).astype(np.float32)

        input = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, weight_shape, weight_data)
        offset = helper.make_tensor('offset', TensorProto.FLOAT, offset_shape, offset_data)

        mul_weight_def = helper.make_node("Mul", inputs=["input", "weight"], outputs=["mul_output"])
        add_offset_def = helper.make_node("Add",
                                          inputs=["mul_output", "offset"],
                                          outputs=["output"])

        graph_def = helper.make_graph([mul_weight_def, add_offset_def],
                                      case_name, [input], [output],
                                      initializer=[weight, offset])

        self.onnx_and_test({"input": input_data}, graph_def)

    def test_Resize(self, case_name):
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
                                      case_name, [input], [output],
                                      initializer=[roi, scales])
        input_data = np.random.randn(*input_shape).astype(np.float32)
        self.onnx_and_test({"input": input_data}, graph_def)

    def test_Resize2(self, case_name):
        input_shape = [1, 32, 208, 30]
        #linear
        output_shape1 = [1, 32, 416, 48]  # by cpu, scale is not integer
        output_shape2 = [1, 32, 416, 90]  # by npu, scale is integer
        output_shape3 = [1, 32, 104, 15]  # by npu, scale is 0.5
        #nearest
        output_shape4 = [1, 32, 416, 60]  # by npu, scale is integer
        output_shape5 = [1, 32, 416, 20]  # by cpu

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output1 = helper.make_tensor_value_info('output1', TensorProto.FLOAT, output_shape1)
        output2 = helper.make_tensor_value_info('output2', TensorProto.FLOAT, output_shape2)
        output3 = helper.make_tensor_value_info('output3', TensorProto.FLOAT, output_shape3)
        output4 = helper.make_tensor_value_info('output4', TensorProto.FLOAT, output_shape4)
        output5 = helper.make_tensor_value_info('output5', TensorProto.FLOAT, output_shape5)
        roi = np.array([], dtype=np.float32)
        scales = np.array([], dtype=np.float32)
        roi_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['roi'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=roi.shape,
                vals=roi.flatten(),
            ),
        )
        scales_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['scales'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=scales.shape,
                vals=scales.flatten(),
            ),
        )
        sizes1_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['sizes1'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=[4],
                vals=np.array(output_shape1, dtype=np.int64),
            ),
        )
        x1_node = helper.make_node(
            'Neg',
            ['input'],
            ['X1'],
        )
        resize1_node = helper.make_node('Resize',
                                        inputs=['X1', 'roi', 'scales', 'sizes1'],
                                        outputs=['output1'],
                                        mode='linear',
                                        coordinate_transformation_mode='half_pixel')
        sizes2_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['sizes2'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=[4],
                vals=np.array(output_shape2, dtype=np.int64),
            ),
        )
        resize2_node = helper.make_node('Resize',
                                        inputs=['input', 'roi', 'scales', 'sizes2'],
                                        outputs=['output2'],
                                        mode='linear',
                                        coordinate_transformation_mode='pytorch_half_pixel')
        sizes3_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['sizes3'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=[4],
                vals=np.array(output_shape3, dtype=np.int64),
            ),
        )
        resize3_node = helper.make_node('Resize',
                                        inputs=['input', 'roi', 'scales', 'sizes3'],
                                        outputs=['output3'],
                                        mode='linear',
                                        coordinate_transformation_mode='half_pixel')
        sizes4_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['sizes4'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=[4],
                vals=np.array(output_shape4, dtype=np.int64),
            ),
        )
        resize4_node = helper.make_node(
            'Resize',
            inputs=['input', 'roi', 'scales', 'sizes4'],
            outputs=['output4'],
            mode='nearest',
        )
        sizes5_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['sizes5'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=[4],
                vals=np.array(output_shape5, dtype=np.int64),
            ),
        )
        resize5_node = helper.make_node(
            'Resize',
            inputs=['input', 'roi', 'scales', 'sizes5'],
            outputs=['output5'],
            mode='nearest',
        )
        graph_def = helper.make_graph([
            x1_node, roi_def, scales_def, sizes1_def, resize1_node, sizes2_def, resize2_node,
            sizes3_def, resize3_node, sizes4_def, resize4_node, sizes5_def, resize5_node
        ], case_name, [input], [output1, output2, output3, output4, output5])
        input_data = np.random.rand(*input_shape).astype(np.float32)
        self.onnx_and_test({"input": input_data}, graph_def)

    def test_Reshape(self, case_name):
        input_shape = [1, 16, 32, 32]
        right_shape = [3]
        output_shape = [1, 16, 1024]
        input_data = np.random.randn(*input_shape).astype(np.float32)
        right_data = np.array([1, 16, 1024], dtype=np.int64)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        right = helper.make_tensor('right', TensorProto.INT64, right_shape, right_data)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        reshape_def = helper.make_node(case_name, inputs=['input', 'right'], outputs=['output'])
        graph_def = helper.make_graph([reshape_def],
                                      case_name, [input], [output],
                                      initializer=[right])
        self.onnx_and_test({'input': input_data}, graph_def)

    def test_Softmax(self, case_name):
        input_shape = [1, 1000, 1, 1]
        axis = 1
        input_data = np.random.randn(*input_shape).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
        softmax_def = helper.make_node(case_name, inputs=['input'], outputs=['output'], axis=axis)
        graph_def = helper.make_graph([softmax_def], case_name, [input], [output])
        self.onnx_and_test({'input': input_data}, graph_def)

    def test_Log(self, case_name):
        input_shape = [1, 3, 32, 32]
        input_data = np.clip(np.random.randn(*input_shape).astype(np.float32) * 10.0, 0.5, 8)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
        log_def = helper.make_node(case_name, inputs=['input'], outputs=['output'])
        graph_def = helper.make_graph([log_def], case_name, [input], [output])
        self.onnx_and_test({'input': input_data}, graph_def)

    def test_Neg(self, case_name):
        input_shape = [4, 16, 27, 27]
        output_shape = [4, 16, 27, 27]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        neg_def = helper.make_node(
            'Neg',  # node name
            ['input'],  # inputs
            ['output'],  # outputs
        )
        graph_def = helper.make_graph(
            [neg_def],
            case_name,
            [input],
            [output],
        )
        input_data = np.random.rand(*input_shape).astype(np.float32)
        self.onnx_and_test({"input": input_data}, graph_def)

    def test_Pad0(self, case_name):
        input_shape = [3, 8, 32, 32]
        output_shape = [3, 8, 44, 46]
        pads = np.array([0, 0, 5, 6, 0, 0, 7, 8]).astype(np.int64)
        input_data = np.random.randn(*input_shape).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        pad_val = helper.make_tensor(name='pads',
                                     data_type=onnx.TensorProto.INT64,
                                     dims=pads.shape,
                                     vals=pads.flatten())
        pad_def = helper.make_node("Pad", ['input', 'pads'], outputs=['output'], mode='constant')
        graph_def = helper.make_graph([pad_def],
                                      case_name, [input], [output],
                                      initializer=[pad_val])
        self.onnx_and_test({'input': input_data}, graph_def)

    def test_Pad1(self, case_name):
        input_shape = [3, 8, 32, 32]
        output_shape = [3, 8, 44, 46]
        pads = np.array([0, 0, 5, 6, 0, 0, 7, 8]).astype(np.int64)
        input_data = np.random.randn(*input_shape).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        pad_shape = helper.make_tensor(name='pads',
                                       data_type=onnx.TensorProto.INT64,
                                       dims=pads.shape,
                                       vals=pads.flatten())
        pad_val = helper.make_tensor(name='pad_val',
                                     data_type=onnx.TensorProto.FLOAT,
                                     dims=[],
                                     vals=[0.6])
        pad_def = helper.make_node("Pad", ['input', 'pads', 'pad_val'],
                                   outputs=['output'],
                                   mode='constant')
        graph_def = helper.make_graph([pad_def],
                                      case_name, [input], [output],
                                      initializer=[pad_shape, pad_val])
        self.onnx_and_test({'input': input_data}, graph_def)

    def test_PadEdge(self, case_name):
        input_shape = [3, 8, 32, 32]
        output_shape = [3, 8, 44, 46]
        pads = np.array([0, 0, 5, 6, 0, 0, 7, 8]).astype(np.int64)
        input_data = np.random.randn(*input_shape).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        pad_val = helper.make_tensor(name='pads',
                                     data_type=onnx.TensorProto.INT64,
                                     dims=pads.shape,
                                     vals=pads.flatten())
        pad_def = helper.make_node("Pad", ['input', 'pads'], outputs=['output'], mode='edge')
        graph_def = helper.make_graph([pad_def],
                                      case_name, [input], [output],
                                      initializer=[pad_val])
        self.onnx_and_test({'input': input_data}, graph_def)

    def test_PadReflect(self, case_name):
        input_shape = [3, 8, 32, 32]
        output_shape = [3, 8, 44, 46]
        pads = np.array([0, 0, 5, 6, 0, 0, 7, 8]).astype(np.int64)
        input_data = np.random.randn(*input_shape).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        pad_val = helper.make_tensor(name='pads',
                                     data_type=onnx.TensorProto.INT64,
                                     dims=pads.shape,
                                     vals=pads.flatten())
        pad_def = helper.make_node("Pad", ['input', 'pads'], outputs=['output'], mode='reflect')
        graph_def = helper.make_graph([pad_def],
                                      case_name, [input], [output],
                                      initializer=[pad_val])
        self.onnx_and_test({'input': input_data}, graph_def)

    def test_DepthToSpace(self, case_name):
        in_shape = [1, 32, 108, 192]
        n, c, h, w = in_shape
        blocksize = 2
        # mode='CRD'
        mode = 'DCR'  # default
        out_shape = [n, c // (blocksize * blocksize), h * blocksize, w * blocksize]
        input_data = np.arange(np.prod(in_shape)).reshape(in_shape).astype(np.float32)
        input_data = np.random.rand(*in_shape).reshape(in_shape).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, list(input_data.shape))
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, out_shape)
        node_def = helper.make_node(
            "DepthToSpace",
            mode=mode,
            blocksize=blocksize,
            inputs=['input'],
            outputs=['output'],
        )
        graph_def = helper.make_graph(
            [node_def],
            case_name,
            [input],
            [output],
        )
        self.onnx_and_test({"input": input_data}, graph_def)

    def test_Div(self, case_name):
        input_shape = {"input1": [1, 3, 27, 27], "input2": [1, 3, 27, 27]}
        output_shape = [1, 3, 27, 27]
        input_data = {k: np.random.randn(*x).astype(np.float32) for k, x in input_shape.items()}
        input_data["input2"] = np.clip(input_data["input2"], 0.01, 10)

        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, x) for k, x in input_shape.items()
        ]
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        div_def = helper.make_node("Div", inputs=list(input_shape.keys()), outputs=["output"])

        graph_def = helper.make_graph([div_def], case_name, inputs, [output])
        self.onnx_and_test(input_data, graph_def)

    def test_ConvTranspose(self, case_name):
        oc, ic = 16, 8
        ih, iw = 16, 16
        kernel_shape = [3, 3]
        pads = [1, 1, 1, 1]
        strides = [2, 2]
        dilations = [1, 1]
        oh = (ih - 1) * strides[0] + (kernel_shape[0] - 1) * dilations[0] + 1 - pads[0] - pads[2]
        ow = (iw - 1) * strides[1] + (kernel_shape[1] - 1) * dilations[1] + 1 - pads[1] - pads[3]
        input_shape = [1, ic, ih, iw]
        output_shape = [1, oc, oh, ow]
        filter_shape = [ic, oc, kernel_shape[0], kernel_shape[1]]

        input_data = np.random.randn(*input_shape).astype(np.float32)
        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        bias_data = np.random.randn(oc).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, list(bias_data.shape), bias_data)
        convtranspose_def = helper.make_node("ConvTranspose",
                                             inputs=['input', 'weight', 'bias'],
                                             outputs=['output'],
                                             kernel_shape=kernel_shape,
                                             pads=pads,
                                             strides=strides,
                                             dilations=dilations,
                                             group=1)
        graph_def = helper.make_graph([convtranspose_def],
                                      case_name, [input], [output],
                                      initializer=[weight, bias])
        self.onnx_and_test({'input': input_data}, graph_def)

    def test_Squeeze(self, case_name):
        axis = [1, 3]
        input_shape = [3, 1, 32, 1]
        output_shape = [input_shape[i] for i in range(len(input_shape)) if i not in axis]
        input_data0 = np.random.randn(*input_shape).astype(np.float32)
        input_data1 = np.random.randn(*input_shape).astype(np.float32)
        input0 = helper.make_tensor_value_info('input0', TensorProto.FLOAT, input_shape)
        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, input_shape)
        axes = helper.make_tensor('axes', TensorProto.INT64, [len(axis)],
                                  axis * np.ones(1).astype(int))
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        add_def = helper.make_node("Add", inputs=['input0', 'input1'], outputs=['x'])
        squeeze_def = helper.make_node("Squeeze", inputs=['x', 'axes'], outputs=['output'])
        graph_def = helper.make_graph([add_def, squeeze_def],
                                      case_name, [input0, input1], [output],
                                      initializer=[axes])
        self.onnx_and_test({'input0': input_data0, 'input1': input_data1}, graph_def)

    def test_Clip(self, case_name):
        input_shape = [1, 3, 32, 32]
        input_data = np.random.randn(*input_shape).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        min = helper.make_tensor('min', TensorProto.FLOAT, [], 0.0 * np.ones(1))
        max = helper.make_tensor('max', TensorProto.FLOAT, [], 6.0 * np.ones(1))
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
        node_def = helper.make_node(case_name, inputs=['input', 'min', 'max'], outputs=['output'])
        graph_def = helper.make_graph([node_def], case_name, [input], [output], [min, max])
        self.onnx_and_test({'input': input_data}, graph_def)

    def test_ReduceL2(self, case_name):
        input_shape = [4, 4, 4, 16, 16, 64]
        output_shape = [4, 4, 4, 64]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('o_l2', TensorProto.FLOAT, output_shape)

        reduce_l2 = helper.make_node(
            'ReduceL2',
            ['input'],
            ['o_l2'],
            keepdims=0,
            axes=[3, 4],
        )

        graph_def = helper.make_graph([reduce_l2], case_name, [input], [output])
        input_data = np.random.rand(*input_shape).astype(np.float32)
        self.onnx_and_test({"input": input_data}, graph_def)

    def test_Sigmoid(self, case_name):
        input_shape = [1, 16, 64, 64]
        output_shape = [1, 16, 64, 64]
        input_data = np.random.randn(*input_shape).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        sigmoid_def = helper.make_node(
            case_name,
            inputs=['input'],
            outputs=['output'],
        )
        graph_def = helper.make_graph([sigmoid_def], case_name, [input], [output])
        self.onnx_and_test({'input': input_data}, graph_def)

    def test_Slice(self, case_name):
        input_shape = [5, 116, 64, 64]
        output_shape = [3, 16, 16, 8]
        input_data = np.random.randn(*input_shape).astype(np.float32)
        starts_data = np.array([2, 10, 10, 12], dtype=np.int64)
        ends_data = np.array([5, 42, 42, 36], dtype=np.int64)
        axes_data = np.array([0, 1, 2, 3], dtype=np.int64)
        steps_data = np.array([1, 2, 2, 3], dtype=np.int64)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        starts = helper.make_tensor('starts', TensorProto.INT64, [4], starts_data)
        ends = helper.make_tensor('ends', TensorProto.INT64, [4], ends_data)
        axes = helper.make_tensor('axes', TensorProto.INT64, [4], axes_data)
        steps = helper.make_tensor('steps', TensorProto.INT64, [4], steps_data)
        slice_def = helper.make_node(
            case_name,
            inputs=['input', 'starts', 'ends', 'axes', 'steps'],
            outputs=['output'],
        )

        graph_def = helper.make_graph([slice_def],
                                      case_name, [input], [output],
                                      initializer=[starts, ends, axes, steps])
        self.onnx_and_test({'input': input_data}, graph_def)

    def test_Split(self, case_name):
        input_shape = [6, 116, 64, 64]
        output1_shape = [3, 116, 64, 64]
        output2_shape = [3, 116, 64, 64]
        input_data = np.random.randn(*input_shape).astype(np.float32)
        split_data = np.array([3, 3], dtype=np.int64)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        split = helper.make_tensor('split', TensorProto.INT64, [2], split_data)
        output_1 = helper.make_tensor_value_info('output_1', TensorProto.FLOAT, output1_shape)
        output_2 = helper.make_tensor_value_info('output_2', TensorProto.FLOAT, output2_shape)
        split_def = helper.make_node(
            "Split",
            inputs=['input', 'split'],
            outputs=['output_1', 'output_2'],
            axis=0,
        )

        graph_def = helper.make_graph([split_def],
                                      case_name, [input], [output_1, output_2],
                                      initializer=[split])
        self.onnx_and_test({'input': input_data}, graph_def)

    def test_Reduce(self, case_name):
        input_shape = [4, 4, 4, 16, 64]
        output_shape = [4, 4, 4, 16]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output0 = helper.make_tensor_value_info('o_mean', TensorProto.FLOAT, output_shape)
        output1 = helper.make_tensor_value_info('o_max', TensorProto.FLOAT, output_shape)
        output2 = helper.make_tensor_value_info('o_min', TensorProto.FLOAT, output_shape)
        reduce_mean = helper.make_node(
            'ReduceMean',
            ['input'],
            ['o_mean'],
            keepdims=0,
            axes=[4],
        )
        reduce_max = helper.make_node(
            'ReduceMax',
            ['input'],
            ['o_max'],
            keepdims=0,
            axes=[4],
        )
        reduce_min = helper.make_node(
            'ReduceMin',
            ['input'],
            ['o_min'],
            keepdims=0,
            axes=[4],
        )

        graph_def = helper.make_graph([reduce_mean, reduce_max, reduce_min], case_name, [input],
                                      [output0, output1, output2])
        input_data = np.random.randn(*input_shape).astype(np.float32)
        self.onnx_and_test(input_data, graph_def)

    def test_Reduce2(self, case_name):
        input_shape = [4, 4, 4, 16, 64]
        output_shape = [4, 4, 1, 1, 64]
        output_shape2 = [4, 4, 64]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output0 = helper.make_tensor_value_info('o_mean', TensorProto.FLOAT, output_shape)
        output1 = helper.make_tensor_value_info('o_max', TensorProto.FLOAT, output_shape)
        output2 = helper.make_tensor_value_info('o_min', TensorProto.FLOAT, output_shape)
        reduce_mean = helper.make_node(
            'ReduceMean',
            ['input'],
            ['o_mean'],
            keepdims=1,
            axes=[2, 3],
        )
        reduce_max = helper.make_node(
            'ReduceMax',
            ['input'],
            ['o_max'],
            keepdims=1,
            axes=[2, 3],
        )
        reduce_min = helper.make_node(
            'ReduceMin',
            ['input'],
            ['o_min'],
            keepdims=1,
            axes=[2, 3],
        )

        graph_def = helper.make_graph([reduce_mean, reduce_max, reduce_min], case_name, [input],
                                      [output0, output1, output2])
        input_data = np.random.randn(*input_shape).astype(np.float32)
        self.onnx_and_test({"input": input_data}, graph_def)

    def test_ReduceMean(self, case_name):
        input_shape = [2, 200, 7, 7]
        output_shape = [2, 1, 1, 7]
        input_data = np.random.randn(*input_shape).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        reducemean_def = helper.make_node(
            "ReduceMean",
            inputs=['input'],
            outputs=['output'],
            axes=[1, 2],
            keepdims=1,
        )

        graph_def = helper.make_graph([reducemean_def], case_name, [input], [output])
        self.onnx_and_test({'input': input_data}, graph_def)

    def test_TorchLayerGroup(self, case_name):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.m1 = nn.Conv2d(3, 8, 3, 1, 0)
                self.m2 = nn.Conv2d(8, 8, 3, 1, 1)

            def forward(self, x):
                y0 = self.m1(x)
                y1 = self.m2(y0)
                y2 = y0 + y1
                return y0, y2

        x = torch.randn(4, 3, 50, 50).float()
        self.torch_and_test(x, Model(), case_name)

    def test_TorchLogSoftmax(self, case_name):

        class Net(torch.nn.Module):

            def __init__(self):
                super(Net, self).__init__()
                self.linear = nn.Linear(32, 72, bias=False)
                self.act = nn.LogSoftmax(dim=2)

            def forward(self, x):
                x = self.linear(x)
                x = self.act(x)
                return x

        input_shape = [3, 100, 32]
        input_data = torch.randn(input_shape)
        self.torch_and_test(input_data, Net(), case_name)

    def test_TorchLayerNorm(self, case_name):

        class Net(torch.nn.Module):

            def __init__(self):
                super(Net, self).__init__()
                self.layer_norm = nn.LayerNorm([24, 50])

            def forward(self, x):
                x = self.layer_norm(x)
                return x

        input_shape = [3, 24, 50]
        input_data = torch.randn(input_shape)
        self.torch_and_test(input_data, Net(), case_name)

    def test_TorchMaskedFill(self, case_name):

        class Net(nn.Module):

            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                y = x.masked_fill(x == 0, value=torch.tensor(-50.0))
                z = x.masked_fill(x != 0, value=torch.tensor(1.0))
                return y + z

        input_shape = [2, 3, 100]
        input_data = torch.randint(0, 1000, input_shape)
        input_data[:, :, 70:] = 0
        self.torch_and_test(input_data, Net(), case_name)

    def test_TorchWhere(self, case_name):

        class Net(torch.nn.Module):

            def __init__(self):
                super(Net, self).__init__()

            def forward(self, mask, a, b):
                x = torch.where(mask.bool(), a, b)
                return x

        a = torch.randn(4, 3, 100, 100).float()
        b = torch.randn(4, 3, 100, 100).float()
        mask = (a > 0).float()
        self.torch_and_test((mask, a, b), Net(), case_name)

    def test_Add(self, case_name):
        input_shape = {"input1": [1, 3, 27, 27], "input2": [1, 3, 27, 27]}
        output_shape = [1, 3, 27, 27]
        input_data = {k: np.random.randn(*x).astype(np.float32) for k, x in input_shape.items()}

        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, x) for k, x in input_shape.items()
        ]
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        add_def = helper.make_node("Add", inputs=list(input_shape.keys()), outputs=["output"])

        graph_def = helper.make_graph([add_def], case_name, inputs, [output])
        self.onnx_and_test(input_data, graph_def)

    def test_AddConst(self, case_name):
        input_shape = [1, 16, 28, 28]
        output_shape = [1, 16, 28, 28]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        w_data = np.random.rand(*input_shape).astype(np.float32)
        w_value = helper.make_tensor(
            name='w',
            data_type=onnx.TensorProto.FLOAT,
            dims=w_data.shape,
            vals=w_data.flatten(),
        )

        add_node = helper.make_node(
            'Add',  # node name
            ['input', 'w'],  # inputs
            ['output'],  # outputs
        )
        graph_def = helper.make_graph([add_node],
                                      case_name, [input], [output],
                                      initializer=[w_value])
        input_data = np.random.rand(*input_shape).astype(np.float32)
        self.onnx_and_test({"input": input_data}, graph_def)

    def test_BroadcastAdd(self, case_name):
        input_shape = {"input1": [1, 3, 1, 27], "input2": [2, 1, 27, 1]}
        output_shape = [2, 3, 27, 27]
        input_data = {k: np.random.randn(*x).astype(np.float32) for k, x in input_shape.items()}
        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, x) for k, x in input_shape.items()
        ]
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
        add_def = helper.make_node("Add", inputs=list(input_shape.keys()), outputs=["output"])
        graph_def = helper.make_graph([add_def], case_name, inputs, [output])
        self.onnx_and_test(input_data, graph_def)

    def test_LRN(self, case_name):
        input_shape = [4, 10, 27, 27]
        output_shape = [4, 10, 27, 27]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        lrn_def = helper.make_node(
            'LRN',  # node name
            ['input'],  # inputs
            ['output'],  # outputs
            size=5,
        )
        graph_def = helper.make_graph(
            [lrn_def],
            case_name,
            [input],
            [output],
        )
        input_data = np.random.rand(*input_shape).astype(np.float32)
        self.onnx_and_test({"input": input_data}, graph_def)

    def test_LSTM(self, case_name):
        seq_length = 75
        batch_size = 2
        num_dir = 2
        input_size = 128
        hidden_size = 64
        direction = 'forward' if num_dir == 1 else 'bidirectional'
        #layout = 0
        np.random.seed(0)
        input_data = np.random.rand(seq_length, batch_size, input_size).astype(np.float32)
        w_data = np.random.rand(num_dir, 4 * hidden_size, input_size).astype(np.float32)
        r_data = np.random.rand(num_dir, 4 * hidden_size, hidden_size).astype(np.float32)
        b_data = np.random.rand(num_dir, 8 * hidden_size).astype(np.float32)
        h0_data = np.random.rand(num_dir, batch_size, hidden_size).astype(np.float32)
        c0_data = np.random.rand(num_dir, batch_size, hidden_size).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, list(input_data.shape))

        output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                               [seq_length, num_dir, batch_size, hidden_size])

        w = helper.make_tensor('w', TensorProto.FLOAT, w_data.shape, w_data)
        r = helper.make_tensor('r', TensorProto.FLOAT, r_data.shape, r_data)
        b = helper.make_tensor('b', TensorProto.FLOAT, b_data.shape, b_data)
        h0 = helper.make_tensor('h0', TensorProto.FLOAT, h0_data.shape, h0_data)
        c0 = helper.make_tensor('c0', TensorProto.FLOAT, c0_data.shape, c0_data)
        sequence_lens = helper.make_tensor('sequence_lens',
                                           TensorProto.FLOAT,
                                           dims=[],
                                           vals=[seq_length])

        node_def = helper.make_node(
            "LSTM",
            inputs=['input', 'w', 'r', 'b', '', 'h0', 'c0'],
            outputs=['output'],
            direction=direction,
            hidden_size=hidden_size,
        )
        graph_def = helper.make_graph([node_def],
                                      case_name, [input], [output],
                                      initializer=[w, r, b, sequence_lens, h0, c0])
        self.onnx_and_test({'input': input_data}, graph_def)

    def test_BroadcastMul(self, case_name):
        input_shape = {"input1": [1, 3, 1, 27], "input2": [2, 1, 27, 1]}
        output_shape = [2, 3, 27, 27]
        input_data = {k: np.random.randn(*x).astype(np.float32) for k, x in input_shape.items()}
        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, x) for k, x in input_shape.items()
        ]
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
        mul_def = helper.make_node("Mul", inputs=list(input_shape.keys()), outputs=["output"])
        graph_def = helper.make_graph([mul_def], case_name, inputs, [output])
        self.onnx_and_test(input_data, graph_def)

    def test_BroadcastMulConst(self, case_name):
        input_shape = [1, 127, 270, 28]
        constant_shape = [2, 1, 1, 28]
        output_shape = [2, 127, 270, 28]

        input_data = {"input": np.random.randn(*input_shape).astype(np.float32)}
        inputs = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        constant = helper.make_tensor(
            "constant",
            TensorProto.FLOAT,
            constant_shape,
            np.random.randn(*constant_shape).astype(np.float32),
        )
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
        sigmoid_def = helper.make_node(
            "Sigmoid",
            inputs=['input'],
            outputs=['sigmoid'],
        )
        mul_def = helper.make_node("Mul", inputs=["sigmoid", "constant"], outputs=["output"])

        graph_def = helper.make_graph([sigmoid_def, mul_def],
                                      case_name, [inputs], [output],
                                      initializer=[constant])
        self.onnx_and_test(input_data, graph_def)

    def test_Gather(self, case_name):
        total_tokens = 60004
        token_shape = [total_tokens, 256]
        input_shape = [1, 13]
        output_shape = [1, 13, 256]
        input_data = {
            "input1": np.random.randint(0, total_tokens, input_shape).astype(np.int64),
            "input2": np.random.rand(*output_shape).astype(np.float32)
        }

        input1 = helper.make_tensor_value_info('input1', TensorProto.INT64, input_shape)
        input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, output_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        token_data = helper.make_tensor(
            name='tokens',
            data_type=onnx.TensorProto.FLOAT,
            dims=token_shape,
            vals=np.random.randn(*token_shape).astype(np.float32).flatten(),
        )

        gather_node = helper.make_node(
            'Gather',  # node name
            ['tokens', 'input1'],  # inputs
            ['x1'],  # outputs
        )

        add_node = helper.make_node(
            'Add',
            ['x1', 'input2'],
            ['output'],
        )

        graph_def = helper.make_graph([gather_node, add_node],
                                      case_name, [input1, input2], [output],
                                      initializer=[token_data])
        self.onnx_and_test(input_data, graph_def)

    def test_Sum(self, case_name):
        input_shape = [1, 3, 27, 27]
        output_shape = [1, 3, 27, 27]

        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, input_shape)
        input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        #test three input
        sum_def = helper.make_node(
            'Sum',  # node name
            ['input1', 'input2'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [sum_def],
            case_name,
            [input1, input2],
            [output],
        )
        input_data1 = np.random.rand(*input_shape).astype(np.float32)
        input_data2 = np.random.rand(*input_shape).astype(np.float32)
        self.onnx_and_test({"input1": input_data1, "input2": input_data2}, graph_def)

    def test_Tile(self, case_name):
        input_shape = [1, 4, 6, 8]
        output_shape = [1, 24, 24, 16]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        tiles = helper.make_tensor(
            name='tiles',
            data_type=onnx.TensorProto.INT64,
            dims=[4],
            vals=np.array([1, 6, 4, 2]),
        )
        tile_node = helper.make_node(
            'Tile',
            ['input', 'tiles'],
            ['output'],
        )
        graph_def = helper.make_graph([tile_node],
                                      case_name, [input], [output],
                                      initializer=[tiles])
        input_data = {'input': np.random.rand(*input_shape).astype(np.float32)}
        self.onnx_and_test(input_data, graph_def)

    def test_Sub(self, case_name):
        input_shape = [4, 3, 27, 27]
        output_shape = [4, 3, 27, 27]

        input0 = helper.make_tensor_value_info('input0', TensorProto.FLOAT, input_shape)
        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        sub_def = helper.make_node(
            'Sub',  # node name
            ['input0', 'input1'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [sub_def],
            case_name,
            [input0, input1],
            [output],
        )
        inputs = {}
        inputs['input0'] = np.random.randn(*input_shape).astype(np.float32)
        inputs['input1'] = np.random.randn(*input_shape).astype(np.float32)
        self.onnx_and_test(inputs, graph_def)

    def test_Sub2(self, case_name):
        input1_shape = [56, 27]
        input2_shape = [56, 1]
        output_shape = [56, 27]

        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, input1_shape)
        input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, input2_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        sub_def = helper.make_node(
            'Sub',  # node name
            ['input1', 'input2'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [sub_def],
            case_name,
            [input1, input2],
            [output],
        )

        input1_data = np.random.randn(*input1_shape).astype(np.float32)
        input2_data = np.random.randn(*input2_shape).astype(np.float32)
        inputs = {
            "input1": input1_data,
            "input2": input2_data,
        }
        self.onnx_and_test(inputs, graph_def)

    def test_Expand(self, case_name):
        input_shape = [1, 3, 1, 16]
        output_shape = [1, 3, 16, 16]
        input_data = np.random.randn(*input_shape).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        shape_def = helper.make_tensor(
            name='new_shape',
            data_type=onnx.TensorProto.INT64,
            dims=[len(output_shape)],
            vals=np.array(output_shape, dtype=np.int64),
        )
        expand_node = helper.make_node(
            'Expand',  # node name
            ['input', 'new_shape'],  # inputs
            ['output'],  # outputs
        )
        graph_def = helper.make_graph([expand_node],
                                      case_name, [input], [output],
                                      initializer=[shape_def])
        self.onnx_and_test({'input': input_data}, graph_def)

    def test_Expand2(self, case_name):
        input_shape = [1, 16]
        output_shape = [1, 16, 16]
        input_data = np.random.randn(*input_shape).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        shape_def = helper.make_tensor(
            name='new_shape',
            data_type=onnx.TensorProto.INT64,
            dims=[len(output_shape)],
            vals=np.array(output_shape, dtype=np.int64),
        )
        expand_node = helper.make_node(
            'Expand',  # node name
            ['input', 'new_shape'],  # inputs
            ['output'],  # outputs
        )
        graph_def = helper.make_graph([expand_node],
                                      case_name, [input], [output],
                                      initializer=[shape_def])
        self.onnx_and_test({'input': input_data}, graph_def)

    def test_GatherToSlice(self, case_name):
        input_shape = {"input": [1, 32, 27, 27], "indices": [5]}
        output_shape = [1, 5, 27, 27]
        input_data = {"input": np.random.randn(*input_shape['input']).astype(np.float32)}

        input = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape['input'])
        indices = helper.make_tensor("indices", TensorProto.INT64, input_shape['indices'],
                                     np.arange(2, 15, 3).astype(np.int64))
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        gather_def = helper.make_node("Gather",
                                      inputs=list(input_shape.keys()),
                                      outputs=["output"],
                                      axis=1)

        graph_def = helper.make_graph([gather_def],
                                      case_name, [input], [output],
                                      initializer=[indices])
        self.onnx_and_test(input_data, graph_def)

    def test_Max(self, case_name):
        input_shape = {"input1": [1, 85, 32, 8], "input2": [1, 85, 32, 8]}
        output_shape = [1, 85, 32, 8]
        input_data = {k: np.random.randn(*x).astype(np.float32) for k, x in input_shape.items()}

        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, x) for k, x in input_shape.items()
        ]
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        max_def = helper.make_node(case_name, inputs=list(input_shape.keys()), outputs=["output"])

        graph_def = helper.make_graph([max_def], case_name, inputs, [output])
        self.onnx_and_test(input_data, graph_def)

    def test_Min(self, case_name):
        input_shape = {"input1": [1, 85, 32, 8], "input2": [1, 85, 32, 8]}
        output_shape = [1, 85, 32, 8]
        input_data = {k: np.random.randn(*x).astype(np.float32) for k, x in input_shape.items()}

        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, x) for k, x in input_shape.items()
        ]
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        min_def = helper.make_node(case_name, inputs=list(input_shape.keys()), outputs=["output"])

        graph_def = helper.make_graph([min_def], case_name, inputs, [output])
        self.onnx_and_test(input_data, graph_def)

    def test_Abs(self, case_name):
        input_shape = [1, 16, 64, 64]
        output_shape = [1, 16, 64, 64]
        input_data = np.random.randn(*input_shape).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        abs_def = helper.make_node(
            "Abs",
            inputs=['input'],
            outputs=['output'],
        )
        graph_def = helper.make_graph([abs_def], case_name, [input], [output])
        self.onnx_and_test({'input': input_data}, graph_def)

    def test_Reciprocal(self, case_name):
        input_shape = [4, 3, 224, 224]
        node_def = helper.make_node(
            "Reciprocal",  # node name
            ['input'],  # inputs
            ['X1'],  # outputs
        )
        neg_def = helper.make_node(
            "Neg",  # node name
            ['X1'],  # inputs
            ['output'],  # outputs
        )
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def, neg_def],
            case_name,
            [input],
            [output],
        )
        input_data = np.random.randn(*input_shape).astype(np.float32)
        # avoid divide 0
        input_data[input_data == 0] = 1
        self.onnx_and_test({"input": input_data}, graph_def)

    def test_PRelu(self, case_name):
        input_shape = [3, 5, 100, 100]
        slope_shape = [1, 5, 1, 1]
        output_shape = [3, 5, 100, 100]
        input = np.random.randn(*input_shape).astype(np.float32)
        slope = helper.make_tensor(name='slope',
                                   data_type=onnx.TensorProto.FLOAT,
                                   dims=slope_shape,
                                   vals=[0.40, 0.37, 0.31, 0.19, 0.11])

        inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)]
        outputs = [helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)]
        prelu_def = helper.make_node("PRelu", ["input", "slope"], ["output"])
        graph_def = helper.make_graph([prelu_def], case_name, inputs, outputs, initializer=[slope])
        self.onnx_and_test({"input": input}, graph_def)


if __name__ == "__main__":
    tester = ONNX_IR_TESTER()
    os.makedirs("onnx_test", exist_ok=True)
    os.chdir("onnx_test")
    if len(sys.argv) == 2:
        tester.test_single(sys.argv[1])
    else:
        tester.test_all()
