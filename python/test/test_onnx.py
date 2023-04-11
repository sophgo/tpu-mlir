#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from copy import deepcopy
import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto
from tools.model_runner import mlir_inference, model_inference, onnx_inference, show_fake_cmd
from tools.npz_tool import npz_compare
from tools.model_transform import *
from utils.auto_remove import file_mark, file_clean
from utils.mlir_shell import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import onnxruntime


class ONNX_IR_TESTER(object):
    # This class is built for testing single operator transform.
    def __init__(self,
                 chip: str = "bm1684x",
                 mode: str = "all",
                 dynamic: bool = False,
                 simple: bool = False,
                 disable_thread: bool = False):
        Y, N = True, False
        # yapf: disable
        self.test_cases = {
            #########################################
            # ONNX Test Case, Alphabetically
            #########################################
            # case: (test, bm1684x_support, bm1686_support, cv183x_support)
            "Abs":          (self.test_Abs,         Y, N, Y),
            "Add":          (self.test_Add,         Y, N, Y),
            "AddBcast":     (self.test_AddBcast,    Y, N, N),
            "AddBcast2":    (self.test_AddBcast2,   Y, N, N),
            "AddBcast3":    (self.test_AddBcast3,   N, N, N),  # failed cases
            "Arg":          (self.test_Arg,         Y, N, Y),
            "AddConst":     (self.test_AddConst,    Y, N, Y),
            "AvgPool1d":    (self.test_AvgPool1d,   Y, N, Y),
            "AvgPool2d":    (self.test_AvgPool2d,   Y, N, Y),
            "AvgPool3d":    (self.test_AvgPool3d,   Y, N, Y),
            "AvgPoolOdd":   (self.test_AvgPoolOdd,  Y, N, Y),
            "PadAvgPool2d": (self.test_PadAvgPool2d,Y, N, Y),
            "BatchMatMul":  (self.test_BatchMatMul, Y, N, Y),
            "BCastAdd":     (self.test_BCastAdd,    Y, N, Y),
            "BCastMul":     (self.test_BCastMul,    Y, N, Y),
            "BCastMulCst":  (self.test_BCastMulCst, Y, N, Y),
            "CompareCst":   (self.test_CompareCst,  Y, N, N),
            "Compare":      (self.test_Compare,     Y, N, N),
            "Compare2":     (self.test_Compare2,    N, N, N),
            "Concat":       (self.test_Concat,      Y, N, Y),
            "Concat2":      (self.test_Concat2,     Y, N, Y),
            "Conv1d":       (self.test_Conv1d,      Y, N, Y),
            "Conv2d":       (self.test_Conv2d,      Y, N, Y),
            "Conv3d":       (self.test_Conv3d,      Y, N, Y),
            "ConvStride":   (self.test_ConvStride,  Y, N, Y),
            "ConvDw":       (self.test_ConvDw,      Y, N, Y),
            "ConvTrans":    (self.test_ConvTrans,   Y, N, Y),
            "ConvTrans2":   (self.test_ConvTrans2,  Y, N, Y),  #no pad
            "Clip":         (self.test_Clip,        Y, N, Y),
            "DepthToSpace": (self.test_DepthToSpace,Y, N, Y),
            "Deconv":       (self.test_Deconv,      N, Y, N),
            "Div":          (self.test_Div,         Y, N, Y),
            "DivBcast":     (self.test_DivBcast,    Y, N, N),
            "DivBcast2":    (self.test_DivBcast2,   Y, N, N),
            "Einsum":       (self.test_Einsum,      Y, N, Y),
            "Einsum2":      (self.test_Einsum2,     Y, N, Y),
            "Elu":          (self.test_Elu,         Y, N, N),
            "Erf":          (self.test_Erf,         Y, N, N),
            "Exp":          (self.test_Exp,         Y, N, Y),
            "Expand":       (self.test_Expand,      Y, N, Y),
            "Expand2":      (self.test_Expand2,     Y, N, Y),
            "Floor":        (self.test_floor,       Y, N, N),
            "Gather":       (self.test_Gather,      Y, N, Y),
            "Gemm":         (self.test_Gemm,        Y, N, Y),
            "GroupFC":      (self.test_GroupFC,     Y, N, Y),
            "GRU":          (self.test_GRU,         Y, N, Y),  # test gru output Y
            "GRU2":         (self.test_GRU2,        Y, N, Y),  # test gru output Yh
            "GRU3":         (self.test_GRU3,        Y, N, Y),  # test gru output Y and Yh
            "LeakyRelu":    (self.test_LeakyRelu,   Y, N, Y),
            "Log":          (self.test_Log,         Y, N, Y),
            "LogSoftmax":   (self.test_LogSoftmax,  Y, N, Y),
            "LRN":          (self.test_LRN,         Y, N, Y),
            "LSTM":         (self.test_LSTM,        Y, N, Y),  # output_y
            "LSTM2":        (self.test_LSTM2,       Y, N, Y),  # output all
            "LSTM3":        (self.test_LSTM3,       Y, N, Y),  # output_yh and output_yc
            "MaxPool1d":    (self.test_MaxPool1d,   Y, N, Y),
            "MaxPool2d":    (self.test_MaxPool2d,   Y, N, Y),
            "MaxPool3d":    (self.test_MaxPool3d,   N, N, Y),
            "MatMul":       (self.test_MatMul,      Y, N, Y),
            "MatMul2":      (self.test_MatMul2,     Y, N, Y),
            "Max":          (self.test_Max,         Y, N, Y),
            "MaxBcast":     (self.test_MaxBcast,    Y, N, N),
            "Mul":          (self.test_Mul,         Y, N, Y),
            "MulBcast":     (self.test_MulBcast,    Y, N, N),
            "MulBcast2":    (self.test_MulBcast2,   Y, N, N),
            "Min":          (self.test_Min,         Y, N, Y),
            "MinBcast":     (self.test_MinBcast,    Y, N, N),
            "MulConst":     (self.test_MulConst,    Y, N, Y),
            "Neg":          (self.test_Neg,         Y, N, Y),
            "Pad":          (self.test_Pad,         Y, N, Y),  # zero pad
            "Pad1":         (self.test_Pad1,        Y, N, Y),  # pad val
            "PadEdge":      (self.test_PadEdge,     Y, N, Y),
            "PadReflect":   (self.test_PadReflect,  Y, N, Y),
            "Pow1":         (self.test_Pow1,        Y, N, Y),  # y = x ^ n
            "Pow2":         (self.test_Pow2,        N, N, N),  # y = n ^ x
            "PRelu":        (self.test_PRelu,       Y, N, Y),
            "Resize":       (self.test_Resize,      Y, N, Y),
            "Resize2":      (self.test_Resize2,     Y, N, Y),
            "Reshape":      (self.test_Reshape,     Y, N, N),
            "Reduce":       (self.test_Reduce,      Y, N, Y),
            "Reduce2":      (self.test_Reduce2,     Y, N, Y),
            "ReduceL2":     (self.test_ReduceL2,    Y, N, Y),
            "ReduceMean":   (self.test_ReduceMean,  Y, N, Y),
            "ReduceSum":    (self.test_ReduceSum,   Y, N, Y),
            "ReduceProd":   (self.test_ReduceProd,  Y, N, N),
            "Reciprocal":   (self.test_Reciprocal,  Y, N, Y),
            "Relu":         (self.test_Relu,        Y, N, Y),
            "ScatterND":    (self.test_ScatterND,   Y, N, N),
            "SiLU":         (self.test_SiLU,        Y, N, Y),
            "Softmax":      (self.test_Softmax,     Y, N, Y),
            "Softplus":     (self.test_Softplus,    Y, N, Y),
            "Squeeze":      (self.test_Squeeze,     Y, N, Y),
            "Sigmoid":      (self.test_Sigmoid,     Y, N, Y),
            "Slice":        (self.test_Slice,       Y, N, Y),
            "Slice2":       (self.test_Slice2,      Y, N, Y),
            "Slice3":       (self.test_Slice3,      Y, N, Y),
            "Split":        (self.test_Split,       Y, N, Y),
            "Scale":        (self.test_Scale,       Y, N, Y),
            "Sqrt":         (self.test_Sqrt,        Y, N, Y),
            "Sub":          (self.test_Sub,         Y, N, Y),
            "Sub2":         (self.test_Sub2,        Y, N, Y),
            "SubBcast":     (self.test_SubBcast,    Y, N, N),
            "SubBcast2":    (self.test_SubBcast2,   Y, N, N),
            "SubConst":     (self.test_SubConst,    Y, N, N),
            "SubConst2":    (self.test_SubConst2,   Y, N, Y),
            "Sum":          (self.test_Sum,         Y, N, Y),
            "Tanh":         (self.test_Tanh,        Y, N, Y),
            "Tile":         (self.test_Tile,        Y, N, Y),
            "Transpose":    (self.test_Transpose,   Y, N, Y),
            "Transpose2":   (self.test_Transpose2,  Y, N, Y),
            "TopK":         (self.test_TopK,        Y, N, N),
            "Upsample":     (self.test_Upsample,    Y, N, N),
            "Where":        (self.test_Where,       Y, N, Y),
            #####################################
            # Torch Test Case, Alphabetically
            #####################################
            # case: (test, bm1684x_support, bm1686_support, cv183x_support)
            "TorchActivation":      (self.test_TorchActivation,     N, N, Y),
            "TorchArgmax":          (self.test_TorchArgmax,         N, N, N),
            "TorchChannelShuffle":  (self.test_TorchChannelShuffle, N, N, N),
            "TorchChunk":           (self.test_TorchChunk,          Y, N, Y),
            "TorchConv2d":          (self.test_TorchConv2d,         N, N, Y),
            "TorchConv3dTrans":     (self.test_TorchConv3dTrans,    Y, N, Y),
            "TorchHardSwish":       (self.test_TorchHardSwish,      Y, N, Y),
            "TorchHardSigmoid":     (self.test_TorchHardSigmoid,    Y, N, Y),
            "TorchGelu":            (self.test_TorchGelu,           Y, N, Y),
            "TorchGroupNorm":       (self.test_TorchGroupNorm,      Y, N, N),
            "TorchGroupNorm2":      (self.test_TorchGroupNorm2,     Y, N, N),
            "TorchGRU":             (self.test_TorchGRU,            Y, N, Y),
            "TorchIdentity":        (self.test_TorchIdentity,       Y, N, Y),
            "TorchIndexCopy":       (self.test_TorchIndexCopy,      N, N, N),
            "TorchInstanceNorm":    (self.test_TorchInstanceNorm,   Y, N, N),
            "TorchInstanceNorm2":   (self.test_TorchInstanceNorm2,  Y, N, N),
            "TorchLayerGroup":      (self.test_TorchLayerGroup,     Y, N, Y),
            "TorchLayerNorm":       (self.test_TorchLayerNorm,      Y, N, Y),
            "TorchLayerNorm2":      (self.test_TorchLayerNorm2,     Y, N, Y),
            "TorchLogSoftmax":      (self.test_TorchLogSoftmax,     Y, N, Y),
            "TorchLSTM":            (self.test_TorchLSTM,           Y, N, Y),
            "TorchMaskedFill":      (self.test_TorchMaskedFill,     Y, N, N),
            "TorchNonZero":         (self.test_TorchNonZero,        Y, N, N),
            "TorchReflectionPad":   (self.test_TorchReflectionPad,  Y, N, Y),
            "TorchRoiAlign":        (self.test_TorchRoiAlign,       Y, N, N),
            "TorchSize":            (self.test_TorchSize,           Y, N, Y),
            "TorchStd":             (self.test_TorchStd,            Y, N, Y),
            "TorchWhere":           (self.test_TorchWhere,          Y, N, N),
            "TorchZeroPad":         (self.test_TorchZeroPad,        Y, N, Y),
            #########################################
            # Special Pass test case, Alphabetically
            #########################################
            # case: (test, bm1684x_support, bm1686_support, cv183x_support)
            "ArgError":         (self.test_ArgError,        Y, N, N),
            "ArgReducefull":    (self.test_ArgReducefull,   Y, N, N),
            "ConcatFuse":       (self.test_ConcatFuse,      Y, N, Y),
            "ConcatToSpace":    (self.test_ConcatToSpace,   Y, N, N),
            "Conv3dTo2d":       (self.test_Conv3dTo2d,      Y, N, Y),
            "Div2Mul":          (self.test_Div2Mul,         Y, N, Y),
            "ConvSlice":        (self.test_ConvSlice,       Y, N, N),
            "GaToSlice":        (self.test_GaToSlice,       Y, N, Y),
            "Mul2Scale":        (self.test_Mul2Scale,       Y, N, Y),
            "MatMulTranspose":  (self.test_MatMulTranspose, Y, N, Y),
            "MatMulTranspose2":  (self.test_MatMulTranspose2, Y, N, Y),
            "MatMulTranspose3":  (self.test_MatMulTranspose3, Y, N, Y),
            "PadConv1d":        (self.test_PadConv1d,       N, N, Y),
            "PadConv2d":        (self.test_PadConv2d,       Y, N, Y),
            "PadConv3d":        (self.test_PadConv3d,       N, N, N),
            "PadPool1d":        (self.test_PadPool1d,       N, N, Y),
            "PadPool2d":        (self.test_PadPool2d,       N, N, Y),
            "PadPool3d":        (self.test_PadPool3d,       N, N, Y),
            "PixelNorm":        (self.test_PixelNorm,       Y, N, N),
            "PixelNorm2":       (self.test_PixelNorm2,      Y, N, N),
            "PermuteFuse":      (self.test_PermuteFuse,     Y, N, Y),
            "PermutePad":       (self.test_PermutePad,      Y, N, N),
            "PermuteToReorg":   (self.test_PermuteToReorg,  Y, N, Y),
            "PermuteToReorg2":  (self.test_PermuteToReorg2, Y, N, Y),
            "PermuteToReshape": (self.test_PermuteToReshape,Y, N, N),
            "Permute5dSplit":   (self.test_Permute5dSplit,  Y, N, Y),
            "PoolAfterRelu":    (self.test_PoolAfterRelu,   Y, N, Y),
            "PoolSignError":    (self.test_PoolSignError,   Y, N, Y),
            "ReshapeFuse":      (self.test_ReshapeFuse,     Y, N, Y),
            "ReduceTranspose":  (self.test_ReduceTranspose, Y, N, N),
            "ReduceFuse":       (self.test_ReduceFuse,      Y, N, Y),
            "SwapDimInner":     (self.test_SwapDimInner,    Y, N, N),
            "SliceToReverse":   (self.test_SliceToReverse,  Y, N, N),
            "StaticDynMixed":   (self.test_StaticDynMixed,  Y, N, N),
            "TransposeArg":     (self.test_TransposeArg,    Y, N, Y),
            #"If":               (self.test_If,    Y, N, N),
            #"If2":               (self.test_If_v2,    Y, N, N),
            #"Loop" :            (self.test_Loop,    Y, N, N)
        }
        # yapf: enable

        # no quantization when quant_mode == "f32"
        self.support_quant_modes = ["f32", "f16", "bf16", "int8"]
        #self.support_quant_modes = ["f32", "f16", "bf16", "int8", "int4"]
        self.support_asym = [True, False]
        self.model_file = ".bmodel"
        self.is_cv18xx = False
        self.chip = chip.lower()
        self.dynamic = dynamic
        self.simple = simple
        self.multithread = not disable_thread
        if self.simple:
            self.support_quant_modes = ["f16", "int8"]
            self.support_asym = [False]
        if self.chip.startswith("cv18"):
            self.support_quant_modes = ["bf16", "int8"]
            self.support_asym = [False]
            self.model_file = ".cvimodel"
            self.is_cv18xx = True
        elif self.chip == "bm1684":
            self.support_quant_modes = ["f32", "int8"]
            self.support_asym = [False]
        self.mode = mode.lower()
        if self.mode == "" or self.mode == "all":
            self.quant_modes = self.support_quant_modes
        else:
            if self.mode not in self.support_quant_modes:
                raise RuntimeError("{} not support mode: {}".format(self.chip, self.mode))
            self.quant_modes = [self.mode]

    def test_single(self, case: str):
        np.random.seed(0)
        torch.manual_seed(7)
        print("Test: {}".format(case))
        if case in self.test_cases:
            func, _, _, _ = self.test_cases[case]
            func(case)
            print("====== TEST {} Success ======".format(case))
        else:
            raise RuntimeError("case [{}] is not exist".format(case))

    def check_support(self, case):
        _, bm1684x_support, bm1686_support, cv183x_support = self.test_cases[case]
        if self.is_cv18xx and cv183x_support:
            return True
        if self.chip == "bm1684x" and bm1684x_support:
            return True
        if self.chip == "bm1686" and bm1686_support:
            return True
        return False

    def create_random_input(self, graph_def: onnx.GraphProto):
        inputs = {}
        for i in graph_def.input:
            # only float input can use this
            assert (i.type.tensor_type.elem_type == onnx.TensorProto.FLOAT)
            name = i.name
            shape = [s.dim_value for s in i.type.tensor_type.shape.dim]
            inputs[name] = np.clip(np.random.randn(*shape).astype(np.float32), -10, 10)
        return inputs

    def onnx_convert(self, input_data: dict, graph_def, model_name: str):
        # onnx --> mlir conversion (origin and optimized mlir models will be generated and saved)
        fp32_mlir = "{}.mlir".format(model_name)
        model_def = helper.make_model(graph_def, producer_name=model_name)
        model_def.opset_import[0].version = 13
        onnx.checker.check_model(model_def)
        tool = OnnxTransformer(model_name, model_def)
        node_name_mapping = tool.converter.node_name_mapping
        tool.model_transform(fp32_mlir)

        onnx_model = "{}_opt.onnx".format(model_name)
        input_npz = "{}_in_fp32.npz".format(model_name)
        file_mark(input_npz)
        for name in input_data:
            if input_data[name].dtype in [np.int64, np.int32]:
                input_data[name] = input_data[name].astype(np.int32)
            else:
                input_data[name] = input_data[name].astype(np.float32)
        np.savez(input_npz, **input_data)
        # top mlir outputs will be inferenced first in case the quant mode is int8
        show_fake_cmd(input_npz, onnx_model, "onnx_out.npz")
        onnx_outs = onnx_inference(input_data, onnx_model, True)
        show_fake_cmd(input_npz, fp32_mlir, "top_out.npz")
        top_mlir_outs = mlir_inference(input_data, fp32_mlir, True)
        # save ref and cali table
        self.ref_npz = "{}_top_out.npz".format(model_name)
        file_mark(self.ref_npz)
        np.savez(self.ref_npz, **top_mlir_outs)
        self.table_name = "{}_cali_table".format(model_name)
        self.make_test_calibration_table(top_mlir_outs, self.table_name)
        return (onnx_outs, top_mlir_outs, input_npz, node_name_mapping)

    def bmodel_generate(self, model_name: str, quant_mode: str, isAsym: bool = False):

        top_mlir = "{}.mlir".format(model_name)
        tpu_mlir = "{}_{}".format(model_name, quant_mode)
        table = None
        if quant_mode == "int8" or quant_mode == "int4":
            tpu_mlir += "_asym" if isAsym else "_sym"
            table = self.table_name

        # lowering
        mlir_lowering(top_mlir,
                      tpu_mlir + ".mlir",
                      mode=quant_mode,
                      chip=self.chip,
                      cali_table=table,
                      asymmetric=isAsym)

        # transform
        tpu_final = tpu_mlir + "_final.mlir"
        bmodel = tpu_mlir + self.model_file
        if self.chip == "bm1684" or self.dynamic:
            # TODO: dynamic cast not support now
            quant_input = True
            quant_output = True
        else:
            quant_input = False
            quant_output = False
        mlir_to_model(tpu_mlir + ".mlir", bmodel, tpu_final, self.dynamic, quant_input,
                      quant_output)
        return (tpu_mlir + ".mlir", bmodel)

    def inference_and_compare(self,
                              tpu_mlir: str,
                              bmodel: str,
                              input_npz: str,
                              quant_mode: str,
                              model_name: str,
                              isAsym: bool = False):
        ref_tpu_tolerance = "0.9,0.9"
        input_data = np.load(input_npz)
        # tpu mlir inference and compare
        if quant_mode == "int8":
            ref_tpu_tolerance = "0.95,0.70" if not isAsym else "0.90,0.54"
        elif quant_mode == "int4":
            ref_tpu_tolerance = "0.90,0.60"
        elif quant_mode == "bf16":
            ref_tpu_tolerance = "0.95,0.85"
        tpu_npz = tpu_mlir.replace(".mlir", "_tpu_out.npz")
        file_mark(tpu_npz)
        show_fake_cmd(input_npz, tpu_mlir, tpu_npz)
        tpu_mlir_outs = mlir_inference(input_data, tpu_mlir, dump_all=True)
        np.savez(tpu_npz, **tpu_mlir_outs)
        npz_compare([self.ref_npz, tpu_npz, "--tolerance", ref_tpu_tolerance, "-v"])
        # bmodel / cvimodel inference and compare
        model_npz = bmodel.replace("." + bmodel.split(".")[-1], "_model_out.npz")
        file_mark(model_npz)
        show_fake_cmd(input_npz, bmodel, model_npz)
        model_outs = model_inference(input_data, bmodel)
        np.savez(model_npz, **model_outs)
        npz_compare([tpu_npz, model_npz, "--tolerance", "0.95,0.80", "-v"])

        msg = quant_mode.upper()
        if quant_mode == "int8" or quant_mode == "int4":
            msg += ", Asymmetric: {}".format(isAsym)

        print("[Success] test {} {}".format(model_name, msg))

    def make_test_calibration_table(self, tensors, table_name):
        # simple calibration table
        with open(table_name, 'w') as f:
            for name in tensors:
                flatten_tensor = tensors[name].flatten()
                max_val = max(flatten_tensor)
                min_val = min(flatten_tensor)
                if max_val == min_val:
                    max_val = max_val + 0.01
                t = 1.1 * max(abs(min_val), abs(max_val)) + 0.01
                f.write("{} {} {} {}\n".format(name, t, min_val, max_val))

    def simple_onnx_inference(self, input_data, onnx_file):
        ort_session = onnxruntime.InferenceSession(onnx_file)
        return ort_session.run(None, input_data)

    def square_rooted(self, x):
        return np.sqrt(sum([a * a for a in x]))

    def cosine_similarity(self, x, y):
        numerator = sum(a * b for a, b in zip(x, y))
        denominator = self.square_rooted(x) * self.square_rooted(y)
        return round(numerator / float(denominator), 3)

    def compare(self, ref_out, targe_out):
        if ref_out.dtype in [np.int64, np.int32, np.int16, np.int8]:
            cos = self.cosine_similarity(ref_out, targe_out)
            assert (cos > 0.999)
        else:
            np.testing.assert_allclose(ref_out, targe_out, rtol=1e-5, atol=1e-01)

    def torch_and_onnx_compare(self, input_data: dict, onnx_file: str, origin_output):
        onnx_outs = self.simple_onnx_inference(input_data, onnx_file)
        num_outputs = len(onnx_outs)
        if isinstance(origin_output, tuple):
            assert (len(origin_output) == num_outputs)
            for i in range(num_outputs):
                x = origin_output[i].data.numpy().flatten()
                y = onnx_outs[i].flatten()
                self.compare(x, y)
        else:
            self.compare(origin_output.data.numpy().flatten(), onnx_outs[0].flatten())
        print("* Torch and Onnx result compared *")

    def torch_and_test(self, inputs, torch_model: nn.Module, model_name: str):
        if isinstance(inputs, tuple):
            origin_output = torch_model(*inputs)
        else:
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

        torch.onnx.export(
            torch_model,
            inputs,
            onnx_file,
            export_params=True,
            verbose=True,
            opset_version=13,  # export hardswish needed
            input_names=in_names)
        onnx_model = onnx.load(onnx_file)
        self.torch_and_onnx_compare(in_data, onnx_file, origin_output)
        self.onnx_and_test(onnx_model.graph, name=model_name, input_data=in_data)

    def onnx_and_test(self, graph_def, name: str = "", input_data: dict = None):
        if input_data is None:
            input_data = self.create_random_input(graph_def)
        model_name = name if name else graph_def.name
        onnx_outs, top_mlir_outs, input_npz, node_name_mapping = self.onnx_convert(
            input_data, graph_def, model_name)
        # test onnx and mlir outputs
        counter = 0
        for name in onnx_outs:
            if name in top_mlir_outs:
                print("Compare mlir and onnx:{}\n".format(name))
                top_mlir_output = top_mlir_outs[name].flatten()
                onnx_output = onnx_outs[name].flatten()
                self.compare(onnx_output, top_mlir_output)
                counter += 1
            elif name in node_name_mapping:
                mapped_name = node_name_mapping[name]
                if mapped_name in top_mlir_outs:
                    print("Compare mlir and onnx:{}\n".format(mapped_name))
                    top_mlir_output = top_mlir_outs[mapped_name].flatten()
                    onnx_output = onnx_outs[name].flatten()
                    self.compare(onnx_output, top_mlir_output)
                    counter += 1
        if counter == 0:
            raise RuntimeError("No compare between onnx outs and mlir outts")
        print("Success: ONNX outs and Mlir outs are equal\n")
        for quant_mode in self.quant_modes:
            if quant_mode == "int8" or quant_mode == "int4":
                for isAsym in self.support_asym:
                    tpu_mlir, bmodel = self.bmodel_generate(model_name, quant_mode, isAsym)
                    self.inference_and_compare(tpu_mlir, bmodel, input_npz, quant_mode, model_name,
                                               isAsym)
            else:
                tpu_mlir, bmodel = self.bmodel_generate(model_name, quant_mode)
                self.inference_and_compare(tpu_mlir, bmodel, input_npz, quant_mode, model_name)

    ##################################
    # adding operators from here
    ##################################
    def AvgPoolBase(self, case_name, input_shape, output_shape, kernel, strides):
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
        self.onnx_and_test(graph_def)

    def test_AvgPool1d(self, case_name):
        input_shape = [1, 32, 128]
        output_shape = [1, 32, 64]
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
        mul_def = helper.make_node('Mul', ['pool_output', 'const_mul'], ['output'])

        graph_def = helper.make_graph([pool_def, mul_def],
                                      case_name, [input], [output],
                                      initializer=[mul_const])
        self.onnx_and_test(graph_def)

    def test_AvgPool2d(self, case_name):
        self.AvgPoolBase(case_name, [1, 32, 128, 128], [1, 32, 64, 64], [2, 2], [2, 2])

    def test_AvgPoolOdd(self, case_name):
        self.AvgPoolBase(case_name, [1, 32, 143, 143], [1, 32, 71, 71], [2, 2], [2, 2])

    def test_AvgPool3d(self, case_name):
        self.AvgPoolBase(case_name, [2, 32, 16, 32, 64], [2, 32, 8, 16, 32], [2, 2, 2], [2, 2, 2])

    def test_PadAvgPool2d(self, case_name):
        input_shape = [1, 16, 56, 56]
        output_shape = [1, 16, 56, 56]
        kernel_shape = [3, 3]
        strides = [1, 1]
        pads = np.array([0, 0, 1, 1, 0, 0, 1, 1]).astype(np.int64)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        pad_val = helper.make_tensor(name='pads',
                                     data_type=onnx.TensorProto.INT64,
                                     dims=pads.shape,
                                     vals=pads.flatten())
        pad_def = helper.make_node("Pad", ['input', 'pads'],
                                   outputs=['pad_output'],
                                   mode='constant')
        avgpool_def = helper.make_node(
            "AveragePool",
            inputs=['pad_output'],
            outputs=['output'],
            kernel_shape=kernel_shape,
            strides=strides,
        )
        graph_def = helper.make_graph([pad_def, avgpool_def],
                                      case_name, [input], [output],
                                      initializer=[pad_val])
        self.onnx_and_test(graph_def)

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
        self.onnx_and_test(graph_def)

    def MaxPoolBase(self, case_name, input_shape, output_shape, kernel, strides):
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
        self.onnx_and_test(graph_def)

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
        self.onnx_and_test(graph_def)

    def test_GRU(self, case_name):
        seq_length = 75
        batch_size = 2
        num_dir = 2
        input_size = 64
        hidden_size = 32
        direction = 'forward' if num_dir == 1 else 'bidirectional'
        h_data = np.random.rand(num_dir, batch_size, hidden_size).astype(np.float32)
        w_data = np.random.rand(num_dir, 3 * hidden_size, input_size).astype(np.float32)
        r_data = np.random.rand(num_dir, 3 * hidden_size, hidden_size).astype(np.float32)
        b_data = np.random.rand(num_dir, 6 * hidden_size).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT,
                                              [seq_length, batch_size, input_size])

        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT,
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
            outputs=['Y', ''],
            direction=direction,
            hidden_size=hidden_size,
            linear_before_reset=1,
        )
        graph_def = helper.make_graph([gru_def],
                                      case_name, [input], [Y],
                                      initializer=[w_value, r_value, b_value, h_value])
        if self.is_cv18xx:
            input_data = {}
            input_data["input"] = np.random.rand(seq_length, batch_size,
                                                 input_size).astype(np.float32)
            self.onnx_and_test(graph_def, input_data=input_data)
            return
        self.onnx_and_test(graph_def)

    def test_GRU2(self, case_name):
        seq_length = 75
        batch_size = 2
        num_dir = 2
        input_size = 128
        hidden_size = 64
        direction = 'forward' if num_dir == 1 else 'bidirectional'
        h_data = np.random.rand(num_dir, batch_size, hidden_size).astype(np.float32)
        w_data = np.random.rand(num_dir, 3 * hidden_size, input_size).astype(np.float32)
        r_data = np.random.rand(num_dir, 3 * hidden_size, hidden_size).astype(np.float32)
        b_data = np.random.rand(num_dir, 6 * hidden_size).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT,
                                              [seq_length, batch_size, input_size])

        Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT,
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
            outputs=['', 'Y_h'],
            direction=direction,
            hidden_size=hidden_size,
            linear_before_reset=1,
        )
        graph_def = helper.make_graph([gru_def],
                                      case_name, [input], [Y_h],
                                      initializer=[w_value, r_value, b_value, h_value])
        if self.is_cv18xx:
            input_data = {}
            input_data["input"] = np.random.rand(seq_length, batch_size,
                                                 input_size).astype(np.float32)
            self.onnx_and_test(graph_def, input_data=input_data)
            return
        self.onnx_and_test(graph_def)

    def test_GRU3(self, case_name):
        seq_length = 75
        batch_size = 2
        num_dir = 2
        input_size = 64
        hidden_size = 32
        direction = 'forward' if num_dir == 1 else 'bidirectional'
        h_data = np.random.rand(num_dir, batch_size, hidden_size).astype(np.float32)
        w_data = np.random.rand(num_dir, 3 * hidden_size, input_size).astype(np.float32)
        r_data = np.random.rand(num_dir, 3 * hidden_size, hidden_size).astype(np.float32)
        b_data = np.random.rand(num_dir, 6 * hidden_size).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT,
                                              [seq_length, batch_size, input_size])

        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT,
                                          [seq_length, num_dir, batch_size, hidden_size])
        Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT,
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
            outputs=['Y', 'Y_h'],
            direction=direction,
            hidden_size=hidden_size,
            linear_before_reset=1,
        )
        graph_def = helper.make_graph([gru_def],
                                      case_name, [input], [Y, Y_h],
                                      initializer=[w_value, r_value, b_value, h_value])
        if self.is_cv18xx:
            input_data = {}
            input_data["input"] = np.random.rand(seq_length, batch_size,
                                                 input_size).astype(np.float32)
            self.onnx_and_test(graph_def, input_data=input_data)
            return
        self.onnx_and_test(graph_def)

    def test_MaxPool1d(self, case_name):
        self.MaxPoolBase(case_name, [1, 32, 128], [1, 32, 64], [2], [2])

    def test_MaxPool2d(self, case_name):
        self.MaxPoolBase(case_name, [1, 32, 128, 128], [1, 32, 64, 64], [2, 2], [2, 2])

    def test_MaxPool3d(self, case_name):
        self.MaxPoolBase(case_name, [1, 32, 16, 32, 64], [1, 32, 8, 16, 63], [2, 1, 2], [2, 2, 1])

    def ConvBase(self, case_name, input_shape, filter_shape, output_shape, kernel, padding, stride,
                 dilation, groups):
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
        self.onnx_and_test(graph_def)

    def test_Conv1d(self, case_name):
        oc = 32
        self.ConvBase(case_name, [1, 16, 100], [oc, 16, 3], [1, oc, 100], [3], [1, 1], [1], [1], 1)

    def test_Conv2d(self, case_name):
        batchs = [1, 2, 4]
        for idx, batch in enumerate(batchs):
            input_shape = [batch, 16, 100, 100]
            filter_shape = [65, 16, 3, 3]
            output_shape = [batch, 65, 100, 100]
            self.ConvBase("{}_{}".format(case_name, idx),
                          input_shape,
                          filter_shape,
                          output_shape,
                          kernel=[3, 3],
                          padding=[1, 1, 1, 1],
                          stride=[1, 1],
                          dilation=[1, 1],
                          groups=1)

    def test_ConvDw(self, case_name):
        input_shape = [1, 16, 100, 100]
        filter_shape = [16, 1, 3, 3]
        output_shape = [1, 16, 100, 100]
        self.ConvBase(case_name,
                      input_shape,
                      filter_shape,
                      output_shape,
                      kernel=[3, 3],
                      padding=[1, 1, 1, 1],
                      stride=[1, 1],
                      dilation=[1, 1],
                      groups=16)

    def test_ConvStride(self, case_name):
        in_shape0 = [1, 32, 320, 320]
        f_shape0 = [64, 32, 3, 3]
        f_shape1 = [32, 64, 1, 1]
        out_shape = [1, 32, 160, 160]

        f_data0 = np.random.randn(*f_shape0).astype(np.float32)
        f_data1 = np.random.randn(*f_shape1).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape0)
        filter0 = helper.make_tensor('filter0', TensorProto.FLOAT, f_shape0, f_data0)
        filter1 = helper.make_tensor('filter1', TensorProto.FLOAT, f_shape1, f_data1)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, out_shape)

        conv0_def = helper.make_node(
            "Conv",
            inputs=['input', 'filter0'],
            outputs=['x1'],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[2, 2],
            dilations=[1, 1],
            group=1,
        )
        sigmoid_def = helper.make_node(
            "Sigmoid",
            inputs=['x1'],
            outputs=['x2'],
        )
        mul_def = helper.make_node(
            "Mul",
            inputs=['x1', 'x2'],
            outputs=['x3'],
        )
        conv1_def = helper.make_node(
            "Conv",
            inputs=['x3', 'filter1'],
            outputs=['output'],
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            dilations=[1, 1],
            group=1,
        )
        graph_def = helper.make_graph([conv0_def, sigmoid_def, mul_def, conv1_def],
                                      case_name, [input], [output],
                                      initializer=[filter0, filter1])
        self.onnx_and_test(graph_def)

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
        self.onnx_and_test(graph_def)

    def test_Concat(self, case_name):
        input_shape = {"input1": [1, 2, 64], "input2": [1, 3, 64], "input3": [1, 4, 64]}
        output_shape = [1, 2 + 3 + 4, 64]
        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, x) for k, x in input_shape.items()
        ]
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        concat_def = helper.make_node("Concat",
                                      inputs=list(input_shape.keys()),
                                      outputs=["output"],
                                      axis=1)

        graph_def = helper.make_graph([concat_def], case_name, inputs, [output])
        self.onnx_and_test(graph_def)

    def test_Concat2(self, case_name):
        input_shape = [1, 192, 256]
        x_shape = [1, 192, 16]
        output_shape = [1, 192, 288]
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        x1_data = np.random.randn(*x_shape).astype(np.float32)
        x1_node_def = onnx.helper.make_tensor(
            name='X1',
            data_type=onnx.TensorProto.FLOAT,
            dims=x_shape,
            vals=x1_data.flatten(),
        )
        x2_data = np.random.randn(*x_shape).astype(np.float32)
        x2_node_def = onnx.helper.make_tensor(
            name='X2',
            data_type=onnx.TensorProto.FLOAT,
            dims=x_shape,
            vals=x2_data.flatten(),
        )
        concat_node = helper.make_node(
            'Concat',
            ['input', 'X1', 'X2'],
            ['output'],
            axis=-1,
        )
        graph_def = helper.make_graph([concat_node],
                                      case_name, [input], [output],
                                      initializer=[x1_node_def, x2_node_def])
        self.onnx_and_test(graph_def)

    def test_Transpose(self, case_name):
        input_shapes = [[1, 16, 32, 32], [4, 3, 85, 20, 20], [1, 4, 2, 16, 20, 40]]
        transpose_orders = {4: [0, 2, 1, 3], 5: [0, 1, 3, 4, 2], 6: [0, 1, 2, 5, 3, 4]}
        for i, input_shape in enumerate(input_shapes):
            input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
            order = transpose_orders[len(input_shape)]
            output_shape = [input_shape[order[i]] for i in range(len(order))]
            output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
            transpose_def = helper.make_node("Transpose",
                                             inputs=['input'],
                                             outputs=['output'],
                                             perm=order)
            graph_def = helper.make_graph([transpose_def], "{}_{}".format(case_name, i), [input],
                                          [output])
            self.onnx_and_test(graph_def)

    def test_Transpose2(self, case_name):
        cases = [
            #case 0, input shape,output shape, order
            [[529, 49, 3, 3, 32], [3, 529, 3, 49, 32], [2, 0, 3, 1, 4]],
            #case 1, input shape,output shape, order
            [[1, 1, 1, 23, 7, 23, 7, 96], [1, 1, 23, 23, 1, 7, 7, 96], [0, 1, 3, 5, 2, 4, 6, 7]]
        ]
        for idx, shapes in enumerate(cases):
            input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shapes[0])
            output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shapes[1])
            transpose_def = helper.make_node("Transpose",
                                             inputs=['input'],
                                             outputs=['output'],
                                             perm=shapes[2])
            graph_def = helper.make_graph([transpose_def], "{}_{}".format(case_name, idx), [input],
                                          [output])
            self.onnx_and_test(graph_def)

    def test_Where(self, case_name):
        # real case
        shape = [10, 40, 224]
        cond_data = np.zeros(shape).astype(np.bool_)
        cond_data[:, :, :100] = 1
        tbrn_data = np.random.randn(*shape).astype(np.float32)
        fbrn_data = np.random.randn(*shape).astype(np.float32)
        cond = helper.make_tensor_value_info('cond', TensorProto.BOOL, shape)
        tbrn = helper.make_tensor_value_info('tbrn', TensorProto.FLOAT, shape)
        fbrn = helper.make_tensor_value_info('fbrn', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)
        where_node = helper.make_node(
            'Where',
            ['cond', 'tbrn', 'fbrn'],
            ['output'],
        )
        graph_def = helper.make_graph([where_node], case_name, [cond, tbrn, fbrn], [output])
        self.onnx_and_test(graph_def,
                           input_data={
                               "cond": cond_data,
                               "tbrn": tbrn_data,
                               "fbrn": fbrn_data
                           })

    def test_Relu(self, case_name):
        oc = 64
        input_shape = [1, 16, 128, 128]
        filter_shape = [oc, 16, 3, 3]
        output_shape = [1, oc, 128, 128]
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

        relu_def = helper.make_node("Relu", inputs=['conv_output'], outputs=['output'])

        graph_def = helper.make_graph([conv_def, relu_def],
                                      case_name, [input], [output],
                                      initializer=[weight, bias])
        self.onnx_and_test(graph_def)

    def test_LeakyRelu(self, case_name):
        oc = 32
        input_shape = [1, 16, 100, 100]
        filter_shape = [oc, 16, 3, 3]
        output_shape = [1, oc, 100, 100]
        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        bias_data = np.random.randn(oc).astype(np.float32)
        alpha_cases = [0.67, 0.2]
        for i, a in enumerate(alpha_cases):
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
                                             alpha=a)

            graph_def = helper.make_graph([conv_def, leakyrelu_def],
                                          "{}_{}".format(case_name, i), [input], [output],
                                          initializer=[weight, bias])
            self.onnx_and_test(graph_def)

    def test_Mul(self, case_name):
        input_shape = {"input1": [1, 3, 27, 27], "input2": [1, 3, 27, 27]}
        output_shape = [1, 3, 27, 27]
        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, x) for k, x in input_shape.items()
        ]
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        mul_def = helper.make_node("Mul", inputs=list(input_shape.keys()), outputs=["output"])

        graph_def = helper.make_graph([mul_def], case_name, inputs, [output])
        self.onnx_and_test(graph_def)

    def test_MulBcast(self, case_name):
        shapes = ([7, 9, 44, 38], )
        bcast_dims = ([[0], [2], [0, 2]], )
        for i, s in enumerate(shapes):
            for dims in bcast_dims[i]:
                bcast_s = s[::]
                for dim in dims:
                    bcast_s[dim] = 1
                a = helper.make_tensor_value_info("a", TensorProto.FLOAT, bcast_s)
                b = helper.make_tensor_value_info("b", TensorProto.FLOAT, s)
                c = helper.make_tensor_value_info("c", TensorProto.FLOAT, bcast_s)
                output = helper.make_tensor_value_info("output", TensorProto.FLOAT, s)
                Mul_def = helper.make_node("Mul", inputs=["a", "b"], outputs=["x"])
                Mul_def_2 = helper.make_node("Mul", inputs=["x", "c"], outputs=["output"])
                graph_def = helper.make_graph([Mul_def, Mul_def_2],
                                              "{}_{}_{}".format(case_name, i,
                                                                "".join(map(str, dims))), [a, b, c],
                                              [output])
                self.onnx_and_test(graph_def)

    def test_MulBcast2(self, case_name):
        shapes = ([4, 7, 1, 15], )
        bcast_shapes = ([1, 7, 13, 15], )
        out_shapes = ([4, 7, 13, 15], )
        for i, s in enumerate(shapes):
            a = helper.make_tensor_value_info("a", TensorProto.FLOAT, bcast_shapes[i])
            b = helper.make_tensor_value_info("b", TensorProto.FLOAT, s)
            c = helper.make_tensor_value_info("c", TensorProto.FLOAT, bcast_shapes[i])
            output = helper.make_tensor_value_info("output", TensorProto.FLOAT, out_shapes[i])
            Mul_def = helper.make_node("Mul", inputs=["a", "b"], outputs=["x"])
            Mul_def_2 = helper.make_node("Mul", inputs=["x", "c"], outputs=["output"])
            graph_def = helper.make_graph([Mul_def, Mul_def_2], "{}_{}".format(
                case_name,
                i,
            ), [a, b, c], [output])
            self.onnx_and_test(graph_def)

    def test_MulConst(self, case_name):
        input_shape = [1, 3, 27, 27]
        output_shape = [1, 3, 27, 27]

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
        self.onnx_and_test(graph_def)

    def test_Gemm(self, case_name):
        M = 50
        K = 100
        N = 25
        input_shape = [M, K]
        weight_shape = [K, N]
        bias_shape = [N]
        output_shape = [M, N]
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
        self.onnx_and_test(graph_def)

    def test_MatMul(self, case_name):
        M = 50
        K = 100
        N = 25
        input_shape = [4, 10, M, K]
        weight_shape = [K, N]
        output_shape = [4, 10, M, N]
        weight_data = np.random.randn(*weight_shape).astype(np.float32)
        input = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, weight_shape, weight_data)
        gemm_def = helper.make_node("MatMul", inputs=["input", "weight"], outputs=["output"])
        graph_def = helper.make_graph([gemm_def],
                                      case_name, [input], [output],
                                      initializer=[weight])
        self.onnx_and_test(graph_def)

    def test_MatMul2(self, case_name):
        M = 50
        K = 100
        N = 25
        input_shape = [4, 10, M, K]
        weight_shape = [K, N]
        bias_shape = [N]
        output_shape = [4, 10, M, N]
        weight_data = np.random.randn(*weight_shape).astype(np.float32)
        bias_data = np.random.randn(*bias_shape).astype(np.float32)
        input = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, weight_shape, weight_data)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, bias_shape, bias_data)
        gemm_def = helper.make_node("MatMul", inputs=["input", "weight"], outputs=["x1"])
        add_def = helper.make_node("Add", inputs=["x1", "bias"], outputs=["output"])
        graph_def = helper.make_graph([gemm_def, add_def],
                                      case_name, [input], [output],
                                      initializer=[weight, bias])
        self.onnx_and_test(graph_def)

    def test_Scale(self, case_name):
        input_shape = [1, 65, 100, 100]
        output_shape = [1, 65, 100, 100]
        weight_shape = [1, 65, 1, 1]
        offset_shape = [1, 65, 1, 1]
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

        self.onnx_and_test(graph_def)

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
        self.onnx_and_test(graph_def)

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
        self.onnx_and_test(graph_def)

    def test_Reshape(self, case_name):
        input_shape = [1, 16, 32, 32]
        right_shape = [3]
        output_shape = [1, 16, 1024]
        right_data = np.array([1, 16, 1024], dtype=np.int64)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        right = helper.make_tensor('right', TensorProto.INT64, right_shape, right_data)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        reshape_def = helper.make_node(case_name, inputs=['input', 'right'], outputs=['output'])
        graph_def = helper.make_graph([reshape_def],
                                      case_name, [input], [output],
                                      initializer=[right])
        self.onnx_and_test(graph_def)

    def test_floor(self, case_name):
        input_shape = [1, 128, 32, 32]
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
        floor_def = helper.make_node(case_name, inputs=['input'], outputs=['output'])
        graph_def = helper.make_graph([floor_def], case_name, [input], [output])
        self.onnx_and_test(graph_def)

    def test_Softmax(self, case_name):
        input_shapes = [[3, 100, 1, 1], [3, 100, 32], [3, 100, 32, 1]]
        axiss = [1, 2]
        for input_shape in input_shapes:
            for axis in axiss:
                input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
                output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
                softmax_def = helper.make_node(case_name,
                                               inputs=['input'],
                                               outputs=['output'],
                                               axis=axis)
                graph_def = helper.make_graph([softmax_def], case_name, [input], [output])
                self.onnx_and_test(graph_def)

    def test_LogSoftmax(self, case_name):
        input_shape = [3, 100, 32]
        axis = 2
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
        softmax_def = helper.make_node(case_name, inputs=['input'], outputs=['output'], axis=axis)
        graph_def = helper.make_graph([softmax_def], case_name, [input], [output])
        self.onnx_and_test(graph_def)

    def test_Softplus(self, case_name):
        input_shape = [200]
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
        softplus_def = helper.make_node(case_name, inputs=['input'], outputs=['output'])
        graph_def = helper.make_graph([softplus_def], case_name, [input], [output])
        self.onnx_and_test(graph_def)

    def test_Exp(self, case_name):
        input_shape = [1, 3, 32, 32]
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
        exp_def = helper.make_node(case_name, inputs=['input'], outputs=['output'])
        graph_def = helper.make_graph([exp_def], case_name, [input], [output])
        self.onnx_and_test(graph_def)

    def test_Tanh(self, case_name):
        input_shape = [1, 3, 32, 32]
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
        tanh_def = helper.make_node(case_name, inputs=['input'], outputs=['output'])
        graph_def = helper.make_graph([tanh_def], case_name, [input], [output])
        self.onnx_and_test(graph_def)

    def test_Log(self, case_name):
        input_shape = [1, 3, 32, 32]
        input_data = np.clip(np.random.randn(*input_shape).astype(np.float32) * 10.0, 0.5, 8)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
        log_def = helper.make_node(case_name, inputs=['input'], outputs=['output'])
        graph_def = helper.make_graph([log_def], case_name, [input], [output])
        self.onnx_and_test(graph_def, input_data={'input': input_data})

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
        self.onnx_and_test(graph_def)

    def test_Pad(self, case_name):
        case0 = [[3, 8, 32, 32], [3, 8, 44, 46], [0, 0, 5, 6, 0, 0, 7, 8]]
        case1 = [[4, 8, 10], [4, 9, 11], [0, 1, 0, 0, 0, 1]]
        case2 = [[1, 1, 160, 160, 96], [1, 1, 161, 161, 96], [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]]
        cases = (case0, case1, case2)
        for idx, case in enumerate(cases):
            pads = np.array(case[2]).astype(np.int64)
            input = helper.make_tensor_value_info('input', TensorProto.FLOAT, case[0])
            output = helper.make_tensor_value_info('output', TensorProto.FLOAT, case[1])
            pad_val = helper.make_tensor(name='pads',
                                         data_type=onnx.TensorProto.INT64,
                                         dims=pads.shape,
                                         vals=pads.flatten())
            pad_def = helper.make_node("Pad", ['input', 'pads'],
                                       outputs=['output'],
                                       mode='constant')
            graph_def = helper.make_graph([pad_def],
                                          "{}_{}".format(case_name, idx), [input], [output],
                                          initializer=[pad_val])
            self.onnx_and_test(graph_def)

    def test_Pad1(self, case_name):
        input_shape = [3, 8, 32, 32]
        output_shape = [3, 8, 44, 46]
        pads = np.array([0, 0, 5, 6, 0, 0, 7, 8]).astype(np.int64)
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
        self.onnx_and_test(graph_def)

    def test_PadEdge(self, case_name):
        input_shape = [3, 8, 32, 32]
        output_shape = [3, 8, 44, 46]
        pads = np.array([0, 0, 5, 6, 0, 0, 7, 8]).astype(np.int64)
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
        self.onnx_and_test(graph_def)

    def test_PadReflect(self, case_name):
        input_shape = [3, 8, 32, 32]
        output_shape = [3, 8, 44, 46]
        pads = np.array([0, 0, 5, 6, 0, 0, 7, 8]).astype(np.int64)

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
        self.onnx_and_test(graph_def)

    def test_DepthToSpace(self, case_name):
        in_shape = [1, 32, 108, 192]
        n, c, h, w = in_shape
        blocksize = 2
        # mode='CRD'
        mode = 'DCR'  # default
        out_shape = [n, c // (blocksize * blocksize), h * blocksize, w * blocksize]
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, in_shape)
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
        self.onnx_and_test(graph_def)

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
        self.onnx_and_test(graph_def, input_data=input_data)

    def test_DivBcast(self, case_name):
        shapes = ([6, 11, 39, 29], )
        bcast_dims = ([[0], [2], [0, 2]], )
        for i, s in enumerate(shapes):
            for dims in bcast_dims[i]:
                bcast_s = s[::]
                for dim in dims:
                    bcast_s[dim] = 1
                a_data = np.random.randn(*bcast_s).astype(np.float32)
                b_data = np.clip(np.random.randn(*s).astype(np.float32), 0.01, 10)
                c_data = np.clip(np.random.randn(*bcast_s).astype(np.float32), 0.01, 10)
                a = helper.make_tensor_value_info("a", TensorProto.FLOAT, bcast_s)
                b = helper.make_tensor_value_info("b", TensorProto.FLOAT, s)
                c = helper.make_tensor_value_info("c", TensorProto.FLOAT, bcast_s)
                output = helper.make_tensor_value_info("output", TensorProto.FLOAT, s)
                Div_def = helper.make_node("Div", inputs=["a", "b"], outputs=["x"])
                Div_def_2 = helper.make_node("Div", inputs=["x", "c"], outputs=["output"])
                graph_def = helper.make_graph([Div_def, Div_def_2],
                                              "{}_{}_{}".format(case_name, i,
                                                                "".join(map(str, dims))), [a, b, c],
                                              [output])
                self.onnx_and_test(graph_def, input_data={"a": a_data, "b": b_data, "c": c_data})

    def test_DivBcast2(self, case_name):
        shapes = ([5, 12, 1, 21], )
        bcast_shapes = ([1, 12, 9, 21], )
        out_shapes = ([5, 12, 9, 21], )
        for i, s in enumerate(shapes):
            a_data = np.random.randn(*bcast_shapes[i]).astype(np.float32)
            b_data = np.clip(np.random.randn(*s).astype(np.float32), 0.01, 10)
            c_data = np.clip(np.random.randn(*bcast_shapes[i]).astype(np.float32), 0.01, 10)
            a = helper.make_tensor_value_info("a", TensorProto.FLOAT, bcast_shapes[i])
            b = helper.make_tensor_value_info("b", TensorProto.FLOAT, s)
            c = helper.make_tensor_value_info("c", TensorProto.FLOAT, bcast_shapes[i])
            output = helper.make_tensor_value_info("output", TensorProto.FLOAT, out_shapes[i])
            Div_def = helper.make_node("Div", inputs=["a", "b"], outputs=["x"])
            Div_def_2 = helper.make_node("Div", inputs=["x", "c"], outputs=["output"])
            graph_def = helper.make_graph([Div_def, Div_def_2], "{}_{}".format(
                case_name,
                i,
            ), [a, b, c], [output])
            self.onnx_and_test(graph_def, input_data={"a": a_data, "b": b_data, "c": c_data})

    def test_ConvTrans(self, case_name):
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

        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        bias_data = np.random.randn(oc).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, list(bias_data.shape), bias_data)
        ConvTrans_def = helper.make_node("ConvTranspose",
                                         inputs=['input', 'weight', 'bias'],
                                         outputs=['output'],
                                         kernel_shape=kernel_shape,
                                         pads=pads,
                                         strides=strides,
                                         dilations=dilations,
                                         group=1)
        graph_def = helper.make_graph([ConvTrans_def],
                                      case_name, [input], [output],
                                      initializer=[weight, bias])
        self.onnx_and_test(graph_def)

    def test_ConvTrans2(self, case_name):
        input_shape = [1, 64, 35, 35]
        filter_shape = [64, 32, 2, 2]
        bias_shape = [32]
        output_shape = [1, 32, 70, 70]
        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        bias_data = np.random.randn(*bias_shape).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, bias_shape, bias_data)
        ConvTrans_def = helper.make_node("ConvTranspose",
                                         inputs=['input', 'weight', 'bias'],
                                         outputs=['output'],
                                         kernel_shape=[2, 2],
                                         pads=[0, 0, 0, 0],
                                         strides=[2, 2],
                                         dilations=[1, 1],
                                         group=1)
        graph_def = helper.make_graph([ConvTrans_def],
                                      case_name, [input], [output],
                                      initializer=[weight, bias])
        self.onnx_and_test(graph_def)

    def test_Squeeze(self, case_name):
        axis = [1, 3]
        input_shape = [3, 1, 32, 1]
        output_shape = [input_shape[i] for i in range(len(input_shape)) if i not in axis]
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
        self.onnx_and_test(graph_def)

    def test_Clip(self, case_name):
        input_shape = [1, 3, 32, 32]
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        min = helper.make_tensor('min', TensorProto.FLOAT, [], -1.0 * np.ones(1))
        max = helper.make_tensor('max', TensorProto.FLOAT, [], 6.0 * np.ones(1))
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
        node_def = helper.make_node(case_name, inputs=['input', 'min', 'max'], outputs=['output'])
        graph_def = helper.make_graph([node_def], case_name, [input], [output], [min, max])
        self.onnx_and_test(graph_def)

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
        self.onnx_and_test(graph_def)

    def test_ReduceSum(self, case_name):
        input_shape = [4, 4, 4, 16, 16, 64]
        output_shape = [4, 4, 4, 16, 64]
        axes = helper.make_tensor(name='axes', data_type=onnx.TensorProto.INT64, dims=[1], vals=[4])
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('o_sum', TensorProto.FLOAT, output_shape)
        reduce_sum = helper.make_node('ReduceSum',
                                      inputs=['input', 'axes'],
                                      outputs=['o_sum'],
                                      keepdims=0)
        graph_def = helper.make_graph([reduce_sum],
                                      case_name, [input], [output],
                                      initializer=[axes])
        self.onnx_and_test(graph_def)

    def test_ReduceProd(self, case_name):
        input_shape = [1, 3, 128, 128]
        output_shape = [1, 3, 128]
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('o_prod', TensorProto.FLOAT, output_shape)
        reduce_prod = helper.make_node("ReduceProd",
                                       inputs=['input'],
                                       outputs=['o_prod'],
                                       keepdims=0,
                                       axes=[2])
        graph_def = helper.make_graph([reduce_prod], case_name, [input], [output])
        self.onnx_and_test(graph_def)

    def test_Sigmoid(self, case_name):
        input_shape = [1, 16, 64, 64]
        output_shape = [1, 16, 64, 64]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        sigmoid_def = helper.make_node(
            case_name,
            inputs=['input'],
            outputs=['output'],
        )
        graph_def = helper.make_graph([sigmoid_def], case_name, [input], [output])
        if 0:  # this code to test local layer, but when model use local layer, compare will not pass.
            a = helper.make_tensor_value_info("a", TensorProto.FLOAT, input_shape)
            add_out = helper.make_tensor_value_info("add_out", TensorProto.FLOAT, input_shape)
            add_def = helper.make_node("Add", inputs=["a", "output"], outputs=["add_out"])
            graph_def = helper.make_graph([sigmoid_def, add_def], case_name, [input, a], [add_out])

        self.onnx_and_test(graph_def)

    def test_Slice(self, case_name):
        input_shape = [5, 116, 64, 64]
        output_shape = [3, 16, 16, 8]
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
            "Slice",
            inputs=['input', 'starts', 'ends', 'axes', 'steps'],
            outputs=['output'],
        )

        graph_def = helper.make_graph([slice_def],
                                      case_name, [input], [output],
                                      initializer=[starts, ends, axes, steps])
        self.onnx_and_test(graph_def)

    def test_Slice2(self, case_name):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                y = x[:, :, 30::2, :42:2]
                y = y + 1
                return y

        x = torch.randn(4, 8, 60, 80).float()
        self.torch_and_test(x, Model(), case_name)

    def test_ConvSlice(self, case_name):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.filter = torch.randn(8, 3, 3, 3)

            def forward(self, x):
                x = F.conv2d(x, self.filter, padding=2)
                y = x[:, :, 0:-1, 0:-1]
                return y

        x = torch.randn(1, 3, 8, 8)
        self.torch_and_test(x, Model(), case_name)

    def test_Upsample(self, case_name):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.Upsample = nn.Upsample(scale_factor=2, mode="nearest")

            def forward(self, x):
                return self.Upsample(x)

        x = torch.randn(2, 64, 184, 320).float()
        self.torch_and_test(x, Model(), case_name)

    def test_Deconv(self, case_name):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.deconv = nn.ConvTranspose2d(in_channels=64,
                                                 out_channels=64,
                                                 kernel_size=2,
                                                 stride=2,
                                                 padding=0,
                                                 output_padding=0,
                                                 groups=1,
                                                 bias=False,
                                                 dilation=1)

            def forward(self, x):
                y = self.deconv(x)
                return y

        x = torch.randn(3, 64, 184, 320).float()
        self.torch_and_test(x, Model(), case_name)

    def test_Slice3(self, case_name):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                y1 = x[:, :16, :, :]
                y2 = x[:, 16:, :, :]
                y = y1 * y2
                return y

        x = torch.randn(1, 32, 64, 88).float()
        self.torch_and_test(x, Model(), case_name)

    def test_Split(self, case_name):
        input_shape = [6, 116, 64, 64]
        output1_shape = [3, 116, 64, 64]
        output2_shape = [3, 116, 64, 64]
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
        self.onnx_and_test(graph_def)

    def test_Arg(self, case_name):
        for keep in [True, False]:
            input_shape = [20, 40, 60]
            output_shape = [20, 1, 60] if keep else [20, 60]

            input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
            output1 = helper.make_tensor_value_info('o_max', TensorProto.INT64, output_shape)
            arg_max = helper.make_node(
                'ArgMax',
                ['input'],
                ['o_max'],
                keepdims=keep,
                axis=1,
                select_last_index=1
            )
            if self.is_cv18xx:
                graph_def = helper.make_graph([arg_max], "{}_{}".format(case_name, keep), [input],
                                              [output1])
                self.onnx_and_test(graph_def)
                continue

            output2 = helper.make_tensor_value_info('o_min', TensorProto.INT64, output_shape)
            arg_min = helper.make_node(
                'ArgMin',
                ['input'],
                ['o_min'],
                keepdims=keep,
                axis=1,
                select_last_index=1
            )

            graph_def = helper.make_graph([arg_max, arg_min], "{}_{}".format(case_name, keep),
                                          [input], [output1, output2])
            self.onnx_and_test(graph_def)

    def test_Reduce(self, case_name):
        for keep in [True, False]:
            input_shape = [4, 4, 4, 16, 64]
            output_shape = [4, 4, 4, 16, 1] if keep else [4, 4, 4, 16]

            input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
            output0 = helper.make_tensor_value_info('o_mean', TensorProto.FLOAT, output_shape)
            output1 = helper.make_tensor_value_info('o_max', TensorProto.FLOAT, output_shape)
            output2 = helper.make_tensor_value_info('o_min', TensorProto.FLOAT, output_shape)
            reduce_mean = helper.make_node(
                'ReduceMean',
                ['input'],
                ['o_mean'],
                keepdims=keep,
                axes=[4],
            )
            reduce_max = helper.make_node(
                'ReduceMax',
                ['input'],
                ['o_max'],
                keepdims=keep,
                axes=[4],
            )
            reduce_min = helper.make_node(
                'ReduceMin',
                ['input'],
                ['o_min'],
                keepdims=keep,
                axes=[4],
            )

            graph_def = helper.make_graph([reduce_mean, reduce_max, reduce_min],
                                          "{}_{}".format(case_name, keep), [input],
                                          [output0, output1, output2])
            self.onnx_and_test(graph_def)

    def test_Reduce2(self, case_name):
        for keep in [True, False]:
            input_shape = [4, 4, 4, 16, 64]
            output_shape = [4, 4, 1, 1, 64] if keep else [4, 4, 64]

            input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
            output0 = helper.make_tensor_value_info('o_mean', TensorProto.FLOAT, output_shape)
            output1 = helper.make_tensor_value_info('o_max', TensorProto.FLOAT, output_shape)
            output2 = helper.make_tensor_value_info('o_min', TensorProto.FLOAT, output_shape)
            reduce_mean = helper.make_node('ReduceMean', ['input'], ['o_mean'],
                                           keepdims=keep,
                                           axes=[2, 3])
            reduce_max = helper.make_node('ReduceMax', ['input'], ['o_max'],
                                          keepdims=keep,
                                          axes=[2, 3])
            reduce_min = helper.make_node('ReduceMin', ['input'], ['o_min'],
                                          keepdims=keep,
                                          axes=[2, 3])

            graph_def = helper.make_graph([reduce_mean, reduce_max, reduce_min],
                                          "{}_{}".format(case_name, keep), [input],
                                          [output0, output1, output2])
            self.onnx_and_test(graph_def)

    def test_ReduceMean(self, case_name):
        input_shape = [2, 200, 7, 7]
        output_shape = [2, 1, 1, 7]

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
        self.onnx_and_test(graph_def)

    def test_TorchHardSigmoid(self, case_name):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                # old torch version not support export this op
                self.hardsigmoid = nn.Hardsigmoid()

            def forward(self, x):
                return self.hardsigmoid(x)
                # return F.hardtanh(x + 3, 0., 6.) / 6.

        x = torch.randn(3, 100, 200).float()
        self.torch_and_test(x, Model(), case_name)

    def test_TorchHardSwish(self, case_name):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                # old torch version not support export this op
                self.hardSwish = nn.Hardswish()

            def forward(self, x):
                return self.hardSwish(x)
                # return x * F.hardtanh(x + 3, 0., 6.) / 6.

        x = torch.randn(10, 2).float()
        self.torch_and_test(x, Model(), case_name)

    def test_TorchIndexCopy(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.index = torch.tensor([0, 4, 2])
                self.t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)

            def forward(self, x):
                y = torch.index_copy(x, 0, self.index, self.t)
                return y

        x = torch.randn(5, 3).float()
        self.torch_and_test(x, Model(), case_name)

    def test_TorchIdentity(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.Identity = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)

            def forward(self, x):
                x = self.Identity(x)
                y = x * x
                return y

        x = torch.randn(1, 3, 100, 100).float()
        self.torch_and_test(x, Model(), case_name)

    def test_TorchChannelShuffle(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.channel_shuffle = nn.ChannelShuffle(2)

            def forward(self, x):
                x = self.channel_shuffle(x)
                return x

        x = torch.randn(1, 4, 100, 100).float()
        self.torch_and_test(x, Model(), case_name)

    def test_TorchReflectionPad(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.ReflectionPad1d = nn.ReflectionPad1d(2)

            def forward(self, x):
                x = self.ReflectionPad1d(x)
                return x

        x = torch.randn(3, 100, 100).float()
        self.torch_and_test(x, Model(), case_name)

    def test_TorchRoiAlign(self, case_name):
        roi_num = 5
        N, C, H, W = 1, 3, 100, 100

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, boxes):
                y = torchvision.ops.roi_align(x, boxes, [8, 8])
                return y

        def gen_rand_rois(N, H, W, roi_num) -> torch.Tensor:
            batch_indice = torch.randint(0, N, (roi_num, ), dtype=torch.int32).float()
            roi_xl = torch.rand(roi_num, dtype=torch.float32) * (W - 1)
            roi_xh = torch.rand(roi_num, dtype=torch.float32) * (W - 1)
            roi_yl = torch.rand(roi_num, dtype=torch.float32) * (H - 1)
            roi_yh = torch.rand(roi_num, dtype=torch.float32) * (H - 1)
            for i in range(roi_num):
                if roi_xl[i] > roi_xh[i]:
                    roi_xl[i], roi_xh[i] = roi_xh[i], roi_xl[i]
                if roi_yl[i] > roi_yh[i]:
                    roi_yl[i], roi_yh[i] = roi_yh[i], roi_yl[i]
            batch_indice.unsqueeze_(1)
            roi_xl.unsqueeze_(1)
            roi_yl.unsqueeze_(1)
            roi_xh.unsqueeze_(1)
            roi_yh.unsqueeze_(1)
            rois = torch.cat((batch_indice, roi_xl, roi_yl, roi_xh, roi_yh), 1)
            return rois

        x = torch.randn(N, C, H, W).float()
        boxes = gen_rand_rois(N, H, W, roi_num)
        self.torch_and_test((x, boxes), Model(), case_name)

    def test_TorchLayerGroup(self, case_name):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.m1 = nn.Conv2d(3, 8, 3, 1, 0)
                self.r1 = nn.ReLU()
                self.m2 = nn.Conv2d(8, 8, 3, 1, 1)

            def forward(self, x):
                y0 = self.m1(x)
                y1 = self.r1(y0)
                y2 = self.m2(y1)
                y3 = y2 + y1
                return y3, y1

        shapes = ([1, 3, 50, 50], [3, 3, 50, 50], [4, 3, 60, 70], [7, 3, 80, 40])
        for idx, shape in enumerate(shapes):
            x = torch.randn(*shape).float()
            self.torch_and_test(x, Model(), "{}_{}".format(case_name, idx))

    def test_TorchLogSoftmax(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.linear = nn.Linear(32, 72, bias=False)
                self.act = nn.LogSoftmax(dim=2)

            def forward(self, x):
                x = self.linear(x)
                x = self.act(x)
                return x

        input_shape = [3, 100, 32]
        input_data = torch.randn(input_shape)
        self.torch_and_test(input_data, Model(), case_name)

    def test_TorchLSTM(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.rnn = nn.LSTM(input_size=100, hidden_size=128, bidirectional=True)

            def forward(self, x, h_0, c_0):
                Y, (Y_h, Y_c) = self.rnn(x, (h_0, c_0))
                return Y, Y_h, Y_c

        input = torch.randn(81, 1, 100)
        h_0 = torch.randn(2, 1, 128)
        c_0 = torch.randn(2, 1, 128)

        inputs = (input, h_0, c_0)
        self.torch_and_test(inputs, Model(), case_name)

    def test_TorchInstanceNorm(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.instance_norm = nn.InstanceNorm2d(100, affine=True)

            def forward(self, x):
                return self.instance_norm(x)

        x = torch.randn(20, 100, 35, 45).float()
        self.torch_and_test(x, Model(), case_name)

    def test_TorchInstanceNorm2(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.instance_norm = nn.InstanceNorm2d(8, affine=True)
                nn.init.normal_(self.instance_norm.weight, std=0.01)
                nn.init.normal_(self.instance_norm.bias, std=0.01)

            def forward(self, x):
                return self.instance_norm(x + 1)

        x = torch.randn(2, 8, 15, 15).float()
        self.torch_and_test(x, Model(), case_name)

    def test_TorchGroupNorm(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.group_norm = nn.GroupNorm(3, 6)
                nn.init.normal_(self.group_norm.weight, std=0.01)
                nn.init.normal_(self.group_norm.bias, std=0.01)

            def forward(self, x):
                return self.group_norm(x)

        x = torch.randn(20, 6, 10, 10).float()
        self.torch_and_test(x, Model(), case_name)

    def test_TorchGroupNorm2(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.group_norm = nn.GroupNorm(3, 6)

            def forward(self, x):
                return self.group_norm(x + 1)

        x = torch.randn(2, 6, 10, 10).float()
        self.torch_and_test(x, Model(), case_name)

    def test_TorchGelu(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.gelu = nn.GELU()

            def forward(self, x):
                return self.gelu(x)

        x = torch.randn(1, 3, 100, 100).float()
        self.torch_and_test(x, Model(), case_name)

    def test_TorchGRU(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.gru = nn.GRU(input_size=100, hidden_size=50, bidirectional=True)

            def forward(self, x, h_0):
                Y, Y_h = self.gru(x, h_0)
                return Y, Y_h

        input = torch.randn(8, 16, 100)
        h_0 = torch.randn(2, 16, 50)
        inputs = (input, h_0)
        self.torch_and_test(inputs, Model(), case_name)

    def test_TorchLayerNorm(self, case_name):
        normalize_shape = [13, 22]

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.layer_norm = nn.LayerNorm(normalize_shape, elementwise_affine=True)

            def forward(self, x):
                x = self.layer_norm(x)
                return x

        input_shape = [14, 25] + normalize_shape
        input_data = torch.randn(input_shape)
        self.torch_and_test(input_data, Model(), case_name)

    def test_TorchLayerNorm2(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.layer_norm = nn.LayerNorm([25, 25])
                nn.init.normal_(self.layer_norm.bias, std=0.01)
                self.conv = nn.Conv2d(32, 32, 3, 1, 1)

            def forward(self, x):
                x = self.layer_norm(x)
                y = self.conv(x)
                return y

        input_shape = [4, 32, 25, 25]
        input_data = torch.randn(input_shape)
        self.torch_and_test(input_data, Model(), case_name)

    def test_TorchMaskedFill(self, case_name):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                y = x.masked_fill(x < 0.2, value=2)
                return y

        input_shape = [2, 3, 100]
        input_data = torch.rand(input_shape)
        self.torch_and_test(input_data, Model(), case_name)

    def test_TorchNonZero(self, case_name):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                y = x.nonzero()
                return y

        input_shape = [1, 4, 64, 32]
        input_data = torch.rand(input_shape)
        self.torch_and_test(input_data, Model(), case_name)

    def test_TorchStd(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                return torch.std(x, -1)

        x = torch.randn(1, 3, 100, 100).float()
        self.torch_and_test(x, Model(), case_name)

    def test_PixelNorm(self, case_name):
        N, C, H, W = 4, 8, 32, 32

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.scale = torch.randn(1, C, 1, 1).float()
                self.bias = torch.randn(1, C, 1, 1).float()

            def forward(self, x):
                m = x.mean(1, keepdim=True)
                var = (x - m).pow(2).mean(1, keepdim=True)
                y = (x - m) / (var + 1e-6).sqrt()
                z = y * self.scale + self.bias
                return z

        x = torch.randn(N, C, H, W).float()
        self.torch_and_test(x, Model(), case_name)

    def test_PixelNorm2(self, case_name):
        N, C, H, W = 4, 8, 32, 32

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.scale = torch.randn(1, C, 1, 1).float()
                self.bias = torch.randn(1, C, 1, 1).float()

            def forward(self, x):
                x = x + 1
                m = x.mean(1, keepdim=True)
                var = (x - m).pow(2).mean(1, keepdim=True)
                y = (x - m) / (var + 1e-6).sqrt()
                z = y * self.scale + self.bias
                return z

        x = torch.randn(N, C, H, W).float()
        self.torch_and_test(x, Model(), case_name)

    def test_ConcatToSpace(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self, no=0):
                super(Model, self).__init__()
                self.conv = nn.Conv2d(160, 8, 3, 1, 1)
                self.no = no

            def case0_f(self, x):
                a = x[:, ::2, ::2, :]
                b = x[:, ::2, 1::2, :]
                c = x[:, 1::2, ::2, :]
                d = x[:, 1::2, 1::2, :]
                y = torch.cat([a, c, b, d], 3)
                return y

            def case1_f(self, x):
                a = x[:, :, ::2, ::2]
                b = x[:, :, ::2, 1::2]
                c = x[:, :, 1::2, ::2]
                d = x[:, :, 1::2, 1::2]
                y = torch.cat([a, c, b, d], 1)
                y = self.conv(y)
                return y

            def case2_f(self, x):
                a = x[:, :, ::2, ::2, :]
                b = x[:, :, ::2, 1::2, :]
                c = x[:, :, 1::2, ::2, :]
                d = x[:, :, 1::2, 1::2, :]
                y = torch.cat([a, c, b, d], 4)
                return y

            def case3_f(self, x):
                a = x[::2, ::2, :]
                b = x[::2, 1::2, :]
                c = x[1::2, ::2, :]
                d = x[1::2, 1::2, :]
                y = torch.cat([a, c, b, d], 2)
                return y

            def forward(self, x):
                if self.no == 0:
                    return self.case0_f(x)
                if self.no == 1:
                    return self.case1_f(x)
                if self.no == 2:
                    return self.case2_f(x)
                if self.no == 3:
                    return self.case3_f(x)

        # case 0
        x = torch.randn(1, 40, 40, 384).float()
        self.torch_and_test(x, Model(0), case_name + "_0")
        # case 1
        x = torch.randn(1, 40, 40, 384).float()
        self.torch_and_test(x, Model(1), case_name + "_1")
        # case 2
        x = torch.randn(4, 2, 40, 40, 80).float()
        self.torch_and_test(x, Model(2), case_name + "_2")
        # case 3
        x = torch.randn(40, 40, 384).float()
        self.torch_and_test(x, Model(3), case_name + "_3")

    def test_ConcatFuse(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(4, 4, 3, 1, 1)
                self.conv2 = nn.Conv2d(4, 4, 3, 1, 1)
                self.conv3 = nn.Conv2d(4, 16, 3, 2, 1)
                self.bias = torch.randn(1, 4, 32, 32)

            def forward(self, x):
                a = self.conv1(x)
                b = self.conv2(x)
                c = self.conv3(x)
                c2 = torch.reshape(c, (1, 4, 32, 32))
                d = torch.cat((a, b, c2), 0)
                e = d + self.bias
                return e

        x = torch.randn(1, 4, 32, 32).float()
        self.torch_and_test(x, Model(), case_name)

    def test_Conv3dTo2d(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.conv3d = nn.Conv3d(3, 96, [10, 4, 4], [10, 4, 4], [0, 0, 0])

            def forward(self, x):
                y = self.conv3d(x)
                return y

        x = torch.randn(1, 3, 10, 320, 320).float()
        self.torch_and_test(x, Model(), case_name)

    def test_GaToSlice(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                a = x[0]
                b = x[1]
                c = x[2]
                d = torch.matmul(a * 0.3, b.transpose(2, 3))
                e = torch.matmul(torch.softmax(d, 3), c)
                return e

        x = torch.randn(3, 36, 12, 49, 32).float()
        self.torch_and_test(x, Model(), case_name)

    def test_PermuteFuse(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.bias = torch.randn(96).float()

            def forward(self, x):
                a = torch.transpose(x, 1, 2)
                b = torch.reshape(a, [1, 96, 1, 160, 160])
                c = torch.permute(b, [0, 2, 3, 4, 1])
                d = c + self.bias
                return d

        x = torch.randn(1, 25600, 96).float()
        self.torch_and_test(x, Model(), case_name)

    def test_PermutePad(self, case_name):

        class Net(torch.nn.Module):

            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                a = torch.permute(x, [0, 1, 3, 4, 2])
                b = torch.nn.functional.pad(a, [0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
                return b

        x = torch.randn(1, 10, 20, 30, 40).float()
        self.torch_and_test(x, Net(), case_name)

    def test_MatMulTranspose(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y):
                a = torch.transpose(y, 2, 3)
                b = torch.matmul(x, a)
                return b

        x = torch.randn(10, 10, 49, 32).float()
        y = torch.randn(10, 10, 49, 32).float()
        self.torch_and_test((x, y), Model(), case_name)

    def test_MatMulTranspose2(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y):
                a = torch.transpose(y, 2, 1)
                b = torch.matmul(x, a)
                return b

        x = torch.randn(10, 10, 49, 32).float()
        y = torch.randn(10, 32, 10, 49).float()
        self.torch_and_test((x, y), Model(), case_name)

    def test_MatMulTranspose3(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y):
                a = torch.transpose(x, 2, 1)
                a1 = torch.transpose(y, 2, 1)
                b = torch.matmul(a, a1)
                return torch.transpose(b, 2, 1)

        x = torch.randn(10, 49, 10, 32).float()
        y = torch.randn(10, 32, 10, 49).float()
        self.torch_and_test((x, y), Model(), case_name)

    def test_ReshapeFuse(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.w0 = torch.randn(1, 3, 49, 49).float()
                self.w1 = torch.randn(1, 529, 1, 49, 49).float()
                self.w2 = torch.randn(1, 3, 49, 49).float()

            def forward(self, x):
                a = x + self.w0
                b = torch.reshape(a, [1, 529, 3, 49, 49])
                c = b + self.w1
                d = torch.reshape(c, [529, 3, 49, 49])
                e = d + self.w2
                return e

        x = torch.randn(529, 3, 49, 49).float()
        self.torch_and_test(x, Model(), case_name)

    def test_SwapDimInner(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                y = torch.cat([x[:, 39:, :, :], x[:, :39, :, :]], 1)
                z = torch.cat([y[:, :, 39:, :], y[:, :, :39, :]], 2)
                return z

        x = torch.randn(1, 42, 42, 384).float()
        self.torch_and_test(x, Model(), case_name)

    def test_TorchWhere(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, a, b):
                x = torch.where(a >= b, a, b)
                return x

        a = torch.randint(-128, 127, (4, 3, 100, 100)).float()
        b = torch.randint(-128, 127, (4, 3, 100, 100)).float()
        self.torch_and_test((a, b), Model(), case_name)

    def test_TorchSize(self, case_name):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                y = torch.ones(x.size(1))
                return torch.add(x, y)

        x = torch.randn(100, 256).float()
        self.torch_and_test(x, Model(), case_name)

    def test_TorchChunk(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.div_num = 2

            def forward(self, x):
                y = torch.negative(x) * 2
                x = torch.cat((x, y), 1)
                x = torch.chunk(x, self.div_num, dim=1)
                x = torch.negative(x[0])
                return x

        x = torch.randn(1, 3, 100, 100).float()
        self.torch_and_test(x, Model(), case_name)

    def test_TorchActivation(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.linear = nn.Linear(100, 200, bias=False)
                self.softplus = nn.Softplus()
                self.hardsigmoid = nn.Hardsigmoid()
                self.prelu = nn.PReLU()
                self.ReLU6 = nn.ReLU6(inplace=True)
                self.mish = nn.Mish()
                self.Softmax = nn.Softmax(dim=1)
                self.Softmax_2d = nn.Softmax2d()

            def forward(self, input):
                #tanh
                x = self.linear(input)
                y0 = torch.tanh(x)
                ##relu
                x = self.linear(input)
                y2 = torch.relu(x)
                ##sigmoid
                x = self.linear(input)
                y1 = torch.sigmoid(x)
                ##leaky_relu
                # x = self.linear(input)
                # y3 = F.leaky_relu(x)
                ##elu
                x = self.linear(input)
                y4 = F.elu(x)
                ##softplus
                x = self.linear(input)
                y5 = self.prelu(x)
                ##hardsigmoid
                x = self.linear(input)
                y6 = self.hardsigmoid(x)
                ##prelu
                x = self.linear(input)
                y7 = self.softplus(x)
                ##relu6
                x = self.linear(input)
                y8 = self.ReLU6(x)
                ##mish
                x = self.linear(input)
                y9 = self.mish(x)
                ##Softmax
                x = self.linear(input)
                y10 = self.Softmax(x)
                ##concat
                y = torch.cat((y0, y1, y2, y4, y5, y6, y7, y8, y9, y10), 0)
                ##Softmax_2d
                x = self.linear(input)
                x = x.unsqueeze(dim=1)
                y2 = self.Softmax_2d(x)
                return y

        x = torch.randn(3, 100, 100).float()
        self.torch_and_test(x, Model(), case_name)

    def test_TorchArgmax(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                a = torch.argmax(x, -1)
                b = torch.argmin(x, -1)
                return a + b

        x = np.arange(0, 128000, step=1, dtype=np.float32)
        np.random.shuffle(x)
        x = torch.from_numpy(x.reshape(40, 40, 80))
        self.torch_and_test(x, Model(), case_name)

    def test_TorchZeroPad(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                ## left:3 right:4 up:4 down:6 postion pad zero
                self.ZeroPad2d = nn.ZeroPad2d(padding=(3, 4, 5, 6))

            def forward(self, x):
                ##input shape = (3, 100, 100)
                x = self.ZeroPad2d(x)  ##output shape = (3, 111, 107)
                return x

        x = torch.randn(4, 3, 100, 100).float()
        self.torch_and_test(x, Model(), case_name)

    def test_TorchConv2d(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.conv2d = nn.Conv2d(in_channels=10,
                                        out_channels=10,
                                        kernel_size=3,
                                        stride=1,
                                        padding=16,
                                        dilation=16)

            def forward(self, x):
                return self.conv2d(x)

        x = torch.randn(1, 10, 64, 64).float()
        self.torch_and_test(x, Model(), case_name)

    def test_TorchConv3dTrans(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.filter = torch.randn(96, 3, 6, 4, 4)

            def forward(self, x):
                x = torch.transpose(x, 1, 2)
                x = F.conv3d(x, self.filter, stride=4)
                return x

        x = torch.randn(1, 6, 3, 640, 640).float()
        self.torch_and_test(x, Model(), case_name)

    def test_StaticDynMixed(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(8, 8, 3, 1, 1)
                self.conv2 = nn.Conv2d(8, 8, 3, 1, 1)

            def forward(self, x):
                a = self.conv1(x) * 100
                (b, indices) = torch.topk(a, 10)
                c = self.conv2(b)
                return c, indices

        x = torch.randn(4, 8, 100, 20).float()
        self.torch_and_test(x, Model(), case_name)

    def test_Add(self, case_name):
        shapes = ([1, 3, 27, 27], [2, 6, 56, 56], [4, 9, 56, 56])
        for i, s in enumerate(shapes):
            a = helper.make_tensor_value_info("a", TensorProto.FLOAT, s)
            b = helper.make_tensor_value_info("b", TensorProto.FLOAT, s)
            output = helper.make_tensor_value_info("output", TensorProto.FLOAT, s)
            add_def = helper.make_node("Add", inputs=["a", "b"], outputs=["output"])
            graph_def = helper.make_graph([add_def], "{}_{}".format(case_name, i), [a, b], [output])
            if 0:  # the follow code used to test local layer, but local layer compare faild. so I just closed it, and commit it.
                c = helper.make_tensor_value_info("c", TensorProto.FLOAT, s)
                add2_out = helper.make_tensor_value_info("add2_out", TensorProto.FLOAT, s)
                add2_def = helper.make_node("Add", inputs=["c", "output"], outputs=["add2_out"])
                graph_def = helper.make_graph([add_def, add2_def], "{}_{}".format(case_name, i),
                                              [a, b, c], [add2_out])
            self.onnx_and_test(graph_def)

    def test_tmp(self, case_name):
        shapes = ([16, 9, 67, 323], )
        bcast_dims = ([[3]], )
        for i, s in enumerate(shapes):
            for dims in bcast_dims[i]:
                bcast_s = s[::]
                for dim in dims:
                    bcast_s[dim] = 1
                a = helper.make_tensor_value_info("a", TensorProto.FLOAT, bcast_s)
                b = helper.make_tensor_value_info("b", TensorProto.FLOAT, s)
                c = helper.make_tensor_value_info("c", TensorProto.FLOAT, bcast_s)
                output = helper.make_tensor_value_info("output", TensorProto.FLOAT, s)
                add_def = helper.make_node("Add", inputs=["a", "b"], outputs=["x"])
                add_def_2 = helper.make_node("Add", inputs=["x", "c"], outputs=["output"])
                graph_def = helper.make_graph([add_def, add_def_2],
                                              "{}_{}_{}".format(case_name, i,
                                                                "".join(map(str, dims))), [a, b, c],
                                              [output])
                self.onnx_and_test(graph_def)

    def test_AddBcast(self, case_name):
        shapes = (
            [
                10,
            ],
            [9, 12],
            [7, 14, 15],
            [16, 9, 323, 67],
            #   [4, 7, 38, 6, 4],
            #   [3, 3, 11, 3, 4, 5],
        )
        bcast_dims = (
            [[0]],
            [[0], [1]],
            [[0], [2], [0, 2], [0, 1, 2]],
            [[0], [2], [0, 2], [0, 3], [2, 3], [0, 2, 3], [0, 1, 2, 3]],
            #   [[0], [2], [0, 2], [3, 4], [2, 3, 4]],
            #   [[0], [2], [0, 2], [3, 4, 5], [2, 3, 4, 5]],
        )
        for i, s in enumerate(shapes):
            for dims in bcast_dims[i]:
                bcast_s = s[::]
                for dim in dims:
                    bcast_s[dim] = 1
                a = helper.make_tensor_value_info("a", TensorProto.FLOAT, bcast_s)
                b = helper.make_tensor_value_info("b", TensorProto.FLOAT, s)
                c = helper.make_tensor_value_info("c", TensorProto.FLOAT, bcast_s)
                output = helper.make_tensor_value_info("output", TensorProto.FLOAT, s)
                add_def = helper.make_node("Add", inputs=["a", "b"], outputs=["x"])
                add_def_2 = helper.make_node("Add", inputs=["x", "c"], outputs=["output"])
                graph_def = helper.make_graph([add_def, add_def_2],
                                              "{}_{}_{}".format(case_name, i,
                                                                "".join(map(str, dims))), [a, b, c],
                                              [output])
                self.onnx_and_test(graph_def)

    def test_AddBcast2(self, case_name):
        shapes = (
            [4, 16, 1, 11],
            [3, 6, 1, 12],
        )
        bcast_shapes = (
            [4, 16, 9, 1],
            [1, 6, 21, 1],
        )
        out_shapes = (
            [4, 16, 9, 11],
            [3, 6, 21, 12],
        )
        for i, s in enumerate(shapes):
            a = helper.make_tensor_value_info("a", TensorProto.FLOAT, bcast_shapes[i])
            b = helper.make_tensor_value_info("b", TensorProto.FLOAT, s)
            c = helper.make_tensor_value_info("c", TensorProto.FLOAT, bcast_shapes[i])
            output = helper.make_tensor_value_info("output", TensorProto.FLOAT, out_shapes[i])
            add_def = helper.make_node("Add", inputs=["a", "b"], outputs=["x"])
            add_def_2 = helper.make_node("Add", inputs=["x", "c"], outputs=["output"])
            graph_def = helper.make_graph([add_def, add_def_2], "{}_{}".format(
                case_name,
                i,
            ), [a, b, c], [output])
            self.onnx_and_test(graph_def)

    # Known issue: failed cases.
    # The first one has a too large w shape and needs broadcast so cannot reshape. W is too large for local mem.
    # Stuck in finding layer groups: is_layer_group_valid->group_one_layer_proc->getGlobalLayerCycle->global codegen->tensor_split_nh->while, as TPU_KERNEL_ASSERT is not enabled.
    # The second one has 6-dim broadcast data and goes global. Most dims cannot be merged so the shape_dim is 5 when merging ends.
    # Forcibly assigning a 5-dim shape into a dim4 shape, it messes up the n-dim shape and raises assert for "n-dim cannot broadcast" in calling local function.
    def test_AddBcast3(self, case_name):
        shapes = ([4, 9, 56, 56 * 27 * 16], )
        bcast_shapes = ([4, 1, 56, 56 * 27 * 16], )
        out_shapes = ([4, 9, 56, 56 * 27 * 16], )
        shapes = ([4, 9, 10, 11, 12, 7], )
        bcast_shapes = ([4, 1, 10, 1, 12, 7], )
        out_shapes = ([4, 9, 10, 11, 12, 7], )
        for i, s in enumerate(shapes):
            a = helper.make_tensor_value_info("a", TensorProto.FLOAT, bcast_shapes[i])
            b = helper.make_tensor_value_info("b", TensorProto.FLOAT, s)
            c = helper.make_tensor_value_info("c", TensorProto.FLOAT, bcast_shapes[i])
            output = helper.make_tensor_value_info("output", TensorProto.FLOAT, out_shapes[i])
            add_def = helper.make_node("Add", inputs=["a", "b"], outputs=["x"])
            add_def_2 = helper.make_node("Add", inputs=["c", "x"], outputs=["output"])
            graph_def = helper.make_graph([add_def, add_def_2], "{}_{}".format(
                case_name,
                i,
            ), [a, b, c], [output])
            self.onnx_and_test(graph_def)

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
        self.onnx_and_test(graph_def)

    def test_BCastAdd(self, case_name):
        # 18xx: only broadcast right opd and broadcast continuous axis is supported
        if self.is_cv18xx:
            input_shape = {"input1": [2, 3, 27, 27], "input2": [2, 1, 1, 27]}
        else:
            input_shape = {"input1": [1, 3, 1, 27], "input2": [2, 1, 27, 1]}
        output_shape = [2, 3, 27, 27]
        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, x) for k, x in input_shape.items()
        ]
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
        add_def = helper.make_node("Add", inputs=list(input_shape.keys()), outputs=["output"])
        graph_def = helper.make_graph([add_def], case_name, inputs, [output])
        self.onnx_and_test(graph_def)

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
        self.onnx_and_test(graph_def)

    def test_LSTM(self, case_name):
        seq_length = 75
        batch_size = 4
        num_dir = 2
        input_size = 128
        hidden_size = 64
        direction = 'forward' if num_dir == 1 else 'bidirectional'
        #layout = 0
        w_data = np.random.randn(num_dir, 4 * hidden_size, input_size).astype(np.float32)
        r_data = np.random.randn(num_dir, 4 * hidden_size, hidden_size).astype(np.float32)
        b_data = np.random.randn(num_dir, 8 * hidden_size).astype(np.float32)
        h0_data = np.random.randn(num_dir, batch_size, hidden_size).astype(np.float32)
        c0_data = np.random.randn(num_dir, batch_size, hidden_size).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT,
                                              [seq_length, batch_size, input_size])

        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT,
                                          [seq_length, num_dir, batch_size, hidden_size])

        w = helper.make_tensor('w', TensorProto.FLOAT, w_data.shape, w_data)
        r = helper.make_tensor('r', TensorProto.FLOAT, r_data.shape, r_data)
        b = helper.make_tensor('b', TensorProto.FLOAT, b_data.shape, b_data)
        h0 = helper.make_tensor('h0', TensorProto.FLOAT, h0_data.shape, h0_data)
        c0 = helper.make_tensor('c0', TensorProto.FLOAT, c0_data.shape, c0_data)

        node_def = helper.make_node(
            "LSTM",
            inputs=['input', 'w', 'r', 'b', '', 'h0', 'c0'],
            outputs=['Y'],
            direction=direction,
            hidden_size=hidden_size,
        )
        graph_def = helper.make_graph([node_def],
                                      case_name, [input], [Y],
                                      initializer=[w, r, b, h0, c0])
        self.onnx_and_test(graph_def)

    def test_LSTM2(self, case_name):
        seq_length = 75
        batch_size = 2
        num_dir = 2
        input_size = 128
        hidden_size = 64
        direction = 'forward' if num_dir == 1 else 'bidirectional'
        w_data = np.random.rand(num_dir, 4 * hidden_size, input_size).astype(np.float32)
        r_data = np.random.rand(num_dir, 4 * hidden_size, hidden_size).astype(np.float32)
        b_data = np.random.rand(num_dir, 8 * hidden_size).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT,
                                              [seq_length, batch_size, input_size])
        h0 = helper.make_tensor_value_info('h0', TensorProto.FLOAT,
                                           [num_dir, batch_size, hidden_size])
        c0 = helper.make_tensor_value_info('c0', TensorProto.FLOAT,
                                           [num_dir, batch_size, hidden_size])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT,
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
        )
        lstm_def = helper.make_node(
            "LSTM",
            inputs=['input', 'w', 'r', 'b', '', 'h0', 'c0'],
            outputs=['Y', 'Y_h', 'Y_c'],
            direction=direction,
            hidden_size=hidden_size,
        )
        graph_def = helper.make_graph([lstm_def],
                                      case_name, [input, h0, c0], [Y, Y_h, Y_c],
                                      initializer=[w_value, r_value, b_value])
        self.onnx_and_test(graph_def)

    def test_LSTM3(self, case_name):
        seq_length = 75
        batch_size = 2
        num_dir = 2
        input_size = 128
        hidden_size = 64
        direction = 'forward' if num_dir == 1 else 'bidirectional'
        w_data = np.random.rand(num_dir, 4 * hidden_size, input_size).astype(np.float32)
        r_data = np.random.rand(num_dir, 4 * hidden_size, hidden_size).astype(np.float32)
        b_data = np.random.rand(num_dir, 8 * hidden_size).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT,
                                              [seq_length, batch_size, input_size])
        h0 = helper.make_tensor_value_info('h0', TensorProto.FLOAT,
                                           [num_dir, batch_size, hidden_size])
        c0 = helper.make_tensor_value_info('c0', TensorProto.FLOAT,
                                           [num_dir, batch_size, hidden_size])
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
        )
        lstm_def = helper.make_node(
            "LSTM",
            inputs=['input', 'w', 'r', 'b', '', 'h0', 'c0'],
            outputs=['', 'Y_h', 'Y_c'],
            direction=direction,
            hidden_size=hidden_size,
        )
        graph_def = helper.make_graph([lstm_def],
                                      case_name, [input, h0, c0], [Y_h, Y_c],
                                      initializer=[w_value, r_value, b_value])
        self.onnx_and_test(graph_def)

    def test_BCastMul(self, case_name):
        input_shape = {"input1": [1, 3, 1, 27], "input2": [2, 1, 27, 1]}
        output_shape = [2, 3, 27, 27]
        if self.is_cv18xx:
            ## 18xx: only broadcast right opd and broadcast continuous axis is supported
            input_shape = {"input1": [2, 3, 4, 27], "input2": [2, 3, 1, 1]}
            output_shape = [2, 3, 4, 27]
        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, x) for k, x in input_shape.items()
        ]
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
        mul_def = helper.make_node("Mul", inputs=list(input_shape.keys()), outputs=["output"])
        graph_def = helper.make_graph([mul_def], case_name, inputs, [output])
        self.onnx_and_test(graph_def)

    def test_BCastMulCst(self, case_name):
        input_shape = [1, 127, 270, 28]
        constant_shape = [2, 1, 1, 28]
        output_shape = [2, 127, 270, 28]
        if self.is_cv18xx:
            #18xx: only broadcast right opd and broadcast continuous axis is supported
            input_shape = [2, 127, 270, 28]
            constant_shape = [1, 1, 1, 28]
            output_shape = [2, 127, 270, 28]
        inputs = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        constant = helper.make_tensor(
            "constant",
            TensorProto.FLOAT,
            constant_shape,
            np.random.rand(*constant_shape).astype(np.float32),
        )
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
        mul_def = helper.make_node("Mul", inputs=["input", "constant"], outputs=["output"])

        graph_def = helper.make_graph([mul_def],
                                      case_name, [inputs], [output],
                                      initializer=[constant])
        self.onnx_and_test(graph_def)

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
        self.onnx_and_test(graph_def, input_data=input_data)

    def test_Sum(self, case_name):
        input_shape = [1, 3, 27, 27]
        output_shape = [1, 3, 27, 27]

        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, input_shape)
        input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, input_shape)
        input3 = helper.make_tensor_value_info('input3', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        #test three input
        sum_def = helper.make_node(
            'Sum',  # node name
            ['input1', 'input2', 'input3'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [sum_def],
            case_name,
            [input1, input2, input3],
            [output],
        )
        self.onnx_and_test(graph_def)

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
        self.onnx_and_test(graph_def)

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
        self.onnx_and_test(graph_def)

    def test_Sub2(self, case_name):
        input1_shape = [4, 3, 27, 27]
        input2_shape = [4, 3, 1, 27]
        output_shape = [4, 3, 27, 27]

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
        self.onnx_and_test(graph_def)

    def test_SubBcast(self, case_name):
        shapes = ([4, 9, 23, 39], )
        bcast_dims = ([[0], [2], [0, 2]], )
        for i, s in enumerate(shapes):
            for dims in bcast_dims[i]:
                bcast_s = s[::]
                for dim in dims:
                    bcast_s[dim] = 1
                a = helper.make_tensor_value_info("a", TensorProto.FLOAT, bcast_s)
                b = helper.make_tensor_value_info("b", TensorProto.FLOAT, s)
                c = helper.make_tensor_value_info("c", TensorProto.FLOAT, bcast_s)
                output = helper.make_tensor_value_info("output", TensorProto.FLOAT, s)
                sub_def = helper.make_node("Sub", inputs=["a", "b"], outputs=["x"])
                sub_def_2 = helper.make_node("Sub", inputs=["x", "c"], outputs=["output"])
                graph_def = helper.make_graph([sub_def, sub_def_2],
                                              "{}_{}_{}".format(case_name, i,
                                                                "".join(map(str, dims))), [a, b, c],
                                              [output])
                self.onnx_and_test(graph_def)

    def test_SubBcast2(self, case_name):
        shapes = ([6, 7, 1, 11], )
        bcast_shapes = ([1, 7, 13, 11], )
        out_shapes = ([6, 7, 13, 11], )
        for i, s in enumerate(shapes):
            a = helper.make_tensor_value_info("a", TensorProto.FLOAT, bcast_shapes[i])
            b = helper.make_tensor_value_info("b", TensorProto.FLOAT, s)
            c = helper.make_tensor_value_info("c", TensorProto.FLOAT, bcast_shapes[i])
            output = helper.make_tensor_value_info("output", TensorProto.FLOAT, out_shapes[i])
            sub_def = helper.make_node("Sub", inputs=["a", "b"], outputs=["x"])
            sub_def_2 = helper.make_node("Sub", inputs=["x", "c"], outputs=["output"])
            graph_def = helper.make_graph([sub_def, sub_def_2], "{}_{}".format(
                case_name,
                i,
            ), [a, b, c], [output])
            self.onnx_and_test(graph_def)

    def test_SubConst(self, case_name):
        input_shape = [4, 3, 27, 27]
        output_shape = [4, 3, 27, 27]

        input0 = helper.make_tensor_value_info('input0', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        w_data = np.random.rand(*input_shape).astype(np.float32)
        w_value = helper.make_tensor(
            name='w',
            data_type=onnx.TensorProto.FLOAT,
            dims=w_data.shape,
            vals=w_data.flatten(),
        )

        sub_def = helper.make_node(
            'Sub',  # node name
            ['input0', 'w'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph([sub_def],
                                      case_name, [input0], [output],
                                      initializer=[w_value])
        self.onnx_and_test(graph_def)

    def test_SubConst2(self, case_name):
        input1_shape = [4, 3, 27, 27]
        w_shape = [4, 3, 1, 27]
        output_shape = [4, 3, 27, 27]

        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, input1_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        w_data = np.random.rand(*w_shape).astype(np.float32)
        w_value = helper.make_tensor(
            name='w',
            data_type=onnx.TensorProto.FLOAT,
            dims=w_data.shape,
            vals=w_data.flatten(),
        )

        sub_def = helper.make_node(
            'Sub',  # node name
            ['input1', 'w'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph([sub_def],
                                      case_name, [input1], [output],
                                      initializer=[w_value])
        self.onnx_and_test(graph_def)

    def test_Expand(self, case_name):
        input_shape = [1, 3, 1, 16]
        output_shape = [1, 3, 16, 16]
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
        self.onnx_and_test(graph_def)

    def test_Expand2(self, case_name):
        input_shape = [1, 16]
        output_shape = [1, 16, 16]
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
        self.onnx_and_test(graph_def)

    def test_Max(self, case_name):
        input_shape = {"input1": [1, 85, 32, 8], "input2": [1, 85, 32, 8]}
        output_shape = [1, 85, 32, 8]

        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, x) for k, x in input_shape.items()
        ]
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        max_def = helper.make_node(case_name, inputs=list(input_shape.keys()), outputs=["output"])

        graph_def = helper.make_graph([max_def], case_name, inputs, [output])
        self.onnx_and_test(graph_def)

    def test_MaxBcast(self, case_name):
        shapes = ([6, 8, 39, 41], )
        bcast_dims = ([[0], [2], [0, 2]], )
        for i, s in enumerate(shapes):
            for dims in bcast_dims[i]:
                bcast_s = s[::]
                for dim in dims:
                    bcast_s[dim] = 1
                a = helper.make_tensor_value_info("a", TensorProto.FLOAT, bcast_s)
                b = helper.make_tensor_value_info("b", TensorProto.FLOAT, s)
                c = helper.make_tensor_value_info("c", TensorProto.FLOAT, bcast_s)
                output = helper.make_tensor_value_info("output", TensorProto.FLOAT, s)
                Max_def = helper.make_node("Max", inputs=["a", "b"], outputs=["x"])
                Max_def_2 = helper.make_node("Max", inputs=["x", "c"], outputs=["output"])
                graph_def = helper.make_graph([Max_def, Max_def_2],
                                              "{}_{}_{}".format(case_name, i,
                                                                "".join(map(str, dims))), [a, b, c],
                                              [output])
                self.onnx_and_test(graph_def)

    def test_Min(self, case_name):
        input_shape = {"input1": [1, 85, 32, 8], "input2": [1, 85, 32, 8]}
        output_shape = [1, 85, 32, 8]
        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, x) for k, x in input_shape.items()
        ]
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        min_def = helper.make_node(case_name, inputs=list(input_shape.keys()), outputs=["output"])

        graph_def = helper.make_graph([min_def], case_name, inputs, [output])
        self.onnx_and_test(graph_def)

    def test_MinBcast(self, case_name):
        shapes = ([5, 11, 23, 27], )
        bcast_dims = ([[0], [2], [0, 2]], )
        for i, s in enumerate(shapes):
            for dims in bcast_dims[i]:
                bcast_s = s[::]
                for dim in dims:
                    bcast_s[dim] = 1
                a = helper.make_tensor_value_info("a", TensorProto.FLOAT, bcast_s)
                b = helper.make_tensor_value_info("b", TensorProto.FLOAT, s)
                c = helper.make_tensor_value_info("c", TensorProto.FLOAT, bcast_s)
                output = helper.make_tensor_value_info("output", TensorProto.FLOAT, s)
                Min_def = helper.make_node("Min", inputs=["a", "b"], outputs=["x"])
                Min_def_2 = helper.make_node("Min", inputs=["x", "c"], outputs=["output"])
                graph_def = helper.make_graph([Min_def, Min_def_2],
                                              "{}_{}_{}".format(case_name, i,
                                                                "".join(map(str, dims))), [a, b, c],
                                              [output])
                self.onnx_and_test(graph_def)

    def test_Abs(self, case_name):
        input_shape = [1, 16, 64, 64]
        output_shape = [1, 16, 64, 64]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        abs_def = helper.make_node(
            "Abs",
            inputs=['input'],
            outputs=['output'],
        )
        graph_def = helper.make_graph([abs_def], case_name, [input], [output])
        self.onnx_and_test(graph_def)

    def test_Reciprocal(self, case_name):
        input_shape = [4, 3, 224, 224]
        node_def = helper.make_node(
            "Reciprocal",  # node name
            ['input'],  # inputs
            ['output'],  # outputs
        )
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def],
            case_name,
            [input],
            [output],
        )
        input_data = np.random.rand(*input_shape).astype(np.float32) + 0.5
        # avoid divide 0
        input_data[input_data == 0] = 1
        self.onnx_and_test(graph_def, input_data={"input": input_data})

    def test_PRelu(self, case_name):
        input_shape = [3, 128, 100, 100]
        slope_shape = [1, 128, 1, 1]
        output_shape = [3, 128, 100, 100]
        scales0 = np.random.rand(*slope_shape).astype(np.float32)
        scales1 = np.negative(np.abs(scales0))
        scales_case = [scales0, scales1]
        for i, s in enumerate(scales_case):
            slope = helper.make_tensor(
                name="slope",
                data_type=onnx.TensorProto.FLOAT,
                dims=slope_shape,
                vals=s,
            )
            inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)]
            outputs = [helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)]
            prelu_def = helper.make_node("PRelu", ["input", "slope"], ["output"])
            graph_def = helper.make_graph([prelu_def],
                                          "{}_{}".format(case_name, i),
                                          inputs,
                                          outputs,
                                          initializer=[slope])
            self.onnx_and_test(graph_def)

    def test_Sqrt(self, case_name):
        shape = [3, 5, 100, 100]
        inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)]
        outputs = [helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)]
        sqrt_def = helper.make_node("Sqrt", ["input"], ["output"])
        graph_def = helper.make_graph([sqrt_def], case_name, inputs, outputs)
        input_data = np.abs(np.random.randn(*shape).astype(np.float32))
        self.onnx_and_test(graph_def, input_data={"input": input_data})

    def test_Pow1(self, case_name):
        shape = [1, 3, 27, 27]
        input_data = np.abs(np.random.randn(*shape).astype(np.float32)) + 1e-6
        input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 27, 27])
        constant = helper.make_tensor("constant", TensorProto.FLOAT, [1],
                                      np.array([1.2]).astype(np.float32))
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)
        pow_def = helper.make_node("Pow", inputs=["input", "constant"], outputs=["output"])
        graph_def = helper.make_graph([pow_def],
                                      case_name, [input], [output],
                                      initializer=[constant])
        self.onnx_and_test(graph_def, input_data={"input": input_data})

    def test_Pow2(self, case_name):
        shape = [1, 3, 27, 27]
        input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 27, 27])
        constant = helper.make_tensor("constant", TensorProto.FLOAT, [1],
                                      np.array([1.2]).astype(np.float32))
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)
        pow_def = helper.make_node("Pow", inputs=["constant", "input"], outputs=["output"])
        graph_def = helper.make_graph([pow_def],
                                      case_name, [input], [output],
                                      initializer=[constant])
        self.onnx_and_test(graph_def)

    def test_CompareCst(self, case_name):
        shape = [1, 3, 27, 27]
        input = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
        constant = helper.make_tensor("constant", TensorProto.FLOAT, [1],
                                      np.array([0.5]).astype(np.float32))
        output = helper.make_tensor_value_info("output", TensorProto.BOOL, shape)
        # "Equal" need not to be tested since equal op between floating number may be invalid
        for cmp_type in ("Greater", "GreaterOrEqual", "Less", "LessOrEqual"):
            cmp_def = helper.make_node(cmp_type, inputs=["input", "constant"], outputs=["output"])
            graph_def = helper.make_graph([cmp_def],
                                          case_name, [input], [output],
                                          initializer=[constant])
            self.onnx_and_test(graph_def)
            print("====== TEST {} Success ======".format(cmp_type))

    def test_Compare(self, case_name):
        shape = [1, 3, 27, 27]
        input_shape = {"input1": shape, "input2": shape}
        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, x) for k, x in input_shape.items()
        ]
        output = helper.make_tensor_value_info("output", TensorProto.BOOL, shape)
        # "Equal" need not to be tested since equal op between floating number may be invalid
        for cmp_type in ("Greater", "GreaterOrEqual", "Less", "LessOrEqual"):
            cmp_def = helper.make_node(cmp_type, inputs=["input1", "input2"], outputs=["output"])
            graph_def = helper.make_graph([cmp_def], case_name, inputs, [output])
            self.onnx_and_test(graph_def)
            print("====== TEST {} Success ======".format(cmp_type))

    def test_Compare2(self, case_name):
        input_shape = {"input1": [1, 3, 1, 27], "input2": [1, 3, 27, 1]}
        output_shape = [1, 3, 27, 27]
        input_data = {
            "input1": np.random.randn(*input_shape["input1"]).astype(np.float32),
            "input2": np.random.randn(*input_shape["input2"]).astype(np.float32)
        }
        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, x) for k, x in input_shape.items()
        ]
        output = helper.make_tensor_value_info("output", TensorProto.BOOL, output_shape)
        # "Equal" need not to be tested since equal op between floating number may be invalid
        for cmp_type in ("Greater", "GreaterOrEqual", "Less", "LessOrEqual"):
            cmp_def = helper.make_node(cmp_type,
                                       inputs=list(input_shape.keys()),
                                       outputs=["output"])
            graph_def = helper.make_graph([cmp_def], case_name, inputs, [output])
            self.onnx_and_test(graph_def, input_data=input_data)
            print("====== TEST {} Success ======".format(cmp_type))

    def test_Einsum(self, case_name):
        input_shape = {"input1": [1, 26, 12, 26], "input2": [12, 26, 312]}
        output_shape = [1, 26, 312]

        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, x) for k, x in input_shape.items()
        ]
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        einsum_def = helper.make_node(
            "Einsum",
            inputs=['input1', 'input2'],
            outputs=['output'],
            equation='bfnd,ndh->bfh',
        )

        graph_def = helper.make_graph([einsum_def], case_name, inputs, [output])
        self.onnx_and_test(graph_def)

    def test_Einsum2(self, case_name):
        input_shape = [1, 26, 12, 26]
        filter_shape = [12, 26, 312]
        output_shape = [1, 26, 312]

        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        einsum_def = helper.make_node(
            "Einsum",
            inputs=['input', 'weight'],
            outputs=['output'],
            equation='bfnd,ndh->bfh',
        )

        graph_def = helper.make_graph([einsum_def],
                                      case_name, [input], [output],
                                      initializer=[weight])
        self.onnx_and_test(graph_def)

    def test_Elu(self, case_name):
        oc = 32
        input_shape = [1, 16, 100, 100]
        filter_shape = [oc, 16, 3, 3]
        output_shape = [1, oc, 100, 100]
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

        elu_def = helper.make_node("Elu", inputs=['conv_output'], outputs=['output'], alpha=0.67)

        graph_def = helper.make_graph([conv_def, elu_def],
                                      case_name, [input], [output],
                                      initializer=[weight, bias])

        self.onnx_and_test(graph_def)

    def test_Erf(self, case_name):
        input_shape = [10, 3, 32, 32]
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
        exp_def = helper.make_node(case_name, inputs=['input'], outputs=['output'])
        graph_def = helper.make_graph([exp_def], case_name, [input], [output])
        self.onnx_and_test(graph_def)

    def test_TopK(self, case_name):
        shape = [10, 1000]
        const = 500
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, shape)
        K = helper.make_tensor("K", TensorProto.INT64, [1], np.array([const]).astype(np.int64))
        o_shape = list(shape)
        o_shape[-1] = const
        Y_Value = helper.make_tensor_value_info('Y_Value', TensorProto.FLOAT, o_shape)
        Y_Index = helper.make_tensor_value_info('Y_Index', TensorProto.INT64, o_shape)
        topk_node = helper.make_node('TopK', ['X', 'K'], ['Y_Value', 'Y_Index'],
                                     axis=-1,
                                     largest=True)
        graph_def = helper.make_graph([topk_node],
                                      case_name, [X], [Y_Value, Y_Index],
                                      initializer=[K])
        self.onnx_and_test(graph_def)

    def test_QDQConv(self, case_name):
        oc = 32
        input_shape = [10, 3, 224, 224]
        filter_shape = [oc, input_shape[1], 3, 3]
        output_shape = [10, oc, 224, 224]
        kernel = [3, 3]
        padding = [1, 1, 1, 1]
        stride = [1, 1]
        dilation = [1, 1]
        groups = 1

        y_scale_data = np.random.uniform(0, 5)
        # y_zero_point_data = np.random.randint(0, 255)
        y_zero_point_data = 0
        x_scale_data = deepcopy(y_scale_data)
        x_zero_point_data = deepcopy(y_zero_point_data)

        weight_data = np.random.randint(-128, 127, filter_shape)
        bias_data = np.random.randn(output_shape[1]).astype(np.float32)

        weight_x_scale_data = np.random.uniform(1e-5, 5, oc)
        weight_x_zero_point_data = np.zeros(oc).astype(np.int32)

        output_y_scale_data = np.random.uniform(1e-5, 2)
        output_y_zero_point_data = 0
        output_x_scale_data = deepcopy(output_y_scale_data)
        output_x_zero_point_data = deepcopy(output_y_zero_point_data)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        weight = helper.make_tensor('weight', TensorProto.INT8, filter_shape, weight_data)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, list(bias_data.shape), bias_data)

        y_scale = helper.make_tensor("y_scale", TensorProto.FLOAT, [1], [y_scale_data])
        y_zero_point = helper.make_tensor("y_zero_point", TensorProto.INT8, [1],
                                          [y_zero_point_data])
        x_scale = helper.make_tensor("x_scale", TensorProto.FLOAT, [1], [x_scale_data])
        x_zero_point = helper.make_tensor("x_zero_point", TensorProto.INT8, [1],
                                          [x_zero_point_data])

        weight_x_scale = helper.make_tensor("weight_x_scale", TensorProto.FLOAT, [oc],
                                            weight_x_scale_data)
        weight_x_zero_point = helper.make_tensor("weight_x_zero_point", TensorProto.INT8, [oc],
                                                 weight_x_zero_point_data)

        quant_node = helper.make_node("QuantizeLinear", ['input', 'y_scale', 'y_zero_point'], ['a'],
                                      axis=0)
        dequant_node = helper.make_node("DequantizeLinear", ['a', 'x_scale', 'x_zero_point'], ['b'],
                                        axis=0)

        weight_dequant_node = helper.make_node("DequantizeLinear",
                                               ['weight', 'weight_x_scale', 'weight_x_zero_point'],
                                               ['weight_output'],
                                               axis=0)

        conv_def = helper.make_node(
            "Conv",
            inputs=['b', 'weight_output', 'bias'],
            outputs=['conv_output'],
            kernel_shape=kernel,
            pads=padding,
            strides=stride,
            dilations=dilation,
            group=groups,
        )

        output_y_scale = helper.make_tensor("output_y_scale", TensorProto.FLOAT, [1],
                                            [output_y_scale_data])
        output_y_zero_point = helper.make_tensor("output_y_zero_point", TensorProto.INT8, [1],
                                                 [output_y_zero_point_data])
        output_x_scale = helper.make_tensor("output_x_scale", TensorProto.FLOAT, [1],
                                            [output_x_scale_data])
        output_x_zero_point = helper.make_tensor("output_x_zero_point", TensorProto.INT8, [1],
                                                 [output_x_zero_point_data])

        output_quant_node = helper.make_node(
            "QuantizeLinear", ['conv_output', 'output_y_scale', 'output_y_zero_point'], ['c'],
            axis=0)
        output_dequant_node = helper.make_node("DequantizeLinear",
                                               ['c', 'output_x_scale', 'output_x_zero_point'],
                                               ['output'],
                                               axis=0)

        graph_def = helper.make_graph([
            quant_node, dequant_node, weight_dequant_node, conv_def, output_quant_node,
            output_dequant_node
        ],
                                      case_name, [input], [output],
                                      initializer=[
                                          y_scale, y_zero_point, x_scale, x_zero_point,
                                          weight_x_scale, weight_x_zero_point, weight, bias,
                                          output_x_scale, output_x_zero_point, output_y_scale,
                                          output_y_zero_point
                                      ])

        self.onnx_and_test(graph_def)

    def test_QDQ(self, case_name):
        input_shape = [10, 3, 224, 224]
        y_scale_data = np.random.uniform(0, 1)
        y_zero_point_data = np.random.randint(0, 255)
        x_scale_data = deepcopy(y_scale_data)
        x_zero_point_data = deepcopy(y_zero_point_data)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
        y_scale = helper.make_tensor("y_scale", TensorProto.FLOAT, [1], [y_scale_data])
        y_zero_point = helper.make_tensor("y_zero_point", TensorProto.INT8, [1],
                                          [y_zero_point_data])
        x_scale = helper.make_tensor("x_scale", TensorProto.FLOAT, [1], [x_scale_data])
        x_zero_point = helper.make_tensor("x_zero_point", TensorProto.INT8, [1],
                                          [x_zero_point_data])
        quant_node = helper.make_node("QuantizeLinear", ['input', 'y_scale', 'y_zero_point'], ['a'])
        dequant_node = helper.make_node("DequantizeLinear", ['a', 'x_scale', 'x_zero_point'],
                                        ['a_output'])

        log_node = helper.make_node("Log", inputs=['a_output'], outputs=['output'])

        graph_def = helper.make_graph([quant_node, dequant_node, log_node],
                                      case_name, [input], [output],
                                      initializer=[y_scale, y_zero_point, x_scale, x_zero_point])

        self.onnx_and_test(graph_def)

    def test_ReduceTranspose(self, case_name):
        input_shape = [8, 16, 32, 64]
        transpose_order = [1, 0, 2, 3]
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        transpose_output_shape = [
            input_shape[transpose_order[i]] for i in range(len(transpose_order))
        ]
        transpose_output = helper.make_tensor_value_info('transpose_output', TensorProto.FLOAT,
                                                         transpose_output_shape)
        transpose_def = helper.make_node("Transpose",
                                         inputs=['input'],
                                         outputs=['transpose_output'],
                                         perm=transpose_order)
        reduce_keepdims = False
        reduce_axes = [1]
        reduce_output_shape = []
        if len(reduce_axes) == len(transpose_output_shape):
            reduce_output_shape = []
        else:
            for i in range(len(transpose_output_shape)):
                keep_sign = True
                for j in range(len(reduce_axes)):
                    if i == reduce_axes[j]:
                        keep_sign = False
                        break
                if keep_sign:
                    reduce_output_shape.append(transpose_output_shape[i])
        reduce_output = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                                      reduce_output_shape)
        reduce_mean_def = helper.make_node(
            'ReduceMean',
            ['transpose_output'],
            ['output'],
            keepdims=reduce_keepdims,
            axes=reduce_axes,
        )
        graph_def = helper.make_graph([transpose_def, reduce_mean_def], case_name, [input],
                                      [reduce_output])
        self.onnx_and_test(graph_def)

    def test_TransposeArg(self, case_name):
        input_shape = [8, 16, 32, 64]
        transpose_order = [0, 2, 1, 3]
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        transpose_def = helper.make_node("Transpose",
                                         inputs=['input'],
                                         outputs=['transpose_output'],
                                         perm=transpose_order)
        arg_keepdims = False
        arg_axis = 1
        reduce_output_shape = [8, 16, 64]
        arg_output = helper.make_tensor_value_info('output', TensorProto.INT64, reduce_output_shape)
        arg_max_def = helper.make_node(
            'ArgMax',
            ['transpose_output'],
            ['output'],
            keepdims=arg_keepdims,
            axis=arg_axis,
            select_last_index=1
        )
        graph_def = helper.make_graph([transpose_def, arg_max_def], case_name, [input],
                                      [arg_output])
        self.onnx_and_test(graph_def)

    def test_ArgError(self, case_name):
        input_shape = [1, 1000, 1, 1]
        output_shape = [1, 1, 1, 1]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output1 = helper.make_tensor_value_info('o_max', TensorProto.INT64, output_shape)
        x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, input_shape)
        relu_def = helper.make_node("Relu", inputs=['input'], outputs=['x2'])
        arg_max = helper.make_node(
            'ArgMax',
            ['x2'],
            ['o_max'],
            keepdims=1,
            axis=1,
            select_last_index=1
        )
        graph_def = helper.make_graph([relu_def, arg_max], "{}".format(case_name), [input],
                                      [output1, x2])
        self.onnx_and_test(graph_def)

    def test_ArgReducefull(self, case_name):
        input_shape = [2, 3, 4]
        arg_axis = 0
        reduce_axes = [0]
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output_shape = [1, 3, 4]
        arg_output = helper.make_tensor_value_info('arg_output', TensorProto.INT64, output_shape)
        arg_def = helper.make_node("ArgMax",
                                   inputs=['input'],
                                   outputs=['arg_output'],
                                   axis=arg_axis,
                                   select_last_index=1
                                   )
        reduce_output_1 = helper.make_tensor_value_info('reduce_output_1', TensorProto.FLOAT,
                                                        output_shape)
        reduce_def_1 = helper.make_node("ReduceMax",
                                        inputs=['input'],
                                        outputs=['reduce_output_1'],
                                        axes=reduce_axes)
        reduce_output_2 = helper.make_tensor_value_info('reduce_output_2', TensorProto.FLOAT,
                                                        output_shape)
        reduce_def_2 = helper.make_node("ReduceMax",
                                        inputs=['input'],
                                        outputs=['reduce_output_2'],
                                        axes=reduce_axes)

        graph_def = helper.make_graph([arg_def, reduce_def_1, reduce_def_2], case_name, [input],
                                      [arg_output, reduce_output_1, reduce_output_2])
        self.onnx_and_test(graph_def)

    # def test_LayerNorm(self, case_name):
    #     axis = 2
    #     io_shape = [3, 4, 5, 2]
    #     wb_shape = [x for x in io_shape]
    #     mr_shape = [x for x in io_shape]
    #     for i in range(axis):
    #         wb_shape[i] = 1
    #     for i in range(axis, 4):
    #         mr_shape[i] = 1
    #     w_data = np.random.randn(*wb_shape).astype(np.float32)
    #     b_data = np.random.randn(*wb_shape).astype(np.float32)
    #     input = helper.make_tensor_value_info('input', TensorProto.FLOAT, io_shape)
    #     weight = helper.make_tensor('weight', TensorProto.FLOAT, w_data.shape, w_data)
    #     bias = helper.make_tensor('bias', TensorProto.FLOAT, b_data.shape, b_data)
    #     output = helper.make_tensor_value_info('output', TensorProto.FLOAT, io_shape)
    #     mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, mr_shape)
    #     rstd = helper.make_tensor_value_info('rstd', TensorProto.FLOAT, mr_shape)

    #     node_def = helper.make_node(
    #         "LayerNormalization",
    #         inputs=['input', 'weight', 'bias'],
    #         outputs=['output', 'mean', 'rstd'],
    #     )
    #     graph_def = helper.make_graph([node_def], case_name,
    #                                   [input], [output, mean, rstd],
    #                                   initializer=[weight, bias])
    #     self.onnx_and_test(graph_def)

    def BinaryWeightBase(self, case_name, input_shape, weight_shape, binary_name, reverse=False):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        if binary_name == "Div":
            weight_data_ = np.random.uniform(0.5, 2, tuple(weight_shape)).astype(np.float32)
            weight_data = np.multiply(weight_data_, np.random.choice(np.array([-1, 1]),
                                                                     weight_shape))
        else:
            weight_data = np.random.randn(*weight_shape).astype(np.float32)
        if binary_name in ("Greater", "GreaterOrEqual", "Less", "LessOrEqual"):
            output_dtype = TensorProto.BOOL
        else:
            output_dtype = TensorProto.FLOAT
        weight = helper.make_tensor('weight1', TensorProto.FLOAT, weight_shape, weight_data)
        output = helper.make_tensor_value_info("output", output_dtype, input_shape)
        binary_def = helper.make_node(binary_name, inputs=['input', 'weight1'], outputs=["output"])
        graph_def = helper.make_graph([binary_def],
                                      case_name, [input], [output],
                                      initializer=[weight])
        self.onnx_and_test(graph_def)

    def test_Div2Mul(self, case_name):
        self.BinaryWeightBase(case_name, [1, 16, 32, 32], [1, 16, 32, 32], "Div")

    def test_Mul2Scale(self, case_name):
        self.BinaryWeightBase(case_name, [1, 32, 32, 32], [1, 32, 1, 1], "Mul")

    #conv_param: [out_shape, filter_shape, conv_pads]
    def PadConvBase(self, case_name, input_shape, pad_param, conv_param_list):
        add_in_shape = {"input1": input_shape, "input2": input_shape}
        pad_val = np.array(pad_param).astype(np.int64)
        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, x) for k, x in add_in_shape.items()
        ]
        paddings = helper.make_tensor(name='paddings',
                                      data_type=onnx.TensorProto.INT64,
                                      dims=pad_val.shape,
                                      vals=pad_val.flatten())
        add1 = helper.make_tensor_value_info("add1", TensorProto.FLOAT, input_shape)
        f_data1 = np.random.randn(*conv_param_list[0][1]).astype(np.float32)
        f_data2 = np.random.randn(*conv_param_list[1][1]).astype(np.float32)
        filter1 = helper.make_tensor('filter1', TensorProto.FLOAT, conv_param_list[0][1], f_data1)
        filter2 = helper.make_tensor('filter2', TensorProto.FLOAT, conv_param_list[1][1], f_data2)
        output1 = helper.make_tensor_value_info("output1", TensorProto.FLOAT, conv_param_list[0][0])
        output2 = helper.make_tensor_value_info("output2", TensorProto.FLOAT, conv_param_list[1][0])

        add_def = helper.make_node("Add", inputs=list(add_in_shape.keys()), outputs=["add1"])
        pad_def = helper.make_node(
            "Pad",
            inputs=['add1', 'paddings'],
            outputs=['pad2'],
            mode='constant',
        )
        conv1_def = helper.make_node("Conv",
                                     inputs=['pad2', 'filter1'],
                                     outputs=['output1'],
                                     kernel_shape=conv_param_list[0][1][2:],
                                     pads=conv_param_list[0][2])
        conv2_def = helper.make_node("Conv",
                                     inputs=['pad2', 'filter2'],
                                     outputs=['output2'],
                                     kernel_shape=conv_param_list[1][1][2:],
                                     pads=conv_param_list[1][2])
        graph_def = helper.make_graph([add_def, pad_def, conv1_def, conv2_def],
                                      case_name,
                                      inputs, [add1, output1, output2],
                                      initializer=[paddings, filter1, filter2])
        self.onnx_and_test(graph_def)

    def test_PadConv1d(self, case_name):
        self.PadConvBase(case_name, [1, 2, 8], [0, 0, 1, 0, 0, 1], \
                        [[[1, 2, 10], [2, 2, 3], [1, 1]], [[1, 2, 8], [2, 2, 3], [0, 0]]])

    def test_PadConv2d(self, case_name):
        self.PadConvBase(case_name, [1, 3, 8, 8], [0, 0, 1, 1, 0, 0, 3, 1], \
                        [[[1, 16, 14, 10], [16, 3, 3, 3], [2, 1, 2, 1]], [[1, 16, 10, 8], [16, 3, 3, 3], [0, 0, 0 ,0]]])

    def test_PadConv3d(self, case_name):
        self.PadConvBase(case_name, [1, 3, 32, 32, 32], [0, 0, 0, 1, 2, 0, 0, 3, 1, 2], \
                        [[[1, 16, 36, 35, 35], [16, 3, 3, 3, 3], [1, 2, 1, 2, 1, 0]], [[1, 16, 33, 32, 34], [16, 3, 3, 3, 3], [0, 0, 0, 0, 0 ,0]]])

    #pool_param: [out_shape, kernel_shape, pool_pads]
    def PadPoolBase(self, case_name, input_shape, pad_param, pool_param_list):
        add_in_shape = {"input1": input_shape, "input2": input_shape}
        pad_val = np.array(pad_param).astype(np.int64)
        inputs = [
            helper.make_tensor_value_info(k, TensorProto.FLOAT, x) for k, x in add_in_shape.items()
        ]
        paddings = helper.make_tensor(name='paddings',
                                      data_type=onnx.TensorProto.INT64,
                                      dims=pad_val.shape,
                                      vals=pad_val.flatten())
        add1 = helper.make_tensor_value_info("add1", TensorProto.FLOAT, input_shape)
        output1 = helper.make_tensor_value_info("output1", TensorProto.FLOAT, pool_param_list[0][0])
        output2 = helper.make_tensor_value_info("output2", TensorProto.FLOAT, pool_param_list[1][0])

        add_def = helper.make_node("Add", inputs=list(add_in_shape.keys()), outputs=["add1"])
        pad_def = helper.make_node("Pad",
                                   inputs=['add1', 'paddings'],
                                   outputs=['pad2'],
                                   mode='constant')
        pool1_def = helper.make_node('AveragePool',
                                     inputs=['pad2'],
                                     outputs=['output1'],
                                     kernel_shape=pool_param_list[0][1],
                                     pads=pool_param_list[0][2],
                                     strides=pool_param_list[0][3],
                                     count_include_pad=1)
        pool2_def = helper.make_node('AveragePool',
                                     inputs=['pad2'],
                                     outputs=['output2'],
                                     kernel_shape=pool_param_list[1][1],
                                     pads=pool_param_list[1][2],
                                     strides=pool_param_list[1][3],
                                     count_include_pad=1)
        graph_def = helper.make_graph([add_def, pad_def, pool1_def, pool2_def],
                                      case_name,
                                      inputs, [add1, output1, output2],
                                      initializer=[paddings])
        self.onnx_and_test(graph_def)

    def test_PadPool1d(self, case_name):
        self.PadPoolBase(case_name, [1, 3, 256], [0, 0, 2, 0, 0, 2], \
                [[[1, 3, 65], [4], [0, 0], [4]], [[1, 3, 34], [8], [6, 6], [8]]])

    def test_PadPool2d(self, case_name):
        self.PadPoolBase(case_name, [1, 32, 12, 12], [0, 0, 1, 2, 0, 0, 1, 2], \
                [[[1, 32, 6, 7], [4, 4], [0, 0, 0, 0], [2, 2]], [[1, 32, 4, 4], [4, 4], [1, 0, 1, 0], [4, 4]]])

    def test_PadPool3d(self, case_name):
        self.PadPoolBase(case_name, [1, 16, 32, 32, 32], [0, 0, 0, 1, 2, 0, 0, 0, 1, 2], \
                [[[1, 16, 15, 16, 17], [4, 4, 4], [0, 0, 0, 0, 0, 0], [2, 2, 2]], [[1, 16, 9, 9, 9], [4, 4, 4], [2, 1, 0, 2, 1, 0], [4, 4, 4]]])

    def test_ScatterND(self, case_name):
        x_shape = [320, 320]
        idx_shape = [160, 160, 2]
        update_shape = [160, 160]
        # indices should not have duplicate entries: that is, if idx1 != idx2, then indices[idx1] != indices[idx2].
        # This ensures that the output value does not depend on the iteration order.
        input_data = {
            # "raw_data": np.random.rand(*input_shape['raw_data']).astype(np.float32),
            "x_data": np.random.rand(*x_shape).astype(np.float32),
            "indices": np.random.randint(0, 64, tuple(idx_shape)),
            "updates": np.random.rand(*update_shape).astype(np.float32)
        }
        raw_data = helper.make_tensor_value_info("x_data", TensorProto.FLOAT, x_shape)
        indices = helper.make_tensor_value_info("indices", TensorProto.INT64, idx_shape)
        updates = helper.make_tensor_value_info("updates", TensorProto.FLOAT, update_shape)
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, x_shape)
        scatternd_def = helper.make_node("ScatterND",
                                         inputs=list(input_data.keys()),
                                         outputs=["output"])
        graph_def = helper.make_graph([scatternd_def], case_name, [raw_data, indices, updates],
                                      [output])
        self.onnx_and_test(graph_def, input_data=input_data)

    def test_SliceToReverse(self, case_name):
        input_shape = [100, 1, 3]
        output_shape = [100, 1, 3]
        starts_data = np.array([-1], dtype=np.int64)
        ends_data = np.array([-4], dtype=np.int64)
        axes_data = np.array([2], dtype=np.int64)
        steps_data = np.array([-1], dtype=np.int64)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        starts = helper.make_tensor('starts', TensorProto.INT64, [1], starts_data)
        ends = helper.make_tensor('ends', TensorProto.INT64, [1], ends_data)
        axes = helper.make_tensor('axes', TensorProto.INT64, [1], axes_data)
        steps = helper.make_tensor('steps', TensorProto.INT64, [1], steps_data)
        slice_def = helper.make_node(
            'Slice',
            inputs=['input', 'starts', 'ends', 'axes', 'steps'],
            outputs=['output'],
        )
        graph_def = helper.make_graph([slice_def],
                                      case_name, [input], [output],
                                      initializer=[starts, ends, axes, steps])
        self.onnx_and_test(graph_def)

    def test_ReduceFuse(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                x = torch.sum(x, 1)
                x = torch.sum(x, 1)
                return x

        x = torch.randn(2, 2, 3, 4).float()
        self.torch_and_test(x, Model(), case_name)

    def test_PermuteToReshape(self, case_name):
        input_shapes = [[4, 3, 28, 1], [4, 1, 3, 20]]
        transpose_orders = [[0, 1, 3, 2], [0, 2, 1, 3]]
        for i, input_shape in enumerate(input_shapes):
            input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
            order = transpose_orders[i]
            output_shape = [input_shape[order[i]] for i in range(len(order))]
            output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
            transpose_def = helper.make_node("Transpose",
                                             inputs=['input'],
                                             outputs=['output'],
                                             perm=order)
            graph_def = helper.make_graph([transpose_def], "{}_{}".format(case_name, i), [input],
                                          [output])
            self.onnx_and_test(graph_def)

    def test_PermuteToReorg(self, case_name):
        input_shape = [1, 4, 6, 6]
        output_shape = [1, 16, 3, 3]
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        rshape1 = [6]
        rshape1_data = np.array([1, 4, 3, 2, 3, 2], dtype=np.int64)
        r1 = helper.make_tensor('shape1', TensorProto.INT64, rshape1, rshape1_data)
        reshape1_def = helper.make_node("Reshape", inputs=['input', 'shape1'], outputs=['out1'])
        order = [0, 1, 3, 5, 2, 4]
        permute_def = helper.make_node("Transpose", inputs=['out1'], outputs=['out2'], perm=order)
        rshape2 = [4]
        rshape2_data = np.array(output_shape, dtype=np.int64)
        r2 = helper.make_tensor('shape2', TensorProto.INT64, rshape2, rshape2_data)
        reshape2_def = helper.make_node("Reshape", inputs=['out2', 'shape2'], outputs=['output'])
        graph_def = helper.make_graph([reshape1_def, permute_def, reshape2_def],
                                      case_name, [input], [output],
                                      initializer=[r1, r2])
        self.onnx_and_test(graph_def)

    def test_PermuteToReorg2(self, case_name):
        input_shape = [1, 16, 200, 200]
        output_shape = [1, 64, 100, 100]
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        rshape1 = [6]
        rshape1_data = np.array([1, 16, 100, 2, 100, 2], dtype=np.int64)
        r1 = helper.make_tensor('shape1', TensorProto.INT64, rshape1, rshape1_data)
        reshape1_def = helper.make_node("Reshape", inputs=['input', 'shape1'], outputs=['out1'])
        order = [0, 1, 3, 5, 2, 4]
        permute_def = helper.make_node("Transpose", inputs=['out1'], outputs=['out2'], perm=order)
        rshape2 = [4]
        rshape2_data = np.array(output_shape, dtype=np.int64)
        r2 = helper.make_tensor('shape2', TensorProto.INT64, rshape2, rshape2_data)
        reshape2_def = helper.make_node("Reshape", inputs=['out2', 'shape2'], outputs=['out3'])
        #add conv define
        filter_shape = [64, 64, 3, 3]
        kernel = [3, 3]
        padding = [1, 1, 1, 1]
        stride = [1, 1]
        dilation = [1, 1]
        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        bias_data = np.random.randn(output_shape[1]).astype(np.float32)
        weight = helper.make_tensor("weight", TensorProto.FLOAT, filter_shape, weight_data)
        bias = helper.make_tensor("bias", TensorProto.FLOAT, list(bias_data.shape), bias_data)
        conv_def = helper.make_node("Conv",
                                    inputs=['out3', 'weight', 'bias'],
                                    outputs=['output'],
                                    kernel_shape=kernel,
                                    pads=padding,
                                    strides=stride,
                                    dilations=dilation,
                                    group=1)
        graph_def = helper.make_graph([reshape1_def, permute_def, reshape2_def, conv_def],
                                      case_name, [input], [output],
                                      initializer=[r1, r2, weight, bias])
        self.onnx_and_test(graph_def)

    def test_Permute5dSplit(self, case_name):
        input_shape = [2, 4, 16, 20, 32]
        orders = [[0, 4, 2, 1, 3], [3, 1, 0, 4, 2], [2, 1, 3, 4, 0], [1, 0, 2, 4, 3],
                  [4, 0, 3, 2, 1], [4, 3, 2, 1, 0]]
        for i in range(0, len(orders)):
            order = orders[i]
            if i >= 3 and self.chip.startswith("cv18"):
                #because cv183x backend don't support all permute4d
                continue
            print("order:", order)
            output_shape = [input_shape[order[i]] for i in range(0, len(order))]
            input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
            output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
            permute_def = helper.make_node("Transpose",
                                           inputs=['input'],
                                           outputs=['output'],
                                           perm=order)
            graph_def = helper.make_graph([permute_def], "{}_{}".format(case_name, i), [input],
                                          [output])
            self.onnx_and_test(graph_def)

    def test_PoolSignError(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.relu = nn.ReLU()
                self.conv = nn.Conv2d(8, 8, 3, 2, 1)
                self.pool = nn.AvgPool2d(2, 2)

            def forward(self, x):
                a = self.relu(x)
                b = self.pool(a)
                c = self.conv(x)
                d = b + c
                return d

        x = torch.randn(4, 8, 32, 32).float()
        self.torch_and_test(x, Model(), case_name)

    def test_PoolAfterRelu(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.relu = nn.ReLU()
                self.matmul = nn.Linear(32, 16)
                self.pool = nn.AvgPool2d(2, 2)

            def forward(self, x):
                a = self.relu(x)
                b = self.pool(a)
                c = self.matmul(x)
                return b, c

        x = torch.randn(4, 8, 32, 32).float()
        self.torch_and_test(x, Model(), case_name)

    def test_If(self, case_name):
        from onnx.numpy_helper import from_array
        from onnx.helper import (make_node, make_graph, make_model, make_tensor_value_info)
        # initializers
        value = np.array([0], dtype=np.float32)
        zero = from_array(value, name='zero')
        value2 = np.array([0, 1], dtype=np.int64)
        axes = from_array(value2, name='axes')
        input_shape = [5, 5]
        # Same as before, X is the input, Z is the output.
        X = make_tensor_value_info('X', onnx.TensorProto.FLOAT, input_shape)
        Z = make_tensor_value_info('Z', onnx.TensorProto.FLOAT, input_shape)
        # The node building the condition. The first one
        # sum over all axes.
        rsum = make_node('ReduceSum', ['X', 'axes'], ['rsum'], keepdims=0)
        # The second compares the result to 0.
        cond = make_node('Greater', ['rsum', 'zero'], ['cond'])

        # Builds the graph is the condition is True.
        # Input for then
        then_out_data = np.random.rand(*input_shape).astype(np.float32)
        then_out2_data = np.random.rand(*input_shape).astype(np.float32)
        then_const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['then_out'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=then_out_data.shape,
                vals=then_out_data.flatten(),
            ),
        )

        then2_const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['then2_out'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=then_out2_data.shape,
                vals=then_out2_data.flatten(),
            ),
        )

        then3_out = make_tensor_value_info('then3_out', onnx.TensorProto.FLOAT, input_shape)

        then_node = helper.make_node(
            "Add",  # node name
            ["then_out", "then2_out"],  # inputs
            ["then3_out"]  # outputs
        )

        # And the graph wrapping these elements.
        then_body = make_graph([then_const_node, then2_const_node, then_node], 'then_body', [],
                               [then3_out])

        # Same process for the else branch.
        else_out_data = np.random.rand(*input_shape).astype(np.float32)
        else_out2_data = np.random.rand(*input_shape).astype(np.float32)
        else_const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['else_out'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=else_out_data.shape,
                vals=else_out_data.flatten(),
            ),
        )

        else2_const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['else2_out'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=else_out2_data.shape,
                vals=else_out2_data.flatten(),
            ),
        )
        else3_out = make_tensor_value_info('else3_out', onnx.TensorProto.FLOAT, input_shape)

        else_node = helper.make_node(
            'Sub',  # node name
            ["else_out", "else2_out"],  # inputs
            ['else3_out']  # outputs
        )

        else_body = make_graph([else_const_node, else2_const_node, else_node], 'else_body', [],
                               [else3_out])

        # Finally the node If taking both graphs as attributes.
        if_node = onnx.helper.make_node("If", ["cond"], ["Z"],
                                        then_branch=then_body,
                                        else_branch=else_body)

        # The final graph.
        graph_def = make_graph([rsum, cond, if_node], "if", [X], [Z], [zero, axes])
        self.onnx_and_test(graph_def)

    def test_If_v2(self, case_name):
        #test if op case2: cover more usage scene of onnx if op
        from onnx.numpy_helper import from_array
        from onnx.helper import (make_node, make_graph, make_model, make_tensor_value_info)
        # initializers
        value = np.array([0], dtype=np.float32)
        zero = from_array(value, name='zero')
        value2 = np.array([0, 1], dtype=np.int64)
        axes = from_array(value2, name='axes')
        shape = [5, 5]
        # Same as before, X is the input, Y is the output.
        X = make_tensor_value_info('X', onnx.TensorProto.FLOAT, shape)
        Z = make_tensor_value_info('Z', onnx.TensorProto.FLOAT, shape)
        Y = make_tensor_value_info('Y', onnx.TensorProto.FLOAT, shape)
        K = make_tensor_value_info('K', onnx.TensorProto.FLOAT, shape)
        # The node building the condition. The first one
        # sum over all axes.
        rsum = make_node('ReduceSum', ['X', 'axes'], ['rsum'], keepdims=0)
        # The second compares the result to 0.
        cond = make_node('Greater', ['rsum', 'zero'], ['cond'])

        # Builds the graph is the condition is True.
        # Input for then
        then_out_data = np.random.rand(*shape).astype(np.float32)
        then_out2_data = np.random.rand(*shape).astype(np.float32)
        then_const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['then_out'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=then_out_data.shape,
                vals=then_out_data.flatten(),
            ),
        )

        then2_const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['then2_out'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=then_out2_data.shape,
                vals=then_out2_data.flatten(),
            ),
        )
        then4_out = make_tensor_value_info('then4_out', onnx.TensorProto.FLOAT, shape)

        then_node = helper.make_node(
            "Add",  # node name
            ["Y", "then2_out"],  # inputs
            ["then3_out"]  # outputs
        )

        mul_node = helper.make_node(
            "Mul",  # node name
            ["then3_out", "then_out"],  # inputs
            ["then4_out"]  # outputs
        )

        # And the graph wrapping these elements.
        then_body = make_graph([then2_const_node, then_node, then_const_node, mul_node],
                               'then_body', [], [then4_out])

        # Same process for the else branch.
        else_out_data = np.random.rand(*shape).astype(np.float32)
        else_out2_data = np.random.rand(*shape).astype(np.float32)
        else_const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['else_out'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=else_out_data.shape,
                vals=else_out_data.flatten(),
            ),
        )

        else2_const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['else2_out'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=else_out2_data.shape,
                vals=else_out2_data.flatten(),
            ),
        )
        else3_out = make_tensor_value_info('else3_out', onnx.TensorProto.FLOAT, shape)

        else_node = helper.make_node(
            'Sub',  # node name
            ["K", "else2_out"],  # inputs
            ['else3_out']  # outputs
        )

        else_body = make_graph([else2_const_node, else_node], 'else_body', [], [else3_out])

        # Finally the node If taking both graphs as attributes.
        if_node = onnx.helper.make_node("If", ["cond"], ["Z"],
                                        then_branch=then_body,
                                        else_branch=else_body)

        # The final graph.
        graph_def = make_graph([rsum, cond, if_node], "if", [X, Y, K], [Z], [zero, axes])
        self.onnx_and_test(graph_def)

    def test_Loop(self, case_name):
        from onnx import numpy_helper
        from onnx.helper import (
            make_node, make_graph, make_model, make_tensor_value_info)
        graph_def=helper.make_graph(
            name="test-loop",
            inputs=[
                helper.make_tensor_value_info(
                    "input_0", TensorProto.FLOAT, shape=[1]
                ),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output_0", TensorProto.INT32, shape=[1]
                )
            ],
            initializer=[
                numpy_helper.from_array(
                    np.array([10], dtype=np.int64),
                    name="while_maximum_iterations_0",
                ),
                numpy_helper.from_array(
                    np.array([-1], dtype=np.int64), name="const_fold_opt__18"
                ),
                numpy_helper.from_array(
                    np.array([10.0], dtype=np.float32), name="const_fold_opt__17"
                ),
                numpy_helper.from_array(
                    np.array([3], dtype=np.int32), name="Const_0"
                ),
                numpy_helper.from_array(
                    np.array([0], dtype=np.int64), name="axes"
                ),
            ],
            nodes=[
                helper.make_node(
                    "Cast",
                    inputs=["input_0"],
                    outputs=["while_cond_158_while_Less__13_0"],
                    name="while_cond_158_while_Less__13",
                    domain="",
                    to=TensorProto.INT32,
                ),
                helper.make_node(
                    "Less",
                    inputs=[
                        "input_0",
                        "const_fold_opt__17",
                    ],
                    outputs=["while_cond_158_while_Less_0"],
                    name="while_cond_158_while_Less",
                    domain="",
                ),
                helper.make_node(
                    "Squeeze",
                    inputs=["while_cond_158_while_Less_0", "axes"],
                    outputs=["while_cond_158_while_Squeeze_0"],
                    name="while_cond_158_while_Squeeze",
                    domain="",
                ),
                helper.make_node(
                    "Loop",
                    inputs=[
                        "while_maximum_iterations_0",
                        "while_cond_158_while_Squeeze_0",
                        "while_cond_158_while_Less__13_0",
                        "Const_0",
                    ],
                    outputs=["while_loop_0", "while_loop_1"],
                    name="while_loop",
                    body=helper.make_graph(
                        name="while_body",
                        inputs=[
                            helper.make_tensor_value_info(
                                "while_while_loop_counter_0",
                                TensorProto.INT64,
                                shape=[],
                            ),
                            helper.make_tensor_value_info(
                                "cond__15_0", TensorProto.BOOL, shape=[]
                            ),
                            helper.make_tensor_value_info(
                                "while_placeholder_0", TensorProto.INT32, shape=[1]
                            ),
                            helper.make_tensor_value_info(
                                "while_add_const_0_0", TensorProto.INT32, shape=[1]
                            ),
                        ],
                        outputs=[
                            helper.make_tensor_value_info(
                                "cond___while_Identity_graph_outputs_Identity__3_0",
                                TensorProto.BOOL,
                                shape=[],
                            ),
                            helper.make_tensor_value_info(
                                "while_Identity_2_0", TensorProto.INT32, shape=[1]
                            ),
                            helper.make_tensor_value_info(
                                "while_add_const_0_0", TensorProto.INT32, shape=[1]
                            ),
                        ],
                        initializer=[
                            numpy_helper.from_array(
                                np.array(8.0, dtype=np.float32),
                                name="const_fold_opt__19",
                            ),
                            numpy_helper.from_array(
                                np.array([0], dtype=np.int64), name="reshape2"
                            ),
                        ],
                        nodes=[
                            helper.make_node(
                                "Add",
                                inputs=[
                                    "while_placeholder_0",
                                    "while_add_const_0_0",
                                ],
                                outputs=["while_Identity_2_0"],
                                name="while_Add",
                            ),
                            helper.make_node(
                                "Cast",
                                inputs=["while_Identity_2_0"],
                                outputs=["cond___while_Less__13_0"],
                                name="cond___while_Less__13",
                                domain="",
                                to=TensorProto.FLOAT,
                            ),
                            helper.make_node(
                                "Less",
                                inputs=[
                                    "cond___while_Less__13_0",
                                    "const_fold_opt__19",
                                ],
                                outputs=["cond___while_Less_0"],
                                name="cond___while_Less",
                                domain="",
                            ),
                            helper.make_node(
                                "Squeeze",
                                inputs=["cond___while_Less_0", "reshape2"],
                                outputs=[
                                    "cond___while_Identity_graph_outputs_Identity__3_0"
                                ],
                                name="cond___while_Squeeze",
                                domain="",
                            ),
                        ],
                    ),
                ),
                helper.make_node(
                    "Unsqueeze",
                    inputs=["while_loop_0","axes"],
                    outputs=["Reshape_tensor_0"],
                    name="Reshape_tensor",
                ),
                helper.make_node(
                    "Reshape",
                    inputs=["Reshape_tensor_0", "const_fold_opt__18"],
                    outputs=["output_0"],
                    name="Reshape",
                ),
            ],
        )
        self.onnx_and_test(graph_def)


def test_one_case_in_all(tester: ONNX_IR_TESTER, case, error_cases, success_cases):
    try:
        tester.test_single(case)
    except:
        error_cases.append(case)
        return
    success_cases.append(case)


def test_all(tester: ONNX_IR_TESTER):
    if tester.multithread:
        import multiprocessing
        process_number = multiprocessing.cpu_count() // 2 + 1
        processes = []
        error_cases = multiprocessing.Manager().list()
        success_cases = multiprocessing.Manager().list()
        for case in tester.test_cases:
            if tester.check_support(case):
                p = multiprocessing.Process(target=test_one_case_in_all,
                                            args=(tester, case, error_cases, success_cases))
                processes.append(p)
            if len(processes) == process_number:
                for p in processes:
                    p.start()
                for j in processes:
                    j.join()
                processes = []
        if processes:
            for p in processes:
                p.start()
            for j in processes:
                j.join()
    else:
        error_cases = []
        success_cases = []
        for case in tester.test_cases:
            if tester.check_support(case):
                test_one_case_in_all(tester, case, error_cases, success_cases)
    print("Success: {}".format(success_cases))
    print("Failure: {}".format(error_cases))
    if error_cases:
        print("====== test_onnx.py --chip {} TEST Failed ======".format(tester.chip))
        # exit(1)
    else:
        print("====== test_onnx.py --chip {} TEST Success ======".format(tester.chip))
    return error_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--chip", default="bm1684x", type=str,
                        choices=['bm1686', 'bm1684x', 'bm1684', 'cv183x', 'cv182x', 'cv181x', 'cv180x'],
                        help="chip platform name")
    parser.add_argument("--case", default="all", type=str, help="test one case, if all, then test all cases")
    parser.add_argument("--mode", default="all", type=str, choices=['all', 'f32', 'f16', 'bf16', 'int8', 'int4'],
                        help="chip platform name")
    parser.add_argument("--dynamic", action="store_true", help='do dynamic compile')
    parser.add_argument("--debug", action="store_true", help='keep middle file if debug')
    parser.add_argument("--simple", action="store_true", help='do simple test for commit test')
    parser.add_argument("--disable_thread", action="store_true", help='do test without multi thread')
    parser.add_argument("--show_all", action="store_true", help='show all cases')
    # yapf: enable
    args = parser.parse_args()
    tester = ONNX_IR_TESTER(args.chip, args.mode, args.dynamic, args.simple, args.disable_thread)
    if args.show_all:
        print("====== Show All Cases ============")
        for case in tester.test_cases:
            print(case)
        exit(0)
    dir = "onnx_test_{}".format(args.chip)
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)
    if args.case == "" or args.case.lower() == "all":
        test_all(tester)
    else:
        tester.test_single(args.case)
    if args.debug == False:
        file_clean()
