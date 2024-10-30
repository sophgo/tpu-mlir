#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import numpy as np
from typing import List, Union

from tools.model_runner import mlir_inference, model_inference, torch_inference, show_fake_cmd
from tools.npz_tool import npz_compare
from tools.model_transform import *
from utils.mlir_shell import *
from utils.auto_remove import clean_kmp_files
from utils.timer import Timer
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
import traceback

from torchvision.ops import DeformConv2d
from torchvision.ops import roi_align


class TORCH_IR_TESTER(object):
    ID = 0
    CURRENT_CASE = ""

    # This class is built for testing single operator transform.
    def __init__(self,
                 chip: str = "bm1684x",
                 mode: str = "all",
                 simple: bool = False,
                 disable_thread: bool = False,
                 quant_input: bool = False,
                 quant_output: bool = False,
                 debug: bool = False,
                 num_core : int = 1,
                 debug_cmd: str = '',
                 cuda: bool = False,
                 dynamic: bool = False):
        Y, N = True, False
        self.quant_input = quant_input
        self.quant_output = quant_output
        self.debug = debug
        self.debug_cmd = debug_cmd
        self.group_opt = 2
        self.test_cuda = cuda
        self.dynamic = dynamic
        # yapf: disable
        self.test_cases = {
            ##################################
            # Torch Test Case, Alphabetically
            ##################################
            # case: (test, bm1684_support, bm1684x_support, bm1688_support, cv183x_support, mars3_support)
            "Abs":              (self.test_Abs,               N, Y, Y, Y, Y),
            "Activation":       (self.test_Activation,        Y, Y, Y, Y, N),
            "AdaptiveAvgPool1d":(self.test_AdaptiveAvgPool1d, N, Y, N, N, Y),
            "AdaptiveAvgPool2d":(self.test_AdaptiveAvgPool2d, N, Y, Y, Y, N),
            "Add":              (self.test_Add,               N, Y, Y, Y, Y),
            "AddLarge":         (self.test_AddLarge,          N, Y, Y, N, Y),
            "Add5d":            (self.test_Add5d,             N, Y, Y, Y, Y),
            "Add6d":            (self.test_Add6d,             N, Y, Y, Y, Y),
            "Addmm":            (self.test_Addmm,             N, Y, Y, Y, Y),
            "Arange":           (self.test_Arange,            N, Y, Y, Y, Y),
            "Arctan":           (self.test_Arctan,            N, Y, Y, N, N),
            "Arctanh":          (self.test_Arctanh,           Y, Y, Y, N, N),
            "Arccos":           (self.test_Arccos,            Y, Y, Y, N, N),
            "Arg":              (self.test_Arg,               Y, Y, Y, N, Y),
            "Attention":        (self.test_Attention,         N, Y, Y, Y, Y),
            "AttentionNew":     (self.test_AttentionNew,      N, Y, N, N, N),
            "AvgPool1d":        (self.test_AvgPool1d,         Y, Y, Y, Y, Y),
            "AvgPool2d":        (self.test_AvgPool2d,         Y, Y, Y, Y, Y),
            "AvgPool3d":        (self.test_AvgPool3d,         N, Y, Y, Y, N),
            "BatchNorm":        (self.test_BatchNorm,         N, Y, Y, Y, Y),
            "BMM":              (self.test_BatchMatMul,       N, Y, Y, Y, Y),
            "Ceil":             (self.test_Ceil,              N, Y, N, N, Y),
            "ChannelShuffle":   (self.test_ChannelShuffle,    N, Y, Y, Y, Y),
            "Chunk":            (self.test_Chunk,             N, Y, Y, Y, Y),
            "Clamp":            (self.test_Clamp,             Y, Y, Y, N, Y),
            "Compare":          (self.test_Compare,           Y, Y, Y, N, Y),
            "Concat":           (self.test_Concat,            N, Y, Y, Y, Y),
            "ConstantLike":     (self.test_ConstantLike,      N, Y, Y, Y, Y),
            "Conv1d":           (self.test_Conv1d,            N, Y, Y, Y, Y),
            "Conv2d":           (self.test_Conv2d,            N, Y, Y, Y, Y),
            "Conv3d":           (self.test_Conv3d,            Y, Y, Y, N, Y),
            "ConvMerge":        (self.test_ConvMerge,         N, Y, Y, Y, Y),
            "ConvGroup":        (self.test_ConvGroup,         N, Y, Y, Y, Y),
            "ConvTrans":        (self.test_ConvTrans,         N, Y, Y, Y, N),
            "ConstantFill":     (self.test_ConstantFill,      Y, Y, Y, Y, Y),
            "DeformConv2D":     (self.test_DeformConv2D,      Y, Y, N, N, N),
            "Div":              (self.test_Div,               N, Y, Y, Y, Y),
            "Dot":              (self.test_Dot,               N, Y, N, N, Y),
            "Dropout":          (self.test_Dropout,           N, Y, Y, Y, Y),
            "Elu":              (self.test_Elu,               Y, Y, Y, Y, N),
            "Embedding":        (self.test_Embedding,         N, Y, Y, Y, N),
            "FAttention":       (self.test_FAttention,        N, Y, N, N, Y),
            "Flatten":          (self.test_Flatten,           N, Y, Y, N, Y),
            "Flip":             (self.test_Flip,              N, Y, Y, N, N),
            "Floor":            (self.test_Floor,             N, Y, Y, N, Y),
            "FloorDiv":         (self.test_FloorDiv,          N, Y, Y, N, Y),
            "FrobeniusNorm":    (self.test_FrobeniusNorm,     N, Y, Y, N, N),
            "Gather":           (self.test_Gather,            N, N, N, N, N),
            "GridSampler":      (self.test_GridSampler,       N, Y, N, Y, N),
            "GridSampler3D":    (self.test_GridSampler3D,     N, N, N, N, N), # bm1684x has random error casued by 2.18 commit
            "GridSampler3DPermute": (self.test_GridSampler3DPermute,     N, N, N, N, N), # bm1684x has random error casued by 2.18 commit
            "GroupNorm":        (self.test_GroupNorm,         Y, Y, Y, N, N),
            "GRU":              (self.test_GRU,               Y, Y, Y, Y, N),
            "IndexPut":         (self.test_IndexPut,          N, Y, Y, N, Y),
            "Index":            (self.test_IndexPut,          N, Y, Y, N, Y),
            "IndexSelect":      (self.test_IndexSelect,       N, Y, Y, Y, Y),
            "InstanceNorm":     (self.test_InstanceNorm,      Y, Y, Y, Y, N),
            "Interp":           (self.test_Interp,            N, Y, Y, Y, Y),
            "Interp2":           (self.test_Interp2,          N, Y, Y, N, Y),
            "LayerNorm":        (self.test_LayerNorm,         N, Y, Y, Y, N),
            "LeakyRelu":        (self.test_LeakyRelu,         N, Y, Y, Y, Y),
            "Linear":           (self.test_Linear,            N, Y, Y, Y, Y),
            "LogSoftmax":       (self.test_LogSoftmax,        N, Y, Y, Y, N),
            "LSTM":             (self.test_LSTM,              N, Y, Y, Y, N),
            "MaskedFill":       (self.test_MaskedFill,        Y, Y, Y, N, Y),
            "Math":             (self.test_Math,              N, Y, Y, N, N),
            "MatMul":           (self.test_MatMul,            N, Y, Y, Y, Y),
            "MatMulSlice":      (self.test_MatMulSlice,       N, Y, Y, Y, Y),
            "MatMulSplit":      (self.test_MatMulSplit,       N, N, Y, N, N),
            "Max":              (self.test_Max,               N, Y, Y, N, Y),
            "MaxPool1d":        (self.test_MaxPool1d,         N, Y, Y, Y, Y),
            "MaxPool2d":        (self.test_MaxPool2d,         N, Y, Y, Y, Y),
            "MaxPool3d":        (self.test_MaxPool3d,         N, Y, Y, Y, Y),
            "MeshGrid":         (self.test_MeshGrid,          N, N, N, N, Y),
            "Min":              (self.test_Min,               N, Y, Y, N, Y),
            "MM":               (self.test_MM,                N, Y, Y, Y, Y),
            "MV":               (self.test_MV,                N, Y, N, N, Y),
            "Mul":              (self.test_Mul,               N, Y, Y, Y, Y),
            "MulConst":         (self.test_MulConst,          N, Y, Y, Y, Y),
            "NewZeros":         (self.test_NewZeros,          N, Y, Y, Y, Y),
            "NonZero":          (self.test_NonZero,           N, Y, Y, N, N),
            # "Normalize":        (self.test_Normalize,         N, Y, Y, N, N),
            "New_full":         (self.test_New_full,          N, Y, Y, Y, Y),
            "Reduce":           (self.test_Reduce,            N, Y, Y, Y, Y),
            "Remainder":        (self.test_Remainder,         Y, Y, N, N, Y),
            "Repeat":           (self.test_Repeat,            N, Y, Y, Y, Y),
            "Reshape":          (self.test_Reshape,           N, Y, Y, Y, Y),
            "Depth2Space":      (self.test_D2SPattern,        N, Y, Y, N, Y),
            "RMSNorm":          (self.test_RMSNorm,           N, Y, Y, N, N),
            "RoiAlign":         (self.test_RoiAlign,          N, Y, Y, N, N),
            "Roll":             (self.test_Roll,              N, Y, Y, N, Y),
            "PixelShuffle":     (self.test_PixelShuffle,      N, Y, Y, Y, Y),
            "PixelUnshuffle":   (self.test_PixelUnshuffle,    N, Y, Y, Y, Y),
            "PRelu":            (self.test_PRelu,             N, Y, Y, Y, Y),
            "PRelu":            (self.test_PRelu,             N, Y, Y, Y, Y),
            "Permute":          (self.test_Permute,           N, Y, Y, Y, Y),
            "Permute2":         (self.test_Permute2,          N, Y, Y, N, Y),
            "Pad1d":            (self.test_Pad1d,             Y, Y, Y, Y, Y),
            "Pad2d":            (self.test_Pad2d,             Y, Y, Y, Y, Y),
            "Pow":              (self.test_Pow,               N, Y, Y, Y, N),
            "Scatter":          (self.test_Scatter,           N, N, N, N, N),
            "ScaleDotAtten":    (self.test_ScaleDotAtten,     N, Y, N, N, Y),
            "Select":           (self.test_Select,            N, Y, Y, Y, Y),
            "Sign":             (self.test_Sign,              N, Y, Y, N, Y),
            "Slice":            (self.test_Slice,             N, Y, Y, Y, Y),
            "Softmax":          (self.test_Softmax,           N, Y, Y, Y, Y),
            "Softmin":          (self.test_Softmin,           N, Y, Y, Y, Y),
            "Split":            (self.test_Split,             N, Y, Y, Y, Y),
            "Squeeze":          (self.test_Squeeze,           N, Y, Y, Y, Y),
            "Stack":            (self.test_Stack,             N, Y, Y, Y, Y),
            "Sub":              (self.test_Sub,               Y, Y, Y, Y, Y),
            "T":                (self.test_T,                 N, Y, Y, Y, Y),
            "Tile":             (self.test_Tile,              Y, Y, Y, Y, Y),
            "To":               (self.test_To,                N, Y, Y, Y, Y),
            "Type_as":          (self.test_Type_as,           N, Y, Y, Y, Y),
            "Transpose":        (self.test_Transpose,         N, Y, Y, Y, Y),
            "Unbind":           (self.test_Unbind,            N, Y, Y, Y, Y),
            "Upsample":         (self.test_Upsample,          N, Y, Y, Y, N),
            "Unary":            (self.test_Unary,             N, Y, Y, Y, Y),
            "Unsqueeze":        (self.test_Unsqueeze,         N, Y, Y, Y, Y),
            "View":             (self.test_View,              N, Y, Y, Y, Y),
            "Where":            (self.test_Where,             N, Y, Y, N, Y),
            ## Special Case
            "Connect":          (self.test_Connect,           N, N, N, N, N),
            "GatherMulConst":   (self.test_GatherMulConst,    N, Y, Y, N, N),
            "InfError":         (self.test_InfError,          N, Y, Y, N, N),
            "SplitReshape":     (self.test_SplitReshape,      N, Y, Y, Y, Y),
            "WeightMultiUse":   (self.test_WeightMultiUse,    Y, Y, Y, Y, Y),
            ## only for test, only run test_single
            "user_define_net":   (self.user_define_net,    N, N, N, N, N),
            ## Canonicalization
            "MovePermuteAfterAdd": (self.test_MovePermuteAfterAdd, N, Y, Y, N, Y)
        }
        # yapf: enable
        self.support_quant_modes = ["f32", "f16", "bf16", "int8"]
        self.support_asym = [False]

        self.model_file = ".bmodel"
        self.is_cv18xx = False
        self.chip = chip.lower()
        self.simple = simple
        self.num_core = num_core
        self.multithread = not disable_thread
        if self.simple:
            self.support_quant_modes = ["f16", "int8"]
            self.support_asym = [False]
        # self.dynamic = dynamic
        if self.chip.startswith("cv18") and self.chip != "cv186x":
            self.support_quant_modes = ["bf16", "int8"]
            self.support_asym = [False]
            self.model_file = ".cvimodel"
            self.is_cv18xx = True
        elif self.chip == "bm1684":
            self.support_quant_modes = ["f32", "int8"]
            self.support_asym = [False]
        elif self.chip == "mars3":
            self.support_quant_modes = ["bf16", "int8"]
            self.support_asym = [False]
        self.mode = mode.lower()
        if self.mode == "" or self.mode == "all":
            self.quant_modes = self.support_quant_modes
        else:
            if self.mode not in self.support_quant_modes:
                raise RuntimeError("{} not support mode: {}".format(self.chip, self.mode))
            self.quant_modes = [self.mode]

    class Desc():

        def __init__(self, dtype, min=-10, max=10) -> None:
            self.dtype = dtype
            self.min = min
            self.max = max

    def test_single(self, case: str):
        np.random.seed(0)
        torch.manual_seed(7)
        TORCH_IR_TESTER.ID = 0
        TORCH_IR_TESTER.CURRENT_CASE = case
        print("Test: {}".format(case))
        if case in self.test_cases:
            os.makedirs(case, exist_ok=True)
            os.chdir(case)
            func, _, _, _, _, _ = self.test_cases[case]
            func()
            print("====== TEST {} Success ======".format(case))
        else:
            raise RuntimeError("case [{}] is not exist".format(case))

    def check_support(self, case):
        _, bm1684_support, bm1684x_support, bm1688_support, cv183x_support, mars3_support = self.test_cases[case]
        if self.is_cv18xx and cv183x_support:
            return True
        if self.chip == "bm1684" and bm1684_support:
            return True
        if self.chip == "bm1684x" and bm1684x_support:
            return True
        if self.chip in ["bm1688","cv186x"] and bm1688_support:
            return True
        if self.chip == "mars3" and mars3_support:
            return True
        return False

    def square_rooted(self, x):
        return np.sqrt(np.sum(np.power(x, 2)))

    def cosine_similarity(self, x, y):
        numerator = np.sum(x * y)
        denominator = self.square_rooted(x) * self.square_rooted(y)
        return round(numerator / float(denominator), 3)

    def compare(self, ref_out, target_out, use_cos: bool = False):
        if ref_out.dtype in [np.int64, np.int32, np.int16, np.int8] or use_cos:
            cos = self.cosine_similarity(ref_out, target_out)
            assert (cos > 0.997 or (np.linalg.norm(ref_out) == 0
                                    and np.linalg.norm(target_out) == 0))
        elif len(target_out) == 1 and len(ref_out) == 1 and ref_out[0] != 0:
            ratio = target_out[0] / ref_out[0]
            assert(ratio > 0.9 and ratio < 1.1)
        else:
            np.testing.assert_allclose(ref_out, target_out, rtol=1e-5, atol=1e-01)

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

    def generate_random(self, shape, dtype='float32', min=-10, max=10):
        scale = max - min
        return (np.random.rand(*shape) * scale + min).astype(dtype)

    def create_random_input(self, shapes, descs: List[Desc]):
        if len(descs) == 0:
            inputs = [self.generate_random(s) for s in shapes]
        else:
            inputs = list()
            for i in range(len(shapes)):
                inputs.append(
                    self.generate_random(shapes[i], descs[i].dtype, descs[i].min, descs[i].max))
        return [torch.from_numpy(inp) for inp in inputs]

    def torch_convert(self, in_shapes, torch_model, model_name: str, descs: List[Desc]):
        # torch --> mlir conversion (origin and optimized mlir models will be generated and saved)
        fp32_mlir = "{}.mlir".format(model_name)
        # input_dtype = [] if len(descs) == 0 else [d.dtype for d in descs]
        input_descs = {}
        for i in range(len(descs)):
            input_descs[i] = descs[i]
        tool = TorchTransformer(model_name, torch_model, input_shapes=in_shapes)
        tool.model_transform(fp32_mlir)

        input_npz = "{}_ref_in_fp32.npz".format(model_name)
        ref_npz = model_name + '_ref_outputs.npz'
        self.top_npz = model_name + "_top_outputs.npz"
        input_data = {}
        for idx, name in enumerate(tool.converter.input_names):
            if len(descs) == 0:
                input_data[name] = self.generate_random(in_shapes[idx])
            else:
                input_data[name] = self.generate_random(in_shapes[idx], descs[idx].dtype,
                                                        descs[idx].min, descs[idx].max)
        np.savez(input_npz, **input_data)
        file_mark(input_npz)
        # # top mlir outputs will be inferenced first in case the quant mode is int8
        show_fake_cmd(input_npz, torch_model, ref_npz)
        torch_outs = torch_inference(input_data, torch_model, True)
        np.savez(ref_npz, **torch_outs)
        file_mark(ref_npz)
        show_fake_cmd(input_npz, fp32_mlir, self.top_npz)
        top_mlir_outs = mlir_inference(input_data, fp32_mlir, True)
        np.savez(self.top_npz, **top_mlir_outs)
        self.table_name = "{}_cali_table".format(model_name)
        self.make_test_calibration_table(top_mlir_outs, self.table_name)
        return (torch_outs, top_mlir_outs, input_npz)

    def model_generate(self, model_name: str, quant_mode: str, isAsym: bool = False):
        top_mlir = "{}.mlir".format(model_name)
        tpu_mlir = "{}_{}".format(model_name, quant_mode)
        table = None
        if quant_mode == "int8":
            tpu_mlir += "_asym" if isAsym else "_sym"
            table = self.table_name

        # lowering
        mlir_lowering(top_mlir,
                      tpu_mlir + ".mlir",
                      mode=quant_mode,
                      num_core=self.num_core,
                      chip=self.chip,
                      cali_table=table,
                      asymmetric=isAsym)

        # transform
        tpu_final = tpu_mlir + "_final.mlir"
        bmodel = tpu_mlir + self.model_file
        mlir_to_model(tpu_mlir + ".mlir", bmodel, tpu_final, opt= self.group_opt, quant_input=self.quant_input, quant_output=self.quant_output, embed_debug_info=self.debug, debug_cmd = f'--debug_cmd={self.debug_cmd}', dynamic=self.dynamic)

        return (tpu_mlir + ".mlir", bmodel)

    def inference_and_compare(self,
                              tpu_mlir: str,
                              bmodel: str,
                              input_npz: str,
                              quant_mode: str,
                              model_name: str,
                              isAsym: bool = False):
        ref_tpu_tolerance = "0.9,0.9"
        if quant_mode == "int8":
            ref_tpu_tolerance = "0.95,0.70" if not isAsym else "0.90,0.54"
        elif quant_mode == "int4":
            ref_tpu_tolerance = "0.90,0.60"
        elif quant_mode == "bf16":
            ref_tpu_tolerance = "0.95,0.80"
        input_data = np.load(input_npz)
        # tpu mlir inference and compare
        tpu_npz = tpu_mlir.replace(".mlir", "_tpu_out.npz")
        show_fake_cmd(input_npz, tpu_mlir, tpu_npz)
        tpu_mlir_outs = mlir_inference(input_data, tpu_mlir, dump_all=True)
        np.savez(tpu_npz, **tpu_mlir_outs)
        file_mark(self.top_npz)
        file_mark(tpu_npz)
        npz_compare([self.top_npz, tpu_npz, "--tolerance", ref_tpu_tolerance, "-v"])
        # bmodel inference and compare
        model_npz = bmodel.replace("." + bmodel.split(".")[-1], "_model_out.npz")
        show_fake_cmd(input_npz, bmodel, model_npz)
        model_outs = model_inference(input_data, bmodel)
        np.savez(model_npz, **model_outs)
        file_mark(model_npz)
        npz_compare([tpu_npz, model_npz, "--tolerance", "0.95,0.80", "-v"])
        # cuda inference and compare
        if self.test_cuda:
            cuda_npz = tpu_mlir.replace(".mlir", "_cuda_out.npz")
            show_fake_cmd(input_npz, tpu_mlir, cuda_npz, True, True)
            cuda_outs = mlir_inference(input_data, tpu_mlir, dump_all=True, use_cuda=True)
            np.savez(cuda_npz, **cuda_outs)
            file_mark(cuda_npz)
            npz_compare([cuda_npz, tpu_npz, "--tolerance", "0.9999,0.9999", "-v"])

        msg = quant_mode.upper()
        if quant_mode == "int8":
            msg += ", Asymmetric: {}".format(isAsym)
        print("[Success] test {} {}".format(model_name, msg))

    def trace_and_test(self, in_shapes, torch_model: nn.Module, descs: List[Desc] = [], use_cos: bool = False):
        """Generic function to generate and compare torch and Tpu-Mlir output"""
        model_name = "{}_{}".format(self.CURRENT_CASE, TORCH_IR_TESTER.ID)
        TORCH_IR_TESTER.ID += 1
        model_def = model_name + ".pt"
        inputs = self.create_random_input(in_shapes, descs)
        jit.trace(torch_model.eval(), inputs).save(model_def)
        torch_outs, top_mlir_outs, input_npz = \
            self.torch_convert(in_shapes, model_def, model_name, descs)
        # test onnx and mlir outputs
        counter = 0
        for name in torch_outs:
            if name in top_mlir_outs:
                print("Compare mlir and torch:{}\n".format(name))
                top_mlir_output = top_mlir_outs[name].flatten()
                torch_output = torch_outs[name].flatten()
                self.compare(torch_output, top_mlir_output, use_cos)
                counter += 1
        if counter == 0:
            raise RuntimeError("No compare between torch outs and mlir outts")
        print("Success: Torch outs and Mlir outs are equal\n")
        for quant_mode in self.quant_modes:
            if quant_mode == "int8" or quant_mode == "int4":
                for isAsym in self.support_asym:
                    tpu_mlir, model = self.model_generate(model_name, quant_mode, isAsym)
                    self.inference_and_compare(tpu_mlir, model, input_npz, quant_mode, model_name,
                                               isAsym)
            else:
                tpu_mlir, model = self.model_generate(model_name, quant_mode)
                self.inference_and_compare(tpu_mlir, model, input_npz, quant_mode, model_name)

    #######################################################################
    # Convolution
    # ------------
    def _test_bcbinary_abnormal(self, op_type, in0_shape, in1_shape):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y):
                return op_type(x, y)

        self.trace_and_test([in0_shape,in1_shape], Model())

    #######################################################################


    def test_AddLarge(self):
        # self._test_bcbinary_abnormal(torch.add, (1, 76760, 8, 5, 4, 2), (1, 76760, 1, 5, 4, 2))
        # # need tile
        self._test_bcbinary_abnormal(torch.add, (1, 76760, 8, 5, 4, 2), (1, 76760, 1, 5, 1, 2))
        # self._test_bcbinary_abnormal(torch.add, (8, 32, 76760, 20), (8, 1, 76760, 20))
        # self._test_bcbinary_abnormal(torch.add, (1, 48, 65536), (1, 48, 1))
        # self._test_bcbinary_abnormal(torch.add, (1, 76760, 8, 5, 4, 2), (1, 1, 1, 5, 1, 2))
        # self._test_bcbinary_abnormal(torch.add, (200, 64, 8, 40), (200, 64, 8,40))
        # self._test_bcbinary_abnormal(torch.add, (1, 76760, 256), (1, 76760, 256))
        # self._test_bcbinary_abnormal(torch.add, (8, 8, 76760, 20), (8, 1, 76760, 20))

    def test_MatMulSplit(self):
        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y):
                y = torch.transpose(y,1,2)
                return torch.matmul(x,y)

        self.trace_and_test([(1,48,65536), (1,48,65536)], Model())

    def _test_Conv(self):

        def case1(conv_fun, input_shape):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.conv1 = conv_fun(8, 8, 3, 1, 1)
                    self.conv2 = conv_fun(8, 8, 3, 1, 1)

                def forward(self, x):
                    y = self.conv1(x)
                    z = self.conv2(x)
                    return y + z

            self.trace_and_test([input_shape], Model())

        def case2(conv_fun,
                  input_shape,
                  kernel_shape,
                  oc,
                  has_bias=False,
                  padding: Union[int, str, List[int]] = 0,
                  stride: Union[int, List[int]] = 1,
                  dilation: Union[int, List[int]] = 1,
                  group=1):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    filter_shape = (oc, input_shape[1] // group, *kernel_shape)
                    self.filter = torch.randn(filter_shape)
                    self.bias = torch.randn(oc) if has_bias else None

                def forward(self, x):
                    y = conv_fun(x,
                                 self.filter,
                                 bias=self.bias,
                                 padding=padding,
                                 stride=stride,
                                 dilation=dilation,
                                 groups=group)
                    return y

            self.trace_and_test([input_shape], Model())

        def case3(conv_fun,
                  input_shape,
                  kernel_shape_0,
                  oc_0,
                  kernel_shape_1,
                  oc_1,
                  has_bias_0=False,
                  padding_0: Union[int, str, List[int]] = 0,
                  stride_0: Union[int, List[int]] = 1,
                  dilation_0: Union[int, List[int]] = 1,
                  group_0=1,
                  has_bias_1=False,
                  padding_1: Union[int, str, List[int]] = 0,
                  stride_1: Union[int, List[int]] = 1,
                  dilation_1: Union[int, List[int]] = 1,
                  group_1=1):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    filter_shape_0 = (oc_0, input_shape[1] // group_0, *kernel_shape_0)
                    self.filter_0 = torch.randn(filter_shape_0)
                    self.bias_0 = torch.randn(oc_0) if has_bias_0 else None

                    filter_shape_1 = (oc_1, oc_0 // group_1, *kernel_shape_1)
                    self.filter_1 = torch.randn(filter_shape_1)
                    self.bias_1 = torch.randn(oc_1) if has_bias_1 else None

                def forward(self, x):
                    y = conv_fun(x,
                                 self.filter_0,
                                 bias=self.bias_0,
                                 padding=padding_0,
                                 stride=stride_0,
                                 dilation=dilation_0,
                                 groups=group_0)

                    z = conv_fun(y,
                                 self.filter_1,
                                 bias=self.bias_1,
                                 padding=padding_1,
                                 stride=stride_1,
                                 dilation=dilation_1,
                                 groups=group_1)
                    return z

            self.trace_and_test([input_shape], Model())

        return dict(case1=case1, case2=case2, case3=case3)

    def test_Conv1d(self):
        """Conv 1D"""

        test = self._test_Conv()
        test["case1"](nn.Conv1d, (4, 8, 28))
        test["case2"](F.conv1d, (1, 3, 32), [3], 12, has_bias=True, group=1, padding="same")
        test["case2"](F.conv1d, (2, 32, 16), [5], 64, padding=2, stride=2, dilation=1)
        # Tpu/Interfaces/BM1684X/Conv1D.cpp::152 Not supported yet.
        test["case2"](F.conv1d, (1, 3, 32), [3], 12, group=3, padding=1, stride=2)

    def test_Conv2d(self):
        """Conv 2D"""

        test = self._test_Conv()
        test["case1"](nn.Conv2d, (4, 8, 28, 28))
        test["case2"](F.conv2d, (1, 3, 32, 32), (3, 3), 12, has_bias=True, group=1, padding="same")
        test["case2"](F.conv2d, (2, 32, 16, 16), (5, 5), 64, padding=2, stride=2, dilation=1)

    def test_ConvGroup(self):
        test = self._test_Conv()
        test["case2"](F.conv2d, (1, 8, 32, 32), (3, 3), 24, group=4, padding=(1, 1))

    def test_Conv3d(self):
        """Conv 3D"""
        # conv3d is a computationally intensive operation that uses a small kernel to reduce runtime.
        test = self._test_Conv()
        test["case1"](nn.Conv3d, (4, 8, 7, 10, 20))
        test["case2"](F.conv3d, (1, 3, 6, 10, 12), (3, 3, 2),
                      12,
                      has_bias=True,
                      group=1,
                      padding="same")
        test["case2"](F.conv3d, (2, 32, 8, 10, 10), (5, 5, 3), 64, padding=2, stride=2, dilation=1)
        # Tpu/Interfaces/BM1684X/Conv3D.cpp::94 Not supported yet.
        # test["case2"](F.conv3d, (1, 3, 32, 32, 32), (3, 3, 3), 12, group=3, padding=(1, 1, 2), stride=(2, 1, 1))

    def test_ConvMerge(self):
        """Conv Merge"""
        test = self._test_Conv()
        test["case3"](F.conv2d, (1, 3, 4, 4), (1, 1), 2, (3, 3), 4, has_bias_0=False,  padding_0=0, has_bias_1=False, padding_1=0)
        test["case3"](F.conv2d, (2, 32, 16, 16), (1, 1), 64, (3, 3), 16, has_bias_0=True,  padding_0=0, has_bias_1=True, padding_1=0)
        test["case3"](F.conv2d, (2, 32, 32, 32), (1, 1), 12, (3, 3), 64, has_bias_0=True,  padding_0=1, has_bias_1=True, padding_1=1)



    #######################################################################
    # Transposed Convolution
    # ------------
    def test_ConvTrans(self):

        def test_deconv(
            input_shape,
            kernel_shape,
            stride=1,
            padding=0,
            output_padding=0,
            groups=1,
            dilation=1,
        ):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.deconv = nn.ConvTranspose2d(16, 32, 3, stride=2)
                    self.weight = torch.randn(kernel_shape)

                def forward(self, x):
                    y = self.deconv(x)
                    z = nn.functional.conv_transpose2d(
                        y,
                        self.weight,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                        groups=groups,
                        dilation=dilation,
                    )
                    return z

            self.trace_and_test([input_shape], Model())

        test_deconv((2, 16, 8, 8), (32, 8, 3, 5))
        test_deconv((2, 16, 8, 8), (32, 4, 3, 4), (2, 3), (1, 1), (0, 0), groups=2)

    #######################################################################
    # AvgPooling
    # ------------
    def _test_AvgPool(self):

        def test_case(pool_funs, input_shape, kernel_size, stride, padding, count_include_pad=True):

            fun1, fun2 = pool_funs

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.pooling = fun1(5, 2)

                def forward(self, x):
                    y = self.pooling(x)
                    z = fun2(y,
                             kernel_size,
                             stride=stride,
                             padding=padding,
                             count_include_pad=count_include_pad)
                    return z

            self.trace_and_test([input_shape], Model())

        return test_case

    def test_AvgPool1d(self):
        test = self._test_AvgPool()
        test((nn.AvgPool1d, F.avg_pool1d), (4, 8, 40), 4, 3, 2)
        test((nn.AvgPool1d, F.avg_pool1d), (1, 8, 40), 2, 1, 1, False)
        # test((nn.AvgPool1d, F.avg_pool1d), (1, 1, 1024), 510, 510, 0, False) /*not supported for bm1684x now*/

    def test_AvgPool2d(self):
        test = self._test_AvgPool()
        test((nn.AvgPool2d, F.avg_pool2d), (4, 8, 40, 30), 4, 3, 2)
        test((nn.AvgPool2d, F.avg_pool2d), (1, 64, 32, 32), (3, 2), (1, 2), (0, 1))
        if not self.is_cv18xx:
            test((nn.AvgPool2d, F.avg_pool2d), (1, 64, 32, 32), (3, 2), (2, 3), (1, 1), False)

    def test_AvgPool3d(self):
        test = self._test_AvgPool()
        test((nn.AvgPool3d, F.avg_pool3d), (4, 8, 12, 20, 24), 4, 3, 2)
        test((nn.AvgPool3d, F.avg_pool3d), (1, 3, 12, 20, 24), (3, 3, 2), (1, 1, 1), (1, 0, 1))

    #######################################################################
    # Max
    # ------------
    def test_Max(self):

        def _test_max(shape, dim, ret_case):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    max_values, max_indices = torch.max(x, dim=dim)
                    if ret_case == 0:
                        return max_indices
                    elif ret_case == 1:
                        return max_values
                    else:
                        return max_values, max_indices

            if (self.chip == "mars3"):
                self.trace_and_test([shape], Model(), [self.Desc('int', 1, 100)])
            else:
                self.trace_and_test([shape], Model())

        for ret_case in [0, 1, 2]:
            _test_max((4, 30), 1, ret_case)
            _test_max((1, 3, 64, 64), 3, ret_case)
            # _test_max((4, 384), 0, ret_case)

    def test_MulConst(self):

        def _test_mulconst(shape):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = x * 0.044
                    z = y * 2
                    return z

            self.trace_and_test([shape], Model())

        for ret_case in [0, 1, 2]:
            _test_mulconst((4, 30))

    #######################################################################
    # Min
    # ------------
    def test_Min(self):

        def _test_min(shape, dim, ret_case):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    min_values, min_indices = torch.min(x, dim=dim)
                    if ret_case == 0:
                        return min_indices
                    elif ret_case == 1:
                        return min_values
                    else:
                        return min_values, min_indices

            if (self.chip == "mars3"):
                self.trace_and_test([shape], Model(), [self.Desc('int', 1, 100)])
            else:
                self.trace_and_test([shape], Model())

        for ret_case in [0, 1, 2]:
            _test_min((4, 30), 1, ret_case)
            _test_min((1, 3, 64, 64), 3, ret_case)
            # _test_min((4, 384), 0, ret_case)

    #######################################################################
    # MaxPooling
    # ------------
    def _test_MaxPool(self):

        def test_case(pool_funs, input_shape, kernel_size, stride, padding, ceil_mode=False):
            fun1, fun2 = pool_funs

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.pooling = fun1(5, 2)

                def forward(self, x):
                    y = self.pooling(x)
                    z = fun2(
                        y,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        ceil_mode=ceil_mode,
                    )
                    return z

            self.trace_and_test([input_shape], Model())

        return test_case

    def test_MaxPool1d(self):
        test = self._test_MaxPool()
        test((nn.MaxPool1d, F.max_pool1d), (4, 8, 40), 4, 3, 2)
        test((nn.MaxPool1d, F.max_pool1d), (1, 8, 40), 2, 1, 0)
        # test((nn.MaxPool1d, F.max_pool1d), (1, 1, 1024), 510, 510, 0, False) /*not supported for bm1684x now*/

    def test_MaxPool2d(self):
        test = self._test_MaxPool()
        test((nn.MaxPool2d, F.max_pool2d), (4, 8, 40, 30), 4, 3, 2)
        test((nn.MaxPool2d, F.max_pool2d), (1, 64, 32, 32), (3, 2), (1, 2), (0, 1))
        test((nn.MaxPool2d, F.max_pool2d), (1, 64, 32, 32), (3, 5), (1, 2), (1, 1))
        test((nn.MaxPool2d, F.max_pool2d), (1, 64, 32, 32), (4, 3), (3, 2), (1, 1), True)

    def test_MaxPool3d(self):
        test = self._test_MaxPool()
        test((nn.MaxPool3d, F.max_pool3d), (4, 8, 10, 64, 64), 2, 1, 1)
        test((nn.MaxPool3d, F.max_pool3d), (1, 3, 10, 30, 40), (3, 3, 2), (1, 2, 1), (1, 0, 1))
        test((nn.MaxPool3d, F.max_pool3d), (1, 3, 10, 30, 40), (3, 3, 5), (1, 2, 2), (1, 0, 1),
             True)

    #######################################################################
    # Binary Base
    # ------------
    def _test_binary(self, op_type, in0_shape, in1_shape, alpha=None, is_reverse=False, min=-10):

        _alpha = {}
        if alpha:
            _alpha = dict(alpha=alpha)

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.weight = torch.randn(in1_shape)

            def forward(self, x):
                if is_reverse:
                    y0 = 3 - x
                else:
                    y0 = x + 3
                y1 = op_type(self.weight, y0, **_alpha)
                y2 = op_type(y0, y1, **_alpha)
                return y2

        self.trace_and_test([in0_shape], Model(), [self.Desc('float32', min)])

    #######################################################################
    # Add
    # ------------
    def test_Add(self):
        """Add"""

        self._test_binary(torch.add, (1, 3, 32, 32), (1, 3, 32, 32), 3)
        self._test_binary(torch.add, (2, 32, 16), (2, 1, 16), 3)
        self._test_binary(torch.add, (32, 32), (32))

    def test_Add5d(self):
        """AddError"""

        self._test_binary(torch.add, (1, 4, 12, 147, 147), (1, 4, 1, 147, 147))

    def test_Add6d(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y):
                return x + y

        self.trace_and_test([(1, 8, 8, 3, 4, 2), (1, 8, 1, 3, 1, 2)], Model())

    #######################################################################
    # Sub
    # ------------
    def test_Sub(self):
        """Sub"""

        self._test_binary(torch.sub, (1, 3, 32, 31), (1, 3, 32, 1), 3)
        self._test_binary(torch.sub, (2, 32, 16), (2, 1, 16), 3, is_reverse=True)
        self._test_binary(torch.sub, (32, 32), (32))

    #######################################################################
    # Mul
    # ------------
    def test_Mul(self):
        """Mul"""

        self._test_binary(torch.multiply, (1, 3, 32, 31), (1, 3, 32, 1))
        self._test_binary(torch.multiply, (2, 32, 16), (2, 1, 16))
        self._test_binary(torch.multiply, (32, 32), (32))

    #######################################################################
    # Div
    # ------------
    def test_Div(self):
        """Div"""
        if self.chip != "bm1688":
            self._test_binary(torch.div, (1, 3, 32, 31), (1, 3, 32, 1), min=0)
            self._test_binary(torch.div, (32, 32), (32), min=0)
        self._test_binary(torch.div, (2, 32, 16), (2, 1, 16), min=0)
    #######################################################################
    # Dot
    # ------------
    def test_Dot(self):
        """Dot"""
        def _test_dot(input_shape, right_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x, y):
                    x = x.view(-1)
                    y = y.view(-1)
                    z = torch.dot(x, y)
                    return z

            self.trace_and_test([input_shape, right_shape], Model())

        _test_dot((32, 1), (32, 1))
    #######################################################################
    # Compare
    # ------------
    def test_Compare(self):
        """Compare
        greater, greater_equal, less, less_equal, equal

        """

        def test_cmp_const(cmp_fun, input_shape, const):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = cmp_fun(x, const)
                    return y

            self.trace_and_test([input_shape], Model())

        self._test_binary(torch.greater, (1, 3, 32, 31), (1, 3, 32, 1))
        self._test_binary(torch.greater, (2, 32, 16), (2, 1, 16))
        self._test_binary(torch.greater, (32, 32), (32))
        self._test_binary(torch.greater_equal, (1, 3, 32, 31), (1, 3, 32, 1))
        self._test_binary(torch.less, (1, 3, 32, 31), (1, 3, 32, 1))
        self._test_binary(torch.less_equal, (1, 3, 32, 31), (1, 3, 32, 1))
        self._test_binary(torch.eq, (1, 3, 32, 31), (1, 3, 32, 1), min=0)
        self._test_binary(torch.ne, (1, 3, 32, 31), (1, 3, 32, 1), min=0)
        test_cmp_const(torch.greater, (1, 2, 3, 4), 0)
        test_cmp_const(lambda x, y: y > x, (1, 2, 3, 4), 0)
        test_cmp_const(torch.greater_equal, (1, 2, 3, 4), 0)
        test_cmp_const(lambda x, y: y >= x, (1, 2, 3, 4), 0)
        test_cmp_const(torch.less, (1, 2, 3, 4), 0)
        test_cmp_const(lambda x, y: y < x, (1, 2, 3, 4), 0)
        test_cmp_const(torch.less_equal, (1, 2, 3, 4), 0)
        test_cmp_const(lambda x, y: y <= x, (1, 2, 3, 4), 0)
        test_cmp_const(torch.eq, (1, 2, 3, 4), 0)
        test_cmp_const(lambda x, y: y == x, (1, 2, 3, 4), 0)
        test_cmp_const(torch.ne, (1, 2, 3, 4), 0)
        test_cmp_const(lambda x, y: y != x, (1, 2, 3, 4), 0)

    #######################################################################
    # LayerNorm
    # ------------
    def test_LayerNorm(self):
        normalize_shape = [40, 80]

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.layer_norm = nn.LayerNorm(normalize_shape, elementwise_affine=True)
                self.prelu = nn.PReLU()

            def forward(self, x):
                x = self.layer_norm(x)
                y = self.prelu(x)
                return x, y

        input_shape = [14, 25] + normalize_shape
        self.trace_and_test([input_shape], Model())

    #######################################################################
    # RMSNorm
    # ------------
    def test_RMSNorm(self):
        normalize_shape = [80]
        eps = 1e-5

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.weight = torch.nn.Parameter(torch.randn(normalize_shape))
                self.prelu = nn.PReLU()

            def forward(self, hidden_states: torch.Tensor):
                input_dtype = hidden_states.dtype
                variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + eps)
                rmsnorm_out = self.weight * hidden_states

                return rmsnorm_out, self.prelu(rmsnorm_out)

        input_shape = [14, 25, 40] + normalize_shape
        self.trace_and_test([input_shape], Model())

    #######################################################################
    # Normalize
    # ------------
    # # Currently(2024/08/14), out chips does not support the operator used in nn.functional.normalize including 'aten::linalg_vector_norm', 'aten::clamp_min'
    # def test_Normalize(self):
    #     p = 2.0
    #     dim = 2
    #     normalize_shape = [40, 80]

    #     class Model(torch.nn.Module):

    #         def __init__(self):
    #             super(Model, self).__init__()
    #             self.prelu = nn.PReLU()

    #         def forward(self, x):
    #             x = nn.functional.normalize(x, p = p, dim = dim) # current chip does not support noramlize operator
    #             y = self.prelu(x)
    #             return x, y

    #     input_shape = [1, 3] + normalize_shape
    #     self.trace_and_test([input_shape], Model())

    #######################################################################
    # Roll
    # ------------
    def test_Roll(self):
        """Concat"""
        def _test_Roll(in0_shape, shifts, dim = None):
            class Model(torch.nn.Module):
                def __init__(self):
                    super(Model, self).__init__()
                    self.b = torch.rand(in0_shape)
                def forward(self, x):
                    a = torch.roll(x, shifts, dim)
                    c = a + self.b
                    return c
            self.trace_and_test([in0_shape], Model())

        _test_Roll((6,8,4,4), 3, 0)
        _test_Roll((1,1,1,8), 4, 3)
        _test_Roll((1,69,8,7), 4, 2)
        _test_Roll((2,8,4,4), 1, 0)
        if not self.simple:
            _test_Roll((4,2), 1)
            _test_Roll((4,2), 1, 0)
            _test_Roll((4,2), -1, 0)
            _test_Roll((4,2), (4,2), (0,1))
            _test_Roll((4,2), (123,-133), (1,0))


    #######################################################################
    # Chunk
    # ------------
    def test_Chunk(self):

        class Model0(torch.nn.Module):

            def __init__(self):
                super(Model0, self).__init__()

            def forward(self, x):
                a, b, c = torch.chunk(x, 3, -1)
                d = a * b + c
                return d

        class Model1(torch.nn.Module):

            def __init__(self):
                super(Model1, self).__init__()
                self.weight = torch.randn((4, 16, 30))

            def forward(self, x):
                a, b, c = torch.chunk(self.weight, 3, -1)
                d = a * b + c + x
                return d

        self.trace_and_test([(4, 16, 30)], Model0())
        #self.trace_and_test([(4, 16, 10)], Model1())

    #######################################################################
    # Clamp
    # ------------
    def test_Clamp(self):

        class Model0(torch.nn.Module):

            def __init__(self):
                super(Model0, self).__init__()

            def forward(self, x):
                y = x * 30
                return torch.clamp(y, -10, 20)

        self.trace_and_test([(4, 16, 30)], Model0())

    #######################################################################
    # SplitUesless
    # ------------
    def test_SplitReshape(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                a, b, c, d = torch.chunk(x, 4, 1)
                a = torch.reshape(a, (1, 1, -1, 3))
                b = torch.reshape(b, (1, 1, -1, 3))
                c = torch.reshape(c, (1, 1, -1, 3))
                d = torch.reshape(d, (1, 1, -1, 3))
                return a, b, c, d

        self.trace_and_test([(1, 4, 16, 30)], Model())

    #######################################################################
    # InfError
    # ------------
    def test_InfError(self):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, mask):
                x = torch.masked_fill(x, mask, float("-inf"))
                y = torch.softmax(x, -1)
                return y

        self.trace_and_test([(1, 32, 128, 128), (1, 1, 128, 128)], Model(),
                                [self.Desc('float', -10, 10), self.Desc('int', 0, 2)])

    #######################################################################
    # MatMul
    # ------------
    def test_MatMul(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y, z):
                out = torch.matmul(x, y) + z
                return out

        self.trace_and_test([(4, 8, 49, 32), (4, 8, 32, 49), (1, 1, 1, 49)], Model())
        self.trace_and_test([(1, 8, 3, 2), (4, 1, 2, 3), (1, 1, 3, 3)], Model())
        self.trace_and_test([(1, 4, 20, 6), (2, 1, 6, 3), (1, 1, 3)], Model())
        self.trace_and_test([(4, 20, 6), (1, 6, 3), (1, 1, 3)], Model())
        if not self.is_cv18xx:
            self.trace_and_test([(7, 256), (5, 256, 64), (1, 1, 64)], Model())

    #######################################################################
    # test Connect Pass
    # ------------
    def test_Connect(self):

        def test_connect_(x_shape: tuple, filter_shape: tuple, bias_shape: tuple):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.filter = torch.randn(*filter_shape)
                    self.bias = torch.randn(*bias_shape)

                def forward(self, x):
                    out = torch.matmul(x, self.filter) + self.bias
                    return out

            self.trace_and_test([x_shape], Model())

        test_connect_((2, 4096, 1024), (2, 1024, 4096), (1, 1, 4096))
        test_connect_((2, 1024, 4096), (2, 4096, 1024), (1, 1, 1024))

    #######################################################################
    # test Weight multiple use Pass
    # ------------
    def test_WeightMultiUse(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.filter = torch.randn(32, 64)
                self.bias = torch.randn(1, 64)

            def forward(self, x, y):
                a = torch.matmul(x, self.filter) + self.bias
                b = y + self.filter + self.bias
                return a + b

        self.trace_and_test([(32, 32), (32, 64)], Model())

    #######################################################################
    # ConstantFill
    # ------------
    def test_ConstantFill(self):

        def _test_constant_fill(func, shape, type=None):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = func(shape, dtype=type)
                    z = y + x
                    return z * 0.5

            self.trace_and_test([shape], Model())

        _test_constant_fill(torch.zeros, (2, 3, 64, 64), torch.float32)
        # _test_constant_fill(torch.zeros, (3, 64, 64))
        # _test_constant_fill(torch.ones, (1, 3, 64, 64), torch.float32)
        _test_constant_fill(torch.ones, (3, 64, 64))

    #######################################################################
    # Embedding
    # ------------
    def test_Embedding(self):

        def _test_embedding(shape, n, d):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.embedding = nn.Embedding(n, d)

                def forward(self, x):
                    y = self.embedding(x)
                    return y

            self.trace_and_test([shape], Model(), [self.Desc('int32', 0, n)])

        _test_embedding((2, 3, 64), 512, 768)
        _test_embedding((2, 64), 20, 30)
        # _test_embedding((1, 384), 30522, 1024)

    #######################################################################
    # To
    # ------------
    def test_To(self):

        def _test_to(shape, dtype):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = x.to(dtype) + 1
                    return y

            self.trace_and_test([shape], Model())

        for dtype in [torch.long, torch.int64, torch.float16]:
            _test_to((2, 3, 64), dtype)

    #######################################################################
    # Type_as
    # ------------
    def test_Type_as(self):

        def _test_type_as(shape, dtype):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.other = torch.tensor(-10, dtype=dtype)

                def forward(self, x):
                    y = x.type_as(self.other) + 1
                    return y

            self.trace_and_test([shape], Model())

        for dtype in [torch.long, torch.int64, torch.float16]:
            _test_type_as((4, 3, 100, 100), dtype)

    #######################################################################
    # Reduce
    # ------------
    def test_Reduce(self):

        def _test_reduce(func, shape, dim=None, keepdim=False):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = func(x, dim, keepdim)
                    return y

            self.trace_and_test([shape], Model())

        # _test_reduce(torch.sum, (2, 3, 64, 64))
        _test_reduce(torch.sum, (1, 3, 64, 64), 1, True)
        _test_reduce(torch.sum, (2, 3, 64, 64), [0, 1, 2])
        # _test_reduce(torch.mean, (2, 3, 64, 64))
        _test_reduce(torch.mean, (1, 3, 64, 64), 1, True)
        _test_reduce(torch.mean, (2, 3, 64, 64), [1, 2])
        # cv183x not supported below
        # _test_reduce(torch.mean, (1, 48, 65536), 2, True)

    #######################################################################
    # Pow
    # ------------
    def test_Pow(self):

        def _test_pow(shape, exp, min=0):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = torch.pow(x, exponent=exp)
                    return y

            self.trace_and_test([shape], Model(), [self.Desc('float32', min)])

        _test_pow((2, 3, 64, 64), 2, -10)
        _test_pow((3, 64, 64), 3)
        _test_pow((64, 64), 0.5)
        _test_pow((1, 3, 100, 100), 5, min=-10)
        _test_pow((1, 3, 100, 100), 4, min=-10)

    #######################################################################
    # MeshGrid
    # ------------
    def test_MeshGrid(self):

        def _test_mesh_grid(shape, indexing=None):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    dim2 = x.size(dim=2)
                    dim3 = x.size(dim=3)
                    in0 = torch.arange(0,dim2,1)
                    in1 = torch.arange(0,dim3*2,2)
                    y0, y1 = torch.meshgrid(in0, in1, indexing=indexing)
                    if indexing == 'xy':
                        y0 = y0.transpose(0,1)
                        y1 = y1.transpose(0,1)
                    y = torch.stack([y0 + x, y1 - x], dim=0)
                    return y

            self.trace_and_test([shape], Model())

        _test_mesh_grid((1, 3, 64, 32), 'ij')
        _test_mesh_grid((1, 3, 32, 64))
        _test_mesh_grid((1, 3, 64, 4), 'xy')

    #######################################################################
    # NewZeros
    # ------------
    def test_NewZeros(self):

        def _test_newzeros(shape, dtype=None):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.coeff = torch.randn(1, 3)

                def forward(self, x):
                    y = self.coeff.new_zeros(shape, dtype=dtype)
                    y = y.to(torch.float32) + x
                    return y * 0.5

            self.trace_and_test([shape], Model())

        _test_newzeros((2, 3, 64, 64), torch.float32)
        _test_newzeros((3, 64, 64), torch.int32)
        _test_newzeros((64, 64))

    #######################################################################
    # NonZero
    # ------------
    def test_NonZero(self):

        def _test_NonZero(in0_shape):
            class Model(nn.Module):
                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = torch.nonzero(x)
                    return y

            self.trace_and_test([in0_shape], Model())

        _test_NonZero([1, 4, 64, 32])
        _test_NonZero([64])

    #######################################################################
    # New_full
    # ------------
    def test_New_full(self):

        def _test_new_full(size, fill_value, dtype=None):

            class Model(torch.nn.Module):
                def __init__(self):
                    super(Model, self).__init__()
                    self.coeff = torch.randn(1, 3)

                def forward(self, x):
                    y = self.coeff.new_full(size, fill_value, dtype=dtype)
                    y = y + x
                    return y * 10

            self.trace_and_test([size], Model())

        _test_new_full((3,4), 3.14, torch.float32)
        _test_new_full((4,), 3)


    #######################################################################
    # Reshape
    # ------------
    def test_Reshape(self):
        """Reshape"""

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                a = torch.reshape(x, (64, -1, 1024))  # 64, 8, 1024
                b = a.transpose(0, 1).contiguous()  # 8, 64, 1024
                c = torch.reshape(b, (8, 64, 64, 16))  # 64, 8, 64, 16
                d = c.to(torch.float32)
                e = d.transpose(1, 2)  # 64, 64, 8, 16
                return e

        in_shape = (512, 1024)
        self.trace_and_test([in_shape], Model())

    #######################################################################
    # Depth2Space
    # ------------
    def test_D2SPattern(self):
        """Depth2Space"""
        def _test_case1(in_shape):
            class Model1(nn.Module):

                def __init__(self):
                    super(Model1, self).__init__()

                def forward(self, x):
                    a = torch.reshape(x, (1,14,14,-1))      # 1,14,14,16
                    b = torch.reshape(a, (1,7,2,7,2,-1))    # 1,7,2,7,2,16
                    c = b.permute(0,1,3,4,2,5)              # 1,7,7,2,2,16
                    d = torch.reshape(c, (1,7,7,64))        # 1,7,7,64
                    return d

            self.trace_and_test([in_shape], Model1())

        def _test_case2(in_shape):
            class Model2(nn.Module):

                def __init__(self):
                    super(Model2, self).__init__()

                def forward(self, x):
                    b = torch.reshape(x, (1,7,2,7,2,-1))    # 1,7,2,7,2,16
                    c = b.permute(0,1,3,4,2,5)              # 1,7,7,2,2,16
                    d = torch.reshape(c, (1,7,7,64))        # 1,7,7,64
                    return d

            self.trace_and_test([in_shape], Model2())

        in_shape1 = (1,196,16)
        _test_case1(in_shape1)

        in_shape2 = (1,14,14,16)
        _test_case2(in_shape2)

    #######################################################################
    # PRelu
    # ------------
    def test_PRelu(self):
        """PRelu"""

        def _test_prelu(input_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    # self.weight0 = torch.randn(1)
                    self.weight1 = torch.randn(input_shape[1])

                def forward(self, x):
                    # y0 = F.prelu(x, self.weight0)
                    y1 = F.prelu(x, self.weight1)
                    return y1

            self.trace_and_test([input_shape], Model())

        _test_prelu((1, 3, 32, 32))
        _test_prelu((2, 32, 16))
        _test_prelu((32, 32))

    #######################################################################
    # BatchNorm
    # ------------
    def test_BatchNorm(self):
        """BatchNorm"""

        def _test_batchnorm(op_type, input_shape, eps=1e-5, affine=True):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    c = input_shape[1]
                    self.bm = op_type(c, affine=affine, eps=eps)
                    if self.bm.weight is not None:
                        torch.nn.init.uniform_(self.bm.weight.data)
                    if self.bm.bias is not None:
                        torch.nn.init.uniform_(self.bm.bias.data)

                def forward(self, x):
                    y = self.bm(x)

                    return y

            self.trace_and_test([input_shape], Model())

        _test_batchnorm(nn.BatchNorm2d, (3, 4, 32, 32), 3e-5)
        _test_batchnorm(nn.BatchNorm2d, (2, 32, 16, 16), affine=False)
        _test_batchnorm(nn.BatchNorm1d, (3, 32), 3e-5)
        _test_batchnorm(nn.BatchNorm1d, (3, 32, 32), affine=False)
        if not self.is_cv18xx:
            _test_batchnorm(nn.BatchNorm3d, (3, 4, 5, 32, 32), 3e-5)
            _test_batchnorm(nn.BatchNorm3d, (2, 32, 16, 16, 3), affine=False)

    #######################################################################
    # InstanceNorm
    # ------------
    def test_InstanceNorm(self):
        """InstanceNorm"""

        def _test_instancenorm(op_type, input_shape, feature, eps=1e-5, affine=True):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.im = op_type(feature, affine=affine, eps=eps)

                def forward(self, x):
                    y = self.im(x)

                    return y

            self.trace_and_test([input_shape], Model())

        _test_instancenorm(nn.InstanceNorm2d, (3, 4, 32, 32), 4, 3e-5)
        # TODO: support squeeze & unsqueeze
        # _test_instancenorm(nn.InstanceNorm2d, (32, 16, 16), 16, affine=False)
        if (not self.is_cv18xx):
            # _test_instancenorm(nn.InstanceNorm1d, (3, 32), 3, 3e-5)
            _test_instancenorm(nn.InstanceNorm1d, (3, 32, 32), 32, affine=False)
            # _test_instancenorm(nn.InstanceNorm3d, (4, 5, 32, 32), 4, 3e-5)
            _test_instancenorm(nn.InstanceNorm3d, (2, 32, 16, 16, 3), 32, affine=False)

    #######################################################################
    # Interp
    # ------------
    def test_Interp(self):
        """Interp"""

        def _test_interp(input_shape, size=None, scale=None, mode=None):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = nn.functional.interpolate(x,
                                                  size,
                                                  scale,
                                                  mode=mode,
                                                  align_corners=False)
                    return y

            self.trace_and_test([input_shape], Model())

        _test_interp((1, 3, 100, 100), None, 4, 'bilinear')
        _test_interp((1, 1, 32, 32), None, 10,'bilinear')
        _test_interp((2, 32, 16, 16), None, 16, 'bilinear')
        if self.chip in ["bm1684x", "bm1688"]:
            _test_interp((2, 3, 224), None, 2, 'linear')
            _test_interp((2, 3, 224), (100), None, 'linear')

    #######################################################################
    # Interp2
    # ------------
    def test_Interp2(self):
        """Interp2"""
        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.new_hw = torch.Tensor([24,32])

            def forward(self, x):
                s = [int(self.new_hw[0]), int(self.new_hw[1])]
                y = nn.functional.interpolate(x,
                                              s,
                                              None,
                                              mode="bilinear",
                                              align_corners=False)
                return y

        self.trace_and_test([(1, 3, 64, 64)], Model())

    #######################################################################
    # BMM
    # ------------
    def test_BatchMatMul(self):
        """BatchMatMul"""

        def _test_batchmatmul(input_shape, right_shape, is_cv18xx):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.weight0 = torch.randn(input_shape)
                    self.weight1 = torch.randn(right_shape)

                def forward(self, x, y):
                    z1 = torch.bmm(x, self.weight1)
                    if is_cv18xx:
                        z2 = torch.bmm(x, y)
                    else:
                        z2 = torch.bmm(self.weight0, y)
                    z3 = torch.transpose(z2, 1, 2)
                    z4 = torch.bmm(z1, z3)
                    return z4

            self.trace_and_test([input_shape, right_shape], Model())

        _test_batchmatmul((3, 32, 32), (3, 32, 64), self.is_cv18xx)
        _test_batchmatmul((2, 32, 16), (2, 16, 34), self.is_cv18xx)

    #######################################################################
    # MM
    # ------------
    def test_MM(self):
        """MM"""

        def _test_mm(input_shape, right_shape, is_cv18xx):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.weight0 = torch.randn(input_shape)
                    self.weight1 = torch.randn(right_shape)
                    self.dim = len(input_shape)

                def forward(self, x, y):
                    z1 = torch.mm(x, self.weight1)
                    if is_cv18xx:
                        z2 = torch.mm(x, y)
                    else:
                        z2 = torch.mm(self.weight0, y)
                    z3 = torch.transpose(z2, 1, 0)
                    z4 = torch.mm(z1, z3)
                    return z4

            self.trace_and_test([input_shape, right_shape], Model())

        _test_mm((32, 32), (32, 64), self.is_cv18xx)
        _test_mm((32, 16), (16, 34), self.is_cv18xx)

    #######################################################################
    # MV
    # ------------
    def test_MV(self):
        """MV"""

        def _test_mv(input_shape, right_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x, y):
                    y = y.view(-1)
                    z = torch.mv(x, y)
                    return z

            self.trace_and_test([input_shape, right_shape], Model())

        _test_mv((64, 32), (32, 1))
        _test_mv((32, 16), (16, 1))

    #######################################################################
    # Addmm
    # ------------
    def test_Addmm(self):
        """Addmm"""

        def _test_addmm(beta, alpha):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x, y, z):
                    o = torch.addmm(beta, x, alpha, y, z)
                    return o

            self.trace_and_test([(24, 32), (24, 16), (16, 32)], Model())

        _test_addmm(1.0, 1.0)
        # _test_addmm(0.5, 0.3) # need to support add with coeff

    #######################################################################
    # Arange
    # ------------
    def test_Arange(self):
        """Arange"""

        def _test_arange(end, start=None, step=None):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    if start is None:
                        a = torch.arange(end)
                    elif step is None:
                        a = torch.arange(start, end)
                    else:
                        a = torch.arange(start, end, step)
                    b = x + a
                    return b

            sta = start if start is not None else 0
            ste = step if step is not None else 1
            out_size = (end - sta) // ste
            self.trace_and_test([(32, out_size)], Model())

        _test_arange(60, 0, 1)
        _test_arange(60, 1)
        _test_arange(60)
        _test_arange(60, 0, 2)

    #######################################################################
    # Unsqueeze
    # ------------
    def test_Unsqueeze(self):
        """Unsqueeze"""

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y):
                a = torch.unsqueeze(y, 1)
                b = x + a
                return b

        self.trace_and_test([(32, 16, 28), (32, 28)], Model())

    #######################################################################
    # Gather
    # ------------
    def test_Gather(self):
        """Gather"""

        def _test_gather(in0_shape, in1_shape):
            input_1 = self.Desc(np.float32)
            input_2 = self.Desc(np.int64, 0, 5538)

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x, y):
                    return torch.gather(x, 2, y)
            self.trace_and_test([in0_shape, in1_shape], Model(), [input_1, input_2])

        _test_gather((10, 349, 5538), (10, 349, 1))

    #######################################################################
    # GatherMulConst
    # ------------
    def test_GatherMulConst(self):
        """Gather"""

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.embed = nn.Embedding(128, 64)

            def forward(self, indice):
                return self.embed(indice) * 1.25
        self.trace_and_test([(4, 28)], Model(), [self.Desc("int64", 0, 64)])


    #######################################################################
    # GroupNorm
    # ------------
    def test_GroupNorm(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.group_norm = nn.GroupNorm(8, 64)
                nn.init.normal_(self.group_norm.weight, std=0.01)
                nn.init.normal_(self.group_norm.bias, std=0.01)

            def forward(self, x):
                return self.group_norm(x)

        self.trace_and_test([(4, 64, 16, 16)], Model())

    #######################################################################
    # Permute
    # ------------
    def test_Permute(self):
        """Permute"""

        def _test_permute(in_shape, dims):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y1 = torch.permute(x, dims=dims)
                    return y1

            self.trace_and_test([in_shape], Model())
        # cv183x regression falure
        #_test_permute((2, 3, 16, 16), (3, 2, 1, 0))

        _test_permute((1, 3, 32, 32), (0, 3, 1, 2))
        _test_permute((2, 32, 16), (2, 0, 1))
        _test_permute((32, 32), (1, 0))

    def test_Permute2(self):
        """Permute"""

        def _test_permute(in_shape, dims):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y1 = torch.permute(x, dims=dims)
                    return y1

            self.trace_and_test([in_shape], Model())

        _test_permute((2, 3, 16, 16), (3, 2, 1, 0))
    #######################################################################
    # T
    # ------------
    def test_T(self):
        """T"""

        def _test_t(in_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    x = torch.concat((x, x))
                    x = torch.t(x)
                    y1 = torch.concat((x, x))
                    return y1

            self.trace_and_test([in_shape], Model())

        _test_t((32, 32))
        if not self.is_cv18xx:
            _test_t((32, ))

    #######################################################################
    # MaskedFill
    # ------------
    def test_MaskedFill(self):
        def _test_masked_fill(in_shape, mask_shape, is_local):
            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x, mask):
                    if is_local:
                        x += x
                        x -= 2
                        x *= 2
                        x += 1
                    x = torch.masked_fill(x, mask, 5)
                    if is_local:
                        x += 1
                    return x

            self.trace_and_test([in_shape, mask_shape], Model(),
                                [self.Desc('float', -10, 10), self.Desc('int', 0, 2)])

        dims = [3, 4, 5]
        shape = [1, 3, 128, 300, 2]
        for dim in dims:
            shapes = [shape[: dim], shape[: dim]]
            odd = True
            for i in range(dim):
                shapes[odd][i] = 1
                odd = not odd
            _test_masked_fill(tuple(shapes[0]), tuple(shapes[1]), False)
        _test_masked_fill(([1, 3, 1, 300]), ([1, 1, 128, 300]), True)

    #######################################################################
    # Math: cos/sin/tan/tanh
    # ------------
    def test_Math(self):
        """Tanh"""

        def _test_math(func):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = func(x)
                    return y

            self.trace_and_test([(4, 3, 16, 16)], Model())

        for f in [torch.cos, torch.cosh, torch.sin, torch.sinh, torch.tan, torch.tanh, torch.exp, torch.sort]:
            _test_math(f)

    #######################################################################
    # arctan
    # ------------
    def test_Arctan(self):
        """Arctan"""

        def _test_arctan(func, min=-10, max=10):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = func(x)
                    return y

            self.trace_and_test([(4, 3, 16, 16)], Model(), [self.Desc('float32', min, max)])

        for f in [torch.arctan]:
            _test_arctan(f)

    #######################################################################
    # arctanh
    # ------------
    def test_Arctanh(self):
        """Arctanh"""

        def _test_arctanh(func, min=-0.99, max=0.99):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = func(x)
                    return y

            self.trace_and_test([(4, 3, 16, 16)], Model(), [self.Desc('float32', min, max)])

        for f in [torch.arctanh]:
            # The value range of arctanh is (-1,1)
            _test_arctanh(f)

    #######################################################################
    # arccos
    # ------------
    def test_Arccos(self):
        """Arccos"""

        def _test_arccos(func, min=-1, max=1):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = func(x)
                    return y

            self.trace_and_test([(4, 3, 16, 16)], Model(), [self.Desc('float32', min, max)])

        for f in [torch.arccos]:
            # The value range of arctanh is (-1,1)
            _test_arccos(f)

    #######################################################################
    # arg
    # ------------
    def test_Arg(self):
        """Arg"""

        def _test_arg(func, axis, keepdim):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = func(x, axis, keepdim=keepdim)
                    return y

            if (self.chip == "mars3"):
                self.trace_and_test([(4, 3, 16, 16)], Model(), [self.Desc('int', 1, 100)])
            else:
                self.trace_and_test([(4, 3, 16, 16)], Model())

        for f in [torch.argmin, torch.argmax]:
            for axis in [0, 1, 2, 3]:
                for keepdim in [True, False]:
                    _test_arg(f, axis, keepdim)

    #######################################################################
    # Tile
    # ------------
    def test_Tile(self):
        """Tile"""

        def _test_tile(in_shape, repeats):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y1 = torch.tile(x, repeats)
                    return y1

            self.trace_and_test([in_shape], Model())

        _test_tile((1, 3, 32, 32), (1, 3, 1, 2))
        _test_tile((2, 32, 16), (2, 1))
        _test_tile((32, 16), (1, 2, 1))

    #######################################################################
    # Repeat
    # ------------
    def test_Repeat(self):
        """Repeat"""

        def _test_repeat(in_shape, repeats):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y1 = x.repeat(repeats)
                    # y1 = torch.tile(x, repeats)
                    return y1

            self.trace_and_test([in_shape], Model())

        _test_repeat((1, 3, 32, 32), (1, 3, 1, 2))
        _test_repeat((2, 32, 16), (2, 1, 1))
        _test_repeat((32, 16), (1, 2, 1))

    #######################################################################
    # Transpose
    # ------------
    def test_Transpose(self):
        """Transpose"""

        def _test_transpose(in_shape, dims):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y1 = torch.transpose(x, dim0=dims[0], dim1=dims[1])
                    return y1

            self.trace_and_test([in_shape], Model())

        _test_transpose((1, 3, 32, 32), (0, 3))
        _test_transpose((2, 32, 16), (2, 0))
        _test_transpose((32, 32), (1, 0))

    #######################################################################
    # View
    # ------------
    def test_View(self):
        """View"""

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                a = x.view(64, -1, 1024)  # 64, 8, 1024
                b = a.transpose(0, 1)  # 8, 64, 1024
                c = b.view(8, 64, 64, 16)  # 64, 8, 64, 16
                d = c.transpose(1, 2)  # 64, 64, 8, 16
                return d

        in_shape = (512, 1024)
        self.trace_and_test([in_shape], Model())

    #######################################################################
    # ChannelShuffle
    # ------------
    def test_ChannelShuffle(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.channel_shuffle = nn.ChannelShuffle(2)

            def forward(self, x):
                x = self.channel_shuffle(x)
                return x

        self.trace_and_test([(1, 4, 100, 100)], Model())

    #######################################################################
    # PixelShuffle
    # ------------
    def test_PixelShuffle(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.pixel_shuffle = nn.PixelShuffle(2)

            def forward(self, x):
                x = self.pixel_shuffle(x)
                return x

        self.trace_and_test([(1, 16, 32, 32)], Model())

    #######################################################################
    # ScaleDotAttention
    # ------------
    def test_ScaleDotAtten(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y, z):
                res = torch.nn.functional.scaled_dot_product_attention(x, y, z)
                return res

        # self.trace_and_test([(1,8,64,64), (1,8,64,64), (1,8,64,64)], Model())
    #######################################################################
    # PixelUnshuffle
    # ------------
    def test_PixelUnshuffle(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.pixel_unshuffle = nn.PixelUnshuffle(2)

            def forward(self, x):
                x = self.pixel_unshuffle(x)
                return x

        self.trace_and_test([(1, 4, 64, 64)], Model())

    #######################################################################
    # Where
    # ------------
    def test_Where(self):
        """Where"""

        def _test_where(in0_shape, in1_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.weight = torch.randn(in1_shape)

                def forward(self, x):
                    y = torch.where(x > 0, 1.0, -1.0)
                    y1 = torch.where(self.weight > 0, y, x)
                    return y1

            desc = self.Desc('float32', -1, 1)
            self.trace_and_test([in0_shape], Model(), [desc])

        _test_where((1, 3, 32, 32), (1, 3, 32, 32))
        _test_where((3, 32, 16), (3, 32, 16))
        # TODO: where backend do not support broadcast
        # _test_where((2, 32, 16), (32, 1))
        # _test_where((32, 32), (1, 32))

    #######################################################################
    # Attention
    # ------------
    def test_Attention(self):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)

            def forward(self, x, y, z):
                out, out_w = self.attention(x, y, z)
                return out, out_w

        self.trace_and_test([(1, 4, 64), (1, 4, 64), (1, 4, 64)], Model())

    def test_AttentionNew(self):

        def _test_attention0(shape, d, head, has_musk=True):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.q_w = torch.randn((shape[2], d*head)) / np.sqrt(d)
                    self.q_b = torch.randn((1, 1, d*head))
                    self.k_w = torch.randn((shape[2], d*head)) / np.sqrt(d)
                    self.k_b = torch.randn((1, 1, d*head))
                    self.v_w = torch.randn((shape[2], d*head)) / np.sqrt(d)
                    self.v_b = torch.randn((1, 1, d*head))
                    self.o_w = torch.randn((d*head, shape[2])) / np.sqrt(d)
                    self.o_b = torch.randn((1, 1, shape[2]))
                    self.musk = -((torch.randn((shape[0],1,1,shape[1])) > 0) * 10000)

                def forward(self, x):
                    q = torch.matmul(x, self.q_w) + self.q_b
                    q = q.reshape((shape[0], shape[1], head, d))
                    q = q.transpose(1,2)
                    k = torch.matmul(x, self.k_w) + self.k_b
                    k = k.reshape((shape[0], shape[1], head, d))
                    k = k.transpose(1,2)
                    k = k.transpose(3,2)
                    m0 = torch.matmul(q, k)
                    m0 = m0 / np.sqrt(d)
                    if has_musk:
                        m0 = m0 + self.musk
                    m0 = torch.softmax(m0, 3)
                    v = torch.matmul(x, self.v_w) + self.v_b
                    v = v.reshape((shape[0], shape[1], head, d))
                    v = v.transpose(1,2)
                    m1 = torch.matmul(m0, v)
                    m1 = m1.transpose(1,2)
                    m1 = m1.reshape(shape[0], shape[1], head*d)
                    y = torch.matmul(m1, self.o_w) + self.o_b
                    y = y + 1
                    return y

            self.trace_and_test([shape], Model(), [self.Desc('float32', -1, 1)])

        def _test_attention1(shape0, shape1, d, head, has_bias=True):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.q_w = torch.randn((shape0[2], d*head)) / np.sqrt(d)
                    self.q_b = torch.randn((1, 1, d*head)) * 0.1
                    self.k_w = torch.randn((shape1[2], d*head)) / np.sqrt(d)
                    self.k_b = torch.randn((1, 1, d*head)) * 0.1
                    self.v_w = torch.randn((shape1[2], d*head)) / np.sqrt(d)
                    self.v_b = torch.randn((1, 1, d*head)) * 0.1
                    self.o_w = torch.randn((d*head, shape0[2])) / np.sqrt(d)
                    self.o_b = torch.randn((1, 1, shape0[2])) * 0.1

                def forward(self, x, x1):
                    q = torch.matmul(x, self.q_w)
                    if has_bias:
                        q = q + self.q_b
                    q = q.reshape((shape0[0], shape0[1], head, d))
                    q = q.transpose(1,2)
                    k = torch.matmul(x1, self.k_w)
                    if has_bias:
                        k = k + self.k_b
                    k = k.reshape((shape1[0], shape1[1], head, d))
                    k = k.transpose(1,2)
                    k = k.transpose(3,2)
                    m0 = torch.matmul(q, k)
                    m0 = m0 / np.sqrt(d)
                    m0 = torch.softmax(m0, 3)
                    v = torch.matmul(x1, self.v_w)
                    if has_bias:
                        v = v + self.v_b
                    v = v.reshape((shape1[0], shape1[1], head, d))
                    v = v.transpose(1,2)
                    m1 = torch.matmul(m0, v)
                    m1 = m1.transpose(1,2)
                    m1 = m1.reshape(shape0[0], shape0[1], head*d)
                    y = torch.matmul(m1, self.o_w) + self.o_b
                    y = y + 1
                    return y

            self.trace_and_test([shape0, shape1], Model(), [self.Desc('float32', -1, 1), self.Desc('float32', -1, 1)])

        # _test_attention0((1, 4096, 128), 64, 2)
        _test_attention0((1, 384, 768), 64, 12)
        # _test_attention0((2, 384, 64), 32, 2)
        # _test_attention0((2, 1024, 640), 80, 2, False)
        # _test_attention0((2, 256, 1280), 160, 2, False)
        if not self.simple:
            # _test_attention0((1, 4096, 320), 40, 2, False)
            # _test_attention1((2, 4096, 320), (2, 128, 768), 40, 8)
            # _test_attention1((2, 256, 1280), (2, 77, 768), 160, 2, False)
            # _test_attention1((2, 1024, 640), (2, 77, 768), 80, 2, False)
            _test_attention1((2, 4096, 320), (2, 77, 768), 40, 8, False)
            # _test_attention1((1, 384, 64), (1, 384, 64), 32, 2, False)

    def test_FAttention(self):

        def _test_flash_attention(batch, d, q_head, kv_head, mq, mk, has_mask):
            def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
                batch, seq_len, num_head, head_dim = kv.shape
                if n_rep == 1:
                    return kv
                kv = kv[:, :, :, None, :].expand(batch, seq_len, num_head, n_rep, head_dim)
                return kv.reshape(batch, seq_len, num_head * n_rep, head_dim)

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.mask = - \
                        ((torch.randn((batch, 1, mq, mk)) > 0) * 10000)
                    self.head_rep = q_head // kv_head
                    assert (kv_head * self.head_rep == q_head)

                def forward(self, q, k, v):
                    q = q + 1
                    k = k + 1
                    v = v + 1
                    k = repeat_kv(k, self.head_rep)
                    v = repeat_kv(v, self.head_rep)
                    q = q.transpose(1, 2)
                    k = k.transpose(1, 2)
                    k = k.transpose(3, 2)
                    m0 = torch.matmul(q, k)
                    m0 = m0 / np.sqrt(d)
                    if has_mask:
                        m0 = m0 + self.mask
                    m0 = torch.softmax(m0, 3)
                    v = v.transpose(1, 2)
                    m1 = torch.matmul(m0, v)
                    m1 = m1.transpose(1, 2)
                    m1 = m1.reshape(batch, mq, q_head*d)
                    return m1

            shape0 = [batch, mq, q_head, d]
            shape1 = [batch, mk, kv_head, d]
            shape2 = [batch, mk, kv_head, d]
            max = 10
            min = -10
            self.trace_and_test([shape0, shape1, shape2], Model(), [self.Desc(
                'float32', min, max), self.Desc('float32', min, max), self.Desc('float32', min, max)])

        # for test accuracy
        # _test_flash_attention(1, 128, 28, 4, 1, 8193,   True)
        # _test_flash_attention(1, 128, 28, 4, 256, 513,  True)
        # _test_flash_attention(1, 128, 32, 32, 1, 8193,  True)
        # _test_flash_attention(1, 128, 32, 32, 256, 513, True)
        # _test_flash_attention(1, 128, 64, 8, 1, 8193,   True)
        # _test_flash_attention(1, 128, 64, 8, 256, 513,  True)
        # _test_flash_attention(1, 128, 4, 4, 128, 128, True)
        # _test_flash_attention(1, 128, 4, 1, 128, True)

        # for test performance
        # _test_flash_attention(1, 128, 32, 32, 512, 512, True)
        # _test_flash_attention(1, 128, 32, 32, 1024, 1024, True)
        # _test_flash_attention(1, 128, 32, 32, 2048, 2048, True)
        # _test_flash_attention(1, 128, 32, 32, 4096, 4096, True)

        # _test_flash_attention(1, 128, 32, 32, 1, 512, True)
        # _test_flash_attention(1, 128, 32, 32, 1, 1024, True)
        # _test_flash_attention(1, 128, 32, 32, 1, 2048, True)
        # _test_flash_attention(1, 128, 32, 32, 1, 4096, True)

    #######################################################################
    # MatmulSlice
    # ------------

    def test_MatMulSlice(self):

        def _test_matmul_slice(shape, size, num):
            import math
            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.w = torch.randn((shape[2], size * num)) / math.sqrt(shape[2])
                    self.b = torch.randn(size * num)

                def forward(self, x):
                    m = torch.matmul(x, self.w)# + self.b
                    for i in range(num):
                      s = m[:, :, i * size : (i+1) * size]
                      if i == 0:
                        y = s
                      else:
                        y *= s
                    return y

            self.trace_and_test([shape], Model(), [self.Desc('float32', -1, 1)])

        _test_matmul_slice((1, 384, 768), 128, 3)
        _test_matmul_slice((1, 384, 768), 96, 4)

    #######################################################################
    # Select
    # ------------
    def test_Select(self):
        """Select"""

        def _test_select(in0_shape, dim, index):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.const = torch.randn(in0_shape, dtype=torch.float32)
                    self.shape_checker = torch.select(
                        self.const, dim=dim, index=index)

                def forward(self, x):
                    y = torch.select(x, dim=dim, index=index)
                    y += self.shape_checker
                    return y

            self.trace_and_test([in0_shape], Model())

        _test_select((1, 3, 32, 32), 2, 13)
        _test_select((3, 32, 16), 0, 2)
        _test_select((32, 16), 1, 4)
        _test_select((10, 20, 30, 40), 2, 5)

    #######################################################################
    # Split
    # ------------
    def test_Split(self):
        """Split"""

        def _test_split(in0_shape, dim, num):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y1 = torch.split(x, dim=dim, split_size_or_sections=num)
                    return y1

            self.trace_and_test([in0_shape], Model())

        _test_split((1, 3, 32, 32), 2, 8)
        _test_split((3, 32, 16), 0, (1, 2))
        _test_split((32, 15), 1, 4)

    #######################################################################
    # Slice
    # ------------
    def test_Slice(self):
        """Slice"""

        class Model0(nn.Module):

            def __init__(self):
                super(Model0, self).__init__()

            def forward(self, x):
                y1 = x[:, 2::2]
                return y1

        class Model1(nn.Module):

            def __init__(self):
                super(Model1, self).__init__()
                self.weight = torch.randn((16, 32, 8))

            def forward(self, x):
                w = self.weight[:, 2:20:2]
                y = x + w
                return y

        self.trace_and_test([(16, 32, 8)], Model0())
        self.trace_and_test([(16, 9, 8)], Model1())

    #######################################################################
    # Squeeze
    # ------------
    def test_Squeeze(self):
        """Squeeze"""

        def _test_squeeze(in0_shape, dim=None):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    x = x + 1
                    if dim is None:
                        y1 = torch.squeeze(x)
                    else:
                        y1 = torch.squeeze(x, dim=dim)
                    return y1

            self.trace_and_test([in0_shape], Model())

        _test_squeeze((1, 3, 1, 32), 0)
        _test_squeeze((1, 3, 1, 16))
        _test_squeeze((32, 1))

    #######################################################################
    # Stack
    # ------------
    def test_Stack(self):
        """Stack"""

        def _test_stack(in0_shape, num, dim=None):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y1 = torch.stack([x] * num, dim)
                    return y1

            self.trace_and_test([in0_shape], Model())

        _test_stack((1, 3, 16, 32), 2, 0)
        _test_stack((2, 3, 16), 3, 1)
        _test_stack((32, 16), 2, 2)

    #######################################################################
    # Upsample
    # ------------
    def test_Upsample(self):
        """Upsample"""

        def _test_upsample(in0_shape, size=None, scale=None, mode='nearest'):

            # If given scale, torch implementation will not recompute scale by default.
            # But mlir and backend will do, so this param is needed.
            args = {}
            if size is None and not scale is None:
                args["recompute_scale_factor"] = True

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.layer = nn.Upsample(size=size, scale_factor=scale, mode=mode, **args)

                def forward(self, x):
                    y1 = self.layer(x)
                    return y1

            self.trace_and_test([in0_shape], Model())

        _test_upsample((1, 3, 34, 34), None, 2)
        _test_upsample((1, 3, 33, 33), None, 2.2)
        _test_upsample((1, 1, 32, 32), None, (2.5))
        _test_upsample((1, 3, 32, 32), None, (2.0, 2.2))
        _test_upsample((1, 3, 32, 32), 64, None)
        _test_upsample((1, 3, 32, 32), (81), None)
        _test_upsample((1, 3, 32, 32), (64, 65), None)
        if self.chip in ["bm1684x", "bm1688"]:
            _test_upsample((1, 3, 224), None, 2)
            _test_upsample((1, 3, 224), (500), None)
        if self.chip in ["bm1684x"]:
            _test_upsample((1, 3, 5, 2, 4), None, (1, 2, 3)) # add for upsample_nearest_3d, only support scale_d = 1

    #######################################################################
    def test_IndexPut(self):
        """IndexPut & Index"""
        def _test_index_put_(in_shape, index_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    high = in_shape[0]
                    self.index = torch.randint(0, high, index_shape)
                    val_shape = [index_shape[0]]
                    val_shape[1:] = in_shape[1:]
                    self.new_val = torch.randn(val_shape)

                def forward(self, x):
                    x[self.index] += self.new_val
                    return x

            self.trace_and_test([in_shape], Model())

        _test_index_put_((10, 3, 64, 64), (2,) )

    #######################################################################
    # IndexSelect
    # ------------
    def test_IndexSelect(self):
        """IndexSelect"""

        def _test_index_select(in_shape, dim, index_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    high = in_shape[dim]
                    self.index = torch.randint(0, high, index_shape)

                def forward(self, x):
                    y1 = torch.index_select(x, dim, self.index)
                    return y1

            self.trace_and_test([in_shape], Model())

        if self.is_cv18xx:
            _test_index_select((1, 3, 32, 32), 2, (1, ))
        else:
            _test_index_select((1, 3, 32, 32), 2, (5, ))
            _test_index_select((2, 32, 16), 0, (3, ))
            _test_index_select((32, 5), 1, (6, ))

    #######################################################################
    # Scatter
    # ------------
    def test_Scatter(self, case_name):
        """Scatter"""

        # in_shape.dims == index_shape.dims && in_shape[i] >=
        def _test_scatter(in_shape, dim, index_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    high = in_shape[dim]
                    self.index = torch.randint(0, high, index_shape)

                def forward(self, x, updates):
                    y1 = torch.scatter(x, dim, self.index, updates)
                    return y1

            self.trace_and_test([in_shape, index_shape], Model())

        _test_scatter((1, 3, 32, 32), 2, (1, 3, 24, 12))
        # _test_scatter((2, 32, 16), 1, (2, 12, 4))
        _test_scatter((32, 5), 0, (13, 5))

    #######################################################################
    # Concat
    # ------------
    def test_Concat(self):
        """Concat"""

        def _test_concat(in0_shape, in1_shape, dim=None):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.weight = torch.randn(in1_shape)

                def forward(self, x):
                    if dim is None:
                        y1 = torch.concat((x, self.weight))
                    else:
                        y1 = torch.concat((x, self.weight), dim=dim)
                    return y1

            self.trace_and_test([in0_shape], Model())

        _test_concat((1, 3, 32, 32), (1, 6, 32, 32), 1)
        _test_concat((2, 32, 16), (3, 32, 16))
        _test_concat((32, 32), (32, 16), -1)

    #######################################################################
    # Dropout
    # ------------
    def test_Dropout(self):
        """Dropout"""

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.dropout = nn.Dropout(0)

            def forward(self, x):
                a = x.view(64, -1, 1024)  # 64, 8, 1024
                b = a.transpose(0, 1)  # 8, 64, 1024
                c = self.dropout(b)
                d = c.transpose(1, 2)  # 8, 1024, 64
                return d

        in_shape = (512, 1024)
        self.trace_and_test([in_shape], Model())

    #######################################################################
    # Elu
    # ------------
    def test_Elu(self):
        """Elu"""

        def _test_elu(input_shape, alpha=1.0):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return F.elu(x)

            self.trace_and_test([input_shape], Model())

        _test_elu((1, 3, 32, 32), 2.0)
        _test_elu((3, 16, 32), 3.5)
        _test_elu((64, 32))

    #######################################################################
    # Floor
    # ------------
    def test_Floor(self):
        """Floor"""

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                o = torch.floor(x)
                return o

        self.trace_and_test([(4, 3, 32, 32)], Model())
        self.trace_and_test([(1, 65, 4, 4)], Model())

    #######################################################################
    # FloorDiv
    # ------------
    def test_FloorDiv(self):
        """FloorDiv"""

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y):
                o = torch.floor_divide(x, y)
                return o

        class Model2(nn.Module):

            def __init__(self):
                super(Model2, self).__init__()

            def forward(self, x):
                o = torch.floor_divide(x, 0.8)
                return o

        self.trace_and_test([(4, 3, 32, 32), (4, 3, 32, 32)], Model())
        self.trace_and_test([(4, 3, 32, 32)], Model2())

    #######################################################################
    # Frobenius Norm
    # ------------
    def test_FrobeniusNorm(self):
        """FloorDiv"""

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                y = torch.linalg.norm(x, ord='fro')
                z = x / y
                return z

        self.trace_and_test([(32,64)], Model(), [self.Desc('float32', -1.0, 1.0)])

    #######################################################################
    # Unary
    # ------------
    def test_Unary(self):
        """Unary Functions"""

        def _test_unary(op_type, in_shape, min=0):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return op_type(x)

            self.trace_and_test([in_shape], Model(), [self.Desc('float32', min)])

        for op_type in [torch.sqrt]:
            _test_unary(op_type, (1, 3, 32, 32))
            _test_unary(op_type, (3, 16, 32))
            _test_unary(op_type, (64, 32))

    #######################################################################
    # Activation
    # ------------
    def test_Activation(self):
        """Activation Functions Without Extra Arguments"""

        def _test_activation(op_type, in_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.activation = op_type()

                def forward(self, x):
                    return self.activation(x)

            self.trace_and_test([in_shape], Model())

        for op_type in [
                nn.GELU, nn.Hardsigmoid, nn.Hardswish, nn.LogSigmoid, nn.Mish, nn.ReLU, nn.ReLU6,
                nn.Sigmoid, nn.SiLU, nn.Softplus, nn.Softsign, nn.Tanh
        ]:
            _test_activation(op_type, (1, 3, 32, 32))
            # _test_activation(op_type, (3, 16, 32))
            # _test_activation(op_type, (64, 32))

    #######################################################################
    # LeakyRelu
    # ------------
    def test_LeakyRelu(self):
        """LeakyRelu"""

        def _test_leaky_relu(in_shape, alpha):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return F.leaky_relu(x, negative_slope=alpha)

            self.trace_and_test([in_shape], Model())

        for alpha in [0.67, -0.2]:
            _test_leaky_relu((1, 3, 32, 32), alpha)
            # _test_leaky_relu((3, 16, 32), alpha)
            # _test_leaky_relu((64, 32), alpha)

    #######################################################################
    # LSTM
    # ------------
    def test_LSTM(self):

        def _test_lstm0(batch, bidir):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.rnn = nn.LSTM(input_size=100, hidden_size=128, bidirectional=bidir)

                def forward(self, x, h_0, c_0):
                    Y, (Y_h, Y_c) = self.rnn(x, (h_0, c_0))
                    return Y, Y_h, Y_c

            num_dir = 2 if bidir else 1
            input_shapes = [(81, batch, 100), (num_dir, batch, 128), (num_dir, batch, 128)]
            self.trace_and_test(input_shapes, Model())

        def _test_lstm1(batch, bidir):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.rnn = nn.LSTM(input_size=100, hidden_size=128, bidirectional=bidir)

                def forward(self, x):
                    Y, _ = self.rnn(x)
                    return Y

            input_shapes = [(81, batch, 100)]
            self.trace_and_test(input_shapes, Model())

        batchs = [1] if self.simple else [1, 4]
        for bidir in [True, False]:
            for batch in batchs:
                _test_lstm0(batch, bidir)
                _test_lstm1(batch, bidir)

    #######################################################################
    # GRU
    # ------------
    def test_GRU(self):

        def _test_gru0(batch, bidir):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.rnn = nn.GRU(input_size=100, hidden_size=128, bidirectional=bidir)

                def forward(self, x, h_0):
                    Y, Y_h = self.rnn(x, h_0)
                    return Y, Y_h

            num_dir = 2 if bidir else 1
            input_shapes = [(81, batch, 100), (num_dir, batch, 128)]
            self.trace_and_test(input_shapes, Model())

        def _test_gru1(batch, bidir):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.rnn = nn.GRU(input_size=100, hidden_size=128, bidirectional=bidir)

                def forward(self, x):
                    Y, _ = self.rnn(x)
                    return Y

            input_shapes = [(81, batch, 100)]
            self.trace_and_test(input_shapes, Model())

        batchs = [1] if self.simple else [1, 4]
        for bidir in [True, False]:
            for batch in batchs:
                _test_gru0(batch, bidir)
                _test_gru1(batch, bidir)

    #######################################################################
    # Softmax
    # ------------
    def test_Softmax(self):
        """Softmax"""

        def _test_softmax(in_shape, dim):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return F.softmax(x, dim=dim)

            self.trace_and_test([in_shape], Model())

        _test_softmax((1, 5, 5538), 2) # test large axis dim
        for dim in [1, 2]:
            _test_softmax((3, 100, 10, 1), dim)
            # _test_softmax((3, 100, 32), dim)
            # _test_softmax((3, 100, 32, 1), dim)

    #######################################################################
    # Flatten
    # ------------
    def test_Flatten(self):
        """Flatten"""

        def _test_flatten(in_shape, start_dim=0, end_dim=-1):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.flatten = nn.Flatten(start_dim=start_dim, end_dim=end_dim)

                def forward(self, x):
                    return self.flatten(x)

            self.trace_and_test([in_shape], Model())

        _test_flatten((3, 16, 32, 64))
        _test_flatten((3, 16, 32, 64), end_dim=2)
        _test_flatten((3, 16, 32, 64), start_dim=1)

    #######################################################################
    # Flip
    # ------------
    def test_Flip(self):
        """Flip"""

        def _test_flip(in_shape, flip_dims):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    # self.flatten = nn.Flatten(start_dim=start_dim, end_dim=end_dim)

                def forward(self, x):
                    flipped_tensor = torch.flip(x, flip_dims)
                    return flipped_tensor

            self.trace_and_test([in_shape], Model())

        _test_flip((3, 16, 32, 64), flip_dims=[-1])
        # _test_flip((3, 16, 32, 64), flip_dims=[0, 1])
        # _test_flip((3, 16, 32, 64), flip_dims=[0, 2,3])

    #######################################################################
    # Adaptive AvgPool2d
    # ------------
    def test_AdaptiveAvgPool2d(self):
        """AdaptiveAvgPool2d"""

        def _test_adaptive_avgpool2d(in_shape, output_size):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return F.adaptive_avg_pool2d(x, output_size)

            self.trace_and_test([in_shape], Model())

        _test_adaptive_avgpool2d((3, 64, 15, 15), (1, 1))
        if self.chip != "cv183x" :
            _test_adaptive_avgpool2d((1, 32, 32, 32), (3, 3))

    #######################################################################
    # Adaptive AvgPool1d
    # ------------
    def test_AdaptiveAvgPool1d(self):
        """AdaptiveAvgPool1d"""

        def _test_adaptive_avgpool1d(in_shape, output_size):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return F.adaptive_avg_pool1d(x, output_size)

            self.trace_and_test([in_shape], Model())

        _test_adaptive_avgpool1d((3, 64, 32), (1))

    #######################################################################
    # Linear
    # ------------
    def test_Linear(self):

        def _test_linear(in_shape, has_bias):

            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.linear = nn.Linear(32, 72, bias=has_bias)

                def forward(self, x):
                    y = self.linear(x)
                    return y

            self.trace_and_test([in_shape], Model())

        _test_linear((3, 100, 32), True)
        _test_linear((64, 32), False)

    #######################################################################
    # LogSoftmax
    # ------------
    def test_LogSoftmax(self):
        """LogSoftmax"""

        def _test_log_softmax(in_shape, dim):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return F.log_softmax(x, dim=dim)

            self.trace_and_test([in_shape], Model())

        for dim in [1, 2]:
            _test_log_softmax((3, 100, 10, 1), dim)
            # _test_log_softmax((3, 100, 32), dim)
            # _test_log_softmax((3, 100, 32, 1), dim)

    #######################################################################
    # Softmin
    # ------------
    def test_Softmin(self):
        """Softmin"""

        def _test_softmin(in_shape, dim):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return F.softmin(x, dim=dim)

            self.trace_and_test([in_shape], Model())

        for dim in [1, 2]:
            _test_softmin((3, 100, 10, 1), dim)
            # _test_softmin((3, 100, 32), dim)
            # _test_softmin((3, 100, 32, 1), dim)

    #######################################################################
    # Pad1D
    # ------------
    def test_Pad1d(self):
        """Pad 1D (ReflectionPad1d, ReplicationPad1d, ConstantPad1d)
        Input: (N,C,W) or (C, W)
        Padding: int (pad_left == pad_right) or tuple (pad_left, pad_right)
      """

        def _test_pad1d(op_type, input_shape, padding: Union[int, tuple], value=0.):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    if op_type == nn.ConstantPad1d:
                        self.pad1d = op_type(padding, value)
                    else:
                        self.pad1d = op_type(padding)

                def forward(self, x):
                    return self.pad1d(x)

            self.trace_and_test([input_shape], Model())

        for op_type in [nn.ReflectionPad1d, nn.ReplicationPad1d, nn.ConstantPad1d]:
            _test_pad1d(op_type, (1, 3, 32), 5, 0.6)
            _test_pad1d(op_type, (2, 3, 32), (5, 3))
            if not self.simple:
                _test_pad1d(op_type, (64, 32), (3, 6), 0.6)
                _test_pad1d(op_type, (64, 32), 7, 0.4)

    #######################################################################
    # Pad2D
    # ------------
    def test_Pad2d(self):
        """Pad 2D (ReflectionPad2d, ReplicationPad2d, ConstantPad2d, ZeroPad2d)
        Input: (C, H, W) or (N, C, H, W)
        Padding: int (pad_left == pad_right == pad_top == pad_bottom)
                  or tuple (pad_left, pad_right, pad_top, pad_bottom)
      """

        def _test_pad2d(op_type, input_shape, padding: Union[int, tuple], value=0.):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    if op_type == nn.ConstantPad2d:
                        self.pad2d = op_type(padding, value)
                    else:
                        self.pad2d = op_type(padding)

                def forward(self, x):
                    return self.pad2d(x)

            self.trace_and_test([input_shape], Model())

        for op_type in [nn.ReflectionPad2d, nn.ReplicationPad2d, nn.ZeroPad2d, nn.ConstantPad2d]:
            if self.is_cv18xx and op_type == nn.ReplicationPad2d:
                _test_pad2d(op_type, (1, 16, 32, 32), 7, 0.6)
                _test_pad2d(op_type, (3, 16, 32, 32), (4, 6, 7, 8))
            else:
                _test_pad2d(op_type, (1, 16, 32), 7, 0.6)
                _test_pad2d(op_type, (3, 16, 32), (4, 6, 7, 8))
            if not self.simple:
                _test_pad2d(op_type, (1, 3, 16, 32), 3, 0.5)
                _test_pad2d(op_type, (2, 4, 16, 32), (3, 4, 5, 6), 0.4)

    #######################################################################
    # Abs
    # ------------
    def test_Abs(self):
        """Abs"""

        def _test_abs(input_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return torch.abs(x)

            self.trace_and_test([input_shape], Model())

        _test_abs((1, 16, 32, 32))

    #######################################################################
    # ConstantLike
    # ------------
    def test_ConstantLike(self):
        """ZerosLike and OnesLike"""

        def _test_constant_like(op_type, input_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = op_type(x)
                    return x + y + 0.5

            self.trace_and_test([input_shape], Model())

        for op_type in [torch.zeros_like, torch.ones_like]:
            _test_constant_like(op_type, (1, 16, 32, 32))

    #######################################################################
    # GridSampler
    # ------------
    def test_GridSampler(self):
        """GridSampler"""

        def _test_grid_sampler(in_shape, grid_shape, mode, padding_mode, align_corners):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    input_tensor = torch.randn(in_shape, dtype=torch.float32)
                    grid = torch.rand(grid_shape, dtype=torch.float32)


                def forward(self, input_tensor, grid):
                    output_tensor = torch.grid_sampler(input_tensor,
                                                       grid,
                                                       interpolation_mode=mode,
                                                       padding_mode=padding_mode,
                                                       align_corners=align_corners)
                    return output_tensor

            self.trace_and_test([in_shape, grid_shape], Model(), [
                                self.Desc('float32', -10, 10), self.Desc('float32', -1.02, 1.02)], use_cos=True)

        mode_list = [0, 1]
        padding_mode_list = [0, 1] if not self.simple else [0]
        align_corners_list = [False, True]

        for mode in mode_list:
            for padding_mode in padding_mode_list:
                for align_corners in align_corners_list:
                    _test_grid_sampler((1, 3, 100, 150), (1, 100, 150, 2), mode,
                                       padding_mode, align_corners)
                    if not self.simple:
                        _test_grid_sampler((1, 3, 100, 150), (1, 40, 80, 2), mode,
                                        padding_mode, align_corners)
                        _test_grid_sampler((1, 3, 50, 50), (1, 1, 1, 2), mode,
                                        padding_mode, align_corners)
        # # max shape in Grouding Dino
        # _test_grid_sampler((8, 32, 100, 100), (8, 13294, 4, 2), 0,
        #                     0, False)
        # # max shape for tpu grid_sample f16 (3 banks)
        # _test_grid_sampler((8, 32, 254, 94), (8, 13294, 4, 2), 0,
        #                     0, False)

        # case for DinoSwinL
        # _test_grid_sampler((8, 32, 320, 180), (8, 76760, 4, 2), 0,
        #                     0, False)
        # _test_grid_sampler((8, 32, 160, 90), (8, 76760, 4, 2), 0,
        #                     0, False)
        # _test_grid_sampler((8, 32, 80, 45), (8, 76760, 4, 2), 0,
        #                     0, False)
        # _test_grid_sampler((8, 32, 40, 23), (8, 76760, 4, 2), 0,
        #                     0, False)
        # _test_grid_sampler((8, 32, 20, 12), (8, 76760, 4, 2), 0,
        #                     0, False)
        # _test_grid_sampler((8, 32, 320, 180), (8, 900, 4, 2), 0,
        #                     0, False)
        # _test_grid_sampler((8, 32, 160, 90), (8, 900, 4, 2), 0,
        #                     0, False)
        # _test_grid_sampler((8, 32, 80, 45), (8, 900, 4, 2), 0,
        #                     0, False)
        # _test_grid_sampler((8, 32, 40, 23), (8, 900, 4, 2), 0,
        #                     0, False)
        # _test_grid_sampler((8, 32, 20, 12), (8, 900, 4, 2), 0,
        #                     0, False)

    #######################################################################
    # GridSampler3D
    # ------------
    def test_GridSampler3D(self):
        """GridSampler3D"""

        def _test_grid_sampler(in_shape, grid_shape, mode, padding_mode, align_corners):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    input_tensor = torch.randn(in_shape, dtype=torch.float32)
                    grid = torch.rand(grid_shape, dtype=torch.float32)


                def forward(self, input_tensor, grid):
                    output_tensor = torch.grid_sampler(input_tensor,
                                                       grid,
                                                       interpolation_mode=mode,
                                                       padding_mode=padding_mode,
                                                       align_corners=align_corners)
                    return output_tensor

            self.trace_and_test([in_shape, grid_shape], Model(), [
                                self.Desc('float32', -10, 10), self.Desc('float32', -1, 1)])

        mode_list = [0, 1]
        padding_mode_list = [0, 1]
        align_corners_list = [False, True]

        # for mode in mode_list:
        #     for padding_mode in padding_mode_list:
        #         for align_corners in align_corners_list:
        #             _test_grid_sampler((1, 3, 100, 150, 150), (1, 100, 150, 150, 3), mode,
        #                                padding_mode, align_corners)
        #             _test_grid_sampler((1, 3, 100, 150, 150), (1, 40, 80, 120, 3), mode,
        #                                padding_mode, align_corners)
        #             _test_grid_sampler((1, 3, 50, 50, 50), (1, 1, 1, 1, 3), mode,
        #                                padding_mode, align_corners)
        _test_grid_sampler((1, 15, 17, 17, 17), (1, 1, 1440, 2560, 3), 0, 1, True)

    #######################################################################
    # GridSampler3DPermute
    # ------------
    def test_GridSampler3DPermute(self):
        """GridSampler3DPermute"""

        def _test_grid_sampler(in_shape, grid_shape, mode, padding_mode, align_corners):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    input_tensor = torch.randn(in_shape, dtype=torch.float32)
                    grid = torch.rand(grid_shape, dtype=torch.float32)


                def forward(self, input_tensor, grid):
                    grid = grid.permute(0, 2, 3, 4, 1)
                    output_tensor = torch.grid_sampler(input_tensor,
                                                       grid,
                                                       interpolation_mode=mode,
                                                       padding_mode=padding_mode,
                                                       align_corners=align_corners)
                    return output_tensor

            self.trace_and_test([in_shape, grid_shape], Model(), [
                                self.Desc('float32', -10, 10), self.Desc('float32', -1, 1)])

        _test_grid_sampler((1, 15, 17, 17, 17), (1, 3, 1, 1440, 2560), 0, 1, True)

    #######################################################################
    # Deformable Convolution
    # ------------
    def test_DeformConv2D(self):
        in_channel = 3
        kernel = [3, 3]
        deform_groups = 1
        out_channel = 64

        class Model0(torch.nn.Module):

            def __init__(self):
                super(Model0, self).__init__()
                num_filter = kernel[0] * kernel[1] * 2 * deform_groups
                self.m0 = nn.Conv2d(in_channel, num_filter, kernel[0], 2, 0, 1)
                self.m1 = DeformConv2d(in_channel, out_channel, kernel[0], 2, 0, 1)

            def forward(self, x):
                y0 = self.m0(x)
                y1 = self.m1(x, y0)
                return y1

        class Model1(torch.nn.Module):

            def __init__(self):
                super(Model1, self).__init__()
                num_filter = kernel[0] * kernel[1] * 2 * deform_groups
                self.m0 = nn.Conv2d(in_channel, num_filter, kernel[0], 2, 0, 1)
                self.m1 = nn.Conv2d(in_channel, num_filter // 2, kernel[0], 2, 0, 1)
                self.m2 = DeformConv2d(in_channel, out_channel, kernel[0], 2, 0, 1)

            def forward(self, x):
                y0 = self.m0(x)
                y1 = self.m1(x)
                y2 = self.m2(x, y0, y1)
                return y2

        self.trace_and_test([(1, 3, 28, 28)], Model0())
        self.trace_and_test([(1, 3, 28, 28)], Model1())

    #######################################################################
    # Ceil
    # ------------
    def test_Ceil(self):
        """Ceil"""

        def _test_ceil(input_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    return torch.ceil(x)

            self.trace_and_test([input_shape], Model())

        _test_ceil((1, 16, 32, 32))

    #######################################################################
    # Remainder
    # ------------
    def test_Remainder(self):
        """Remainder"""

        def _test_remainder(op_type,in0_shape,in1_shape):

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()
                    self.weight = torch.randn(in1_shape)

                def forward(self, x):
                    y2 = op_type(x, self.weight)
                    return y2

            self.trace_and_test([in0_shape], Model(), [self.Desc('float32')])

        _test_remainder(torch.remainder, (1, 16, 32, 32), (1, 16, 32, 32))

    #######################################################################
    # Sign
    # ------------
    def test_Sign(self):
        """Sign"""

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                o = torch.sign(x)
                return o

        self.trace_and_test([(4, 3, 32, 32)], Model())
        self.trace_and_test([(1, 65, 4, 4)], Model())

    #######################################################################
    # RoiAlign
    # ------------
    def test_RoiAlign(self):
        roi_num = 5
        N, C, H, W = 3, 3, 64, 64

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.boxes = self.gen_rand_rois(N, H, W, roi_num)

            def gen_rand_rois(self, N, H, W, roi_num) -> torch.Tensor:
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

            def forward(self, x):
                y = roi_align(x, self.boxes, [8, 8])
                return y

        self.trace_and_test([(N, C, H, W)], Model())

    #######################################################################
    # Unbind
    # ------------
    def test_Unbind(self):
        def _test_unbind(shape, dim):
            class Model(torch.nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, x):
                    y = torch.unbind(x, dim=dim)
                    return y
            self.trace_and_test(shape, Model())

        _test_unbind([(10, 10, 10, 10)], dim=0)
        _test_unbind([(10, 10, 10, 10)], dim=1)
        _test_unbind([(10, 10, 10, 10)], dim=2)
        _test_unbind([(10, 10, 10, 10)], dim=3)

    #######################################################################
    # MovePermuteAfterAdd
    # ------------
    def test_MovePermuteAfterAdd(self):
        class Model0(torch.nn.Module):

            def __init__(self):
                super(Model0, self).__init__()

            def forward(self, qk, mask, v):
                v1 = v.permute(0, 2, 1, 3)
                qk1 = qk.permute(0, 2, 1, 3)
                qk2 = qk1 + mask.permute(0, 2, 1, 3)
                w = torch.nn.functional.softmax(qk2, dim=-1)
                wv1 = (w @ v1).permute(0, 2, 1, 3)
                v1 = v.permute(0, 2, 1, 3)
                qk1 = qk.permute(0, 2, 1, 3)
                qk2 = qk1 + mask.permute(0, 2, 1, 3)
                w = torch.nn.functional.softmax(qk2, dim=-1)
                wv2 = (w @ v1).permute(0, 2, 1, 3)
                return wv1, wv2

        self.trace_and_test([[5,16,8,16],[5,16,8,16],[5,16,8,64]], Model0())
        # Permute will be converted to Reshape when there is 1 in shape
        # MovePermuteAfterAdd will be affected by PermuteToReshape Pattern
        # self.trace_and_test([[5,1,8,16],[5,1,8,16],[5,16,8,64]], Model0())

    def user_define_net(self):
        """user_define_net"""
        # return
        self.group_opt = 3
        batch_size = self.num_core

        print('start test test_model1')
        from tools.train.test_model import test_model1
        model = test_model1()
        self.trace_and_test([(batch_size,3,224,224)], model)

        print('start test test_model2')
        from tools.train.test_model import test_model2
        model = test_model2()
        self.trace_and_test([(batch_size,3,224,224)], model)

        print('start test test_model3')
        from tools.train.test_model import test_model3
        model = test_model3()
        self.trace_and_test([(batch_size,3,224,224)], model)

        print('start test test_model4')
        from tools.train.test_model import test_model4
        model = test_model4()
        self.trace_and_test([(batch_size,3,224,224)], model)

        print('start test resnet18')
        from tools.train.resnet import resnet18
        model = resnet18()
        self.trace_and_test([(batch_size,3,224,224)], model)

        print('start test mobilenet_v2')
        import torchvision.models as models
        model = models.mobilenet_v2()
        self.trace_and_test([(batch_size,3,224,224)], model)

        print('start test resnet50')
        from tools.train.resnet import resnet50
        model = resnet50()
        self.trace_and_test([(batch_size,3,224,224)], model)

        print('start test yolov5s')
        from nets.yolo import YoloBody
        anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        num_classes = 80
        phi = 's'
        model = YoloBody(anchors_mask, num_classes, phi)
        self.trace_and_test([(batch_size,3,224,224)], model)

        # print('start test test_model6')
        # from tools.train.test_model import test_model6
        # model = test_model6()
        # self.trace_and_test([(1,3,224,224)], model)

        # print('start test test_model5') #error
        # from tools.train.test_model import test_model5
        # model = test_model5()
        # self.trace_and_test([(1,3,224,224)], model)


        # print('start test test_model7')
        # from tools.train.test_model import test_model7
        # model = test_model7()
        # self.trace_and_test([(1,3,224,224), (1,3,224,224)], model)

        # d_model=10 #768  #test ok at 0918
        # from tools.train.gpt2 import TransformerBlocks #backward can not use this
        # mod = TransformerBlocks(d_model=d_model, nlayers=2) #.train()
        # self.trace_and_test([(1,4,d_model)], mod)

def test_one_case_in_all(tester: TORCH_IR_TESTER, case, error_cases, success_cases):
    t = Timer()
    try:
        tester.test_single(case)
    except:
        error_cases.append("{}:{}s".format(case, int(t.elapsed_time())))
        traceback.print_exc()
        return
    success_cases.append("{}:{}s".format(case, int(t.elapsed_time())))


def test_all(tester: TORCH_IR_TESTER):
    if tester.multithread:
        import multiprocessing
        from utils.misc import collect_process
        process_number = multiprocessing.cpu_count() // 2 + 1
        processes = []
        error_cases = multiprocessing.Manager().list()
        success_cases = multiprocessing.Manager().list()
        for case in tester.test_cases:
            if tester.check_support(case):
                p = multiprocessing.Process(target=test_one_case_in_all, name = case,
                                            args=(tester, case, error_cases, success_cases))
                processes.append(p)
            if len(processes) == process_number:
                collect_process(processes, error_cases)
                processes = []
        collect_process(processes, error_cases)
        processes = []
    else:
        error_cases = []
        success_cases = []
        for case in tester.test_cases:
            if tester.check_support(case):
                test_one_case_in_all(tester, case, error_cases, success_cases)
    print("Success: {}".format(success_cases))
    print("Failure: {}".format(error_cases))
    if error_cases:
        print("====== test_torch.py --chip {} TEST Failed ======".format(tester.chip))
        # exit(1)
    else:
        print("====== test_torch.py --chip {} TEST Success ======".format(tester.chip))
    clean_kmp_files()
    return error_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--chip", default="bm1684x", type=str,
                        choices=['bm1684', 'bm1684x', 'bm1688', 'cv183x', 'mars3', 'cv186x', 'bm1690'], help="chip platform name")
    parser.add_argument("--case", default="all", type=str, help="test one case, if all, then test all cases")
    parser.add_argument("--mode", default="all", type=str, choices=['all', 'f32', 'f16', 'bf16', 'int8'],
                        help="chip platform name")
    parser.add_argument("--num_core", default=1, type=int, help='The numer of TPU cores used for parallel computation')
    parser.add_argument("--debug", action="store_true", help='keep middle file if debug')
    parser.add_argument("--simple", action="store_true", help='do simple test for commit test')
    parser.add_argument("--disable_thread", action="store_true", help='do test without multi thread')
    parser.add_argument("--show_all", action="store_true", help='show all cases')
    parser.add_argument("--quant_input", action="store_true", help='quant input')
    parser.add_argument("--quant_output", action="store_true", help='quant output')
    parser.add_argument("--dynamic", action="store_true", help='dynamic mode in model deployment')
    parser.add_argument("--debug_cmd", default="", type=str, help="debug_cmd")
    parser.add_argument("--cuda", action="store_true", help="test cuda inference")
    # yapf: enable
    args = parser.parse_args()
    tester = TORCH_IR_TESTER(args.chip, args.mode, args.simple, args.disable_thread,
                             args.quant_input, args.quant_output, args.debug, args.num_core,
                             args.debug_cmd, args.cuda, args.dynamic)
    if args.show_all:
        print("====== Show All Cases ============")
        for case in tester.test_cases:
            print(case)
        exit(0)
    dir = "torch_test_{}".format(args.chip)
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)
    if args.case == "" or args.case.lower() == "all":
        test_all(tester)
    else:
        tester.test_single(args.case)
    if args.debug == False:
        file_clean()
