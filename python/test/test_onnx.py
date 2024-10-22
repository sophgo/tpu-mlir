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
from utils.auto_remove import file_mark, file_clean, clean_kmp_files
from utils.mlir_shell import *
from utils.timer import Timer
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
                 disable_thread: bool = False,
                 num_core: int = 1,
                 debug_cmd: str = '',
                 cuda: bool = False):
        Y, N = True, False
        # yapf: disable
        self.test_cases = {
            #########################################
            # ONNX Test Case, Alphabetically
            #########################################
            # case: (test, bm1684_support, bm1684x_support, bm1688_support, cv183x_support, bm1690_support,mars3_support)
            "Abs":          (self.test_Abs,           Y, Y, Y, Y, Y, Y),
            "Add":          (self.test_Add,           Y, Y, Y, Y, Y, Y),
            "Add_5dim_bc":  (self.test_Add_5dim_bc,   N, Y, Y, N, N, Y),
            "And":          (self.test_And,           N, Y, Y, N, Y, N),
            "AddBcast":     (self.test_AddBcast,      N, N, N, N, Y, N), # bm1684x has random error caused by 2.27 commit
            "AddBcast2":    (self.test_AddBcast2,     Y, Y, Y, N, Y, Y),
            "AddBcast3":    (self.test_AddBcast3,     N, N, N, N, N, N), # failed cases
            "Acos":         (self.test_Arccos,        Y, Y, Y, N, Y, N),
            "Atan":         (self.test_Arctan,        N, Y, Y, N, Y, N),
            "Atanh":        (self.test_Arctanh,       Y, Y, Y, N, Y, N),
            "Arg":          (self.test_Arg,           Y, Y, Y, Y, Y, N),
            "AddWeight":    (self.test_AddWeight,     Y, Y, Y, Y, Y, Y),
            "AddWeight2":   (self.test_AddWeight2,    Y, Y, Y, Y, Y, Y),
            "AvgPool1d":    (self.test_AvgPool1d,     Y, Y, Y, Y, Y, Y),
            "AvgPool2d":    (self.test_AvgPool2d,     Y, Y, Y, Y, Y, Y),
            "AvgPool3d":    (self.test_AvgPool3d,     N, Y, Y, Y, Y, N),
            "AvgPoolOdd":   (self.test_AvgPoolOdd,    Y, Y, Y, Y, Y, Y),
            "PadAvgPool2d": (self.test_PadAvgPool2d,  Y, Y, Y, Y, Y, Y),
            "BatchMatMul":  (self.test_BatchMatMul,   Y, Y, Y, Y, Y, Y),
            "BCastAdd":     (self.test_BCastAdd,      Y, Y, Y, Y, Y, Y),
            "BCastMul":     (self.test_BCastMul,      Y, Y, Y, Y, Y, Y),
            "BCastMulCst":  (self.test_BCastMulCst,   Y, Y, Y, Y, Y, Y),
            "Cast":         (self.test_Cast,          Y, Y, Y, Y, Y, N),
            "CompareCst":   (self.test_CompareCst,    Y, Y, Y, Y, Y, Y),
            "Compare":      (self.test_Compare,       Y, Y, Y, N, Y, Y),
            "Compare2":     (self.test_Compare2,      Y, N, N, N, N, N),
            "Concat":       (self.test_Concat,        Y, Y, Y, Y, Y, Y),
            "Concat2":      (self.test_Concat2,       Y, Y, Y, Y, Y, N),
            "Concat3":      (self.test_Concat3,       N, Y, Y, N, Y, Y),
            "ConstOfShape": (self.test_ConstOfShape,  N, Y, Y, N, Y, Y),
            "ConstantFillDyn": (self.test_ConstantFillDyn, N, Y, Y, N, Y, Y),
            "Conv1d":       (self.test_Conv1d,        Y, Y, Y, Y, Y, Y),
            "Conv1dbigd":   (self.test_Conv1d_bigd,   Y, N, N, N, N, N),
            "Conv2d":       (self.test_Conv2d,        Y, Y, Y, Y, Y, Y),
            "Conv2dbigd":   (self.test_Conv2d_bigd,   Y, N, N, N, N, N),
            "Conv3d":       (self.test_Conv3d,        N, Y, Y, Y, Y, Y),
            "ConvPool3d":   (self.test_ConvPool3d,    Y, Y, Y, N, Y, Y),
            "ConvStride":   (self.test_ConvStride,    Y, Y, Y, Y, Y, Y),
            "ConvDw":       (self.test_ConvDw,        Y, Y, Y, Y, Y, Y),
            "ConvTrans":    (self.test_ConvTrans,     N, Y, Y, Y, Y, N),
            "ConvTrans2":   (self.test_ConvTrans2,    N, Y, Y, Y, Y, N), # no pad
            "Clip":         (self.test_Clip,          Y, Y, Y, Y, Y, N),
            "CumSum":       (self.test_CumSum,        N, Y, N, N, Y, Y),
            "DepthToSpace":  (self.test_DepthToSpace,  Y, Y, Y, Y, Y, N),
            "Deconv":       (self.test_Deconv,        Y, Y, Y, Y, Y, N),
            "DeconvDF":     (self.test_DeconvDynW,    N, Y, N, N, Y, N),
            "Deconv2":      (self.test_Deconv2,       Y, N, N, Y, N, N),
            "Deconv3d":     (self.test_Deconv3d,      Y, N, N, N, N, N),
            "Div":          (self.test_Div,           Y, Y, Y, Y, Y, Y),
            "DivBcast":     (self.test_DivBcast,      Y, Y, Y, N, Y, Y),
            "DivBcast2":    (self.test_DivBcast2,     Y, Y, Y, N, Y, Y),
            "Einsum":       (self.test_Einsum,        Y, Y, Y, Y, Y, Y),
            "Einsum2":      (self.test_Einsum2,       Y, Y, Y, Y, Y, Y),
            "Einsum3":      (self.test_Einsum3,       Y, Y, Y, Y, Y, Y),
            "Einsum4":      (self.test_Einsum4,       N, Y, Y, N, Y, Y),
            "Einsum5":      (self.test_Einsum5,       N, Y, Y, N, Y, Y),
            "Einsum6":      (self.test_Einsum6,       N, Y, Y, N, Y, Y),
            "Einsum7":      (self.test_Einsum7,       N, Y, Y, N, Y, Y),
            "Einsum8":      (self.test_Einsum8,       N, Y, Y, N, Y, Y),
            "Einsum9":      (self.test_Einsum9,       N, Y, Y, N, Y, Y),
            "Einsum10":     (self.test_Einsum10,      N, Y, Y, N, Y, Y),
            "Einsum11":     (self.test_Einsum11,      N, Y, Y, N, Y, N),
            "Einsum12":     (self.test_Einsum12,      N, Y, Y, N, Y, Y),
            "Einsum13":     (self.test_Einsum13,      N, Y, N, N, Y, N),
            "Einsum14":     (self.test_Einsum14,      N, Y, Y, N, Y, N),
            "Elu":          (self.test_Elu,           Y, Y, Y, N, Y, N),
            "Erf":          (self.test_Erf,           N, Y, Y, N, Y, N),
            "Exp":          (self.test_Exp,           Y, Y, Y, Y, Y, Y),
            "Expand":       (self.test_Expand,        Y, Y, Y, Y, Y, Y),
            "Expand2":      (self.test_Expand2,       Y, Y, Y, Y, Y, Y),
            "ExpandDyn":    (self.test_ExpandDyn,     N, Y, Y, N, Y, N),
            "Flatten":      (self.test_Flatten,       Y, Y, Y, Y, Y, Y),
            "Flip":         (self.test_Flip,          Y, Y, Y, N, Y, N),
            "Floor":        (self.test_Floor,         Y, Y, Y, N, Y, N),
            "FitPermute2Hdim": (self.test_FitPermute2Hdim, N, N, Y, Y, N, N),
            "Gather":       (self.test_Gather,        Y, Y, Y, Y, Y, N),
            "GatherElements": (self.test_GatherElements, Y, Y, N, N, Y, N),
            "GatherND":     (self.test_GatherND,      Y, Y, Y, Y, Y, N),
            "Gather2":      (self.test_Gather2,       N, Y, Y, N, Y, N),
            "Gather3":      (self.test_Gather3,       Y, Y, Y, N, Y, N),
            "Gemm":         (self.test_Gemm,          Y, Y, Y, Y, Y, Y),
            "GlobalAveragePool": (self.test_GlobalAveragePool, Y, Y, Y, Y, Y, Y),
            "GlobalMaxPool": (self.test_GlobalMaxPool, Y, Y, Y, Y, Y, Y),
            "GroupFC":      (self.test_GroupFC,       Y, Y, Y, Y, Y, N), # test gru output Y
            "GRU":          (self.test_GRU,           N, Y, Y, Y, Y, N), # test gru output Yh
            "GRU2":         (self.test_GRU2,          N, Y, Y, Y, Y, N), # test gru output Y and Yh
            "GRU3":         (self.test_GRU3,          N, Y, Y, Y, Y, N),
            "GRU4":         (self.test_GRU4,          N, Y, Y, N, Y, N), # test gru output Y and Yh,also splitting batch dim
            "LeakyRelu":    (self.test_LeakyRelu,     Y, Y, Y, Y, Y, N),
            "Log":          (self.test_Log,           Y, Y, Y, Y, Y, N),
            "LogSoftmax":   (self.test_LogSoftmax,    Y, Y, Y, Y, Y, N),
            "LRN":          (self.test_LRN,           Y, Y, Y, Y, Y, N), # output_y
            "LSTM":         (self.test_LSTM,          N, Y, Y, Y, Y, N), # output all
            "LSTM2":        (self.test_LSTM2,         N, Y, Y, Y, Y, N), # output_yh and output_yc
            "LSTM3":        (self.test_LSTM3,         Y, Y, Y, Y, Y, N),
            "MaxPool1d":    (self.test_MaxPool1d,     Y, Y, Y, Y, Y, Y),
            "MaxPool2d":    (self.test_MaxPool2d,     Y, Y, Y, Y, Y, Y),
            "MaxPool3d":    (self.test_MaxPool3d,     N, Y, Y, Y, Y, Y),
            "MatMul":       (self.test_MatMul,        Y, Y, Y, Y, Y, Y),
            "MatMul2":      (self.test_MatMul2,       Y, Y, Y, Y, Y, Y),
            "MatMul2PC":    (self.test_MatMul2PC,     N, Y, Y, N, N, Y),
            "Max":          (self.test_Max,           Y, Y, Y, Y, Y, Y),
            "MaxBcast":     (self.test_MaxBcast,      Y, Y, Y, N, Y, Y),
            "Not":          (self.test_Not,           N, Y, Y, N, Y, Y),
            "Mod":          (self.test_Mod,           N, Y, Y, N, N, N),
            "Mul":          (self.test_Mul,           Y, Y, Y, Y, Y, Y),
            "MulMerge":     (self.test_MulMerge,      Y, Y, Y, N, Y, Y),
            "MulBcast":     (self.test_MulBcast,      Y, Y, Y, N, Y, Y),
            "MulBcast2":    (self.test_MulBcast2,     Y, Y, Y, N, Y, Y),
            "Min":          (self.test_Min,           Y, Y, Y, Y, Y, Y),
            "MinBcast":     (self.test_MinBcast,      Y, Y, Y, N, Y, Y),
            "MulConst":     (self.test_MulConst,      Y, Y, Y, Y, Y, Y),
            "Neg":          (self.test_Neg,           Y, Y, Y, Y, Y, Y),
            "Pad":          (self.test_Pad,           Y, Y, Y, Y, Y, Y), # zero pad
            "Pad1":         (self.test_Pad1,          Y, Y, Y, Y, Y, Y), # pad val
            "PadEdge":      (self.test_PadEdge,       N, Y, Y, Y, Y, Y),
            "PadReflect":   (self.test_PadReflect,    Y, Y, Y, Y, Y, Y),
            "Pow1":         (self.test_Pow1,          Y, Y, Y, Y, Y, N), # y = x ^ n
            "Pow2":         (self.test_Pow2,          Y, Y, Y, N, Y, N), # y = n ^ x
            "Pow3":         (self.test_Pow3,          Y, Y, Y, N, Y, N), # y = x1 ^ x2
            "PRelu":        (self.test_PRelu,         Y, Y, Y, Y, Y, Y),
            "Range":        (self.test_Range,         N, Y, Y, N, Y, N),
            "Resize":       (self.test_Resize,        Y, Y, Y, Y, Y, Y),
            "Resize2":      (self.test_Resize2,       N, Y, Y, Y, Y, N),
            "Reshape":      (self.test_Reshape,       Y, Y, Y, N, Y, Y),
            "Reduce":       (self.test_Reduce,        Y, Y, Y, Y, Y, Y),
            "Reduce2":      (self.test_Reduce2,       Y, Y, Y, Y, Y, Y),
            "ReduceL1":     (self.test_ReduceL1,      Y, Y, Y, N, Y, Y),
            "ReduceL2":     (self.test_ReduceL2,      Y, Y, Y, Y, Y, Y),
            "ReduceMean":   (self.test_ReduceMean,    Y, Y, Y, Y, Y, Y),
            "ReduceSum":    (self.test_ReduceSum,     Y, Y, Y, Y, Y, Y),
            "ReduceProd":   (self.test_ReduceProd,    Y, Y, Y, N, Y, Y),
            "Reciprocal":   (self.test_Reciprocal,    Y, Y, Y, Y, Y, N),
            "Relu":         (self.test_Relu,          Y, Y, Y, Y, Y, Y),
            "ReluOnly":     (self.test_ReluOnly,      Y, N, Y, N, N, Y),
            "Round":        (self.test_Round,         N, Y, N, N, Y, Y),
            "ScatterElements": (self.test_ScatterElements, N, Y, N, N, Y, N),
            "ScatterND":    (self.test_ScatterND,     N, Y, Y, N, Y, N),
            "Shape":        (self.test_Shape,         Y, Y, Y, N, Y, Y),
            "ShapeCast":    (self.test_ShapeCast,     N, N, N, N, N, N),
            "ShapeSlice":   (self.test_ShapeSlice,    Y, N, N, N, N, Y),
            "SiLU":         (self.test_SiLU,          Y, Y, Y, Y, Y, Y),
            "Softmax":      (self.test_Softmax,       Y, Y, Y, Y, Y, N),
            "Softplus":     (self.test_Softplus,      Y, Y, Y, Y, Y, N),
            "Space2Depth":  (self.test_Space2Depth,   Y, Y, Y, N, Y, N),
            "Squeeze":      (self.test_Squeeze,       Y, Y, Y, Y, Y, Y),
            "Sigmoid":      (self.test_Sigmoid,       Y, Y, Y, Y, Y, Y),
            "Sign":         (self.test_Sign,          N, Y, Y, N, Y, Y),
            "Slice":        (self.test_Slice,         Y, Y, Y, Y, Y, Y),
            "Slice2":       (self.test_Slice2,        Y, Y, Y, Y, Y, Y),
            "Slice3":       (self.test_Slice3,        Y, Y, Y, Y, Y, Y),
            "Dynamic_Slice": (self.test_Dynamic_Slice, N, Y, Y, N, Y, Y),
            "Split":        (self.test_Split,         Y, Y, Y, Y, Y, Y),
            "Split2":        (self.test_Split2,       Y, Y, Y, Y, Y, Y),
            "Scale":        (self.test_Scale,         Y, Y, Y, Y, Y, Y),
            "Sqrt":         (self.test_Sqrt,          Y, Y, Y, Y, Y, Y),
            "Sub":          (self.test_Sub,           Y, Y, Y, Y, Y, Y),
            "Sub2":         (self.test_Sub2,          Y, Y, Y, Y, Y, Y),
            "SubBcast":     (self.test_SubBcast,      Y, Y, Y, N, Y, Y),
            "SubBcast2":    (self.test_SubBcast2,     Y, Y, Y, N, Y, Y),
            "SubConst":     (self.test_SubConst,      Y, Y, Y, Y, Y, Y),
            "SubConst2":    (self.test_SubConst2,     Y, Y, Y, Y, Y, Y),
            "Sum":          (self.test_Sum,           Y, Y, Y, Y, Y, Y),
            "Tanh":         (self.test_Tanh,          Y, Y, Y, Y, Y, N),
            "Tile":         (self.test_Tile,          Y, Y, Y, Y, Y, Y),
            "TileDyn":      (self.test_TileDyn,       N, Y, Y, N, Y, N),
            "Transpose":    (self.test_Transpose,     Y, Y, Y, Y, Y, Y),
            "Transpose2":   (self.test_Transpose2,    Y, Y, Y, Y, Y, Y),
            "Trilu":        (self.test_Trilu,         N, Y, N, N, N, N),
            "TopK":         (self.test_TopK,          N, Y, Y, N, Y, N),
            "TopK2":        (self.test_TopK2,         N, Y, N, N, Y, N),
            "TopK4":        (self.test_TopK4,         N, N, N, Y, N, N),
            "Upsample":     (self.test_Upsample,      Y, Y, Y, N, Y, Y),
            "Unsqueeze":    (self.test_Unsqueeze,     Y, Y, Y, N, Y, Y),
            # Only 1D shape is supported currently
            # "ShapeUnsqueeze":  (self.test_ShapeUnsqueeze,  N, Y, Y, N),
            # "ShapeSqueeze":    (self.test_ShapeSqueeze,    N, Y, Y, N),
            "Where":        (self.test_Where,         N, Y, Y, Y, Y,Y),
            "Xor":          (self.test_Xor,           N, Y, Y, N, N,Y),
            #####################################
            # Torch Test Case, Alphabetically
            #####################################
            # case: (test, bm1684_support, bm1684x_support, bm1688_support, cv183x_support,mars3_support)
            "TorchActivation":      (self.test_TorchActivation,     N, Y, Y, Y, Y, N),
            "TorchArg":             (self.test_TorchArg,            N, Y, Y, N, Y, N),
            "TorchBatchNorm":       (self.test_TorchBatchNorm,      Y, Y, Y, Y, Y, Y),
            "TorchChannelShuffle":  (self.test_TorchChannelShuffle, N, N, N, N, N, N),
            "TorchChunk":           (self.test_TorchChunk,          N, Y, Y, Y, Y, Y),
            "TorchConv2d":          (self.test_TorchConv2d,         N, N, N, Y, N, N),
            "TorchConv3dTrans":     (self.test_TorchConv3dTrans,    N, Y, Y, Y, Y, Y),
            "TorchHardSwish":       (self.test_TorchHardSwish,      Y, Y, Y, Y, Y, Y),
            "TorchHardSigmoid":     (self.test_TorchHardSigmoid,    Y, Y, Y, Y, Y, Y),
            "TorchGelu":            (self.test_TorchGelu,           Y, Y, Y, Y, Y, Y),
            "TorchGroupNorm":       (self.test_TorchGroupNorm,      Y, Y, Y, N, Y, N),
            "TorchGroupNorm2":      (self.test_TorchGroupNorm2,     Y, Y, Y, N, Y, N),
            "TorchGRU":             (self.test_TorchGRU,            N, Y, Y, Y, Y, N),
            "TorchIdentity":        (self.test_TorchIdentity,       Y, Y, Y, Y, Y, Y),
            "TorchIndexCopy":       (self.test_TorchIndexCopy,      N, N, N, N, N, N),
            "TorchInstanceNorm":    (self.test_TorchInstanceNorm,   N, Y, Y, N, Y, N),
            "TorchInstanceNorm2":   (self.test_TorchInstanceNorm2,  N, Y, Y, N, Y, N),
            "TorchLayerGroup":      (self.test_TorchLayerGroup,     Y, Y, Y, Y, Y, Y),
            "TorchLayerNorm":       (self.test_TorchLayerNorm,      Y, Y, Y, Y, Y, Y),
            "TorchLayerNorm2":      (self.test_TorchLayerNorm2,     Y, Y, Y, Y, Y, N),
            "TorchLogSoftmax":      (self.test_TorchLogSoftmax,     Y, Y, Y, Y, Y, N),
            "TorchLSTM":            (self.test_TorchLSTM,           Y, Y, Y, Y, Y, N),
            "TorchMaskedFill":      (self.test_TorchMaskedFill,     N, Y, Y, N, Y, Y),
            "TorchNonZero":         (self.test_TorchNonZero,        N, Y, Y, N, Y, N),
            "TorchNormalize":       (self.test_TorchNormalize,      N, Y, Y, N, N, N),
            "TorchReflectionPad":   (self.test_TorchReflectionPad,  N, Y, Y, Y, Y, Y),
            "TorchRMSNorm":         (self.test_TorchRMSNorm,        N, Y, Y, N, Y, N),
            "TorchRoiAlign":        (self.test_TorchRoiAlign,       N, Y, Y, N, Y, N),
            "TorchScatterND":       (self.test_TorchScatterND,      N, Y, Y, Y, Y, N),
            "TorchSize":            (self.test_TorchSize,           Y, Y, Y, Y, Y, Y),
            "TorchStd":             (self.test_TorchStd,            N, Y, Y, Y, Y, N),
            "TorchWhere":           (self.test_TorchWhere,          N, Y, Y, N, Y, Y),
            "TorchZeroPad":         (self.test_TorchZeroPad,        N, Y, Y, Y, Y, Y),
            #########################################
            # Special Pass test case, Alphabetically
            #########################################
            # case: (test, bm1684_support, bm1684x_support, bm1688_support, cv183x_support,mars3_support)
            "ArgError":         (self.test_ArgError,        N, Y, Y, Y, Y, Y),
            "ArgReducefull":    (self.test_ArgReducefull,   Y, Y, Y, N, Y, N),
            "ConcatFuse":       (self.test_ConcatFuse,      Y, Y, Y, Y, Y, Y),
            "ConcatToSpace":    (self.test_ConcatToSpace,   N, Y, Y, N, Y, Y),
            "Conv3dTo2d":       (self.test_Conv3dTo2d,      N, Y, Y, Y, Y, Y),
            "Depth2SpaceWithPermute": (self.test_Depth2SpaceWithPermute, Y, Y, Y, N, Y, Y),
            "Div2Mul":          (self.test_Div2Mul,         Y, Y, Y, Y, Y, Y),
            "ConvSlice":        (self.test_ConvSlice,       Y, Y, Y, N, Y, Y),
            "Gather2Slice":     (self.test_Gather2Slice,    Y, Y, Y, Y, Y, Y),
            "Gather2Slice2":    (self.test_Gather2Slice2,   N, Y, Y, Y, Y, Y),
            "GatherUnsqueeze":  (self.test_GatherUnsqueeze,  Y, Y, Y, Y, Y, Y),
            "GLMTilePermute":   (self.test_GLMTilePermute,  N, Y, N, N, Y, Y),
            "GLMTilePermute2":  (self.test_GLMTilePermute2, N, Y, N, N, Y, N),
            "Mul2Scale":        (self.test_Mul2Scale,       Y, Y, Y, Y, Y, Y),
            "MatMulTranspose":  (self.test_MatMulTranspose, N, Y, Y, Y, Y, Y),
            "MatMulTranspose2": (self.test_MatMulTranspose2, N, Y, Y, Y, Y, Y),
            "MatMulTranspose3": (self.test_MatMulTranspose3, N, Y, Y, Y, Y, Y),
            "PadConv1d":        (self.test_PadConv1d,       N, N, N, Y, N, N),
            "PadConv2d":        (self.test_PadConv2d,       Y, Y, Y, Y, Y, Y),
            "PadConv3d":        (self.test_PadConv3d,       N, N, N, N, N, Y),
            "PadPool1d":        (self.test_PadPool1d,       N, Y, Y, Y, Y, Y),
            "PadPool2d":        (self.test_PadPool2d,       N, N, N, Y, N, Y),
            "PadPool3d":        (self.test_PadPool3d,       N, N, N, Y, N, N),
            "PixelNorm":        (self.test_PixelNorm,       N, Y, Y, N, Y, Y),
            "PixelNorm2":       (self.test_PixelNorm2,      N, Y, Y, N, Y, Y),
            "PenaltySample":    (self.test_PenaltySample,   N, N, Y, N, N, N),
            "PermuteBinary":    (self.test_PermuteBinary,   N, Y, Y, Y, Y, Y),
            "PermuteFuse":      (self.test_PermuteFuse,     N, Y, Y, Y, Y, Y),
            "PermutePad":       (self.test_PermutePad,      N, Y, Y, N, Y, Y),
            "PermuteToReorg":   (self.test_PermuteToReorg,  N, Y, Y, Y, Y, N),
            "PermuteToReorg2":  (self.test_PermuteToReorg2, N, Y, Y, Y, Y, Y),
            "PermuteToReshape": (self.test_PermuteToReshape, Y, Y, Y, N, Y, Y),
            "Permute5dSplit":   (self.test_Permute5dSplit,  Y, Y, Y, Y, Y, Y),
            "Permute7d":        (self.test_Permute7d,       Y, Y, Y, N, Y, Y),
            "PoolAfterRelu":    (self.test_PoolAfterRelu,   N, Y, Y, Y, Y, Y),
            "PoolSignError":    (self.test_PoolSignError,   N, Y, Y, Y, Y, Y),
            "ReshapeFuse":      (self.test_ReshapeFuse,     Y, Y, Y, Y, Y, Y),
            "ReshapeN":         (self.test_ReshapeN,        Y, N, N, N, N, Y),
            "ReduceTranspose":  (self.test_ReduceTranspose, Y, Y, Y, N, Y, Y),
            "ReduceFuse":       (self.test_ReduceFuse,      Y, Y, Y, Y, Y, Y),
            "SwapDimInner":     (self.test_SwapDimInner,    Y, Y, Y, N, Y, Y),
            "SliceToReverse":   (self.test_SliceToReverse,  N, Y, Y, N, Y, N),
            "StaticDynMixed":   (self.test_StaticDynMixed,  N, Y, Y, N, Y, N),
            "TransposeArg":     (self.test_TransposeArg,    Y, Y, Y, Y, Y, N),
            "If":               (self.test_If,              N, Y, Y, N, Y, N),
            # "Loop":            (self.test_Loop,             N, Y, N, N, Y, N),
            ## only for test
            "user_define_net":   (self.user_define_net,    Y, Y, Y, Y, Y, N)
        }
        # yapf: enable

        # no quantization when quant_mode == "f32"
        self.support_quant_modes = ["f32", "f16", "bf16", "int8"]
        # self.support_quant_modes = ["f32", "f16", "bf16", "int8", "int4"]
        self.support_asym = [False]
        self.model_file = ".bmodel"
        self.is_cv18xx = False
        self.chip = chip.lower()
        self.debug_cmd = debug_cmd
        self.test_cuda = cuda
        self.dynamic = dynamic
        self.simple = simple
        self.multithread = not disable_thread
        self.num_core = num_core
        self.opt = 2
        if self.simple:
            self.support_quant_modes = ["f16", "int8"]
            self.support_asym = [False]
        if self.chip.startswith("cv18") and self.chip != "cv186x":
            self.support_quant_modes = ["bf16", "int8"]
            self.support_asym = [False]
            self.model_file = ".cvimodel"
            self.is_cv18xx = True
        elif self.chip == "bm1684":
            self.support_quant_modes = ["f32", "int8"]
            self.support_asym = [False]
        elif self.chip == "bm1690" and not self.simple:
            # only full test
            self.support_quant_modes.append("f8e4m3")
            self.support_quant_modes.append("f8e5m2")
        elif self.chip =="mars3":
            self.support_quant_modes = ["bf16", "int8"]
            self.support_asym = [False]
        self.mode = mode.lower()
        if self.mode == "" or self.mode == "all":
            self.quant_modes = self.support_quant_modes
            # current supported fp8 ops are limited, so use test_fp8 instead
            for quant_mode in self.quant_modes:
                if "f8" in quant_mode:
                    self.quant_modes.remove(quant_mode)
        elif self.mode == "f8":
            assert self.chip == "bm1690" and "only bm1690 support fp8"
            self.quant_modes = ["f8e4m3", "f8e5m2"]
        else:
            if self.chip == "bm1688" or self.chip == "cv186x" or self.chip == "sg2380":
                self.support_quant_modes.append("int4")
            if self.mode not in self.support_quant_modes:
                raise RuntimeError("{} not support mode: {}".format(self.chip, self.mode))
            self.quant_modes = [self.mode]

    def test_single(self, case: str):
        np.random.seed(0)
        torch.manual_seed(7)
        print("Test: {}".format(case))
        if case in self.test_cases:
            os.makedirs(case, exist_ok=True)
            os.chdir(case)
            func, _, _, _, _, _, _ = self.test_cases[case]
            func(case)
            print("====== TEST {} Success ======".format(case))
        else:
            raise RuntimeError("case [{}] is not exist".format(case))

    def check_support(self, case):
        _, bm1684_support, bm1684x_support, bm1688_support, cv183x_support, bm1690_support,mars3_support= self.test_cases[
            case]
        if self.is_cv18xx and cv183x_support:
            return True
        if self.chip == "bm1684" and bm1684_support:
            return True
        if self.chip == "bm1684x" and bm1684x_support:
            return True
        if self.chip == "bm1688" and bm1688_support:
            return True
        if self.chip == "bm1690" and bm1690_support:
            return True
        if self.chip == "mars3" and mars3_support:
            return True
        return False

    def create_random_input(self, graph_def: onnx.GraphProto):
        inputs = {}
        for i in graph_def.input:
            name = i.name
            shape = [s.dim_value for s in i.type.tensor_type.shape.dim]
            if i.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
                inputs[name] = np.clip(np.random.randn(*shape).astype(np.float32), -10, 10)
            elif i.type.tensor_type.elem_type == onnx.TensorProto.BOOL:
                # create random input data for bool type
                inputs[name] = np.random.randint(0, 2, shape).astype(np.bool_)
            elif i.type.tensor_type.elem_type == onnx.TensorProto.INT64:
                inputs[name] = np.random.randint(1, 10, shape).astype(np.int64)
            else:
                assert(0)
        return inputs

    def onnx_convert(self,
                     input_data: dict,
                     graph_def,
                     model_name: str,
                     quant_modes: list,
                     static_shape=True,
                     version=14,
                     dynamic=False):
        # onnx --> mlir conversion (origin and optimized mlir models will be generated and saved)
        fp32_mlir = "{}.mlir".format(model_name)
        model_def = helper.make_model(graph_def, producer_name=model_name)
        model_def.opset_import[0].version = version
        input_npz = "{}_in_fp32.npz".format(model_name)
        file_mark(input_npz)
        for name in input_data:
            if input_data[name].dtype in [np.int64, np.int32]:
                input_data[name] = input_data[name].astype(np.int32)
            else:
                input_data[name] = input_data[name].astype(np.float32)
        np.savez(input_npz, **input_data)
        onnx.checker.check_model(model_def)
        tool = OnnxTransformer(model_name, model_def, test_input=input_npz, static_shape=static_shape, dynamic=dynamic)
        node_name_mapping = tool.converter.node_name_mapping
        tool.model_transform(fp32_mlir)

        onnx_model = "{}_opt.onnx".format(model_name)
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
        for qmode in quant_modes:
            self.make_test_calibration_table(top_mlir_outs, self.table_name, qmode)

        return (onnx_outs, top_mlir_outs, input_npz, node_name_mapping)

    def bmodel_generate(self, model_name: str, quant_mode: str, isAsym: bool = False, matmul_perchannel: bool = False):

        top_mlir = "{}.mlir".format(model_name)
        tpu_mlir = "{}_{}".format(model_name, quant_mode)
        table = None
        if quant_mode == "int8" or quant_mode == "int4":
            tpu_mlir += "_asym" if isAsym else "_sym"
            table = self.table_name
        elif quant_mode == "f8e4m3" or quant_mode == "f8e5m2":
            table = self.table_name + "_" + quant_mode

        # lowering
        mlir_lowering(top_mlir,
                      tpu_mlir + ".mlir",
                      mode=quant_mode,
                      chip=self.chip,
                      num_core=self.num_core,
                      cali_table=table,
                      asymmetric=isAsym,
                      matmul_perchannel=matmul_perchannel)

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
        mlir_to_model(tpu_mlir + ".mlir",
                      bmodel,
                      tpu_final,
                      self.dynamic,
                      quant_input,
                      quant_output,
                      opt = self.opt,
					  debug_cmd = f'--debug_cmd={self.debug_cmd}')
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
            ref_tpu_tolerance = "0.95,0.80"
        elif quant_mode == "f8e4m3" or quant_mode == "f8e5m2":
            ref_tpu_tolerance = "0.95,0.80"
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
        # cuda inference and compare
        if self.test_cuda:
            cuda_npz = tpu_mlir.replace(".mlir", "_cuda_out.npz")
            show_fake_cmd(input_npz, tpu_mlir, cuda_npz, True, True)
            cuda_outs = mlir_inference(input_data, tpu_mlir, dump_all=True, use_cuda=True)
            np.savez(cuda_npz, **cuda_outs)
            file_mark(cuda_npz)
            npz_compare([cuda_npz, tpu_npz, "--tolerance", "0.9999,0.9999", "-v"])
        msg = quant_mode.upper()
        if quant_mode == "int8" or quant_mode == "int4":
            msg += ", Asymmetric: {}".format(isAsym)

        print("[Success] test {} {}".format(model_name, msg))

    def inference_and_compare_bmodel(self,
                                     bmodel: str,
                                     input_npz: str,
                                     onnx_outs,
                                     quant_mode: str,
                                     model_name: str,
                                     isAsym: bool = False,
                                     node_name_mapping=None):
        ref_bmodel_tolerance = "0.9,0.9"
        input_data = np.load(input_npz)
        if quant_mode == "int8":
            ref_bmodel_tolerance = "0.95,0.70" if not isAsym else "0.90,0.54"
        elif quant_mode == "int4":
            ref_bmodel_tolerance = "0.90,0.60"
        elif quant_mode == "bf16":
            ref_bmodel_tolerance = "0.95,0.80"
        elif quant_mode == "f8e4m3" or quant_mode == "f8e5m2":
            ref_bmodel_tolerance = "0.95,0.80"
        model_npz = bmodel.replace("." + bmodel.split(".")[-1], "_model_out.npz")
        file_mark(model_npz)
        show_fake_cmd(input_npz, bmodel, model_npz)
        model_outs = model_inference(input_data, bmodel)
        np.savez(model_npz, **model_outs)

        counter = 0
        onnx_transformed_model = {}
        for name in onnx_outs:
            key_vals = {key_val[0:len(name)]: key_val for key_val in model_outs.keys()}
            found = False
            if name in key_vals.keys():
                mapped_name = key_vals[name]
                found = True
            elif name in node_name_mapping:
                mapped_name = node_name_mapping[name]
                found = mapped_name in model_outs
            if found:
                if model_outs[mapped_name].shape != onnx_outs[name].shape:
                    raise RuntimeError(
                        f"onnx and bmodel output shape not equal, {model_outs[mapped_name].shape} vs {onnx_outs[name].shape}"
                    )
                onnx_transformed_model[mapped_name] = onnx_outs[name]
                counter += 1
        if counter == 0:
            raise RuntimeError("No compare between onnx outs and bmodel outs")
        onnx_transformed_npz = bmodel.replace("." + bmodel.split(".")[-1], "_onnx_transformed.npz")
        file_mark(onnx_transformed_npz)
        np.savez(onnx_transformed_npz, **onnx_transformed_model)
        npz_compare([onnx_transformed_npz, model_npz, "--tolerance", ref_bmodel_tolerance, "-v"])

        msg = quant_mode.upper()
        if quant_mode == "int8" or quant_mode == "int4":
            msg += ", Asymmetric: {}".format(isAsym)

        print("[Success] test {} {}".format(model_name, msg))

    def make_test_calibration_table(self, tensors, table_name, qmode=None):
        # simple calibration table
        if qmode == 'f8e4m3' or qmode == 'f8e5m2':
            table_name = table_name + "_" + qmode
        elif qmode not in ['int8','int4']:
            return
        with open(table_name, 'w') as f:
            if qmode == 'f8e4m3' or qmode == 'f8e5m2':
                f.write("#tpu-mlir-fp8 caliration table\n")
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
        return np.sqrt(np.sum(np.power(x, 2)))

    def cosine_similarity(self, x, y):
        numerator = np.sum(x * y)
        denominator = self.square_rooted(x) * self.square_rooted(y)
        return round(numerator / float(denominator), 3)

    def compare(self, ref_out, targe_out):
        if ref_out.dtype in [np.int64, np.int32, np.int16, np.int8]:
            cos = self.cosine_similarity(ref_out, targe_out)
            assert (cos > 0.999)
        # else:
        #     np.testing.assert_allclose(ref_out, targe_out, rtol=1e-5, atol=1e-01)

    def torch_and_onnx_compare(self, input_data: dict, onnx_file: str, origin_output):
        onnx_outs = self.simple_onnx_inference(input_data, onnx_file)
        num_outputs = len(onnx_outs)
        if isinstance(origin_output, tuple):
            assert (len(origin_output) == num_outputs)
            for i in range(num_outputs):
                x = origin_output[i].data.numpy()
                y = onnx_outs[i]
                self.compare(x.ravel(), y.ravel())
        else:
            self.compare(origin_output.data.numpy().ravel(), onnx_outs[0].ravel())
        print("* Torch and Onnx result compared *")

    def torch_and_test(self, inputs, torch_model: nn.Module, model_name: str, static_shape=True, dynamic=False):
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
                if input.data.dtype == torch.float64:
                    in_data[name] = input.data.numpy().astype(np.float64)
                elif input.data.dtype == torch.float32:
                    in_data[name] = input.data.numpy().astype(np.float32)
                elif input.data.dtype == torch.int64:
                    in_data[name] = input.data.numpy().astype(np.int64)
                elif input.data.dtype == torch.int32:
                    in_data[name] = input.data.numpy().astype(np.int32)
                else:
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
            opset_version=14,  # export hardswish needed
            input_names=in_names)
        onnx_model = onnx.load(onnx_file)
        self.torch_and_onnx_compare(in_data, onnx_file, origin_output)
        self.onnx_and_test(onnx_model.graph,
                           name=model_name,
                           input_data=in_data,
                           static_shape=static_shape,
                           dynamic=dynamic)

    def onnx_and_test(self,
                      graph_def,
                      name: str = "",
                      input_data: dict = None,
                      static_shape=True,
                      check_last: bool = False,
                      support_modes=None,
                      version=14,
                      dynamic=False,
                      matmul_perchannel=False):
        print(matmul_perchannel)
        if support_modes is None:
            quant_modes = self.quant_modes
        else:
            quant_modes = list(set(self.quant_modes) & set(support_modes))
        if input_data is None:
            input_data = self.create_random_input(graph_def)
        model_name = name if name else graph_def.name
        onnx_outs, top_mlir_outs, input_npz, node_name_mapping = self.onnx_convert(
            input_data,
            graph_def,
            model_name,
            quant_modes,
            static_shape=static_shape,
            version=version,
            dynamic=dynamic)
        # this assumes that outputs are in order, i.e. the last one is the output
        if check_last:
            top_mlir_outs[list(onnx_outs.keys())[-1]] = list(top_mlir_outs.values())[-1]
        # test onnx and mlir outputs
        counter = 0
        for name in onnx_outs:
            if name in top_mlir_outs:
                print("Compare onnx and mlir:{}\n".format(name))
                top_mlir_output = top_mlir_outs[name].ravel()
                onnx_output = onnx_outs[name].ravel()
                self.compare(onnx_output, top_mlir_output)
                counter += 1
            elif name in node_name_mapping:
                mapped_name = node_name_mapping[name]
                if mapped_name in top_mlir_outs:
                    print("Compare onnx and mlir:{}\n".format(mapped_name))
                    top_mlir_output = top_mlir_outs[mapped_name].ravel()
                    onnx_output = onnx_outs[name].ravel()
                    self.compare(onnx_output, top_mlir_output)
                    counter += 1
        if counter == 0:
            raise RuntimeError("No compare between onnx outs and mlir outts")
        print("Success: ONNX outs and Mlir outs are equal\n")
        for quant_mode in quant_modes:
            if quant_mode == "int8" or quant_mode == "int4":
                for isAsym in self.support_asym:
                    tpu_mlir, bmodel = self.bmodel_generate(model_name, quant_mode, isAsym, matmul_perchannel)
                    self.inference_and_compare(tpu_mlir, bmodel, input_npz, quant_mode, model_name,
                                               isAsym)
            else:
                tpu_mlir, bmodel = self.bmodel_generate(model_name, quant_mode)
                self.inference_and_compare(tpu_mlir, bmodel, input_npz, quant_mode, model_name)

    def onnx_and_test_bmodel(self,
                             graph_def,
                             name: str = "",
                             input_data: dict = None,
                             static_shape=True,
                             check_last: bool = False,
                             quant_modes=None,
                             only_cmp_with_bmodel=False,
                             version=14):
        if quant_modes is None:
            quant_modes = self.quant_modes
        if input_data is None:
            input_data = self.create_random_input(graph_def)
        model_name = name if name else graph_def.name
        onnx_outs, top_mlir_outs, input_npz, node_name_mapping = self.onnx_convert(
            input_data,
            graph_def,
            model_name,
            quant_modes,
            static_shape=static_shape,
            version=version)
        # this assumes that outputs are in order, i.e. the last one is the output
        if check_last:
            top_mlir_outs[list(onnx_outs.keys())[-1]] = list(top_mlir_outs.values())[-1]
        for quant_mode in quant_modes:
            if quant_mode == "int8" or quant_mode == "int4":
                for isAsym in self.support_asym:
                    _, bmodel = self.bmodel_generate(model_name, quant_mode, isAsym)
                    self.inference_and_compare_bmodel(bmodel, input_npz, onnx_outs, quant_mode,
                                                      model_name, isAsym, node_name_mapping)
            else:
                _, bmodel = self.bmodel_generate(model_name, quant_mode)
                self.inference_and_compare_bmodel(bmodel,
                                                  input_npz,
                                                  onnx_outs,
                                                  quant_mode,
                                                  model_name,
                                                  node_name_mapping=node_name_mapping)

    ##################################
    # adding operators from here
    ##################################
    def AvgPoolBase(self, case_name, input_shape, output_shape, kernel, strides):
        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = AveragePool<kernel_shape=%s, strides=%s>(input)
            }
            """ % (case_name, input_shape, output_shape, kernel, strides)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Unsqueeze(self, case_name):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                y = x.unsqueeze(0)
                return y

        x = torch.randn(76800,2).float()
        self.torch_and_test(x, Model(), case_name)

    def test_Space2Depth(self, case_name):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                y = F.pixel_unshuffle(x,2)
                return y

        x = torch.arange(48*4).reshape(1,3,8,8).float()
        self.torch_and_test(x, Model(), case_name)

    def test_AvgPool1d(self, case_name):
        input_shape = [1, 32, 128]
        output_shape = [1, 32, 64]

        mul_const = helper.make_tensor(name='const_mul',
                                       data_type=TensorProto.FLOAT,
                                       dims=[],
                                       vals=[2.0])

        graph_txt = """
            %s (float%s input) => (float%s output)
            <float const_mul>
            {
                pool_output = AveragePool<kernel_shape=[2], strides=[2]>(input)
                output = Mul(pool_output, const_mul)
            }
            """ % (case_name, input_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([mul_const])
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

        graph_txt = """
            %s (float%s input) => (float%s output)
            <int64[%s] pads = %s>
            {
                pad_output = Pad<mode="constant">(input, pads)
                output = AveragePool<kernel_shape=%s, strides=%s>(pad_output)
            }
            """ % (case_name, input_shape, output_shape, pads.shape[0], str(list(pads)).replace(
            '[', '{').replace(']', '}'), kernel_shape, strides)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_BatchMatMul(self, case_name):
        # matmul(1x16x40x64, 1x16x64x40) => 1x16x40x40
        input1_shape = [4, 16, 40, 64]
        input2_shape = [4, 16, 64, 40]
        output_shape = [4, 16, 40, 40]

        graph_txt = """
            %s (float%s input1, float%s input2) => (float%s output)
            {
                output = MatMul(input1, input2)
            }
            """ % (case_name, input1_shape, input2_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def MaxPoolBase(self, case_name, input_shape, output_shape, kernel, strides):
        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = MaxPool<kernel_shape=%s, strides=%s>(input)
            }
            """ % (case_name, input_shape, output_shape, kernel, strides)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_GlobalAveragePool(self, case_name):
        input_shape = [1, 3, 64, 64]
        output_shape = [1, 3, 1, 1]
        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = GlobalAveragePool(input)
            }
            """ % (case_name, input_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_GlobalMaxPool(self, case_name):
        input_shape = [1, 3, 64, 64]
        output_shape = [1, 3, 1, 1]
        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = GlobalMaxPool(input)
            }
            """ % (case_name, input_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_GroupFC(self, case_name):
        input_shape = [16, 40, 43]
        filter_shape = [16, 43, 48]
        bias_shape = [16, 1, 48]
        output_shape = [16, 40, 48]

        filter_data = np.random.rand(*filter_shape).astype(np.float32).flatten()
        bias_data = np.random.rand(*bias_shape).astype(np.float32).flatten()
        filter = helper.make_tensor('filter', TensorProto.FLOAT, filter_shape, filter_data)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, bias_shape, bias_data)
        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s filter, float%s bias>
            {
                fc = MatMul(input, filter)
                output = Add(fc, bias)
            }
            """ % (case_name, input_shape, output_shape, filter_shape, bias_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([filter, bias])
        self.onnx_and_test(graph_def)

    def test_GRU(self, case_name):
        seq_length = 75
        batch_size = 2
        num_dir = 2
        input_size = 64
        hidden_size = 32
        input_shape = [seq_length, batch_size, input_size]
        Y_shape = [seq_length, num_dir, batch_size, hidden_size]
        direction = 'forward' if num_dir == 1 else 'bidirectional'
        h_data = np.random.rand(num_dir, batch_size, hidden_size).astype(np.float32)
        w_data = np.random.rand(num_dir, 3 * hidden_size, input_size).astype(np.float32)
        r_data = np.random.rand(num_dir, 3 * hidden_size, hidden_size).astype(np.float32)
        b_data = np.random.rand(num_dir, 6 * hidden_size).astype(np.float32)

        graph_txt = """
            %s (float%s input) => (float%s Y)
            <float%s w, float%s r, float%s b, float%s h>
            {
                Y, = GRU<direction="%s",hidden_size=%d,linear_before_reset=1>(input, w, r, b, ,h)
            }
            """ % (case_name, input_shape, Y_shape, list(w_data.shape), list(r_data.shape), list(
            b_data.shape), list(h_data.shape), direction, hidden_size)
        graph_def = onnx.parser.parse_graph(graph_txt)
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
        graph_def.initializer.extend([w_value, r_value, b_value, h_value])
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
        input_shape = [seq_length, batch_size, input_size]
        Y_shape = [seq_length, num_dir, batch_size, hidden_size]
        Y_h_shape = [num_dir, batch_size, hidden_size]
        h_data = np.random.rand(num_dir, batch_size, hidden_size).astype(np.float32)
        w_data = np.random.rand(num_dir, 3 * hidden_size, input_size).astype(np.float32)
        r_data = np.random.rand(num_dir, 3 * hidden_size, hidden_size).astype(np.float32)
        b_data = np.random.rand(num_dir, 6 * hidden_size).astype(np.float32)

        graph_txt = """
            %s (float%s input) => (float%s Y_h)
            <float%s w, float%s r, float%s b, float%s h, float%s Y>
            {
                Y, Y_h = GRU<direction="%s",hidden_size=%d,linear_before_reset=1>(input, w, r, b, ,h)
            }
            """ % (case_name, input_shape, Y_h_shape, list(w_data.shape), list(r_data.shape), list(
            b_data.shape), list(h_data.shape), Y_shape, direction, hidden_size)
        graph_def = onnx.parser.parse_graph(graph_txt)

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
        graph_def.initializer.extend([w_value, r_value, b_value, h_value])
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
        input_shape = [seq_length, batch_size, input_size]
        Y_shape = [seq_length, num_dir, batch_size, hidden_size]
        Y_h_shape = [num_dir, batch_size, hidden_size]

        h_data = np.random.rand(num_dir, batch_size, hidden_size).astype(np.float32)
        w_data = np.random.rand(num_dir, 3 * hidden_size, input_size).astype(np.float32)
        r_data = np.random.rand(num_dir, 3 * hidden_size, hidden_size).astype(np.float32)
        b_data = np.random.rand(num_dir, 6 * hidden_size).astype(np.float32)

        graph_txt = """
            %s (float%s input) => (float%s Y, float%s Y_h)
            <float%s w, float%s r, float%s b, float%s h>
            {
                Y, Y_h = GRU<direction="%s",hidden_size=%d,linear_before_reset=1>(input, w, r, b, ,h)
            }
            """ % (case_name, input_shape, Y_shape, Y_h_shape, list(w_data.shape), list(
            r_data.shape), list(b_data.shape), list(h_data.shape), direction, hidden_size)
        graph_def = onnx.parser.parse_graph(graph_txt)
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
        graph_def.initializer.extend([w_value, r_value, b_value, h_value])
        if self.is_cv18xx:
            input_data = {}
            input_data["input"] = np.random.rand(seq_length, batch_size,
                                                 input_size).astype(np.float32)
            self.onnx_and_test(graph_def, input_data=input_data)
            return
        self.onnx_and_test(graph_def)

    def test_GRU4(self, case_name):
        seq_length = 101
        batch_size = 640
        num_dir = 2
        input_size = 64
        hidden_size = 128
        direction = 'forward' if num_dir == 1 else 'bidirectional'
        input_shape = [seq_length, batch_size, input_size]
        Y_shape = [seq_length, num_dir, batch_size, hidden_size]
        Y_h_shape = [num_dir, batch_size, hidden_size]

        h_data = np.random.rand(num_dir, batch_size, hidden_size).astype(np.float32)
        w_data = np.random.rand(num_dir, 3 * hidden_size, input_size).astype(np.float32)
        r_data = np.random.rand(num_dir, 3 * hidden_size, hidden_size).astype(np.float32)
        b_data = np.random.rand(num_dir, 6 * hidden_size).astype(np.float32)

        graph_txt = """
            %s (float%s input) => (float%s Y, float%s Y_h)
            <float%s w, float%s r, float%s b, float%s h>
            {
                Y, Y_h = GRU<direction="%s",hidden_size=%d,linear_before_reset=1>(input, w, r, b, ,h)
            }
            """ % (case_name, input_shape, Y_shape, Y_h_shape, list(w_data.shape), list(
            r_data.shape), list(b_data.shape), list(h_data.shape), direction, hidden_size)
        graph_def = onnx.parser.parse_graph(graph_txt)
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
        graph_def.initializer.extend([w_value, r_value, b_value, h_value])
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

        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s weight, float%s bias>
            {
                output = Conv<kernel_shape=%s,pads=%s,strides=%s,dilations=%s,group=%d>(input, weight, bias)
            }
            """ % (case_name, input_shape, output_shape, filter_shape, list(
            bias_data.shape), kernel, padding, stride, dilation, groups)
        graph_def = onnx.parser.parse_graph(graph_txt)

        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, list(bias_data.shape), bias_data)
        graph_def.initializer.extend([weight, bias])
        self.onnx_and_test(graph_def)

    def test_Conv1d(self, case_name):
        oc = 32
        self.ConvBase(case_name, [1, 16, 100], [oc, 16, 3], [1, oc, 100], [3], [1, 1], [1], [1], 1)

    def test_Conv1d_bigd(self, case_name):
        # batchs = [1]
        # for idx, batch in enumerate(batchs):
        input_shape = [1, 1024, 163]
        filter_shape = [1024, 1024, 3]
        output_shape = [1, 1024, 1]
        self.ConvBase(case_name,
                      input_shape,
                      filter_shape,
                      output_shape,
                      kernel=[3],
                      padding=[0, 0],
                      stride=[1],
                      dilation=[81],
                      groups=1)

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

    def test_Conv2d_bigd(self, case_name):
        batchs = [1]
        for idx, batch in enumerate(batchs):
            input_shape = [batch, 2048, 80, 40]
            filter_shape = [2048, 1, 3, 3]
            output_shape = [batch, 2048, 80, 40]
            self.ConvBase("{}_{}".format(case_name, idx),
                          input_shape,
                          filter_shape,
                          output_shape,
                          kernel=[3, 3],
                          padding=[24, 24, 24, 24],
                          stride=[1, 1],
                          dilation=[24, 24],
                          groups=2048)

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

        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s filter0, float%s filter1>
            {
                x1 = Conv<kernel_shape=[3, 3],pads=[1, 1, 1, 1],strides=[2, 2],dilations=[1, 1],group=1>(input, filter0)
                x2 = Sigmoid(x1)
                x3 = Mul(x1, x2)
                output = Conv<kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1], dilations=[1, 1], group=1>(x3, filter1)
            }
            """ % (case_name, in_shape0, out_shape, f_shape0, f_shape1)
        graph_def = onnx.parser.parse_graph(graph_txt)

        filter0 = helper.make_tensor('filter0', TensorProto.FLOAT, f_shape0, f_data0)
        filter1 = helper.make_tensor('filter1', TensorProto.FLOAT, f_shape1, f_data1)
        graph_def.initializer.extend([filter0, filter1])
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

    def test_ConvPool3d(self, case_name):
        # oc = 32
        input_shape = [1, 3, 16, 112, 112]
        filter_shape = [64, 3, 3, 3, 3]
        output_shape = [1, 64, 16, 56, 56]
        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        bias_data = np.random.randn(output_shape[1]).astype(np.float32)

        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s weight, float%s bias>
            {
                conv_output = Conv<kernel_shape=[3, 3, 3],pads=[1, 1, 1, 1, 1, 1],strides=[1, 1, 1],dilations=[1, 1, 1],group=1>(input, weight, bias)
                relu_output = Relu(conv_output)
                output = MaxPool<kernel_shape=[1,2,2], pads=[0,0,0,0,0,0], strides=[1,2,2]>(relu_output)
            }
            """ % (case_name, input_shape, output_shape, filter_shape, list(bias_data.shape))
        graph_def = onnx.parser.parse_graph(graph_txt)

        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, list(bias_data.shape), bias_data)

        graph_def.initializer.extend([weight, bias])
        self.onnx_and_test(graph_def)

    def test_SiLU(self, case_name):
        input_shape = [1, 16, 64, 64]
        output_shape = [1, 16, 64, 64]

        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                x1 = Sigmoid(input)
                output = Mul(input, x1)
            }
            """ % (case_name, input_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Trilu(self, case_name):
        input_shape = [1, 16, 64, 64]

        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = Trilu<upper=1>(input)
            }
            """ % (case_name, input_shape, input_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Concat(self, case_name):
        input_shape = {"input1": [1, 2, 64], "input2": [1, 3, 64], "input3": [1, 4, 64]}
        output_shape = [1, 2 + 3 + 4, 64]
        graph_txt = """
            %s (float[1, 2, 64] input1, float[1, 3, 64] input2, float[1, 4, 64] input3) => (float%s output)
            {
                output = Concat<axis=1>(input1, input2, input3)
            }
            """ % (case_name, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Concat2(self, case_name):
        input_shape = [1, 192, 256]
        x_shape = [1, 192, 16]
        output_shape = [1, 192, 288]

        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s X1, float%s X2>
            {
                output = Concat<axis=-1>(input, X1, X2)
            }
            """ % (case_name, input_shape, output_shape, x_shape, x_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)

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
        graph_def.initializer.extend([x1_node_def, x2_node_def])
        self.onnx_and_test(graph_def)

    def test_Transpose(self, case_name):
        input_shapes = [[1, 16, 32, 32], [4, 3, 85, 20, 20], [1, 4, 2, 16, 20, 40]]
        transpose_orders = {4: [0, 2, 1, 3], 5: [0, 1, 3, 4, 2], 6: [0, 1, 2, 5, 3, 4]}
        for i, input_shape in enumerate(input_shapes):
            order = transpose_orders[len(input_shape)]
            output_shape = [input_shape[order[i]] for i in range(len(order))]
            graph_txt = """
                %s (float%s input) => (float%s output)
                {
                    output = Transpose<perm=%s>(input)
                }
                """ % (case_name, input_shape, output_shape, order)
            graph_def = onnx.parser.parse_graph(graph_txt)
            self.onnx_and_test(graph_def)

    def test_Transpose2(self, case_name):
        cases = [
            # case 0, input shape,output shape, order
            [[529, 49, 3, 3, 32], [3, 529, 3, 49, 32], [2, 0, 3, 1, 4]],
            # case 1, input shape,output shape, order
            [[1, 1, 1, 23, 7, 23, 7, 96], [1, 1, 23, 23, 1, 7, 7, 96], [0, 1, 3, 5, 2, 4, 6, 7]]
        ]
        for idx, shapes in enumerate(cases):
            graph_txt = """
                %s (float%s input) => (float%s output)
                {
                    output = Transpose<perm=%s>(input)
                }
                """ % (case_name, shapes[0], shapes[1], shapes[2])
            graph_def = onnx.parser.parse_graph(graph_txt)
            self.onnx_and_test(graph_def)

    def test_Where(self, case_name):
        # real case
        # normal case, broadcast case
        cond_shape = [10, 40, 224], [1, 1, 2, 1]
        tbrn_shape = [10, 40, 224], [1, 256]
        fbrn_shape = [10, 40, 224], [1, 1, 2, 256]
        out_shape = [10, 40, 224], [1, 1, 2, 256]

        cond_data = [np.zeros(e).astype(np.bool_) for e in cond_shape]
        cond_data[0][:, :, :100] = 1
        cond_data[1][..., 1:, :] = 1

        for idx in range(len(cond_shape)):
            tbrn_data = np.random.randn(*tbrn_shape[idx]).astype(np.float32)
            fbrn_data = np.random.randn(*fbrn_shape[idx]).astype(np.float32)
            graph_txt = """
                %s (bool%s cond, float%s tbrn, float%s fbrn) => (float%s output)
                <float const_mul = {2.0}>
                {
                    output = Where(cond, tbrn, fbrn)
                }
                """ % (case_name, cond_shape[idx], tbrn_shape[idx], fbrn_shape[idx], out_shape[idx])
            graph_def = onnx.parser.parse_graph(graph_txt)
            self.onnx_and_test(graph_def,
                               input_data={
                                   "cond": cond_data[idx],
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

        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s weight, float%s bias>
            {
                conv_output = Conv<kernel_shape=[3, 3],pads=[1, 1, 1, 1],strides=[1, 1],dilations=[1, 1],group=1>(input, weight, bias)
                output = Relu(conv_output)
            }
            """ % (case_name, input_shape, output_shape, filter_shape, list(bias_data.shape))
        graph_def = onnx.parser.parse_graph(graph_txt)

        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, list(bias_data.shape), bias_data)
        graph_def.initializer.extend([weight, bias])
        self.onnx_and_test(graph_def)

    def test_ReluOnly(self, case_name):
        oc = 64
        input_shape = [1, 16, 128, 128]
        filter_shape = [oc, 16, 3, 3]
        output_shape = [1, oc, 128, 128]
        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        bias_data = np.random.randn(oc).astype(np.float32)

        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s weight, float%s bias>
            {
                relu_output = Relu(input)
                output = Conv<kernel_shape=[3, 3],pads=[1, 1, 1, 1],strides=[1, 1],dilations=[1, 1],group=1>(relu_output, weight, bias)
            }
            """ % (case_name, input_shape, output_shape, filter_shape, list(bias_data.shape))
        graph_def = onnx.parser.parse_graph(graph_txt)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, list(bias_data.shape), bias_data)
        graph_def.initializer.extend([weight, bias])
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
            graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s weight, float%s bias>
            {
                conv_output = Conv<kernel_shape=[3, 3],pads=[1, 1, 1, 1],strides=[1, 1],dilations=[1, 1],group=1>(input, weight, bias)
                output = LeakyRelu<alpha=%f>(conv_output)
            }
            """ % (case_name, input_shape, output_shape, filter_shape, list(bias_data.shape), a)
            graph_def = onnx.parser.parse_graph(graph_txt)
            weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
            bias = helper.make_tensor('bias', TensorProto.FLOAT, list(bias_data.shape), bias_data)
            graph_def.initializer.extend([weight, bias])
            self.onnx_and_test(graph_def)

    def test_Mul(self, case_name):
        input_shape = [
            {"input1": [1, 3, 27, 27], "input2": [1, 3, 27, 27]},
            {"input1": [1, 4420, 2], "input2": [1, 4420, 2]}
        ]
        output_shape = [[1, 3, 27, 27], [1, 4420, 2]]

        for i in range(len(input_shape)):
            input1 = input_shape[i]["input1"]
            input2 = input_shape[i]["input2"]
            output = output_shape[i]
            graph_txt = """
                %s (float%s input1, float%s input2) => (float%s output)
                {
                    output = Mul(input1, input2)
                }
                """ % (case_name, input1, input2, output)
            graph_def = onnx.parser.parse_graph(graph_txt)
            self.onnx_and_test(graph_def)

    def test_MulBcast(self, case_name):
        shapes = ([7, 9, 44, 38], )
        bcast_dims = ([[0], [2], [0, 2]], )
        for i, s in enumerate(shapes):
            for dims in bcast_dims[i]:
                bcast_s = s[::]
                for dim in dims:
                    bcast_s[dim] = 1
                graph_txt = """
                %s_%s_%s (float%s a, float%s b, float%s c) => (float%s output)
                {
                    x = Mul(a, b)
                    output = Mul(x, c)
                }
                """ % (case_name, i, "".join(map(str, dims)), bcast_s, s, bcast_s, s)
                graph_def = onnx.parser.parse_graph(graph_txt)
                self.onnx_and_test(graph_def)

    def test_MulBcast2(self, case_name):
        shapes = ([4, 7, 1, 15], )
        bcast_shapes = ([1, 7, 13, 15], )
        out_shapes = ([4, 7, 13, 15], )
        for i, s in enumerate(shapes):
            graph_txt = """
                %s_%s (float%s a, float%s b, float%s c) => (float%s output)
                {
                    x = Mul(a, b)
                    output = Mul(x, c)
                }
                """ % (case_name, i, bcast_shapes[i], s, bcast_shapes[i], out_shapes[i])
            graph_def = onnx.parser.parse_graph(graph_txt)
            self.onnx_and_test(graph_def)

    def test_MulConst(self, case_name):
        input_shape = [1, 3, 27, 27]
        output_shape = [1, 3, 27, 27]

        graph_txt = """
                %s (float%s input) => (float%s output)
                <float const_mul = {-2.0}>
                {
                    output = Mul(input, const_mul)
                }
                """ % (case_name, input_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_MulMerge(self, case_name):
        input_shape = [1, 3, 24, 24]
        output_shape = [1, 3, 24, 24]
        w_data = np.random.rand(*input_shape).astype(np.float32)

        graph_txt = """
                %s (float%s input1) => (float%s output2)
                <float%s w>
                {
                    output1 = Mul(input1, w)
                    output2 = Mul(output1, w)
                }
                """ % (case_name, input_shape, output_shape, list(w_data.shape))
        graph_def = onnx.parser.parse_graph(graph_txt)

        w_value = helper.make_tensor(
            name='w',
            data_type=onnx.TensorProto.FLOAT,
            dims=w_data.shape,
            vals=w_data.flatten(),
        )
        graph_def.initializer.extend([w_value])
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
        graph_txt = """
                %s (float%s input) => (float%s output)
                <float%s weight, float%s bias>
                {
                    output = Gemm(input, weight, bias)
                }
                """ % (case_name, input_shape, output_shape, weight_shape, bias_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)

        weight = helper.make_tensor('weight', TensorProto.FLOAT, weight_shape, weight_data)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, bias_shape, bias_data)
        graph_def.initializer.extend([weight, bias])
        self.onnx_and_test(graph_def)

    def test_MatMul(self, case_name):
        M = 50
        K = 100
        N = 25
        input_shape = [4, 10, M, K]
        weight_shape = [K, N]
        output_shape = [4, 10, M, N]
        bias_shape = [1, 1, 1, N]
        weight_data = np.random.randn(*weight_shape).astype(np.float32)
        bias_data = np.random.randn(*bias_shape).astype(np.float32)

        graph_txt = """
                %s (float%s input) => (float%s output)
                <float%s weight, float%s bias>
                {
                    mm = MatMul(input, weight)
                    output = Add(mm, bias)
                }
                """ % (case_name, input_shape, output_shape, weight_shape, bias_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, weight_shape, weight_data)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, bias_shape, bias_data)

        graph_def.initializer.extend([weight, bias])
        self.onnx_and_test(graph_def)

    def test_MatMul2(self, case_name, per_channel=False):
        M = 50
        K = 100
        N = 25
        input_shape = [4, 10, M, K]
        weight_shape = [K, N]
        bias_shape = [N]
        output_shape = [4, 10, M, N]
        weight_data = np.random.randn(*weight_shape).astype(np.float32)
        bias_data = np.random.randn(*bias_shape).astype(np.float32)
        graph_txt = """
                %s (float%s input) => (float%s output)
                <float%s weight, float%s bias>
                {
                    x1 = MatMul(input, weight)
                    output = Add(x1, bias)
                }
                """ % (case_name, input_shape, output_shape, weight_shape, bias_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)

        weight = helper.make_tensor('weight', TensorProto.FLOAT, weight_shape, weight_data)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, bias_shape, bias_data)
        graph_def.initializer.extend([weight, bias])
        self.onnx_and_test(graph_def, matmul_perchannel=per_channel)

    def test_MatMul2PC(self, case_name):
        return self.test_MatMul2(case_name, per_channel=True)

    # def test_MatMul3(self, case_name):
    #     M = 50
    #     K = 100
    #     N = 25
    #     input_shape = [4, 10, M, K]
    #     weight_shape = [4, 10, K, N]
    #     bias_shape = [N]
    #     output_shape = [4, 10, M, N]
    #     weight_data = np.random.randn(*weight_shape).astype(np.float32)
    #     bias_data = np.random.randn(*bias_shape).astype(np.float32)
    #     input = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
    #     output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
    #     weight = helper.make_tensor('weight', TensorProto.FLOAT, weight_shape, weight_data)
    #     bias = helper.make_tensor('bias', TensorProto.FLOAT, bias_shape, bias_data)
    #     gemm_def = helper.make_node("MatMul", inputs=["input", "weight"], outputs=["x1"])
    #     add_def = helper.make_node("Add", inputs=["x1", "bias"], outputs=["output"])
    #     graph_def = helper.make_graph([gemm_def, add_def],
    #                                   case_name, [input], [output],
    #                                   initializer=[weight, bias])
    #     self.onnx_and_test(graph_def)

    def test_Scale(self, case_name):
        input_shape = [1, 65, 100, 100]
        output_shape = [1, 65, 100, 100]
        weight_shape = [1, 65, 1, 1]
        offset_shape = [1, 65, 1, 1]
        weight_data = np.random.randn(*weight_shape).astype(np.float32)
        offset_data = np.random.randn(*offset_shape).astype(np.float32)

        graph_txt = """
                %s (float%s input) => (float%s output)
                <float%s weight, float%s offset>
                {
                    mul_output = Mul(input, weight)
                    output = Add(mul_output, offset)
                }
                """ % (case_name, input_shape, output_shape, weight_shape, offset_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)

        weight = helper.make_tensor('weight', TensorProto.FLOAT, weight_shape, weight_data)
        offset = helper.make_tensor('offset', TensorProto.FLOAT, offset_shape, offset_data)
        graph_def.initializer.extend([weight, offset])
        self.onnx_and_test(graph_def)

    def test_Resize(self, case_name):
        case0 = [[1, 16, 32, 32], [1, 16, 64, 64], [1, 1, 2, 2], [4], 'nearest', 'asymmetric']
        case1 = [[2, 3, 224], [2, 3, 448], [1, 1, 2], [3], 'linear', 'half_pixel']
        cases = (case0, case1) if self.chip in ['bm1684x', 'bm1688', 'sg2380'] else (case0, )
        for idx, case in enumerate(cases):
            input_shape, output_shape, scales, dim, mode, coor_mode = case
            graph_txt = """
                %s (float%s input) => (float%s output)
                <float[0] roi, float%s scales>
                {
                    output = Resize<mode="%s",nearest_mode="floor", coordinate_transformation_mode="%s">(input, roi, scales)
                }
                """ % (case_name, input_shape, output_shape, dim, mode, coor_mode)
            graph_def = onnx.parser.parse_graph(graph_txt)

            roi_data = np.array([], dtype=np.float32)
            scales_data = np.array(scales, dtype=np.float32)
            roi = helper.make_tensor('roi', TensorProto.FLOAT, [0], roi_data)
            scales = helper.make_tensor('scales', TensorProto.FLOAT, dim, scales_data)
            graph_def.initializer.extend([roi, scales])
            self.onnx_and_test(graph_def)

    def test_Resize2(self, case_name):
        input_shape = [1, 32, 208, 30]
        # linear
        output_shape1 = [1, 32, 416, 48]  # by cpu, scale is not integer
        output_shape2 = [1, 32, 416, 90]  # by npu, scale is integer
        output_shape3 = [1, 32, 104, 15]  # by npu, scale is 0.5
        # nearest
        output_shape4 = [1, 32, 416, 60]  # by npu, scale is integer
        output_shape5 = [1, 32, 416, 20]  # by cpu
        roi = np.array([], dtype=np.float32)
        scales = np.array([], dtype=np.float32)

        graph_txt = """
            %s (float%s input) => (float%s output1, float%s output2, float%s output3, float%s output4, float%s output5)
            <float%s roi, float%s scales, int64[4] sizes1=%s, int64[4] sizes2=%s, int64[4] sizes3=%s, int64[4] sizes4=%s, int64[4] sizes5=%s>
            {
                X1 = Neg(input)
                output1 = Resize<mode="linear",coordinate_transformation_mode="half_pixel">(X1, roi, scales, sizes1)
                output2 = Resize<mode="linear",coordinate_transformation_mode="pytorch_half_pixel">(input, roi, scales, sizes2)
                output3 = Resize<mode="linear",coordinate_transformation_mode="half_pixel">(input, roi, scales, sizes3)
                output4 = Resize<mode="nearest">(input, roi, scales, sizes4)
                output5 = Resize<mode="nearest">(input, roi, scales, sizes5)
            }
            """ % (case_name, input_shape, output_shape1, output_shape2, output_shape3, output_shape4,
                   output_shape5, list(roi.shape), list(scales.shape), str(output_shape1).replace(
                       '[', '{').replace(']', '}'), str(output_shape2).replace('[', '{').replace(
                           ']', '}'), str(output_shape3).replace('[', '{').replace(
                               ']', '}'), str(output_shape4).replace('[', '{').replace(']', '}'),
                   str(output_shape5).replace('[', '{').replace(']', '}'))
        roi_def = onnx.helper.make_tensor(name='roi',
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=roi.shape,
                                          vals=roi.flatten())
        scales_def = onnx.helper.make_tensor(name='scales',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=scales.shape,
                                             vals=scales.flatten())
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([roi_def, scales_def])
        self.onnx_and_test(graph_def)

    def test_Flatten(self, case_name):
        input_shape = [2, 3, 4, 5]
        output_shape = [6, 20]
        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                x = Flatten<axis=2>(input)
                output = Softmax<axis=-1>(x)
            }
            """ % (case_name, input_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Flip(self, case_name):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):

                axes = 5
                y1 = x.flip(dims=[axes])

                # reversed_indices = [i for i in range(x.size(axes) - 1, -1, -1)]
                # y = x.index_select(axes, torch.tensor(reversed_indices))

                return y1

        x = torch.randn([1, 4, 3, 32, 2, 6])
        self.torch_and_test((x), Model(), case_name)

    def test_Reshape(self, case_name):
        input_shape = [1, 16, 32, 32]
        right_shape = [3]
        output_shape = [1, 16, 1024]
        right_data = np.array([1, 16, 1024], dtype=np.int64)
        graph_txt = """
            %s (float%s input) => (float%s output)
            <int64%s right>
            {
                output = %s(input, right)
            }
            """ % (case_name, input_shape, output_shape, right_shape, case_name)
        graph_def = onnx.parser.parse_graph(graph_txt)

        right = helper.make_tensor('right', TensorProto.INT64, right_shape, right_data)
        graph_def.initializer.extend([right])
        self.onnx_and_test(graph_def)

    def test_Reshape2(self, case_name):
        # [3, 6] -> [2, 9]
        # onnx vs Top compare failed due to dynamic shape not support in Top
        # Tpu vs model compare failed due to dynamic shape not support in Tpu
        # onnx vs model compare success
        input_shape = [3, 6]
        graph_txt = """
            %s (float%s input) => (int64 output)
            <int64 indices0 = {0}, int64 indices = {0}, int64 dim1 = {-1}>
            {
                gather0 = Gather<axis=0>(input,indices0)
                nonzero = NonZero(gather0)
                transpose = Transpose<perm=[1, 0]>(nonzero)
                shape = Shape(transpose)
                dim0 = Gather<axis=0>(shape, indices)
                new_shape = Concat<axis=0>(dim0, dim1)
                output_tmp = Reshape(input, new_shape)
                output = Shape(output_tmp)
            }
            """ % (case_name, input_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)

        indices0 = helper.make_tensor('indices0', TensorProto.INT64, [], vals=[0])
        indices = helper.make_tensor('indices', TensorProto.INT64, [1], vals=[0])
        dim1 = helper.make_tensor('dim1', TensorProto.INT64, [1], vals=[-1])
        graph_def.initializer.extend([indices, dim1, indices0])
        input_data = {
            'input':
            np.array([[1, 0, 3, 0, 0, 0], [4, 5, 6, 3, 2, 1], [7, 8, 9, 2, 1, 1]], dtype=np.float32)
        }
        self.onnx_and_test(graph_def, static_shape=False, input_data=input_data)

    def test_Floor(self, case_name):
        input_shape = [1, 128, 32, 32]
        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = %s(input)
            }
            """ % (case_name, input_shape, input_shape, case_name)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Softmax(self, case_name):
        input_shapes = [[3, 100, 1, 1], [3, 100, 32], [3, 100, 32, 1]]
        axiss = [1, 2]
        for input_shape in input_shapes:
            for axis in axiss:
                graph_txt = """
                    %s (float%s input) => (float%s output)
                    {
                        output = %s<axis=%d>(input)
                    }
                    """ % (case_name, input_shape, input_shape, case_name, axis)
                graph_def = onnx.parser.parse_graph(graph_txt)
                self.onnx_and_test(graph_def)

    def test_LogSoftmax(self, case_name):
        input_shape = [3, 100, 32]
        axis = 2
        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = %s<axis=%d>(input)
            }
            """ % (case_name, input_shape, input_shape, case_name, axis)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Softplus(self, case_name):
        input_shape = [200]
        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = %s(input)
            }
            """ % (case_name, input_shape, input_shape, case_name)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Exp(self, case_name):
        input_shape = [1, 3, 32, 32]
        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = %s(input)
            }
            """ % (case_name, input_shape, input_shape, case_name)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Tanh(self, case_name):
        input_shape = [1, 3, 32, 32]
        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = Tanh(input)
            }
            """ % (case_name, input_shape, input_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Arctan(self, case_name):
        input_shape = [1, 3, 32, 32]
        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = %s(input)
            }
            """ % (case_name, input_shape, input_shape, case_name)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Arctanh(self, case_name):
        input_shape = [1, 3, 32, 32]
        # The value range of arctanh is (-1,1)
        input_data = np.clip(np.random.randn(*input_shape).astype(np.float32), -0.99, 0.99)
        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = %s(input)
            }
            """ % (case_name, input_shape, input_shape, case_name)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def, input_data={'input': input_data})

    def test_Arccos(self, case_name):
        input_shape = [1, 3, 32, 32]
        # The value range of arccos is (-1,1)
        input_data = np.clip(np.random.randn(*input_shape).astype(np.float32), -0.99, 0.99)
        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = %s(input)
            }
            """ % (case_name, input_shape, input_shape, case_name)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def, input_data={'input': input_data})

    def test_Log(self, case_name):
        input_shape = [1, 3, 32, 32]
        input_data = np.clip(np.random.randn(*input_shape).astype(np.float32) * 10.0, 0.5, 8)
        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = %s(input)
            }
            """ % (case_name, input_shape, input_shape, case_name)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def, input_data={'input': input_data})

    def test_Neg(self, case_name):
        input_shape = [4, 16, 27, 27]
        output_shape = [4, 16, 27, 27]

        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = Neg(input)
            }
            """ % (case_name, input_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Nms2(self, case_name):
        # params for Nms
        num_batches = 1
        num_classes = 80
        spatial_dimension = 50  # 15200
        in_shape = [num_batches, spatial_dimension, 4]
        score_shape = [num_batches, num_classes, spatial_dimension]
        nonzero_input_shape = [2, 6]
        graph_txt = """
            %s (float%s nonzero_input, float%s boxes, float%s scores) => (int64 Y_Value)
            <float[1] iou_threshold = {0.5}, float[1] score_threshold = {0.05}, int64 indices0 = {0}, int64 indices1 = {0}>
            {
                gather0 = Gather<axis=0>(nonzero_input, indices0)
                nonzero = NonZero(gather0)
                transpose = Transpose<perm=[1, 0]>(nonzero)
                shape = Shape(transpose)
                max_output_boxes_per_class = Gather<axis=0>(shape,indices1)
                Y_Value = NonMaxSuppression(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
            }
            """ % (case_name, nonzero_input_shape, in_shape, score_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        input_data = {
            'nonzero_input': np.array([[1, 0, 3, 0, 4, 0], [2.1, 2.5, 0, 0, 2.6, 0]],
                                      dtype=np.float32),
            'boxes': np.random.rand(*in_shape).astype(np.float32),
            'scores': np.random.rand(*score_shape).astype(np.float32)
        }
        self.onnx_and_test_bmodel(graph_def,
                                  static_shape=False,
                                  input_data=input_data,
                                  only_cmp_with_bmodel=True)

    def test_Nms(self, case_name):
        num_batches = 1
        num_classes = 80
        spatial_dimension = 15200
        max_out = 200
        in_shape = [num_batches, spatial_dimension, 4]
        score_shape = [num_batches, num_classes, spatial_dimension]
        y_shape = [max_out * num_classes, 3]

        graph_txt = """
            %s (float%s boxes, float%s scores) => (int64 selected_indices)
            <int64 max_output_boxes_per_class = {200}, float iou_threshold = {0.5}, float score_threshold = {0.05}>
            {
                selected_indices = NonMaxSuppression(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
            }
            """ % (case_name, in_shape, score_shape, y_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Pad(self, case_name):
        case0 = [[3, 8, 32, 32], [3, 8, 44, 46], [0, 0, 5, 6, 0, 0, 7, 8]]
        case1 = [[4, 8, 10], [4, 9, 11], [0, 1, 0, 0, 0, 1]]
        case2 = [[1, 1, 160, 160, 96], [1, 1, 161, 161, 96], [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]]
        cases = (case0, case1, case2)
        for idx, case in enumerate(cases):
            pads = np.array(case[2]).astype(np.int64)
            graph_txt = """
                %s_%s (float%s input) => (float%s output)
                <int64%s pads>
                {
                    output = Pad(input, pads)
                }
                """ % (case_name, idx, case[0], case[1], list(pads.shape))
            graph_def = onnx.parser.parse_graph(graph_txt)
            pad_val = helper.make_tensor(name='pads',
                                         data_type=onnx.TensorProto.INT64,
                                         dims=pads.shape,
                                         vals=pads.flatten())
            graph_def.initializer.extend([pad_val])
            self.onnx_and_test(graph_def)

    def test_Pad1(self, case_name):
        input_shape = [3, 8, 32, 32]
        output_shape = [3, 8, 44, 46]
        pads = np.array([0, 0, 5, 6, 0, 0, 7, 8]).astype(np.int64)
        graph_txt = """
                %s (float%s input) => (float%s output)
                <int64%s pads, float pad_val>
                {
                    output = Pad(input, pads, pad_val)
                }
                """ % (case_name, input_shape, output_shape, list(pads.shape))
        graph_def = onnx.parser.parse_graph(graph_txt)

        pad_shape = helper.make_tensor(name='pads',
                                       data_type=onnx.TensorProto.INT64,
                                       dims=pads.shape,
                                       vals=pads.flatten())
        pad_val = helper.make_tensor(name='pad_val',
                                     data_type=onnx.TensorProto.FLOAT,
                                     dims=[],
                                     vals=[0.6])
        graph_def.initializer.extend([pad_shape, pad_val])
        self.onnx_and_test(graph_def)

    def test_PadEdge(self, case_name):
        input_shape = [3, 8, 32, 32]
        output_shape = [3, 8, 44, 46]
        pads = np.array([0, 0, 5, 6, 0, 0, 7, 8]).astype(np.int64)
        graph_txt = """
                %s (float%s input) => (float%s output)
                <int64%s pads>
                {
                    output = Pad<mode="edge">(input, pads)
                }
                """ % (case_name, input_shape, output_shape, list(pads.shape))
        graph_def = onnx.parser.parse_graph(graph_txt)

        pad_val = helper.make_tensor(name='pads',
                                     data_type=onnx.TensorProto.INT64,
                                     dims=pads.shape,
                                     vals=pads.flatten())
        graph_def.initializer.extend([pad_val])
        self.onnx_and_test(graph_def)

    def test_PadReflect(self, case_name):
        input_shape = [3, 8, 32, 32]
        output_shape = [3, 8, 44, 46]
        pads = np.array([0, 0, 5, 6, 0, 0, 7, 8]).astype(np.int64)

        graph_txt = """
                %s (float%s input) => (float%s output)
                <int64%s pads>
                {
                    output = Pad<mode="reflect">(input, pads)
                }
                """ % (case_name, input_shape, output_shape, list(pads.shape))
        graph_def = onnx.parser.parse_graph(graph_txt)

        pad_val = helper.make_tensor(name='pads',
                                     data_type=onnx.TensorProto.INT64,
                                     dims=pads.shape,
                                     vals=pads.flatten())
        graph_def.initializer.extend([pad_val])
        self.onnx_and_test(graph_def)

    def test_DepthToSpace(self, case_name):
        in_shape = [1, 32, 108, 192]
        n, c, h, w = in_shape
        blocksize = 2
        # mode='CRD'
        mode = 'DCR'  # default
        out_shape = [n, c // (blocksize * blocksize), h * blocksize, w * blocksize]
        graph_txt = """
                %s (float%s input) => (float%s output)
                {
                    output = DepthToSpace<mode="%s", blocksize=%d>(input)
                }
                """ % (case_name, in_shape, out_shape, mode, blocksize)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Div(self, case_name):
        input_shape = {"input1": [1, 3, 27, 27], "input2": [1, 3, 27, 27]}
        output_shape = [1, 3, 27, 27]
        input_data = {k: np.random.randn(*x).astype(np.float32) for k, x in input_shape.items()}
        input_data["input2"] = np.clip(input_data["input2"], 0.01, 10)

        graph_txt = """
                %s (float[1, 3, 27, 27] input1,float[1, 3, 27, 27] input2) => (float%s output)
                {
                    output = Div(input1, input2)
                }
                """ % (case_name, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
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
                graph_txt = """
                    %s_%s_%s (float%s a, float%s b, float%s c) => (float%s output)
                    {
                        x = Div(a, b)
                        output = Div(x, c)
                    }
                    """ % (case_name, i, "".join(map(str, dims)), bcast_s, s, bcast_s, s)
                graph_def = onnx.parser.parse_graph(graph_txt)
                self.onnx_and_test(graph_def, input_data={"a": a_data, "b": b_data, "c": c_data})

    def test_DivBcast2(self, case_name):
        shapes = ([5, 12, 1, 21], )
        bcast_shapes = ([1, 12, 9, 21], )
        out_shapes = ([5, 12, 9, 21], )
        for i, s in enumerate(shapes):
            a_data = np.random.randn(*bcast_shapes[i]).astype(np.float32)
            b_data = np.clip(np.random.randn(*s).astype(np.float32), 0.01, 10)
            c_data = np.clip(np.random.randn(*bcast_shapes[i]).astype(np.float32), 0.01, 10)
            graph_txt = """
                %s_%s (float%s a, float%s b, float%s c) => (float%s output)
                {
                    x = Div(a, b)
                    output = Div(x, c)
                }
                """ % (case_name, i, bcast_shapes[i], s, bcast_shapes[i], out_shapes[i])
            graph_def = onnx.parser.parse_graph(graph_txt)
            self.onnx_and_test(graph_def, input_data={"a": a_data, "b": b_data, "c": c_data})

    def test_ConvTrans(self, case_name):
        oc, ic = 16, 8
        ih, iw = 16, 16
        kernel_shape = [3, 3]
        pads = [1, 1, 1, 1]
        strides = [2, 2]
        dilations = [1, 1]
        output_padding = [1, 1]
        oh = (ih - 1) * strides[0] + output_padding[0] + (kernel_shape[0] -
                                                          1) * dilations[0] + 1 - pads[0] - pads[2]
        ow = (iw - 1) * strides[1] + output_padding[0] + (kernel_shape[1] -
                                                          1) * dilations[1] + 1 - pads[1] - pads[3]
        input_shape = [1, ic, ih, iw]
        output_shape = [1, oc, oh, ow]
        filter_shape = [ic, oc, kernel_shape[0], kernel_shape[1]]

        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        bias_data = np.random.randn(oc).astype(np.float32)

        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s weight, float%s bias>
            {
                output = ConvTranspose<kernel_shape=%s, pads=%s, output_padding=%s,
                strides=%s, dilations=%s, group=1>(input, weight, bias)
            }
            """ % (case_name, input_shape, output_shape, filter_shape, list(
            bias_data.shape), kernel_shape, pads, output_padding, strides, dilations)
        graph_def = onnx.parser.parse_graph(graph_txt)

        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, list(bias_data.shape), bias_data)
        graph_def.initializer.extend([weight, bias])
        self.onnx_and_test(graph_def)

    def test_ConvTrans2(self, case_name):
        input_shape = [1, 64, 35, 35]
        filter_shape = [64, 32, 2, 2]
        bias_shape = [32]
        output_shape = [1, 32, 71, 71]
        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        bias_data = np.random.randn(*bias_shape).astype(np.float32)

        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s weight, float%s bias>
            {
                output = ConvTranspose<kernel_shape=[2, 2], pads=[0, 0, 0, 0], output_padding=[1, 1],
                strides=[2, 2], dilations=[1, 1], group=1>(input, weight, bias)
            }
            """ % (case_name, input_shape, output_shape, filter_shape, bias_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)

        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, bias_shape, bias_data)
        graph_def.initializer.extend([weight, bias])
        self.onnx_and_test(graph_def)

    def test_Squeeze(self, case_name):
        axis = [1, 3]
        input_shape = [3, 1, 32, 1]
        output_shape = [input_shape[i] for i in range(len(input_shape)) if i not in axis]

        graph_txt = """
            %s (float%s input0, float%s input1) => (float%s output_1, float%s output_2)
            <int64%s axes>
            {
                x = Add(input0, input1)
                output_1 = Squeeze(x, axes)
                output_2 = Squeeze(x)
            }
            """ % (case_name, input_shape, input_shape, output_shape, output_shape, [len(axis)])
        graph_def = onnx.parser.parse_graph(graph_txt)
        axes = helper.make_tensor('axes', TensorProto.INT64, [len(axis)],
                                  axis * np.ones(1).astype(int))
        graph_def.initializer.extend([axes])
        self.onnx_and_test(graph_def)

    def test_Clip(self, case_name):
        input_shape = [1, 3, 32, 32]
        graph_txt = """
            %s (float%s input) => (float%s output)
            <float min = {-1.0}, float max = {6.0}>
            {
                output = %s(input, min, max)
            }
            """ % (case_name, input_shape, input_shape, case_name)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_ReduceL2(self, case_name):
        input_shape = [4, 4, 4, 16, 16, 64]
        output_shape = [4, 4, 4, 64]

        graph_txt = """
            %s (float%s input) => (float%s o_l2)
            {
                o_l2 = ReduceL2<keepdims=0, axes=[3, 4]>(input)
            }
            """ % (case_name, input_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_ReduceL1(self, case_name):
        input_shape = [4, 4, 4, 16, 16, 64]
        output_shape = [4, 4, 4, 64]

        graph_txt = """
            %s (float%s input) => (float%s o_l1)
            {
                o_l1 = ReduceL1<keepdims=0, axes=[3, 4]>(input)
            }
            """ % (case_name, input_shape, output_shape)

        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_ReduceSum(self, case_name):
        input_shape = [4, 4, 4, 16, 16, 64]
        output_shape = [4, 4, 4, 16, 64]

        graph_txt = """
            %s (float%s input) => (float%s o_sum)
            <int64[1] axes = {4}>
            {
                o_sum = ReduceSum<keepdims=0>(input, axes)
            }
            """ % (case_name, input_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_ReduceProd(self, case_name):
        input_shape = [1, 3, 128, 128]
        output_shape = [1, 3, 128]

        graph_txt = """
            %s (float%s input) => (float%s o_prod)
            {
                o_prod = ReduceProd<keepdims=0, axes=[2]>(input)
            }
            """ % (case_name, input_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Sigmoid(self, case_name):
        input_shape = [1, 16, 64, 64]
        output_shape = [1, 16, 64, 64]

        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = %s(input)
            }
            """ % (case_name, input_shape, output_shape, case_name)
        graph_def = onnx.parser.parse_graph(graph_txt)
        if 0:  # this code to test local layer, but when model use local layer, compare will not pass.
            graph_txt = """
                %s (float%s input, float%s a) => (float%s add_out)
                {
                    output = %s(input)
                    add_out = Add(a, output)
                }
                """ % (case_name, input_shape, input_shape, input_shape, case_name)
            graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Sign(self, case_name):
        input_shape = [4, 2, 200, 100]
        output_shape = [4, 2, 200, 100]

        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = %s(input)
            }
            """ % (case_name, input_shape, output_shape, case_name)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Slice(self, case_name):
        input_shape = [5, 116, 64, 64]
        output_shape = [3, 16, 16, 8]
        starts_data = np.array([2, 10, 10, 12], dtype=np.int64)
        ends_data = np.array([5, 42, 42, 36], dtype=np.int64)
        axes_data = np.array([0, 1, 2, 3], dtype=np.int64)
        steps_data = np.array([1, 3, 2, 3], dtype=np.int64)

        graph_txt = """
            %s (float%s input) => (float%s output)
            <int64[4] starts, int64[4] ends, int64[4] axes, int64[4] steps>
            {
                output = Slice(input, starts, ends, axes, steps)
            }
            """ % (case_name, input_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)

        starts = helper.make_tensor('starts', TensorProto.INT64, [4], starts_data)
        ends = helper.make_tensor('ends', TensorProto.INT64, [4], ends_data)
        axes = helper.make_tensor('axes', TensorProto.INT64, [4], axes_data)
        steps = helper.make_tensor('steps', TensorProto.INT64, [4], steps_data)
        graph_def.initializer.extend([starts, ends, axes, steps])
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

    def test_Dynamic_Slice(self, case_name):
        if not self.dynamic:
            pass
        else:

            class Model(nn.Module):

                def __init__(self):
                    super(Model, self).__init__()

                def forward(self, input, a, b, c, d, e):
                    a = a.to(torch.int64)
                    b = b.to(torch.int64)
                    c = c.to(torch.int64)
                    d = d.to(torch.int64)
                    e = e.to(torch.int64)
                    res = input[a[0]:b[0], b[0]:c[0], d[0] - a[0]:, 3:e[0]]
                    return res

            shape1 = (4, 6, 224, 224)
            input = torch.randn(shape1, dtype=torch.float32)
            a = torch.tensor([1.0])
            b = torch.tensor([3.0])
            c = torch.tensor([4.0])
            d = torch.tensor([66.0])
            e = torch.tensor([200.0])
            self.torch_and_test((input, a, b, c, d, e), Model(), case_name)

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
                self.deconv = nn.ConvTranspose2d(in_channels=128,
                                                 out_channels=128,
                                                 kernel_size=4,
                                                 stride=2,
                                                 padding=0,
                                                 output_padding=0,
                                                 groups=128,
                                                 bias=False,
                                                 dilation=1)

            def forward(self, x):
                y = self.deconv(x)
                return y

        x = torch.randn(1, 128, 24, 44).float()
        self.torch_and_test(x, Model(), case_name)

    def test_DeconvDynW(self, case_name):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.deconv = nn.ConvTranspose2d(in_channels=8,
                                                 out_channels=8,
                                                 kernel_size=2,
                                                 stride=2,
                                                 padding=0,
                                                 output_padding=0,
                                                 groups=1,
                                                 bias=False,
                                                 dilation=1)

            def forward(self, x, y):
                output_padding = self.deconv._output_padding(
                    x,
                    None,
                    self.deconv.stride,
                    self.deconv.padding,
                    self.deconv.kernel_size,  # type: ignore[arg-type]
                    2,
                    self.deconv.dilation)  # type: ignore[arg-type]

                out = F.conv_transpose2d(x, y, None, [2, 2], 0, output_padding, 1, 1)
                return out

        x = torch.randn(3, 8, 16, 32).float()
        y = torch.randn(8, 8, 2, 2).float()
        self.torch_and_test((x, y), Model(), case_name)

    def test_Deconv2(self, case_name):
        groups, kernel = 4, 4

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.filter = torch.arange(0, groups * kernel * kernel,
                                           dtype=torch.float32).reshape(groups, 1, kernel, kernel)

            def forward(self, x):
                y = F.conv_transpose2d(x,
                                       self.filter,
                                       padding=1,
                                       stride=2,
                                       dilation=1,
                                       groups=groups)
                return y

        x = torch.randn(1, groups, 32, 32).float()
        self.torch_and_test(x, Model(), case_name)

    def test_Deconv3d(self, case_name):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.deconv = nn.ConvTranspose3d(in_channels=4,
                                                 out_channels=12,
                                                 kernel_size=2,
                                                 stride=2,
                                                 padding=0,
                                                 output_padding=0,
                                                 groups=2,
                                                 bias=True,
                                                 dilation=1)

            def forward(self, x):
                y = self.deconv(x)
                return y

        x = torch.randn(3, 4, 8, 8, 8).float()
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

        graph_txt = """
            %s (float%s input) => (float%s output_1, float%s output_2)
            <int64[2] split>
            {
                output_1, output_2 = Split<axis=0>(input, split)
            }
            """ % (case_name, input_shape, output1_shape, output2_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)

        split = helper.make_tensor('split', TensorProto.INT64, [2], split_data)
        graph_def.initializer.extend([split])
        self.onnx_and_test(graph_def)

    def test_Split2(self, case_name):
        input_shape = [6, 116, 64, 64]
        output1_shape = [3, 116, 64, 64]
        output2_shape = [3, 116, 64, 64]
        split_data = np.array([3, 3], dtype=np.int64)

        graph_txt = """
            %s (float%s input) => (float%s output_1, float%s output_2)
            {
                output_1, output_2 = Split<axis=0>(input)
            }
            """ % (case_name, input_shape, output1_shape, output2_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Arg(self, case_name):
        for keep in [True, False]:
            input_shape = [20, 40, 60]
            output_shape = [20, 1, 60] if keep else [20, 60]
            if self.is_cv18xx:
                graph_txt = """
                %s_%s (float%s input) => (int64%s o_max)
                {
                    o_max = ArgMax<keepdims=%s, axis=1, select_last_index=1>(input)
                }
                """ % (case_name, keep, input_shape, output_shape, 1 if keep else 0)
                graph_def = onnx.parser.parse_graph(graph_txt)
                self.onnx_and_test(graph_def)
                continue

            graph_txt = """
                %s_%s (float%s input) => (int64%s o_max, int64%s o_min)
                {
                    o_max = ArgMax<keepdims=%s, axis=1, select_last_index=1>(input)
                    o_min = ArgMin<keepdims=%s, axis=1, select_last_index=1>(input)
                }
                """ % (case_name, keep, input_shape, output_shape, output_shape, 1 if keep else 0,
                       1 if keep else 0)
            graph_def = onnx.parser.parse_graph(graph_txt)
            self.onnx_and_test(graph_def)

    def test_Reduce(self, case_name):
        for keep in [True, False]:
            input_shape = [4, 4, 4, 16, 64]
            output_shape = [4, 4, 4, 16, 1] if keep else [4, 4, 4, 16]

            graph_txt = """
                %s_%s (float%s input) => (float%s o_mean, float%s o_max, float%s o_min)
                {
                    o_mean = ReduceMean<keepdims=%d, axes=[4]>(input)
                    o_max = ReduceMax<keepdims=%d, axes=[4]>(input)
                    o_min = ReduceMin<keepdims=%d, axes=[4]>(input)
                }
                """ % (case_name, keep, input_shape, output_shape, output_shape, output_shape, keep, keep,
                       keep)
            graph_def = onnx.parser.parse_graph(graph_txt)
            self.onnx_and_test(graph_def)

    def test_Reduce2(self, case_name):
        for keep in [True, False]:
            input_shape = [4, 4, 4, 16, 64]
            output_shape = [4, 4, 1, 1, 64] if keep else [4, 4, 64]

            graph_txt = """
                %s_%s (float%s input) => (float%s o_mean, float%s o_max, float%s o_min)
                {
                    o_mean = ReduceMean<keepdims=%d, axes=[2, 3]>(input)
                    o_max = ReduceMax<keepdims=%d, axes=[2, 3]>(input)
                    o_min = ReduceMin<keepdims=%d, axes=[2, 3]>(input)
                }
                """ % (case_name, keep, input_shape, output_shape, output_shape, output_shape, keep, keep,
                       keep)
            graph_def = onnx.parser.parse_graph(graph_txt)
            self.onnx_and_test(graph_def)

    def test_ReduceMean(self, case_name):
        input_shape = [2, 200, 7, 7]
        output_shape = [2, 1, 1, 7]

        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = ReduceMean<keepdims=1, axes=[1, 2]>(input)
            }
            """ % (case_name, input_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
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

    def test_TorchNormalize(self, case_name):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                y = nn.functional.normalize(x,p=3.0, dim=[1,2])
                return y

        # x = torch.randn(1, 3, 224, 224).float()

        np.random.seed(10)
        t = np.clip(np.random.randn(1,3,224,224).astype('float16'), a_min=-10, a_max=10)
        x = torch.tensor(t).float()

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

            def __init__(self, C):
                super(Model, self).__init__()
                self.instance_norm = nn.InstanceNorm2d(C, affine=True)
                nn.init.normal_(self.instance_norm.weight, std=0.01)
                nn.init.normal_(self.instance_norm.bias, std=0.01)

            def forward(self, x):
                return self.instance_norm(x + 1)

        x = torch.randn(2, 8, 15, 15).float()
        self.torch_and_test(x, Model(x.shape[1]), case_name)
        x = torch.randn(4, 1, 1, 100).float()
        self.torch_and_test(x, Model(x.shape[1]), case_name)

    def test_TorchGroupNorm(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self, C, num_groups):
                super(Model, self).__init__()
                self.group_norm = nn.GroupNorm(num_groups, C)
                nn.init.normal_(self.group_norm.weight, std=0.01)
                nn.init.normal_(self.group_norm.bias, std=0.01)

            def forward(self, x):
                return self.group_norm(x)

        x = torch.randn(20, 6, 10, 10).float()
        self.torch_and_test(x, Model(x.shape[1], 3), case_name)
        if self.chip in ["bm1684x", "bm1690", "bm1688", "sg2380"]:
            x = torch.randn(2, 129, 4, 4).float()
            self.torch_and_test(x, Model(x.shape[1], 43), case_name)

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

    def test_TorchRMSNorm(self, case_name):
        normalize_shape = [4096]
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

                return self.prelu(rmsnorm_out)

        input_shape = [1, 513] + normalize_shape
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

    def test_Depth2SpaceWithPermute(self, case_name):
        input_shape = [1, 108, 192, 32]
        transpose_order = [0, 3, 1, 2]
        block_size = 2
        mode = 'DCR'
        permute_output_shape = [
            input_shape[transpose_order[j]] for j in range(len(transpose_order))
        ]
        depth2space_input_shape = permute_output_shape
        depth2space_output_shape = [
            depth2space_input_shape[0], depth2space_input_shape[1] // (block_size * block_size),
            depth2space_input_shape[2] * block_size, depth2space_input_shape[3] * block_size
        ]

        graph_txt = """
            %s (float%s input) => (float%s depth2space_output)
            {
                permute_output = Transpose<perm=%s>(input)
                depth2space_output = DepthToSpace<blocksize=%d, mode="%s">(permute_output)
            }
            """ % (case_name, input_shape, depth2space_output_shape, transpose_order, block_size, mode)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)


    def test_FitPermute2Hdim(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                # self.bias = torch.randn(96).float()
                self.conv2d = nn.Conv2d(in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=(1, 1),
                        padding=1,
                        dilation=1)
            def forward(self, x, x2):
                x = x[2:3,:,:,64:128]

                x = torch.reshape(x,[1,14,2,1,28,64])
                x = torch.permute(x,[0,1,3,5,2,4])
                x = torch.reshape(x,[14,64,2,28])
                y1 = self.conv2d(x)
                y1 = torch.reshape(y1,[14,2,32,56])
                y1 = torch.permute(y1,[0,1,3,2])

                x = torch.reshape(x,[14,2,32,56])
                x = torch.permute(x,[0,1,3,2])
                x2 = torch.permute(x2,[0,2,1,3])
                y2 = torch.matmul(x2, x)
                return y1 + y2

        x = torch.randn(3, 1, 784,128).float()
        x2 = torch.randn(14, 56, 2,56).float()
        self.torch_and_test((x,x2), Model(), case_name)

    def test_Gather2Slice(self, case_name):

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

    def test_GatherUnsqueeze(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                s = x[1]
                return s.unsqueeze(0)

        for i, s in enumerate([3, [2, 3, 2, 4]]):
            x = torch.randn(s).float()
            self.torch_and_test(x, Model(), case_name + str(i), static_shape=False)

    def test_Gather2Slice2(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                a = x[..., 0]
                b = x[..., 1]
                c = torch.matmul(a, b)
                return c

        x = torch.randn(32, 128, 128, 2).float()
        self.torch_and_test(x, Model(), case_name)

    def test_GLMTilePermute(self, case_name):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y):
                y = y.unsqueeze(3).repeat(1, 1, 1, 2, 1)
                y = y.reshape(13, 4, 8).transpose(1, 0)
                res = torch.matmul(x, y)
                return res

        x = 0.1 * torch.randn(4, 1, 13).float()
        y = 0.1 * torch.randn(13, 1, 2, 8)
        self.torch_and_test((x, y), Model(), case_name)

    def test_GLMTilePermute2(self, case_name):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y):
                y = y.unsqueeze(3).repeat(1, 1, 1, 16, 1)
                y = y.reshape(512, 64, 128).transpose(1, 0)
                res = torch.matmul(x, y)
                return res

        x = 0.1 * torch.randn(64, 512, 512).float()
        y = 0.1 * torch.randn(512, 2, 2, 128)
        self.torch_and_test((x, y), Model(), case_name)

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

    def test_ReshapeN(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.w0 = torch.randn(1, 1, 3, 49, 49).float()
                self.w1 = torch.randn(1, 3, 49, 49).float()

            def forward(self, x):
                a = x + self.w0
                b = torch.reshape(a, [529, 3, 49, 49])
                c = b - self.w1
                return c

        x = torch.randn(1, 529, 3, 49, 49).float()
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
                # tanh
                x = self.linear(input)
                y0 = torch.tanh(x)
                # relu
                x = self.linear(input)
                y2 = torch.relu(x)
                # sigmoid
                x = self.linear(input)
                y1 = torch.sigmoid(x)
                # leaky_relu
                # x = self.linear(input)
                # y3 = F.leaky_relu(x)
                # elu
                x = self.linear(input)
                y4 = F.elu(x)
                # softplus
                x = self.linear(input)
                y5 = self.prelu(x)
                # hardsigmoid
                x = self.linear(input)
                y6 = self.hardsigmoid(x)
                # prelu
                x = self.linear(input)
                y7 = self.softplus(x)
                # relu6
                x = self.linear(input)
                y8 = self.ReLU6(x)
                # mish
                x = self.linear(input)
                y9 = self.mish(x)
                # Softmax
                x = self.linear(input)
                y10 = self.Softmax(x)
                # concat
                y = torch.cat((y0, y1, y2, y4, y5, y6, y7, y8, y9, y10), 0)
                # Softmax_2d
                x = self.linear(input)
                x = x.unsqueeze(dim=1)
                y2 = self.Softmax_2d(x)
                return y

        x = torch.randn(3, 100, 100).float()
        self.torch_and_test(x, Model(), case_name)

    def test_TorchArg(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                a = torch.argmax(x, -1)
                b = torch.argmin(x, -1)
                return a, b

        # generate data with duplicate values
        x = np.random.randint(-666, 666, size=(2, 32, 40, 80)).astype(np.float32)
        x = torch.from_numpy(x)
        self.torch_and_test(x, Model(), case_name)

    def test_TorchBatchNorm(self, case_name):

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.bm = nn.BatchNorm2d(3, affine=True, eps=9.9999997473787516E-6)
                if self.bm.weight is not None:
                    torch.nn.init.uniform_(self.bm.weight.data)
                if self.bm.bias is not None:
                    torch.nn.init.uniform_(self.bm.bias.data)

            def forward(self, x):
                y = self.bm(x)

                return y

        # generate data with duplicate values
        x = np.random.randint(-666, 666, size=(32, 3, 112, 112)).astype(np.float32)
        x = torch.from_numpy(x)
        self.torch_and_test(x, Model(), case_name)

    def test_TorchZeroPad(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                # left:3 right:4 up:4 down:6 postion pad zero
                self.ZeroPad2d = nn.ZeroPad2d(padding=(3, 4, 5, 6))

            def forward(self, x):
                # input shape = (3, 100, 100)
                x = self.ZeroPad2d(x)  # output shape = (3, 111, 107)
                return x

        x = torch.randn(4, 3, 100, 100).float()
        self.torch_and_test(x, Model(), case_name)

    def test_TorchScatterND(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y):
                x[:, ::3, 2::2, ::1] = y
                return x

        x = torch.randn(2, 51, 40, 2).float()
        y = torch.randn(2, 17, 19, 1).float()
        self.torch_and_test((x, y), Model(), case_name)

    def test_TorchConv2d(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.conv2d = nn.Conv2d(in_channels=160,
                                        out_channels=128,
                                        kernel_size=3,
                                        stride=(2, 1),
                                        padding=16,
                                        dilation=16)

            def forward(self, x):
                return self.conv2d(x)

        x = torch.randn(1, 160, 20, 30).float()
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
            graph_txt = """
                %s (float%s a, float%s b) => (float%s output)
                {
                    output = Add(a, b)
                }
                """ % (case_name, s, s, s)
            graph_def = onnx.parser.parse_graph(graph_txt)
            if 0:  # the follow code used to test local layer, but local layer compare faild. so I just closed it, and commit it.
                c = helper.make_tensor_value_info("c", TensorProto.FLOAT, s)
                add2_out = helper.make_tensor_value_info("add2_out", TensorProto.FLOAT, s)
                add2_def = helper.make_node("Add", inputs=["c", "output"], outputs=["add2_out"])
                graph_def = helper.make_graph([add_def, add2_def], "{}_{}".format(case_name, i),
                                              [a, b, c], [add2_out])
            self.onnx_and_test(graph_def)

    def test_Add_5dim_bc(self, case_name):
        shape_A = [25,196,12,14,1]
        shape_B = [25,196,12,1,14]
        shape_res = [25,196,12,14,14]
        graph_txt = """
                %s (float%s a, float%s b, float%s c) => (float%s output)
                {
                    x = Add(a, b)
                    output = Sub(x, c)
                }
                """ % (case_name, shape_A, shape_B, shape_res, shape_res)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_AddBcast(self, case_name):
        shapes = (
            [
                10,
            ],
            [9, 12],
            [7, 14, 15],
            [16, 9, 323, 67],
            [4, 7, 38, 6, 4],
            [3, 3, 11, 3, 4, 5],
        )
        bcast_dims = (
            [[0]],
            [[0], [1]],
            [[0], [2], [0, 2], [0, 1, 2]],
            [[0], [2], [0, 2], [0, 3], [2, 3], [0, 2, 3], [0, 1, 2, 3]],
            [[0], [2], [0, 2], [3, 4], [2, 3, 4]],
            [[0], [2], [0, 2], [3, 4, 5], [2, 3, 4, 5]],
        )
        for i, s in enumerate(shapes):
            for dims in bcast_dims[i]:
                bcast_s = s[::]
                for dim in dims:
                    bcast_s[dim] = 1
                graph_txt = """
                    %s_%s_%s (float%s a, float%s b, float%s c) => (float output)
                    {
                        x = Add(a, b)
                        output = Add(x, c)
                    }
                    """ % (case_name, i, "".join(map(str, dims)), bcast_s, s, bcast_s)
                graph_def = onnx.parser.parse_graph(graph_txt)
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
            graph_txt = """
                %s_%s (float%s a, float%s b, float%s c) => (float%s output)
                {
                    x = Add(a, b)
                    output = Add(x, c)
                }
                """ % (case_name, i, bcast_shapes[i], s, bcast_shapes[i], out_shapes[i])
            graph_def = onnx.parser.parse_graph(graph_txt)
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
            graph_txt = """
                %s_%s (float%s a, float%s b, float%s c) => (float%s output)
                {
                    x = Add(a, b)
                    output = Add(x, c)
                }
                """ % (case_name, i, bcast_shapes[i], s, bcast_shapes[i], out_shapes[i])
            graph_def = onnx.parser.parse_graph(graph_txt)
            self.onnx_and_test(graph_def)

    def test_AddWeight2(self, case_name):
        input_shape = [1, 16, 8, 8]
        wshape = [16, 1, 1]
        output_shape = [1, 16, 8, 8]
        w_data = np.random.randn(*wshape).astype(np.float32)

        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s w>
            {
                output = Add(input, w)
            }
            """ % (case_name, input_shape, list(w_data.shape), output_shape)
        w_value = helper.make_tensor(
            name='w',
            data_type=onnx.TensorProto.FLOAT,
            dims=w_data.shape,
            vals=w_data.flatten(),
        )
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([w_value])
        self.onnx_and_test(graph_def)

    def test_AddWeight(self, case_name):
        input_shape = [1, 16, 28, 28]
        wshape = [28, 28]
        output_shape = [1, 16, 28, 28]
        w_data = np.random.randn(*wshape).astype(np.float32)

        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s w>
            {
                output = Add(input, w)
            }
            """ % (case_name, input_shape, output_shape, list(w_data.shape))
        w_value = helper.make_tensor(
            name='w',
            data_type=onnx.TensorProto.FLOAT,
            dims=w_data.shape,
            vals=w_data.flatten(),
        )

        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([w_value])
        self.onnx_and_test(graph_def)

    def test_BCastAdd(self, case_name):
        # 18xx: only broadcast right opd and broadcast continuous axis is supported
        if self.is_cv18xx:
            input_shape = {"input1": [2, 3, 27, 27], "input2": [2, 1, 1, 27]}
        else:
            input_shape = {"input1": [1, 3, 1, 27], "input2": [2, 1, 27, 1]}
        output_shape = [2, 3, 27, 27]

        graph_txt = """
            %s (float%s input1, float%s input2) => (float%s output)
            {
                output = Add(input1, input2)
            }
            """ % (case_name, input_shape['input1'], input_shape['input2'], output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_LRN(self, case_name):
        input_shape = [4, 10, 27, 27]
        output_shape = [4, 10, 27, 27]

        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = LRN<size=5>(input)
            }
            """ % (case_name, input_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_LSTM(self, case_name):
        seq_length = 75
        batch_size = 4
        num_dir = 2
        input_size = 128
        hidden_size = 64
        direction = 'forward' if num_dir == 1 else 'bidirectional'
        input_shape = [seq_length, batch_size, input_size]
        output_shape = [seq_length, num_dir, batch_size, hidden_size]
        # layout = 0
        w_data = np.random.randn(num_dir, 4 * hidden_size, input_size).astype(np.float32)
        r_data = np.random.randn(num_dir, 4 * hidden_size, hidden_size).astype(np.float32)
        b_data = np.random.randn(num_dir, 8 * hidden_size).astype(np.float32)
        h0_data = np.random.randn(num_dir, batch_size, hidden_size).astype(np.float32)
        c0_data = np.random.randn(num_dir, batch_size, hidden_size).astype(np.float32)

        graph_txt = """
            %s (float%s input) => (float%s Y)
            <float%s w, float%s r, float%s b, float%s h0, float%s c0>
            {
                Y = LSTM<direction="%s", hidden_size=%d>(input, w, r, b, , h0, c0)
            }
            """ % (case_name, input_shape, output_shape, list(w_data.shape), list(
            r_data.shape), list(b_data.shape), list(h0_data.shape), list(
                c0_data.shape), direction, hidden_size)
        graph_def = onnx.parser.parse_graph(graph_txt)

        w = helper.make_tensor('w', TensorProto.FLOAT, w_data.shape, w_data)
        r = helper.make_tensor('r', TensorProto.FLOAT, r_data.shape, r_data)
        b = helper.make_tensor('b', TensorProto.FLOAT, b_data.shape, b_data)
        h0 = helper.make_tensor('h0', TensorProto.FLOAT, h0_data.shape, h0_data)
        c0 = helper.make_tensor('c0', TensorProto.FLOAT, c0_data.shape, c0_data)
        graph_def.initializer.extend([w, r, b, h0, c0])
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

        input_s = [seq_length, batch_size, input_size]
        h0_s = [num_dir, batch_size, hidden_size]
        c0_s = [num_dir, batch_size, hidden_size]
        Y_s = [seq_length, num_dir, batch_size, hidden_size]
        Y_h_s = [num_dir, batch_size, hidden_size]
        Y_c_s = [num_dir, batch_size, hidden_size]

        graph_txt = """
            %s (float%s input, float%s h0, float%s c0) => (float%s Y, float%s Y_h, float%s Y_c)
            <float%s w, float%s r, float%s b>
            {
                Y, Y_h, Y_c= LSTM<direction="%s", hidden_size=%d>(input, w, r, b, , h0, c0)
            }
            """ % (case_name, input_s, h0_s, c0_s, Y_s, Y_h_s, Y_c_s, list(w_data.shape), list(
            r_data.shape), list(b_data.shape), direction, hidden_size)
        graph_def = onnx.parser.parse_graph(graph_txt)

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

        graph_def.initializer.extend([w_value, r_value, b_value])
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

        input_s = [seq_length, batch_size, input_size]
        h0_s = [num_dir, batch_size, hidden_size]
        c0_s = [num_dir, batch_size, hidden_size]
        Y_h_s = [num_dir, batch_size, hidden_size]
        Y_c_s = [num_dir, batch_size, hidden_size]
        graph_txt = """
            %s (float%s input, float%s h0, float%s c0) => (float%s Y_h, float%s Y_c)
            <float%s w, float%s r, float%s b, float Y>
            {
                Y, Y_h, Y_c= LSTM<direction="%s", hidden_size=%d>(input, w, r, b, , h0, c0)
            }
            """ % (case_name, input_s, h0_s, c0_s, Y_h_s, Y_c_s, list(w_data.shape), list(
            r_data.shape), list(b_data.shape), direction, hidden_size)
        graph_def = onnx.parser.parse_graph(graph_txt)

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
        graph_def.initializer.extend([w_value, r_value, b_value])
        self.onnx_and_test(graph_def)

    def test_BCastMul(self, case_name):
        input_shape = {"input1": [1, 3, 1, 27], "input2": [2, 1, 27, 1]}
        output_shape = [2, 3, 27, 27]
        if self.is_cv18xx:
            # 18xx: only broadcast right opd and broadcast continuous axis is supported
            input_shape = {"input1": [2, 3, 4, 27], "input2": [2, 3, 1, 1]}
            output_shape = [2, 3, 4, 27]
        graph_txt = """
            %s (float%s input1, float%s input2) => (float%s output)
            {
                output = Mul(input1, input2)
            }
            """ % (case_name, input_shape['input1'], input_shape['input2'], output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_BCastMulCst(self, case_name):
        input_shape = [1, 127, 270, 28]
        constant_shape = [2, 1, 1, 28]
        output_shape = [2, 127, 270, 28]
        if self.is_cv18xx:
            # 18xx: only broadcast right opd and broadcast continuous axis is supported
            input_shape = [2, 127, 270, 28]
            constant_shape = [1, 1, 1, 28]
            output_shape = [2, 127, 270, 28]
        constant = helper.make_tensor(
            "constant",
            TensorProto.FLOAT,
            constant_shape,
            np.random.rand(*constant_shape).astype(np.float32),
        )
        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s constant>
            {
                output = Mul(input, constant)
            }
            """ % (case_name, input_shape, output_shape, constant_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([constant])
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

        graph_txt = """
            %s (int64%s input1, float%s input2) => (float%s output)
            <float%s tokens>
            {
                x1 = Gather(tokens, input1)
                output = Add(x1, input2)
            }
            """ % (case_name, input_shape, output_shape, output_shape, token_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)

        token_data = helper.make_tensor(
            name='tokens',
            data_type=onnx.TensorProto.FLOAT,
            dims=token_shape,
            vals=np.random.randn(*token_shape).astype(np.float32).flatten(),
        )
        graph_def.initializer.extend([token_data])
        self.onnx_and_test(graph_def, input_data=input_data)

    def test_GatherElements(self, case_name):
        if self.chip in ['bm1684x', 'bm1690']:
            # samples too large for regression is commented
            # but all samples following should be passed
            input_data = [
                {
                    "data": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
                },
                {
                    "data": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
                },
                {
                    "data": np.random.randn(1, 13294, 4).astype(np.float32)
                },
                {
                    "data": np.arange(256).reshape(4, 4, 4, 4).astype(np.float32)
                },
                {
                    "data": np.random.randn(16, 17, 18, 19).astype(np.float32)
                    # "data": np.arange(16 * 17 * 18 * 19).reshape(16, 17, 18, 19).astype(np.float32) # for f32 debug
                },
                {
                    "data": np.random.randn(16, 17, 18, 19, 20).astype(np.float32)
                    # }, {
                    #     "data": np.random.randn(5, 8, 512, 512).astype(np.float32)
                    # }, {
                    #     "data": np.random.randn(5, 8, 512, 512).astype(np.float32)
                    # }, {
                    #     "data": np.random.randn(5, 8, 8, 512, 512).astype(np.float32)
                    # }, {
                    #     "data": np.random.randn(655340, 2).astype(np.float32)
                }
            ]
            indices_data = [
                np.array([[2, 1, 0, 2], [1, 2, 0, 1], [0, 2, 1, 0]], dtype=np.int64),
                np.array([[2, 1, 0], [1, 2, 0], [0, 2, 1], [0, 1, 2]], dtype=np.int64),
                np.random.randint(0, 13294, [1, 900, 4]),
                np.random.randint(0, 4, [2, 3, 9, 4]),
                np.random.randint(0, 18, [15, 12, 39, 15]),
                np.random.randint(0, 18, [15, 12, 39, 15, 13]),
                # np.random.randint(0, 512, [3, 7, 1025, 510]),
                # np.random.randint(0, 8, [3, 128, 511, 88]),
                # np.random.randint(0, 8, [3, 6, 128, 511, 88]),
                # np.random.randint(0, 2, [65536, 2]),
            ]
            # axis_data = [1, 0, 1, 2, 2, 2, 2, 1, 2, 1]
            axis_data = [1, 0, 1, 2, 2, 2]
        elif self.chip in ['bm1684']:
            input_data = [{
                "data": np.array([[[0, 0], [2, 2]], [[4, 5], [6, 7]]], dtype=np.float32),
            }]
            indices_data = [np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=np.int64)]
            axis_data = [2]
        for i in range(len(input_data)):
            # if i != 8:
            #     continue
            input_ = input_data[i]
            indices_ = indices_data[i]
            axis_ = axis_data[i]
            indices = helper.make_tensor(
                "indices",
                TensorProto.INT64,
                indices_.shape,
                indices_,
            )

            if self.chip in ['bm1684x', 'bm1690']:
                graph_txt = """
                    %s (float%s data) => (float%s gather_output)
                    <int64%s indices>
                    {
                        gather_output = GatherElements<axis=%d>(data, indices)
                    }
                    """ % (case_name, list(input_["data"].shape), list(
                    indices_data[i].shape), list(indices_.shape), axis_)
                graph_def = onnx.parser.parse_graph(graph_txt)
                graph_def.initializer.extend([indices])
            elif self.chip in ['bm1684']:
                graph_txt = """
                    %s (float%s data) => (float%s output)
                    <int64%s indices, float const_add>
                    {
                        gather_output = GatherElements<axis=%d>(data, indices)
                        output = Add(gather_output, const_add)
                    }
                    """ % (case_name, list(input_["data"].shape), list(
                    indices_data[i].shape), list(indices_.shape), axis_)
                graph_def = onnx.parser.parse_graph(graph_txt)

                add_const = helper.make_tensor(name='const_add',
                                               data_type=TensorProto.FLOAT,
                                               dims=[],
                                               vals=[2.0])
                graph_def.initializer.extend([indices, add_const])
            self.onnx_and_test(graph_def, input_data=input_)

    def test_GatherND(self, case_name):
        input_datas = [{
            "data": np.array([[0, 1], [2, 3]], dtype=np.float32)
        }, {
            "data": np.array([[0, 1], [2, 3]], dtype=np.float32)
        }, {
            "data": np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
        }, {
            "data": np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
        }, {
            "data": np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
        }]
        indices_datas = [
            np.array([[0, 0], [1, 1]], dtype=np.int64),
            np.array([[1], [0]], dtype=np.int64),
            np.array([[0, 1], [1, 0]], dtype=np.int64),
            np.array([[[0, 1]], [[1, 0]]], dtype=np.int64),
            np.array([[1], [0]], dtype=np.int64)
        ]
        output_shapes = [[2], [2, 2], [2, 2], [2, 1, 2], [2, 2]]
        batch_dims = [0, 0, 0, 0, 1]
        for i in range(len(input_datas)):
            input_shape = input_datas[i]["data"].shape
            output_shape = output_shapes[i]
            indices_data = indices_datas[i]
            input_data = input_datas[i]

            graph_txt = """
                %s_%s (float%s data) => (float%s output)
                <int64%s indices, float const_add>
                {
                    gather_output = GatherND<batch_dims=%d>(data, indices)
                    output = Add(gather_output, const_add)
                }
                """ % (case_name, i, list(input_shape), output_shape, list(indices_data.shape), batch_dims[i])
            graph_def = onnx.parser.parse_graph(graph_txt)

            add_const = helper.make_tensor(name='const_add',
                                           data_type=TensorProto.FLOAT,
                                           dims=[],
                                           vals=[2.0])
            indices = helper.make_tensor(
                "indices",
                TensorProto.INT64,
                indices_data.shape,
                indices_data,
            )
            graph_def.initializer.extend([indices, add_const])
            self.onnx_and_test(graph_def, input_data=input_data)

    def test_Sum(self, case_name):
        input_shape = [1, 3, 27, 27]
        output_shape = [1, 3, 27, 27]

        graph_txt = """
            %s (float%s input1, float%s input2, float%s input3) => (float%s output)
            {
                output = Sum(input1, input2, input3)
            }
            """ % (case_name, input_shape, input_shape, input_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Tile(self, case_name):
        input_shape = [1, 4, 6, 8]
        output_shape = [3, 24, 24, 16]

        graph_txt = """
            %s (float%s input) => (float%s output)
            <int64[4] tiles = {1, 6, 4, 2}>
            {
                output = Tile(input, tiles)
            }
            """ % (case_name, input_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Tile2(self, case_name):
        # test local_codegen, now"N N N N"
        # When testing the case, please open the LocalGenSupport in lib/Dialect/Tpu/Interfaces/Common/Tile.cpp
        input_shape = [1, 4, 6, 8]
        output_shape = [3, 24, 24, 16]

        graph_txt = """
            %s (float%s input,float%s right) => (float%s output)
            <int64[4] tiles = {1, 6, 4, 2}>
            {
                left = Tile(input, tiles)
                output = Add(left, right)
            }
            """ % (case_name, input_shape, output_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_TileDyn(self, case_name):
        nonzero_input_shape = [2, 6]
        tile_input_shape = [3, 5]
        graph_txt = """
            %s (float%s nonzero_input, float%s tile_input) => (float Y_Value)
            <int64 indices0 = {0}, int64[1] indices = {0}, int64[1] dim0 = {3}>
            {
                gather0 = Gather<axis=0>(nonzero_input, indices0)
                nonzero = NonZero(gather0)
                transpose = Transpose<perm=[1, 0]>(nonzero)
                shape = Shape(transpose)
                dim1 = Gather<axis=0>(shape,indices)
                times = Concat<axis=0>(dim0, dim1)
                Y_Value = Tile(tile_input, times)
            }
            """ % (case_name, nonzero_input_shape, tile_input_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        input_data = {
            'nonzero_input':
            np.array([[1, 0, 3, 0, 4, 5], [2.1, 2.5, 0, 0, 2.6, 0]], dtype=np.float32),
            'tile_input':
            np.array([[1.5, 1.2, 0, 0, 1.6], [2.1, 2.5, 0, 0, 2.6], [3.5, 3.2, 0, 0, 3.6]],
                     dtype=np.float32)
        }
        self.onnx_and_test_bmodel(graph_def,
                                  static_shape=False,
                                  input_data=input_data,
                                  only_cmp_with_bmodel=True)

    def test_ConstantFillDyn(self, case_name):
        tensor_input_shape = [2, 6]

        graph_txt = """
            %s (float%s tensor_input) => (float Y_Value)
            <float constant>
            {
                shape = Shape(tensor_input)
                Y_Value = ConstantOfShape(shape)
            }
            """ % (case_name, tensor_input_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        input_data = {
            'tensor_input':
            np.array([[1, 0, 3, 0, 4, 5], [2.1, 2.5, 0, 0, 2.6, 0]], dtype=np.float32),
        }
        self.onnx_and_test_bmodel(graph_def,
                                  static_shape=False,
                                  input_data=input_data,
                                  only_cmp_with_bmodel=True)

    def test_Sub(self, case_name):
        input_shape = [4, 3, 27, 27]
        output_shape = [4, 3, 27, 27]
        graph_txt = """
            %s (float%s input0,float%s input1) => (float%s output)
            {
                output = Sub(input0, input1)
            }
            """ % (case_name, input_shape, input_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Sub2(self, case_name):
        input1_shape = [4, 3, 27, 27]
        input2_shape = [4, 3, 1, 27]
        output_shape = [4, 3, 27, 27]

        graph_txt = """
            %s (float%s input1,float%s input2) => (float%s output)
            {
                output = Sub(input1, input2)
            }
            """ % (case_name, input1_shape, input2_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_SubBcast(self, case_name):
        shapes = ([4, 9, 23, 39], )
        bcast_dims = ([[0], [2], [0, 2]], )
        for i, s in enumerate(shapes):
            for dims in bcast_dims[i]:
                bcast_s = s[::]
                for dim in dims:
                    bcast_s[dim] = 1
                graph_txt = """
                    %s_%s_%s (float%s a, float%s b, float%s c) => (float%s output)
                    {
                        x = Sub(a, b)
                        output = Sub(x, c)
                    }
                    """ % (case_name, i, "".join(map(str, dims)), bcast_s, s, bcast_s, s)
                graph_def = onnx.parser.parse_graph(graph_txt)
                self.onnx_and_test(graph_def)

    def test_SubBcast2(self, case_name):
        shapes = ([6, 7, 1, 11], )
        bcast_shapes = ([1, 7, 13, 11], )
        out_shapes = ([6, 7, 13, 11], )
        for i, s in enumerate(shapes):
            graph_txt = """
                %s_%s (float%s a, float%s b, float%s c) => (float%s output)
                {
                    x = Sub(a, b)
                    output = Sub(x, c)
                }
                """ % (case_name, i, bcast_shapes[i], s, bcast_shapes[i], out_shapes[i])
            graph_def = onnx.parser.parse_graph(graph_txt)
            self.onnx_and_test(graph_def)

    def test_SubConst(self, case_name):
        input_shape = [4, 3, 27, 27]
        output_shape = [4, 3, 27, 27]

        w_data = np.random.rand(*input_shape).astype(np.float32)
        w_value = helper.make_tensor(
            name='w',
            data_type=onnx.TensorProto.FLOAT,
            dims=w_data.shape,
            vals=w_data.flatten(),
        )
        const = helper.make_tensor(name='const', data_type=TensorProto.FLOAT, dims=[], vals=[1.0])
        graph_txt = """
            %s (float%s input0) => (float%s output0, float%s output1, float%s output2)
            <float%s w, float const>
            {
                output0 = Sub(input0, w)
                output1 = Sub(input0, const)
                output2 = Sub(const, input0)
            }
            """ % (case_name, input_shape, output_shape, output_shape, output_shape, list(w_data.shape))
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([w_value, const])
        self.onnx_and_test(graph_def)

    def test_SubConst2(self, case_name):
        input1_shape = [4, 3, 27, 27]
        w_shape = [4, 3, 1, 27]
        output_shape = [4, 3, 27, 27]

        w_data = np.random.rand(*w_shape).astype(np.float32)
        w_value = helper.make_tensor(
            name='w',
            data_type=onnx.TensorProto.FLOAT,
            dims=w_data.shape,
            vals=w_data.flatten(),
        )
        graph_txt = """
            %s (float%s input1) => (float%s output)
            <float%s w>
            {
                output = Sub(input1, w)
            }
            """ % (case_name, input1_shape, output_shape, list(w_data.shape))
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([w_value])
        self.onnx_and_test(graph_def)

    def test_Expand(self, case_name):
        input_shape = [1, 3, 1, 16]
        output_shape = [1, 3, 16, 16]
        graph_txt = """
            %s (float%s input) => (float%s output)
            <int64%s new_shape>
            {
                output = Expand(input, new_shape)
            }
            """ % (case_name, input_shape, output_shape, [len(output_shape)])
        graph_def = onnx.parser.parse_graph(graph_txt)

        shape_def = helper.make_tensor(
            name='new_shape',
            data_type=onnx.TensorProto.INT64,
            dims=[len(output_shape)],
            vals=np.array(output_shape, dtype=np.int64),
        )
        graph_def.initializer.extend([shape_def])
        self.onnx_and_test(graph_def)

    def test_Expand2(self, case_name):
        input_shape = [1, 16]
        output_shape = [4, 16, 16]
        graph_txt = """
            %s (float%s input) => (float%s output)
            <int64%s new_shape>
            {
                output = Expand(input, new_shape)
            }
            """ % (case_name, input_shape, output_shape, [len(output_shape)])
        graph_def = onnx.parser.parse_graph(graph_txt)

        shape_def = helper.make_tensor(
            name='new_shape',
            data_type=onnx.TensorProto.INT64,
            dims=[len(output_shape)],
            vals=np.array([4, 16, 1], dtype=np.int64),
        )
        graph_def.initializer.extend([shape_def])
        self.onnx_and_test(graph_def)

    def test_ExpandDyn(self, case_name):
        nonzero_input_shape = [2, 6]
        expand_input_shape = [3, 1]
        indices0 = helper.make_tensor('indices0', TensorProto.INT64, [], vals=[0])
        indices = helper.make_tensor('indices', TensorProto.INT64, [1], vals=[0])
        dim0 = helper.make_tensor('dim0', TensorProto.INT64, [1], vals=[3])

        graph_txt = """
            %s (float%s nonzero_input, float%s expand_input) => (float Y_Value)
            <int64 indices0, int64[1] indices, int64[1] dim0>
            {
                gather0 = Gather<axis=0>(nonzero_input, indices0)
                nonzero = NonZero(gather0)
                transpose = Transpose<perm=[1, 0]>(nonzero)
                shape = Shape(transpose)
                dim1 = Gather<axis=0>(shape, indices)
                times = Concat<axis=0>(dim0, dim1)
                Y_Value = Expand(expand_input, times)
            }
            """ % (case_name, nonzero_input_shape, expand_input_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([indices0, indices, dim0])
        input_data = {
            'nonzero_input': np.array([[1, 0, 3, 0, 4, 0], [2.1, 2.5, 0, 0, 2.6, 0]],
                                      dtype=np.float32),
            'expand_input': np.array([[1.5], [2.1], [3.5]], dtype=np.float32)
        }
        self.onnx_and_test_bmodel(graph_def,
                                  static_shape=False,
                                  input_data=input_data,
                                  only_cmp_with_bmodel=True)

    def test_Max(self, case_name):
        input_shape = {"input1": [1, 85, 32, 8], "input2": [1, 85, 32, 8]}
        output_shape = [1, 85, 32, 8]

        graph_txt = """
            %s (float%s input1, float%s input2) => (float%s output)
            {
                output = %s(input1, input2)
            }
            """ % (case_name, input_shape['input1'], input_shape["input2"], output_shape, case_name)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_MaxBcast(self, case_name):
        shapes = ([6, 8, 39, 41], )
        bcast_dims = ([[0], [2], [0, 2]], )
        for i, s in enumerate(shapes):
            for dims in bcast_dims[i]:
                bcast_s = s[::]
                for dim in dims:
                    bcast_s[dim] = 1
                graph_txt = """
                    %s_%s_%s (float%s a, float%s b, float%s c) => (float%s output)
                    {
                        x = Max(a, b)
                        output = Max(x, c)
                    }
                    """ % (case_name, i, "".join(map(str, dims)), bcast_s, s, bcast_s, s)
                graph_def = onnx.parser.parse_graph(graph_txt)
                self.onnx_and_test(graph_def)

    def test_Min(self, case_name):
        input_shape = {"input1": [1, 85, 32, 8], "input2": [1, 85, 32, 8]}
        output_shape = [1, 85, 32, 8]
        graph_txt = """
            %s (float%s input1, float%s input2) => (float%s output)
            {
                output = %s(input1, input2)
            }
            """ % (case_name, input_shape['input1'], input_shape["input2"], output_shape, case_name)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_MinBcast(self, case_name):
        shapes = ([5, 11, 23, 27], )
        bcast_dims = ([[0], [2], [0, 2]], )
        for i, s in enumerate(shapes):
            for dims in bcast_dims[i]:
                bcast_s = s[::]
                for dim in dims:
                    bcast_s[dim] = 1
                graph_txt = """
                    %s_%s_%s (float%s a, float%s b, float%s c) => (float%s output)
                    {
                        x = Min(a, b)
                        output = Min(x, c)
                    }
                    """ % (case_name, i, "".join(map(str, dims)), bcast_s, s, bcast_s, s)
                graph_def = onnx.parser.parse_graph(graph_txt)
                self.onnx_and_test(graph_def)

    def test_Abs(self, case_name):
        input_shape = [1, 16, 64, 64]
        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = Abs(input)
            }
            """ % (case_name, input_shape, input_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Reciprocal(self, case_name):
        input_shape = [4, 3, 224, 224]
        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = Reciprocal(input)
            }
            """ % (case_name, input_shape, input_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        input_data = np.random.rand(*input_shape).astype(np.float32) + 0.5
        # avoid divide 0
        input_data[input_data == 0] = 1
        self.onnx_and_test(graph_def, input_data={"input": input_data})

    def test_PRelu(self, case_name):
        input_shape = [2, 128, 100, 100]
        slope_shape = [1, 128, 1, 1]
        output_shape = [2, 128, 100, 100]
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
            graph_txt = """
            %s (float%s input, float%s input1) => (float%s output)
                {
                    output0 = PRelu(input, slope)
                    output = Add(output0, input1)
                }
                """ % (case_name, input_shape, input_shape, output_shape)
            graph_def = onnx.parser.parse_graph(graph_txt)
            graph_def.initializer.extend([slope])
            self.onnx_and_test(graph_def)

    def test_Sqrt(self, case_name):
        shape = [3, 5, 100, 100]
        graph_txt = """
            %s (float%s input) => (float%s output)
                {
                    output = Sqrt(input)
                }
                """ % (case_name, shape, shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        input_data = np.abs(np.random.randn(*shape).astype(np.float32))
        self.onnx_and_test(graph_def, input_data={"input": input_data})

    def test_Pow1(self, case_name):
        shape = [1, 3, 27, 27]
        input_data = np.abs(np.random.randn(*shape).astype(np.float32)) + 1e-6
        constant = helper.make_tensor("constant", TensorProto.FLOAT, [1],
                                      np.array([1.2]).astype(np.float32))
        graph_txt = """
            %s (float%s input) => (float%s output)
            <float[1] constant>
            {
                output = Pow(input, constant)
            }
            """ % (case_name, shape, shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([constant])
        self.onnx_and_test(graph_def, input_data={"input": input_data})

    def test_Pow2(self, case_name):
        shape = [1, 3, 27, 27]
        constant = helper.make_tensor("constant", TensorProto.FLOAT, [1],
                                      np.array([1.2]).astype(np.float32))
        graph_txt = """
            %s (float%s input) => (float%s output)
            <float[1] constant>
            {
                output = Pow(constant, input)
            }
            """ % (case_name, shape, shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([constant])
        self.onnx_and_test(graph_def)

    def test_Pow3(self, case_name):
        shape = [1, 3, 27, 27]
        input1_data = np.abs(np.random.randn(*shape).astype(np.float32)) + 1e-6
        input2_data = np.random.randn(*shape).astype(np.float32)
        graph_txt = """
            %s (float%s input1, float%s input2) => (float%s output)
            {
                output = Pow(input1, input2)
            }
            """ % (case_name, shape, shape, shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def, input_data={"input1": input1_data, "input2": input2_data})

    def test_Not(self, case_name):
        shape = [1, 3, 27, 27]
        graph_txt = """
            %s (bool%s input) => (bool%s output)
            {
                output = Not(input)
            }
            """ % (case_name, shape, shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Mod(self, case_name):
        shape = [1, 5, 15, 15]
        input1_info = helper.make_tensor_value_info('input1', TensorProto.INT64, shape)
        input2_info = helper.make_tensor_value_info('input2', TensorProto.INT64, [1])
        output_info = helper.make_tensor_value_info('output', TensorProto.INT64, shape)

        node_def = helper.make_node("Mod",
                                   inputs=["input1", "input2"],
                                   outputs=["output"],
                                   fmod=0)
        graph_def = helper.make_graph([node_def],
                                      case_name,
                                      [input1_info, input2_info],
                                      [output_info])
        self.onnx_and_test(graph_def)

    def test_Cast(self, case_name):
        input_shape = [1, 32, 64]
        output1_shape, output2_shape = [1, 32, 64, 1], [1, 1, 32, 64]
        dim4_shape = [4]
        shape1_data = np.array(output1_shape, dtype=np.int64)
        shape2_data = np.array(output2_shape, dtype=np.int64)
        input_data = {"input": np.random.randint(0, 255, input_shape).astype(np.float32)}
        graph_txt = """
            %s (float%s input) => (float%s output1, int64%s output2)
            <int64%s shape1, int64%s shape2, float const_mul>
            {
                shape1_out = Reshape(input, shape1)
                cast1_out = Cast<to=%d>(shape1_out)
                output2 = Reshape(cast1_out, shape2)
                cast2_out = Cast<to=%d>(cast1_out)
                output1 = Mul(cast2_out, const_mul)
            }
            """ % (case_name, input_shape, output1_shape, output2_shape, dim4_shape, dim4_shape,
                   TensorProto.INT64, TensorProto.FLOAT)
        graph_def = onnx.parser.parse_graph(graph_txt)

        shape1 = helper.make_tensor('shape1', TensorProto.INT64, dim4_shape, shape1_data)
        shape2 = helper.make_tensor('shape2', TensorProto.INT64, dim4_shape, shape2_data)
        mul_const = helper.make_tensor(name='const_mul',
                                       data_type=TensorProto.FLOAT,
                                       dims=[],
                                       vals=[2.0])
        graph_def.initializer.extend([shape1, shape2, mul_const])
        self.onnx_and_test(graph_def, input_data=input_data)

    def test_CompareCst(self, case_name):
        shape = [1, 3, 27, 27]
        # "Equal" need not to be tested since equal op between floating number may be invalid
        cases = ("Greater", "GreaterOrEqual", "Less", "LessOrEqual")
        const_value = 0.5
        if self.is_cv18xx:
            cases = ("Equal", )
            const_value = 0.0
        constant = helper.make_tensor("constant", TensorProto.FLOAT, [1],
                                      np.array([const_value]).astype(np.float32))
        for cmp_type in cases:
            graph_txt = """
                %s (float%s input) => (bool%s output)
                <float[1] constant>
                {
                    output = %s(input, constant)
                }
                """ % (case_name, shape, shape, cmp_type)
            graph_def = onnx.parser.parse_graph(graph_txt)
            graph_def.initializer.extend([constant])
            self.onnx_and_test(graph_def)
            print("====== TEST {} Success ======".format(cmp_type))

    def test_And(self, case_name):
        shape = [1, 3, 27, 27]
        graph_txt = """
            %s (bool[1, 3, 27, 27] input, bool[1] constant = {1}) => (bool[1, 3, 27, 27] output)
            {
                output = And(input, constant)
            }
            """ % (case_name)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Xor(self, case_name):
        input1_shape = [1, 3, 27, 27]
        input2_shapes = [[1, 3, 27, 27], [1]]
        for input2_shape in input2_shapes:
            graph_txt = """
                %s (bool%s input1, bool%s input2) => (bool%s output)
                {
                    output = Xor(input1, input2)
                }
                """ % (case_name, input1_shape, input2_shape, input1_shape)
            graph_def = onnx.parser.parse_graph(graph_txt)
            self.onnx_and_test(graph_def)

    def test_Compare(self, case_name):
        shape = [1, 3, 27, 27]
        input_shape = {"input1": shape, "input2": shape}
        # "Equal" need not to be tested since equal op between floating number may be invalid
        for cmp_type in ("Greater", "GreaterOrEqual", "Less", "LessOrEqual"):
            graph_txt = """
                %s (float%s input1, float%s input2) => (bool%s output)
                {
                    output = %s(input1, input2)
                }
                """ % (case_name, shape, shape, shape, cmp_type)
            graph_def = onnx.parser.parse_graph(graph_txt)
            self.onnx_and_test(graph_def)
            print("====== TEST {} Success ======".format(cmp_type))

    def test_Compare2(self, case_name):
        input_shape = {"input1": [1, 3, 1, 27], "input2": [1, 3, 27, 1]}
        output_shape = [1, 3, 27, 27]
        input_data = {
            "input1": np.random.randn(*input_shape["input1"]).astype(np.float32),
            "input2": np.random.randn(*input_shape["input2"]).astype(np.float32)
        }
        # "Equal" need not to be tested since equal op between floating number may be invalid
        for cmp_type in ("Greater", "GreaterOrEqual", "Less", "LessOrEqual"):
            graph_txt = """
                %s (float%s input1, float%s input2) => (bool%s output)
                {
                    output = %s(input1, input2)
                }
                """ % (case_name, input_shape["input1"], input_shape["input2"], output_shape, cmp_type)
            graph_def = onnx.parser.parse_graph(graph_txt)
            self.onnx_and_test(graph_def, input_data=input_data)
            print("====== TEST {} Success ======".format(cmp_type))

    def test_Einsum(self, case_name):
        input_shape = {"input1": [1, 26, 12, 26], "input2": [12, 26, 312]}
        output_shape = [1, 26, 312]
        graph_txt = """
            %s (float%s input1, float%s input2) => (float%s output)
            {
                output = Einsum<equation="bfnd,ndh->bfh">(input1, input2)
            }
            """ % (case_name, input_shape["input1"], input_shape["input2"], output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Einsum2(self, case_name):
        input_shape = [1, 26, 12, 26]
        filter_shape = [12, 26, 312]
        output_shape = [1, 26, 312]

        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s weight>
            {
                output = Einsum<equation="bfnd,ndh->bfh">(input, weight)
            }
            """ % (case_name, input_shape, output_shape, filter_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([weight])
        self.onnx_and_test(graph_def)

    def test_Einsum3(self, case_name):
        input_shape = {"input1": [4], "input2": [32]}
        output_shape = [4, 32]
        graph_txt = """
            %s (float%s input1, float%s input2) => (float%s output)
            {
                output = Einsum<equation="i,j->ij">(input1, input2)
            }
            """ % (case_name, input_shape["input1"], input_shape["input2"], output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_Einsum4(self, case_name):
        for i, equation in enumerate(['bhwc,hkc->bhwk', 'bhwc,wkc->bhwk']):
            input_shape = [5, 16, 15, 64]
            filter_shape = [16, 14, 64] if equation == 'bhwc,hkc->bhwk' else [15, 14, 64]
            output_shape = [5, 16, 15, 14]

            weight_data = np.random.randn(*filter_shape).astype(np.float32)
            weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
            graph_txt = """
                %s (float%s input) => (float%s output)
                <float%s weight>
                {
                    output = Einsum<equation="%s">(input, weight)
                }
                """ % (case_name, input_shape, output_shape, filter_shape, equation)
            graph_def = onnx.parser.parse_graph(graph_txt)
            graph_def.initializer.extend([weight])
            self.onnx_and_test(graph_def)

    def test_Einsum5(self, case_name):
        for i, equation in enumerate(['bhwc,hkc->bhwk', 'bhwc,wkc->bhwk']):
            input_shape = {"input1": [5, 16, 15, 64], "input2": [16, 14, 64]}
            if equation == 'bhwc,wkc->bhwk':
                input_shape["input2"][0] = input_shape["input1"][2]
            output_shape = [5, 15, 15, 14]
            graph_txt = """
                %s (float%s input1, float%s input2) => (float%s output)
                {
                    output = Einsum<equation="%s">(input1, input2)
                }
                """ % (case_name, input_shape["input1"], input_shape["input2"], output_shape, equation)
            graph_def = onnx.parser.parse_graph(graph_txt)
            self.onnx_and_test(graph_def)

    def test_Einsum6(self, case_name):
        for i, equation in enumerate(['bhwc,bhkc->bhwk', 'bhwc,bhck->bhwk']):
            input_shape = {"input1": [5, 16, 15, 64], "input2": [5, 16, 14, 64]}
            if equation == 'bhwc,bhck->bhwk':
                input_shape["input2"][3] = input_shape["input2"][2]
                input_shape["input2"][2] = input_shape["input1"][3]
            output_shape = [5, 16, 15, 14]
            graph_txt = """
                Einsum6_%s (float%s input1, float%s input2) => (float%s output)
                {
                    output = Einsum<equation="%s">(input1, input2)
                }
                """ % (i, input_shape["input1"], input_shape["input2"], output_shape, equation)
            graph_def = onnx.parser.parse_graph(graph_txt)
            self.onnx_and_test(graph_def)

    def test_Einsum7(self, case_name):
        input_shape = [1, 26, 12, 26]
        filter_shape = [12, 26, 312]
        output_shape = [1, 26, 12, 312]

        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s weight>
            {
                output = Einsum<equation="abcd,cde->abce">(input, weight)
            }
            """ % (case_name, input_shape, output_shape, filter_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([weight])
        self.onnx_and_test(graph_def)

    def test_Einsum8(self, case_name):
        input_shape = [12, 26, 13]
        filter_shape = [12, 13, 32, 16]
        output_shape = [12, 26, 32, 16]

        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s weight>
            {
                output = Einsum<equation="bqc,bchw->bqhw">(input, weight)
            }
            """ % (case_name, input_shape, output_shape, filter_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([weight])
        self.onnx_and_test(graph_def)

    def test_Einsum9(self, case_name):
        input_shape = [12, 26, 13]
        filter_shape = [12, 26, 32, 16]
        output_shape = [12, 13, 32, 16]

        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s weight>
            {
                output = Einsum<equation="bqc,bqhw->bchw">(input, weight)
            }
            """ % (case_name, input_shape, output_shape, filter_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([weight])
        self.onnx_and_test(graph_def)
    def test_Einsum10(self, case_name):
        input_shape = [12, 26, 13]
        filter_shape = [26, 32]
        output_shape = [12, 26, 13, 32]

        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s weight>
            {
                output = Einsum<equation="abc,bd->abcd">(input, weight)
            }
            """ % (case_name, input_shape, output_shape, filter_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([weight])
        self.onnx_and_test(graph_def)

    def test_Einsum11(self, case_name):
        input_shape = [1, 12, 26, 13]
        filter_shape = [1, 32, 26, 13]
        output_shape = [1, 32, 12]

        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s weight>
            {
                output = Einsum<equation="abcd,aecd->aeb">(input, weight)
            }
            """ % (case_name, input_shape, output_shape, filter_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([weight])
        self.onnx_and_test(graph_def)

    def test_Einsum12(self, case_name):
        input_shape = [1, 26, 13]
        filter_shape = [13, 12, 32]
        output_shape = [1, 26, 12, 32]

        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s weight>
            {
                output = Einsum<equation="abc,cde->abde">(input, weight)
            }
            """ % (case_name, input_shape, output_shape, filter_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([weight])
        self.onnx_and_test(graph_def)

    def test_Einsum13(self, case_name):
        input_shape = [1, 8, 32, 50, 50]
        filter_shape = [1, 80, 8, 32]
        output_shape = [1, 8, 50, 50, 80]

        # bm1688 f16、int8 raise failure
        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s weight>
            {
                output = Einsum<equation="abcde,afbc->abdef">(input, weight)
            }
            """ % (case_name, input_shape, output_shape, filter_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([weight])
        self.onnx_and_test(graph_def)

    def test_Einsum14(self, case_name):
        input_shape = [1, 768, 50, 50]
        filter_shape = [1, 80, 768]
        output_shape = [1, 80, 50, 50]

        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s weight>
            {
                output = Einsum<equation="abcd,aeb->aecd">(input, weight)
            }
            """ % (case_name, input_shape, output_shape, filter_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([weight])
        self.onnx_and_test(graph_def)

    def test_Elu(self, case_name):
        oc = 32
        input_shape = [1, 16, 100, 100]
        filter_shape = [oc, 16, 3, 3]
        output_shape = [1, oc, 100, 100]
        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        bias_data = np.random.randn(oc).astype(np.float32)

        weight = helper.make_tensor('weight', TensorProto.FLOAT, filter_shape, weight_data)
        bias = helper.make_tensor('bias', TensorProto.FLOAT, list(bias_data.shape), bias_data)

        graph_txt = """
            %s (float%s input) => (float%s output)
            <float%s weight, float%s bias>
            {
                conv_output = Conv<kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1], dilations=[1, 1], group=1>(input, weight, bias)
                output = Elu<alpha=0.67>(conv_output)
            }
            """ % (case_name, input_shape, output_shape, filter_shape, list(bias_data.shape))
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([weight, bias])
        self.onnx_and_test(graph_def)

    def test_Erf(self, case_name):
        input_shape = [10, 3, 32, 32]
        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = %s(input)
            }
            """ % (case_name, input_shape, input_shape, case_name)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_TopK(self, case_name):
        shape = [10, 1000]
        const = 500
        o_shape = list(shape)
        o_shape[-1] = const
        graph_txt = """
            %s (float%s X) => (float%s Y_Value, int64%s Y_Index)
            <int64[1] K = {%d}>
            {
                Y_Value, Y_Index = TopK<axis=-1, largest=1>(X, K)
            }
            """ % (case_name, shape, o_shape, o_shape, const)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_TopK2(self, case_name):
        input_shape = [3, 6]
        graph_txt = """
            %s (float%s input) => (float Y_Value, int64 Y_Index)
            <int64 indices0 = {0}, int64[1] indices = {0}>
            {
                gather0 = Gather(input, indices0)
                nonzero = NonZero(gather0)
                transpose = Transpose<perm=[1, 0]>(nonzero)
                shape = Shape(transpose)
                K = Gather<axis=0>(shape,indices)
                Y_Value, Y_Index = TopK<axis=-1, largest=1>(input, K)
            }
            """ % (case_name, input_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        input_data = {
            'input':
            np.array([[1, 0, 3, 0, 0, 0], [4, 5, 6, 3, 2, 1], [7, 8, 9, 2, 1, 1]], dtype=np.float32)
        }
        self.onnx_and_test_bmodel(graph_def,
                                  static_shape=False,
                                  input_data=input_data,
                                  only_cmp_with_bmodel=True)

    def test_TopK3(self, case_name):
        # This is for testing topk gmem overflow hardware bug
        shape = [1, 11400]
        const = 200
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, shape)
        C = helper.make_tensor("C", TensorProto.FLOAT, [1], np.array([1.0]).astype(np.float64))
        K = helper.make_tensor("K", TensorProto.INT64, [1], np.array([const]).astype(np.int64))
        o_shape = list(shape)
        o_shape[-1] = const
        Y_Index = helper.make_tensor_value_info('Y_Index', TensorProto.INT64, o_shape)
        Add_Value = helper.make_tensor_value_info('add_out', TensorProto.FLOAT, shape)
        add_node = helper.make_node("Add", inputs=["X", "C"], outputs=["add_out"])
        topk_node = helper.make_node('TopK', ['X', 'K'], ['Y_Value', 'Y_Index'],
                                     axis=-1,
                                     largest=True)
        graph_def = helper.make_graph([add_node, topk_node],
                                      case_name, [X], [Y_Index, Add_Value],
                                      initializer=[K, C])
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

        graph_txt = """
            %s (float%s input) => (float%s output)
            <int8%s weight, float%s bias, float y_scale, int8 y_zero_point, float x_scale, int8 x_zero_point,
            float weight_x_scale, int8 weight_x_zero_point, float output_y_scale, int8 output_y_zero_point, float output_x_scale, int8 output_x_zero_point>
            {
                a = QuantizeLinear<axis=0>(input, y_scale, y_zero_point)
                b = DequantizeLinear<axis=0>(a, x_scale, x_zero_point)
                weight_output = DequantizeLinear<axis=0>(weight, weight_x_scale, weight_x_zero_point)
                conv_output = Conv<kernel_shape=%s, pads=%s, strides=%s, dilations=%s, group=%d>(b, weight_output, bias)
                c = QuantizeLinear<axis=0>(conv_output, output_y_scale, output_y_zero_point)
                output = DequantizeLinear<axis=0>(c, output_x_scale, output_x_zero_point)
            }
            """ % (case_name, input_shape, output_shape, filter_shape, list(
            bias_data.shape), kernel, padding, stride, dilation, groups)
        graph_def = onnx.parser.parse_graph(graph_txt)

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
        output_y_scale = helper.make_tensor("output_y_scale", TensorProto.FLOAT, [1],
                                            [output_y_scale_data])
        output_y_zero_point = helper.make_tensor("output_y_zero_point", TensorProto.INT8, [1],
                                                 [output_y_zero_point_data])
        output_x_scale = helper.make_tensor("output_x_scale", TensorProto.FLOAT, [1],
                                            [output_x_scale_data])
        output_x_zero_point = helper.make_tensor("output_x_zero_point", TensorProto.INT8, [1],
                                                 [output_x_zero_point_data])

        graph_def.initializer.extend([
            y_scale, y_zero_point, x_scale, x_zero_point, weight_x_scale, weight_x_zero_point,
            weight, bias, output_x_scale, output_x_zero_point, output_y_scale, output_y_zero_point
        ])
        self.onnx_and_test(graph_def)

    def test_QDQ(self, case_name):
        input_shape = [10, 3, 224, 224]
        y_scale_data = np.random.uniform(0, 1)
        y_zero_point_data = np.random.randint(0, 255)
        x_scale_data = deepcopy(y_scale_data)
        x_zero_point_data = deepcopy(y_zero_point_data)

        y_scale = helper.make_tensor("y_scale", TensorProto.FLOAT, [1], [y_scale_data])
        y_zero_point = helper.make_tensor("y_zero_point", TensorProto.INT8, [1],
                                          [y_zero_point_data])
        x_scale = helper.make_tensor("x_scale", TensorProto.FLOAT, [1], [x_scale_data])
        x_zero_point = helper.make_tensor("x_zero_point", TensorProto.INT8, [1],
                                          [x_zero_point_data])
        graph_txt = """
            %s (float%s input) => (float%s output)
            <float y_scale, int8 y_zero_point, float x_scale, int8 x_zero_point>
            {
                a = QuantizeLinear(input, y_scale, y_zero_point)
                a_output = DequantizeLinear(a, x_scale, x_zero_point)
                output = Log(a_output)
            }
            """ % (case_name, input_shape, input_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([y_scale, y_zero_point, x_scale, x_zero_point])
        self.onnx_and_test(graph_def)

    def test_ReduceTranspose(self, case_name):
        input_shape = [8, 16, 32, 64]
        transpose_order = [1, 0, 2, 3]
        transpose_output_shape = [
            input_shape[transpose_order[i]] for i in range(len(transpose_order))
        ]
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
        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                transpose_output = Transpose<perm=%s>(input)
                output = ReduceMean<keepdims=%d, axes=%s>(transpose_output)
            }
            """ % (case_name, input_shape, reduce_output_shape, transpose_order, reduce_keepdims, reduce_axes)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_TransposeArg(self, case_name):
        input_shape = [8, 16, 32, 64]
        transpose_order = [0, 2, 1, 3]
        arg_keepdims = False
        arg_axis = 1
        reduce_output_shape = [8, 16, 64]
        graph_txt = """
            %s (float%s input) => (int64%s output)
            {
                transpose_output = Transpose<perm=%s>(input)
                output = ArgMax<keepdims=%d, axis=%s, select_last_index=1>(transpose_output)
            }
            """ % (case_name, input_shape, reduce_output_shape, transpose_order, arg_keepdims, arg_axis)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_ArgError(self, case_name):
        input_shape = [1, 1000, 1, 1]
        output_shape = [1, 1, 1, 1]
        graph_txt = """
            %s (float%s input) => (int64%s o_max, float%s x2)
            {
                x2 = Relu(input)
                o_max = ArgMax<keepdims=1, axis=1, select_last_index=1>(x2)
            }
            """ % (case_name, input_shape, output_shape, input_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def test_ArgReducefull(self, case_name):
        input_shape = [2, 3, 4]
        arg_axis = 0
        reduce_axes = [0]
        output_shape = [1, 3, 4]
        graph_txt = """
            %s (float%s input) => (int64%s arg_output, float%s reduce_output_1, float%s reduce_output_2)
            {
                arg_output = ArgMax<axis=%d, select_last_index=1>(input)
                reduce_output_1 = ReduceMax<axes=%s>(input)
                reduce_output_2 = ReduceMax<axes=%s>(input)
            }
            """ % (case_name, input_shape, output_shape, output_shape, output_shape, arg_axis, reduce_axes,
                   reduce_axes)
        graph_def = onnx.parser.parse_graph(graph_txt)
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
        if binary_name == "Div":
            weight_data_ = np.random.uniform(0.5, 2, tuple(weight_shape)).astype(np.float32)
            weight_data = np.multiply(weight_data_, np.random.choice(np.array([-1, 1]),
                                                                     weight_shape))
        else:
            weight_data = np.random.randn(*weight_shape).astype(np.float32)
        if binary_name in ("Greater", "GreaterOrEqual", "Less", "LessOrEqual"):
            output_dtype = 'bool'
        else:
            output_dtype = 'float'
        graph_txt = """
            %s (float%s input) => (%s%s output)
            <float%s weight1>
            {
                output = %s(input, weight1)
            }
            """ % (case_name, input_shape, output_dtype, input_shape, weight_shape, binary_name)
        graph_def = onnx.parser.parse_graph(graph_txt)
        weight = helper.make_tensor('weight1', TensorProto.FLOAT, weight_shape, weight_data)

        graph_def.initializer.extend([weight])
        self.onnx_and_test(graph_def)

    def test_Div2Mul(self, case_name):
        self.BinaryWeightBase(case_name, [1, 16, 32, 32], [1, 16, 32, 32], "Div")

    def test_Mul2Scale(self, case_name):
        self.BinaryWeightBase(case_name, [1, 32, 32, 32], [1, 32, 1, 1], "Mul")

    # conv_param: [out_shape, filter_shape, conv_pads]
    def PadConvBase(self, case_name, input_shape, pad_param, conv_param_list):
        add_in_shape = {"input1": input_shape, "input2": input_shape}
        pad_val = np.array(pad_param).astype(np.int64)
        paddings = helper.make_tensor(name='paddings',
                                      data_type=onnx.TensorProto.INT64,
                                      dims=pad_val.shape,
                                      vals=pad_val.flatten())
        graph_txt = """
            %s (float%s input1, float%s input2) => (float%s add1, float%s output1, float%s output2)
            <float filter1, float filter2, int64 paddings>
            {
                add1 = Add(input1, input2)
                pad2 = Pad<mode="constant">(add1, paddings)
                output1 = Conv<kernel_shape=%s, pads=%s>(pad2, filter1)
                output2 = Conv<kernel_shape=%s, pads=%s>(pad2, filter2)
            }
            """ % (case_name, add_in_shape["input1"], add_in_shape["input2"], input_shape,
                   conv_param_list[0][0], conv_param_list[1][0], conv_param_list[0][1][2:],
                   conv_param_list[0][2], conv_param_list[1][1][2:], conv_param_list[1][2])
        graph_def = onnx.parser.parse_graph(graph_txt)

        f_data1 = np.random.randn(*conv_param_list[0][1]).astype(np.float32)
        f_data2 = np.random.randn(*conv_param_list[1][1]).astype(np.float32)
        filter1 = helper.make_tensor('filter1', TensorProto.FLOAT, conv_param_list[0][1], f_data1)
        filter2 = helper.make_tensor('filter2', TensorProto.FLOAT, conv_param_list[1][1], f_data2)

        graph_def.initializer.extend([paddings, filter1, filter2])
        self.onnx_and_test(graph_def)

    def test_PadConv1d(self, case_name):
        self.PadConvBase(case_name, [1, 2, 8], [0, 0, 1, 0, 0, 1],
                         [[[1, 2, 10], [2, 2, 3], [1, 1]], [[1, 2, 8], [2, 2, 3], [0, 0]]])

    def test_PadConv2d(self, case_name):
        self.PadConvBase(case_name, [1, 3, 8, 8], [0, 0, 1, 1, 0, 0, 3, 1],
                         [[[1, 16, 14, 10], [16, 3, 3, 3], [2, 1, 2, 1]],
                          [[1, 16, 10, 8], [16, 3, 3, 3], [0, 0, 0, 0]]])

    def test_PadConv3d(self, case_name):
        self.PadConvBase(case_name, [1, 3, 32, 32, 32], [0, 0, 0, 1, 2, 0, 0, 3, 1, 2],
                         [[[1, 16, 36, 35, 35], [16, 3, 3, 3, 3], [1, 2, 1, 2, 1, 0]],
                          [[1, 16, 33, 32, 34], [16, 3, 3, 3, 3], [0, 0, 0, 0, 0, 0]]])

    # pool_param: [out_shape, kernel_shape, pool_pads]
    def PadPoolBase(self, case_name, input_shape, pad_param, pool_param_list):
        add_in_shape = {"input1": input_shape, "input2": input_shape}
        pad_val = np.array(pad_param).astype(np.int64)
        paddings = helper.make_tensor(name='paddings',
                                      data_type=onnx.TensorProto.INT64,
                                      dims=pad_val.shape,
                                      vals=pad_val.flatten())

        graph_txt = """
            %s (float%s input1, float%s input2) => (float%s add1, float%s output1, float%s output2)
            <int64 paddings>
            {
                add1 = Add(input1, input2)
                pad2 = Pad<mode="constant">(add1, paddings)
                output1 = AveragePool<kernel_shape=%s, pads=%s, strides=%s, count_include_pad=1>(pad2)
                output2 = AveragePool<kernel_shape=%s, pads=%s, strides=%s, count_include_pad=1>(pad2)
            }
            """ % (case_name, add_in_shape["input1"], add_in_shape["input2"], input_shape,
                   pool_param_list[0][0], pool_param_list[1][0], pool_param_list[0][1],
                   pool_param_list[0][2], pool_param_list[0][3], pool_param_list[1][1],
                   pool_param_list[1][2], pool_param_list[1][3])
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([paddings])
        self.onnx_and_test(graph_def)

    def test_PadPool1d(self, case_name):
        self.PadPoolBase(case_name, [1, 3, 256], [0, 0, 2, 0, 0, 2],
                         [[[1, 3, 65], [4], [0, 0], [4]], [[1, 3, 34], [8], [6, 6], [8]]])

    def test_PadPool2d(self, case_name):
        self.PadPoolBase(case_name, [1, 32, 12, 12], [0, 0, 1, 2, 0, 0, 1, 2],
                         [[[1, 32, 6, 7], [4, 4], [0, 0, 0, 0], [2, 2]],
                          [[1, 32, 4, 4], [4, 4], [1, 0, 1, 0], [4, 4]]])

    def test_PadPool3d(self, case_name):
        self.PadPoolBase(case_name, [1, 16, 32, 32, 32], [0, 0, 0, 1, 2, 0, 0, 0, 1, 2],
                         [[[1, 16, 15, 16, 17], [4, 4, 4], [0, 0, 0, 0, 0, 0], [2, 2, 2]],
                          [[1, 16, 9, 9, 9], [4, 4, 4], [2, 1, 0, 2, 1, 0], [4, 4, 4]]])

    def test_ScatterElements(self, case_name):

        input_data = [{
            "data": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        }, {
            "data":
            np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype=np.float32)
        }, {
            "data": np.random.randn(1, 13294, 4).astype(np.float32)
        }, {
            "data": np.arange(256).reshape(4, 4, 4, 4).astype(np.float32)
        }, {
            "data": np.random.randn(16, 17, 18, 19).astype(np.float32)
        }, {
            "data": np.random.randn(16, 17, 18, 19, 20).astype(np.float32)
        }, {
            "data": np.random.randn(15, 16, 17, 18, 19, 20).astype(np.float32)
        }, {
            "data": np.random.randn(2, 2, 2, 2, 2, 2, 2).astype(np.float32)
        }, {
            "data": np.random.randn(2, 2, 2, 2, 2, 2, 2, 2).astype(np.float32)
        }, {
            "data": np.random.randn(2, 65536, 2).astype(np.float32)
        }]

        indices_data = [
            np.array([[2, 1, 0, 2], [1, 2, 0, 1], [0, 2, 1, 0]], dtype=np.int32),
            np.array([[2, 1, 0], [1, 2, 0], [0, 2, 1], [0, 1, 2]], dtype=np.int32),
            np.random.randint(0, 13294, [1, 900, 4]),
            np.random.randint(0, 4, [4, 4, 9, 4]),
            np.random.randint(0, 18, [16, 17, 25, 19]),
            np.random.randint(0, 20, [16, 17, 18, 19, 21]),
            np.random.randint(0, 20, [15, 16, 17, 18, 19, 21]),
            np.random.randint(0, 2, [2, 2, 2, 2, 2, 2, 8]),
            np.random.randint(0, 2, [2, 2, 2, 2, 10, 2, 2, 2]),
            np.random.randint(0, 2, [2, 65536, 2])
        ]

        updates_data = [
            np.array([[2, 1, 0, 2], [1, 2, 0, 1], [0, 2, 1, 0]], dtype=np.float32),
            np.array([[2, 1, 0], [1, 2, 0], [0, 2, 1], [0, 1, 2]], dtype=np.float32),
            np.random.randn(1, 900, 4).astype(np.float32),
            np.random.randn(4, 4, 9, 4).astype(np.float32),
            np.random.randn(16, 17, 25, 19).astype(np.float32),
            np.random.randn(16, 17, 18, 19, 21).astype(np.float32),
            np.random.randn(15, 16, 17, 18, 19, 21).astype(np.float32),
            np.random.randn(2, 2, 2, 2, 2, 2, 8).astype(np.float32),
            np.random.randn(2, 2, 2, 2, 10, 2, 2, 2).astype(np.float32),
            np.random.randn(2, 65536, 2).astype(np.float32)
        ]

        axis_data = [1, 0, 1, 2, 2, 4, 5, 6, 4, 2]
        test_len =  len(input_data)
        if self.simple:
            test_len = 2
        for i in range(len(input_data)):
            # if i != 9:
            #     continue
            data_, indices_, updates_, axis_ = input_data[i]["data"], indices_data[i], updates_data[
                i], axis_data[i]
            input_ = {"data": data_, "indices": indices_, "updates": updates_}
            graph_txt = """
            %s (float%s data, int64%s indices, float%s updates) => (float%s scatter_output)
                {
                    scatter_output = ScatterElements<axis=%d>(data, indices, updates)
                }
                """ % (case_name, list(data_.shape), list(indices_.shape), list(
                updates_.shape), list(data_.shape), axis_)
            graph_def = onnx.parser.parse_graph(graph_txt)
            self.onnx_and_test(graph_def, input_data=input_)

    def test_ScatterND(self, case_name):
        if self.chip in ['bm1684x', 'bm1688', 'cv186x', 'sg2380'] and not self.simple:
            x_shapes = [[320, 320], [1, 3, 128, 128, 186], [2, 3, 20, 40], [1, 13294, 256],
                        [1, 900, 256], [320, 320], [1, 3, 128, 20, 20], [1, 13294, 256],
                        [8, 900, 2], [4, 4, 4]]
            idx_shapes = [[160, 160, 2], [1, 3, 128, 128, 150, 5], [2, 3, 20, 4],
                          [1, 13294, 256, 3], [1, 900, 256, 3], [160, 1], [1, 3, 64, 3],
                          [1, 13294, 2], [8, 1], [4, 4, 4, 3]]
            update_shapes = [[160, 160], [1, 3, 128, 128, 150], [2, 3, 20], [1, 13294, 256],
                             [1, 900, 256], [160, 320], [1, 3, 64, 20, 20], [1, 13294, 256],
                             [8, 900, 2], [4, 4, 4]]
        else:
            x_shapes = [[320, 320], [1, 5, 256]]
            idx_shapes = [[160, 160, 2], [1, 1, 2]]
            update_shapes = [[160, 160], [1, 1, 256]]

        for i in range(0, len(x_shapes)):
            # if i != 4:
            #     continue
            x_shape, idx_shape, update_shape = x_shapes[i], idx_shapes[i], update_shapes[i]
            # indices should not have duplicate entries: that is, if idx1 != idx2, then indices[idx1] != indices[idx2].
            # This ensures that the output value does not depend on the iteration order.
            x_dims = len(x_shape)
            k = idx_shape[-1]
            assert (k <= x_dims)
            # 生成不重复的 indices 数组
            index_num = np.prod(idx_shape[:-1])
            if k == x_dims:
                offset_max = np.prod(x_shape)
            else:
                offset_max = np.prod(x_shape[:-1])

            offset = np.random.choice(offset_max, size=index_num, replace=False)
            indices = np.zeros(idx_shape, dtype=np.int32)
            for i in range(k):
                indices[...,
                        k - i - 1] = (offset % x_shape[k - i - 1]).reshape(indices[...,
                                                                                   k - i - 1].shape)
                offset = offset // x_shape[k - i - 1]

            input_data = {
                "x_data": np.random.rand(*x_shape).astype(np.float32),
                "indices": indices,
                "updates": np.random.rand(*update_shape).astype(np.float32),
            }
            add_data = helper.make_tensor('add_tensor', TensorProto.FLOAT, x_shape,
                                          np.random.rand(*x_shape).astype(np.float32))
            graph_txt = """
                %s (float%s x_data, int64%s indices, float%s updates) => (float%s output)
                <float add_tensor>
                {
                    scatter_output = ScatterND(x_data, indices, updates)
                    output = Add(scatter_output, add_tensor)
                }
                """ % (case_name, x_shape, idx_shape, update_shape, x_shape)
            graph_def = onnx.parser.parse_graph(graph_txt)
            graph_def.initializer.extend([add_data])
            self.onnx_and_test(graph_def, input_data=input_data)

    def test_SliceToReverse(self, case_name):
        input_shape = [100, 1, 3]
        output_shape = [100, 1, 3]
        starts_data = np.array([-1], dtype=np.int64)
        ends_data = np.array([-4], dtype=np.int64)
        axes_data = np.array([2], dtype=np.int64)
        steps_data = np.array([-1], dtype=np.int64)

        starts = helper.make_tensor('starts', TensorProto.INT64, [1], starts_data)
        ends = helper.make_tensor('ends', TensorProto.INT64, [1], ends_data)
        axes = helper.make_tensor('axes', TensorProto.INT64, [1], axes_data)
        steps = helper.make_tensor('steps', TensorProto.INT64, [1], steps_data)

        graph_txt = """
            %s (float%s input) => (float%s output)
            <int64 starts, int64 ends, int64 axes, int64 steps>
            {
                output = Slice(input, starts, ends, axes, steps)
            }
            """ % (case_name, input_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([starts, ends, axes, steps])
        self.onnx_and_test(graph_def)

    def test_TopK4(self, case_name):
        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                x = x + x
                v, _ = torch.topk(x, 256, 2)
                v = v + v
                return v

        x = torch.randn(15, 5, 256).float()
        self.torch_and_test(x, Model(), case_name)

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
            order = transpose_orders[i]
            output_shape = [input_shape[order[i]] for i in range(len(order))]
            graph_txt = """
                %s (float%s input) => (float%s output)
                {
                    output = Transpose<perm=%s>(input)
                }
                """ % (case_name, input_shape, output_shape, order)
            graph_def = onnx.parser.parse_graph(graph_txt)
            self.onnx_and_test(graph_def)

    def test_PermuteToReorg(self, case_name):
        input_shape = [1, 4, 6, 6]
        output_shape = [1, 16, 3, 3]
        rshape1 = [6]
        rshape1_data = np.array([1, 4, 3, 2, 3, 2], dtype=np.int64)
        r1 = helper.make_tensor('shape1', TensorProto.INT64, rshape1, rshape1_data)
        order = [0, 1, 3, 5, 2, 4]
        rshape2 = [4]
        rshape2_data = np.array(output_shape, dtype=np.int64)
        r2 = helper.make_tensor('shape2', TensorProto.INT64, rshape2, rshape2_data)
        graph_txt = """
            %s (float%s input) => (float%s output)
            <int64 shape1, int64 shape2>
            {
                out1 = Reshape(input, shape1)
                out2 = Transpose<perm=%s>(out1)
                output = Reshape(out2, shape2)
            }
            """ % (case_name, input_shape, output_shape, order)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([r1, r2])
        self.onnx_and_test(graph_def)

    def test_PermuteToReorg2(self, case_name):
        input_shape = [1, 16, 200, 200]
        output_shape = [1, 64, 100, 100]
        rshape1 = [6]
        rshape1_data = np.array([1, 16, 100, 2, 100, 2], dtype=np.int64)
        r1 = helper.make_tensor('shape1', TensorProto.INT64, rshape1, rshape1_data)
        order = [0, 1, 3, 5, 2, 4]
        rshape2 = [4]
        rshape2_data = np.array(output_shape, dtype=np.int64)
        r2 = helper.make_tensor('shape2', TensorProto.INT64, rshape2, rshape2_data)
        # add conv define
        filter_shape = [64, 64, 3, 3]
        kernel = [3, 3]
        padding = [1, 1, 1, 1]
        stride = [1, 1]
        dilation = [1, 1]
        weight_data = np.random.randn(*filter_shape).astype(np.float32)
        bias_data = np.random.randn(output_shape[1]).astype(np.float32)
        weight = helper.make_tensor("weight", TensorProto.FLOAT, filter_shape, weight_data)
        bias = helper.make_tensor("bias", TensorProto.FLOAT, list(bias_data.shape), bias_data)
        graph_txt = """
            %s (float%s input) => (float%s output)
            <int64 shape1, int64 shape2, float weight, float bias>
            {
                out1 = Reshape(input, shape1)
                out2 = Transpose<perm=%s>(out1)
                out3 = Reshape(out2, shape2)
                output = Conv<kernel_shape=%s, pads=%s, strides=%s, dilations=%s, group=1>(out3, weight, bias)
            }
            """ % (case_name, input_shape, output_shape, order, kernel, padding, stride, dilation)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([r1, r2, weight, bias])
        self.onnx_and_test(graph_def)

    def test_Permute5dSplit(self, case_name):
        input_shape = [2, 4, 16, 20, 32]
        orders = [[0, 4, 2, 1, 3], [3, 1, 0, 4, 2], [2, 1, 3, 4, 0], [1, 0, 2, 4, 3],
                  [4, 0, 3, 2, 1], [4, 3, 2, 1, 0]]
        for i in range(0, len(orders)):
            order = orders[i]
            if i >= 3 and self.chip.startswith("cv18"):
                # because cv183x backend don't support all permute4d
                continue
            print("order:", order)
            output_shape = [input_shape[order[i]] for i in range(0, len(order))]
            graph_txt = """
                %s_%s (float%s input) => (float%s output)
                {
                    output = Transpose<perm=%s>(input)
                }
                """ % (case_name, i, input_shape, output_shape, order)
            graph_def = onnx.parser.parse_graph(graph_txt)
            self.onnx_and_test(graph_def)

    def test_Permute7d(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                return x.permute(0, 1, 3, 5, 2, 4, 6)

        x = torch.randn(1, 2, 4, 4, 4, 2, 4).float()
        self.torch_and_test(x, Model(), case_name)

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
        import copy
        # initializers
        value = np.array([0], dtype=np.float32)
        zero = from_array(value, name='zero')
        value2 = np.array([0, 1, 2, 3], dtype=np.int64)
        axes = from_array(value2, name='axes')
        input_shape = [1, 3, 100, 10]
        out_shape = copy.deepcopy(input_shape)
        shape = copy.deepcopy(input_shape)
        '''
        Note:at int8 mode, the topk's index out always compare failed,
        so set the K to small number
        '''
        kk = 6
        kkk = 4
        out_shape[-1] = kk
        shape[-1] = kkk
        graph_txt = """
            %s (float[1] X, float%s Y, float%s Z, float%s M, float%s N, float%s J) => (float%s out)
            <float zero = {0}, int64[1] K = {4}>
            {
                cond = Greater(X, zero)
                O = If (cond)<
                    then_branch = then_body() => (float%s then2_out)
                    <int64[1] then_K = {6}>
                    {
                        then_out = Add(Y, Z)
                        Y_Value, Y_Index = TopK<axis=-1, largest=0>(then_out, then_K)
                        then2_out = Mul(Y_Value, N)
                    },
                    else_branch = else_body() => (float%s else2_out)
                    <int64[1] else_k = {6}>
                    {
                        else_out = Sub(Z, M)
                        else_y_value, else_y_index = TopK<axis=-1, largest=0>(else_out, else_k)
                        else2_out = Add(else_y_value, N)
                    }
                >
                out1 = Sub(O, N)
                y_value, y_index = TopK<axis=-1, largest=0>(out1, K)
                out = Add(y_value, J)
            }
            """ % (case_name, input_shape, input_shape, input_shape, out_shape, shape, shape, out_shape,
                   out_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def, static_shape=False)

    def test_Loop(self, case_name):
        graph_txt = """
            %s (float[1] input_0) => (int32[1] up_bound, int32[1] new_start_v2)
            <float[1] const_fold_opt__17 = {10}, int64[1] reshape = {0}, int32[1] init_v = {1}, int32[1] up_value = {30}, int32[1] start_value = {10}, int64[1] while_maximum_iterations_0 = {30}>
            {
                while_cond_158_while_Less_0 = Less(input_0, const_fold_opt__17)
                while_cond_158_while_Squeeze_0 = Squeeze(while_cond_158_while_Less_0, reshape)
                up, new_cout = Loop(while_maximum_iterations_0, while_cond_158_while_Squeeze_0, up_value, start_value)<
                body = while_body(int64 while_while_loop_counter_0, bool cond__15_0, int32[1] while_placeholder_0, int32[1] while_add_const_0_0) => (bool[1] cond___while_Less_0, int32[1] while_placeholder_0, int32[1] while_Identity_2_0)
                <int32[1] step_value = {2}>
                {
                    while_Identity_2_0 = Add(while_add_const_0_0, step_value)
                    cond___while_Less_0 = Less(while_Identity_2_0, while_placeholder_0)
                }
                >
                up_bound = Add(up, init_v)
                new_start_v2 = Sub(new_cout, init_v)
            }
            """ % (case_name)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def, support_modes=["f32", "f16", "bf16"])

    def test_Shape(self, case_name):
        input_shape = [2, 3, 4]
        output_shape = [1]
        graph_txt = """
            %s (float%s input) => (int64%s output)
            <int64 starts, int64 ends, int64 axes, int64 steps>
            {
                shapeinfo = Shape<start=0, end=2>(input)
                output = Slice(shapeinfo, starts, ends, axes, steps)
            }
            """ % (case_name, input_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        starts = helper.make_tensor('starts', TensorProto.INT64, [1], np.array([0], np.int64))
        ends = helper.make_tensor('ends', TensorProto.INT64, [1], np.array([1], np.int64))
        axes = helper.make_tensor('axes', TensorProto.INT64, [1], np.array([0], np.int64))
        steps = helper.make_tensor('steps', TensorProto.INT64, [1], np.array([1], np.int64))
        graph_def.initializer.extend([starts, ends, axes, steps])
        self.onnx_and_test(graph_def, case_name, static_shape=False, version=15)

    def test_ShapeSlice(self, case_name):
        shape = [10, 1000]
        o_shape = list(shape)
        o_shape[-1] = shape[0]
        starts = helper.make_tensor('starts', TensorProto.INT64, [1], np.array([0], np.int64))
        ends = helper.make_tensor('ends', TensorProto.INT64, [1], np.array([1], np.int64))
        axes = helper.make_tensor('axes', TensorProto.INT64, [1], np.array([0], np.int64))
        steps = helper.make_tensor('steps', TensorProto.INT64, [1], np.array([1], np.int64))
        graph_txt = """
            %s (float%s X) => (int64[1] K)
            <int64 starts, int64 ends, int64 axes, int64 steps>
            {
                X_Shape = Shape(X)
                K = Slice(X_Shape, starts, ends, axes, steps)
            }
            """ % (case_name, shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([starts, ends, axes, steps])
        self.onnx_and_test(graph_def, case_name, static_shape=False)

    def test_ShapeCast(self, case_name):
        shape = [10, 1000]
        o_shape = list(shape)
        o_shape[-1] = shape[0]
        starts = helper.make_tensor('starts', TensorProto.INT64, [1], np.array([0], np.int64))
        ends = helper.make_tensor('ends', TensorProto.INT64, [1], np.array([1], np.int64))
        axes = helper.make_tensor('axes', TensorProto.INT64, [1], np.array([0], np.int64))
        steps = helper.make_tensor('steps', TensorProto.INT64, [1], np.array([1], np.int64))
        graph_txt = """
            %s (float%s X) => (float%s Y_Value, int64%s Y_Index)
            <int64 starts, int64 ends, int64 axes, int64 steps>
            {
                X_Shape = Shape(X)
                K = Slice(X_Shape, starts, ends, axes, steps)
                Y_Value, Y_Index = TopK<axis=-1, largest=1>(X, K)
            }
            """ % (case_name, shape, o_shape, o_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([starts, ends, axes, steps])
        self.onnx_and_test(graph_def, case_name, static_shape=False)

    def test_ConstOfShape(self, case_name):
        input_shape = [4, 10, 27, 27]
        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                soutput = Shape(input)
                output = ConstantOfShape(soutput)
            }
            """ % (case_name, input_shape, input_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def, case_name, static_shape=False)

    def test_ShapeUnsqueeze(self, case_name):
        from onnx import numpy_helper
        from onnx.helper import (make_node, make_graph, make_model, make_tensor_value_info)
        input_shape = [2, 3]
        output_shape = [len(input_shape)]
        axes = [0, 2]
        axes_tensor = numpy_helper.from_array(np.array(axes, dtype=np.int64), name="axes")
        unsqueeze_shape = [1, output_shape[0], 1]
        graph_txt = """
            %s (float%s input) => (int64%s unsqueezeinfo)
            <int64 axes>
            {
                shapeinfo = Shape(input)
                unsqueezeinfo = Unsqueeze(shapeinfo, axes)
            }
            """ % (case_name, input_shape, unsqueeze_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([axes_tensor])
        self.onnx_and_test(graph_def, case_name, static_shape=False)

    def test_ShapeSqueeze(self, case_name):
        from onnx import numpy_helper
        from onnx.helper import (make_node, make_graph, make_model, make_tensor_value_info)
        input_shape = [1, 1, 2, 3]
        output_shape = [2, 3]
        axes = [0, 1]
        unsqueeze_shape = [1, 1, output_shape[0]]
        squeeze_shape = [len(input_shape)]
        axes_unsqueeze_tensor = numpy_helper.from_array(np.array(axes, dtype=np.int64),
                                                        name="axes_unsqueeze")
        axes_squeeze_tensor = numpy_helper.from_array(np.array(axes, dtype=np.int64),
                                                      name="axes_squeeze")
        graph_txt = """
            %s (float%s input) => (int64%s squeezeinfo)
            <int64 axes_unsqueeze, int64 axes_squeeze>
            {
                shapeinfo = Shape(input)
                unsqueezeinfo = Unsqueeze(shapeinfo, axes_unsqueeze)
                squeezeinfo = Squeeze(unsqueezeinfo, axes_squeeze)
            }
            """ % (case_name, input_shape, squeeze_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([axes_squeeze_tensor, axes_unsqueeze_tensor])
        self.onnx_and_test(graph_def, case_name, static_shape=False)

    def test_Gather2(self, case_name):
        indices_data = [1]
        input_shape = [2, 3]
        indices = helper.make_tensor(name='indices',
                                     data_type=TensorProto.INT64,
                                     dims=[1],
                                     vals=indices_data)
        graph_txt = """
            %s (float%s input) => (int64 output)
            <int64 indices>
            {
                shapeinfo = Shape(input)
                output = Gather(shapeinfo, indices)
            }
            """ % (case_name, input_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([indices])
        self.onnx_and_test(graph_def, case_name, static_shape=False)

    def test_Gather3(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.index = np.array([15, 16, 17, 21, 22, 23, 27, 28, 29, 66, 67,
                                       68]).astype(np.int64)
                self.axis = 1

            def forward(self, x):
                return torch.index_select(x, self.axis, torch.from_numpy(self.index)) * 2

        x = torch.randn(1, 72, 9).float()
        self.torch_and_test(x, Model(), case_name)

    def test_Concat3(self, case_name):
        input_shape = [2, 3]
        input2 = helper.make_tensor(name='input2',
                                    data_type=TensorProto.INT64,
                                    dims=[2],
                                    vals=[1, 2])
        graph_txt = """
            %s (float%s input) => (int64 output)
            <int64 input2>
            {
                shapeinfo = Shape(input)
                output = Concat<axis=0>(shapeinfo, input2)
            }
            """ % (case_name, input_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        graph_def.initializer.extend([input2])
        self.onnx_and_test(graph_def, case_name, static_shape=False)

    def test_Range(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                x = torch.nonzero(x)
                x_shape = x.shape
                y = torch.arange(1, x_shape[1], dtype=torch.float32)
                return y

        x = torch.randn(4, 8, 32, 32).float()
        self.torch_and_test(x, Model(), case_name, static_shape=False, dynamic=True)

    def test_PenaltySample(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.top_k = 50
                self.min_tokens_to_keep = 5
                self.keep_matrix = torch.zeros((1, self.top_k), dtype=torch.float)
                self.keep_matrix[0, :self.min_tokens_to_keep] = -1

            def forward(self, m_logits, input_ids, top_p, temperature, penalty):
                # repeat penalty
                logits = torch.gather(m_logits, 1, input_ids)
                logits = torch.where(logits < 0, logits * penalty, logits / penalty)
                m_logits.scatter_(1, input_ids, logits)

                # top_k
                logits, token = torch.topk(m_logits.float(), self.top_k)

                # temperature
                logits = logits / temperature

                # top_p
                cumulative_probs = logits.softmax(dim=1).cumsum(dim=1)
                cumulative_probs = cumulative_probs + self.keep_matrix
                filtered_logits = torch.where(cumulative_probs < top_p, logits, torch.FloatTensor([-1.]))
                probs = filtered_logits.softmax(dim=1)
                return probs

        m_logits = torch.randn(1, 512).float()
        input_ids = torch.tensor([range(10)]).long()
        top_p = torch.tensor([0.8]).float()
        temperature = torch.tensor([0.98]).float()
        penalty = torch.tensor([0.98]).float()
        self.torch_and_test((m_logits, input_ids, top_p, temperature, penalty), Model(), case_name)

    def test_PermuteBinary(self, case_name):

        class Model(torch.nn.Module):

            def __init__(self, f):
                super(Model, self).__init__()
                self.f = f

            def forward(self, x1, x2):
                x1 = torch.transpose(x1, 1, 3)
                x2 = torch.transpose(x2, 1, 3)
                y = self.f(x1, x2)
                return y

        x1 = torch.randn(4, 8, 32, 32).float()
        x2 = torch.randn(4, 1, 32, 32).float()
        self.torch_and_test((x1, x2), Model(torch.add), case_name + "Add")
        self.torch_and_test((x1, x2), Model(torch.sub), case_name + "Sub")
        self.torch_and_test((x1, x2), Model(torch.mul), case_name + "Mul")

    def test_CumSum(self, case_name):
        input_shape = [2, 3, 4, 5]
        for d in range(len(input_shape)):
            dim_input = helper.make_tensor(name="dim",
                                           data_type=onnx.TensorProto.INT64,
                                           dims=[1],
                                           vals=np.array([d], dtype=np.int64))
            graph_txt = """
                %s (float%s input) => (float%s output)
                <int64 dim>
                {
                    output = CumSum(input, dim)
                }
                """ % (case_name, input_shape, input_shape)
            graph_def = onnx.parser.parse_graph(graph_txt)
            graph_def.initializer.extend([dim_input])
            input_data = {
                "input": np.random.uniform(-100, 100, size=input_shape).astype(np.float32)
            }
            self.onnx_and_test(graph_def, input_data=input_data)

    def test_Round(self, case_name):
        input_shape = [1, 16, 64, 64]
        output_shape = [1, 16, 64, 64]
        graph_txt = """
            %s (float%s input) => (float%s output)
            {
                output = Round(input)
            }
            """ % (case_name, input_shape, output_shape)
        graph_def = onnx.parser.parse_graph(graph_txt)
        self.onnx_and_test(graph_def)

    def user_define_net(self, case_name):
        """user_define_net"""

        self.opt = 2
        # from tools.train.resnet import resnet18
        # model = resnet18()
        # from tools.train.resnet import resnet50
        # model = resnet50()

        # print('start test mobilenet_v2')
        # import torchvision.models as models
        # x = torch.randn(1,3,224,224).float()
        # model = models.mobilenet_v2()
        # self.torch_and_test(x, model, "mobilenet_v2")

        # print('start test resnet50')
        # x = torch.randn(4,3,224,224).float()
        # from tools.train.resnet import resnet50
        # model = resnet50()
        # self.torch_and_test(x, model, "user_define_net")

        # print('start test resnet18')
        # x = torch.randn(1,3,224,224).float()
        # from tools.train.resnet import resnet18
        # model = resnet18()
        # self.torch_and_test(x, model, "user_define_net")

        # print('start test test_model0')
        # x = torch.randn(4,3,224,224).float()
        # from tools.train.test_model import test_model0
        # model = test_model0()
        # self.torch_and_test(x, model, "user_define_net")

        # print('start test test_model1')
        # x = torch.randn(4,3,110,110).float()
        # from tools.train.test_model import test_model1
        # model = test_model1()
        # self.torch_and_test(x, model, "user_define_net")

        # x = torch.randn(1,3,224,224).float()
        # from tools.train.test_model import test_model1
        # model = test_model1()
        # self.torch_and_test(x, model, "user_define_net")

        # print('start test test_model2')
        # x = torch.randn(1,3,224,224).float()
        # from tools.train.test_model import test_model2
        # model = test_model2()
        # self.torch_and_test(x, model, "user_define_net")

        # print('start test test_model3')
        # x = torch.randn(1,3,224,224).float() #中间输出功能有误
        # from tools.train.test_model import test_model3
        # model = test_model3()
        # self.torch_and_test(x, model, "test_model3")

        # print('start test test_model4')
        # from tools.train.test_model import test_model4
        # x = torch.randn(1,3,224,224).float()
        # model = test_model4()
        # self.torch_and_test(x, model, "test_model4")

        print('start test test_model5')
        from tools.train.test_model import test_model5
        x = torch.randn(1,3,224,224).float()
        model = test_model5()
        self.torch_and_test(x, model, "test_model5")

        # print('start test test_model6')
        # from tools.train.test_model import test_model6
        # model = test_model6()
        # self.trace_and_test([(1,3,224,224)], model)

        # print('start test test_model7')
        # from tools.train.test_model import test_model7
        # model = test_model7()
        # self.trace_and_test([(1,3,224,224), (1,3,224,224)], model)

        # d_model=10 #768  #test ok at 0918
        # from tools.train.gpt2 import TransformerBlocks #backward can not use this
        # mod = TransformerBlocks(d_model=d_model, nlayers=2) #.train()
        # self.trace_and_test([(1,4,d_model)], mod)

def test_one_case_in_all(tester: ONNX_IR_TESTER, case, error_cases, success_cases):
    t = Timer()
    try:
        tester.test_single(case)
    except:
        error_cases.append("{}:{}s".format(case, int(t.elapsed_time())))
        return
    success_cases.append("{}:{}s".format(case, int(t.elapsed_time())))


def test_int4(tester: ONNX_IR_TESTER):
    tester.chip = "bm1688"
    tester.mode = "int4"
    tester.dynamic = False
    tester.simple = False
    tester.quant_modes = ["int4"]
    tester.support_asym = [False]
    Y, N = True, False
    test_cases = {
        "Conv2d": (tester.test_Conv2d, N, N, Y, N),
        "MatMul": (tester.test_MatMul, N, N, Y, N),
        "MatMul2": (tester.test_MatMul2, N, N, Y, N),
    }
    if tester.multithread:
        import multiprocessing
        from utils.misc import collect_process
        process_number = multiprocessing.cpu_count() // 2 + 1
        processes = []
        error_cases = multiprocessing.Manager().list()
        success_cases = multiprocessing.Manager().list()
        for case in test_cases:
            if tester.check_support(case):
                p = multiprocessing.Process(target=test_one_case_in_all,
                                            name=case,
                                            args=(tester, case, error_cases, success_cases))
                processes.append(p)
            if len(processes) == process_number:
                collect_process(processes, error_cases)
                processes = []
        if processes:
            collect_process(processes, error_cases)
            processes = []
    else:
        error_cases = []
        success_cases = []
        for case in test_cases:
            if tester.check_support(case):
                test_one_case_in_all(tester, case, error_cases, success_cases)
    print("Success: {}".format(success_cases))
    print("Failure: {}".format(error_cases))
    if error_cases:
        print("====== test_onnx.py --chip {} TEST Failed ======".format(tester.chip))
    else:
        print("====== test_onnx.py --chip {} TEST Success ======".format(tester.chip))
    return error_cases


def test_fp8(tester: ONNX_IR_TESTER):
    tester.chip = "bm1690"
    tester.mode = "f8"
    tester.dynamic = False
    tester.simple = False
    tester.quant_modes = ["f8e4m3", "f8e5m2"]
    tester.support_asym = [False]
    Y, N = True, False
    test_cases_maxidx = len(tester.test_cases["Add"])-1
    tester.test_cases = {
        # case: [test,     bm1690_support]
        "Add": [tester.test_Add, Y],
        "AddBcast": [tester.test_AddBcast, N],
        "AddWeight": [tester.test_AddWeight, Y],
        "AvgPool1d": [tester.test_AvgPool1d, N],
        "AvgPool2d": [tester.test_AvgPool2d, N],
        "AvgPool3d": [tester.test_AvgPool3d, N],
        "Conv2d": [tester.test_Conv2d, Y],
        "Gather": [tester.test_Gather, Y],
        "GlobalAveragePool": [tester.test_GlobalAveragePool, Y],
        "MatMul": [tester.test_MatMul, N],
        "MatMul2": [tester.test_MatMul2, N],
        "MaxPool1d": [tester.test_MaxPool1d, Y],
        "MaxPool2d": [tester.test_MaxPool2d, Y],
        "MaxPool3d": [tester.test_MaxPool3d, Y],
        "Mul": [tester.test_Mul, Y],
        "MulConst": [tester.test_MulConst, Y],
        "Transpose": [tester.test_Transpose, N],
        "Reshape": [tester.test_Reshape, Y],
        "Scale": [tester.test_Scale, N],
        "Squeeze": [tester.test_Squeeze, N],
        "Sub": [tester.test_Sub, Y],
        "SubConst": [tester.test_SubConst, Y],
        "Unsqueeze": [tester.test_Unsqueeze, Y],
    }
    bm1690_idx = 5
    bm1690_pre = bm1690_idx -1
    for name in tester.test_cases.keys():
        tester.test_cases[name] = tester.test_cases[name][:1] + \
            [N] * bm1690_pre + tester.test_cases[name][1:] + [N] * (test_cases_maxidx - bm1690_idx)
    if tester.multithread:
        import multiprocessing
        from utils.misc import collect_process
        process_number = multiprocessing.cpu_count() // 2 + 1
        processes = []
        error_cases = multiprocessing.Manager().list()
        success_cases = multiprocessing.Manager().list()
        for case in tester.test_cases:
            if tester.check_support(case):
                p = multiprocessing.Process(target=test_one_case_in_all,
                                            name=case,
                                            args=(tester, case, error_cases, success_cases))
                processes.append(p)
            if len(processes) == process_number:
                collect_process(processes, error_cases)
                processes = []
        if processes:
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
        print("====== test_onnx.py --chip {} FP8 TEST Failed ======".format(tester.chip))
    else:
        print("====== test_onnx.py --chip {} FP8 TEST Success ======".format(tester.chip))
    return error_cases

def test_all(tester: ONNX_IR_TESTER):
    if tester.multithread:
        import multiprocessing
        from utils.misc import collect_process
        process_number = multiprocessing.cpu_count() // 2 + 1
        processes = []
        error_cases = multiprocessing.Manager().list()
        success_cases = multiprocessing.Manager().list()
        for case in tester.test_cases:
            if tester.check_support(case):
                print("====== test_onnx.py --case {} --chip {} TEST START PROCESSING ======".format(
                    case, tester.chip))
                p = multiprocessing.Process(target=test_one_case_in_all,
                                            name=case,
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
        print("====== test_onnx.py --chip {} TEST Failed ======".format(tester.chip))
        # exit(1)
    else:
        print("====== test_onnx.py --chip {} TEST Success ======".format(tester.chip))
    clean_kmp_files()
    return error_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--chip", default="bm1684x", type=str,
                        choices=['bm1684', 'bm1684x', 'bm1688', 'cv183x', 'cv182x', 'cv181x', 'cv180x', 'cv186x', 'bm1690', 'sg2380', 'mars3'],
                        help="chip platform name")
    parser.add_argument("--case", default="all", type=str, help="test one case, if all, then test all cases")
    parser.add_argument("--mode", default="all", type=str, choices=['all', 'f32', 'f16', 'bf16', 'int8', 'int4', 'f8', 'f8e4m3', 'f8e5m2'],
                        help="quantize modes")
    parser.add_argument("--dynamic", action="store_true", help='do dynamic compile')
    parser.add_argument("--debug", action="store_true", help='keep middle file if debug')
    parser.add_argument("--simple", action="store_true", help='do simple test for commit test')
    parser.add_argument("--disable_thread", action="store_true", help='do test without multi thread')
    parser.add_argument("--num_core", default=1, type=int, help='The numer of TPU cores used for parallel computation')
    parser.add_argument("--show_all", action="store_true", help='show all cases')
    parser.add_argument("--debug_cmd", default="", type=str, help="debug_cmd")
    parser.add_argument("--cuda", action="store_true", help="test cuda inference")
    # yapf: enable
    args = parser.parse_args()
    tester = ONNX_IR_TESTER(args.chip, args.mode, args.dynamic, args.simple, args.disable_thread,
                            args.num_core, args.debug_cmd, args.cuda)
    if args.show_all:
        print("====== Show All Cases ============")
        for case in tester.test_cases:
            print(case)
        exit(0)
    dir = "onnx_test_{}".format(args.chip)
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)
    if args.mode == 'int4':
        test_int4(tester)
    elif args.mode == 'f8':
        test_fp8(tester)
    elif args.case == "" or args.case.lower() == "all":
        test_all(tester)
    else:
        tester.test_single(args.case)
    if args.debug == False:
        file_clean()
