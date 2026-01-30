#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import os
import sys
import gc
import re
import time
import copy
import math
import numpy as np
import pymlir
import torch
from ctypes import *
from tqdm import tqdm
import datetime
from utils.preprocess import preprocess
from utils.mlir_parser import *
from utils.misc import *
#import graphviz as gz
from math import *
from collections import Counter
from scipy import spatial
from .utils import *

pymlir.set_mem_mode("force_value_mem")

sub_blocks = {
    "eva_block": [
        'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.Reshape', 'top.Slice', 'top.Slice',
        'top.Slice', 'top.Squeeze', 'top.Squeeze', 'top.Squeeze', 'top.Permute', 'top.Slice',
        'top.Slice', 'top.Mul', 'top.Slice', 'top.MulConst', 'top.Slice', 'top.Unsqueeze',
        'top.Unsqueeze', 'top.Concat', 'top.Reshape', 'top.Mul', 'top.Add', 'top.Concat',
        'top.Permute', 'top.Slice', 'top.Slice', 'top.Mul', 'top.Slice', 'top.MulConst',
        'top.Slice', 'top.Unsqueeze', 'top.Unsqueeze', 'top.Concat', 'top.Reshape', 'top.Mul',
        'top.Add', 'top.Concat', 'top.MulConst', 'top.Permute', 'top.MatMul', 'top.Softmax',
        'top.Permute', 'top.MatMul', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Add',
        'top.LayerNorm', 'top.MatMul', 'top.MatMul', 'top.SiLU', 'top.Mul', 'top.MatMul'
    ],
    "eva02_block": [
        "top.LayerNorm", "top.MatMul", "top.MatMul", "top.MatMul", "top.Reshape", "top.Permute",
        "top.Reshape", "top.Permute", "top.Reshape", "top.Permute", "top.Rope", "top.Rope",
        "top.Permute", "top.MulConst", "top.MulConst", "top.MatMul", "top.Softmax", "top.MatMul",
        "top.Permute", "top.Reshape", "top.MatMul", "top.Add", "top.LayerNorm", "top.MatMul",
        "top.MatMul", "top.SiLU", "top.Mul", "top.LayerNorm", "top.MatMul", "top.Add"
    ],
    "eva02_smc_block": [
        "top.LayerNorm", "top.MatMul", "top.MatMul", "top.MatMul", "top.Reshape", "top.Permute",
        "top.Reshape", "top.Permute", "top.Reshape", "top.Permute", "top.Rope", "top.Rope",
        "top.Permute", "top.MulConst", "top.MulConst", "top.MatMul", "top.Softmax", "top.Mul",
        "top.MatMul", "top.Mul", "top.Permute", "top.Reshape", "top.MatMul", "top.Add",
        "top.LayerNorm", "top.MatMul", "top.MatMul", "top.SiLU", "top.Mul", "top.LayerNorm",
        "top.MatMul", "top.Add"
    ],
    "eva02_block_v1": [
        "top.Add", "top.LayerNorm", "top.MatMul", "top.Reshape", "top.Slice", "top.Slice",
        "top.Slice", "top.Squeeze", "top.Squeeze", "top.Squeeze", "top.Permute", "top.Rope",
        "top.Permute", "top.Rope", "top.MulConst", "top.Permute", "top.MatMul", "top.Softmax",
        "top.Permute", "top.MatMul", "top.Permute", "top.Reshape", "top.MatMul", "top.Add",
        "top.LayerNorm", "top.MatMul", "top.MatMul", "top.SiLU", "top.Mul", "top.MatMul"
    ],
    "bert_block": [
        'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.MatMul', 'top.MatMul', 'top.Reshape',
        'top.Permute', 'top.Reshape', 'top.Reshape', 'top.Permute', 'top.Permute', 'top.MatMul',
        'top.MulConst', 'top.Add', 'top.Softmax', 'top.MatMul', 'top.Permute', 'top.Reshape',
        'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.GELU', 'top.MatMul'
    ],
    "bert_block_1": [
        'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.MatMul', 'top.Reshape', 'top.MatMul',
        'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Permute', 'top.Permute', 'top.MatMul',
        'top.MulConst', 'top.Add', 'top.Softmax', 'top.MatMul', 'top.Permute', 'top.Reshape',
        'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.GELU', 'top.MatMul'
    ],
    "bert_block_large_pt": [
        'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.MatMul', 'top.Reshape', 'top.Permute',
        'top.MatMul', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Permute', 'top.Permute',
        'top.MatMul', 'top.MulConst', 'top.Add', 'top.Softmax', 'top.MatMul', 'top.Permute',
        'top.Reshape', 'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.GELU',
        'top.MatMul'
    ],
    "insert_mul_bert_block": [
        'top.Add', 'top.LayerNorm', 'top.Mul', 'top.MatMul', 'top.MatMul', 'top.MatMul',
        'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Reshape', 'top.Permute', 'top.Permute',
        'top.MatMul', 'top.MulConst', 'top.Add', 'top.Softmax', 'top.MatMul', 'top.Permute',
        'top.Reshape', 'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.Mul', 'top.MatMul',
        'top.GELU', 'top.MatMul'
    ],
    "insert_mul_bert_block_1": [
        'top.Add', 'top.LayerNorm', 'top.Mul', 'top.MatMul', 'top.MatMul', 'top.Reshape',
        'top.MatMul', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Permute', 'top.Permute',
        'top.MatMul', 'top.MulConst', 'top.Add', 'top.Softmax', 'top.MatMul', 'top.Permute',
        'top.Reshape', 'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.Mul', 'top.MatMul',
        'top.GELU', 'top.MatMul'
    ],
    "insert_mul_bert_block_2": [
        'top.Add', 'top.LayerNorm', 'top.Mul', 'top.MatMul', 'top.MatMul', 'top.Reshape',
        'top.Permute', 'top.MatMul', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Permute',
        'top.Permute', 'top.MatMul', 'top.MulConst', 'top.Add', 'top.Softmax', 'top.MatMul',
        'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.Mul',
        'top.MatMul', 'top.GELU', 'top.MatMul'
    ],
    "deit_block": [
        'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.Reshape', 'top.Slice', 'top.Squeeze',
        'top.Slice', 'top.Squeeze', 'top.Slice', 'top.Squeeze', 'top.Permute', 'top.Permute',
        'top.Permute', 'top.MatMul', 'top.MulConst', 'top.Softmax', 'top.Permute', 'top.MatMul',
        'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.MatMul',
        'top.GELU', 'top.MatMul'
    ],
    "swin_block": [
        'top.LayerNorm', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape',
        'top.Slice', 'top.Slice', 'top.Slice', 'top.Squeeze', 'top.Squeeze', 'top.Squeeze',
        'top.Permute', 'top.MulConst', 'top.Permute', 'top.Permute', 'top.MatMul', 'top.Add',
        'top.Softmax', 'top.Permute', 'top.MatMul', 'top.Permute', 'top.Reshape', 'top.MatMul',
        'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Add', 'top.Reshape', 'top.LayerNorm',
        'top.MatMul', 'top.GELU', 'top.MatMul', 'top.Add', 'top.Reshape', 'top.LayerNorm',
        'top.Slice', 'top.Slice', 'top.Concat', 'top.Slice', 'top.Slice', 'top.Concat',
        'top.Reshape', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape', 'top.Slice',
        'top.Slice', 'top.Slice', 'top.Squeeze', 'top.Squeeze', 'top.Squeeze', 'top.Permute',
        'top.MulConst', 'top.Permute', 'top.Permute', 'top.MatMul', 'top.Add', 'top.Softmax',
        'top.Permute', 'top.MatMul', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape',
        'top.Permute', 'top.Reshape', 'top.Slice', 'top.Slice', 'top.Concat', 'top.Slice',
        'top.Slice', 'top.Concat', 'top.Add', 'top.Reshape', 'top.LayerNorm', 'top.MatMul',
        'top.GELU', 'top.MatMul', 'top.Add'
    ],
    "_swin_block": [
        'top.LayerNorm', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape',
        'top.Slice', 'top.Slice', 'top.Slice', 'top.Squeeze', 'top.Squeeze', 'top.Squeeze',
        'top.Permute', 'top.MulConst', 'top.Permute', 'top.Permute', 'top.MatMul', 'top.Add',
        'top.Softmax', 'top.Permute', 'top.MatMul', 'top.Permute', 'top.Reshape', 'top.MatMul',
        'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Reshape', 'top.Add', 'top.LayerNorm',
        'top.MatMul', 'top.GELU', 'top.MatMul', 'top.Add', 'top.Reshape', 'top.LayerNorm',
        'top.Slice', 'top.Slice', 'top.Concat', 'top.Slice', 'top.Slice', 'top.Concat',
        'top.Reshape', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape', 'top.Slice',
        'top.Slice', 'top.Slice', 'top.Squeeze', 'top.Squeeze', 'top.Squeeze', 'top.Permute',
        'top.MulConst', 'top.Permute', 'top.Permute', 'top.MatMul', 'top.Add', 'top.Softmax',
        'top.Permute', 'top.MatMul', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape',
        'top.Permute', 'top.Reshape', 'top.Slice', 'top.Slice', 'top.Concat', 'top.Slice',
        'top.Slice', 'top.Concat', 'top.Reshape', 'top.Add', 'top.LayerNorm', 'top.MatMul',
        'top.GELU', 'top.MatMul', 'top.Add'
    ],
    "vit_block": [
        'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.Reshape', 'top.Slice', 'top.Slice',
        'top.Slice', 'top.Squeeze', 'top.Squeeze', 'top.Squeeze', 'top.Permute', 'top.MulConst',
        'top.Permute', 'top.Permute', 'top.MatMul', 'top.Softmax', 'top.Permute', 'top.MatMul',
        'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.MatMul',
        'top.GELU', 'top.MatMul'
    ],
    "detr_block": [
        'top.Add', 'top.LayerNorm', 'top.Add', 'top.MatMul', 'top.MatMul', 'top.MatMul',
        'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Reshape', 'top.Permute', 'top.MulConst',
        'top.Permute', 'top.MatMul', 'top.Softmax', 'top.MatMul', 'top.Permute', 'top.Reshape',
        'top.MatMul', 'top.Reshape', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.MatMul'
    ],
    "detr_rc_50_block": [
        'top.Add', 'top.LayerNorm', 'top.Add', 'top.MatMul', 'top.MatMul', 'top.MatMul',
        'top.Reshape', 'top.Permute', 'top.MulConst', 'top.Reshape', 'top.Permute', 'top.Reshape',
        'top.Permute', 'top.MatMul', 'top.Softmax', 'top.MatMul', 'top.Permute', 'top.Reshape',
        'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.MatMul'
    ],
    "swin_1_block": [
        'top.LayerNorm', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape',
        'top.Permute', 'top.Permute', 'top.Slice', 'top.Slice', 'top.Reshape', 'top.Slice',
        'top.Reshape', 'top.MulConst', 'top.Reshape', 'top.Permute', 'top.MatMul', 'top.Add',
        'top.Softmax', 'top.MatMul', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape',
        'top.Permute', 'top.Reshape', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.GELU',
        'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.Reshape', 'top.Slice', 'top.Slice',
        'top.Concat', 'top.Slice', 'top.Slice', 'top.Concat', 'top.Reshape', 'top.Permute',
        'top.Reshape', 'top.MatMul', 'top.Reshape', 'top.Permute', 'top.Permute', 'top.Slice',
        'top.Slice', 'top.Reshape', 'top.Slice', 'top.Reshape', 'top.MulConst', 'top.Reshape',
        'top.Permute', 'top.MatMul', 'top.Add', 'top.Softmax', 'top.MatMul', 'top.Permute',
        'top.Reshape', 'top.MatMul', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Slice',
        'top.Slice', 'top.Concat', 'top.Slice', 'top.Slice', 'top.Concat', 'top.Reshape', 'top.Add',
        'top.LayerNorm', 'top.MatMul', 'top.GELU', 'top.MatMul', 'top.Add'
    ],
    "_swin_1_block": [
        'top.LayerNorm', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape',
        'top.Slice', 'top.Slice', 'top.Slice', 'top.Squeeze', 'top.Squeeze', 'top.Squeeze',
        'top.Permute', 'top.MulConst', 'top.Permute', 'top.Permute', 'top.MatMul', 'top.Add',
        'top.Softmax', 'top.Permute', 'top.MatMul', 'top.Permute', 'top.Reshape', 'top.MatMul',
        'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Add', 'top.LayerNorm', 'top.MatMul',
        'top.GELU', 'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.Reshape', 'top.Slice',
        'top.Slice', 'top.Concat', 'top.Slice', 'top.Slice', 'top.Concat', 'top.Reshape',
        'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape', 'top.Slice', 'top.Slice',
        'top.Slice', 'top.Squeeze', 'top.Squeeze', 'top.Squeeze', 'top.Permute', 'top.MulConst',
        'top.Permute', 'top.Permute', 'top.MatMul', 'top.Add', 'top.Softmax', 'top.Permute',
        'top.MatMul', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape', 'top.Permute',
        'top.Reshape', 'top.Slice', 'top.Slice', 'top.Concat', 'top.Slice', 'top.Slice',
        'top.Concat', 'top.Reshape', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.GELU',
        'top.MatMul', 'top.Add'
    ],
    "_eva_block": [
        'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.Reshape', 'top.Permute', 'top.MatMul',
        'top.Reshape', 'top.Permute', 'top.MatMul', 'top.Reshape', 'top.Permute', 'top.Slice',
        'top.Slice', 'top.Mul', 'top.Slice', 'top.MulConst', 'top.Slice', 'top.Unsqueeze',
        'top.Unsqueeze', 'top.Concat', 'top.Reshape', 'top.Mul', 'top.Add', 'top.Concat',
        'top.Slice', 'top.Slice', 'top.Mul', 'top.Slice', 'top.MulConst', 'top.Slice',
        'top.Unsqueeze', 'top.Unsqueeze', 'top.Concat', 'top.Reshape', 'top.Mul', 'top.Add',
        'top.Concat', 'top.MulConst', 'top.Permute', 'top.MatMul', 'top.Softmax', 'top.MatMul',
        'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.MatMul',
        'top.MatMul', 'top.SiLU', 'top.Mul', 'top.LayerNorm', 'top.MatMul'
    ],
    "cswin_block": [
        'top.LayerNorm', 'top.MatMul', 'top.Reshape', 'top.Permute', 'top.Slice', 'top.Reshape',
        'top.Slice', 'top.Reshape', 'top.Slice', 'top.Reshape', 'top.Permute', 'top.Reshape',
        'top.Permute', 'top.Reshape', 'top.Permute', 'top.Permute', 'top.Reshape', 'top.Permute',
        'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Conv',
        'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Permute', 'top.MulConst', 'top.Permute',
        'top.MatMul', 'top.Softmax', 'top.MatMul', 'top.Add', 'top.Permute', 'top.Reshape',
        'top.Permute', 'top.Reshape', 'top.Slice', 'top.Reshape', 'top.Slice', 'top.Reshape',
        'top.Slice', 'top.Reshape', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Permute',
        'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Conv', 'top.Reshape', 'top.Permute',
        'top.Reshape', 'top.Permute', 'top.MulConst', 'top.Permute', 'top.MatMul', 'top.Softmax',
        'top.MatMul', 'top.Add', 'top.Permute', 'top.Reshape', 'top.Permute', 'top.Reshape',
        'top.Concat', 'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.GELU',
        'top.MatMul', 'top.Add'
    ],
    "yolo_block": ['top.MaxPool', 'top.MaxPool', 'top.MaxPool', 'top.Concat'],
    "yolo_block_12": [
        'top.Sub', 'top.Add', 'top.Add', 'top.Sub', 'top.MulConst', 'top.Concat', 'top.Mul',
        'top.Concat'
    ],
    "clip_m2_encoder_block": [
        'top.LayerNorm', 'top.MatMul', 'top.MatMul', 'top.MatMul', 'top.Reshape', 'top.Permute',
        'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Permute', 'top.Permute', 'top.MatMul',
        'top.MulConst', 'top.Softmax', 'top.MatMul', 'top.Permute', 'top.Reshape', 'top.LayerNorm',
        'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.GELU', 'top.LayerNorm',
        'top.MatMul', 'top.Add'
    ]
}
openclip_blocks = {
    'openclip_vision_block': [
        'top.LayerNorm', 'top.MatMul', 'top.MatMul', 'top.Reshape', 'top.Permute', 'top.MatMul',
        'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Reshape',
        'top.Reshape', 'top.Permute', 'top.MatMul', 'top.Softmax', 'top.MatMul', 'top.Reshape',
        'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.MatMul',
        'top.MulConst', 'top.Sigmoid', 'top.Mul', 'top.MatMul', 'top.Add'
    ],
    'openclip_text_block': [
        'top.LayerNorm', 'top.MatMul', 'top.MatMul', 'top.Reshape', 'top.Permute', 'top.MatMul',
        'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Reshape',
        'top.Reshape', 'top.Permute', 'top.MatMul', 'top.Reshape', 'top.Add', 'top.Add',
        'top.Reshape', 'top.Softmax', 'top.MatMul', 'top.Reshape', 'top.Permute', 'top.Reshape',
        'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.MulConst', 'top.Sigmoid',
        'top.Mul', 'top.MatMul', 'top.Add'
    ],
    'l2_norm_block': ['top.Abs', 'top.Mul', 'top.Reduce', 'top.Sqrt', 'top.Div'],
}

# "detr_pattern": ['top.Conv', 'top.Scale', 'top.Conv', 'top.Scale', 'top.Conv', 'top.Scale', 'top.Add']

N_mode = [
    'top.Relu', 'top.MaxPool', 'top.Conv', 'top.MatMul', 'top.PRelu', 'top.AvgPool', 'top.Add',
    'top.Sigmoid', 'top.Deconv'
]
H_mode = ['top.Conv', 'top.MatMul', 'top.AvgPool', 'top.Deconv', 'top.MaxPool']


class MatchPattern:

    def __init__(self, args):
        self.args = args
        self.fp32_mlir = args.mlir_file
        self.chip = args.chip
        self.cali_table_name = args.calibration_table
        self.module = pymlir.module()
        self.module.load(args.mlir_file)
        self.parser = MlirParser(args.mlir_file)
        self.cali_method = args.cali_method
        self.log_level = "DEBUG" if args.debug_log else "INFO"
        self._logs = []

        if self.args.part_quantize == 'N_mode':
            self.quantize_ops = N_mode
        elif self.args.part_quantize == 'H_mode':
            self.quantize_ops = H_mode
        elif self.args.part_quantize == 'custom_mode':
            self.quantize_ops = ['top.' + op for op in self.args.custom_operator]
        else:
            pass

    def _log(self, msg: str):
        self._logs.append(msg)
        if self.args.debug_log:
            print(msg)

    def run(self):
        flag = 0
        count = 0
        type_tensors = []
        fp_layer_list = []
        all_tensors = self.parser.get_op_name_list()
        num_tensors = len(all_tensors)
        model_block_name = None
        append_unduplicated = lambda lst, item: lst.append(item) if item not in lst else None

        if self.args.part_quantize:
            for j in range(num_tensors):
                op_type = self.parser.get_op_type_by_op_name(all_tensors[j])
                if op_type in self.quantize_ops:
                    pass
                else:
                    append_unduplicated(fp_layer_list, all_tensors[j])

        for i in range(num_tensors):
            op_type = self.parser.get_op_type_by_op_name(all_tensors[i])
            type_tensors.append(op_type)
        type_tensors_str = ''.join(map(str, type_tensors))
        for name, sub_block in sub_blocks.items():
            sub_str = ''.join(map(str, sub_block))
            if sub_str in type_tensors_str:
                count = type_tensors_str.count(sub_str)
                flag = 1
                model_block_name = name
                self._log(
                    f"{sub_block} (Name: {name}) is a subset of the main list. Count: {count}")
                break
        else:
            openclip_block_counts = {
                name: type_tensors_str.count(''.join(map(str, sub_block)))
                for name, sub_block in openclip_blocks.items()
            }
            if all(count for count in
                   openclip_block_counts.values()) and openclip_block_counts['l2_norm_block'] == 2:
                flag = 1
                model_block_name = 'openclip_block'
                for name, sub_block in openclip_blocks.items():
                    count = openclip_block_counts[name]
                    self._log(
                        f"{sub_block} (Name: {name}) is a subset of the main list. Count: {count}")
                last_matmul_index = num_tensors - type_tensors_str[
                    type_tensors_str.rfind('top.MatMul'):].count('top')
                first_text_block_index = type_tensors_str[:type_tensors_str.index(''.join(
                    openclip_blocks['openclip_text_block']))].count('top')
                first_text_mlp_start_index = first_text_block_index + 27  # 27 is the index of the fisrt mlp matmul in openclip text block
                first_text_mlp_end_index = first_text_block_index + 31  # 31 is the index of the second mlp matmul in openclip text block

        split_fuse_fp_layer_list = []
        if flag == 1:
            if model_block_name == 'yolo_block' or model_block_name == 'yolo_block_12':
                for op_name in all_tensors:
                    # detect the last conv of yolo head
                    # the pre op of the last conv should be SiLU or LeakyRelu
                    # the next op of the last conv should be Concat or Reshape
                    # the next op of the next op should not be Conv
                    op_type = self.parser.get_op_type_by_op_name(op_name)
                    if op_type != 'top.Conv': continue
                    next_ops = self.parser.get_next_op_by_op_name(op_name)
                    pre_ops = self.parser.get_pre_op_by_op_name(op_name)
                    if len(next_ops) != 1 or len(pre_ops) != 1: continue
                    next_op_type = self.parser.get_op_type_by_op_name(next_ops[0])
                    pre_op_type = self.parser.get_op_type_by_op_name(pre_ops[0])
                    if next_op_type not in ['top.Concat', 'top.Reshape']:
                        continue
                    if pre_op_type not in ['top.SiLU', 'top.LeakyRelu']:
                        continue
                    next_next_ops = self.parser.get_next_op_by_op_name(next_ops[0])
                    nn_op_types = [self.parser.get_op_type_by_op_name(_) for _ in next_next_ops]
                    if any(_ == 'top.Conv' for _ in nn_op_types):
                        continue
                    # recursively find all the ops after the last conv
                    ops_after_last_conv = next_ops
                    while ops_after_last_conv:
                        current_op = ops_after_last_conv.pop(0)
                        if current_op in fp_layer_list:
                            continue
                        append_unduplicated(fp_layer_list, current_op)
                        next_ops = self.parser.get_next_op_by_op_name(current_op)
                        ops_after_last_conv.extend(next_ops)

                for item in fp_layer_list:
                    if not is_fuseop(item):
                        if isinstance(item, (list, tuple)):
                            split_fuse_fp_layer_list.extend(item)
                        else:
                            split_fuse_fp_layer_list.append(item)
                    else:
                        new_ops = split_fuseop[item]
                        for op in new_ops:
                            if op not in fp_layer_list:
                                split_fuse_fp_layer_list.append(op)
                return split_fuse_fp_layer_list, flag, self._logs

            for i in range(num_tensors):
                op_type = self.parser.get_op_type_by_op_name(all_tensors[i])
                if op_type == 'top.LayerNorm':
                    pre_op = self.parser.get_pre_op_by_op_name(all_tensors[i])
                    if len(pre_op) == 1 and self.parser.get_op_type_by_op_name(
                            pre_op[0]) == 'top.Add':
                        append_unduplicated(fp_layer_list, pre_op[0])
                if op_type == 'top.SiLU' or op_type == 'top.GELU':
                    append_unduplicated(fp_layer_list, all_tensors[i])
                if model_block_name == 'vit_block':
                    first_ln_index = type_tensors_str[:type_tensors_str.find('top.LayerNorm'
                                                                             )].count('top')
                    if op_type == 'top.GELU' and all_tensors[i] in fp_layer_list:
                        fp_layer_list.remove(all_tensors[i])
                    if i < first_ln_index and op_type != 'top.Input' and count >= 20:
                        if op_type == 'top.Add':
                            next_ops = self.parser.get_next_op_by_op_name(all_tensors[i])
                            if len(next_ops) != 1 or self.parser.get_op_type_by_op_name(
                                    next_ops[0]) != 'top.LayerNorm':
                                append_unduplicated(
                                    fp_layer_list,
                                    all_tensors[i])  # avoid to add top.Add repeatedly
                        else:
                            append_unduplicated(fp_layer_list, all_tensors[i])
                if model_block_name == 'eva_block':
                    if op_type == 'top.Add':
                        append_unduplicated(fp_layer_list, all_tensors[i])
                    if op_type == 'top.SiLU':
                        next_op = self.parser.get_next_op_by_op_name(all_tensors[i])
                        next_op_type = self.parser.get_op_type_by_op_name(next_op[0])
                        if len(next_op) == 1 and next_op_type == 'top.Mul':
                            next_next_op = self.parser.get_next_op_by_op_name(next_op[0])
                            next_next_op_type = self.parser.get_op_type_by_op_name(next_next_op[0])
                            if next_next_op_type == 'top.MatMul':
                                append_unduplicated(fp_layer_list, next_next_op[0])
                if model_block_name == '_eva_block' or model_block_name == 'eva02_block_v1':
                    if op_type == 'top.Add':
                        append_unduplicated(fp_layer_list, all_tensors[i])
                if model_block_name == 'eva02_block' or model_block_name == 'eva02_smc_block':
                    if op_type == 'top.SiLU' and all_tensors[i] in fp_layer_list:
                        fp_layer_list.remove(all_tensors[i])
                    if op_type == 'top.MatMul':
                        pre_ops = self.parser.get_pre_op_by_op_name(all_tensors[i])
                        pre_op_types = [
                            self.parser.get_op_type_by_op_name(pre_op) for pre_op in pre_ops
                        ]
                        if len(pre_ops) == 2 and any(pre_op_type == 'top.Softmax'
                                                     for pre_op_type in pre_op_types):
                            append_unduplicated(fp_layer_list, all_tensors[i])
                    if op_type == 'top.Mul':
                        pre_ops = self.parser.get_pre_op_by_op_name(all_tensors[i])
                        pre_op_types = [
                            self.parser.get_op_type_by_op_name(pre_op) for pre_op in pre_ops
                        ]
                        if len(pre_ops) == 1 and pre_op_types[0] == 'top.Softmax':
                            append_unduplicated(fp_layer_list, all_tensors[i])
                    if op_type == 'top.Add':
                        next_ops = self.parser.get_next_op_by_op_name(all_tensors[i])
                        next_op_types = [
                            self.parser.get_op_type_by_op_name(next_op) for next_op in next_ops
                        ]
                        if all(next_op_type != 'top.LayerNorm' for next_op_type in next_op_types):
                            append_unduplicated(fp_layer_list, all_tensors[i])
                if model_block_name == 'swin_block' or model_block_name == '_swin_block' or model_block_name == 'swin_1_block':
                    if op_type == 'top.LayerNorm':
                        pre_op = self.parser.get_pre_op_by_op_name(all_tensors[i])
                        if len(pre_op) == 1 and self.parser.get_op_type_by_op_name(
                                pre_op[0]) == 'top.Reshape':
                            _pre_op = self.parser.get_pre_op_by_op_name(pre_op[0])
                            if len(_pre_op) == 1 and self.parser.get_op_type_by_op_name(
                                    _pre_op[0]) == 'top.Permute':
                                append_unduplicated(fp_layer_list, _pre_op[0])
                    if op_type == 'top.Add':
                        append_unduplicated(fp_layer_list, all_tensors[i])
                if model_block_name == '_swin_1_block':
                    if op_type == 'top.Add' or op_type == 'top.Depth2Space':
                        append_unduplicated(fp_layer_list, all_tensors[i])
                if model_block_name == 'bert_block' or model_block_name == 'bert_block_1':
                    if op_type == 'top.GELU':
                        next_op = self.parser.get_next_op_by_op_name(all_tensors[i])
                        next_op_type = self.parser.get_op_type_by_op_name(next_op[0])
                        if next_op_type == 'top.MatMul':
                            append_unduplicated(fp_layer_list, next_op[0])
                if model_block_name.startswith(
                        'insert_mul_bert_block') or model_block_name == 'bert_block_large_pt':
                    if op_type == 'top.GELU' and all_tensors[i] in fp_layer_list:
                        fp_layer_list.remove(all_tensors[i])
                    elif op_type == 'top.Add':
                        next_ops = self.parser.get_next_op_by_op_name(all_tensors[i])
                        if len(next_ops) != 1 or self.parser.get_op_type_by_op_name(
                                next_ops[0]) != 'top.LayerNorm':
                            append_unduplicated(fp_layer_list, all_tensors[i])
                    elif op_type == 'top.Mul':
                        pre_ops = self.parser.get_pre_op_by_op_name(all_tensors[i])
                        if len(pre_ops) == 1 and self.parser.get_op_type_by_op_name(
                                pre_ops[0]) == 'top.LayerNorm':
                            append_unduplicated(fp_layer_list, all_tensors[i])
                if model_block_name == 'detr_block' or model_block_name == 'detr_rc_50_block':
                    if op_type in ['top.Conv', 'top.Scale', 'top.Reshape']:
                        pass
                    elif op_type == 'top.MatMul':
                        pre_op = self.parser.get_pre_op_by_op_name(all_tensors[i])
                        pre_op_type = self.parser.get_op_type_by_op_name(pre_op[0])
                        if len(pre_op) == 1 and (pre_op_type == 'top.LayerNorm'
                                                 or pre_op_type == 'top.Reshape'):
                            pass
                        else:
                            append_unduplicated(fp_layer_list, all_tensors[i])
                    else:
                        append_unduplicated(fp_layer_list, all_tensors[i])
                if model_block_name == "clip_m2_encoder_block":  # very basic config, need further optimization
                    if op_type == "top.Add":
                        append_unduplicated(fp_layer_list, all_tensors[i])
                if model_block_name == 'openclip_block':
                    if i >= last_matmul_index or (first_text_mlp_start_index <= i <=
                                                  first_text_mlp_end_index):
                        append_unduplicated(fp_layer_list, all_tensors[i])
                    elif op_type in [
                            'top.Abs', 'top.Reduce', 'top.Sqrt', 'top.Softmax', 'top.Gather',
                            'top.Slice', 'top.Squeeze', 'top.Arg', 'top.Concat'
                    ]:
                        append_unduplicated(fp_layer_list, all_tensors[i])
                    elif op_type == 'top.Div':
                        # Div op name will be changed by adding '_inv' suffix when deploying.
                        append_unduplicated(fp_layer_list, f'{all_tensors[i]}_inv')
                    elif op_type == 'top.Mul':
                        next_op = self.parser.get_next_op_by_op_name(all_tensors[i])
                        next_op_type = self.parser.get_op_type_by_op_name(next_op[0])
                        if len(next_op) == 1 and next_op_type == 'top.Reduce':
                            append_unduplicated(fp_layer_list, all_tensors[i])
                    elif op_type == 'top.Permute':
                        pre_op = self.parser.get_pre_op_by_op_name(all_tensors[i])
                        pre_op_type = self.parser.get_op_type_by_op_name(pre_op[0])
                        if len(pre_op) == 1 and pre_op_type == 'top.Div':
                            append_unduplicated(fp_layer_list, all_tensors[i])
                    elif op_type == 'top.Add':
                        next_op = self.parser.get_next_op_by_op_name(all_tensors[i])
                        next_op_type = self.parser.get_op_type_by_op_name(next_op[0])
                        pre_ops = self.parser.get_pre_op_by_op_name(all_tensors[i])
                        pre_op_types = [
                            self.parser.get_op_type_by_op_name(pre_op) for pre_op in pre_ops
                        ]
                        if (len(next_op) == 1 and next_op_type in ['top.Add', 'top.Slice', 'top.Gather']) or \
                           all(pre_op_type == 'top.Add' for pre_op_type in pre_op_types):
                            append_unduplicated(fp_layer_list, all_tensors[i])
        return fp_layer_list, flag, self._logs
