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
pymlir.set_mem_mode("force_value_mem")
from ctypes import *
from tqdm import tqdm
import datetime
from utils.preprocess import preprocess
from utils.mlir_parser import *
from utils.log_setting import setup_logger
from utils.misc import *
from math import *

from calibration.transformer_pattern import FLOAT_MAP
from calibration.transformer_pattern import chip_support_mix_fp_type


class ShapeOps:
    def __init__(self, args):
        self.args = args
        self.fp32_mlir = args.mlir_file
        self.chip = args.chip
        self.module = pymlir.module()
        self.module.load(args.mlir_file)
        self.parser = MlirParser(args.mlir_file)
        self.quantize_table = self.parser.module_name+"_shape_ops"
        if args.fp_type == 'auto':
            self.mix_mode = FLOAT_MAP[self.chip]
        else:
            self.mix_mode = args.fp_type
            if args.fp_type not in chip_support_mix_fp_type[self.chip]:
                print('parameter error, fp_type:{args.fp_type} not support by {self.chip}')
                exit(1)

    def gen_qtable(self, fp_layer_list):
        with open(self.quantize_table, "w") as f:
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# chip: {}  mix_mode: {}\n".format(self.chip, self.mix_mode))
            f.write("# number of {} layer: {}\n".format(self.mix_mode, len(fp_layer_list)))
            f.write("###\n")
            f.write("# op_name   quantize_mode\n")
            for layer in fp_layer_list:
                f.write("{} {}\n".format(layer, self.mix_mode))

    def find_forward_all_input_shapes(self, shape_ops):
        found_new_shape_ops = False
        for op in shape_ops:
            next_ops = self.parser.get_next_op_by_op_name(op)
            for op_ in next_ops:
                if op_ in shape_ops:
                    continue
                pre_ops = self.parser.get_pre_op_by_op_name(op_)
                all_shape_inputs = True
                for op__ in pre_ops:
                    if op__ in shape_ops:
                        continue
                    else:
                        all_shape_inputs = False
                        break
                if all_shape_inputs:
                    shape_ops.append(op_)
                    found_new_shape_ops = True
        return found_new_shape_ops, shape_ops

    def run(self):
        shape_layer_list = []

        shape_generator = ['top.Shape', 'top.Size']
        shape_consumer = ['top.Reshape', 'top.Interp']
        transparent_ops = ['top.Concat', 'top.Slice', 'top.Tile', 'top.Split', 'top.Pack', 'top.Repeat']
        is_generator = lambda op_name: self.parser.get_op_type_by_op_name(op_name) in shape_generator
        is_consumer = lambda op_name: self.parser.get_op_type_by_op_name(op_name) in shape_consumer
        is_transparent = lambda op_name: self.parser.get_op_type_by_op_name(op_name) in transparent_ops

        inputs = self.parser.inputs #list of operation
        outputs = self.parser.get_output_op_names_n_shapes()  #dict of name and shape
        is_input = lambda op_name: op_name in [op.name for op in inputs]
        is_output = lambda op_name: op_name in outputs

        shape_ops = [x.name for x in self.parser.ops if is_generator(x.name)] #op names

        while True:
            no_new_shape_op = True
            for op in shape_ops:
                if is_output(op):
                    continue
                next_ops = self.parser.get_next_op_by_op_name(op)
                for op_ in next_ops:
                    if op_ in shape_ops:
                        continue
                    elif is_output(op_):
                        shape_ops.append(op_)
                        no_new_shape_op = False
                    elif is_consumer(op_):
                        continue
                    elif is_transparent(op_):
                        shape_ops.append(op_)
                        no_new_shape_op = False
                    else:
                        continue
            for op in shape_ops:
                pre_ops = self.parser.get_pre_op_by_op_name(op)
                for op in pre_ops:
                    if op in shape_ops or is_generator(op):
                        continue
                    elif is_transparent(op) or is_input(op):
                        no_new_shape_op = False
                        shape_ops.append(op)
                    else:  #maybe some math op is calculating ratio and shape, the output may be shape but the ratio may be not from shape
                        continue
            new_op, shape_ops = self.find_forward_all_input_shapes(shape_ops)
            if new_op:
                no_new_shape_op = False
            if no_new_shape_op:
                break
        self.gen_qtable(shape_ops)
        return
