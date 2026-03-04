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
from ctypes import *
from tqdm import tqdm
import datetime
from utils.preprocess import preprocess
from utils.mlir_parser import *
from utils.log_setting import setup_logger
from utils.misc import *
from math import *
from .utils import *

pymlir.set_mem_mode("force_value_mem")


class ShapeOps:

    def __init__(self, args):
        self.args = args
        self.fp32_mlir = args.mlir_file
        self.chip = args.chip
        self.module = pymlir.module()
        self.module.load(args.mlir_file)
        self.parser = MlirParser(args.mlir_file)

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
        shape_generator_second_output = ['top.TopK']  # maybe maxpooling
        shape_generator_first_output = ['top.Arg']
        shape_consumer_second_input = ['top.GatherElements'
                                       ]  # maybe 'top.Gather', 'top.Scatter', 'top.ScatterND',
        transparent_ops = [
            'top.Concat',
            'top.Slice',
            'top.Tile',
            'top.Split',
            'top.Pack',
            'top.Repeat',
            'top.Squeeze',
            'top.Unsqueeze',
        ]
        is_cast_to_int = lambda op_name: self.parser.get_op_type_by_op_name(
            op_name) == 'top.Cast' and self.parser.get_op_by_op_name(op_name).attrs[
                'to'] in ['"INT32"', '"INT16"', '"INT8"']
        is_cast_to_float = lambda op_name: self.parser.get_op_type_by_op_name(
            op_name) == 'top.Cast' and self.parser.get_op_by_op_name(op_name).attrs[
                'to'] in ['"F32"', '"F16"']
        is_where_index = lambda op_name, input_op: self.parser.get_op_type_by_op_name(
            op_name) == 'top.Where' and input_op == self.parser.get_op_by_op_name(op_name).opds[0]
        is_generator = lambda op_name: (self.parser.get_op_type_by_op_name(op_name) in
                                        shape_generator) or is_cast_to_int(op_name)
        is_second_output_generator = lambda op_name: (self.parser.get_op_type_by_op_name(
            op_name) in shape_generator_second_output) and \
            (self.parser.get_op_by_op_name(op_name).outputs[1] == op_name if len(self.parser.get_op_by_op_name(op_name).outputs) > 1 else False)
        is_first_output_generator = lambda op_name: (self.parser.get_op_type_by_op_name(
            op_name) in shape_generator_first_output) and \
            (self.parser.get_op_by_op_name(op_name).outputs[0] == op_name)
        is_consumer = lambda op_name: (self.parser.get_op_type_by_op_name(op_name) in shape_consumer
                                       ) or is_cast_to_float(op_name)
        is_transparent = lambda op_name: self.parser.get_op_type_by_op_name(op_name
                                                                            ) in transparent_ops

        inputs = self.parser.inputs  #list of operation
        outputs = self.parser.get_output_op_names_n_shapes()  #dict of name and shape
        is_input = lambda op_name: op_name in [op.name for op in inputs]
        is_output = lambda op_name: op_name in outputs

        shape_ops = [x.name for x in self.parser.ops if is_generator(x.name)]  #op names
        ext_shape_ops_first_output = [
            x for x in self.parser.ops if is_first_output_generator(x.name)
        ]
        multi_output_ops = [x for x in self.parser.ops if len(x.outputs) > 1]
        ext_shape_ops_second_output = [
            x for x in multi_output_ops if is_second_output_generator(x.name)
        ]
        has_input = lambda op_name, op: op_name in op.opds
        ext_shape_ops = []
        for op in ext_shape_ops_first_output:
            ext_shape_ops.extend([
                x for x in self.parser.get_next_op_by_op_name(op.name)
                if has_input(op.outputs[0], self.parser.get_op_by_op_name(x))
            ])
        for op in ext_shape_ops_second_output:
            ext_shape_ops.extend([
                x for x in self.parser.get_next_op_by_op_name(op.name)
                if has_input(op.outputs[1], self.parser.get_op_by_op_name(x))
            ])

        shape_ops = list(set(shape_ops + ext_shape_ops))

        shape_consumer_second_input_tensor = [
            x.opds[1] for x in self.parser.ops
            if x.type in shape_consumer_second_input and len(x.opds) > 1
        ]
        is_second_input_consumer = lambda op_name: op_name in shape_consumer_second_input_tensor
        while True:
            no_new_shape_op = True
            for op in shape_ops:  # in fact, tensor names
                if is_output(op):
                    continue
                next_ops = self.parser.get_next_op_by_op_name(op)
                for op_ in next_ops:
                    if op_ in shape_ops:
                        continue
                    elif is_output(op_):
                        shape_ops.append(op_)
                        no_new_shape_op = False
                    elif is_consumer(op_) or is_second_input_consumer(op_) or is_where_index(
                            op_, op):
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
        shape_ops = list(
            set(shape_ops + [x.name for x in ext_shape_ops_second_output] +
                [x.name for x in ext_shape_ops_first_output]))  # remove duplicates
        return shape_ops
