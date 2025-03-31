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
from utils.log_setting import logger, setup_logger
from utils.misc import *
#import graphviz as gz
from math import *
from collections import Counter
from scipy import spatial
from .utils import *

pymlir.set_mem_mode("force_value_mem")

sub_blocks = {
    "eva_block":['top.Add', 'top.LayerNorm', 'top.MatMul', 'top.Reshape', 'top.Slice', 'top.Slice', 'top.Slice', 'top.Squeeze',
                 'top.Squeeze', 'top.Squeeze', 'top.Permute', 'top.Slice', 'top.Slice', 'top.Mul', 'top.Slice', 'top.MulConst',
                 'top.Slice', 'top.Unsqueeze', 'top.Unsqueeze', 'top.Concat', 'top.Reshape', 'top.Mul', 'top.Add', 'top.Concat',
                 'top.Permute', 'top.Slice', 'top.Slice', 'top.Mul', 'top.Slice', 'top.MulConst', 'top.Slice', 'top.Unsqueeze',
                 'top.Unsqueeze', 'top.Concat', 'top.Reshape', 'top.Mul', 'top.Add', 'top.Concat', 'top.MulConst', 'top.Permute',
                 'top.MatMul', 'top.Softmax', 'top.Permute', 'top.MatMul', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Add',
                 'top.LayerNorm', 'top.MatMul', 'top.MatMul', 'top.SiLU', 'top.Mul', 'top.MatMul'],
    "bert_block":['top.Add', 'top.LayerNorm', 'top.MatMul', 'top.MatMul', 'top.MatMul', 'top.Reshape', 'top.Permute', 'top.Reshape',
                  'top.Reshape', 'top.Permute', 'top.Permute', 'top.MatMul', 'top.MulConst', 'top.Add', 'top.Softmax', 'top.MatMul',
                  'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.GELU', 'top.MatMul'],
    "bert_block_1":['top.Add', 'top.LayerNorm', 'top.MatMul', 'top.MatMul', 'top.Reshape', 'top.MatMul', 'top.Reshape', 'top.Permute',
                    'top.Reshape', 'top.Permute', 'top.Permute', 'top.MatMul', 'top.MulConst', 'top.Add', 'top.Softmax', 'top.MatMul',
                    'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.GELU', 'top.MatMul'],
    "deit_block":['top.Add', 'top.LayerNorm', 'top.MatMul', 'top.Reshape', 'top.Slice', 'top.Squeeze', 'top.Slice', 'top.Squeeze',
                  'top.Slice', 'top.Squeeze', 'top.Permute', 'top.Permute', 'top.Permute', 'top.MatMul', 'top.MulConst', 'top.Softmax',
                  'top.Permute', 'top.MatMul', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.MatMul',
                  'top.GELU', 'top.MatMul'],
    "swin_block":['top.LayerNorm', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape', 'top.Slice', 'top.Slice',
                  'top.Slice', 'top.Squeeze', 'top.Squeeze', 'top.Squeeze', 'top.Permute', 'top.MulConst', 'top.Permute', 'top.Permute',
                  'top.MatMul', 'top.Add', 'top.Softmax', 'top.Permute', 'top.MatMul', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape',
                  'top.Permute', 'top.Reshape', 'top.Add', 'top.Reshape', 'top.LayerNorm', 'top.MatMul', 'top.GELU', 'top.MatMul', 'top.Add',
                  'top.Reshape', 'top.LayerNorm', 'top.Slice', 'top.Slice', 'top.Concat', 'top.Slice', 'top.Slice', 'top.Concat', 'top.Reshape',
                  'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape', 'top.Slice', 'top.Slice', 'top.Slice', 'top.Squeeze', 'top.Squeeze',
                  'top.Squeeze', 'top.Permute', 'top.MulConst', 'top.Permute', 'top.Permute', 'top.MatMul', 'top.Add', 'top.Softmax', 'top.Permute',
                  'top.MatMul', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Slice', 'top.Slice',
                  'top.Concat', 'top.Slice', 'top.Slice', 'top.Concat', 'top.Add', 'top.Reshape', 'top.LayerNorm', 'top.MatMul', 'top.GELU',
                  'top.MatMul', 'top.Add'],
    "_swin_block":['top.LayerNorm', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape', 'top.Slice', 'top.Slice',
                  'top.Slice', 'top.Squeeze', 'top.Squeeze', 'top.Squeeze', 'top.Permute', 'top.MulConst', 'top.Permute', 'top.Permute',
                  'top.MatMul', 'top.Add', 'top.Softmax', 'top.Permute', 'top.MatMul', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape',
                  'top.Permute', 'top.Reshape', 'top.Reshape', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.GELU', 'top.MatMul', 'top.Add',
                  'top.Reshape', 'top.LayerNorm', 'top.Slice', 'top.Slice', 'top.Concat', 'top.Slice', 'top.Slice', 'top.Concat', 'top.Reshape',
                  'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape', 'top.Slice', 'top.Slice', 'top.Slice', 'top.Squeeze', 'top.Squeeze',
                  'top.Squeeze', 'top.Permute', 'top.MulConst', 'top.Permute', 'top.Permute', 'top.MatMul', 'top.Add', 'top.Softmax', 'top.Permute',
                  'top.MatMul', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Slice', 'top.Slice',
                  'top.Concat', 'top.Slice', 'top.Slice', 'top.Concat', 'top.Reshape', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.GELU',
                  'top.MatMul', 'top.Add'],
    "vit_block": ['top.Add', 'top.LayerNorm', 'top.MatMul', 'top.Reshape', 'top.Slice', 'top.Slice', 'top.Slice', 'top.Squeeze', 'top.Squeeze',
                  'top.Squeeze', 'top.Permute', 'top.MulConst', 'top.Permute', 'top.Permute', 'top.MatMul', 'top.Softmax', 'top.Permute', 'top.MatMul',
                  'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.GELU', 'top.MatMul'],
    "detr_block": ['top.Add', 'top.LayerNorm', 'top.Add', 'top.MatMul', 'top.MatMul', 'top.MatMul', 'top.Reshape', 'top.Permute', 'top.Reshape',
                   'top.Reshape', 'top.Permute', 'top.MulConst', 'top.Permute', 'top.MatMul', 'top.Softmax', 'top.MatMul', 'top.Permute',
                   'top.Reshape', 'top.MatMul', 'top.Reshape', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.MatMul'],
    "detr_rc_50_block": ['top.Add', 'top.LayerNorm', 'top.Add', 'top.MatMul', 'top.MatMul', 'top.MatMul', 'top.Reshape', 'top.Permute', 'top.MulConst',
                        'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Permute', 'top.MatMul', 'top.Softmax', 'top.MatMul', 'top.Permute', 'top.Reshape',
                        'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.MatMul'],
    "swin_1_block": ['top.LayerNorm', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape', 'top.Permute', 'top.Permute',
                     'top.Slice', 'top.Slice', 'top.Reshape', 'top.Slice', 'top.Reshape', 'top.MulConst', 'top.Reshape', 'top.Permute',
                     'top.MatMul', 'top.Add', 'top.Softmax', 'top.MatMul', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape', 'top.Permute',
                     'top.Reshape', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.GELU', 'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.Reshape',
                     'top.Slice', 'top.Slice', 'top.Concat', 'top.Slice', 'top.Slice', 'top.Concat', 'top.Reshape', 'top.Permute', 'top.Reshape',
                     'top.MatMul', 'top.Reshape', 'top.Permute', 'top.Permute', 'top.Slice', 'top.Slice', 'top.Reshape', 'top.Slice', 'top.Reshape',
                     'top.MulConst', 'top.Reshape', 'top.Permute', 'top.MatMul', 'top.Add', 'top.Softmax', 'top.MatMul', 'top.Permute', 'top.Reshape',
                     'top.MatMul', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Slice', 'top.Slice', 'top.Concat', 'top.Slice', 'top.Slice', 'top.Concat',
                     'top.Reshape', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.GELU', 'top.MatMul', 'top.Add'],
    "_swin_1_block":['top.LayerNorm', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape', 'top.Slice', 'top.Slice',
                     'top.Slice', 'top.Squeeze', 'top.Squeeze', 'top.Squeeze', 'top.Permute', 'top.MulConst', 'top.Permute', 'top.Permute',
                     'top.MatMul', 'top.Add', 'top.Softmax', 'top.Permute', 'top.MatMul', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape',
                     'top.Permute', 'top.Reshape', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.GELU', 'top.MatMul', 'top.Add',
                     'top.LayerNorm', 'top.Reshape', 'top.Slice', 'top.Slice', 'top.Concat', 'top.Slice', 'top.Slice', 'top.Concat', 'top.Reshape',
                     'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape', 'top.Slice', 'top.Slice', 'top.Slice', 'top.Squeeze', 'top.Squeeze',
                     'top.Squeeze', 'top.Permute', 'top.MulConst', 'top.Permute', 'top.Permute', 'top.MatMul', 'top.Add', 'top.Softmax', 'top.Permute',
                     'top.MatMul', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Slice', 'top.Slice',
                     'top.Concat', 'top.Slice', 'top.Slice', 'top.Concat', 'top.Reshape', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.GELU',
                     'top.MatMul', 'top.Add'],
    "_eva_block":['top.Add', 'top.LayerNorm', 'top.MatMul', 'top.Reshape', 'top.Permute', 'top.MatMul', 'top.Reshape', 'top.Permute', 'top.MatMul',
                  'top.Reshape', 'top.Permute', 'top.Slice', 'top.Slice', 'top.Mul', 'top.Slice', 'top.MulConst', 'top.Slice', 'top.Unsqueeze',
                  'top.Unsqueeze', 'top.Concat', 'top.Reshape', 'top.Mul', 'top.Add', 'top.Concat', 'top.Slice', 'top.Slice', 'top.Mul', 'top.Slice',
                  'top.MulConst', 'top.Slice', 'top.Unsqueeze', 'top.Unsqueeze', 'top.Concat', 'top.Reshape', 'top.Mul', 'top.Add', 'top.Concat',
                  'top.MulConst', 'top.Permute', 'top.MatMul', 'top.Softmax', 'top.MatMul', 'top.Permute', 'top.Reshape', 'top.MatMul', 'top.Add',
                  'top.LayerNorm', 'top.MatMul', 'top.MatMul', 'top.SiLU', 'top.Mul', 'top.LayerNorm', 'top.MatMul'],
    "cswin_block":['top.LayerNorm', 'top.MatMul', 'top.Reshape', 'top.Permute', 'top.Slice', 'top.Reshape', 'top.Slice', 'top.Reshape', 'top.Slice',
                   'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Permute', 'top.Permute', 'top.Reshape', 'top.Permute',
                   'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Conv', 'top.Reshape', 'top.Permute', 'top.Reshape',
                   'top.Permute', 'top.MulConst', 'top.Permute', 'top.MatMul', 'top.Softmax', 'top.MatMul', 'top.Add', 'top.Permute', 'top.Reshape',
                   'top.Permute', 'top.Reshape', 'top.Slice', 'top.Reshape', 'top.Slice', 'top.Reshape', 'top.Slice', 'top.Reshape', 'top.Reshape',
                   'top.Permute', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Conv', 'top.Reshape', 'top.Permute',
                   'top.Reshape', 'top.Permute', 'top.MulConst', 'top.Permute', 'top.MatMul', 'top.Softmax', 'top.MatMul', 'top.Add', 'top.Permute',
                   'top.Reshape', 'top.Permute', 'top.Reshape', 'top.Concat', 'top.MatMul', 'top.Add', 'top.LayerNorm', 'top.MatMul', 'top.GELU',
                   'top.MatMul', 'top.Add'],
    "yolo_block":['top.MaxPool', 'top.MaxPool', 'top.MaxPool', 'top.Concat'],
    "yolo_block_12":['top.Sub', 'top.Add', 'top.Add', 'top.Sub', 'top.MulConst', 'top.Concat', 'top.Mul', 'top.Concat']

}
yolo_post_blocks = {
    "yolov5":['top.MulConst', 'top.Add', 'top.MulConst', 'top.MulConst', 'top.Mul', 'top.Mul', 'top.Concat', 'top.Reshape'],
    "yolov7":['top.MulConst', 'top.Add', 'top.Mul', 'top.Mul', 'top.Concat', 'top.Reshape'],
    "yolov6_8_9_11_12":['top.Sub', 'top.Add', 'top.Add', 'top.Sub', 'top.MulConst', 'top.Concat', 'top.Mul', 'top.Concat'],
    "yolov10":['top.Sub', 'top.Add', 'top.Concat', 'top.Mul', 'top.Concat', 'top.Permute', 'top.Slice', 'top.Slice',
               'top.Reduce', 'top.TopK', 'top.Unsqueeze', 'top.Tile', 'top.Tile', 'top.GatherElements', 'top.GatherElements',
               'top.Reshape', 'top.Reshape', 'top.TopK', 'top.MulConst', 'top.Floor', 'top.Unsqueeze', 'top.Mod', 'top.Unsqueeze',
               'top.Gather', 'top.Concat'],
    "yolov10":['top.Sub', 'top.Add', 'top.Concat', 'top.Mul', 'top.Concat', 'top.Permute', 'top.Slice', 'top.Slice',
               'top.Reduce', 'top.TopK', 'top.Unsqueeze', 'top.Tile', 'top.Tile', 'top.GatherElements', 'top.GatherElements',
               'top.Reshape', 'top.Reshape', 'top.TopK', 'top.MulConst', 'top.Floor', 'top.Unsqueeze', 'top.Mod', 'top.Unsqueeze',
               'top.Gather', 'top.Concat'],
}
# "detr_pattern": ['top.Conv', 'top.Scale', 'top.Conv', 'top.Scale', 'top.Conv', 'top.Scale', 'top.Add']

N_mode = ['top.Relu', 'top.MaxPool', 'top.Conv', 'top.MatMul', 'top.PRelu', 'top.AvgPool', 'top.Add', 'top.Sigmoid', 'top.Deconv']
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
        self.logger = logger('Match_Pattern', log_level=self.log_level)

        if self.args.part_quantize == 'N_mode':
            self.quantize_ops = N_mode
        elif self.args.part_quantize == 'H_mode':
            self.quantize_ops = H_mode
        elif self.args.part_quantize == 'custom_mode':
            self.quantize_ops = ['top.' + op for op in self.args.custom_operator]
        else:
            pass

        if '/' in self.cali_table_name:
            last_index = self.cali_table_name.rfind('/')
            self.quantize_table = self.cali_table_name[:last_index + 1] + "qtable"
        else:
            self.quantize_table = "qtable"
        if args.fp_type == 'auto':
            self.mix_mode = FLOAT_MAP[self.chip]
        else:
            self.mix_mode = args.fp_type
            if args.fp_type not in chip_support_mix_fp_type[self.chip]:
                self.logger.print_info(f'parameter error, fp_type:{args.fp_type} not support by {self.chip}')
                exit(1)

    def gen_qtable(self, fp_layer_list, flag):
        self.logger.print_info(f'The qtable has been generated in: {self.quantize_table} !!!')
        with open(self.quantize_table, "w") as f:
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# chip: {}  mix_mode: {}\n".format(self.chip, self.mix_mode))
            f.write("# number of {} layer: {}\n".format(self.mix_mode, len(fp_layer_list)))
            if self.args.part_quantize and flag == 0:
                f.write("# part_quantize \n")
            f.write("###\n")
            f.write("# op_name   quantize_mode\n")
            for layer in fp_layer_list:
                f.write("{} {}\n".format(layer, self.mix_mode))

    def run(self):
        flag = 0
        count = 0
        type_tensors = []
        fp_layer_list = []
        fp_layer_list_part_quantize = []
        all_tensors = self.parser.get_op_name_list()
        num_tensors = len(all_tensors)
        model_block_name = None

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
                self.logger.print_info(f"{sub_block} (Name: {name}) is a subset of the main list. Count: {count}")
                break
        if flag == 1:
            if model_block_name == 'yolo_block' or model_block_name == 'yolo_block_12':
                from collections import Counter
                def find_sublist(main_list, sub_list):
                    sub_len = len(sub_list)
                    if sub_len == 0:
                        return []
                    sub_counter = Counter(sub_list)
                    matches = []
                    for i in range(len(main_list) - sub_len + 1):
                        window = main_list[i:i+sub_len]
                        window_counter = Counter(window)
                        if window_counter == sub_counter:
                            end_idx = i + sub_len
                            matches.append((i, end_idx))
                    return matches

                for name, yolo_post_block in yolo_post_blocks.items():
                    matches = find_sublist(type_tensors, yolo_post_block)
                    if matches:
                        match_count = len(matches)
                        self.logger.print_info(f"The [{name}] post-processing pattern matches this model. Block count: {match_count}")
                        self.logger.print_info(f"The [{name}] post-processing pattern is: {yolo_post_block}")
                        for idx, (start, end) in enumerate(matches, 1):
                            fp_layer_list.extend(all_tensors[start:end])
                            for next_op in self.parser.get_next_op_by_op_name(all_tensors[end-1]):
                                if next_op:
                                    next_op_type = self.parser.get_op_type_by_op_name(next_op)
                                    # v5 && v7
                                    if next_op_type == 'top.Concat' and next_op not in fp_layer_list:
                                        fp_layer_list.append(next_op)
                        break
                split_fuse_fp_layer_list = []
                for item in fp_layer_list:
                    if item.startswith('fused['):
                        match = re.search(r'fused\["(.*?)"\]', item)
                        if match:
                            fused_paths = [p.strip('"') for p in match.group(1).split(', ')]
                            split_fuse_fp_layer_list.extend(fused_paths)
                        else:
                            split_fuse_fp_layer_list.append(item)
                    else:
                        split_fuse_fp_layer_list.append(item)
                self.gen_qtable(split_fuse_fp_layer_list, flag)
                return

            for i in range(num_tensors):
                op_type = self.parser.get_op_type_by_op_name(all_tensors[i])
                if op_type == 'top.LayerNorm':
                    pre_op = self.parser.get_pre_op_by_op_name(all_tensors[i])
                    if len(pre_op) == 1 and self.parser.get_op_type_by_op_name(pre_op[0]) == 'top.Add':
                        fp_layer_list.append(pre_op[0])
                if op_type == 'top.SiLU' or op_type == 'top.GELU':
                    fp_layer_list.append(all_tensors[i])
                if model_block_name == 'vit_block':
                    if count < 20 :
                        if op_type == 'top.Softmax':
                            next_op = self.parser.get_next_op_by_op_name(all_tensors[i])
                            next_op_type = self.parser.get_op_type_by_op_name(next_op[0])
                            if next_op_type == 'top.MatMul':
                                fp_layer_list.append(next_op[0])
                    else:
                        if op_type == 'top.GELU':
                            next_op = self.parser.get_next_op_by_op_name(all_tensors[i])
                            next_op_type = self.parser.get_op_type_by_op_name(next_op[0])
                            if next_op_type == 'top.MatMul':
                                fp_layer_list.append(next_op[0])
                if model_block_name == 'eva_block':
                    if op_type == 'top.Add':
                        fp_layer_list.append(all_tensors[i])
                    if op_type == 'top.SiLU':
                        next_op = self.parser.get_next_op_by_op_name(all_tensors[i])
                        next_op_type = self.parser.get_op_type_by_op_name(next_op[0])
                        if len(next_op) == 1 and next_op_type == 'top.Mul':
                            next_next_op = self.parser.get_next_op_by_op_name(next_op[0])
                            next_next_op_type = self.parser.get_op_type_by_op_name(next_next_op[0])
                            if next_next_op_type == 'top.MatMul':
                                fp_layer_list.append(next_next_op[0])
                if model_block_name == '_eva_block':
                    if op_type == 'top.Add':
                        fp_layer_list.append(all_tensors[i])
                if model_block_name == 'swin_block' or model_block_name == '_swin_block' or model_block_name == 'swin_1_block':
                    if op_type == 'top.LayerNorm':
                        pre_op = self.parser.get_pre_op_by_op_name(all_tensors[i])
                        if len(pre_op) == 1 and self.parser.get_op_type_by_op_name(pre_op[0]) == 'top.Reshape':
                            _pre_op = self.parser.get_pre_op_by_op_name(pre_op[0])
                            if len(_pre_op) == 1 and self.parser.get_op_type_by_op_name(_pre_op[0]) == 'top.Permute':
                                fp_layer_list.append(_pre_op[0])
                    if op_type == 'top.Add':
                        fp_layer_list.append(all_tensors[i])
                if model_block_name == '_swin_1_block':
                    if op_type == 'top.Add' or op_type == 'top.Depth2Space':
                        fp_layer_list.append(all_tensors[i])
                if model_block_name == 'bert_block' or model_block_name == 'bert_block_1':
                    if op_type == 'top.GELU':
                        next_op = self.parser.get_next_op_by_op_name(all_tensors[i])
                        next_op_type = self.parser.get_op_type_by_op_name(next_op[0])
                        if next_op_type == 'top.MatMul':
                            fp_layer_list.append(next_op[0])
                if model_block_name == 'detr_block' or  model_block_name == 'detr_rc_50_block':
                    if op_type in ['top.Conv','top.Scale','top.Reshape']:
                        pass
                    elif op_type == 'top.MatMul':
                        pre_op = self.parser.get_pre_op_by_op_name(all_tensors[i])
                        pre_op_type = self.parser.get_op_type_by_op_name(pre_op[0])
                        if len(pre_op) == 1 and (pre_op_type == 'top.LayerNorm' or pre_op_type == 'top.Reshape'):
                            pass
                        else:
                            fp_layer_list.append(all_tensors[i])
                    else:
                        fp_layer_list.append(all_tensors[i])
            self.gen_qtable(fp_layer_list, flag)
        if flag == 0 and self.args.part_quantize:
            for j in range(num_tensors):
                op_type = self.parser.get_op_type_by_op_name(all_tensors[j])
                if op_type in self.quantize_ops:
                    pass
                else:
                    fp_layer_list_part_quantize.append(all_tensors[j])
            self.gen_qtable(fp_layer_list_part_quantize, flag)
        return
