#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import numpy as np
import argparse
import pymlir
import onnx
import ast
import onnxruntime
import onnxsim.onnx_simplifier as onnxsim
from utils.preprocess import get_preprocess_parser, preprocess
from utils.mlir_parser import *
from utils.misc import *

class mlir_inference(object):
    def __init__(self, args):
        self.idx = 0
        self.module = pymlir.module()
        self.module.load(args.model_file)
        self.postprocess_type = args.postprocess_type
        self.module_parsered = MlirParser(args.model_file)
        self.batch_size = self.module_parsered.get_batch_size()
        args.batch_size = self.batch_size
        self.input_num = self.module_parsered.get_input_num()
        self.img_proc = preprocess(args.debug_cmd)
        self.img_proc.load_config(self.module_parsered.get_input_op_by_idx(0))
        args.net_input_dims = self.img_proc.net_input_dims
        self.batched_labels = []
        self.batched_imgs = ''
        exec('from eval.postprocess_and_score_calc.{name} import {name}'.format(name = args.postprocess_type))
        self.score = eval('{}(args)'.format(args.postprocess_type))
        self.debug_cmd = parse_debug_cmd(args.debug_cmd)
        print('batch_size:', self.batch_size)

    def run(self, idx, img_path, target = None):
        self.idx = idx
        self.batched_imgs += '{},'.format(img_path)
        if target is not None:
            self.batched_labels.append(target)
        if (idx+1) % self.batch_size == 0:
            self.batched_imgs = self.batched_imgs[:-1]
            self.model_invoke()

    def get_result(self):
        if self.batched_imgs != '':
            print('get_result do the remained imgs')
            tmp = self.batched_imgs[:-1].split(',')
            n = self.batch_size - len(tmp)
            if n > 0:
                for i in range(n):
                    tmp.append(tmp[-1])
                    if len(self.batched_labels) > 0:
                        self.batched_labels.append(self.batched_labels[-1])
            self.batched_imgs = ','.join(tmp)
            self.idx += n
            self.model_invoke()
        return self.score.get_result()

    def model_invoke(self):
        ratio_list = None
        if 'not_use_preprocess' in self.debug_cmd:
            img = self.score.preproc(self.batched_imgs)
            self.module.set_tensor(self.img_proc.input_name, img)
        else:
            x= self.img_proc.run(self.batched_imgs)
            ratio_list = self.img_proc.get_config('ratio')
            self.module.set_tensor(self.img_proc.input_name, x)
        self.module.invoke()
        all_tensors = self.module.get_all_tensor()
        outputs = []
        for i in self.module.output_names:
            outputs.append(all_tensors[i])
        if len(self.batched_labels) > 0:
            self.score.update(self.idx, outputs, labels = self.batched_labels, ratios = ratio_list)
        else:
            self.score.update(self.idx, outputs, img_paths = self.batched_imgs, ratios = ratio_list)
        self.batched_labels.clear()
        self.batched_imgs = ''
        if (self.idx + 1) % 5 == 0:
            self.score.print_info()

class onnx_inference(object):
    def __init__(self, args):
        self.img_proc = preprocess()
        self.img_proc.config(**vars(args))
        self.batched_labels = []
        args.batch_size = self.img_proc.batch_size
        args.net_input_dims = self.img_proc.net_input_dims
        self.batched_imgs = ''
        self.net = onnxruntime.InferenceSession(args.model_file,providers=['CPUExecutionProvider'])
        exec('from eval.postprocess_and_score_calc.{name} import {name}'.format(name = args.postprocess_type))
        self.score = eval('{}(args)'.format(args.postprocess_type))
        self.batch_size = args.batch_size
        self.debug_cmd = parse_debug_cmd(args.debug_cmd)
        print('onnx batch size:', self.batch_size)

    def run(self, idx, img_path, target = None):
        self.idx = idx
        self.batched_imgs += '{},'.format(img_path)
        if target is not None:
            self.batched_labels.append(target)
        if (idx+1) % self.batch_size == 0:
            self.batched_imgs = self.batched_imgs[:-1]
            self.model_invoke()

    def model_invoke(self):
        ratio_list = None
        if 'not_use_preprocess' in self.debug_cmd:
            img = self.score.preproc(self.batched_imgs)
        else:
            img= self.img_proc.run(self.batched_imgs)
            ratio_list = self.img_proc.get_config('ratio')
        input_name = self.net.get_inputs()[0].name
        outs = self.net.run(None, {input_name:img})
        output = outs[0:1] if type(outs) == list else [outs]
        if len(self.batched_labels) > 0:
            self.score.update(self.idx, output, self.batched_imgs, labels = self.batched_labels, ratios = ratio_list)
        else:
            self.score.update(self.idx, output, self.batched_imgs, ratios = ratio_list)
        self.batched_imgs = ''
        self.batched_labels.clear()
        if (self.idx + 1) % 5 == 0:
            self.score.print_info()

    def get_result(self):
        if self.batched_imgs != '':
            print('get_result do the remained imgs')
            tmp = self.batched_imgs[:-1].split(',')
            n = self.batch_size - len(tmp)
            if n > 0:
                for i in range(n):
                    tmp.append(tmp[-1])
                    if len(self.batched_labels) > 0:
                        self.batched_labels.append(self.batched_labels[-1])
            self.batched_imgs = ','.join(tmp)
            self.idx += n
            self.model_invoke()
        self.score.get_result()

#class tflite_inference()
class model_inference(object):
    def __init__(self, parser):
        args, _ = parser.parse_known_args()
        if args.postprocess_type == 'topx':
            from eval.postprocess_and_score_calc.topx import score_Parser
        elif args.postprocess_type == 'coco_mAP':
            from eval.postprocess_and_score_calc.coco_mAP import score_Parser
        else:
            print('postprocess_type error')
            exit(1)
        new_parser = argparse.ArgumentParser(
            parents=[parser, score_Parser().parser],
            conflict_handler='resolve')
        if args.model_file.endswith('.mlir'):
            args = new_parser.parse_args()
            engine = mlir_inference(args)
        elif args.model_file.endswith('.onnx'):
            parser = get_preprocess_parser(existed_parser=new_parser)
            parser.add_argument("--input_shapes", type=str, help="input_shapes")
            args = parser.parse_args()
            engine = onnx_inference(args)
        else:
            print('model_file:{}, ext_name error'.format(args.model_file))
            exit(1)
        self.engine = engine

    def run(self, idx, img_path, target = None):
        self.engine.run(idx, img_path, target)

    def get_result(self):
        self.engine.get_result()
