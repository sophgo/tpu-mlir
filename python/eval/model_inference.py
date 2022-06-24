#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
#
# Licensed under the Apache License v2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ==============================================================================

import numpy as np
import argparse
import pymlir
import pyruntime
import onnx
import onnxruntime
from data.preprocess import preprocess
from utils.mlir_parser import *


class mlir_inference(object):
    def __init__(self, args):
        self.module = pymlir.module()
        self.module.load(args.mlir_file)
        self.postprocess_type = args.postprocess_type
        self.module_parsered = MlirParser(args.mlir_file)
        self.batch_size = self.module_parsered.get_batch_size()
        args.batch_size = self.batch_size
        self.input_num = self.module_parsered.get_input_num()
        self.img_proc = preprocess()
        self.img_proc.load_config(self.module_parsered.get_input_op_by_idx(0))
        self.batched_labels = []
        self.batched_imgs = ''
        exec('from eval.postprocess_and_score_calc.{name} import {name}'.format(name = args.postprocess_type))
        self.score = eval('{}(args)'.format(args.postprocess_type))

    def run(self, idx, img_path, target = None) -> dict:
        self.batched_imgs += '{},'.format(img_path)
        if target is not None:
            self.batched_labels.append(target)
        if (idx+1) % self.batch_size == 0:
            x = self.img_proc.run(self.batched_imgs[:-1])
            self.module.set_tensor(self.img_proc.input_name, x)
            self.module.invoke()
            outputs = self.module.get_all_tensor()[self.module.output_names[0]]
            if len(self.batched_labels) > 0:
                self.score.update(idx, outputs, self.batched_labels)
            else:
                self.score.update(idx, outputs)
            self.batched_labels.clear()
            self.batched_imgs = ''

            if (idx + 1) % 50 == 0:
                self.score.print_info()

    def get_result(self):
        return self.score.get_result()

#class onnx_inference()
#class tflite_inference()
