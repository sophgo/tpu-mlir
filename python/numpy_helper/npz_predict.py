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
import sys
import argparse
from .npz_compare import *

def predict_args(args_list):
    parser = argparse.ArgumentParser(description='Compare two npz tensor files.')
    parser.add_argument("ref_file",
                        help="Reference file with fp32 data")
    parser.add_argument('--op_info', type=str, required=True,
                        help="A csv file op_info, including order and dequant threshold")
    args = parser.parse_args(args_list)
    return args

def quantize(d1, threshold):
  d1 = np.round(d1 * 127.0 / threshold)
  d1[d1 > 127.0] = 127.0
  d1[d1 < -128.0] = -128.0
  return d1

def npz_predict(args_list):
  args = predict_args(args_list)
  npzfile = np.load(args.ref_file)
  ordered_names, operations, quant_types, thresholds = load_op_info(args.op_info)
  predict = {}
  for name in ordered_names:
    if name in npzfile:
      data = npzfile[name]
      threshold = thresholds[name]
      if data.dtype != np.float32 or threshold == 0.0:
        continue
      predict[name] = dequantize(quantize(data, threshold), threshold)
  np.savez("predict.npz", **predict)
  args_compare = ["predict.npz", args.ref_file, "--tolerance", "0.95,0.95,0.85", "-v"]
  npz_compare(args_compare)
