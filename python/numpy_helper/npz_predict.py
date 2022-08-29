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
