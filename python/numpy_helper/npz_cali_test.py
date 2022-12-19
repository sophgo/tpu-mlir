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
import re
from .npz_compare import *


def ctest_args(args_list):
    parser = argparse.ArgumentParser(description='Compare two npz tensor files.')
    parser.add_argument("npz_file", help="Reference file with fp32 data")
    parser.add_argument('--calibration_table',
                        type=str,
                        required=True,
                        help="calibration table of npz file")
    args = parser.parse_args(args_list)
    return args


def quantize(d1, threshold):
    d1 = np.round(d1 * 127.0 / threshold)
    d1[d1 > 127.0] = 127.0
    d1[d1 < -128.0] = -128.0
    return d1


def npz_cali_test(args_list):
    args = ctest_args(args_list)
    npzfile = np.load(args.npz_file)
    thresholds = {}
    pattern_info = re.compile(r'#.*')
    pattern_cali = re.compile(r'\S+\s+[-0-9.e]+\s+[-0-9.e]+\s+[-0-9.e]+')
    pattern_weight = re.compile(r'#weight_scale')
    with open(args.calibration_table, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if pattern_weight.match(line):
                break
            if pattern_info.match(line):
                continue
            if pattern_cali.match(line):
                name, threshold, min, max = line.split(' ')
                thresholds[name] = eval(threshold)

    new_data = {}
    for name, threshold in thresholds.items():
        if name in npzfile:
            data = npzfile[name]
            if data.dtype != np.float32 or threshold == 0.0:
                continue
            new_data[name] = dequantize(quantize(data, threshold), threshold)
    np.savez("requant.npz", **new_data)
    args_compare = ["requant.npz", args.npz_file, "--tolerance", "0.95,0.85", "-vv"]
    npz_compare(args_compare)

if __name__ == '__main__':
    npz_cali_test(sys.argv)
