#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import re
import argparse
import pymlir

from calibration.kld_calibrator import ActivationCalibrator
from calibration.data_selector import DataSelector


def buffer_size_type(arg):
    try:
        val = re.match('(\d+)G', arg).groups()[0]
    except:
        raise argparse.ArgumentTypeError("must be [0-9]G")
    return val


if __name__ == '__main__':
    print("SOPHGO Toolchain {}".format(pymlir.module().version))
    parser = argparse.ArgumentParser()
    parser.add_argument('mlir_file', metavar='mlir_file', help='mlir file')
    parser.add_argument('--dataset', type=str, help='dataset for calibration')
    parser.add_argument('--data_list', type=str, help='Input list file contain all input')
    parser.add_argument('--input_num',
                        type=int,
                        default=0,
                        help='num of images for calibration')
    parser.add_argument('--tune_num', type=int, default=5,
                        help='num of images for tune')
    parser.add_argument('--histogram_bin_num',
                        type=int,
                        default=2048,
                        help='Specify histogram bin numer for kld calculate')
    parser.add_argument('-o', '--calibration_table', type=str, help='output threshold table')
    parser.add_argument('--debug_cmd', type=str, default='', help='debug cmd')
    args = parser.parse_args()

    selector = DataSelector(args.dataset, args.input_num, args.data_list)
    # calibration
    calibrator = ActivationCalibrator(args, selector.data_list)
    calibrator.run()
