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

from calibration.kld_calibrator import ActivationCalibrator2
from calibration.data_selector import DataSelector


if __name__ == '__main__':
    print("SOPHGO Toolchain {}".format(pymlir.module().version))
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument('mlir_file', metavar='mlir_file', help='mlir file')
    parser.add_argument('--dataset', type=str, help='dataset for calibration')
    parser.add_argument('--data_list', type=str, help='Input list file contain all input')
    parser.add_argument('--input_num', type=int, default=0, help='num of images for calibration')
    parser.add_argument('--tune_list', type=str, default='', help='Tune list file contain all input for tune')
    parser.add_argument('--tune_num', type=int, default=5, help='num of images for tune')
    parser.add_argument('--histogram_bin_num', type=int, default=2048,
                        help='Specify histogram bin numer for kld calculate')
    parser.add_argument('-o', '--calibration_table', type=str, help='output threshold table')
    parser.add_argument('--debug_cmd', type=str, default='', help='debug cmd')
    # yapf: enable
    args = parser.parse_args()
    dump_list = True if 'dump_list' in args.debug_cmd else False
    selector = DataSelector(args.dataset, args.input_num, args.data_list)
    tune_ds = None
    if args.tune_list:
        tune_ds = DataSelector(None, args.tune_num, args.tune_list)
        args.tune_num = len(tune_ds.data_list)
    if dump_list:
        selector.dump("./selected_image_list.txt")
        if tune_ds is not None:
            tune_ds.dump("./selected_tune_image_list.txt")
    # calibration
    calibrator = ActivationCalibrator2(args, selector, tune_ds)
    calibrator.run()
