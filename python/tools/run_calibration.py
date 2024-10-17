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
from calibration.search_threshold import SearchThreshold
from calibration.search_qtable import SearchQtable
from calibration.mix_precision import MixPrecSearcher
from calibration.transformer_pattern import MatchPattern
from calibration.shape_ops import ShapeOps
from utils.log_setting import logger

def parse_method_list(input_str):
    return input_str.split(',')

if __name__ == '__main__':
    print("TPU-MLIR {}".format(pymlir.__version__))
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument('mlir_file', metavar='mlir_file', help='mlir file')
    parser.add_argument('--we', help='weight equalization', action="store_true")
    parser.add_argument('--bc', help='bias correction', action="store_true")
    parser.add_argument('--dataset', type=str, help='dataset for calibration')
    parser.add_argument('--data_list', type=str, help='Input list file contain all input')
    parser.add_argument('--input_num', type=int, default=0, help='num of images for calibration')
    parser.add_argument('--inference_num', type=int, default=30,
                        help='num of inputs for inference during optimal threshold searching')
    parser.add_argument('--bc_inference_num', type=int, default=30,
                        help='num of inputs for inference during bias correction')
    parser.add_argument('--tune_list', type=str, default='', help='Tune list file contain all input for tune')
    parser.add_argument('--tune_num', type=int, default=5, help='num of images for tune')
    parser.add_argument('--histogram_bin_num', type=int, default=2048,
                        help='Specify histogram bin numer for kld calculate')
    parser.add_argument('--expected_cos', type=float, default=0.99, help='expected net output cos')
    parser.add_argument('--min_layer_cos', type=float, default=0.99, help='minimum cos of layer')
    parser.add_argument('--max_float_layers', type=int, default=5, help='num of float layers in search qtable')
    parser.add_argument('--cali_method', type=str, default='use_kl',help='method of calibration')
    parser.add_argument('--chip', '--processor', default='bm1684x', type=str,
                        choices=['bm1684x', 'bm1684', 'cv183x', 'cv182x', 'cv181x', 'cv180x', 'cv186x', 'bm1688', 'bm1690', 'mars3'],
                        help='chip platform name')
    parser.add_argument('--fp_type', default='auto', type=str,choices=['auto', 'F16', 'F32', 'BF16'],
                        help='float type of mix precision')
    parser.add_argument('--post_process', type=str, default=None,help='post_process program path')
    parser.add_argument('--global_compare_layers', type=str, default='',
                        help='global compare layers, for example:\'layer1,layer2\' or \'layer1:0.3,layer2:0.7\'')
    parser.add_argument('--search', type=str, default='False', choices=['search_threshold', 'search_qtable','False'], help='choose quantization scheme')
    parser.add_argument('--transformer', type=str, default='False', help='model include attention structure')
    parser.add_argument('--quantize_method_list', type=parse_method_list, default='MSE', help='threshold method for search qtable')
    parser.add_argument('--benchmark_method', type=str, default='cos', choices=['cos', 'snr'], help='method for search optimal threshold')
    parser.add_argument('--kurtosis_analysis', help='kurtosis analysis', action="store_true")
    parser.add_argument('--part_quantize', help='only quantize conv and matmul', action="store_true")
    parser.add_argument('--part_asymmtric', help='some pattern use asymmetric quantize', action='store_true')
    parser.add_argument('-o', '--calibration_table', type=str, help='output threshold table')
    parser.add_argument('--quantize_table', help='output search qtable')
    parser.add_argument('--debug_cmd', type=str, default='', help='debug cmd')
    parser.add_argument('--debug_log', action='store_true', help='Enable DEBUG logging level')
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

    log_level = "DEBUG" if args.debug_log else "INFO"

    shape_ops=ShapeOps(args)
    shape_ops.run()

    # mix precision
    if args.search == 'search_qtable':
        args._logger = logger('Search_Qtable', log_level=log_level)
        searcherQ = SearchQtable(args, selector, tune_ds)
        searcherQ.run()
    else:
        # weight equalization
        if args.we:
            args._logger = logger('Weight_Equalization', log_level=log_level)
            searcher = MixPrecSearcher(args)
            searcher.weight_equalization()
        # calibration
        if args.search == 'search_threshold':
            args._logger = logger('Search_Threshold', log_level=log_level)
            searcherT = SearchThreshold(args, selector, tune_ds)
            searcherT.run_search_calitable()
        elif args.search == 'False':
            calibrator = ActivationCalibrator(args, selector, tune_ds)
            calibrator.run()
        # bias correction
        if args.bc:
            input_num = args.input_num
            args.input_num = args.bc_inference_num
            args._logger = logger('Bias_Correction', log_level=log_level)
            searcher = MixPrecSearcher(args)
            searcher.run_bias_correction()
            args.input_num = input_num
            if args.search == 'search_threshold':
                args._logger = logger('Search_Threshold', log_level=log_level)
                searcher = SearchThreshold(args, selector, tune_ds)
                searcher.run_search_calitable()
            elif args.search == 'False':
                calibrator = ActivationCalibrator(args, selector, tune_ds)
                calibrator.run()
        match_pattern = MatchPattern(args)
        match_pattern.run()
