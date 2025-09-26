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
import time
import datetime
import pymlir
import numpy as np
import copy
import logging
from utils.mlir_parser import *
from calibration.mix_precision import MixQuantModel
from calibration.mix_precision import MixPrecSearcher
from calibration.kld_calibrator import CalibrationTable, ActivationCalibrator
from pathlib import Path
from utils.net_dot_log import net_dot_log
from utils.log_setting import logger, setup_logger
from utils.mlir_parser import MlirParser
from utils.misc import parse_debug_cmd
from .utils import *

pymlir.set_mem_mode("force_value_mem")


class SearchThreshold:

    def __init__(self, args, selector, tune_ds, fixed_fp_layers):
        self.args = args
        self.fp32_mlir = args.mlir_file
        self.chip = args.chip
        self.cali_table_name = args.calibration_table
        self.module = pymlir.module()
        self.module.load(args.mlir_file)
        self.parser = MlirParser(args.mlir_file)
        self.selector = selector
        self.tune_ds = tune_ds
        self.benchmark_method = args.benchmark_method
        args.loss_table = ""
        if args.input_num > args.inference_num:
            args.input_num = args.inference_num
        self.mix_prec = MixPrecSearcher(args)
        self.debug_cmd = args.debug_cmd
        self.fixed_fp_layers = shape_pattern_fp_layers or []

    def run_search_calitable(self):

        float_model = MixQuantModel(self.fp32_mlir, self.chip)
        predictions_gt = []
        global_compare_layers, layers_rate, _ = self.mix_prec.extract_global_layers()
        quantize_method_list = ["kl", "max", "percentile9999", "mse"]
        self.mix_prec.logger.print_info(
            "Quantization threshold calculation method: {}".format(quantize_method_list))
        all_op_names = self.parser.get_op_name_list()
        all_op_names = get_no_fused_tensors(self.parser, all_op_names)
        calibrator = ActivationCalibrator(self.args, self.selector, self.tune_ds)
        calibrator.calibration_method = quantize_method_list
        layer_th_dict = calibrator.gen_multiple_thresholds(all_op_names, quantize_method_list)
        del calibrator
        mix_table = self.mix_prec._gen_mix_table(self.fixed_fp_layers)

        result = {}
        _ = self.mix_prec.run_model(float_model, True, global_compare_layers, layers_rate,
                                    predictions_gt)
        for method in quantize_method_list:
            cali_table = self.cali_table_name + "_" + method + ".1"
            if method == 'KL' and self.args.tune_num > 0:
                cali_table = self.cali_table_name + "_" + method + "_tune"
            new_cali_table_name = "new_cali_table.txt"
            with open(cali_table, 'r') as file:
                data = file.read()
            with open(new_cali_table_name, 'w') as file:
                file.write(data)
            int8_model = MixQuantModel(self.fp32_mlir, self.chip, new_cali_table_name, mix_table)
            if self.benchmark_method == 'cos':
                outputs = self.mix_prec.run_model(int8_model, False, global_compare_layers,
                                                  layers_rate, predictions_gt)
            elif self.benchmark_method == 'snr':
                outputs = 1 - self.mix_prec.run_model_loss_snr(
                    int8_model, False, global_compare_layers, layers_rate, predictions_gt)
            result[method] = outputs

        optimal_method = max(result.items(), key=lambda item: item[1])
        optimal_method_name = optimal_method[0]
        optimal_value = optimal_method[1]
        self.mix_prec.logger.print_info("{} similarity scores: {}".format(
            self.benchmark_method, result))
        self.mix_prec.logger.print_info("optimal_method_name: {}, optimal_value: {}".format(
            optimal_method_name, optimal_value))

        if optimal_method_name == 'KL':
            cali_table = self.cali_table_name + "_" + optimal_method_name + "_tune"
        else:
            cali_table = self.cali_table_name + "_" + optimal_method_name + ".1"
        new_cali_table_name = self.cali_table_name
        with open(cali_table, 'r') as file:
            data = file.read()
        with open(new_cali_table_name, 'w') as file:
            file.write(data)
        self.mix_prec.logger.print_info("success")
        return
