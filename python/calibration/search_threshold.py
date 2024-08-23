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
pymlir.set_mem_mode("force_value_mem")
import numpy as np
import copy
import logging
from utils.mlir_parser import *
from calibration.mix_precision import MixQuantModel
from calibration.mix_precision import MixPrecSearcher
from calibration.kld_calibrator import CalibrationTable, ActivationCalibrator, SimpleTuner
from pathlib import Path
from utils.net_dot_log import net_dot_log
from utils.log_setting import logger, setup_logger
from utils.mlir_parser import MlirParser
from utils.misc import parse_debug_cmd

def is_fuseop(op_name):
    return re.match(r'^fused\[".*?"\]$', op_name)

def split_fuseop(op_name):
    if is_fuseop(op_name):
        new_ops = re.findall(r'"([^"]+)"', op_name)
        return new_ops[0]
    else:
        return op_name
class SearchThreshold:
    def __init__(self, args, selector, tune_ds):
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
        self.debug_cmd = parse_debug_cmd(args.debug_cmd)

    def gen_multiple_thresholds_new(self, quantize_method_list):
        calibrator = ActivationCalibrator(self.args, self.selector, self.tune_ds)
        thresholds_map, thresholds_map_absmax, thresholds_map_scale, thresholds_map_zp, _, _, _, _, thresholds, thresholds4= calibrator.activation_collect_and_calc_th_parallel()
        calibrator._clean_resource()
        for i, method_name in enumerate(quantize_method_list):
            thresholds_map = thresholds[method_name]
            op_layers = self.parser.get_op_name_list()
            cali_table = self.args.calibration_table + "_" + method_name
            if self.args.tune_num > 0:
                cali_table += ".1"
            with open(cali_table, 'w') as f:
                f.write("# mlir version: {}\n".format(pymlir.__version__))
                f.write("# mlir: {}\n".format(self.args.mlir_file))
                f.write("# genetated time: {}\n".format(datetime.datetime.now()))
                f.write("# quantization threshold calculation method: {}\n".format(method_name))
                if 'fp8' in self.debug_cmd:
                    f.write("#tpu-mlir-fp8 caliration table\n")
                else:
                    f.write("# histogram number: {}\n".format(calibrator.histogram_bin_num))
                f.write("# sample number: {}\n###\n".format(calibrator.num_samples))
                f.write("# op_name    threshold    min    max\n")
                for i, op_name in enumerate(op_layers):
                    outputs = self.parser.get_outputs_by_op_name(op_name)
                    for out in outputs:
                        if out not in thresholds_map:
                            continue   # possible useless leaf output
                        if 'fp8' in self.debug_cmd:
                            threshold = thresholds_map[out]
                            min_value, max_value = -threshold*128.0/127.0, threshold
                            f.write("{} {:.7f} {:.7f} {:.7f}\n".format(out, threshold, min_value,
                                                               max_value))
                        else:
                            if out in thresholds_map:
                                threshold = thresholds_map[out]
                            else:
                                threshold = 1.0
                            if out in calibrator._activations_statistics:
                                min_value, max_value, *_ = calibrator._activations_statistics[op_name]
                            else:
                                min_value, max_value = -1,1
                            if threshold <= 1e-5 or np.isnan(threshold):
                                threshold = 1e-5
                                self.mix_prec.logger.print_info("WARNING: layer {} threshold is zero. Please check the input data correctness.".format(op_name))
                        f.write("{} {:.7f} {:.7f} {:.7f}\n".format(op_name, threshold, min_value, max_value))
                if 'int4' in self.debug_cmd:
                    thresholds_map4 = thresholds4[method_name]
                    f.write("\n")
                    f.write("#int4_th\n")
                    for i, op_name in enumerate(op_layers):
                        outputs = self.parser.get_outputs_by_op_name(op_name)
                        for out in outputs:
                            if out not in thresholds_map:
                                continue
                            threshold = thresholds_map4[out]
                            if threshold <= 1e-5 or np.isnan(threshold):
                                threshold = 1e-5
                                self.mix_prec.logger.print_info("WARNING: layer {} threshold is zero. Please check the input data correctness.".format(out))
                            min_value, max_value = -threshold*8.0/7.0, threshold
                            f.write("{} {:.7f} {:.7f} {:.7f}\n".format(out, threshold, min_value,
                                                                    max_value))
            if calibrator.args.tune_num <= 0 or 'int4' in self.debug_cmd or method_name != 'KL':
                continue

            if self.args.tune_num > 0:
                cali_table = cali_table.rsplit(".1", 1)[0]
            calibrator.args.calibration_table = cali_table
            tunner = SimpleTuner(calibrator.args, calibrator.tune_ds, calibrator.ppa_list, thresholds_map_absmax)
            thresholds_map = tunner.run()

            layer_name_list = []
            cali_table += "_tune"
            with open(cali_table, 'w') as f:
                f.write("# mlir version: {}\n".format(pymlir.__version__))
                f.write("# mlir: {}\n".format(self.args.mlir_file))
                f.write("# genetated time: {}\n".format(datetime.datetime.now()))
                f.write("# quantization threshold calculation method: {}\n".format(method_name))
                f.write("# histogram number: {}\n".format(calibrator.histogram_bin_num))
                f.write("# sample number: {}\n".format(calibrator.num_samples))
                f.write("# tune number: {}\n###\n".format(self.args.tune_num))
                f.write("# op_name    threshold    min    max\n")
                for i, op_name in enumerate(op_layers):
                    op_name = split_fuseop(op_name)
                    threshold = thresholds_map[op_name]
                    if threshold <= 1e-5:
                        threshold = 1e-5
                        self.mix_prec.logger.print_info("WARNING: layer {} threshold is zero. Please check the input data correctness.".format(op_name))
                    layer_name_list.append('{}_{}'.format(i, op_name))
                    min_value, max_value, *_ = calibrator._activations_statistics[op_name]
                    f.write("{} {:.7f} {:.7f} {:.7f}\n".format(op_name, threshold, min_value, max_value))
        return

    def run_search_calitable(self):

        float_model = MixQuantModel(self.fp32_mlir, self.chip)
        predictions_gt = []
        global_compare_layers, layers_rate, _ = self.mix_prec.extract_global_layers()
        quantize_method_list = ["KL","MAX", "Percentile9999","MSE"]
        self.mix_prec.logger.print_info("Quantization threshold calculation method: {}".format(quantize_method_list))
        self.gen_multiple_thresholds_new(quantize_method_list)

        result = {}
        _ = self.mix_prec.run_model(float_model, True, global_compare_layers, layers_rate, predictions_gt)
        for method in quantize_method_list:
            cali_table = self.cali_table_name + "_" + method + ".1"
            if method == 'KL':
                cali_table = self.cali_table_name + "_" + method + "_tune"
            new_cali_table_name = "new_cali_table.txt"
            with open(cali_table, 'r') as file:
                data = file.read()
            with open(new_cali_table_name, 'w') as file:
                file.write(data)
            int8_model = MixQuantModel(self.fp32_mlir, self.chip, new_cali_table_name)
            if self.benchmark_method == 'cos':
                outputs = self.mix_prec.run_model(int8_model, False, global_compare_layers, layers_rate, predictions_gt)
            elif self.benchmark_method == 'snr':
                outputs = 1 - self.mix_prec.run_model_loss_snr(int8_model, False, global_compare_layers, layers_rate, predictions_gt)
            result[method] = outputs

        optimal_method = max(result.items(), key=lambda item: item[1])
        optimal_method_name = optimal_method[0]
        optimal_value = optimal_method[1]
        self.mix_prec.logger.print_info("{} similarity scores: {}".format(self.benchmark_method,result))
        self.mix_prec.logger.print_info("optimal_method_name: {}, optimal_value: {}".format(optimal_method_name,optimal_value))

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
