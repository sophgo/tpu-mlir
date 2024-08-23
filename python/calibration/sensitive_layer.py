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
import tools.run_calibration
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

class SensitiveLayer:
    def __init__(self, args, selector, tune_ds):
        self.args = args
        self.fp32_mlir = args.mlir_file
        self.chip = args.chip
        self.cali_table_name = args.calibration_table
        self.cali_table = CalibrationTable(self.cali_table_name)
        self.module = pymlir.module()
        self.module.load(args.mlir_file)
        self.parser = MlirParser(args.mlir_file)
        self.selector = selector
        self.tune_ds = tune_ds
        args.input_num = args.inference_num
        args.loss_table = ""
        self.mix_prec = MixPrecSearcher(args)
        self.mix_prec.dot_log = net_dot_log('sensitive_layer_search_result', self.parser, self.mix_prec.logger)


    def gen_multiple_thresholds(self, all_op_names, quantize_method_list):
        layer_th_dicts = {} # method_name: {op_name: [op_fmax, op_th]}
        for i, method_name in enumerate(quantize_method_list):
            tmp_th_dict = {}
            calibrator = ActivationCalibrator(self.args, self.selector, self.tune_ds)
            if method_name == "MAX":
                calibrator.debug_cmd = 'use_max'
            elif method_name == "Percentile9999":
                calibrator.debug_cmd = 'use_percentile9999'
            thresholds_map, thresholds_map_absmax, thresholds_map_scale, thresholds_map_zp, _, _, _, _ = calibrator.activation_collect_and_calc_th()
            calibrator._clean_resource()
            # step 3: dump threshold table of default histogram bins
            thresholds_map_list = []
            op_layers = self.parser.get_op_name_list()
            cali_table = self.args.calibration_table + "_" + method_name
            if self.args.tune_num > 0:
                cali_table += ".1"
            with open(cali_table, 'w') as f:
                f.write("# genetated time: {}\n".format(datetime.datetime.now()))
                f.write("# histogram number: {}\n".format(calibrator.histogram_bin_num))
                f.write("# sample number: {}\n###\n".format(calibrator.num_samples))
                f.write("# op_name    threshold    min    max\n")
                for i, op_name in enumerate(op_layers):
                    if 'use_torch_observer_for_cali' in calibrator.debug_cmd:
                        qmin, qmax = -128, 127
                        scale = thresholds_map_scale[op_name]
                        zp = thresholds_map_zp[op_name]
                        threshold = float(scale * max(-(qmin-zp), qmax-zp))
                        min_value = float(scale * (qmin - zp))
                        max_value = float(scale * (qmax - zp))
                    else:
                        threshold = thresholds_map[op_name]
                        min_value, max_value, _ = calibrator.activations_statistics[op_name]
                    thresholds_map_list.append(threshold)
                    f.write("{} {:.7f} {:.7f} {:.7f}\n".format(op_name, threshold, min_value, max_value))
            if calibrator.args.tune_num <= 0:
                return

            if self.args.tune_num > 0:
                cali_table = cali_table.rsplit(".1", 1)[0]
            calibrator.args.calibration_table = cali_table
            tunner = SimpleTuner(calibrator.args, calibrator.tune_ds, calibrator.ppa_list, thresholds_map_absmax)
            thresholds_map = tunner.run()

            tuned_threshold_list = []
            layer_name_list = []
            cali_table += "_tune"
            with open(cali_table, 'w') as f:
                f.write("# genetated time: {}\n".format(datetime.datetime.now()))
                f.write("# histogram number: {}\n".format(calibrator.histogram_bin_num))
                f.write("# sample number: {}\n".format(calibrator.num_samples))
                f.write("# tune number: {}\n###\n".format(self.args.tune_num))
                f.write("# op_name    threshold    min    max\n")
                for i, op_name in enumerate(op_layers):
                    threshold = thresholds_map[op_name]
                    layer_name_list.append('{}_{}'.format(i, op_name))
                    tuned_threshold_list.append(threshold)
                    min_value, max_value, _ = calibrator.activations_statistics[op_name]
                    f.write("{} {:.7f} {:.7f} {:.7f}\n".format(op_name, threshold, min_value, max_value))

            for op_name in all_op_names:
                tmp_th_dict[op_name] = [thresholds_map_absmax[op_name], thresholds_map[op_name]]
            layer_th_dicts[method_name] = tmp_th_dict
        return layer_th_dicts


    def check_layer_names(self, all_op_names, int8_model, layer_th_dicts, quantize_method_list):
        layer_names = []
        layer_name2layer_type_dict = {}
        ignored_layers = ["Coeff", "Accuracy"]
        for layer_name in all_op_names:
            ignore = False
            layer_proto = int8_model.parser.get_op_by_op_name(layer_name)
            if layer_proto == None or layer_proto.type in ignored_layers:
                ignore = True
            if not ignore:
                layer_names.append(layer_name)
                layer_name2layer_type_dict[layer_name] = layer_proto.type
        for layer_name_check in layer_names:
            if layer_name_check not in layer_th_dicts[quantize_method_list[0]].keys():
                self.mix_prec.logger.print_dbg(
                    "layer name of prototxt {} not match layer name of log/layer_name.txt, please cheak whether there is '! or /' in layer name ".format(
                        layer_name_check))
                exit(1)
        self.mix_prec.logger.print_info("layer name check pass !")
        return layer_names

    def set_layer_new_th(self, model, layer_name, value):
        op = model.parser.get_op_by_op_name(layer_name)
        threshold = float(value)
        self.cali_table.thresholds_map[layer_name][0] = threshold
        new_cali_table_name = "new_cali_table.txt"
        self.cali_table.update_to(new_cali_table_name, layer_name, threshold)
        return new_cali_table_name


    def search_sensitve_layer(self, layer_names, quantize_method_list, float_model, int8_model, layer_th_dicts, global_compare_layers, layers_rate, predictions_gt):
        num_quantize_method = len(quantize_method_list)
        fp_layer_list = []
        for op_name in layer_names:
            fp_layer_list.append(op_name)
        modified_layers = {}
        last_tried_method = quantize_method_list[0]
        sensitive_layer_analysis_dict = {}
        for layer_name in layer_names:
            layer_type = self.parser.get_op_type_by_op_name(layer_name)
            self.mix_prec.logger.print_info("start to handle layer: {}, type: {}".format(layer_name, layer_type))
            fp_layer_list.remove(layer_name)
            mix_table = self.mix_prec._gen_mix_table(fp_layer_list)
            ret = False
            while not ret:
                if layer_name not in modified_layers:
                    last_tried_method = quantize_method_list[0]
                    modified_layers[layer_name] = [1, float('inf'), layer_th_dicts[last_tried_method][layer_name][1], last_tried_method]
                    method = last_tried_method
                    new_th = layer_th_dicts[last_tried_method][layer_name][1]  # layer_th_dicts{quantize_method: {layer_name:{fmax, th}}}
                    new_cali_table_name = self.set_layer_new_th(int8_model, layer_name, new_th)
                    last_tried_method = method
                    self.mix_prec.logger.print_info("adjust layer {} th, with method {}, and threshlod {}".format(layer_name, method, new_th))
                    mixmodel = MixQuantModel(self.fp32_mlir, self.chip, new_cali_table_name, mix_table)
                    outputs_cos = 1 - self.mix_prec.run_model(mixmodel, False, global_compare_layers, layers_rate, predictions_gt)
                    self.mix_prec.logger.print_info("outputs_cos_los = {}".format(outputs_cos))
                elif modified_layers[layer_name][0] < num_quantize_method:
                    method_idx = modified_layers[layer_name][0]
                    method = quantize_method_list[method_idx]
                    if outputs_cos < modified_layers[layer_name][1]:
                        modified_layers[layer_name][1] = outputs_cos
                        modified_layers[layer_name][2] = layer_th_dicts[last_tried_method][layer_name][1]
                        modified_layers[layer_name][3] = last_tried_method
                    new_th = layer_th_dicts[method][layer_name][1]
                    new_cali_table_name = self.set_layer_new_th(int8_model, layer_name, new_th)
                    last_tried_method = method
                    self.mix_prec.logger.print_info("adjust layer {} th, with method {}, and threshlod {}".format(layer_name, method, new_th))
                    modified_layers[layer_name][0] += 1
                    mixmodel = MixQuantModel(self.fp32_mlir, self.chip, new_cali_table_name, mix_table)
                    outputs_cos = 1 - self.mix_prec.run_model(mixmodel, False, global_compare_layers, layers_rate, predictions_gt)
                    self.mix_prec.logger.print_info("outputs_cos_los = {}".format(outputs_cos))
                elif modified_layers[layer_name][0] == num_quantize_method:
                    if outputs_cos < modified_layers[layer_name][1]:
                        modified_layers[layer_name][1] = outputs_cos
                        modified_layers[layer_name][2] = layer_th_dicts[last_tried_method][layer_name][1]
                        modified_layers[layer_name][3] = last_tried_method
                    best_th = modified_layers[layer_name][2]
                    modified_layers[layer_name][0] += 1
                    new_cali_table_name = self.set_layer_new_th(int8_model, layer_name, best_th)
                    self.mix_prec.logger.print_info("layer {}, layer type is {}, best_th = {}, best_method = {}, best_cos_loss = {}"
                                           .format(layer_name, layer_type, best_th, modified_layers[layer_name][3], modified_layers[layer_name][1]))
                    sensitive_layer_analysis_dict[layer_name] = [modified_layers[layer_name][1], layer_type]
                    ret = True

            fp_layer_list.append(layer_name)
        return sensitive_layer_analysis_dict, new_cali_table_name

    def analysis_sensitive_layers(self, sensitive_layer_analysis_dict):
        num = 0
        num_fp32 = 0
        set_fp_layer_list = []
        sensitive_layer_analysis_dict = sorted(sensitive_layer_analysis_dict.items(), key = lambda x:x[1][0], reverse = True)
        for sensitive_layer_tuple in sensitive_layer_analysis_dict:
            self.mix_prec.logger.print_info("the layer {} is {} sensitive layer, loss is {}, type is {}".format(sensitive_layer_tuple[0], num, sensitive_layer_tuple[1][0], sensitive_layer_tuple[1][1]))
            if self.args.max_float_layers > 0 and self.args.max_float_layers > num_fp32:
                if sensitive_layer_tuple[0] in set_fp_layer_list:
                    continue
                set_fp_layer_list.append(sensitive_layer_tuple[0])
                num_fp32 += 1
                op = self.parser.get_op_by_op_name(sensitive_layer_tuple[0])
                if op.type == "top.Conv":
                    next_op_name_list = self.parser.get_next_op_by_op_name(sensitive_layer_tuple[0])
                    for next_op_name in next_op_name_list:
                        next_op = self.parser.get_op_by_op_name(next_op_name)
                        if next_op.type == "top.Scale":
                            self.mix_prec.logger.print_info("next scale op: {}".format(next_op_name))
                            set_fp_layer_list.append(next_op_name)
                            num_fp32 += 1
                elif op.type == "top.Scale":
                    pre_op_name_list = self.parser.get_pre_op_by_op_name(sensitive_layer_tuple[0])
                    for pre_op_name in pre_op_name_list:
                        pre_op = self.parser.get_op_by_op_name(pre_op_name)
                        if pre_op.type == "top.Conv":
                            self.mix_prec.logger.print_info("pre conv op: {}".format(pre_op_name))
                            set_fp_layer_list.append(pre_op_name)
                            num_fp32 += 1

            num += 1
        self.mix_prec.logger.print_info("set_fp_layer_list = {}".format(set_fp_layer_list))
        return set_fp_layer_list


    def print_log_info(self, layer_cos_list, fp_layer_list, all_int8_cos, outputs_cos, t0):
        self.mix_prec.logger.print_info('>>>run result:')
        layer_cos_list = sorted(layer_cos_list, key=lambda x: x[1], reverse=False)
        with open(self.mix_prec.quantize_table, "w") as f:
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# sample number: {}\n".format(self.mix_prec.num_sample))
            f.write("# chip: {}  mix_mode: {}\n".format(self.mix_prec.chip, self.mix_prec.mix_mode))
            f.write("# number of {} layer: {}\n".format(self.mix_prec.mix_mode, len(fp_layer_list)))
            f.write("###\n")
            f.write("# op_name   quantize_mode\n")
            for layer in fp_layer_list:
                f.write("{} {}\n".format(layer, self.mix_prec.mix_mode))
        self.mix_prec.logger.print_info(f'int8 outputs_cos:{all_int8_cos:.6f} old')
        self.mix_prec.logger.print_info(f"mix model outputs_cos:{outputs_cos:.6f}")
        self.mix_prec.logger.print_info("Output mix quantization table to {}".format(self.mix_prec.quantize_table))
        self.mix_prec.logger.print_info("total time:{}".format(time.time() - t0))

    def run(self):
        t0 = time.time()

        #setp1: float_model and int8_model inference
        float_model = MixQuantModel(self.fp32_mlir, self.chip)
        int8_model = MixQuantModel(self.fp32_mlir, self.chip, self.cali_table_name)
        layer_cos_list, predictions_gt = [], []
        global_compare_layers, layers_rate, _ = self.mix_prec.extract_global_layers()
        _ = self.mix_prec.run_model(float_model, True, global_compare_layers, layers_rate, predictions_gt)
        outputs_cos = self.mix_prec.run_model(int8_model, False, global_compare_layers, layers_rate, predictions_gt)
        if outputs_cos > self.args.expected_cos:
            float_model.clean()
            int8_model.clean()
            self.mix_prec.enable_print()
            self.mix_prec.logger.print_info(
                f'job success, current int8 cos:{outputs_cos} is higher than expected_cos:{self.args.expected_cos},no need for mix precsion')
            exit(0)
        all_int8_cos = outputs_cos
        self.mix_prec.logger.print_info("all_int8_cos={}".format(all_int8_cos))

        #setp2: generate op th dict of three defined methods(KL, Max, Percentile9999)
        all_op_names = self.parser.get_op_name_list()
        quantize_method_list = ["MAX", "Percentile9999", "KL"]
        layer_th_dicts = self.gen_multiple_thresholds(all_op_names, quantize_method_list)

        #step3: check layer names
        layer_names = self.check_layer_names(all_op_names, int8_model, layer_th_dicts, quantize_method_list)
        self.mix_prec.logger.print_info("Global metrics layer is : {}".format(global_compare_layers))

        #step4: search sensitive layer
        sensitive_layer_analysis_dict, new_cali_table_name = self.search_sensitve_layer(layer_names, quantize_method_list, float_model, int8_model, layer_th_dicts, global_compare_layers, layers_rate, predictions_gt)

        #step5: analysis sensitive layers
        self.mix_prec.enable_print()
        set_fp_layer_list = self.analysis_sensitive_layers(sensitive_layer_analysis_dict)

        #step6: generate final mix model and print info
        self.mix_prec.dot_log.gen_dot_graph()
        mix_table = self.mix_prec._gen_mix_table(set_fp_layer_list)
        mixmodel = MixQuantModel(self.fp32_mlir, self.chip, new_cali_table_name, mix_table)
        outputs_cos = self.mix_prec.run_model(mixmodel, False, global_compare_layers, layers_rate, predictions_gt)
        self.print_log_info(layer_cos_list, set_fp_layer_list, all_int8_cos, outputs_cos, t0)
        print("success sensitive layer search")
        return 'success'
