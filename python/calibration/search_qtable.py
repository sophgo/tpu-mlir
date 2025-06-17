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
import collections
from utils.mlir_parser import *
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from calibration.mix_precision import MixQuantModel
from calibration.mix_precision import MixPrecSearcher
from calibration.kld_calibrator import CalibrationTable, ActivationCalibrator, SimpleTuner
from pathlib import Path
from utils.net_dot_log import net_dot_log
from utils.log_setting import logger, setup_logger
from utils.mlir_parser import MlirParser
from utils.misc import parse_debug_cmd
from .utils import *

pymlir.set_mem_mode("force_value_mem")


class SearchQtable:

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
        args.input_num = args.inference_num
        args.loss_table = ""
        self.mix_prec = MixPrecSearcher(args)
        self.mix_prec.dot_log = net_dot_log('search_qtable_result', self.parser,
                                            self.mix_prec.logger)
        self.quantize_method_list = args.quantize_method_list
        self.debug_cmd = parse_debug_cmd(args.debug_cmd)

    def gen_multiple_thresholds(self, all_op_names, quantize_method_list):
        layer_th_dicts = {}
        for i, method_name in enumerate(quantize_method_list):
            tmp_th_dict = {}
            calibrator = ActivationCalibrator(self.args, self.selector, self.tune_ds)
            if method_name == "MAX":
                calibrator.debug_cmd = 'use_max'
            elif method_name == "Percentile9999":
                calibrator.debug_cmd = 'use_percentile9999'
            elif method_name == "MSE":
                calibrator.debug_cmd = 'use_mse'
            thresholds_map, thresholds_map_absmax, thresholds_map_scale, thresholds_map_zp, _, _, _, _ = calibrator.activation_collect_and_calc_th_new(
            )
            calibrator._clean_resource()

            thresholds_map_list = []
            op_layers = self.parser.get_op_name_list()
            cali_table = self.args.calibration_table + "_" + method_name
            if self.args.tune_num > 0:
                cali_table += ".1"
            with open(cali_table, 'w') as f:
                f.write("# mlir version: {}\n".format(pymlir.__version__))
                f.write("# mlir: {}\n".format(self.args.mlir_file))
                f.write("# genertated time: {}\n".format(datetime.datetime.now()))
                f.write("# histogram number: {}\n".format(calibrator.histogram_bin_num))
                f.write("# sample number: {}\n###\n".format(calibrator.num_samples))
                f.write("# op_name    threshold    min    max\n")
                for i, op_name in enumerate(op_layers):
                    outputs = self.parser.get_outputs_by_op_name(op_name)
                    for out in outputs:
                        if out not in thresholds_map:
                            continue
                        else:
                            if 'use_torch_observer_for_cali' in calibrator.debug_cmd:
                                qmin, qmax = -128, 127
                                scale = thresholds_map_scale[op_name]
                                zp = thresholds_map_zp[op_name]
                                threshold = float(scale * max(-(qmin - zp), qmax - zp))
                                min_value = float(scale * (qmin - zp))
                                max_value = float(scale * (qmax - zp))
                            else:
                                if out in thresholds_map:
                                    threshold = thresholds_map[out]
                                else:
                                    threshold = 1.0
                                if out in calibrator.activations_statistics:
                                    min_value, max_value, _ = calibrator.activations_statistics[out]
                                else:
                                    min_value, max_value = -1, 1
                            thresholds_map_list.append(threshold)
                            f.write("{} {:.7f} {:.7f} {:.7f}\n".format(
                                out, threshold, min_value, max_value))
            if calibrator.args.tune_num <= 0:
                return

            if self.args.tune_num > 0:
                cali_table = cali_table.rsplit(".1", 1)[0]
            calibrator.args.calibration_table = cali_table
            tunner = SimpleTuner(calibrator.args, calibrator.tune_ds, calibrator.ppa_list,
                                 thresholds_map_absmax)
            thresholds_map = tunner.run()

            tuned_threshold_list = []
            layer_name_list = []
            cali_table += "_tune"
            op_layers = calibrator.get_no_fused_tensors(op_layers)
            with open(cali_table, 'w') as f:
                f.write("# mlir version: {}\n".format(pymlir.__version__))
                f.write("# mlir: {}\n".format(self.args.mlir_file))
                f.write("# genertated time: {}\n".format(datetime.datetime.now()))
                f.write("# histogram number: {}\n".format(calibrator.histogram_bin_num))
                f.write("# sample number: {}\n".format(calibrator.num_samples))
                f.write("# tune number: {}\n###\n".format(self.args.tune_num))
                f.write("# op_name   threshold    min    max\n")
                for i, op_name in enumerate(op_layers):
                    threshold = thresholds_map[op_name]
                    if threshold <= 1e-5 or np.isnan(threshold):
                        threshold = 1e-5
                        self.mix_prec.logger.print_info(
                            "WARNING: layer {} threshold is zero. Please check the "
                            "input data correctness.".format(op_name))
                    layer_name_list.append('{}_{}'.format(i, op_name))
                    tuned_threshold_list.append(threshold)
                    min_value, max_value, _ = calibrator.activations_statistics[op_name]
                    f.write("{} {:.7f} {:.7f} {:.7f}\n".format(op_name, threshold, min_value,
                                                               max_value))
            for op_name in all_op_names:
                if op_name not in thresholds_map:
                    pass
                else:
                    if thresholds_map[op_name] <= 1e-5 or np.isnan(thresholds_map[op_name]):
                        thresholds_map[op_name] = 1e-5
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
                    "layer name of prototxt {} not match layer name of log/layer_name.txt, please cheak whether there is '! or /' in layer name "
                    .format(layer_name_check))
                exit(1)
        self.mix_prec.logger.print_info("layer name check pass !")
        return layer_names

    def search_layer_type_no_need_quant(self, layer_names, float_outputs_cos, global_compare_layers,
                                        layers_rate, predictions_gt):
        op_types = set()
        for layer_name in layer_names:
            op_type = self.parser.get_op_type_by_op_name(layer_name)
            op_types.add(op_type)

        sensitive_op_type = []
        layer_op_map = {
            layer_name: self.parser.get_op_type_by_op_name(layer_name)
            for layer_name in layer_names
        }
        cos_threshold = max(0.999, self.args.expected_cos)
        for op_type in op_types:
            fp_list = []
            for layer_name in layer_names:
                if layer_op_map[layer_name] == op_type:
                    pass
                else:
                    fp_list.append(layer_name)
            mix_table = self.mix_prec._gen_mix_table(fp_list)
            mix_model = MixQuantModel(self.fp32_mlir, self.chip, self.cali_table_name, mix_table)
            similarity = 1 - self.mix_prec.run_model_new(mix_model, False, global_compare_layers,
                                                         layers_rate, predictions_gt, -1, ['cos'])
            self.mix_prec.logger.print_info(f"op_type : {op_type}, similarity : {similarity}")
            if similarity < float_outputs_cos * cos_threshold:
                sensitive_op_type.append(op_type)
        self.mix_prec.logger.print_info(
            f"sensitive_op_type : {sensitive_op_type}, please pay attention to these types of operations"
        )
        return sensitive_op_type

    def set_layer_new_th(self, model, layer_name, value):
        op = model.parser.get_op_by_op_name(layer_name)
        threshold = float(value)
        self.cali_table.thresholds_map[layer_name][0] = threshold
        new_cali_table_name = "new_cali_table.txt"
        self.cali_table.update_to(new_cali_table_name, layer_name, threshold)
        return new_cali_table_name

    def compare_loss(self, layer_name, loss_dict, outputs_cos, outputs_snr):
        existing_cos = loss_dict[layer_name][0]
        existing_snr = loss_dict[layer_name][1]

        if outputs_cos < existing_cos:
            existing_cos = outputs_cos
        if outputs_snr < existing_snr:
            existing_snr = outputs_snr

        loss_dict[layer_name] = [existing_cos, existing_snr]

    def search_sensitve_layer(self, layer_names, quantize_method_list, float_model, int8_model,
                              layer_th_dicts, global_compare_layers, layers_rate, predictions_gt):
        if not layer_names:
            self.mix_prec.logger.print_info(
                "Layer names are empty. All operators skipped in search phase.")
            sys.exit(1)
        num_quantize_method = len(quantize_method_list)
        loss_dict = collections.defaultdict(list)
        fp_layer_list = []
        for op_name in layer_names:
            fp_layer_list.append(op_name)
        modified_layers = {}
        last_tried_method = quantize_method_list[0]
        sensitive_layer_analysis_dict = {}
        for layer_name in layer_names:
            layer_type = self.parser.get_op_type_by_op_name(layer_name)
            self.mix_prec.logger.print_info("start to handle layer: {}, type: {}".format(
                layer_name, layer_type))
            fp_layer_list.remove(layer_name)
            mix_table = self.mix_prec._gen_mix_table(fp_layer_list)
            ret = False
            while not ret:
                if layer_name not in modified_layers:
                    last_tried_method = quantize_method_list[0]
                    modified_layers[layer_name] = [
                        1,
                        float('inf'), layer_th_dicts[last_tried_method][layer_name][1],
                        last_tried_method
                    ]
                    method = last_tried_method
                    new_th = layer_th_dicts[last_tried_method][layer_name][
                        1]  # layer_th_dicts{quantize_method: {layer_name:{fmax, th}}}
                    new_cali_table_name = self.set_layer_new_th(int8_model, layer_name, new_th)
                    last_tried_method = method
                    self.mix_prec.logger.print_info(
                        "adjust layer {} th, with method {}, and threshlod {}".format(
                            layer_name, method, new_th))
                    mixmodel = MixQuantModel(self.fp32_mlir, self.chip, new_cali_table_name,
                                             mix_table)
                    if not self.args.cluster:
                        outputs_cos = 1 - self.mix_prec.run_model(
                            mixmodel, False, global_compare_layers, layers_rate, predictions_gt)
                    else:
                        outputs_cos, outputs_snr = self.mix_prec.run_model_new(
                            mixmodel, False, global_compare_layers, layers_rate, predictions_gt)
                        loss_dict[layer_name].extend([outputs_cos, outputs_snr])
                    self.mix_prec.logger.print_info("outputs_cos_los = {}".format(outputs_cos))
                elif modified_layers[layer_name][0] < num_quantize_method:
                    method_idx = modified_layers[layer_name][0]
                    method = quantize_method_list[method_idx]
                    if outputs_cos < modified_layers[layer_name][1]:
                        modified_layers[layer_name][1] = outputs_cos
                        modified_layers[layer_name][2] = layer_th_dicts[last_tried_method][
                            layer_name][1]
                        modified_layers[layer_name][3] = last_tried_method
                    new_th = layer_th_dicts[method][layer_name][1]
                    new_cali_table_name = self.set_layer_new_th(int8_model, layer_name, new_th)
                    last_tried_method = method
                    self.mix_prec.logger.print_info(
                        "adjust layer {} th, with method {}, and threshlod {}".format(
                            layer_name, method, new_th))
                    modified_layers[layer_name][0] += 1
                    mixmodel = MixQuantModel(self.fp32_mlir, self.chip, new_cali_table_name,
                                             mix_table)
                    if not self.args.cluster:
                        outputs_cos = 1 - self.mix_prec.run_model(
                            mixmodel, False, global_compare_layers, layers_rate, predictions_gt)
                    else:
                        outputs_cos, outputs_snr = self.mix_prec.run_model_new(
                            mixmodel, False, global_compare_layers, layers_rate, predictions_gt)
                        self.compare_loss(layer_name, loss_dict, outputs_cos, outputs_snr)
                    self.mix_prec.logger.print_info("outputs_cos_los = {}".format(outputs_cos))
                elif modified_layers[layer_name][0] == num_quantize_method:
                    if outputs_cos < modified_layers[layer_name][1]:
                        modified_layers[layer_name][1] = outputs_cos
                        modified_layers[layer_name][2] = layer_th_dicts[last_tried_method][
                            layer_name][1]
                        modified_layers[layer_name][3] = last_tried_method
                    best_th = modified_layers[layer_name][2]
                    modified_layers[layer_name][0] += 1
                    new_cali_table_name = self.set_layer_new_th(int8_model, layer_name, best_th)
                    self.mix_prec.logger.print_info(
                        "layer {}, layer type is {}, best_th = {}, best_method = {}, best_cos_loss = {}"
                        .format(layer_name, layer_type, best_th, modified_layers[layer_name][3],
                                modified_layers[layer_name][1]))
                    sensitive_layer_analysis_dict[layer_name] = [
                        modified_layers[layer_name][1], layer_type
                    ]
                    ret = True

            fp_layer_list.append(layer_name)
        return sensitive_layer_analysis_dict, new_cali_table_name, loss_dict

    def search_sensitve_layer_int4(self, layer_names, all_op_names, global_compare_layers,
                                   layers_rate, predictions_gt):
        loss_dict = collections.defaultdict(list)
        fp_layer_list = []
        mix_mode = 'INT8'
        with open(self.cali_table_name, 'a') as file:
            file.write("\n#int4_op\n")
        for op_name in all_op_names:
            fp_layer_list.append(op_name)
        sensitive_layer_analysis_dict = {}
        for layer_name in layer_names:
            layer_type = self.parser.get_op_type_by_op_name(layer_name)
            self.mix_prec.logger.print_info("start to handle layer: {}, type: {}".format(
                layer_name, layer_type))
            fp_layer_list.remove(layer_name)
            mix_table = self.mix_prec._gen_mix_table_4_8(fp_layer_list, mix_mode)
            with open(self.cali_table_name, 'a') as file:
                file.write(f"{layer_name}")
            int4_mix_model = MixQuantModel(self.fp32_mlir, self.chip, self.cali_table_name,
                                           mix_table, "int4")
            outputs_cos, outputs_snr = self.mix_prec.run_model_new(int4_mix_model, False,
                                                                   global_compare_layers,
                                                                   layers_rate, predictions_gt)
            if layer_name not in loss_dict:
                loss_dict[layer_name].extend([outputs_cos, outputs_snr])
            else:
                self.compare_loss(layer_name, loss_dict, outputs_cos, outputs_snr)
            with open(self.cali_table_name, 'r') as file:
                lines = file.readlines()
            with open(self.cali_table_name, 'w') as file:
                for line in lines:
                    if line.strip() != layer_name:
                        file.write(line)
            fp_layer_list.append(layer_name)
            self.mix_prec.logger.print_info("layer {}, outputs_cos:{}, outputs_snr:{}".format(
                layer_name, outputs_cos, outputs_snr))
        return loss_dict

    def search_sensitve_layer_fast(self, layer_names, all_op_names, sensitive_layer,
                                   global_compare_layers, layers_rate, predictions_gt, count):
        loss_dict = collections.defaultdict(list)
        fp_layer_list = copy.deepcopy(sensitive_layer)
        fp_layer_list += [layer for layer in layer_names if layer not in sensitive_layer]
        for layer_name in layer_names:
            fp_layer_list.remove(layer_name)
            layer_type = self.parser.get_op_type_by_op_name(layer_name)
            self.mix_prec.logger.print_info("start to handle layer: {}, type: {}".format(
                layer_name, layer_type))
            mix_table = self.mix_prec._gen_mix_table(fp_layer_list)
            int8_mix_model = MixQuantModel(self.fp32_mlir, self.chip, self.cali_table_name,
                                           mix_table)
            outputs_cos, outputs_snr = self.mix_prec.run_model_fast(int8_mix_model, False,
                                                                    global_compare_layers,
                                                                    layers_rate, predictions_gt,
                                                                    count)
            if layer_name not in loss_dict:
                loss_dict[layer_name].extend([outputs_cos, outputs_snr])
            else:
                self.compare_loss(layer_name, loss_dict, outputs_cos, outputs_snr)
            self.mix_prec.logger.print_info("layer {}, outputs_cos:{}, outputs_snr:{}".format(
                layer_name, outputs_cos, outputs_snr))
            fp_layer_list.append(layer_name)
        return loss_dict

    def analysis_sensitive_layers(self, sensitive_layer_analysis_dict, pr):
        num = 0
        num_fp32 = 0
        set_fp_layer_list = []
        sensitive_layer_analysis_dict = sorted(sensitive_layer_analysis_dict.items(),
                                               key=lambda x: x[1][0],
                                               reverse=True)
        for sensitive_layer_tuple in sensitive_layer_analysis_dict:
            if pr == True:
                self.mix_prec.logger.print_info(
                    "the layer {} is {} sensitive layer, loss is {}, type is {}".format(
                        sensitive_layer_tuple[0], num, sensitive_layer_tuple[1][0],
                        sensitive_layer_tuple[1][1]))
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
                            self.mix_prec.logger.print_info(
                                "next scale op: {}".format(next_op_name))
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

    def cluster(self, loss_dict, num_cluster):
        layer_names = list(loss_dict.keys())
        X = np.array([losses for losses in loss_dict.values()])

        kmeans = KMeans(n_clusters=num_cluster, random_state=42)
        labels = kmeans.fit_predict(X)

        result = {name: label for name, label in zip(layer_names, labels)}

        centroids = kmeans.cluster_centers_
        target_cluster = np.argmax(centroids[:, 0])
        selected_layers = [name for name, label in result.items() if label == target_cluster]

        self.mix_prec.logger.print_info("selected_layers = {}".format(selected_layers))
        return selected_layers

    def auto_select_clusters(self, X, max_clusters=10):
        best_score = -1
        best_n_clusters = 3  # 聚类数至少从3开始
        for n in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n, random_state=42)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            self.mix_prec.logger.print_info(f"n_clusters={n}, silhouette_score={score:.4f}")
            if score > best_score:
                best_score = score
                best_n_clusters = n
        return best_n_clusters

    def find_best_eps(self, X, eps_min=0.005, eps_max=0.01, n_points=10, min_samples=2):
        """
        自动选择 DBSCAN 中最佳的 eps 参数，基于轮廓系数的评价指标。

        参数:
            X (array-like): 数据集，形状为 (n_samples, n_features)
            eps_min (float): eps 搜索的最小值，默认为 0.005
            eps_max (float): eps 搜索的最大值，默认为 0.01
            n_points (int): 在 eps_min 与 eps_max 之间生成的 eps 值个数，默认为 100
            min_samples (int): DBSCAN 中的 min_samples 参数，默认为 2

        返回:
            best_eps (float or None): 根据轮廓系数选出的最佳 eps 值；
                                    若所有 eps 下聚类效果均不理想，则返回 None
        """
        eps_values = np.linspace(eps_min, eps_max, n_points)
        best_eps = None
        best_score = -1
        for eps in eps_values:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X)

            # 如果所有点都归为同一个簇或全为噪声，则跳过
            if len(set(labels)) <= 1:
                continue

            # 计算轮廓系数
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_eps = eps

        return best_eps

    def cluster_4_8(self, loss_dict):
        layer_names = list(loss_dict.keys())
        X = np.array([losses for losses in loss_dict.values()])

        # best_n_clusters = self.auto_select_clusters(X)
        # kmeans = KMeans(n_clusters= best_n_clusters, random_state=42)
        # labels = kmeans.fit_predict(X)
        # best_eps = self.find_best_eps(X)
        db = DBSCAN(eps=0.01, min_samples=2)
        db.fit(X)
        labels = db.labels_

        clusters = collections.defaultdict(list)
        for name, label in zip(layer_names, labels):
            clusters[label].append(name)

        # 获取聚类中心，centroids.shape = (n_clusters, n_features)
        #centroids = kmeans.cluster_centers_
        centroids = {}
        unique_labels = np.unique(labels)  # 包括噪声标签 -1
        for label in unique_labels:
            points = X[labels == label]
            centroids[label] = np.mean(points, axis=0)

        # 根据 centroids 的第一个元素排序各个聚类
        # sorted_labels 中的 label 顺序，就是对应聚类中心第一维从小到大的顺序
        sorted_labels = sorted(clusters.keys(), key=lambda label: centroids[label][0])
        sorted_clusters = [clusters[label] for label in sorted_labels]

        self.mix_prec.logger.print_info("sorted_clusters = {}".format(sorted_clusters))
        return sorted_clusters

    def remove_lines_from_file(self, file_path, lines_to_remove):
        """
        从文件中删除指定的行。
        参数:
        file_path (str): 文件路径
        lines_to_remove (Iterable[str]): 需要移除的内容列表（注意内容比较时去除换行符）
        """
        # 读取当前文件中的所有行
        with open(file_path, 'r') as f:
            lines = f.readlines()
        # 过滤掉与 lines_to_remove 中的内容匹配的行（去除两端空白后比较）
        filtered_lines = []
        for line in lines:
            if line.strip() not in {s.strip() for s in lines_to_remove}:
                filtered_lines.append(line)
        # 将过滤后的结果写回文件（覆盖原有内容）
        with open(file_path, 'w') as f:
            f.writelines(filtered_lines)

    def adjust_qtable(self, outputs_cos, layer_names_quant, sensitive_layer_analysis_dict,
                      new_cali_table_name, global_compare_layers, layers_rate, predictions_gt):
        if outputs_cos < self.args.expected_cos and (len(layer_names_quant) //
                                                     5) > self.args.max_float_layers:
            base_float_layers = self.args.max_float_layers
            self.args.max_float_layers = len(layer_names_quant) // 5
            set_fp_layer_list = self.analysis_sensitive_layers(sensitive_layer_analysis_dict, False)
            mix_table = self.mix_prec._gen_mix_table(set_fp_layer_list)
            mixmodel = MixQuantModel(self.fp32_mlir, self.chip, new_cali_table_name, mix_table)
            outputs_cos = self.mix_prec.run_model(mixmodel, False, global_compare_layers,
                                                  layers_rate, predictions_gt)
            self.mix_prec.logger.print_info(
                "float layer number: {}, mix model outputs_cos: {}".format(
                    self.args.max_float_layers, outputs_cos))
            if outputs_cos > self.args.expected_cos:
                lower_bound = base_float_layers
                upper_bound = len(layer_names_quant) // 5

                while lower_bound <= upper_bound:
                    self.args.max_float_layers = (lower_bound + upper_bound) // 2
                    set_fp_layer_list = self.analysis_sensitive_layers(
                        sensitive_layer_analysis_dict, False)
                    mix_table = self.mix_prec._gen_mix_table(set_fp_layer_list)
                    mixmodel = MixQuantModel(self.fp32_mlir, self.chip, new_cali_table_name,
                                             mix_table)
                    outputs_cos = self.mix_prec.run_model(mixmodel, False, global_compare_layers,
                                                          layers_rate, predictions_gt)
                    self.mix_prec.logger.print_info(
                        "float layer number: {}, mix model outputs_cos: {}".format(
                            self.args.max_float_layers, outputs_cos))

                    if outputs_cos > self.args.expected_cos:
                        upper_bound = self.args.max_float_layers - 1
                    else:
                        lower_bound = self.args.max_float_layers + 1

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
        self.mix_prec.logger.print_info("Output mix quantization table to {}".format(
            self.mix_prec.quantize_table))
        self.mix_prec.logger.print_info("total time:{}".format(time.time() - t0))

    def print_log_info_4_8(self, fp_layer_list, int8_outputs_cos, t0):
        self.mix_prec.logger.print_info('>>>run result:')
        with open(self.mix_prec.quantize_table, "w") as f:
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# sample number: {}\n".format(self.mix_prec.num_sample))
            f.write("# chip: {}  mix_mode: {}\n".format(self.mix_prec.chip, "INT8"))
            f.write("# number of {} layer: {}\n".format("INT8", len(fp_layer_list)))
            f.write("###\n")
            f.write("# op_name   quantize_mode\n")
            for layer in fp_layer_list:
                f.write("{} {}\n".format(layer, "INT8"))
        self.mix_prec.logger.print_info(f'int8 outputs_cos:{int8_outputs_cos:.6f} old')
        self.mix_prec.logger.print_info("Output mix quantization table to {}".format(
            self.mix_prec.quantize_table))
        self.mix_prec.logger.print_info("total time:{}".format(time.time() - t0))

    def run(self):
        t0 = time.time()

        #setp1: generate op th dict of defined methods(KL, Max, Percentile9999, MSE)
        all_op_names = self.parser.get_op_name_list()
        quantize_method_list = self.quantize_method_list
        layer_th_dicts = self.gen_multiple_thresholds(all_op_names, quantize_method_list)
        self.mix_prec.logger.print_info("quantize_method_list={}".format(quantize_method_list))

        try:
            mse_cali_table = self.args.calibration_table + "_MSE_tune"
            with open(mse_cali_table, 'r') as file:
                data = file.read()
        except Exception as e:
            _cali_table = self.args.calibration_table + "_" + quantize_method_list[0] + "_tune"
            with open(_cali_table, 'r') as file:
                data = file.read()
        with open(self.cali_table_name, 'w') as file:
            file.write(data)

        #setp2: float_model and int8_model inference
        float_model = MixQuantModel(self.fp32_mlir, self.chip)
        int8_model = MixQuantModel(self.fp32_mlir, self.chip, self.cali_table_name)
        float_outputs_cos = 1.0
        layer_cos_list, predictions_gt = [], []
        global_compare_layers, layers_rate, _ = self.mix_prec.extract_global_layers()
        _ = self.mix_prec.run_model(float_model, True, global_compare_layers, layers_rate,
                                    predictions_gt)
        outputs_cos = self.mix_prec.run_model(int8_model, False, global_compare_layers, layers_rate,
                                              predictions_gt)
        if outputs_cos > self.args.expected_cos:
            float_model.clean()
            int8_model.clean()
            self.mix_prec.enable_print()
            self.mix_prec.logger.print_info(
                f'job success, current int8 cos:{outputs_cos} is higher than expected_cos:{self.args.expected_cos},no need for mix precsion'
            )
            exit(0)
        all_int8_cos = outputs_cos
        self.mix_prec.logger.print_info("all_int8_cos={}".format(all_int8_cos))
        self.cali_table = CalibrationTable(self.cali_table_name)

        #step3: check layer names
        float_ops = self.mix_prec.get_fixed_float_layers(int8_model, global_compare_layers,
                                                         layers_rate, predictions_gt)
        layer_names = self.check_layer_names(all_op_names, int8_model, layer_th_dicts,
                                             quantize_method_list)
        self.mix_prec.logger.print_info("all layer number: {}".format(len(layer_names)))
        layer_names_quant = [layer for layer in layer_names if layer not in float_ops]
        self.mix_prec.logger.print_info("all layer number no float: {}".format(
            len(layer_names_quant)))
        sensitive_op_type = self.search_layer_type_no_need_quant(layer_names_quant,
                                                                 float_outputs_cos,
                                                                 global_compare_layers, layers_rate,
                                                                 predictions_gt)
        layer_names = [
            layer for layer in layer_names_quant
            if self.parser.get_op_type_by_op_name(layer) in sensitive_op_type
        ]
        self.mix_prec.logger.print_info("transformer model: {}, all search layer number: {}".format(
            self.args.transformer, len(layer_names)))
        self.mix_prec.logger.print_info(
            "Global metrics layer is : {}".format(global_compare_layers))

        #step4: search sensitive layer
        sensitive_layer_analysis_dict, new_cali_table_name, loss_dict = self.search_sensitve_layer(
            layer_names, quantize_method_list, float_model, int8_model, layer_th_dicts,
            global_compare_layers, layers_rate, predictions_gt)

        #step5: analysis sensitive layers
        self.mix_prec.enable_print()
        if self.args.cluster:
            selected_fp_layers = self.cluster(loss_dict, 2)
            mix_table = self.mix_prec._gen_mix_table(selected_fp_layers)
        else:
            set_fp_layer_list = self.analysis_sensitive_layers(sensitive_layer_analysis_dict, True)
            mix_table = self.mix_prec._gen_mix_table(set_fp_layer_list)

        #step6: generate final mix model and print info
        self.mix_prec.dot_log.gen_dot_graph()
        mixmodel = MixQuantModel(self.fp32_mlir, self.chip, new_cali_table_name, mix_table)
        outputs_cos = self.mix_prec.run_model(mixmodel, False, global_compare_layers, layers_rate,
                                              predictions_gt)
        self.mix_prec.logger.print_info("float layer number: {}, mix model outputs_cos: {}".format(
            self.args.max_float_layers, outputs_cos))

        if not self.args.cluster:
            self.adjust_qtable(outputs_cos, layer_names_quant, sensitive_layer_analysis_dict,
                               new_cali_table_name, global_compare_layers, layers_rate,
                               predictions_gt)
            self.print_log_info(layer_cos_list, set_fp_layer_list, all_int8_cos, outputs_cos, t0)
        else:
            self.print_log_info(layer_cos_list, selected_fp_layers, all_int8_cos, outputs_cos, t0)
        print("success search qtable")
        return 'success'

    def run_4_8(self):

        t0 = time.time()
        layer_cos_list, predictions_gt = [], []
        float_model = MixQuantModel(self.fp32_mlir, self.chip)
        int8_model = MixQuantModel(self.fp32_mlir, self.chip, self.cali_table_name)
        global_compare_layers, layers_rate, _ = self.mix_prec.extract_global_layers()
        _ = self.mix_prec.run_model(float_model, True, global_compare_layers, layers_rate,
                                    predictions_gt)
        int8_outputs_cos = self.mix_prec.run_model(int8_model, False, global_compare_layers,
                                                   layers_rate, predictions_gt)
        self.mix_prec.logger.print_info(f'current int8 cos:{int8_outputs_cos}')

        all_op_names = self.parser.get_op_name_list()
        search_op_type = ['top.Conv', 'top.MatMul']
        layer_names = [
            layer for layer in all_op_names
            if self.parser.get_op_type_by_op_name(layer) in search_op_type
        ]

        loss_dict = self.search_sensitve_layer_int4(layer_names, all_op_names,
                                                    global_compare_layers, layers_rate,
                                                    predictions_gt)

        sorted_loss_items = sorted(loss_dict.items(), key=lambda item: item[1][0])
        for layer_name, values in sorted_loss_items:
            outputs_cos, outputs_snr = values
            self.mix_prec.logger.print_info("Layer: {}, outputs_cos: {}, outputs_snr: {}".format(
                layer_name, outputs_cos, outputs_snr))

        sorted_clusters = self.cluster_4_8(loss_dict)
        fp_layer_list = copy.deepcopy(all_op_names)
        for cluster in sorted_clusters:
            for op_name in cluster:
                with open(self.cali_table_name, 'a') as file:
                    file.write(f"{op_name}\n")
                fp_layer_list.remove(op_name)
            mix_table = self.mix_prec._gen_mix_table_4_8(fp_layer_list, 'INT8')
            int4_mix_model = MixQuantModel(self.fp32_mlir, self.chip, self.cali_table_name,
                                           mix_table, "int4")
            outputs_cos, outputs_snr = self.mix_prec.run_model_new(int4_mix_model, False,
                                                                   global_compare_layers,
                                                                   layers_rate, predictions_gt)
            if 1 - outputs_cos < int8_outputs_cos * 0.95:
                self.remove_lines_from_file(self.cali_table_name, cluster)
                for op_name in cluster:
                    fp_layer_list.append(op_name)
                break
        self.print_log_info_4_8(fp_layer_list, int8_outputs_cos, t0)

    def run_fast(self):
        t0 = time.time()
        layer_cos_list, predictions_gt = [], []
        float_model = MixQuantModel(self.fp32_mlir, self.chip)
        int8_model = MixQuantModel(self.fp32_mlir, self.chip, self.cali_table_name)
        global_compare_layers, layers_rate, _ = self.mix_prec.extract_global_layers()
        _ = self.mix_prec.run_model(float_model, True, global_compare_layers, layers_rate,
                                    predictions_gt)
        int8_outputs_cos = self.mix_prec.run_model(int8_model, False, global_compare_layers,
                                                   layers_rate, predictions_gt)
        if int8_outputs_cos > self.args.expected_cos:
            float_model.clean()
            int8_model.clean()
            self.mix_prec.enable_print()
            self.mix_prec.logger.print_info(
                f'job success, current int8 cos:{int8_outputs_cos} is higher than expected_cos:{self.args.expected_cos},no need for mix precsion'
            )
            exit(0)

        float_outputs_cos = 1.0
        all_op_names = self.parser.get_op_name_list()
        sensitive_op_type = self.search_layer_type_no_need_quant(all_op_names, float_outputs_cos,
                                                                 global_compare_layers, layers_rate,
                                                                 predictions_gt)
        layer_names = [
            layer for layer in all_op_names
            if self.parser.get_op_type_by_op_name(layer) in sensitive_op_type
        ]
        self.mix_prec.logger.print_info("transformer model: {}, all search layer number: {}".format(
            self.args.transformer, len(layer_names)))

        sensitive_layer = []
        cos_sim = int8_outputs_cos
        count = 0
        eps = 0
        while int8_outputs_cos < self.args.expected_cos:
            loss_dict = self.search_sensitve_layer_fast(layer_names, all_op_names, sensitive_layer,
                                                        global_compare_layers, layers_rate,
                                                        predictions_gt, count)

            keys = list(loss_dict.keys())
            sorted_by_first = sorted(keys, key=lambda k: loss_dict[k][0])
            rank_first = {key: idx for idx, key in enumerate(sorted_by_first)}
            sorted_by_second = sorted(keys, key=lambda k: loss_dict[k][1])
            rank_second = {key: idx for idx, key in enumerate(sorted_by_second)}
            total_rank = {key: rank_first[key] + rank_second[key] for key in keys}
            sorted_keys = sorted(keys, key=lambda k: (total_rank[k], loss_dict[k][0]))

            top_5_layers = sorted_keys[-5:]
            layer_names = [layer for layer in layer_names if layer not in top_5_layers]
            sensitive_layer += [layer for layer in top_5_layers if layer not in sensitive_layer]

            mix_table = self.mix_prec._gen_mix_table(sensitive_layer)
            int8_model = MixQuantModel(self.fp32_mlir, self.chip, self.cali_table_name, mix_table)
            outputs_cos = self.mix_prec.run_model(int8_model, False, global_compare_layers,
                                                  layers_rate, predictions_gt)
            if (outputs_cos - cos_sim < eps and outputs_cos - cos_sim
                    < 0.001) or len(sensitive_layer) > 0.2 * len(all_op_names):
                break
            else:
                if count == 0:
                    eps = abs(outputs_cos - cos_sim) / 2
                cos_sim = outputs_cos
                count += 1
                if count > self.args.inference_num - 1:
                    break
        self.print_log_info(layer_cos_list, sensitive_layer, int8_outputs_cos, outputs_cos, t0)
