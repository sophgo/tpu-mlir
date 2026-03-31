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
from calibration.kld_calibrator import CalibrationTable, ActivationCalibrator
from pathlib import Path
from utils.net_dot_log import net_dot_log
from utils.log_setting import logger, setup_logger
from utils.mlir_parser import MlirParser
from utils.misc import parse_debug_cmd
from .utils import *
import gc

pymlir.set_mem_mode("force_value_mem")


class SearchQtable:

    def __init__(self, args, selector, tune_ds, qtable=None):
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
        self.debug_cmd = args.debug_cmd
        self.qtable = qtable
        self.mix_prec.qtable = qtable
        self.low_prec, self.high_prec = get_mix_prec(self.args.chip, self.args.mix_mode,
                                                     self.args.fp_type)

    def check_layer_names(self, all_op_names, int8_model, layer_th_dicts, quantize_method_list):
        layer_names = []
        layer_name2layer_type_dict = {}
        ignored_layers = ["Coeff", "Accuracy"]
        for layer_name in all_op_names:
            ignore = False
            layer_proto = int8_model.parser.get_op_by_op_name(layer_name)
            if layer_proto is not None:
                ignore = True if layer_proto.type in ignored_layers else False
                if not ignore:
                    layer_names.append(layer_name)
                    layer_name2layer_type_dict[layer_name] = layer_proto.type
            else:
                layer_names.append(layer_name)
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
            mix_table = self.mix_prec._gen_mix_table(fp_list, self.qtable)
            mix_model = MixQuantModel(self.fp32_mlir, self.chip, self.low_prec, self.high_prec,
                                      self.cali_table_name, mix_table)
            similarity = 1 - self.mix_prec.run_model_new(mix_model, False, global_compare_layers,
                                                         layers_rate, predictions_gt, -1, ['cos'])
            self.mix_prec.logger.print_info(f"op_type : {op_type}, similarity : {similarity}")
            if similarity < float_outputs_cos * cos_threshold:
                sensitive_op_type.append(op_type)
        self.mix_prec.logger.print_info(
            f"sensitive_op_type : {sensitive_op_type}, please pay attention to these types of operations"
        )
        return sensitive_op_type

    def set_layer_new_th(self, layer_name, value):
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

    def search_sensitve_layer(self, layer_names, quantize_method_list, layer_th_dicts,
                              global_compare_layers, layers_rate, predictions_gt):
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
            if self.qtable.exists(layer_name):
                continue
            else:
                layer_type = self.parser.get_op_type_by_op_name(layer_name)
                self.mix_prec.logger.print_info("start to handle layer: {}, type: {}".format(
                    layer_name, layer_type))
                fp_layer_list.remove(layer_name)
                mix_table = self.mix_prec._gen_mix_table(fp_layer_list, self.qtable)
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
                        new_cali_table_name = self.set_layer_new_th(layer_name, new_th)
                        last_tried_method = method
                        self.mix_prec.logger.print_info(
                            "adjust layer {} th, with method {}, and threshlod {}".format(
                                layer_name, method, new_th))
                        mixmodel = MixQuantModel(
                            self.fp32_mlir,
                            self.chip,
                            self.low_prec,
                            self.high_prec,
                            new_cali_table_name,
                            mix_table,
                            #  using_cuda=False)
                            using_cuda=True)
                        if not self.args.cluster:
                            outputs_cos = 1 - self.mix_prec.run_model(
                                mixmodel, False, global_compare_layers, layers_rate, predictions_gt)
                        else:
                            outputs_cos, outputs_snr = self.mix_prec.run_model_new(
                                mixmodel, False, global_compare_layers, layers_rate, predictions_gt)
                            loss_dict[layer_name].extend([outputs_cos, outputs_snr])
                        self.mix_prec.logger.print_info("outputs_cos_los = {}".format(outputs_cos))
                        mixmodel.clean()
                        del mixmodel
                    elif modified_layers[layer_name][0] < num_quantize_method:
                        method_idx = modified_layers[layer_name][0]
                        method = quantize_method_list[method_idx]
                        if outputs_cos < modified_layers[layer_name][1]:
                            modified_layers[layer_name][1] = outputs_cos
                            modified_layers[layer_name][2] = layer_th_dicts[last_tried_method][
                                layer_name][1]
                            modified_layers[layer_name][3] = last_tried_method
                        new_th = layer_th_dicts[method][layer_name][1]
                        new_cali_table_name = self.set_layer_new_th(layer_name, new_th)
                        last_tried_method = method
                        self.mix_prec.logger.print_info(
                            "adjust layer {} th, with method {}, and threshlod {}".format(
                                layer_name, method, new_th))
                        modified_layers[layer_name][0] += 1
                        mixmodel = MixQuantModel(self.fp32_mlir,
                                                 self.chip,
                                                 self.low_prec,
                                                 self.high_prec,
                                                 new_cali_table_name,
                                                 mix_table,
                                                 using_cuda=True)
                        #  using_cuda=False)
                        if not self.args.cluster:
                            outputs_cos = 1 - self.mix_prec.run_model(
                                mixmodel, False, global_compare_layers, layers_rate, predictions_gt)
                        else:
                            outputs_cos, outputs_snr = self.mix_prec.run_model_new(
                                mixmodel, False, global_compare_layers, layers_rate, predictions_gt)
                            self.compare_loss(layer_name, loss_dict, outputs_cos, outputs_snr)
                        self.mix_prec.logger.print_info("outputs_cos_los = {}".format(outputs_cos))
                        mixmodel.clean()
                        del mixmodel
                    elif modified_layers[layer_name][0] == num_quantize_method:
                        if outputs_cos < modified_layers[layer_name][1]:
                            modified_layers[layer_name][1] = outputs_cos
                            modified_layers[layer_name][2] = layer_th_dicts[last_tried_method][
                                layer_name][1]
                            modified_layers[layer_name][3] = last_tried_method
                        best_th = modified_layers[layer_name][2]
                        modified_layers[layer_name][0] += 1
                        new_cali_table_name = self.set_layer_new_th(layer_name, best_th)
                        self.mix_prec.logger.print_info(
                            "layer {}, layer type is {}, best_th = {}, best_method = {}, best_cos_loss = {}"
                            .format(layer_name, layer_type, best_th, modified_layers[layer_name][3],
                                    modified_layers[layer_name][1]))
                        sensitive_layer_analysis_dict[layer_name] = [
                            modified_layers[layer_name][1], layer_type
                        ]
                        ret = True
                fp_layer_list.append(layer_name)
            gc.collect()
        return sensitive_layer_analysis_dict, new_cali_table_name, loss_dict

    def search_sensitve_layer_int4(self, layer_names, all_op_names, global_compare_layers,
                                   layers_rate, predictions_gt):
        loss_dict = collections.defaultdict(list)
        fp_layer_list = []
        with open(self.cali_table_name, 'a') as file:
            file.write("\n#int4_op\n")
        for op_name in all_op_names:
            fp_layer_list.append(op_name)
        sensitive_layer_analysis_dict = {}
        for layer_name in layer_names:
            if self.qtable.exists(layer_name):
                continue
            else:
                layer_type = self.parser.get_op_type_by_op_name(layer_name)
                self.mix_prec.logger.print_info("start to handle layer: {}, type: {}".format(
                    layer_name, layer_type))
                fp_layer_list.remove(layer_name)
                mix_table = self.mix_prec._gen_mix_table(mix_ops=fp_layer_list,
                                                         qtable=self.qtable,
                                                         high_prec_type=self.high_prec)
                with open(self.cali_table_name, 'a') as file:
                    file.write(f"{layer_name}")
                int4_mix_model = MixQuantModel(self.fp32_mlir, self.chip, self.low_prec,
                                               self.high_prec, self.cali_table_name, mix_table)
                outputs_cos, outputs_snr = self.mix_prec.run_model_new(
                    int4_mix_model, False, global_compare_layers, layers_rate, predictions_gt)
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

    def search_sensitve_layer_w4a8(self, layer_names, all_op_names, global_compare_layers,
                                   layers_rate, predictions_gt):
        # w4a8 is trying w4a8 ops based on int8, use low_prec as mixed type
        loss_dict = collections.defaultdict(list)
        fp_layer_list = []
        for layer_name in layer_names:
            if self.qtable.exists(layer_name):
                continue
            else:
                layer_type = self.parser.get_op_type_by_op_name(layer_name)
                self.mix_prec.logger.print_info("start to handle layer: {}, type: {}".format(
                    layer_name, layer_type))
                fp_layer_list.append(layer_name)
                mix_table = self.mix_prec._gen_mix_table(mix_ops=fp_layer_list,
                                                         qtable=self.qtable,
                                                         high_prec_type=self.low_prec)
                mix_model = MixQuantModel(self.fp32_mlir, self.chip, None, self.low_prec,
                                          self.cali_table_name, mix_table)
                outputs_cos, outputs_snr = self.mix_prec.run_model_new(
                    mix_model, False, global_compare_layers, layers_rate, predictions_gt)
                if layer_name not in loss_dict:
                    loss_dict[layer_name].extend([outputs_cos, outputs_snr])
                else:
                    self.compare_loss(layer_name, loss_dict, outputs_cos, outputs_snr)
                fp_layer_list.remove(layer_name)
                self.mix_prec.logger.print_info("layer {}, outputs_cos:{}, outputs_snr:{}".format(
                    layer_name, outputs_cos, outputs_snr))
        return loss_dict

    def search_sensitve_layer_fast(self, layer_names, sensitive_layer, global_compare_layers,
                                   layers_rate, predictions_gt, count):
        loss_dict = collections.defaultdict(list)
        fp_layer_list = copy.deepcopy(sensitive_layer)
        fp_layer_list += [layer for layer in layer_names if layer not in sensitive_layer]
        for layer_name in layer_names:
            if self.qtable.exists(layer_name):
                continue
            else:
                fp_layer_list.remove(layer_name)
                layer_type = self.parser.get_op_type_by_op_name(layer_name)
                self.mix_prec.logger.print_info("start to handle layer: {}, type: {}".format(
                    layer_name, layer_type))
                mix_table = self.mix_prec._gen_mix_table(fp_layer_list, self.qtable)
                int8_mix_model = MixQuantModel(self.fp32_mlir, self.chip, self.low_prec,
                                               self.high_prec, self.cali_table_name, mix_table)
                outputs_cos, outputs_snr = self.mix_prec.run_model_fast(
                    int8_mix_model, False, global_compare_layers, layers_rate, predictions_gt,
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
        best_n_clusters = 3  # Cluster number must be at least 3
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

            # If all points are assigned to the same cluster or all are noise, skip.
            if len(set(labels)) <= 1:
                continue

            # Calculate Silhouette Coefficient
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

        # Get the cluster centers, centroids.shape = (n_clusters, n_features)
        #centroids = kmeans.cluster_centers_
        centroids = {}
        unique_labels = np.unique(labels)  # Includes noise labels -1
        for label in unique_labels:
            points = X[labels == label]
            centroids[label] = np.mean(points, axis=0)

        # Sort each cluster by the first element of centroids.
        # sorted_labels' label order corresponds to the first dimension of cluster centers in ascending order.
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
        # Read all lines in the current file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        # Filter out lines that match the content in lines_to_remove (trim whitespace before comparison)
        filtered_lines = []
        for line in lines:
            if line.strip() not in {s.strip() for s in lines_to_remove}:
                filtered_lines.append(line)
        # Write the filtered results back to the file (overwriting existing content)
        with open(file_path, 'w') as f:
            f.writelines(filtered_lines)

    def adjust_qtable(self, outputs_cos, layer_names_quant, sensitive_layer_analysis_dict,
                      new_cali_table_name, global_compare_layers, layers_rate, predictions_gt):
        if outputs_cos < self.args.expected_cos and (len(layer_names_quant) //
                                                     5) > self.args.max_float_layers:
            base_float_layers = self.args.max_float_layers
            self.args.max_float_layers = len(layer_names_quant) // 5
            set_fp_layer_list = self.analysis_sensitive_layers(sensitive_layer_analysis_dict, False)
            total_fp_layers_analysis = set_fp_layer_list
            mix_table = self.mix_prec._gen_mix_table(total_fp_layers_analysis, self.qtable)
            mixmodel = MixQuantModel(self.fp32_mlir, self.chip, self.low_prec, self.high_prec,
                                     new_cali_table_name, mix_table)
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
                    total_fp_layers_analysis = set_fp_layer_list
                    mix_table = self.mix_prec._gen_mix_table(total_fp_layers_analysis, self.qtable)
                    mixmodel = MixQuantModel(self.fp32_mlir, self.chip, self.low_prec,
                                             self.high_prec, new_cali_table_name, mix_table)
                    outputs_cos = self.mix_prec.run_model(mixmodel, False, global_compare_layers,
                                                          layers_rate, predictions_gt)
                    self.mix_prec.logger.print_info(
                        "float layer number: {}, mix model outputs_cos: {}".format(
                            self.args.max_float_layers, outputs_cos))

                    if outputs_cos > self.args.expected_cos:
                        upper_bound = self.args.max_float_layers - 1
                    else:
                        lower_bound = self.args.max_float_layers + 1
            return total_fp_layers_analysis, outputs_cos
        return list(sensitive_layer_analysis_dict), outputs_cos

    def print_log_info(self, layer_cos_list, fp_layer_list, all_int8_cos, outputs_cos, t0):
        self.mix_prec.logger.print_info('>>>run result:')
        layer_cos_list = sorted(layer_cos_list, key=lambda x: x[1], reverse=False)
        if self.qtable is not None:
            qtable = copy.deepcopy(self.qtable)
        else:
            qtable = QuantizeTable()
        if self.args.mix_mode in ['wi8ai8_fp', 'wf8af8_fp']:
            fp_type = FLOAT_MAP[
                self.args.chip] if self.args.fp_type == 'auto' else self.args.fp_type
        elif self.args.mix_mode in ['wi4ai8_wi8ai8']:
            fp_type = self.low_prec
        else:
            fp_type = self.high_prec
        qtable.append_custom(fp_layer_list, [fp_type] * len(fp_layer_list))
        qtable.dump(self.mix_prec.quantize_table)
        self.mix_prec.logger.print_info(f'int8 outputs_cos:{all_int8_cos:.6f} old')
        self.mix_prec.logger.print_info(f"mix model outputs_cos:{outputs_cos:.6f}")
        self.mix_prec.logger.print_info("Output mix quantization table to {}".format(
            self.mix_prec.quantize_table))
        self.mix_prec.logger.print_info("total time:{}".format(time.time() - t0))

    def print_log_info_4bit(self, fp_layer_list, int8_outputs_cos, t0, mix_mode: str = 'INT8'):
        self.mix_prec.logger.print_info('>>>run result:')
        if self.qtable is not None:
            qtable = copy.deepcopy(self.qtable)
        else:
            qtable = QuantizeTable()
        qtable.append_custom(fp_layer_list, [mix_mode] * len(fp_layer_list))
        qtable.dump(self.mix_prec.quantize_table)
        self.mix_prec.logger.print_info(f'int8 outputs_cos:{int8_outputs_cos:.6f} old')
        self.mix_prec.logger.print_info("Output mix quantization table to {}".format(
            self.mix_prec.quantize_table))
        self.mix_prec.logger.print_info("total time:{}".format(time.time() - t0))

    def run(self):
        t0 = time.time()

        #setp1: generate op th dict of defined methods(KL, Max, Percentile9999, MSE)
        all_op_names = self.parser.get_op_name_list()
        all_op_names = get_no_fused_tensors(self.parser, all_op_names)
        quantize_method_list = [x.lower() for x in self.quantize_method_list]
        suffix = "_tune" if self.args.tune_num > 0 else ""
        calibrator = ActivationCalibrator(self.args, self.selector, self.tune_ds, using_cuda=True)
        # calibrator = ActivationCalibrator(self.args, self.selector, self.tune_ds, using_cuda=False)
        calibrator.calibration_method = quantize_method_list
        layer_th_dicts = calibrator.gen_multiple_thresholds(all_op_names, quantize_method_list)
        del calibrator
        self.mix_prec.logger.print_info("quantize_method_list={}".format(quantize_method_list))

        try:
            mse_cali_table = self.args.calibration_table + "_mse" + suffix
            with open(mse_cali_table, 'r') as file:
                data = file.read()
        except Exception as e:
            _cali_table = self.args.calibration_table + "_" + quantize_method_list[0] + suffix
            with open(_cali_table, 'r') as file:
                data = file.read()
        with open(self.cali_table_name, 'w') as file:
            file.write(data)
        #setp2: float_model and int8_model inference
        mix_table = None if self.qtable is None else self.mix_prec._gen_mix_table([], self.qtable)
        float_model = MixQuantModel(self.fp32_mlir, self.chip, None, self.high_prec)
        int8_model = MixQuantModel(self.fp32_mlir, self.chip, self.low_prec, self.high_prec,
                                   self.cali_table_name, mix_table)
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
            if mix_table is None:
                self.mix_prec.logger.print_info(
                    f'job success, current int8 cos:{outputs_cos} is higher than expected_cos:{self.args.expected_cos},no need for mix precsion'
                )
            else:
                self.mix_prec.logger.print_info(
                    f'job success, current int8 cos:{outputs_cos} is higher than expected_cos:{self.args.expected_cos} with layers:{self.qtable.get_all_fp_layers()}'
                )
            exit(0)
        all_int8_cos = outputs_cos
        self.mix_prec.logger.print_info(
            "all_int8_cos={} with default mse_tune calitable".format(all_int8_cos))
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
        self.mix_prec.logger.print_info("all search layer number: {}".format(len(layer_names)))
        self.mix_prec.logger.print_info(
            "Global metrics layer is : {}".format(global_compare_layers))

        float_model.clean()
        # int8_model.clean()
        del float_model
        # del int8_model

        #step4: search sensitive layer
        t1 = time.time()
        sensitive_layer_analysis_dict, new_cali_table_name, loss_dict = self.search_sensitve_layer(
            layer_names, quantize_method_list, layer_th_dicts, global_compare_layers, layers_rate,
            predictions_gt)
        t2 = time.time()
        self.mix_prec.logger.print_info("total time of sensitive_layer search is: {}".format(t2 -
                                                                                             t1))

        #step5: analysis sensitive layers
        self.mix_prec.enable_print()
        if self.args.cluster:
            selected_fp_layers = self.cluster(loss_dict, 2)
            mix_table = self.mix_prec._gen_mix_table(selected_fp_layers, self.qtable)
        else:
            set_fp_layer_list = self.analysis_sensitive_layers(sensitive_layer_analysis_dict, True)
            mix_table = self.mix_prec._gen_mix_table(set_fp_layer_list, self.qtable)

        #step6: generate final mix model and print info
        self.mix_prec.dot_log.gen_dot_graph()
        mixmodel = MixQuantModel(
            self.fp32_mlir,
            self.chip,
            self.low_prec,
            self.high_prec,
            new_cali_table_name,
            mix_table,
            #  using_cuda=False)
            using_cuda=True)
        outputs_cos = self.mix_prec.run_model(mixmodel, False, global_compare_layers, layers_rate,
                                              predictions_gt)
        self.mix_prec.logger.print_info("float layer number: {}, mix model outputs_cos: {}".format(
            self.args.max_float_layers, outputs_cos))

        if not self.args.cluster:
            final_fp_layers, outputs_cos = self.adjust_qtable(outputs_cos, layer_names_quant,
                                                              sensitive_layer_analysis_dict,
                                                              new_cali_table_name,
                                                              global_compare_layers, layers_rate,
                                                              predictions_gt)
            self.print_log_info(layer_cos_list, final_fp_layers, all_int8_cos, outputs_cos, t0)
        else:
            self.print_log_info(layer_cos_list, selected_fp_layers, all_int8_cos, outputs_cos, t0)
        print("success search qtable")
        return 'success'

    def run_4_8(self):

        t0 = time.time()
        layer_cos_list, predictions_gt = [], []
        fp_type = 'F32' if (
            self.args.fp_type == 'auto'
            and 'F32' in chip_support_mix_fp_type[self.args.chip]) else FLOAT_MAP[self.args.chip]
        float_model = MixQuantModel(self.fp32_mlir, self.chip, None, fp_type)
        mix_table = None if self.qtable is None else self.mix_prec._gen_mix_table([], self.qtable)
        int8_model = MixQuantModel(self.fp32_mlir, self.chip, self.low_prec, self.high_prec,
                                   self.cali_table_name, mix_table)
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
            mix_table = self.mix_prec._gen_mix_table(mix_ops=fp_layer_list,
                                                     qtable=self.qtable,
                                                     high_prec_type=self.high_prec)
            int4_mix_model = MixQuantModel(self.fp32_mlir, self.chip, self.low_prec, self.high_prec,
                                           self.cali_table_name, mix_table)
            outputs_cos, outputs_snr = self.mix_prec.run_model_new(int4_mix_model, False,
                                                                   global_compare_layers,
                                                                   layers_rate, predictions_gt)
            if 1 - outputs_cos < int8_outputs_cos * 0.95:
                self.remove_lines_from_file(self.cali_table_name, cluster)
                for op_name in cluster:
                    fp_layer_list.append(op_name)
                break
        self.print_log_info_4bit(fp_layer_list, int8_outputs_cos, t0, mix_mode='INT8')

    def run_fast(self):
        t0 = time.time()
        layer_cos_list, predictions_gt = [], []
        float_model = MixQuantModel(self.fp32_mlir, self.chip, None,
                                    self.high_prec)  # assume for float mix
        mix_table = None if self.qtable is None else self.mix_prec._gen_mix_table([], self.qtable)
        int8_model = MixQuantModel(self.fp32_mlir, self.chip, self.low_prec, self.high_prec,
                                   self.cali_table_name, mix_table)
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
        self.mix_prec.logger.print_info("all search layer number: {}".format(len(layer_names)))

        sensitive_layer = []
        cos_sim = int8_outputs_cos
        count = 0
        eps = 0
        while int8_outputs_cos < self.args.expected_cos:
            loss_dict = self.search_sensitve_layer_fast(layer_names, sensitive_layer,
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

            mix_table = self.mix_prec._gen_mix_table(sensitive_layer, self.qtable)
            int8_model = MixQuantModel(self.fp32_mlir, self.chip, self.low_prec, self.high_prec,
                                       self.cali_table_name, mix_table)
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

    def run_w4a8(self):
        t0 = time.time()
        layer_cos_list, predictions_gt = [], []
        fp_type = 'F32' if (
            self.args.fp_type == 'auto'
            and 'F32' in chip_support_mix_fp_type[self.args.chip]) else FLOAT_MAP[self.args.chip]
        float_model = MixQuantModel(self.fp32_mlir, self.chip, None, fp_type)
        mix_table = None if self.qtable is None else self.mix_prec._gen_mix_table([], self.qtable)
        # select w4a8 ops based on int8 base
        int8_model = MixQuantModel(self.fp32_mlir, self.chip, None, self.low_prec,
                                   self.cali_table_name, mix_table)
        global_compare_layers, layers_rate, _ = self.mix_prec.extract_global_layers()
        _ = self.mix_prec.run_model(float_model, True, global_compare_layers, layers_rate,
                                    predictions_gt)
        int8_outputs_cos = self.mix_prec.run_model(int8_model, False, global_compare_layers,
                                                   layers_rate, predictions_gt)
        self.mix_prec.logger.print_info(f'current int8 cos:{int8_outputs_cos}')

        all_op_names = self.parser.get_op_name_list()
        all_ops = self.parser.ops
        search_op_type = ['top.Conv', 'top.MatMul']
        weight_file = self.parser.module_weight_file
        weights = np.load(weight_file)
        layer_names = [
            layer.name for layer in all_ops
            if (self.parser.get_op_type_by_op_name(layer.name) in search_op_type and (
                layer.opds[1] in weights if self.parser.get_op_type_by_op_name(layer.name) ==
                'top.MatMul' else True))
        ]

        loss_dict = self.search_sensitve_layer_w4a8(layer_names, all_op_names,
                                                    global_compare_layers, layers_rate,
                                                    predictions_gt)

        sorted_loss_items = sorted(loss_dict.items(), key=lambda item: item[1][0])
        for layer_name, values in sorted_loss_items:
            outputs_cos, outputs_snr = values
            self.mix_prec.logger.print_info("Layer: {}, outputs_cos: {}, outputs_snr: {}".format(
                layer_name, outputs_cos, outputs_snr))

        sorted_clusters = self.cluster_4_8(loss_dict)
        fp_layer_list = []
        for cluster in sorted_clusters:
            for op_name in cluster:
                fp_layer_list.append(op_name)
            mix_table = self.mix_prec._gen_mix_table(mix_ops=fp_layer_list,
                                                     qtable=self.qtable,
                                                     high_prec_type=self.low_prec)
            mix_model = MixQuantModel(self.fp32_mlir, self.chip, None, self.low_prec,
                                      self.cali_table_name, mix_table)
            outputs_cos, outputs_snr = self.mix_prec.run_model_new(mix_model, False,
                                                                   global_compare_layers,
                                                                   layers_rate, predictions_gt)
            if 1 - outputs_cos < int8_outputs_cos * 0.95:
                for op_name in cluster:
                    fp_layer_list.remove(op_name)
                break
        self.print_log_info_4bit(fp_layer_list, int8_outputs_cos, t0, mix_mode='W4INT8')
