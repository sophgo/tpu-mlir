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
from tqdm import tqdm
from .utils import *
import gc

pymlir.set_mem_mode("force_value_mem")


class FloatOpClassifier:
    """Unified float-op set (fixed + pattern + shape + custom) and the two
    search-skip predicates derived from the lowering semantics.

    The lowering uses the *producer's* cali-table threshold at the
    float->quantized boundary (``getQuantInt8Type`` in
    ``lib/Conversion/TopToTpu/TopLowering.cpp`` reads the producer's
    ``CalibratedQuantizedType``), and a float lowering preserves that type on
    its output. Hence a float op's threshold is only dead when its output
    never reaches a quantized op; it is *not* dead just because the op itself
    is float. The two searches therefore need different skip rules.
    """

    def __init__(self, parser, qtable, fixed_float_ops=None):
        self.parser = parser
        self.qtable = qtable
        self._fixed = set(fixed_float_ops or [])
        self._reach_cache = {}

    def is_float(self, name):
        return name in self._fixed or (self.qtable is not None and self.qtable.exists(name))

    def _reaches_quantized(self, name, on_stack):
        # True iff a non-float op is reachable from `name`'s output through any
        # consumer chain. Float consumers are traversed because transparent
        # float ops (Reshape/Permute/Slice/...) carry the producer's
        # CalibratedQuantizedType forward (Forward/BackwardCalibartion).
        if name in self._reach_cache:
            return self._reach_cache[name]
        if name in on_stack:
            return False
        on_stack.add(name)
        result = False
        for consumer in self.parser.get_next_op_by_op_name(name):
            if not self.is_float(consumer):
                result = True
                break
            if self._reaches_quantized(consumer, on_stack):
                result = True
                break
        on_stack.discard(name)
        self._reach_cache[name] = result
        return result

    def skip_threshold_search(self, name):
        # Skip iff op is float AND its threshold is dead (no quantized consumer
        # reachable -> the float->quantized boundary cast never fires).
        return self.is_float(name) and not self._reaches_quantized(name, set())

    def skip_sensitive_search(self, name):
        # Skip iff op is float (pre-decided; demotion high_prec<->low_prec is
        # undefined for an op the lowering already keeps float).
        return self.is_float(name)


class SearchQtableBase:

    def __init__(self, args, selector, tune_ds, qtable=None):
        self.args = args
        self.fp32_mlir = args.mlir_file
        self.chip = args.chip
        self.cali_table_name = args.calibration_table
        self.module = pymlir.module()
        self.module.load(args.mlir_file)
        self.module.set_progress_silent(True)
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
        # q_group_size for dynamic-quant (F4/F8) weight/activation grouping;
        # 0 leaves int8/int4 (non-DYN) paths untouched.
        self.q_group_size = getattr(self.args, 'q_group_size', 0)
        # Default classifier (fixed-float empty); run() upgrades it once the
        # search baseline model is available. Pattern/shape/custom float are
        # covered via the qtable from the start.
        self.classifier = FloatOpClassifier(self.parser, self.qtable, [])

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
        min_similarity = float('inf')
        min_similarity_op_type = None
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
            mix_model.module.set_progress_silent(True)
            similarity = self.mix_prec.run_model(mix_model,
                                                 False,
                                                 global_compare_layers,
                                                 layers_rate,
                                                 predictions_gt,
                                                 sample_num=1,
                                                 loss_methods=['cos'])
            self.mix_prec.logger.print_info(f"op_type : {op_type}, similarity : {similarity}")
            if similarity < min_similarity:
                min_similarity = similarity
                min_similarity_op_type = op_type
            if similarity < float_outputs_cos * cos_threshold:
                sensitive_op_type.append(op_type)
        if len(sensitive_op_type) == 0:
            sensitive_op_type.append(min_similarity_op_type)
        self.mix_prec.logger.print_info(
            f"sensitive_op_type : {sensitive_op_type}, please pay attention to these types of operations"
        )
        return sensitive_op_type

    def compare_loss(self, layer_name, loss_dict, outputs_cos, outputs_snr):
        existing_cos = loss_dict[layer_name][0]
        existing_snr = loss_dict[layer_name][1]

        if outputs_cos < existing_cos:
            existing_cos = outputs_cos
        if outputs_snr < existing_snr:
            existing_snr = outputs_snr

        loss_dict[layer_name] = [existing_cos, existing_snr]

    def search_best_threshold_per_op(self,
                                     layer_names,
                                     quantize_method_list,
                                     layer_th_dicts,
                                     global_compare_layers,
                                     layers_rate,
                                     predictions_gt,
                                     low_prec=None,
                                     high_prec=None,
                                     chip=None,
                                     set_th_fn=None,
                                     isolate=True,
                                     high_prec_type=None,
                                     cluster=None,
                                     desc='search_best_th'):
        """Per-op best-threshold search shared by the int8 ``search_sensitve_layer`` and
        the int4/w4a8 stages.

        For each op it tries every method's threshold (read from ``layer_th_dicts``),
        benchmarks the model, and keeps the best (method, th) per op, committing it via
        ``set_th_fn``. When ``isolate`` is True the current op is quantized while the
        rest are kept at ``high_prec_type`` (the int8 routine); when False the full
        model (``_gen_mix_table([])``) is used for every op (int4 stage 2 / w4a8 stage 1).
        """
        if low_prec is None:
            low_prec = self.low_prec
        if high_prec is None:
            high_prec = self.high_prec
        if chip is None:
            chip = self.chip
        if set_th_fn is None:
            set_th_fn = self.set_layer_new_th
        if cluster is None:
            cluster = self.args.cluster
        if not layer_names:
            self.mix_prec.logger.print_info(
                "Layer names are empty. All operators skipped in search phase.")
            sys.exit(1)
        num_quantize_method = len(quantize_method_list)
        total = len(layer_names) * num_quantize_method
        pbar = tqdm(total=total, desc=desc)
        loss_dict = collections.defaultdict(list)
        fp_layer_list = list(layer_names)
        modified_layers = {}
        last_tried_method = quantize_method_list[0]
        sensitive_layer_analysis_dict = {}
        new_cali_table_name = self.cali_table_name
        for layer_idx, layer_name in enumerate(layer_names):
            if self.classifier.skip_threshold_search(layer_name):
                continue
            layer_type = self.parser.get_op_type_by_op_name(layer_name)
            pbar.set_postfix_str(f"{layer_idx}/{len(layer_names)} {layer_name}")
            self.mix_prec.logger.print_info("start to handle layer: {}, type: {}".format(
                layer_name, layer_type))
            fp_layer_list.remove(layer_name)
            if isolate:
                mix_table = self.mix_prec._gen_mix_table(fp_layer_list,
                                                         self.qtable,
                                                         high_prec_type=high_prec_type)
            else:
                mix_table = self.mix_prec._gen_mix_table([],
                                                         self.qtable,
                                                         high_prec_type=high_prec_type)
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
                    new_th = layer_th_dicts[last_tried_method][layer_name][1]
                    new_cali_table_name = set_th_fn(layer_name, new_th)
                    last_tried_method = method
                    self.mix_prec.logger.print_info(
                        "adjust layer {} th, with method {}, and threshlod {}".format(
                            layer_name, method, new_th))
                    mixmodel = MixQuantModel(self.fp32_mlir,
                                             chip,
                                             low_prec,
                                             high_prec,
                                             new_cali_table_name,
                                             mix_table,
                                             using_cuda=True)
                    mixmodel.module.set_progress_silent(True)
                    if not cluster:
                        outputs_cos = 1 - self.mix_prec.run_model(
                            mixmodel, False, global_compare_layers, layers_rate, predictions_gt)
                    else:
                        result = self.mix_prec.run_model(mixmodel,
                                                         False,
                                                         global_compare_layers,
                                                         layers_rate,
                                                         predictions_gt,
                                                         loss_methods=['cos', 'snr'])
                        outputs_cos = 1 - result['cos']
                        outputs_snr = result['snr']
                        loss_dict[layer_name].extend([outputs_cos, outputs_snr])
                    self.mix_prec.logger.print_info("outputs_cos_los = {}".format(outputs_cos))
                    pbar.set_postfix_str(
                        f"{layer_idx}/{len(layer_names)} {layer_name} | m={method} cos={1-outputs_cos:.4f}"
                    )
                    mixmodel.clean()
                    del mixmodel
                    pbar.update(1)
                elif modified_layers[layer_name][0] < num_quantize_method:
                    method_idx = modified_layers[layer_name][0]
                    method = quantize_method_list[method_idx]
                    if outputs_cos < modified_layers[layer_name][1]:
                        modified_layers[layer_name][1] = outputs_cos
                        modified_layers[layer_name][2] = layer_th_dicts[last_tried_method][
                            layer_name][1]
                        modified_layers[layer_name][3] = last_tried_method
                    new_th = layer_th_dicts[method][layer_name][1]
                    new_cali_table_name = set_th_fn(layer_name, new_th)
                    last_tried_method = method
                    self.mix_prec.logger.print_info(
                        "adjust layer {} th, with method {}, and threshlod {}".format(
                            layer_name, method, new_th))
                    modified_layers[layer_name][0] += 1
                    mixmodel = MixQuantModel(self.fp32_mlir,
                                             chip,
                                             low_prec,
                                             high_prec,
                                             new_cali_table_name,
                                             mix_table,
                                             using_cuda=True)
                    mixmodel.module.set_progress_silent(True)
                    if not cluster:
                        outputs_cos = 1 - self.mix_prec.run_model(
                            mixmodel, False, global_compare_layers, layers_rate, predictions_gt)
                    else:
                        result = self.mix_prec.run_model(mixmodel,
                                                         False,
                                                         global_compare_layers,
                                                         layers_rate,
                                                         predictions_gt,
                                                         loss_methods=['cos', 'snr'])
                        outputs_cos = 1 - result['cos']
                        outputs_snr = result['snr']
                        self.compare_loss(layer_name, loss_dict, outputs_cos, outputs_snr)
                    self.mix_prec.logger.print_info("outputs_cos_los = {}".format(outputs_cos))
                    pbar.set_postfix_str(
                        f"{layer_idx}/{len(layer_names)} {layer_name} | m={method} cos={1-outputs_cos:.4f}"
                    )
                    mixmodel.clean()
                    del mixmodel
                    pbar.update(1)
                elif modified_layers[layer_name][0] == num_quantize_method:
                    if outputs_cos < modified_layers[layer_name][1]:
                        modified_layers[layer_name][1] = outputs_cos
                        modified_layers[layer_name][2] = layer_th_dicts[last_tried_method][
                            layer_name][1]
                        modified_layers[layer_name][3] = last_tried_method
                    best_th = modified_layers[layer_name][2]
                    modified_layers[layer_name][0] += 1
                    new_cali_table_name = set_th_fn(layer_name, best_th)
                    self.mix_prec.logger.print_info(
                        "layer {}, layer type is {}, best_th = {}, best_method = {}, best_cos_loss = {}"
                        .format(layer_name, layer_type, best_th, modified_layers[layer_name][3],
                                modified_layers[layer_name][1]))
                    # Float ops still pass through here when their threshold is
                    # live (feeds a quantized op); their threshold gets tuned
                    # above, but they are already float, so exclude them from
                    # the demotion ranking to avoid wasting a float-layer slot.
                    if not self.classifier.is_float(layer_name):
                        sensitive_layer_analysis_dict[layer_name] = [
                            modified_layers[layer_name][1], layer_type
                        ]
                    ret = True
            fp_layer_list.append(layer_name)
            gc.collect()
        pbar.close()
        return sensitive_layer_analysis_dict, new_cali_table_name, loss_dict

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

    def print_log_info(self, fp_layer_list, all_int8_cos, outputs_cos, t0, mix_mode=None):
        self.mix_prec.logger.print_info('>>>run result:')
        if self.qtable is not None:
            qtable = copy.deepcopy(self.qtable)
        else:
            qtable = QuantizeTable()
        if mix_mode is None:
            if self.args.mix_mode in ['wi8ai8_fp', 'wf8af8_fp']:
                fp_type = FLOAT_MAP[
                    self.args.chip] if self.args.fp_type == 'auto' else self.args.fp_type
            elif self.args.mix_mode in ['wi4ai8_wi8ai8']:
                fp_type = self.low_prec
            else:
                fp_type = self.high_prec
        else:
            fp_type = mix_mode
        qtable.append_custom(fp_layer_list, [fp_type] * len(fp_layer_list))
        qtable.dump(self.mix_prec.quantize_table)
        self.mix_prec.logger.print_info(
            "float layer number: {}, mix model outputs_cos: {:.6f}".format(
                len(fp_layer_list), outputs_cos))
        self.mix_prec.logger.print_info(f'int8 outputs_cos:{all_int8_cos:.6f} old')
        self.mix_prec.logger.print_info(f"mix model outputs_cos:{outputs_cos:.6f}")
        self.mix_prec.logger.print_info("Output mix quantization table to {}".format(
            self.mix_prec.quantize_table))
        self.mix_prec.logger.print_info("total time:{}".format(time.time() - t0))

    def search_sensitive_layer(self, layer_names, global_compare_layers, layers_rate,
                               predictions_gt):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


class SearchQtable(SearchQtableBase):

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

    def set_layer_new_th(self, layer_name, value):
        threshold = float(value)
        self.cali_table.thresholds_map[layer_name][0] = threshold
        new_cali_table_name = "new_cali_table.txt"
        self.cali_table.update_to(new_cali_table_name, layer_name, threshold)
        return new_cali_table_name

    def search_sensitve_layer(self, layer_names, quantize_method_list, layer_th_dicts,
                              global_compare_layers, layers_rate, predictions_gt):
        return self.search_best_threshold_per_op(layer_names,
                                                 quantize_method_list,
                                                 layer_th_dicts,
                                                 global_compare_layers,
                                                 layers_rate,
                                                 predictions_gt,
                                                 isolate=True,
                                                 desc='search_sensitve_layer')

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
        Automatically select the best eps parameter for DBSCAN, based on the
        silhouette score.

        Args:
            X (array-like): dataset, shape (n_samples, n_features)
            eps_min (float): minimum eps to search, default 0.005
            eps_max (float): maximum eps to search, default 0.01
            n_points (int): number of eps values to generate between eps_min
                and eps_max, default 100
            min_samples (int): min_samples parameter for DBSCAN, default 2

        Returns:
            best_eps (float or None): the best eps value chosen by silhouette
                score; returns None if clustering quality is poor for every eps
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
            mixmodel.module.set_progress_silent(True)
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
                    mixmodel.module.set_progress_silent(True)
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

    def run(self):
        t0 = time.time()

        #step1: generate op th dict of defined methods(KL, Max, Percentile9999, MSE)
        all_op_names = self.parser.get_op_name_list()
        all_op_names = get_no_fused_tensors(self.parser, all_op_names)
        quantize_method_list = [x.lower() for x in self.quantize_method_list]
        suffix = "_tune" if self.args.tune_num > 0 else ""
        reuse_cali_table = self.cali_table_name is not None and os.path.exists(self.cali_table_name)
        if reuse_cali_table:
            self.mix_prec.logger.print_warning(
                f'[search_qtable] {self.cali_table_name} already exists; '
                'skip per-method threshold generation, run sensitivity search against '
                'the single existing table.')
            quantize_method_list = quantize_method_list[:1]
            existing_table = CalibrationTable(self.cali_table_name)
            tmp_th_dict = {}
            for op, vals in existing_table.thresholds_map.items():
                th, mn, mx = vals
                tmp_th_dict[op] = [max(abs(mn), abs(mx)), th]
            layer_th_dicts = {quantize_method_list[0]: tmp_th_dict}
        else:
            calibrator = ActivationCalibrator(self.args,
                                              self.selector,
                                              self.tune_ds,
                                              using_cuda=True)
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
        #step2: float_model and int8_model inference
        mix_table = None if self.qtable is None else self.mix_prec._gen_mix_table([], self.qtable)
        float_model = MixQuantModel(self.fp32_mlir, None, None, "F32")
        float_model.module.set_progress_silent(True)
        int8_model = MixQuantModel(self.fp32_mlir, self.chip, self.low_prec, self.high_prec,
                                   self.cali_table_name, mix_table)
        int8_model.module.set_progress_silent(True)
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
        self.classifier = FloatOpClassifier(self.parser, self.qtable, float_ops)
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
        mixmodel.module.set_progress_silent(True)
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
            self.print_log_info(final_fp_layers, all_int8_cos, outputs_cos, t0)
        else:
            self.print_log_info(selected_fp_layers, all_int8_cos, outputs_cos, t0)
        print("success search qtable")
        return 'success'


class Int4SearchStrategy:
    name = 'int4'
    desc = 'search_sensitve_layer_int4'

    def before_search(self, searcher):
        # strip any stale #int4_op block from a previous run, then write a
        # fresh header so the search starts clean.
        searcher._strip_int4_op_block()
        with open(searcher.cali_table_name, 'a') as file:
            file.write("\n#int4_op\n")

    def before_trial(self, searcher, op_name):
        with open(searcher.cali_table_name, 'a') as file:
            file.write(f"{op_name}\n")

    def rollback_trial(self, searcher, op_name):
        searcher.remove_lines_from_file(searcher.cali_table_name, [op_name])

    def qtable_keep_mode(self, searcher):
        return searcher.high_prec

    def make_trial_model(self, searcher, mix_table):
        return MixQuantModel(searcher.fp32_mlir, searcher.chip, searcher.low_prec,
                             searcher.high_prec, searcher.cali_table_name, mix_table)


class W4A8SearchStrategy:
    name = 'w4a8'
    desc = 'search_sensitve_layer_w4a8'

    def before_search(self, searcher):
        pass

    def before_trial(self, searcher, op_name):
        pass

    def rollback_trial(self, searcher, op_name):
        pass

    def qtable_keep_mode(self, searcher):
        return searcher.high_prec

    def make_trial_model(self, searcher, mix_table):
        return MixQuantModel(searcher.fp32_mlir, searcher.chip, None, searcher.low_prec,
                             searcher.cali_table_name, mix_table)


class F4F8SearchStrategy:
    name = 'f4f8'
    desc = 'search_sensitve_layer_f4f8'

    # F4/F8 (DYN): activations stay F16/BF16; weight quant to F4/F8 happens in
    # the lowering via per-op dq_type read from the qtable. No #int4_op marking,
    # no cali table. Base mode = low_prec (F4); ops in the mix_table are kept at
    # high_prec (F8) via qtable_keep_mode.

    def before_search(self, searcher):
        pass

    def before_trial(self, searcher, op_name):
        pass

    def rollback_trial(self, searcher, op_name):
        pass

    def qtable_keep_mode(self, searcher):
        return searcher.high_prec

    def make_trial_model(self, searcher, mix_table):
        return MixQuantModel(searcher.fp32_mlir,
                             searcher.chip,
                             searcher.low_prec,
                             searcher.high_prec,
                             None,
                             mix_table,
                             q_group_size=searcher.q_group_size)


class SearchQtable4Bit(SearchQtableBase):

    SEARCH_STRATEGIES = {
        'wi4ai4_wi8ai8': Int4SearchStrategy(),
        'wi4ai8_wi8ai8': W4A8SearchStrategy(),
        'wf4af16dyn_wf8af16dyn': F4F8SearchStrategy(),
        'wf4abf16dyn_wf8abf16dyn': F4F8SearchStrategy(),
    }

    def _get_search_strategy(self):
        return self.SEARCH_STRATEGIES[self.args.mix_mode]

    def search_sensitive_layer_low_prec(self, layer_names, global_compare_layers, layers_rate,
                                        predictions_gt, strategy):
        loss_dict = collections.defaultdict(list)
        keep_int8_layers = copy.deepcopy(layer_names)
        strategy.before_search(self)
        pbar = tqdm(total=len(layer_names), desc=strategy.desc)
        for layer_idx, layer_name in enumerate(layer_names):
            if self.classifier.skip_sensitive_search(layer_name):
                continue
            layer_type = self.parser.get_op_type_by_op_name(layer_name)
            pbar.set_postfix_str(f"{layer_idx}/{len(layer_names)} {layer_name}")
            self.mix_prec.logger.print_info("start to handle layer: {}, type: {}".format(
                layer_name, layer_type))
            keep_int8_layers.remove(layer_name)
            strategy.before_trial(self, layer_name)
            try:
                mix_table = self.mix_prec._gen_mix_table(
                    mix_ops=keep_int8_layers,
                    qtable=self.qtable,
                    high_prec_type=strategy.qtable_keep_mode(self))
                mix_model = strategy.make_trial_model(self, mix_table)
                mix_model.module.set_progress_silent(True)
                result = self.mix_prec.run_model(mix_model,
                                                 False,
                                                 global_compare_layers,
                                                 layers_rate,
                                                 predictions_gt,
                                                 loss_methods=['cos', 'snr'])
                outputs_cos = 1 - result['cos']
                outputs_snr = result['snr']
                if layer_name not in loss_dict:
                    loss_dict[layer_name].extend([outputs_cos, outputs_snr])
                else:
                    self.compare_loss(layer_name, loss_dict, outputs_cos, outputs_snr)
            finally:
                strategy.rollback_trial(self, layer_name)
                keep_int8_layers.append(layer_name)
            self.mix_prec.logger.print_info("layer {}, outputs_cos:{}, outputs_snr:{}".format(
                layer_name, outputs_cos, outputs_snr))
            pbar.update(1)
        pbar.close()
        return loss_dict

    def remove_lines_from_file(self, file_path, lines_to_remove):
        """
        Remove specified lines from a file.

        Args:
        file_path (str): file path
        lines_to_remove (Iterable[str]): list of contents to remove (whitespace
            is trimmed before comparison)
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

    def _sort_cluster_by_loss(self, cluster, loss_dict):
        return sorted(cluster, key=lambda op_name: loss_dict[op_name][0])

    def _strip_int4_op_block(self):
        """Remove any existing ``#int4_op`` block (header + op names) from the
        cali table in place, so a re-run doesn't carry stale int4-op markings
        from a previous search. Keeps ``#int4_th`` (the thresholds). Safe for
        all modes - no-op if the table has no ``#int4_op``."""
        if self.cali_table_name is None or not os.path.exists(self.cali_table_name):
            return
        skip = False
        out = []
        with open(self.cali_table_name, 'r') as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith('#int4_op'):
                    skip = True
                    continue
                if stripped.startswith('#'):
                    skip = False
                    out.append(line)
                    continue
                if skip:
                    continue
                out.append(line)
        with open(self.cali_table_name, 'w') as f:
            f.writelines(out)

    def _ensure_cali_table(self):
        if self.cali_table_name is None:
            self.cali_table_name = Path(self.fp32_mlir).stem + '_cali_table'
            self.args.calibration_table = self.cali_table_name
        if os.path.exists(self.cali_table_name):
            self._table_pre_existed = True
            # a pre-existing table may carry a stale #int4_op block from a
            # previous search run; strip it so the search starts clean.
            self._strip_int4_op_block()
            self.mix_prec.logger.print_warning(
                f'[search_best_th] {self.cali_table_name} already exists; '
                'skip best-threshold search and use it as-is.')
            return
        self._table_pre_existed = False
        calibrator_args = copy.deepcopy(self.args)
        if self.args.mix_mode == 'wi4ai4_wi8ai8':
            calibrator_args.debug_cmd = copy.deepcopy(self.args.debug_cmd)
            calibrator_args.debug_cmd['int4'] = None
        calibrator = ActivationCalibrator(calibrator_args, self.selector, self.tune_ds)
        calibrator.calibration_method = [self.args.cali_method[0]]
        calibrator.run()

    def _collect_candidate_ops(self, is_w4a8):
        all_op_names = self.parser.get_op_name_list()
        search_op_type = ['top.Conv', 'top.MatMul']
        if self.args.mix_mode in ('wf4af16dyn_wf8af16dyn', 'wf4abf16dyn_wf8abf16dyn'):
            # F4/F8 DYN path is MatMul-only.
            return [
                n for n in all_op_names if self.parser.get_op_type_by_op_name(n) == 'top.MatMul'
            ]
        if is_w4a8:
            weight_file = self.parser.module_weight_file
            weights = np.load(weight_file)
            layer_names = [
                layer.name for layer in self.parser.ops
                if (self.parser.get_op_type_by_op_name(layer.name) in search_op_type and (
                    layer.opds[1] in weights if self.parser.get_op_type_by_op_name(layer.name) ==
                    'top.MatMul' else not self._is_depthwise_conv(layer)))
            ]
        else:
            layer_names = [
                layer for layer in all_op_names
                if self.parser.get_op_type_by_op_name(layer) in search_op_type
            ]
        return layer_names

    def _stage1_methods(self):
        methods = [m.lower() for m in self.args.quantize_method_list]
        if not methods or methods == ['mse']:
            methods = ['kl', 'max', 'percentile9999', 'mse']
        return methods

    def _stage2_methods(self):
        qlist = [m.lower() for m in self.args.quantize_method_list]
        sel = [m for m in ['kl', 'mse'] if m in qlist]
        if not sel:
            sel = ['kl', 'mse']
        return sel

    def _parse_int4_th(self, table_path):
        result = {}
        in_int4 = False
        if not os.path.exists(table_path):
            return result
        with open(table_path, 'r') as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith('#int4_th'):
                    in_int4 = True
                    continue
                if stripped.startswith('#'):
                    in_int4 = False
                    continue
                fields = stripped.split()
                if in_int4 and len(fields) >= 4:
                    th = float(fields[1])
                    result[fields[0]] = [th, th]
        return result

    def _set_layer_th_filebased(self, layer_name, new_th, update_int4):
        """Update the 8-bit row (``update_int4=False``) or the ``#int4_th`` row
        (``update_int4=True``) for ``layer_name`` in the cali table, preserving all
        other sections. Accumulates across calls via the ``new_cali_table.txt``
        scratch file (falls back to ``self.cali_table_name`` on first call)."""
        src = 'new_cali_table.txt' if os.path.exists('new_cali_table.txt') else self.cali_table_name
        in_int4 = False
        out = []
        with open(src, 'r') as fin:
            for line in fin:
                stripped = line.strip()
                if stripped.startswith('#int4_th'):
                    in_int4 = True
                    out.append(line)
                    continue
                if stripped.startswith('#'):
                    in_int4 = False
                    out.append(line)
                    continue
                fields = stripped.split()
                if len(fields) >= 4 and fields[0] == layer_name and \
                   ((update_int4 and in_int4) or (not update_int4 and not in_int4)):
                    th = float(new_th)
                    if update_int4:
                        mn, mx = -th * 8.0 / 7.0, th
                    else:
                        mn, mx = float(fields[2]), float(fields[3])
                    out.append("{} {:.7f} {:.7f} {:.7f}\n".format(layer_name, th, mn, mx))
                else:
                    out.append(line)
        with open('new_cali_table.txt', 'w') as fout:
            fout.writelines(out)
        return 'new_cali_table.txt'

    def _commit_scratch_table(self):
        if os.path.exists('new_cali_table.txt'):
            import shutil
            shutil.copyfile('new_cali_table.txt', self.cali_table_name)
            os.remove('new_cali_table.txt')

    def _gen_per_method_thresholds(self, ops, methods, is_int4_mode):
        calibrator_args = copy.deepcopy(self.args)
        calibrator_args.calibration_table = self.cali_table_name
        if is_int4_mode:
            calibrator_args.debug_cmd = copy.deepcopy(self.args.debug_cmd)
            calibrator_args.debug_cmd['int4'] = None
            calibrator_args.tune_num = 0
        calibrator = ActivationCalibrator(calibrator_args,
                                          self.selector,
                                          self.tune_ds,
                                          using_cuda=True)
        calibrator.calibration_method = methods
        layer_th_dicts = calibrator.gen_multiple_thresholds(ops, methods)
        del calibrator
        suffix = '' if (is_int4_mode or calibrator_args.tune_num <= 0) else '_tune'
        layer_th_dicts4 = {}
        if is_int4_mode:
            for method in methods:
                path = f'{self.cali_table_name}_{method}{suffix}'
                if os.path.exists(path):
                    layer_th_dicts4[method] = self._parse_int4_th(path)
        return layer_th_dicts, layer_th_dicts4

    def _run_best_threshold_stages(self, candidate_ops, global_compare_layers, layers_rate,
                                   predictions_gt, mix_table):
        is_int4_mode = self.args.mix_mode == 'wi4ai4_wi8ai8'
        is_w4a8 = self.args.mix_mode == 'wi4ai8_wi8ai8'

        if is_w4a8:
            # w4a8: only Stage 1 — tune INT8 (8-bit activation) thresholds in the full
            # W4A8 model. There is no #int4_th section (weight-only int4).
            methods1 = self._stage1_methods()
            layer_th_dicts, _ = self._gen_per_method_thresholds(candidate_ops,
                                                                methods1,
                                                                is_int4_mode=False)
            if not layer_th_dicts:
                self.mix_prec.logger.print_warning(
                    '[search_best_th] stage1 produced no method tables, skip.')
            else:
                self.search_best_threshold_per_op(
                    candidate_ops,
                    methods1,
                    layer_th_dicts,
                    global_compare_layers,
                    layers_rate,
                    predictions_gt,
                    low_prec=None,
                    high_prec=self.low_prec,
                    chip=self.chip,
                    set_th_fn=lambda l, th: self._set_layer_th_filebased(l, th, False),
                    isolate=False,
                    high_prec_type=None,
                    cluster=False,
                    desc='search_best_th_stage1_w4a8')
                self._commit_scratch_table()
        elif is_int4_mode:
            # wi4ai4: drop Stage 1. The 8-bit ths for Conv/MatMul don't affect the
            # int4 model (those ops use #int4_th), so tuning them only risked
            # regressing the int8 baseline reference. Only Stage 2: tune #int4_th in
            # the full INT4 model on bm1688. The default cali_method is appended to
            # the candidate methods so the baseline th is always an option (can't
            # regress below the default).
            methods2 = self._stage2_methods()
            default_method = self.args.cali_method[0].lower()
            if default_method not in methods2:
                methods2 = methods2 + [default_method]
            _, layer_th_dicts4 = self._gen_per_method_thresholds(candidate_ops,
                                                                 methods2,
                                                                 is_int4_mode=True)
            # Fallback: if the default method's #int4_th wasn't generated, parse the
            # baseline table (which carries it from _ensure_cali_table).
            if default_method not in layer_th_dicts4:
                layer_th_dicts4[default_method] = self._parse_int4_th(self.cali_table_name)
            ld4_sel = {m: layer_th_dicts4[m] for m in methods2 if m in layer_th_dicts4}
            if not ld4_sel:
                self.mix_prec.logger.print_warning(
                    '[search_best_th] stage2 produced no int4 method tables, skip.')
            else:
                self.search_best_threshold_per_op(
                    candidate_ops,
                    methods2,
                    ld4_sel,
                    global_compare_layers,
                    layers_rate,
                    predictions_gt,
                    low_prec=self.low_prec,
                    high_prec=self.high_prec,
                    chip=self.chip,
                    set_th_fn=lambda l, th: self._set_layer_th_filebased(l, th, True),
                    isolate=False,
                    cluster=False,
                    desc='search_best_th_stage2_int4')
                self._commit_scratch_table()

    def _gen_int8_cali_table(self):
        int8_cali_table = self.cali_table_name + '.int8'
        skip_int4_block = False
        with open(self.cali_table_name, 'r') as src, open(int8_cali_table, 'w') as dst:
            for line in src:
                stripped = line.strip()
                if stripped.startswith('#'):
                    if stripped.startswith('#int4_th') or stripped.startswith('#int4_op'):
                        skip_int4_block = True
                        continue
                    skip_int4_block = False
                if not skip_int4_block:
                    dst.write(line)
        return int8_cali_table

    def _is_depthwise_conv(self, layer):
        if layer.type != 'top.Conv':
            return False
        group = int(layer.attrs.get('group', '1').split(':')[0])
        if group <= 1:
            return False
        in_c = layer.in_shapes[0][1]
        out_c = layer.shape[1]
        return in_c == out_c == group

    def run(self):
        is_int4_mode = self.args.mix_mode == 'wi4ai4_wi8ai8'
        is_w4a8 = self.args.mix_mode == 'wi4ai8_wi8ai8'
        is_f4f8 = self.args.mix_mode in ('wf4af16dyn_wf8af16dyn', 'wf4abf16dyn_wf8abf16dyn')
        if not (is_int4_mode or is_w4a8 or is_f4f8):
            raise RuntimeError(f'unsupported mix_mode for SearchQtable4Bit: {self.args.mix_mode}')
        if not is_f4f8:
            # int4/w4a8 need a cali table (#int4_th / 8-bit thresholds). F4/F8 (DYN)
            # keeps activations F16/BF16 and quantizes weights in the lowering, so no
            # cali table / threshold computation is needed.
            self._ensure_cali_table()
        t0 = time.time()
        layer_cos_list, predictions_gt = [], []
        fp_type = 'F32' if (
            self.args.fp_type == 'auto'
            and 'F32' in chip_support_mix_fp_type[self.args.chip]) else FLOAT_MAP[self.args.chip]
        float_model = MixQuantModel(self.fp32_mlir, self.chip, None, fp_type)
        float_model.module.set_progress_silent(True)
        global_compare_layers, layers_rate, _ = self.mix_prec.extract_global_layers()
        _ = self.mix_prec.run_model(float_model, True, global_compare_layers, layers_rate,
                                    predictions_gt)
        mix_table = None if self.qtable is None else self.mix_prec._gen_mix_table([], self.qtable)
        if not is_f4f8 and not getattr(self, '_table_pre_existed', False):
            candidate_ops = self._collect_candidate_ops(is_w4a8)
            self._run_best_threshold_stages(candidate_ops, global_compare_layers, layers_rate,
                                            predictions_gt, mix_table)
        # Baseline = int8 (int4/w4a8) or all-high/F8 (F4/F8). target = baseline cos.
        if is_f4f8:
            candidate_ops = self._collect_candidate_ops(is_w4a8)
            mix_all_high = self.mix_prec._gen_mix_table(candidate_ops,
                                                        self.qtable,
                                                        high_prec_type=self.high_prec)
            baseline_model = MixQuantModel(self.fp32_mlir,
                                           self.chip,
                                           self.low_prec,
                                           self.high_prec,
                                           None,
                                           mix_all_high,
                                           q_group_size=self.q_group_size)
            baseline_model.module.set_progress_silent(True)
            baseline_outputs_cos = self.mix_prec.run_model(baseline_model, False,
                                                           global_compare_layers, layers_rate,
                                                           predictions_gt)
            target_outputs_cos = baseline_outputs_cos
            self.mix_prec.logger.print_info(f'current f8 cos:{target_outputs_cos}')
            baseline_model.clean()
        else:
            # _run_best_threshold_stages above called _gen_mix_table many times,
            # overwriting tmp_mix_table.txt; re-generate the default (qtable fp)
            # mix_table for the int4/w4a8 baseline and the informational models
            # below (w4a8/int4), which all read this same mix_table.
            mix_table = None if self.qtable is None else self.mix_prec._gen_mix_table([],
                                                                                      self.qtable)
            int8_low_prec, int8_high_prec = get_mix_prec(self.args.chip, 'wi8ai8_fp',
                                                         self.args.fp_type)
            int8_cali_table = self._gen_int8_cali_table()
            baseline_model = MixQuantModel(self.fp32_mlir, self.chip, int8_low_prec, int8_high_prec,
                                           int8_cali_table, mix_table)
            baseline_model.module.set_progress_silent(True)
            baseline_outputs_cos = self.mix_prec.run_model(baseline_model, False,
                                                           global_compare_layers, layers_rate,
                                                           predictions_gt)
            target_outputs_cos = baseline_outputs_cos
            self.mix_prec.logger.print_info(f'current int8 cos:{target_outputs_cos}')
            # Discover fixed-float Conv/MatMul on the int8 baseline (I8/U8
            # dtypes are reliable here; f8 baselines are not, see get_fixed_float_layers)
            # so the sensitive search can skip them. The best-th stages above
            # already ran with the __init__ default (fixed empty); only the
            # sensitive search needs the upgraded classifier.
            fixed_float_ops = self.mix_prec.get_fixed_float_layers(baseline_model,
                                                                   global_compare_layers,
                                                                   layers_rate, predictions_gt)
            self.classifier = FloatOpClassifier(self.parser, self.qtable, fixed_float_ops)
            baseline_model.clean()
            if os.path.exists(int8_cali_table):
                os.remove(int8_cali_table)
        # Informational all-low model.
        if is_f4f8:
            # mix_all_high above overwrote tmp_mix_table.txt; the all-F4 model
            # needs the default mix (no MatMul marked -> all at base F4), so
            # re-generate it into a fresh file.
            f4_mix_table = None if self.qtable is None else self.mix_prec._gen_mix_table(
                [], self.qtable)
            f4_model = MixQuantModel(self.fp32_mlir,
                                     self.chip,
                                     self.low_prec,
                                     self.high_prec,
                                     None,
                                     f4_mix_table,
                                     q_group_size=self.q_group_size)
            f4_model.module.set_progress_silent(True)
            f4_outputs_cos = self.mix_prec.run_model(f4_model, False, global_compare_layers,
                                                     layers_rate, predictions_gt)
            self.mix_prec.logger.print_info(f'current f4 cos:{f4_outputs_cos}')
        elif is_w4a8:
            w4a8_model = MixQuantModel(self.fp32_mlir, self.chip, None, self.low_prec,
                                       self.cali_table_name, mix_table)
            w4a8_model.module.set_progress_silent(True)
            w4a8_outputs_cos = self.mix_prec.run_model(w4a8_model, False, global_compare_layers,
                                                       layers_rate, predictions_gt)
            self.mix_prec.logger.print_info(f'current w4a8 cos:{w4a8_outputs_cos}')
        else:
            int4_model = MixQuantModel(self.fp32_mlir, self.chip, self.low_prec, self.high_prec,
                                       self.cali_table_name, mix_table)
            int4_model.module.set_progress_silent(True)
            int4_outputs_cos = self.mix_prec.run_model(int4_model, False, global_compare_layers,
                                                       layers_rate, predictions_gt)
            self.mix_prec.logger.print_info(f'current int4 cos:{int4_outputs_cos}')
            # int4_model.clean()

        all_op_names = self.parser.get_op_name_list()
        strategy = self._get_search_strategy()
        layer_names = self._collect_candidate_ops(is_w4a8)
        loss_dict = self.search_sensitive_layer_low_prec(layer_names, global_compare_layers,
                                                         layers_rate, predictions_gt, strategy)

        sorted_loss_items = sorted(loss_dict.items(), key=lambda item: item[1][0])
        for layer_name, values in sorted_loss_items:
            outputs_cos, outputs_snr = values
            self.mix_prec.logger.print_info("Layer: {}, outputs_cos: {}, outputs_snr: {}".format(
                layer_name, outputs_cos, outputs_snr))

        sorted_clusters = self.cluster_4_8(loss_dict)
        print(f'sorted_clusters: {sorted_clusters}')
        keep_int8_layers = copy.deepcopy(layer_names)
        accepted_outputs_cos = target_outputs_cos
        target_cos_threshold = target_outputs_cos * self.args.expected_cos
        try_cluster_one_by_one = 'cluster_one_by_one' in self.debug_cmd
        for cluster in sorted_clusters:
            for op_name in cluster:
                strategy.before_trial(self, op_name)
                keep_int8_layers.remove(op_name)
            high_prec_type = strategy.qtable_keep_mode(self)
            mix_table = self.mix_prec._gen_mix_table(mix_ops=keep_int8_layers,
                                                     qtable=self.qtable,
                                                     high_prec_type=high_prec_type)
            mix_model = strategy.make_trial_model(self, mix_table)
            mix_model.module.set_progress_silent(True)
            result = self.mix_prec.run_model(mix_model,
                                             False,
                                             global_compare_layers,
                                             layers_rate,
                                             predictions_gt,
                                             loss_methods=['cos', 'snr'])
            outputs_snr = result['snr']
            trial_outputs_cos = result['cos']
            print(
                f'Cluster: {cluster}, outputs_cos: {trial_outputs_cos}, outputs_snr: {outputs_snr} target_output_cos {target_outputs_cos}'
            )
            if trial_outputs_cos < target_cos_threshold:
                for op_name in cluster:
                    strategy.rollback_trial(self, op_name)
                    keep_int8_layers.append(op_name)
                if try_cluster_one_by_one:
                    sorted_cluster = self._sort_cluster_by_loss(cluster, loss_dict)
                    self.mix_prec.logger.print_info(
                        f'Cluster {cluster} failed, try ops one by one: {sorted_cluster}')
                    for op_name in sorted_cluster:
                        strategy.before_trial(self, op_name)
                        keep_int8_layers.remove(op_name)
                        mix_table = self.mix_prec._gen_mix_table(mix_ops=keep_int8_layers,
                                                                 qtable=self.qtable,
                                                                 high_prec_type=high_prec_type)
                        mix_model = strategy.make_trial_model(self, mix_table)
                        mix_model.module.set_progress_silent(True)
                        result = self.mix_prec.run_model(mix_model,
                                                         False,
                                                         global_compare_layers,
                                                         layers_rate,
                                                         predictions_gt,
                                                         loss_methods=['cos', 'snr'])
                        single_outputs_snr = result['snr']
                        single_outputs_cos = result['cos']
                        print(
                            f'Layer: {op_name}, outputs_cos: {single_outputs_cos}, outputs_snr: {single_outputs_snr} target_output_cos {target_outputs_cos}'
                        )
                        if single_outputs_cos < target_cos_threshold:
                            strategy.rollback_trial(self, op_name)
                            keep_int8_layers.append(op_name)
                            continue
                        accepted_outputs_cos = single_outputs_cos
                break
            accepted_outputs_cos = trial_outputs_cos
        self.mix_prec.enable_print()
        keep_int8_num = len(keep_int8_layers)
        mixed_num = len(layer_names) - keep_int8_num
        if is_f4f8:
            self.mix_prec.logger.print_info('>>>final f4/f8 search statistics:')
            self.mix_prec.logger.print_info(
                f'cos change vs f8:{accepted_outputs_cos - target_outputs_cos:.6f}')
            self.mix_prec.logger.print_info(f'final F8 keep layers:{keep_int8_num}')
            self.mix_prec.logger.print_info(f'final F4 layers:{mixed_num}')
            self.print_log_info(keep_int8_layers,
                                target_outputs_cos,
                                accepted_outputs_cos,
                                t0,
                                mix_mode=self.high_prec)
        else:
            self.mix_prec.logger.print_info('>>>final int4/w4a8 search statistics:')
            self.mix_prec.logger.print_info(
                f'cos change vs int8:{accepted_outputs_cos - target_outputs_cos:.6f}')
            self.mix_prec.logger.print_info(f'final INT8 keep layers:{keep_int8_num}')
            if is_w4a8:
                self.mix_prec.logger.print_info(f'w4a8 outputs_cos:{w4a8_outputs_cos:.6f} old')
                self.mix_prec.logger.print_info(f'final W4A8 layers:{mixed_num}')
            else:
                self.mix_prec.logger.print_info(f'int4 outputs_cos:{int4_outputs_cos:.6f} old')
                self.mix_prec.logger.print_info(f'final INT4 layers:{mixed_num}')
            self.print_log_info(keep_int8_layers,
                                target_outputs_cos,
                                accepted_outputs_cos,
                                t0,
                                mix_mode='INT8')


class SearchQtableFast(SearchQtableBase):

    def search_sensitive_layer(self, layer_names, global_compare_layers, layers_rate,
                               predictions_gt):
        loss_dict = collections.defaultdict(list)
        fp_layer_list = copy.deepcopy(self.sensitive_layer)
        fp_layer_list += [layer for layer in layer_names if layer not in self.sensitive_layer]
        pbar = tqdm(total=len(layer_names),
                    desc=f"round_{self.iteration_count} layer_search",
                    position=1,
                    leave=True)
        for layer_idx, layer_name in enumerate(layer_names):
            fp_layer_list.remove(layer_name)
            layer_type = self.parser.get_op_type_by_op_name(layer_name)
            pbar.set_postfix_str(f"{layer_idx}/{len(layer_names)} {layer_name}")
            self.mix_prec.logger.print_info("start to handle layer: {}, type: {}".format(
                layer_name, layer_type))
            mix_table = self.mix_prec._gen_mix_table(fp_layer_list, self.qtable)
            int8_mix_model = MixQuantModel(self.fp32_mlir, self.chip, self.low_prec, self.high_prec,
                                           self.cali_table_name, mix_table)
            result = self.mix_prec.run_model(int8_mix_model,
                                             False,
                                             global_compare_layers,
                                             layers_rate,
                                             predictions_gt,
                                             sample_num=1,
                                             loss_methods=['cos', 'snr'])
            outputs_cos = 1 - result['cos']
            outputs_snr = result['snr']
            if layer_name not in loss_dict:
                loss_dict[layer_name].extend([outputs_cos, outputs_snr])
            else:
                self.compare_loss(layer_name, loss_dict, outputs_cos, outputs_snr)
            self.mix_prec.logger.print_info("layer {}, outputs_cos:{}, outputs_snr:{}".format(
                layer_name, outputs_cos, outputs_snr))
            fp_layer_list.append(layer_name)
            pbar.update(1)
        pbar.close()
        return loss_dict

    def run(self):
        t0 = time.time()
        if self.cali_table_name is None:
            self.cali_table_name = Path(self.fp32_mlir).stem + '_cali_table'
            self.args.calibration_table = self.cali_table_name
        if not os.path.exists(self.cali_table_name):
            calibrator = ActivationCalibrator(self.args, self.selector, self.tune_ds)
            calibrator.calibration_method = [self.args.cali_method[0]]
            calibrator.run()
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
        float_ops = self.mix_prec.get_fixed_float_layers(int8_model, global_compare_layers,
                                                         layers_rate, predictions_gt)
        self.classifier = FloatOpClassifier(self.parser, self.qtable, float_ops)
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
        layer_names = [
            layer for layer in layer_names if not self.classifier.skip_sensitive_search(layer)
        ]
        self.mix_prec.logger.print_info("all search layer number: {}".format(len(layer_names)))

        self.sensitive_layer = []
        cos_sim = int8_outputs_cos
        self.iteration_count = 0
        eps = 0
        max_iterations = self.args.inference_num
        outer_pbar = tqdm(total=max_iterations,
                          desc="fast_search",
                          bar_format='{desc}: {n_fmt}/{total_fmt} rounds | cos={postfix}',
                          position=0,
                          leave=True)
        outer_pbar.set_postfix_str(f"{int8_outputs_cos:.4f}")
        while int8_outputs_cos < self.args.expected_cos:
            self.iteration_count += 1
            loss_dict = self.search_sensitive_layer(layer_names, global_compare_layers, layers_rate,
                                                    predictions_gt)

            keys = list(loss_dict.keys())
            sorted_by_first = sorted(keys, key=lambda k: loss_dict[k][0])
            rank_first = {key: idx for idx, key in enumerate(sorted_by_first)}
            sorted_by_second = sorted(keys, key=lambda k: loss_dict[k][1])
            rank_second = {key: idx for idx, key in enumerate(sorted_by_second)}
            total_rank = {key: rank_first[key] + rank_second[key] for key in keys}
            sorted_keys = sorted(keys, key=lambda k: (total_rank[k], loss_dict[k][0]))

            top_5_layers = sorted_keys[-5:]
            layer_names = [layer for layer in layer_names if layer not in top_5_layers]
            self.sensitive_layer += [
                layer for layer in top_5_layers if layer not in self.sensitive_layer
            ]

            mix_table = self.mix_prec._gen_mix_table(self.sensitive_layer, self.qtable)
            int8_model = MixQuantModel(self.fp32_mlir, self.chip, self.low_prec, self.high_prec,
                                       self.cali_table_name, mix_table)
            outputs_cos = self.mix_prec.run_model(int8_model, False, global_compare_layers,
                                                  layers_rate, predictions_gt)
            outer_pbar.update(1)
            outer_pbar.set_postfix_str(f"{outputs_cos:.4f}")
            if (outputs_cos - cos_sim < eps and outputs_cos - cos_sim < 0.001) or len(
                    self.sensitive_layer) > 0.2 * len(all_op_names):
                break
            else:
                if self.iteration_count == 1:
                    eps = abs(outputs_cos - cos_sim) / 2
                cos_sim = outputs_cos
                if self.iteration_count >= self.args.inference_num:
                    break
        outer_pbar.close()
        self.print_log_info(self.sensitive_layer, int8_outputs_cos, outputs_cos, t0)
