#!/usr/bin/env python3
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
import gc
import re
import time
import copy
import math
import numpy as np
import pymlir
import torch
import warnings
from ctypes import *
from tqdm import tqdm
import datetime
from utils.preprocess import preprocess
from utils.mlir_parser import *
from utils.log_setting import setup_logger
from utils.misc import *
from math import *
from calibration.data_selector import DataSelector
from calibration.data_driven_runner import DataDrivenDAGRunner
from .utils import *
from .simple_tunner import SimpleTuner

from .cali_math import cosine_sim, math_impl, POOL_THREADS

pymlir.set_mem_mode("force_value_mem")

cur_dir_path = os.path.join(os.path.dirname(__file__))
calibration_math_path = os.path.join("/".join(cur_dir_path.split("/")[:-2]),
                                     "lib/calibration_math.so")
if not os.path.exists(calibration_math_path):
    calibration_math_path = "calibration_math.so"


class BaseKldCalibrator:

    def __init__(self, math_lib_path=calibration_math_path):
        self.calib_lib = CDLL(math_lib_path)
        self.calib_lib.kl_diversity.restype = c_float
        self.calib_lib.kl_diversity_hist.restype = c_float

    def kld_threshold(self, hist, width, bin_num, dst_bins):
        threshold = self.calib_lib.kl_diversity_hist(hist.ctypes.data_as(POINTER(c_int)),
                                                     c_float(width), c_longlong(bin_num),
                                                     c_longlong(dst_bins))
        return threshold


class ActivationCalibrator(BaseKldCalibrator):

    def __init__(self, args, ds: DataSelector, tune_ds: DataSelector):
        super().__init__()
        self.args = copy.deepcopy(args)
        self.start_time = time.time()
        self.tuned_op_list = []
        self.debug_cmd = args.debug_cmd
        self.fuseop_list = {}
        if 'fp8' in self.debug_cmd:
            if 'int4' in self.debug_cmd:
                print('can not calibration both for int4 and fp8')
                sys.exit(1)
            if 'use_torch_observer_for_cali' in self.debug_cmd:
                print('not use use_torch_observer_for_cali for fp8')
                self.debug_cmd.pop('use_torch_observer_for_cali')
            if 'max' not in self.debug_cmd:
                self.debug_cmd['max'] = 1
            if 'percentile9999' in self.debug_cmd:
                print('only use max for fp8')
                self.debug_cmd.pop('percentile9999')
            if 'tune_steps' in self.debug_cmd:
                self.debug_cmd.pop('tune_steps')
            print(f'final dbg cmd is {self.debug_cmd}')
            self.args.tune_num = 0
        # if 'input_calibration_table' in self.debug_cmd:
        self.module = pymlir.module()
        self.module.load(args.mlir_file)
        self.torchObserver_dict = {}
        if 'use_torch_observer_for_cali' in self.debug_cmd:
            if "int4" in self.debug_cmd:
                print('can not use int4 in torch observer')
                sys.exit(1)
            from torch import qint8, per_tensor_affine
            Observer_type = 'HistogramObserver'
            if 'Observer_type' in self.debug_cmd:
                Observer_type = self.debug_cmd['Observer_type']
            if Observer_type == 'MovingAverageMinMaxObserver':
                from torch.quantization import MovingAverageMinMaxObserver
                for tensor in self.module.all_tensor_names:
                    self.torchObserver_dict[tensor] = MovingAverageMinMaxObserver(
                        averaging_constant=0.1, dtype=qint8, qscheme=per_tensor_affine)
            elif Observer_type == 'HistogramObserver':
                from torch.quantization import HistogramObserver
                for tensor in self.module.all_tensor_names:
                    self.torchObserver_dict[tensor] = HistogramObserver(bins=args.histogram_bin_num,
                                                                        dtype=qint8,
                                                                        qscheme=per_tensor_affine)
            else:
                print('Observer_type in debug_cmd is error')
                exit(1)
        self.parser = MlirParser(args.mlir_file)
        for op_name in self.parser.get_op_name_list():
            fuseop_list_append(op_name, self.fuseop_list)
        self.batch_size = self.parser.get_batch_size()
        self.input_num = self.parser.get_input_num()
        self.ppa_list = []
        self.calibration_method = parse_calibration_methods(self.args.cali_method, self.debug_cmd)

        for i in range(self.input_num):
            tmp = preprocess()
            tmp.load_config(self.parser.get_input_op_by_idx(i))
            self.ppa_list.append(tmp)
        self.ds = ds
        self.tune_ds = ds if tune_ds is None else tune_ds
        self.data_list = ds.data_list
        self.args.input_num = len(self.data_list)
        if ds.all_image or ds.all_yuv:
            n = self.args.input_num % self.batch_size
            if n != 0:
                for i in range(self.batch_size - n):
                    self.data_list.append(self.data_list[-1])
                    self.args.input_num += 1
            self.args.input_num = self.args.input_num // self.batch_size
        log_level = "DEBUG" if 'debug_log' in self.debug_cmd else "INFO"
        self.logger = setup_logger('ActivationCalibrator', log_level=log_level)
        self.histogram_bin_num = args.histogram_bin_num
        self.tune_steps = 20
        self.num_samples = self.args.input_num
        if 'tune_steps' in self.debug_cmd:
            self.tune_steps = int(self.debug_cmd['tune_steps'])
        self.load_net_input()

    def _clean_resource(self):
        del self.module
        self.module = None

    def load_net_input(self):
        self.dq_activations = {}
        self.ref_activations = {}
        inp_ref_dict = {}
        for input in self.module.input_names:
            inp_ref_dict[input] = self.parser.get_use_count_by_op_name(input)

        if self.ds.all_image or self.ds.all_yuv:
            batched_inputs = self.input_num * ['']
        else:
            batched_inputs = {}
        idx, tune_idx = 0, 0
        self.dq_activations[tune_idx] = {}
        self.ref_activations[tune_idx] = {}
        only_one = len(self.module.input_names) == 1
        for data in self.data_list:
            if self.ds.all_npz:
                x = np.load(data)
                if only_one:
                    assert (len(x.files) == 1)
                    n0 = self.module.input_names[0]
                    n1 = x.files[0]
                    if x[n1].shape[0] > 1:
                        self.dq_activations[tune_idx][n0] = [x[n1], inp_ref_dict[n0]]
                        self.ref_activations[tune_idx][n0] = [x[n1], inp_ref_dict[n0]]
                    else:
                        batched_inputs[n1] = (np.concatenate(
                            [batched_inputs[n1], x[n1].astype(np.float32)], axis=0)
                                              if n1 in batched_inputs else x[n1].astype(np.float32))
                        if batched_inputs[n1].shape[0] >= self.batch_size:
                            self.dq_activations[tune_idx][n0] = [
                                batched_inputs[n1][:self.batch_size], inp_ref_dict[n0]
                            ]
                            self.ref_activations[tune_idx][n0] = [
                                batched_inputs[n1][:self.batch_size], inp_ref_dict[n0]
                            ]
                            batched_inputs.pop(n1)
                        else:
                            continue
                else:
                    for input in self.module.input_names:
                        assert (input in x)
                        if x[input].shape[0] > 1:
                            self.dq_activations[tune_idx][input] = [x[input], inp_ref_dict[input]]
                            self.ref_activations[tune_idx][input] = [x[input], inp_ref_dict[input]]
                            batch_size = self.batch_size
                        else:
                            batched_inputs[input] = (np.concatenate(
                                [batched_inputs[input], x[input].astype(np.float32)],
                                axis=0) if input in batched_inputs else x[input].astype(np.float32))
                            batch_size = batched_inputs[input].shape[0]
                            if batched_inputs[input].shape[0] >= self.batch_size:
                                real_batch_size = self.parser.get_op_by_op_name(input).shape[0]
                                self.dq_activations[tune_idx][input] = [
                                    batched_inputs[input][:real_batch_size], inp_ref_dict[input]
                                ]
                                self.ref_activations[tune_idx][input] = [
                                    batched_inputs[input][:real_batch_size], inp_ref_dict[input]
                                ]
                                batched_inputs.pop(input)

                    if batch_size < self.batch_size:
                        continue

            elif self.ds.all_image or self.ds.all_yuv:
                idx += 1
                inputs = data.split(',')
                inputs = [s.strip() for s in inputs]
                assert (self.input_num == len(inputs))
                for i in range(self.input_num):
                    batched_inputs[i] += '{},'.format(inputs[i])
                    if idx == self.batch_size:
                        x = self.ppa_list[i].run(batched_inputs[i][:-1])
                        name = self.ppa_list[i].input_name
                        self.dq_activations[tune_idx][name] = [x, inp_ref_dict[name]]
                        self.ref_activations[tune_idx][name] = [x, inp_ref_dict[name]]
                if idx == self.batch_size:
                    idx = 0
                    batched_inputs = self.input_num * ['']
                else:
                    continue
            else:
                self.dq_activations[tune_idx] = {}
                self.ref_activations[tune_idx] = {}
                inputs = data.split(',')
                inputs = [s.strip() for s in inputs]
                assert (self.input_num == len(inputs))
                for name, input in zip(self.module.input_names, inputs):
                    x = np.load(input)
                    self.dq_activations[tune_idx][name] = [x, inp_ref_dict[name]]
                    self.ref_activations[tune_idx][name] = [x, inp_ref_dict[name]]
            tune_idx += 1
            self.dq_activations[tune_idx] = {}
            self.ref_activations[tune_idx] = {}

        if len(self.ref_activations[tune_idx]) == 0:
            self.ref_activations.pop(tune_idx)
        print(f"input_num = {self.args.input_num}, ref = {len(self.ref_activations)}")
        self.args.input_num = min(self.args.input_num, len(self.ref_activations))
        print(f"real input_num = {self.args.input_num}")
        assert self.args.input_num > 0

    def clear_ref_tensor(self, i, evaled_op):
        if self.ref_activations[i][evaled_op][1] == 0:  # Clear residual network output
            self.ref_activations[i].pop(evaled_op)
        input_ops = self.parser.get_pre_op_by_op_name(evaled_op)
        for input_op in input_ops:
            if input_op in self.ref_activations[i]:
                if self.ref_activations[i][input_op][1] == 1:
                    self.ref_activations[i].pop(input_op)
                else:
                    self.ref_activations[i][input_op][1] -= 1

    def get_ref_tensor(self, i, evaled_op):
        # print(i, self.ref_activations[i])
        if evaled_op in self.ref_activations[i]:
            return self.ref_activations[i][evaled_op][0]
        print('error, idx:{} evaled_op:{} not in ref_activations'.format(i, evaled_op))
        return None

    def gen_ref_tensor(self, i, op_name):
        op_name = split_fuseop(op_name)[0]
        if op_name in self.ref_activations[i]:
            return

        def set_func(layer_name):
            if layer_name == op_name:
                input_ops = self.parser.get_pre_op_by_op_name(op_name)
                for input_op in input_ops:
                    if input_op in self.ref_activations[i]:
                        data = self.ref_activations[i][input_op][0]
                        self.module.set_tensor(input_op, data, data.shape)

        def get_func(layer_name):
            if layer_name == op_name:
                count = self.parser.get_use_count_by_op_name(op_name)
                self.ref_activations[i][op_name] = [
                    self.module.get_tensor(layer_name).copy(), count
                ]
                outputs = self.parser.get_outputs_by_op_name(op_name)
                if outputs is not None:
                    for output in outputs:
                        if output == op_name:
                            continue
                        count = self.parser.get_use_count_by_op_name(output)
                        if count > 0:
                            self.ref_activations[i][output] = [
                                self.module.get_tensor(output).copy(), count
                            ]
                elif outputs is None and op_name in self.fuseop_list:
                    fused_op_name = self.fuseop_list[op_name]
                    outputs = self.parser.get_outputs_by_op_name(fused_op_name)
                    for output in outputs:
                        if output == op_name:
                            continue
                        count = self.parser.get_use_count_by_op_name(output)
                        if count > 0:
                            self.ref_activations[i][output] = [
                                self.module.get_tensor(output).copy(), count
                            ]

        self.module.before_invoke(set_func)
        self.module.after_invoke(get_func)
        if len(self.parser.get_pre_op_by_op_name(op_name)) > 0 or op_name in self.fuseop_list:
            self.module.invoke_at(op_name)
        self.module.clear_hooks()

    def find_threshold(self, histogram_data_map, histogram_width_map, dst_bins=128):
        thresholds = {}
        num = len(histogram_data_map)
        pbar = tqdm(range(num), total=num, position=0, leave=True)
        for item in histogram_data_map:
            pbar.set_description("[{}] threshold: {}".format(self.histogram_bin_num, item))
            pbar.update(1)

            def _padding(data_map, bin_num):
                remainder = len(data_map) % bin_num
                padding_size = 0 if remainder == 0 else bin_num - remainder
                padding = np.zeros(padding_size, dtype=data_map.dtype)
                data_map = np.concatenate((data_map, padding))
                return data_map

            histogram_data_map[item] = _padding(histogram_data_map[item], dst_bins)
            MAX_BINS = 64 * 2048  # give a huge number of bins to avoid too many thread in cpp cal function, max threads would be 64*2048 / 128 (8bit) = 1024
            while len(histogram_data_map[item]) > MAX_BINS:
                histogram_data_map[
                    item] = histogram_data_map[item][::2] + histogram_data_map[item][1::2]
                histogram_data_map[item] = _padding(histogram_data_map[item], dst_bins)
                histogram_width_map[item] *= 2.0
            thresholds[item] = self.kld_threshold(histogram_data_map[item],
                                                  histogram_width_map[item],
                                                  len(histogram_data_map[item]), dst_bins)
        pbar.close()
        return thresholds

    def choose_asym_op(self, tensor_list):
        """Set the op to use asymmetric quantization.
        """
        for op in tensor_list:
            op_type = self.parser.get_op_type_by_op_name(op)
            if op_type in ['top.SiLU', 'top.GELU']:
                next_op = self.parser.get_next_op_by_op_name(op)
                if all(
                        self.parser.get_op_type_by_op_name(nxt_op) in ['top.Conv', 'top.MatMul']
                        for nxt_op in next_op):
                    self.asym_op.append(op)
                    self.asym_op1.append(op)
                    for nxt_op in next_op:
                        self.asym_op1.append(nxt_op)

    def process_statistic(self, evaled_op, i, idx, all_tensors):
        per = 99.99 + i * self.step
        self.perd[evaled_op] = per
        outputs = self.parser.get_outputs_by_op_name(evaled_op)
        res_length = 0
        for out in outputs:
            if out not in all_tensors and out not in self.tensor_list:
                continue
            abs_value = None
            activation = self.module.get_tensor(out).flatten()
            if activation is None:
                continue
            self.size[out] = activation.size
            res_length = int(self.args.input_num * self.size[out] * (1 - per / 100)) + 1
            self.res_length_dict[out] = res_length
            if out in self.asym_op:
                tmp = math_impl.prepare(activation)
                tmp = math_impl.sort_distr_percentile(tmp, res_length)
                if type(tmp) != np.ndarray and tmp.device != 'cpu':
                    self.all_data_test[out] = np.concatenate((self.all_data_test[out], tmp.get()))
                else:
                    self.all_data_test[out] = np.concatenate((self.all_data_test[out], tmp))
            else:
                if 'use_torch_observer_for_cali' in self.debug_cmd:
                    from torch import Tensor
                    self.torchObserver_dict[out](Tensor(activation.astype(np.float32)))
                else:
                    self.min_value[out] = min(np.min(activation), self.min_value[out])
                    self.max_value[out] = max(np.max(activation), self.max_value[out])
                    abs_value = max(abs(self.min_value[out]), abs(self.max_value[out]))
                    bits = [8, 4] if 'int4' in self.debug_cmd else [8]
                    activation = math_impl.prepare(activation.flatten())
                    if self.args.kurtosis_analysis:
                        kurtosis = math_impl.get_kurtosis(activation)
                        if out not in self.kurtosis:
                            self.kurtosis[out] = []
                            self.kurtosis[out].append(kurtosis)
                        else:
                            self.kurtosis[out].append(kurtosis)
                    if out in self.last_five_tensors:
                        tmp = math_impl.sort_distr_percentile(activation, res_length)
                        if type(tmp) != np.ndarray and tmp.device != 'cpu':
                            self.all_data_test[out] = np.concatenate(
                                (self.all_data_test[out], tmp.get()))
                        else:
                            self.all_data_test[out] = np.concatenate((self.all_data_test[out], tmp))
                    if 'percentile9999' in self.calibration_method and out not in self.last_five_tensors:
                        tmp = math_impl.sort_distr_percentile(activation, res_length)
                        if type(tmp) != np.ndarray and tmp.device != 'cpu':
                            self.all_data_test[out] = np.concatenate(
                                (self.all_data_test[out], tmp.get()))
                        else:
                            self.all_data_test[out] = np.concatenate((self.all_data_test[out], tmp))
                    if 'mse' in self.calibration_method:
                        th_mses = math_impl.threshold_mse(activation, bits)
                        if out not in self.mse:
                            self.mse[out] = []
                            self.mse[out].append(th_mses[8])
                        else:
                            self.mse[out].append(th_mses[8])
                        if 4 in bits:
                            if out not in self.mse:
                                self.mse4[out] = []
                                self.mse4[out].append(th_mses[4])
                            else:
                                self.mse4[out].append(th_mses[4])
                    if 'aciq_gauss' in self.calibration_method:
                        th_aciq_gauss = math_impl.threshold_aciq_gauss(activation, bits)
                        if out not in self.aciq_g:
                            self.aciq_g[out] = []
                            self.aciq_g[out].append(th_aciq_gauss[8])
                        else:
                            self.aciq_g[out].append(th_aciq_gauss[8])
                        if 4 in bits:
                            if out not in self.aciq_g:
                                self.aciq_g4[out] = []
                                self.aciq_g4[out].append(th_aciq_gauss[4])
                            else:
                                self.aciq_g4[out].append(th_aciq_gauss[4])
                    if 'aciq_laplace' in self.calibration_method:
                        th_aciq_laplace = math_impl.threshold_aciq_laplace(activation, bits)
                        if out not in self.aciq_l:
                            self.aciq_l[out] = []
                            self.aciq_l[out].append(th_aciq_laplace[8])
                        else:
                            self.aciq_l[out].append(th_aciq_laplace[8])
                        if 4 in bits:
                            if out not in self.aciq_l:
                                self.aciq_l4[out] = []
                                self.aciq_l4[out].append(th_aciq_laplace[4])
                            else:
                                self.aciq_l4[out].append(th_aciq_laplace[4])
                    if 'max' in self.calibration_method:
                        self.max_abs_value[out] = max(abs_value, self.max_abs_value[out])
                    if out not in self.histogram_data_map:
                        hist, width = math_impl.histogram(activation, abs_value,
                                                          self.histogram_bin_num)
                        self.histogram_data_map[out] = hist
                        self.histogram_width_map[out] = width
                        self.hist_dict[out] = (hist, width, self.min_value[out],
                                               self.max_value[out], abs_value)
                    else:
                        self.hist_dict[out] = math_impl.combine_histogram(
                            self.hist_dict[out], activation, self.min_value[out],
                            self.max_value[out], abs_value, self.histogram_bin_num)
                        self.histogram_data_map[out] = self.hist_dict[out][0]
                        self.histogram_width_map[out] = self.hist_dict[out][1]
        return

    def process_compute_threshold(self, evaled_op, i, all_tensors):
        abs_value = None
        outputs = self.parser.get_outputs_by_op_name(evaled_op)
        for out in outputs:
            if out not in all_tensors and out not in self.tensor_list:
                continue
            abs_value = max(abs(self.min_value[out]), abs(self.max_value[out]))
            bits = [8, 4] if 'int4' in self.debug_cmd else [8]
            result = calibration_result(min_val=self.min_value[out],
                                        max_val=self.max_value[out],
                                        abs_max=abs_value)
            result4 = calibration_result(min_val=self.min_value[out],
                                         max_val=self.max_value[out],
                                         abs_max=abs_value)
            if out in self.asym_op:
                res = np.sort(self.all_data_test[out])
                res_max = res[-self.res_length_dict[out]:]
                res_min = res[:self.res_length_dict[out]][::-1]
                inter = self.args.input_num * self.size[out] - 1
                idx = int((self.perd[evaled_op] / 100) * inter)
                ratio = (self.perd[evaled_op] / 100) * inter - idx
                v_max = res_max[0] + ratio * (
                    res_max[1] - res_max[0]) if self.res_length_dict[out] != 1 else res_max[0]
                v_min = res_min[0] + ratio * (
                    res_min[1] - res_min[0]) if self.res_length_dict[out] != 1 else res_min[0]
                self.min_value[out] = v_min
                self.max_value[out] = v_max
                result.abs_max = max(abs(v_max), abs(v_min))
                result.min_val = v_min
                result.max_val = v_max
                result.abs_max = max(abs(v_min), abs(v_max))
                result.kl = result.abs_max
                result.mse = result.abs_max
                result.aciq_g = result.abs_max
                result.aciq_l = result.abs_max
                result.p99 = result.abs_max
                result.p99_min = v_min
                result.p99_max = v_max
                if 4 in bits:
                    result4.abs_max = max(abs(v_max), abs(v_min))
                    result4.min_val = v_min
                    result4.max_val = v_max
                    result4.abs_max = max(abs(v_min), abs(v_max))
                    result4.kl = result.abs_max
                    result4.mse = result.abs_max
                    result4.aciq_g = result.abs_max
                    result4.aciq_l = result.abs_max
                    result4.p99 = result.abs_max
                    result4.p99_min = v_min
                    result4.p99_max = v_max
            else:
                if 'percentile9999' in self.calibration_method:
                    res = np.sort(self.all_data_test[out])
                    res_max = res[-self.res_length_dict[out]:]
                    res_min = res[:self.res_length_dict[out]][::-1]
                    inter = self.args.input_num * self.size[out] - 1
                    idx = int((self.perd[evaled_op] / 100) * inter)
                    ratio = (self.perd[evaled_op] / 100) * inter - idx
                    v_max = res_max[0] + ratio * (
                        res_max[1] - res_max[0]) if self.res_length_dict[out] != 1 else res_max[0]
                    v_min = res_min[0] + ratio * (
                        res_min[1] - res_min[0]) if self.res_length_dict[out] != 1 else res_min[0]
                    self.min_value[out] = v_min
                    self.max_value[out] = v_max
                    result.p99 = max(abs(v_max), abs(v_min))
                    result.p99_min = v_min
                    result.p99_max = v_max
                if 'mse' in self.calibration_method or 'aciq_gauss' in self.calibration_method or 'aciq_laplace' in self.calibration_method:
                    if out in self.last_five_tensors:
                        res = np.sort(np.abs(self.all_data_test[out]))[-self.res_length_dict[out]:]
                        inter = self.args.input_num * self.size[out] - 1
                        idx = int((self.perd[evaled_op] / 100) * inter)
                        ratio = (self.perd[evaled_op] / 100) * inter - idx
                        self.last_five_tensors_threshold[out] = res[0] + ratio * (
                            res[1] - res[0]) if self.res_length_dict[out] != 1 else res[0]
                if 'mse' in self.calibration_method:
                    mean_s_n = sum(self.mse[out]) / len(self.mse[out])
                    mean_s_n = min(mean_s_n, abs_value)
                    result.mse = mean_s_n
                    if 4 in bits:
                        mean_s_n = sum(self.mse4[out]) / len(self.mse4[out])
                        mean_s_n = min(mean_s_n, abs_value)
                        result4.mse = mean_s_n
                if 'aciq_gauss' in self.calibration_method:
                    mean_gauss = sum(self.aciq_g[out]) / len(self.aciq_g[out])
                    if mean_gauss > abs_value:
                        mean_gauss = abs_value
                    result.aciq_g = mean_gauss
                    if 4 in bits:
                        mean_gauss = sum(self.aciq_g4[out]) / len(self.aciq_g4[out])
                        if mean_gauss > abs_value:
                            mean_gauss = abs_value
                        result4.aciq_g = mean_gauss
                if 'aciq_laplace' in self.calibration_method:
                    mean_laplace = sum(self.aciq_l[out]) / len(self.aciq_l[out])
                    if mean_laplace > abs_value:
                        mean_laplace = abs_value
                    result.aciq_l = mean_laplace
                    if 4 in bits:
                        mean_laplace = sum(self.aciq_l4[out]) / len(self.aciq_l4[out])
                        if mean_laplace > abs_value:
                            mean_laplace = abs_value
                        result4.aciq_l = mean_laplace
                if abs_value != None and abs_value <= 1e-5:
                    # if op's outputs are all close to zero, change it to 1e-5 for them.
                    self.min_value[out] = -1e-5 if self.min_value[out] < 0 else 0
                    self.max_value[out] = 1e-5
                    abs_value = 1e-5
                    result.min_val = self.min_value[out]
                    result.max_val = self.max_value[out]
                    result.abs_max = abs_value
                    result4.min_val = self.min_value[out]
                    result4.max_val = self.max_value[out]
                    result4.abs_max = abs_value
                    print("WARNING: layer {} is all zeros. Please check the "
                          "input data correctness.".format(out))
                if self.args.kurtosis_analysis:
                    mean_kurtosis = sum(self.kurtosis[out]) / len(self.kurtosis[out])
                    #print("op:{} and kurtosis:{}".format(out,mean_kurtosis))
                    if out not in self.kurtosis_result:
                        self.kurtosis_result[out] = []
                        self.kurtosis_result[out].append(mean_kurtosis)
                    else:
                        self.kurtosis_result[out].append(mean_kurtosis)
            self.activations_statistics[out] = [result, result4] if 4 in bits else [result]
        return

    def parallel_statistic(self, all_tensors, idx):
        from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED
        with ThreadPoolExecutor(
                max_workers=POOL_THREADS) as executor:  # You can adjust max_workers as needed
            future_to_tensor = {
                executor.submit(self.process_statistic, evaled_op, i, idx, all_tensors): evaled_op
                for i, evaled_op in enumerate(all_tensors)
            }
            _, _ = wait(future_to_tensor, return_when=ALL_COMPLETED)
        return

    def parallel_compute_threshold(self, all_tensors):
        from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED
        with ThreadPoolExecutor(
                max_workers=POOL_THREADS) as executor:  # You can adjust max_workers as needed
            future_to_tensor = {
                executor.submit(self.process_compute_threshold, evaled_op, i, all_tensors):
                evaled_op
                for i, evaled_op in enumerate(all_tensors)
            }
            _, _ = wait(future_to_tensor, return_when=ALL_COMPLETED)
        return

    def activation_collect_and_calc_th(self):
        t0 = time.time()
        all_tensors = self.parser.get_op_name_list()
        thresholds_out = []
        muti_output_tensor = []
        self.size = {}
        self.hist_dict = {}
        self.res_length_dict = {}
        self.perd = {}
        self.mse = {}
        self.mse4 = {}
        self.aciq_g = {}
        self.aciq_l = {}
        self.aciq_g4 = {}
        self.aciq_l4 = {}
        self.kurtosis = {}
        self.kurtosis_result = {}
        self.input_op = []
        self.asym_op = []
        self.asym_op1 = []
        self.last_five_tensors_threshold = {}
        self.histogram_data_map = {}
        self.histogram_width_map = {}
        self.activations_statistics = {}
        self.tensor_list = []
        self.tensor_list = get_no_fused_tensors(self.parser, all_tensors)
        self.last_five_tensors = self.tensor_list[-5:][::-1]
        self.min_value = {tensor: float('inf') for tensor in self.tensor_list}
        self.max_value = {tensor: float('-inf') for tensor in self.tensor_list}
        self.max_abs_value = {tensor: float('-inf') for tensor in self.tensor_list}
        self.all_data_test = {tensor: [] for tensor in self.tensor_list}

        for op in all_tensors:
            if len(self.parser.get_outputs_by_op_name(op)) > 1:
                muti_output_tensor.append(op)
            if self.parser.get_pre_op_by_op_name(op) == []:
                self.input_op.append(op)
        if self.args.part_asymmetric:
            self.choose_asym_op(self.tensor_list)
        self.step = (99.999999 - 99.99) / len(all_tensors)
        input_number = [i for i in range(self.args.input_num)]
        pbar = tqdm(input_number, total=self.args.input_num, position=0, leave=True)
        use_data_driven_runner = False
        if use_data_driven_runner:
            from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
            executor = ThreadPoolExecutor(max_workers=POOL_THREADS)
            futures = []

            def statistic_callback(op_name, idx):
                future = executor.submit(self.process_statistic, op_name,
                                         all_tensors.index(op_name), idx, all_tensors)
                futures.append(future)

            runner = DataDrivenDAGRunner(mlir_file=None,
                                         module=self.module,
                                         parser=self.parser,
                                         ds=None,
                                         img_list=None,
                                         ref_act=self.ref_activations,
                                         op_callback=statistic_callback)
            for idx in range(self.args.input_num):
                pbar.set_description("activation_collect_and_calc_th for sample: {}".format(idx))
                pbar.update(1)
                runner.parallel_thread_execute(idx)
                # runner.execute(idx)
                done, not_done = wait(futures, return_when=ALL_COMPLETED)
                futures = []
        else:
            for idx in range(self.args.input_num):
                pbar.set_description("activation_collect_and_calc_th for sample: {}".format(idx))
                pbar.update(1)
                data = []
                for name in list(self.ref_activations[idx].keys()):
                    data.append(self.ref_activations[idx][name][0])
                for k, v in zip(self.module.input_names, data):
                    self.module.set_tensor(k, v, v.shape)
                self.module.invoke()
                self.parallel_statistic(all_tensors, idx)
        pbar.close()
        for tensor in self.histogram_data_map:
            if type(self.histogram_data_map[tensor]
                    ) != np.ndarray and self.histogram_data_map[tensor].device != 'cpu':
                self.histogram_data_map[tensor] = self.histogram_data_map[tensor].get()

        self.parallel_compute_threshold(all_tensors)
        if 'use_torch_observer_for_cali' in self.debug_cmd:
            for i, evaled_op in enumerate(all_tensors):
                assert 'int4' not in self.debug_cmd, "int4 not support use_torch_observer_for_cali"
                outputs = self.parser.get_outputs_by_op_name(evaled_op)
                for out in outputs:
                    if out not in all_tensors:
                        continue
                    qmin, qmax = -128, 127
                    scale, zp = self.torchObserver_dict[out].calculate_qparams()
                    threshold = float(scale * max(-(qmin - zp), (qmax - zp)))
                    threshold = 1e-5 if (threshold <= 1e-5) else threshold  # fix me
                    self.activations_statistics[out][0].kl = threshold
                    self.activations_statistics[out][0].min = scale * (qmin - zp)
                    self.activations_statistics[out][0].max = scale * (qmax - zp)
        if 'mse' in self.calibration_method or 'aciq_gauss' in self.calibration_method or 'aciq_laplace' in self.calibration_method:
            for tensor in self.last_five_tensors:
                if self.last_five_tensors_threshold[tensor] != max(
                        abs(self.activations_statistics[tensor][0].min_val),
                        abs(self.activations_statistics[tensor][0].max_val)):
                    break
                thresholds_out.append(tensor)
                if self.parser.get_op_type_by_op_name(
                        tensor) == 'top.MatMul' or self.parser.get_op_type_by_op_name(
                            tensor) == 'top.Conv':
                    break
        if 'use_torch_observer_for_cali' not in self.debug_cmd:
            thresholds_map = self.find_threshold(self.histogram_data_map, self.histogram_width_map,
                                                 128)
            if 'int4' in self.debug_cmd:
                thresholds_map4 = self.find_threshold(self.histogram_data_map,
                                                      self.histogram_width_map, 8)
            for k, result in self.activations_statistics.items():
                result8 = result[0]
                result4 = result[1] if len(result) == 2 else None
                if k in thresholds_map:
                    result8.kl = thresholds_map[k]
                    if result4 is not None:
                        result4.kl = thresholds_map4[k]
                result8.canonicalize()
                if result4 is not None:
                    result4.canonicalize()
                mi = result8.min_val
                ma = result8.max_val
                if 'mse' in self.calibration_method:
                    if k in thresholds_out:
                        result8.mse = self.last_five_tensors_threshold[k]
                        if result4 is not None:
                            result4.mse = self.last_five_tensors_threshold[k]
                        continue
                    if k in self.input_op:
                        result8.mse = max(abs(mi), abs(ma))
                        if result4 is not None:
                            result4.mse = max(abs(mi), abs(ma))
                        continue
                if 'aciq_gauss' in self.calibration_method:
                    if k in thresholds_out:
                        result8.aciq_g = self.last_five_tensors_threshold[k]
                        if result4 is not None:
                            result4.aciq_g = self.last_five_tensors_threshold[k]
                        continue
                    if k in self.input_op:
                        result8.aciq_g = max(abs(mi), abs(ma))
                        if result4 is not None:
                            result4.aciq_g = max(abs(mi), abs(ma))
                        continue
                if 'aciq_laplace' in self.calibration_method:
                    if k in thresholds_out:
                        result8.aciq_l = self.last_five_tensors_threshold[k]
                        if result4 is not None:
                            result4.aciq_l = self.last_five_tensors_threshold[k]
                        continue
                    if k in self.input_op:
                        result8.aciq_l = max(abs(mi), abs(ma))
                        if result4 is not None:
                            result4.aciq_l = max(abs(mi), abs(ma))
                        continue
        time2 = time.time()
        # return thresholds_map, thresholds_map_absmax, thresholds_map_scale, thresholds_map_zp, thresholds_map4, thresholds_map_absmax4, thresholds_map_scale4, thresholds_map_zp4

    def dump_cali_table(self, cali_table: str, cali_method: str, thresholds_map: dict = {}):
        thresholds_map_list = []
        layer_name_list = []
        op_layers = self.parser.get_op_name_list()
        use_dict = len(thresholds_map) > 0
        with open(cali_table, 'w') as f:
            f.write("# mlir version: {} cali version 1.0\n".format(pymlir.__version__))
            f.write("# mlir: {}\n".format(self.args.mlir_file))
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# quantization threshold calculation method: {}\n".format(cali_method))
            if 'fp8' in self.debug_cmd:
                f.write("#tpu-mlir-fp8 caliration table\n")
            else:
                f.write("# histogram number: {}\n".format(self.histogram_bin_num))
            f.write("# sample number: {}\n###\n".format(self.num_samples))
            f.write("# tune number: {}\n###\n".format(self.args.tune_num))
            f.write("# op_name    threshold    min    max\n")
            for i, op_name in enumerate(op_layers):
                outputs = self.parser.get_outputs_by_op_name(op_name)
                no_fused_outputs = get_no_fused_tensors(self.parser, outputs)
                for out in no_fused_outputs:
                    if out not in self.activations_statistics:
                        continue  # possible useless leaf output
                    if 'fp8' in self.debug_cmd:
                        if use_dict:
                            threshold = thresholds_map[out]
                        else:
                            threshold = self.activations_statistics[out][0].get_threshold('max')
                        if threshold <= 1e-5 or np.isnan(threshold):
                            threshold = 1e-5
                            print("WARNING: layer {} threshold is zero. Please check the "
                                  "input data correctness.".format(op_name))
                        min_value, max_value = -threshold, threshold
                        thresholds_map_list.append(threshold)
                        layer_name_list.append('{}_{}'.format(i, op_name))
                        f.write("{} {:.7f} {:.7f} {:.7f}\n".format(out, threshold, min_value,
                                                                   max_value))
                    else:
                        if out in self.activations_statistics:
                            if use_dict:
                                threshold = thresholds_map[out]
                            else:
                                threshold = self.activations_statistics[out][0].get_threshold(
                                    cali_method)
                            if threshold <= 1e-5 or np.isnan(threshold):
                                threshold = 1e-5
                                print("WARNING: layer {} threshold is zero. Please check the "
                                      "input data correctness.".format(op_name))
                            if cali_method == 'percentile9999':  # hack to keep same with former version
                                min_value = self.activations_statistics[out][0].p99_min
                                max_value = self.activations_statistics[out][0].p99_max
                            else:
                                min_value = self.activations_statistics[out][0].min_val
                                max_value = self.activations_statistics[out][0].max_val
                        else:
                            threshold = 1.0
                            min_value, max_value = -1, 1
                        thresholds_map_list.append(threshold)
                        layer_name_list.append('{}_{}'.format(i, op_name))
                        f.write("{} {:.7f} {:.7f} {:.7f}\n".format(out, threshold, min_value,
                                                                   max_value))
            if 'int4' in self.debug_cmd and not use_dict:
                f.write("\n")
                f.write("#int4_th\n")
                for i, op_name in enumerate(op_layers):
                    outputs = self.parser.get_outputs_by_op_name(op_name)
                    no_fused_outputs = get_no_fused_tensors(self.parser, outputs)
                    for out in no_fused_outputs:
                        if out not in self.activations_statistics or len(
                                self.activations_statistics[out]) < 2:
                            continue
                        threshold = self.activations_statistics[out][1].get_threshold(cali_method)
                        min_value, max_value = -threshold * 8.0 / 7.0, threshold
                        f.write("{} {:.7f} {:.7f} {:.7f}\n".format(out, threshold, min_value,
                                                                   max_value))
            if self.args.part_asymmetric:
                f.write("\n")
                f.write("#asym_op\n")
                for i, op_name in enumerate(self.asym_op1):
                    f.write("{}\n".format(op_name))
        return thresholds_map_list, layer_name_list

    def gen_multiple_thresholds(self, all_op_names, quantize_method_list):
        layer_th_dicts = {}  # method_name: {op_name: [op_fmax, op_th]}
        tmp_th_dict = {}
        quantize_method_list = [x.lower() for x in quantize_method_list]
        self.args.calibration_method = quantize_method_list

        self.activation_collect_and_calc_th()
        self._clean_resource()
        # step 3: dump threshold table of default histogram bins
        for method_name in quantize_method_list:
            cali_table_orig = self.args.calibration_table
            cali_table = self.args.calibration_table + "_" + method_name
            if self.args.tune_num > 0:
                cali_table += ".1"
            _, _ = self.dump_cali_table(cali_table, method_name)
            if self.args.tune_num <= 0:
                self.args.calibration_table = cali_table_orig
                continue
            else:
                cali_table = cali_table.rsplit(".1", 1)[0]
            self.args.calibration_table = cali_table
            thresholds_map_absmax = {}
            for op in self.activations_statistics:
                if method_name == 'percentile9999':  # hack to use p99 as absmax to keep same with former version
                    thresholds_map_absmax[op] = self.activations_statistics[op][0].p99
                else:
                    thresholds_map_absmax[op] = self.activations_statistics[op][0].abs_max

            tunner = SimpleTuner(self.args, self.tune_ds, self.ppa_list, thresholds_map_absmax)
            thresholds_map = tunner.run()
            cali_table += "_tune"
            _, _ = self.dump_cali_table(cali_table, method_name, thresholds_map)
            for op_name in all_op_names:
                if op_name not in thresholds_map:
                    pass
                else:
                    tmp_th_dict[op_name] = [thresholds_map_absmax[op_name], thresholds_map[op_name]]
            layer_th_dicts[method_name] = tmp_th_dict
            self.args.calibration_table = cali_table_orig
        return layer_th_dicts

    def run(self):
        layer_name_list = []
        thresholds_map_list = []
        op_layers = self.parser.get_op_name_list()
        assert len(self.calibration_method
                   ) == 1, "only support one calibration method in simple calibration"
        if 'input_calibration_table' in self.debug_cmd:
            assert self.args.tune_num > 0
            input_calibration_table = self.debug_cmd['input_calibration_table']
            if input_calibration_table != '' and os.path.exists(input_calibration_table):
                os.system('cp -f {name} {name}.1'.format(name=input_calibration_table))
                threshold_table = CalibrationTable(input_calibration_table)
                for op_name in op_layers:
                    op_name = split_fuseop(op_name)
                    for op_ in op_name:
                        thresholds_map_list.append(threshold_table.thresholds_map[op_name][0])
            else:
                print('input_calibration_table error')
                exit(1)
        else:
            self.activation_collect_and_calc_th()
            if self.args.smc:  # fix softmax correction mul threshold as 1
                for i, op_name in enumerate(op_layers):
                    op_type = self.parser.get_op_type_by_op_name(op_name)
                    pre_op = self.parser.get_pre_op_by_op_name(op_name)
                    if op_type != 'top.Mul' or len(pre_op) != 1: continue
                    pre_op_type = self.parser.get_op_type_by_op_name(pre_op[0])
                    if pre_op_type == 'top.Softmax':
                        output = self.parser.get_outputs_by_op_name(op_name)[0]
                        result8 = self.activations_statistics[output][0]
                        result4 = self.activations_statistics[output][1] if len(
                            self.activations_statistics[output]) == 2 else None
                        result8.set_unsigned_all(1.0)
                        if result4 is not None:
                            result4.set_unsigned_all(1.0)
            self._clean_resource()
            # step 3: dump threshold table of default histogram bins
            cali_table = self.args.calibration_table
            if self.args.tune_num > 0:
                cali_table += ".1"
            for method_ in self.calibration_method:
                thresholds_map_list, _ = self.dump_cali_table(cali_table, method_)
        if self.args.tune_num <= 0 or 'int4' in self.debug_cmd or 'input_calibration_table' in self.debug_cmd or 'percentile9999' in self.calibration_method or 'max' in self.calibration_method or 'mse' in self.calibration_method:
            # set absmax to threshold in former version, infact tune is not done for mse
            os.rename(cali_table, self.args.calibration_table)
            return
        # setp 4: tune to get better threshold of each layers.
        thresholds_map_absmax = {}
        for op in self.activations_statistics:
            thresholds_map_absmax[op] = self.activations_statistics[op][0].abs_max
        tunner = SimpleTuner(self.args, self.tune_ds, self.ppa_list, thresholds_map_absmax)
        thresholds = tunner.run()
        if self.args.smc:  # fix softmax correction mul threshold as 1
            for i, op_name in enumerate(op_layers):
                op_type = self.parser.get_op_type_by_op_name(op_name)
                pre_op = self.parser.get_pre_op_by_op_name(op_name)
                if op_type != 'top.Mul' or len(pre_op) != 1: continue
                pre_op_type = self.parser.get_op_type_by_op_name(pre_op[0])
                if pre_op_type == 'top.Softmax':
                    output = self.parser.get_outputs_by_op_name(op_name)[0]
                    thresholds[output] = 1.0

        # step 5: dump threshold table after tuning
        op_layers = get_no_fused_tensors(self.parser, op_layers)
        tuned_threshold_list, layer_name_list = self.dump_cali_table(self.args.calibration_table,
                                                                     self.calibration_method,
                                                                     thresholds)
        if cali_table.endswith('.1'):
            os.remove(cali_table)
        if 'print_debug_info' in self.debug_cmd:
            th_before_tuned = np.array(thresholds_map_list)
            th_after_tuned = np.array(tuned_threshold_list)
            file_prefix = './{}_{}pics_{}_times_tuned_th_statistic'.format(
                self.args.mlir_file.split('.')[0], tunner.args.tune_num, tunner.tune_steps)
            save_tensor_diff_subplot(th_before_tuned, th_after_tuned, layer_name_list,
                                     'before_tuned', 'after_tuned', file_prefix)
        if self.args.kurtosis_analysis:
            if '/' in self.args.calibration_table:
                last_index = self.args.calibration_table.rfind('/')
                kurtosis_result = self.args.calibration_table[:last_index + 1] + "kurtosis_analysis"
            else:
                kurtosis_result = "kurtosis_analysis"
            # Sort by value in descending order
            sorted_kurtosis_result = dict(
                sorted(self.kurtosis_result.items(), key=lambda item: item[1], reverse=True))
            with open(kurtosis_result, "w") as f:
                f.write("# op_name   kurtosis   op_type\n")
                for layer, kurtosis in sorted_kurtosis_result.items():
                    op_type = self.parser.get_op_type_by_op_name(layer)
                    if op_type in ['top.LayerNorm', 'top.Softmax']:
                        pass
                    else:
                        f.write("{}  {:.7f}  {}\n".format(layer, kurtosis[0], op_type))
