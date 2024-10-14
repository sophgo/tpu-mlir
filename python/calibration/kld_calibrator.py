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
pymlir.set_mem_mode("force_value_mem")
from ctypes import *
from tqdm import tqdm
import datetime
from utils.preprocess import preprocess
from utils.mlir_parser import *
from utils.log_setting import setup_logger
from utils.misc import *
#import graphviz as gz
from math import *
from scipy import spatial
from calibration.data_selector import DataSelector

cur_dir_path = os.path.join(os.path.dirname(__file__))
calibration_math_path = os.path.join("/".join(cur_dir_path.split("/")[:-2]), "lib/calibration_math.so")
if not os.path.exists(calibration_math_path):
    calibration_math_path = "calibration_math.so"

class BaseKldCalibrator:

    def __init__(self, math_lib_path=calibration_math_path):
        self.calib_lib = CDLL(math_lib_path)
        self.calib_lib.kl_diversity.restype = c_float
        self.calib_lib.kl_diversity_hist.restype = c_float

    def histogram(self, ndarray, abs_max, bin_num):
        t = np.abs(ndarray.flatten())
        t = t[t != 0]
        width = abs_max / (bin_num - 1)
        if t.size > 0:
            hist, _ = np.histogram(np.floor(t / width + 0.5),
                                   bins=bin_num,
                                   range=(0, bin_num - 1),
                                   density=False)
        else:
            hist = np.zeros(bin_num)
        hist = hist.astype(np.int32)
        return hist, width

    def kld_threshold(self, hist, width, bin_num, dst_bins):
        threshold = self.calib_lib.kl_diversity_hist(hist.ctypes.data_as(POINTER(c_int)),
                                                     c_float(width), c_longlong(bin_num), c_longlong(dst_bins))
        return threshold


class CalibrationTable:

    def __init__(self, table):
        self.headers, self.thresholds_map = self.parse(table)

    def parse(self, table):
        thresholds_map = dict()
        headers = []
        with open(table, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith('#'):
                    headers.append(line)
                    continue
                # op_name    threshold    min    max
                fields = line.split(' ')
                if len(fields) != 4:
                    print("Table format should be 'op_name, threshold, min, max'")
                    raise RuntimeError("Error with parse {} in {}".format(line, table))

                op_name, threshold, _min, _max = fields
                thresholds_map[op_name] = [float(threshold), float(_min), float(_max)]
        return headers, thresholds_map

    def dump(self, dest_table):
        with open(dest_table, "w") as f:
            for line in self.headers:
                f.write(line + "\n")
            for k, v in self.thresholds_map.items():
                f.write("{} {:.7f} {:.7f} {:.7f}\n".format(k, *v))

    def update_to(self, dest_table, target_op, new_threshold):
        with open(dest_table, "w") as f:
            for line in self.headers:
                f.write(line + "\n")
            for k, v in self.thresholds_map.items():
                if k == target_op:
                    f.write("{} {:.7f} {:.7f} {:.7f}\n".format(k, new_threshold, v[1], v[2]))
                else:
                    f.write("{} {:.7f} {:.7f} {:.7f}\n".format(k, *v))


def is_npz(image):
    return True if image.split('.')[-1] == 'npz' else False


def is_npy(image):
    return True if image.split('.')[-1] == 'npy' else False

def is_fuseop(op_name):
    return re.match(r'^fused\[".*?"\]$', op_name)

def split_fuseop(op_name):
    if is_fuseop(op_name):
        new_ops = re.findall(r'"([^"]+)"', op_name)
        return new_ops[0]
    else:
        return op_name

def fuseop_list_append(op_name,fuseop_list):
    if is_fuseop(op_name):
        new_ops = re.findall(r'"([^"]+)"', op_name)
        if op_name not in fuseop_list:
            fuseop_list[op_name] = new_ops
            fuseop_list[new_ops[0]] = op_name
    return

def import_quant_bias(value, threshold):
    scale = 127 / threshold
    value = np.round(value * scale)
    value[value > 127] = 127
    value[value < -128] = -128
    value /= scale
    return value


def sort_distr(array, length):
    def first_k(a, k):
        a_sort = np.sort(a)
        return a_sort[-k:]
    return first_k(array, length)

def sort_distr_per(array, length):
    def first_k(a, k):
        a_sort = np.sort(a)
        f = a_sort[:k].copy()
        r = a_sort[-k:].copy()
        a = np.concatenate((f,r))
        return a
    return first_k(array, length)

def cosine_sim(x, y):
    x[np.isnan(x)] = 0.0
    y[np.isnan(y)] = 0.0
    cosine_similarity = 1 - spatial.distance.cosine(x.flatten().astype(np.float32),
                                                    y.flatten().astype(np.float32))
    return cosine_similarity


class SimpleTuner:

    def __init__(self, args, ds: DataSelector, ppa_list, abs_max_dict):
        self.args = copy.deepcopy(args)
        self.start_time = time.time()
        self.abs_max_dict = abs_max_dict
        self.args.tune_num = min(len(ds.data_list), args.tune_num)
        self.tuned_op_list = []
        self.data_list = ds.data_list  # [:self.args.tune_num]
        self.ppa_list = ppa_list
        self.debug_cmd = parse_debug_cmd(args.debug_cmd)
        self.fuseop_list = {}
        if self.debug_cmd:
            print('debug_cmd:', self.debug_cmd)
        if 'input_calibration_table' in self.debug_cmd:
            self.threshold_table = CalibrationTable(self.debug_cmd['input_calibration_table'] +
                                                    ".1")
        else:
            self.threshold_table = CalibrationTable(args.calibration_table + ".1")
        self.module = pymlir.module()
        self.module.load(args.mlir_file)
        self.parser = MlirParser(args.mlir_file)
        for op_name in self.parser.get_op_name_list():
            fuseop_list_append(op_name,self.fuseop_list)
        self.batch_size = self.parser.get_batch_size()
        self.input_num = self.parser.get_input_num()
        self.ds = ds
        if ds.all_image:
            n = self.args.tune_num % self.batch_size
            if n != 0:
                for i in range(self.batch_size - n):
                    self.data_list.append(self.data_list[-1])
                    self.args.tune_num += 1
            self.args.tune_num = self.args.tune_num // self.batch_size
        log_level = "DEBUG" if 'debug_log' in self.debug_cmd else "INFO"
        self.logger = setup_logger('auto_tune', log_level=log_level)
        self.tune_steps = 21
        if 'tune_steps' in self.debug_cmd:
            self.tune_steps = int(self.debug_cmd['tune_steps'])
        self.module_dq = pymlir.module()
        self.module_dq.load(args.mlir_file)
        self.module_dq.fake_quant_weight()
        self.load_net_input()
        self.dot = None
        #self.dot = gz.Digraph()

    def print_dbg(self, *para):
        tmp = [str(item) for item in para]
        tmpStr = ' '.join(tmp)
        self.logger.debug(tmpStr)

    def print_info(self, *para):
        tmp = [str(item) for item in para]
        tmpStr = ' '.join(tmp)
        self.logger.info(tmpStr)

    def load_net_input(self):
        self.dq_activations = {}
        self.ref_activations = {}
        inp_ref_dict = {}
        for input in self.module.input_names:
            inp_ref_dict[input] = self.parser.get_use_count_by_op_name(input)

        if self.ds.all_image:
            batched_inputs = self.input_num * ['']
        else:
            batched_inputs = {}

        idx, tune_idx = 0, 0
        self.dq_activations[tune_idx] = {}
        self.ref_activations[tune_idx] = {}
        only_one = len(self.module.input_names) == 1
        print(f'prepare data from {len(self.data_list)}')
        for data in self.data_list:
            if len(self.ref_activations) > self.args.tune_num + 1:
                break
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
                                [batched_inputs[input], x[input].astype(np.float32)], axis=0) if input
                                                    in batched_inputs else x[input].astype(np.float32))
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
            elif self.ds.all_image:
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
            # print(f'last tune data (tune_idx={tune_idx}) not valid, droped')
            self.ref_activations.pop(tune_idx)
        self.args.tune_num = min(self.args.tune_num, len(self.ref_activations))
        # print(f"tune_num = {self.args.tune_num}, ref = {len(self.ref_activations)}")
        # print(f"real tune_num = {self.args.tune_num}")
        assert self.args.tune_num > 0

    def get_input_tensor(self, i, op_name):
        if op_name in self.dq_activations[i]:
            return self.dq_activations[i][op_name][0]
        print('error, idx:{} op_name:{} not in dq_activations'.format(i, op_name))
        return None

    def clear_input_tensor(self, i, op_name, node_label):
        if i == 0:
            node_label[0] += '\nclear_input_tensor {}\'s input'.format(op_name)
        input_ops = self.parser.get_pre_op_by_op_name(op_name)
        for input_op in input_ops:
            if input_op in self.dq_activations[i]:
                if self.dq_activations[i][input_op][1] == 1:
                    self.dq_activations[i].pop(input_op)
                    if i == 0:
                        node_label[0] += '\npop its input:{}'.format(input_op)
                else:
                    self.dq_activations[i][input_op][1] -= 1
                    tmp = self.dq_activations[i][input_op][1]
                    if i == 0:
                        node_label[0] += '\ndec {}\'s refcount:{} > {}'.format(
                            input_op, tmp + 1, tmp)

    def clear_ref_tensor(self, i, evaled_op, node_label):
        if i == 0:
            node_label[0] += '\nclear_ref_tensor {}\'s input'.format(evaled_op)
        # evaled_op = split_fuseop(evaled_op
        if self.ref_activations[i][evaled_op][1] == 0:  #清除残留的网络输出
            self.ref_activations[i].pop(evaled_op)
        input_ops = self.parser.get_pre_op_by_op_name(evaled_op)
        for input_op in input_ops:
            if input_op in self.ref_activations[i]:
                if self.ref_activations[i][input_op][1] == 1:
                    self.ref_activations[i].pop(input_op)
                    if i == 0:
                        node_label[0] += '\npop its input:{}'.format(input_op)
                else:
                    self.ref_activations[i][input_op][1] -= 1
                    tmp = self.ref_activations[i][input_op][1]
                    if i == 0:
                        node_label[0] += '\ndec {}\'s refcount:{} > {}'.format(
                            input_op, tmp + 1, tmp)

    def gen_input_tensor(self, i, input_tensor_of_evaled_op, node_label):
        if i == 0:
            tmp = 'gen_input_tensor:{}'.format(input_tensor_of_evaled_op)
        if input_tensor_of_evaled_op in self.dq_activations[i]:
            if i == 0:
                tmp += '\nit exsit'
                node_label[0] += '\n{}'.format(tmp)
            return True

        input_ops = self.parser.get_pre_op_by_op_name(input_tensor_of_evaled_op)
        for input_op in input_ops:
            data = self.get_input_tensor(i, input_op)
            if data is None:
                return False
            threshold = self.threshold_table.thresholds_map[input_op][0]
            data_q = import_quant_bias(data, threshold)
            cos_sim = cosine_sim(data, data_q)
            self.module_dq.set_tensor(input_op, data_q)
            if i == 0:
                tmp += '\nits input:{} import_quant_bias, th:{}, cos:{:.4f}'.format(
                    input_op, threshold, cos_sim)
        if len(input_ops) > 0:
            value = self.module_dq.invoke_at(input_tensor_of_evaled_op)
            target_fp32_activations = self.get_ref_tensor(i, input_tensor_of_evaled_op)
            cos_sim = cosine_sim(target_fp32_activations, value)
            if input_tensor_of_evaled_op in self.layer_cos_sim:
                self.layer_cos_sim[input_tensor_of_evaled_op] += cos_sim
            else:
                self.layer_cos_sim[input_tensor_of_evaled_op] = cos_sim
            count = self.parser.get_use_count_by_op_name(input_tensor_of_evaled_op)
            self.dq_activations[i][input_tensor_of_evaled_op] = [value.copy(), count]
            if i == 0:
                tmp += '\nrun it, refcount:{}, cos:{}\n'.format(count, cos_sim)
        if i == 0:
            node_label[0] += '\n{}'.format(tmp)
        return True

    def get_ref_tensor(self, i, evaled_op):
        if evaled_op in self.ref_activations[i]:
            return self.ref_activations[i][evaled_op][0]
        print('error, idx:{} evaled_op:{} not in ref_activations'.format(i, evaled_op))
        return None

    def gen_ref_tensor(self, i, op_name, node_label):
        if i == 0:
            tmp = 'gen_ref_tensor:{}'.format(op_name)
        if op_name in self.ref_activations[i]:
            if i == 0:
                tmp += '\nit exsit'
                node_label[0] += '\n{}'.format(tmp)
            return
        input_ops = self.parser.get_pre_op_by_op_name(op_name)
        if op_name in self.fuseop_list and input_ops == []:
            fused_op_name = self.fuseop_list[op_name]
            input_ops = self.parser.get_pre_op_by_op_name(fused_op_name)
        # print(op_name, input_ops)
        for input_op in input_ops:
            data = self.ref_activations[i][input_op][0]
            refcount = self.ref_activations[i][input_op][1]
            if i == 0:
                tmp += '\nits input:{}, refcount:{}'.format(input_op, refcount)

            self.module.set_tensor(input_op, data)
        if len(input_ops) > 0:
            value = self.module.invoke_at(op_name)
            outputs = self.parser.get_outputs_by_op_name(op_name)
            if outputs is None and op_name in self.fuseop_list:
                fused_op_name = self.fuseop_list[op_name]
                outputs = self.parser.get_outputs_by_op_name(fused_op_name)
            count = 0
            for o_ in outputs:
                count += self.parser.get_use_count_by_op_name(o_)
            for o_ in outputs:
                self.ref_activations[i][o_] = [value.copy(), count]
            if i == 0:
                tmp += '\ninvoke_at:{}, refcount:{}'.format(op_name, count)
                #self.print_dbg('have {} users as refcount'.format(count))
        if i == 0:
            node_label[0] += '\n{}'.format(tmp)

    def calc_distance(self, evaled_op, threshold):
        distance = 0
        total_cosine_similarity = 0
        for idx in range(self.args.tune_num):
            for input in self.parser.get_pre_op_by_op_name(evaled_op):
                if idx == 0:
                    self.print_dbg('{}\'s input:{} import_quant_bias, th:{}'.format(
                        evaled_op, input, threshold))
                if 'not_use_fp32_tensor_as_ref' in self.debug_cmd:
                    value = self.get_input_tensor(idx, input)
                else:
                    value = self.get_ref_tensor(idx, input)
                if value is None:
                    print('error, calc_distance get tensor fail')
                    return None, None
                value = import_quant_bias(value, threshold)
                self.module_dq.set_tensor(input, value)
            target_activations = self.module_dq.invoke_at(evaled_op)
            target_fp32_activations = self.get_ref_tensor(idx, evaled_op)
            total_cosine_similarity += cosine_sim(target_activations, target_fp32_activations)
            diff = target_fp32_activations.flatten() - target_activations.flatten()
            norm_2 = np.linalg.norm(diff)
            norm_1 = np.linalg.norm(target_fp32_activations.flatten(), ord=1)
            distance += norm_2 / norm_1
        return distance / self.args.tune_num, total_cosine_similarity / self.args.tune_num

    def find_better_threshold(self, evaled_op, tuned_op, node_label):
        prev_distance = -1
        threshold = self.initial_threshold[tuned_op][0]
        # abs_max = max(map(abs, self.initial_threshold[tuned_op][1:]))
        abs_max = self.abs_max_dict[tuned_op]
        #op_no = self.module.all_tensor_names.index(tuned_op)
        if tuned_op in self.parser.get_op_name_list():
            op_no = self.parser.get_op_name_list().index(tuned_op)
        else:
            op_no = 0
        self.print_dbg('>>>tuned_op_idx:', op_no, ', tuned_op:', tuned_op, ', threshold:',
                       threshold, 'abs_max:', abs_max, ', evaled_op:', evaled_op)
        if threshold > abs_max:
            self.print_dbg('waring, threshold > abs_max, do not tune the threshold')
        th_min = threshold
        if 'find_lower_th' in self.debug_cmd:
            th_min = threshold / 3
        #print(f'tuned_op:{tuned_op}, th_min:{th_min}, abs_max:{abs_max}')
        best_threshold = threshold
        best_cos_sim = 0
        init_cos_sim = 0

        pre_ops = self.parser.get_pre_op_by_op_name(evaled_op)
        for idx in range(self.args.tune_num):
            if 'not_use_fp32_tensor_as_ref' in self.debug_cmd:
                for pre_op in pre_ops:
                    if not self.gen_input_tensor(idx, pre_op, node_label):
                        return False
        min_tuned_diff = 0.01
        if 'min_tuned_diff' in self.debug_cmd:
            min_tuned_diff = self.debug_cmd['min_tuned_diff']
        diff = abs(abs_max - th_min)
        step = (abs_max - th_min) / self.tune_steps
        ranges = range(self.tune_steps + 1)
        if step > 0 and diff > min_tuned_diff:
            r = 1
            if 'find_lower_th' in self.debug_cmd :
                r = 2
            for n in range(r):
                if n == 1 and 'find_lower_th' in self.debug_cmd:
                    th_min = best_threshold - step
                    th_max = best_threshold + step
                    diff = abs(th_max - th_min)
                    times = self.tune_steps // 2
                    step = (th_max - th_min) / times
                    ranges = range(times + 1)[1:-1]
                    #print(f'find_lower_th enable,tuned_op:{tuned_op},best_threshold:{best_threshold},step:{step},th_min:{th_min},th_max:{th_max}, times:{times}')
                for i in ranges:
                    cur_threshold = th_min + step * i
                    # print(1+((i-1-self.tune_steps/2)/self.tune_steps))
                    cur_distance, cur_cos_sim = self.calc_distance(evaled_op, cur_threshold)
                    if cur_distance is None and cur_cos_sim is None:
                        return False
                    if prev_distance == -1:
                        self.print_dbg("### tuning i:{}, tuned_op_idx:{}, tuned_op:{}, threshold:"
                                       "{:5f}, distance: {}".format(i, op_no, tuned_op,
                                                                    cur_threshold, cur_distance))
                        prev_distance = cur_distance
                        init_cos_sim = cur_cos_sim
                        continue
                    elif cur_distance < prev_distance:  # and cur_cos_sim > best_cos_sim
                        # self.print_dbg(
                        #     "### tuning i:{}, tuned_op_idx:{}, tuned_op:{}, find a better threshold:"
                        #     "{:5f}".format(i, op_no, tuned_op,cur_threshold))
                        prev_distance = cur_distance
                        best_threshold = cur_threshold
                        # print(f'tuned_op:{tuned_op}, find best_threshold:{best_threshold}')
                        best_cos_sim = cur_cos_sim
                    # else:
                    # self.print_dbg(
                    #     "### tuning i:{}, tuned_op_idx:{}, tuned_op:{}, not a better threshold:"
                    #     "{:5f}".format(i, op_no, tuned_op,cur_threshold))

        for idx in range(self.args.tune_num):
            if 'not_use_fp32_tensor_as_ref' in self.debug_cmd:
                self.clear_input_tensor(idx, tuned_op, node_label)
        if step > 0:
            if threshold != best_threshold:
                node_label[
                    0] += '\ntune, find:{} th:{:.4f}->{:.4f}, abs_max:{:.4f}, cos_sim:{:.3f}->{:.3f}'.format(
                        tuned_op, threshold, best_threshold, abs_max, init_cos_sim, best_cos_sim)
            else:
                node_label[
                    0] += '\ntune:{} no better th, init th:{}, cos_sim:{:.3f}->{:.3f}'.format(
                        tuned_op, threshold, init_cos_sim, best_cos_sim)
        else:
            node_label[0] += '\nnot tune {}, init th:{}'.format(tuned_op, threshold)

        #若当前tune的best_threshold大于先前该th tune后的最佳门限,则保存，否则保留上次tune更大的结果
        if best_threshold > self.threshold_table.thresholds_map[tuned_op][0]:
            node_label[0] += '\nupdate {} th: {}>{}'.format(
                tuned_op, self.threshold_table.thresholds_map[tuned_op][0], best_threshold)
            self.threshold_table.thresholds_map[tuned_op][0] = best_threshold
        return True

    def isAllInputTuned(self, evaled_op):
        pre_ops = self.parser.get_pre_op_by_op_name(evaled_op)
        for tuned_op in pre_ops:
            if tuned_op not in self.tuned_op_list:
                return False
        return True

    def run(self):
        self.layer_cos_sim = {}
        all_tensors = self.parser.get_op_name_list()
        self.initial_threshold = copy.deepcopy(self.threshold_table.thresholds_map)
        pbar = tqdm(all_tensors, total=len(all_tensors), position=0, leave=True)
        for i, evaled_op in enumerate(all_tensors):
            pbar.set_description("tune op: {}".format(evaled_op))
            pbar.update(1)
            type = self.parser.get_op_type_by_op_name(evaled_op)
            if type is None:
                continue
            node_label = ['idx:{}  name:{}  type:{}\n'.format(i, evaled_op, type.split('.')[-1])]
            if type == 'top.Input':
                if self.dot is not None:
                    self.dot.node(evaled_op, node_label[0], shape='box')
                continue

            pre_ops = self.parser.get_pre_op_by_op_name(evaled_op)
            if self.dot is not None:
                for pre_op in pre_ops:
                    evaled_op = split_fuseop(evaled_op)
                    self.dot.edge(pre_op, evaled_op, label=pre_op)

            for idx in range(self.args.tune_num):
                evaled_op = split_fuseop(evaled_op)
                self.gen_ref_tensor(idx, evaled_op, node_label)

            #若op的多个输入都已调节过，那任挑其中1个来调节，暂定第1个
            if self.isAllInputTuned(evaled_op):
                ret = self.find_better_threshold(evaled_op, pre_ops[0], node_label)
                if not ret:
                    break
                self.tuned_op_list.append(pre_ops[0])
                if self.dot is not None:
                    self.dot.node(evaled_op, node_label[0], shape='box')
                for idx in range(self.args.tune_num):
                    self.clear_ref_tensor(idx, evaled_op, node_label)
                continue
            faild = False
            for tuned_op in pre_ops:
                #op的多个输入，调节那些没有调节过的；
                if tuned_op not in self.tuned_op_list:
                    ret = self.find_better_threshold(evaled_op, tuned_op, node_label)
                    if not ret:
                        faild = True
                        break
                    self.tuned_op_list.append(tuned_op)
            if faild:
                break

            for idx in range(self.args.tune_num):
                self.clear_ref_tensor(idx, evaled_op, node_label)

            self.print_dbg('>>>>buffered_tensors info:')
            self.print_dbg('dq_activations keys:', list(self.dq_activations[0].keys()))
            for op_name in self.dq_activations[0]:
                self.print_dbg('op:', op_name, ' refcount:', self.dq_activations[0][op_name][1])
            self.print_dbg('ref_activations keys:', list(self.ref_activations[0].keys()))
            for op_name in self.ref_activations[0]:
                self.print_dbg('op:', op_name, ' refcount:', self.ref_activations[0][op_name][1])
            if self.dot is not None:
                self.dot.node(evaled_op, node_label[0], shape='box')
        pbar.close()
        print('auto tune end, run time:{}'.format(time.time() - self.start_time))

        if 'print_debug_info' in self.debug_cmd:
            index_list = []
            cos_tensor = []
            for i, layer in enumerate(self.layer_cos_sim):
                index_list.append('{}_{}'.format(i, layer))
                cos_tensor.append(self.layer_cos_sim[layer] / self.args.tune_num)
            index_list = np.array(index_list)
            cos_tensor = np.array(cos_tensor)
            save_tensor_subplot(cos_tensor, index_list, 'layer_cos_sim', 'layer_cos_sim')
        if self.dot is not None:
            self.dot.render(filename='auto_tune_result', directory='./', view=False)
            os.system('dot -Tsvg ./auto_tune_result -o ./auto_tune_result.svg')
        tuned_th_dict = {
            i: self.threshold_table.thresholds_map[i][0]
            for i in self.threshold_table.thresholds_map
        }
        return tuned_th_dict


class ActivationCalibrator(BaseKldCalibrator):

    def __init__(self, args, ds: DataSelector, tune_ds: DataSelector):
        super().__init__()
        self.args = copy.deepcopy(args)
        self.start_time = time.time()
        self.tuned_op_list = []
        self.debug_cmd = parse_debug_cmd(args.debug_cmd)
        self.fuseop_list = {}
        if 'fp8' in self.debug_cmd:
            if 'int4' in self.debug_cmd:
                print('can not calibration both for int4 and fp8')
                sys.exit(1)
            if 'use_torch_observer_for_cali' in self.debug_cmd:
                print('not use use_torch_observer_for_cali for fp8')
                self.debug_cmd.pop('use_torch_observer_for_cali')
            if 'use_max' not in self.debug_cmd:
                self.debug_cmd['use_max']=1
            if 'use_percentile9999' in self.debug_cmd:
                print('only use max for fp8')
                self.debug_cmd.pop('use_percentile9999')
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
            fuseop_list_append(op_name,self.fuseop_list)
        self.batch_size = self.parser.get_batch_size()
        self.input_num = self.parser.get_input_num()
        self.ppa_list = []

        for i in range(self.input_num):
            tmp = preprocess()
            tmp.load_config(self.parser.get_input_op_by_idx(i))
            self.ppa_list.append(tmp)
        self.ds = ds
        self.tune_ds = ds if tune_ds is None else tune_ds
        self.data_list = ds.data_list
        self.args.input_num = len(self.data_list)
        if ds.all_image:
            n = self.args.input_num % self.batch_size
            if n != 0:
                for i in range(self.batch_size - n):
                    self.data_list.append(self.data_list[-1])
                    self.args.input_num += 1
            self.args.input_num = self.args.input_num // self.batch_size
        log_level = "DEBUG" if 'debug_log' in self.debug_cmd else "INFO"
        self.logger = setup_logger('auto_tune', log_level=log_level)
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

        if self.ds.all_image:
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
                                [batched_inputs[input], x[input].astype(np.float32)], axis=0) if input
                                                    in batched_inputs else x[input].astype(np.float32))
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

            elif self.ds.all_image:
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
            print(f'last input data (idx={tune_idx}) not valid, droped')
            self.ref_activations.pop(tune_idx)
        self.args.input_num = min(self.args.input_num, len(self.ref_activations))
        print(f"input_num = {self.args.input_num}, ref = {len(self.ref_activations)}")
        print(f"real input_num = {self.args.input_num}")
        assert self.args.input_num > 0

    def clear_ref_tensor(self, i, evaled_op):
        if self.ref_activations[i][evaled_op][1] == 0:  #清除残留的网络输出
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
        op_name = split_fuseop(op_name)
        if op_name in self.ref_activations[i]:
            return
        def set_func(layer_name):
            if layer_name==op_name:
                input_ops = self.parser.get_pre_op_by_op_name(op_name)
                for input_op in input_ops:
                    if input_op in self.ref_activations[i]:
                        data = self.ref_activations[i][input_op][0]
                        self.module.set_tensor(input_op, data)
        def get_func(layer_name):
            if layer_name==op_name:
                count = self.parser.get_use_count_by_op_name(op_name)
                self.ref_activations[i][op_name] = [self.module.get_tensor(layer_name).copy(), count]
                outputs = self.parser.get_outputs_by_op_name(op_name)
                if outputs is not None:
                    for output in outputs:
                        if output == op_name:
                            continue
                        count = self.parser.get_use_count_by_op_name(output)
                        if count > 0:
                            self.ref_activations[i][output] = [self.module.get_tensor(output).copy(), count]
                elif outputs is None and op_name in self.fuseop_list:
                    fused_op_name = self.fuseop_list[op_name]
                    outputs = self.parser.get_outputs_by_op_name(fused_op_name)
                    for output in outputs:
                        if output == op_name:
                            continue
                        count = self.parser.get_use_count_by_op_name(output)
                        if count > 0:
                            self.ref_activations[i][output] = [self.module.get_tensor(output).copy(), count]
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
            remainder = len(histogram_data_map[item]) % dst_bins
            padding_size = 0 if remainder == 0 else dst_bins - remainder
            padding = np.zeros(padding_size,dtype=histogram_data_map[item].dtype)
            histogram_data_map[item] = np.concatenate((histogram_data_map[item],padding))
            thresholds[item] = self.kld_threshold(histogram_data_map[item],
                                                  histogram_width_map[item],len(histogram_data_map[item]), dst_bins)
        pbar.close()
        return thresholds

    def combine_histogram(self, old_hist, arr, new_min, new_max, new_th):
        """ Collect layer histogram for arr and combine it with old histogram.
        """
        (old_hist, old_hist_edges, old_min, old_max, old_th) = old_hist
        if new_th <= old_th:
            hist,width = self.histogram(arr,old_th,len(old_hist))
            return (old_hist + hist, old_hist_edges, min(old_min, new_min), max(old_max, new_max), old_th)
        else:
            # Need to generate new histogram with new_th
            old_num_bins = len(old_hist)
            old_step = old_th / old_num_bins
            half_increased_bins = int((new_th - old_th) // old_step + 1)
            new_num_bins = half_increased_bins  + old_num_bins
            # if new_num_bins > 40000000 :
            #     import warnings
            #     warnings.warn("校准图片差异过大,如果方便请调整校准图片顺序或者替换校准图片", UserWarning)
            #     new_num_bins = 40000000
            #     half_increased_bins = new_num_bins - old_num_bins
            if new_num_bins > 8000000:
                import warnings
                warnings.warn("校准图片差异过大,如果方便请调整校准图片顺序或者替换校准图片", UserWarning)
                hist, hist_edges = self.histogram(arr, new_th, self.histogram_bin_num)
                return (hist, hist_edges, min(old_min, new_min), max(old_max, new_max), new_th)
            new_th = half_increased_bins * old_step + old_th
            hist, hist_edges = self.histogram(arr, new_th,new_num_bins)
            hist[0:new_num_bins - half_increased_bins] += old_hist
            return (hist, hist_edges, min(old_min, new_min), max(old_max, new_max), new_th)

    def _combine_histogram(self, old_hist, arr, new_min, new_max, new_th):
        """ Collect layer histogram for arr and combine it with old histogram.
        """
        upsample_rate = 128
        (old_hist, old_hist_edges, old_min, old_max, old_th) = old_hist
        combined_min = 0
        combined_max = new_th
        downsample_rate = int(
            torch.ceil(
                ((combined_max - combined_min) / old_th) * upsample_rate
            ).item()
        )
        e = downsample_rate / upsample_rate * old_th - (combined_max - combined_min)
        start_idx = int(
            torch.round((0 - combined_min) / (old_th) * 2048 * upsample_rate).item()
        )
        combined_max = combined_max + e
        hist, hist_edges = self.histogram(arr,combined_max,2048)
        if combined_max == old_th :
            old_hist = old_hist + hist
        else:
            upsampled_histogram = old_hist.repeat_interleave(upsample_rate)
            histogram_with_output_range = torch.zeros(
                (2048 * downsample_rate)
            )
            histogram_with_output_range[
                start_idx : 2048 * upsample_rate + start_idx
            ] = upsampled_histogram
            integral_histogram = torch.cumsum(
                histogram_with_output_range, 0, dtype=torch.double
            )[downsample_rate - 1 :: downsample_rate]
            shifted_integral_histogram = torch.zeros((2048))
            shifted_integral_histogram[1:2048] = integral_histogram[0:-1]
            interpolated_histogram = (
                integral_histogram - shifted_integral_histogram
            ) / upsample_rate
            old_hist = hist + interpolated_histogram.to(torch.float)
        return (old_hist,hist_edges,min(old_min,new_min),max(old_max,new_max),combined_max)

    def get_no_fused_tensors(self, all_tensors):
        tensor_list = []
        for op in all_tensors:
            if "fused" in op:
                fused_ops = op.split('["')[1].split('"]')[0].split(', ')
                tensor_list.extend([fused_op.strip('"') for fused_op in fused_ops])
            else:
                tensor_list.append(op)
        return tensor_list

    def calculate_kurtosis(self, signal):
        """Calculates the kurtosis of a given signal.
        """
        signal = signal.flatten()
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return float('inf')

        standardized_signal = (signal - mean) / (std + 1e-8)
        kurtosis = np.mean(standardized_signal**4)

        return kurtosis.item()

    def choose_asym_op(self, tensor_list):
        """Set the op to use asymmetric quantization.
        """
        for op in tensor_list:
            op_type = self.parser.get_op_type_by_op_name(op)
            if op_type in ['top.SiLU','top.GELU']:
                next_op = self.parser.get_next_op_by_op_name(op)
                if all(self.parser.get_op_type_by_op_name(nxt_op) in ['top.Conv', 'top.MatMul'] for nxt_op in next_op):
                    self.asym_op.append(op)
                    self.asym_op1.append(op)
                    for nxt_op in next_op:
                        self.asym_op1.append(nxt_op)

    def process_statistic_muti(self, muti_output_tensor, idx):
        if muti_output_tensor == []:
            return
        for i, evaled_op in enumerate(muti_output_tensor):
            self.process_statistic(evaled_op, i, idx, muti_output_tensor)
        return

    def process_statistic(self, evaled_op, i, idx, all_tensors):
        per = 99.99 + i * self.step
        self.perd[evaled_op] = per
        outputs = self.parser.get_outputs_by_op_name(evaled_op)
        res_length = 0
        for out in outputs:
            if out not in all_tensors and out not in self.tensor_list:
                continue
            abs_value = None
            activation=self.module.get_tensor(out).copy()
            if activation is None:
                continue
            self.size[out] = activation.size
            res_length = int(self.args.input_num * self.size[out] * (1 - per / 100)) + 1
            self.res_length_dict[out] = res_length
            if out in self.asym_op:
                tmp = activation.flatten()
                tmp = sort_distr_per(tmp, res_length)
                self.all_data_test[out] = np.concatenate((self.all_data_test[out],tmp))
            else:
                if 'use_torch_observer_for_cali' in self.debug_cmd:
                    from torch import Tensor
                    self.torchObserver_dict[out](Tensor(activation.astype(np.float32)))
                else:
                    self.min_value[out] = min(np.min(activation), self.min_value[out])
                    self.max_value[out] = max(np.max(activation), self.max_value[out])
                    abs_value = max(abs(self.min_value[out]), abs(self.max_value[out]))
                    if 'use_percentile9999' in self.debug_cmd or 'use_percentile9999' in self.args.cali_method:
                        tmp = activation.flatten()
                        tmp = sort_distr_per(tmp, res_length)
                        self.all_data_test[out] = np.concatenate((self.all_data_test[out],tmp))
                    elif 'use_mse' in self.debug_cmd or 'use_mse' in self.args.cali_method:
                        bit = 8
                        if 'int4' in self.debug_cmd :
                            bit = 4
                        if out in self.last_five_tensors:
                            tmp = np.abs(activation.flatten())
                            tmp = sort_distr(tmp, res_length)
                            self.all_data_test[out][idx * res_length : (idx + 1) * res_length] = tmp
                        if np.abs(np.min(activation) - 0) < 1e-6 :
                            unsigned = 4
                        else:
                            unsigned = 1
                        abs_x = np.abs(activation.flatten())
                        s_n = abs_x.sum() / abs_x[abs_x > 0].size
                        for _ in range(20):
                            s_n_plus_1 = abs_x[abs_x > s_n].sum() / (1 / (4 ** bit) / 3 / unsigned * abs_x[abs_x <= s_n].size + abs_x[abs_x > s_n].size)
                            if np.abs(s_n_plus_1 - s_n) < 1e-6:
                                break
                            s_n = s_n_plus_1
                        if out not in self.mse:
                            self.mse[out] = []
                            self.mse[out].append(s_n)
                        else:
                            self.mse[out].append(s_n)
                    elif 'use_aciq_gauss' in self.debug_cmd or 'use_aciq_gauss' in self.args.cali_method:
                        alpha = 3.92403337
                        if 'int4' in self.debug_cmd:
                            alpha = 2.55913642
                        if out in self.last_five_tensors:
                            tmp = np.abs(activation.flatten())
                            tmp = sort_distr(tmp, res_length)
                            self.all_data_test[out][idx * res_length : (idx + 1) * res_length] = tmp
                        gaussian_const = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) ** 0.5)
                        N = activation.size
                        std = ((np.max(activation) - np.min(activation)) * gaussian_const) / ((2 * math.log(N)) ** 0.5)
                        if out not in self.aciq:
                            self.aciq[out] = []
                            self.aciq[out].append(alpha * std)
                        else:
                            self.aciq[out].append(alpha * std)
                    elif 'use_aciq_laplace' in self.debug_cmd or 'use_aciq_laplace' in self.args.cali_method:
                        beta = 9.89675982
                        if 'int4' in self.debug_cmd :
                            beta = 5.02864014
                        if out in self.last_five_tensors:
                            tmp = np.abs(activation.flatten())
                            tmp = sort_distr(tmp, res_length)
                            self.all_data_test[out][idx * res_length : (idx + 1) * res_length] = tmp
                        b = np.mean(abs(activation-np.mean(activation)))
                        if out not in self.aciq:
                            self.aciq[out] = []
                            self.aciq[out].append(beta * b)
                        else:
                            self.aciq[out].append(beta * b)
                    elif 'use_max' in self.debug_cmd or 'use_max' in self.args.cali_method:
                        self.max_abs_value[out] = max(np.max(np.abs(activation)), self.max_abs_value[out])
                    if self.args.kurtosis_analysis:
                        kurtosis = self.calculate_kurtosis(activation)
                        if out not in self.kurtosis:
                            self.kurtosis[out] = []
                            self.kurtosis[out].append(kurtosis)
                        else:
                            self.kurtosis[out].append(kurtosis)
                    if out not in self.histogram_data_map:
                        hist, width = self.histogram(activation, abs_value, self.histogram_bin_num)
                        self.histogram_data_map[out] = hist
                        self.histogram_width_map[out] = width
                        self.hist_dict[out] = (hist,width, self.min_value[out],self.max_value[out], abs_value)
                    else:
                        self.hist_dict[out]=self.combine_histogram(self.hist_dict[out],activation,self.min_value[out],self.max_value[out],abs_value)
                        self.histogram_data_map[out] = self.hist_dict[out][0]
                        self.histogram_width_map[out] = self.hist_dict[out][1]
        return

    def process_compute_threshold_muti(self, muti_output_tensor):
        if muti_output_tensor == []:
            return
        for i, evaled_op in enumerate(muti_output_tensor):
            self.process_compute_threshold(evaled_op, i, muti_output_tensor)
        return

    def process_compute_threshold(self,evaled_op,i,all_tensors):
        abs_value = None
        outputs = self.parser.get_outputs_by_op_name(evaled_op)
        for out in outputs:
            if out not in all_tensors and out not in self.tensor_list:
                continue
            abs_value = max(abs(self.min_value[out]), abs(self.max_value[out]))
            if out in self.asym_op:
                res = np.sort(self.all_data_test[out])
                res_max = res[-self.res_length_dict[out]:]
                res_min = res[:self.res_length_dict[out]][::-1]
                inter = self.args.input_num * self.size[out] - 1
                idx = int((self.perd[evaled_op]/ 100) * inter)
                ratio = (self.perd[evaled_op]/ 100) * inter - idx
                v_max = res_max[0] + ratio * (res_max[1] - res_max[0]) if self.res_length_dict[out] != 1 else res_max[0]
                v_min = res_min[0] + ratio * (res_min[1] - res_min[0]) if self.res_length_dict[out] != 1 else res_min[0]
                self.min_value[out] = v_min
                self.max_value[out] = v_max
                abs_value = max(abs(v_max),abs(v_min))
            else:
                if 'use_percentile9999' in self.debug_cmd or 'use_percentile9999' in self.args.cali_method:
                    res = np.sort(self.all_data_test[out])
                    res_max = res[-self.res_length_dict[out]:]
                    res_min = res[:self.res_length_dict[out]][::-1]
                    inter = self.args.input_num * self.size[out] - 1
                    idx = int((self.perd[evaled_op]/ 100) * inter)
                    ratio = (self.perd[evaled_op]/ 100) * inter - idx
                    v_max = res_max[0] + ratio * (res_max[1] - res_max[0]) if self.res_length_dict[out] != 1 else res_max[0]
                    v_min = res_min[0] + ratio * (res_min[1] - res_min[0]) if self.res_length_dict[out] != 1 else res_min[0]
                    self.min_value[out] = v_min
                    self.max_value[out] = v_max
                    abs_value = max(abs(v_max),abs(v_min))
                elif 'use_mse' in self.debug_cmd or 'use_mse' in self.args.cali_method:
                    if out in self.last_five_tensors:
                        res = np.sort(self.all_data_test[out])[-self.res_length_dict[out]:]
                        inter = self.args.input_num * self.size[out] - 1
                        idx = int((self.perd[evaled_op]/ 100) * inter)
                        ratio = (self.perd[evaled_op]/ 100) * inter - idx
                        self.last_five_tensors_threshold[out]= res[0] + ratio * (res[1] - res[0]) if self.res_length_dict[out] != 1 else res[0]
                    mean_s_n = sum(self.mse[out])/len(self.mse[out])
                    mean_s_n = min(mean_s_n,abs_value)
                    abs_value = mean_s_n
                elif 'use_aciq_gauss' in self.debug_cmd or 'use_aciq_gauss' in self.args.cali_method:
                    if out in self.last_five_tensors:
                        res = np.sort(self.all_data_test[out])[-self.res_length_dict[out]:]
                        inter = self.args.input_num * self.size[out] - 1
                        idx = int((self.perd[evaled_op]/ 100) * inter)
                        ratio = (self.perd[evaled_op]/ 100) * inter - idx
                        self.last_five_tensors_threshold[out]= res[0] + ratio * (res[1] - res[0]) if self.res_length_dict[out] != 1 else res[0]
                    mean_gauss = sum(self.aciq[out])/len(self.aciq[out])
                    if mean_gauss > abs_value :
                        mean_gauss = abs_value
                    abs_value = mean_gauss
                elif 'use_aciq_laplace' in self.debug_cmd or 'use_aciq_laplace' in self.args.cali_method:
                    if out in self.last_five_tensors:
                        res = np.sort(self.all_data_test[out])[-self.res_length_dict[out]:]
                        inter = self.args.input_num * self.size[out] - 1
                        idx = int((self.perd[evaled_op]/ 100) * inter)
                        ratio = (self.perd[evaled_op]/ 100) * inter - idx
                        self.last_five_tensors_threshold[out]= res[0] + ratio * (res[1] - res[0]) if self.res_length_dict[out] != 1 else res[0]
                    mean_laplace = sum(self.aciq[out])/len(self.aciq[out])
                    if mean_laplace > abs_value :
                        mean_laplace = abs_value
                    abs_value = mean_laplace
                elif 'use_max' in self.debug_cmd or 'use_max' in self.args.cali_method:
                    #t0 = time.time()
                    abs_value = self.max_abs_value[out]
                if abs_value != None and abs_value <= 1e-5:
                    # if op's outputs are all close to zero, change it to 1e-5 for them.
                    self.min_value[out] = -1e-5 if self.min_value[out] < 0 else 0
                    self.max_value[out] = 1e-5
                    abs_value = 1e-5
                    print("WARNING: layer {} is all zeros. Please check the "
                            "input data correctness.".format(out))
                if self.args.kurtosis_analysis:
                    mean_kurtosis = sum(self.kurtosis[out])/len(self.kurtosis[out])
                    #print("op:{} and kurtosis:{}".format(out,mean_kurtosis))
                    if out not in self.kurtosis_result:
                        self.kurtosis_result[out] = []
                        self.kurtosis_result[out].append(mean_kurtosis)
                    else:
                        self.kurtosis_result[out].append(mean_kurtosis)
            self.activations_statistics[out] = (self.min_value[out], self.max_value[out], abs_value)
        return

    def parallel_statistic(self, all_tensors,idx):
        from concurrent.futures import ThreadPoolExecutor,as_completed
        with ThreadPoolExecutor(max_workers=6) as executor:  # You can adjust max_workers as needed
            future_to_tensor = {executor.submit(self.process_statistic, evaled_op, i,idx,all_tensors): evaled_op for i, evaled_op in enumerate(all_tensors)}
        return

    def parallel_compute_threshold(self, all_tensors):
        from concurrent.futures import ThreadPoolExecutor,as_completed
        with ThreadPoolExecutor(max_workers=6) as executor:  # You can adjust max_workers as needed
            future_to_tensor = {executor.submit(self.process_compute_threshold, evaled_op, i,all_tensors): evaled_op for i, evaled_op in enumerate(all_tensors)}
        return

    def activation_collect_and_calc_th_new(self):
        t0 = time.time()
        all_tensors = self.parser.get_op_name_list()

        thresholds_map = {}
        thresholds_map_absmax = {}
        thresholds_map_scale = {}
        thresholds_map_zp = {}
        thresholds_map4 = {}
        thresholds_map_absmax4 = {}
        thresholds_map_scale4 = {}
        thresholds_map_zp4 = {}
        thresholds_out = []
        muti_output_tensor = []
        self.size = {}
        self.hist_dict = {}
        self.res_length_dict = {}
        self.perd = {}
        self.mse = {}
        self.aciq = {}
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
        self.tensor_list = self.get_no_fused_tensors(all_tensors)
        # self.last_five_tensors = all_tensors[-5:][::-1]
        # self.min_value={tensor:float('inf') for tensor in all_tensors}
        # self.max_value={tensor:float('-inf') for tensor in all_tensors}
        # self.max_abs_value={tensor:float('-inf') for tensor in all_tensors}
        # self.all_data_test={tensor:[] for tensor in all_tensors}
        self.last_five_tensors = self.tensor_list[-5:][::-1]
        self.min_value={tensor:float('inf') for tensor in self.tensor_list}
        self.max_value={tensor:float('-inf') for tensor in self.tensor_list}
        self.max_abs_value={tensor:float('-inf') for tensor in self.tensor_list}
        self.all_data_test={tensor:[] for tensor in self.tensor_list}

        for op in all_tensors:
            if len(self.parser.get_outputs_by_op_name(op)) > 1:
                muti_output_tensor.append(op)
            if self.parser.get_pre_op_by_op_name(op) == []:
                self.input_op.append(op)
        if self.args.part_asymmtric:
            self.choose_asym_op(self.tensor_list)
        self.step = (99.999999 - 99.99) / len(all_tensors)
        input_number = [i for i in range(self.args.input_num)]
        pbar = tqdm(input_number, total = self.args.input_num, position = 0, leave = True)
        for idx in range(self.args.input_num):
            pbar.set_description("activation_collect_and_calc_th for sample: {}".format(idx))
            pbar.update(1)
            data = []
            for name in list(self.ref_activations[idx].keys()):
                data.append(self.ref_activations[idx][name][0])
            for k, v in zip(self.module.input_names, data):
                self.module.set_tensor(k, v)
            self.module.invoke()
            self.parallel_statistic(all_tensors,idx)
            self.process_statistic_muti(muti_output_tensor,idx)
        pbar.close()

        self.parallel_compute_threshold(all_tensors)
        self.process_compute_threshold_muti(muti_output_tensor)
        if 'use_torch_observer_for_cali' in self.debug_cmd:
            for i, evaled_op in enumerate(all_tensors):
                outputs = self.parser.get_outputs_by_op_name(evaled_op)
                for out in outputs:
                    if out not in all_tensors:
                        continue
                    qmin, qmax = -128, 127
                    scale, zp = self.torchObserver_dict[out].calculate_qparams()
                    threshold = float(scale * max(-(qmin-zp), (qmax-zp)))
                    threshold = 1e-5 if (threshold <= 1e-5) else threshold  # fix me
                    thresholds_map[out] = threshold
                    thresholds_map_absmax[out] = threshold
                    thresholds_map_scale[out] = scale.numpy()[0]
                    thresholds_map_zp[out] = zp.numpy()[0]
        if 'use_mse' in self.debug_cmd or 'use_aciq_gauss' in self.debug_cmd or 'use_aciq_laplace' in self.debug_cmd or 'use_mse' in self.args.cali_method or 'use_aciq_gauss' in self.args.cali_method or 'use_aciq_laplace' in self.args.cali_method:
            for tensor in self.last_five_tensors:
                if self.last_five_tensors_threshold[tensor] != max(abs(self.activations_statistics[tensor][0]), abs(self.activations_statistics[tensor][1])):
                    break
                #self.last_five_tensors_threshold[tensor] = max(abs(self.activations_statistics[tensor][0]), abs(self.activations_statistics[tensor][1]))
                thresholds_out.append(tensor)
                if self.parser.get_op_type_by_op_name(tensor) == 'top.MatMul' or self.parser.get_op_type_by_op_name(tensor) == 'top.Conv':
                    break
        if 'use_torch_observer_for_cali' not in self.debug_cmd:
            thresholds_map = self.find_threshold(self.histogram_data_map, self.histogram_width_map, 128)
            if 'int4' in self.debug_cmd :
                thresholds_map4 = self.find_threshold(self.histogram_data_map, self.histogram_width_map, 8)
            for k, v in self.activations_statistics.items():
                mi, ma, abs_val = v
                thresholds_map_absmax[k] = abs_val
                if k in self.asym_op:
                    thresholds_map[k] = abs_val
                    continue
                if thresholds_map[k] > abs_val:
                    thresholds_map[k] = abs_val
                    thresholds_map4[k] = abs_val
                if 'use_percentile9999' in self.debug_cmd or 'use_percentile9999' in self.args.cali_method:
                    thresholds_map[k] = abs_val
                    thresholds_map4[k] = abs_val
                elif 'use_mse' in self.debug_cmd or 'use_mse' in self.args.cali_method:
                    # thresholds_map_absmax[k] = max(abs(mi),abs(ma))
                    if k in thresholds_out:
                        thresholds_map[k] = self.last_five_tensors_threshold[k]
                        thresholds_map4[k] = self.last_five_tensors_threshold[k]
                        continue
                    if k in self.input_op:
                        thresholds_map[k] = max(abs(mi),abs(ma))
                        thresholds_map4[k] = max(abs(mi),abs(ma))
                        continue
                    thresholds_map[k] = abs_val
                    thresholds_map4[k] = abs_val
                elif 'use_aciq_gauss' in self.debug_cmd or 'use_aciq_gauss' in self.args.cali_method:
                    thresholds_map_absmax[k] = max(abs(mi),abs(ma))
                    if k in thresholds_out:
                        thresholds_map[k] = self.last_five_tensors_threshold[k]
                        thresholds_map4[k] = self.last_five_tensors_threshold[k]
                        continue
                    if k in self.input_op:
                        thresholds_map[k] = max(abs(mi),abs(ma))
                        thresholds_map4[k] = max(abs(mi),abs(ma))
                        continue
                    thresholds_map[k] = abs_val
                    thresholds_map4[k] = abs_val
                elif 'use_aciq_laplace' in self.debug_cmd or 'use_aciq_laplace' in self.args.cali_method:
                    thresholds_map_absmax[k] = max(abs(mi),abs(ma))
                    if k in thresholds_out:
                        thresholds_map[k] = self.last_five_tensors_threshold[k]
                        thresholds_map4[k] = self.last_five_tensors_threshold[k]
                        continue
                    if k in self.input_op:
                        thresholds_map[k] = max(abs(mi),abs(ma))
                        thresholds_map4[k] = max(abs(mi),abs(ma))
                        continue
                    thresholds_map[k] = abs_val
                    thresholds_map4[k] = abs_val
                elif 'use_max' in self.debug_cmd or 'use_max' in self.args.cali_method:
                    thresholds_map[k] = abs_val
                    thresholds_map4[k] = abs_val
        time2 = time.time()
        return thresholds_map, thresholds_map_absmax, thresholds_map_scale, thresholds_map_zp, thresholds_map4, thresholds_map_absmax4, thresholds_map_scale4, thresholds_map_zp4

    def _process_statistic_muti(self, muti_output_tensor, idx):
        if muti_output_tensor == []:
            return
        for i, evaled_op in enumerate(muti_output_tensor):
            self._process_statistic(evaled_op, i, idx, muti_output_tensor)
        return

    def _process_statistic(self,evaled_op, i,idx,all_tensors):
        per = 99.99 + i * self._step
        self._perd[evaled_op] = per
        outputs = self.parser.get_outputs_by_op_name(evaled_op)
        res_length = 0
        for out in outputs:
            if out not in all_tensors and out not in self._tensor_list:
                continue
            abs_value = None
            activation=self.module.get_tensor(out).copy()
            if activation is None:
                continue
            self._size[out] = activation.size
            res_length = int(self.args.input_num * self._size[out] * (1 - per / 100)) + 1
            self._res_length_dict[out] = res_length
            # parallel statistic minmax,octav,kl+tune,percentile9999
            self._min_value[out] = min(np.min(activation), self._min_value[out])
            self._max_value[out] = max(np.max(activation), self._max_value[out])
            abs_value = max(abs(self._min_value[out]), abs(self._max_value[out]))

            tmp = np.abs(activation.flatten())
            tmp = sort_distr(tmp, res_length)
            self._all_data_test[out][idx * res_length : (idx + 1) * res_length] = tmp

            bit = 8
            if 'int4' in self.debug_cmd :
                bit = 4
            if np.abs(np.min(activation) - 0) < 1e-6 :
                unsigned = 4
            else:
                unsigned = 1
            abs_x = np.abs(activation.flatten())
            s_n = abs_x.sum() / abs_x[abs_x > 0].size
            for _ in range(20):
                s_n_plus_1 = abs_x[abs_x > s_n].sum() / (1 / (4 ** bit) / 3 / unsigned * abs_x[abs_x <= s_n].size + abs_x[abs_x > s_n].size)
                if np.abs(s_n_plus_1 - s_n) < 1e-6:
                    break
                s_n = s_n_plus_1
            if out not in self._mse:
                self._mse[out] = []
                self._mse[out].append(s_n)
            else:
                self._mse[out].append(s_n)

            self._max_abs_value[out] = max(np.max(np.abs(activation)), self._max_abs_value[out])

            if out not in self._histogram_data_map:
                hist, width = self.histogram(activation, abs_value, self.histogram_bin_num)
                self._histogram_data_map[out] = hist
                self._histogram_width_map[out] = width
                self._hist_dict[out] = (hist,width, self._min_value[out],self._max_value[out], abs_value)
            else:
                self._hist_dict[out]=self.combine_histogram(self._hist_dict[out],activation,self._min_value[out],self._max_value[out],abs_value)
                self._histogram_data_map[out] = self._hist_dict[out][0]
                self._histogram_width_map[out] = self._hist_dict[out][1]
        return

    def _process_compute_threshold_muti(self, muti_output_tensor):
        if muti_output_tensor == []:
            return
        for i, evaled_op in enumerate(muti_output_tensor):
            self._process_compute_threshold(evaled_op, i, muti_output_tensor)
        return

    def _process_compute_threshold(self,evaled_op,i,all_tensors):
        abs_value = None
        outputs = self.parser.get_outputs_by_op_name(evaled_op)
        for out in outputs:
            if out not in all_tensors and out not in self._tensor_list:
                continue
            abs_value = max(abs(self._min_value[out]), abs(self._max_value[out]))
            #percentile9999
            res_max = np.sort(self._all_data_test[out])[-self._res_length_dict[out]:]
            res_min = np.sort(self._all_data_test[out])[:-self._res_length_dict[out]][::-1]
            inter = self.args.input_num * self._size[out] - 1
            idx = int((self._perd[evaled_op]/ 100) * inter)
            ratio = (self._perd[evaled_op]/ 100) * inter - idx
            percentile9999_max = res_max[0] + ratio * (res_max[1] - res_max[0]) if self._res_length_dict[out] != 1 else res_max[0]
            percentile9999_min = res_min[0] + (1-ratio) * (res_min[1] - res_min[0]) if self._res_length_dict[out] != 1 else res_min[0]
            percentile9999 = max(abs(percentile9999_max),abs(percentile9999_min))
            if percentile9999 != None and percentile9999 <= 1e-5:
                percentile9999 = 1e-5
            #mse
            if 'use_percentile9999' in self.debug_cmd:
                self._min_value[out] = percentile9999_min
                self._max_value[out] = percentile9999_max

            if out in self._last_five_tensors:
                self._last_five_tensors_threshold[out] = percentile9999
            mean_s_n = sum(self._mse[out])/len(self._mse[out])
            mean_s_n = min(mean_s_n,abs_value)
            mse = mean_s_n
            if mse != None and mse <= 1e-5:
                mse = 1e-5
            #max
            minmax = self._max_abs_value[out]
            if abs_value != None and abs_value <= 1e-5:
                # if op's outputs are all close to zero, change it to 1e-5 for them.
                self._min_value[out] = -1e-5 if self._min_value[out] < 0 else 0
                self._max_value[out] = 1e-5
                abs_value = 1e-5
                print("WARNING: layer {} is all zeros. Please check the "
                        "input data correctness.".format(out))
            self._activations_statistics[out] = (self._min_value[out], self._max_value[out], abs_value, percentile9999, mse)
        return

    def _parallel_statistic(self, all_tensors,idx):
        from concurrent.futures import ThreadPoolExecutor,as_completed
        with ThreadPoolExecutor(max_workers=6) as executor:  # You can adjust max_workers as needed
            future_to_tensor = {executor.submit(self._process_statistic, evaled_op, i,idx,all_tensors): evaled_op for i, evaled_op in enumerate(all_tensors)}
        return

    def _parallel_compute_threshold(self, all_tensors):
        from concurrent.futures import ThreadPoolExecutor,as_completed
        with ThreadPoolExecutor(max_workers=6) as executor:  # You can adjust max_workers as needed
            future_to_tensor = {executor.submit(self._process_compute_threshold, evaled_op, i,all_tensors): evaled_op for i, evaled_op in enumerate(all_tensors)}
        return

    def activation_collect_and_calc_th_parallel(self):
        all_tensors = self.parser.get_op_name_list()

        thresholds_map = {}
        thresholds_map_absmax = {}
        thresholds_map_scale = {}
        thresholds_map_zp = {}
        thresholds_map4 = {}
        thresholds_map_absmax4 = {}
        thresholds_map_scale4 = {}
        thresholds_map_zp4 = {}
        thresholds = {}
        thresholds4 = {}
        thresholds_max = {}
        thresholds_percentile = {}
        thresholds_mse = {}
        muti_output_tensor = []
        self._size = {}
        self._hist_dict = {}
        self._res_length_dict = {}
        self._perd = {}
        self._mse = {}
        self._aciq = {}
        self._input_op = []
        self._last_five_tensors_threshold = {}
        self._histogram_data_map = {}
        self._histogram_width_map = {}
        self._activations_statistics = {}
        self._tensor_list = []
        self._tensor_list = self.get_no_fused_tensors(all_tensors)
        # self._last_five_tensors = all_tensors[-5:][::-1]
        # self._min_value={tensor:float('inf') for tensor in all_tensors}
        # self._max_value={tensor:float('-inf') for tensor in all_tensors}
        # self._max_abs_value={tensor:float('-inf') for tensor in all_tensors}
        # self._all_data_test={tensor:[] for tensor in all_tensors}
        self._last_five_tensors = self._tensor_list[-5:][::-1]
        self._min_value={tensor:float('inf') for tensor in self._tensor_list}
        self._max_value={tensor:float('-inf') for tensor in self._tensor_list}
        self._max_abs_value={tensor:float('-inf') for tensor in self._tensor_list}
        self._all_data_test={tensor:[] for tensor in self._tensor_list}

        for op in all_tensors:
            if len(self.parser.get_outputs_by_op_name(op)) > 1:
                muti_output_tensor.append(op)
            if self.parser.get_pre_op_by_op_name(op) == []:
                self._input_op.append(op)
        self._step = (99.999999 - 99.99) / len(all_tensors)
        input_number = [i for i in range(self.args.input_num)]
        pbar = tqdm(input_number, total = self.args.input_num, position = 0, leave = True)
        for idx in range(self.args.input_num):
            pbar.set_description("activation_collect_and_calc_th for sample: {}".format(idx))
            pbar.update(1)
            data = []
            for name in list(self.ref_activations[idx].keys()):
                data.append(self.ref_activations[idx][name][0])
            for k, v in zip(self.module.input_names, data):
                self.module.set_tensor(k, v)
            self.module.invoke()
            self._parallel_statistic(all_tensors,idx)
            self._process_statistic_muti(muti_output_tensor,idx)
        pbar.close()

        self._parallel_compute_threshold(all_tensors)
        self._process_compute_threshold_muti(muti_output_tensor)
        tensors = []
        for tensor in self._last_five_tensors:
            if self._last_five_tensors_threshold[tensor] != max(abs(self._activations_statistics[tensor][0]), abs(self._activations_statistics[tensor][1])):
                break
            #self._last_five_tensors_threshold[tensor] = max(abs(self._activations_statistics[tensor][0]), abs(self._activations_statistics[tensor][1]))
            # self._activations_statistics[tensor][4] = self._last_five_tensors_threshold[tensor]
            tensors.append(tensor)
            if self.parser.get_op_type_by_op_name(tensor) == 'top.MatMul' or self.parser.get_op_type_by_op_name(tensor) == 'top.Conv':
                break

        thresholds_map = self.find_threshold(self._histogram_data_map, self._histogram_width_map, 128)
        if 'int4' in self.debug_cmd :
            thresholds_map4 = self.find_threshold(self._histogram_data_map, self._histogram_width_map, 8)
        for k, v in self._activations_statistics.items():
            _, _, abs_val, percentile9999, mse = v
            thresholds_map_absmax[k] = abs_val
            if thresholds_map[k] > abs_val:
                thresholds_map[k] = abs_val
                thresholds_map4[k] = abs_val
            thresholds_percentile[k] = percentile9999
            thresholds_max[k] = abs_val
            if k in tensors:
                thresholds_mse[k] = self._last_five_tensors_threshold[k]
            elif k in self._input_op:
                thresholds_mse[k] = abs_val
            else:
                thresholds_mse[k] = mse
        thresholds['KL'] = thresholds_map
        thresholds['MAX'] = thresholds_max
        thresholds['MSE'] = thresholds_mse
        thresholds['Percentile9999'] = thresholds_percentile
        thresholds4['KL'] = thresholds_map4
        thresholds4['MAX'] = thresholds_max
        thresholds4['MSE'] = thresholds_mse
        thresholds4['Percentile9999'] = thresholds_percentile
        return thresholds_map, thresholds_map_absmax, thresholds_map_scale, thresholds_map_zp, thresholds_map4, thresholds_map_absmax4, thresholds_map_scale4, thresholds_map_zp4, thresholds, thresholds4

    def activation_collect_and_calc_th(self):
        t0 = time.time()
        histogram_data_map = {}
        histogram_width_map = {}
        self.activations_statistics = {}
        thresholds_map = {}
        thresholds_map_absmax = {}
        thresholds_map_scale = {}
        thresholds_map_zp = {}
        thresholds_map4 = {}
        thresholds_map_absmax4 = {}
        thresholds_map_scale4 = {}
        thresholds_map_zp4 = {}

        all_tensors = self.parser.get_op_name_list()
        step = (99.999999 - 99.99) / len(all_tensors)
        pbar = tqdm(all_tensors, total=len(all_tensors), position=0, leave=True)
        for i, evaled_op in enumerate(all_tensors):
            pbar.set_description("activation_collect_and_calc_th for op: {}".format(evaled_op))
            pbar.update(1)
            for idx in range(self.args.input_num):
                self.gen_ref_tensor(idx, evaled_op)
            min_value = inf
            max_value = -inf
            abs_value = None
            max_abs_value = -inf
            outputs = self.parser.get_outputs_by_op_name(evaled_op)
            for out in outputs:
                tensor = self.get_ref_tensor(0, out)
                if tensor is None:
                    continue
                tensor_size = (self.get_ref_tensor(0, out)).size
                # all_data = np.zeros(tensor_size * self.args.input_num, dtype = np.float32)
                num = self.args.input_num
                per = 99.99 + i * step
                res_length = int(num * tensor_size * (1 - per / 100)) + 1
                all_data_test = np.zeros(self.args.input_num * res_length)

                for idx in range(self.args.input_num):
                    activation = self.get_ref_tensor(idx, out)
                    if activation is None:
                        continue
                    if 'use_torch_observer_for_cali' in self.debug_cmd:
                        from torch import Tensor
                        self.torchObserver_dict[out](Tensor(activation.astype(np.float32)))
                    else:
                        min_value = min(np.min(activation), min_value)
                        max_value = max(np.max(activation), max_value)
                        abs_value = max(abs(min_value), abs(max_value))
                        if 'use_percentile9999' in self.debug_cmd:
                            # all_data[idx * tensor_size : (idx + 1) * tensor_size] = activation.flatten()
                            tmp = np.abs(activation.flatten())
                            tmp = sort_distr(tmp, res_length)
                            all_data_test[idx * res_length : (idx + 1) * res_length] = tmp
                        elif 'use_max' in self.debug_cmd:
                            max_abs_value = max(np.max(np.abs(activation)), max_abs_value)

                if 'use_percentile9999' in self.debug_cmd:
                    # t0 = time.time()
                    # time1 = time.time()
                    # abs_value = np.percentile(np.abs(all_data), 99.99 + i * step)
                    # time2 = time.time()
                    res = np.sort(all_data_test)[-res_length:]
                    inter = num * tensor_size - 1
                    idx = int((per / 100) * inter)
                    ratio = (per / 100) * inter - idx
                    abs_value = res[0] + ratio * (res[1] - res[0]) if res_length != 1 else res[0]
                    # time3 = time.time()
                    # print(abs_value)
                    # print(abs_value_test)
                    # print("并行方法时间： {}s".format(time3 - time2))
                    # print("numpy percentile方法时间： {}s".format(time2 - time1))
                elif 'use_max' in self.debug_cmd:
                    #t0 = time.time()
                    abs_value = max_abs_value
                if abs_value != None and abs_value <= 1e-5:
                    # if op's outputs are all close to zero, change it to 1e-5 for them.
                    min_value = -1e-5 if min_value < 0 else 0
                    max_value = 1e-5
                    abs_value = 1e-5
                    print("WARNING: layer {} is all zeros. Please check the "
                          "input data correctness.".format(out))
                self.activations_statistics[out] = (min_value, max_value, abs_value)

                if 'use_torch_observer_for_cali' not in self.debug_cmd:
                    for idx in range(self.args.input_num):
                        activation = self.get_ref_tensor(idx, out)
                        _, _, abs_value = self.activations_statistics[out]
                        hist, width = self.histogram(activation, abs_value, self.histogram_bin_num)
                        if out not in histogram_data_map:
                            histogram_data_map[out] = hist
                            histogram_width_map[out] = width
                        else:
                            histogram_data_map[out] += hist
                else:
                    qmin, qmax = -128, 127
                    scale, zp = self.torchObserver_dict[out].calculate_qparams()
                    threshold = float(scale * max(-(qmin-zp), (qmax-zp)))
                    threshold = 1e-5 if (threshold <= 1e-5) else threshold  # fix me
                    thresholds_map[out] = threshold
                    thresholds_map_absmax[out] = threshold
                    thresholds_map_scale[out] = scale.numpy()[0]
                    thresholds_map_zp[out] = zp.numpy()[0]

                for idx in range(self.args.input_num):
                    self.clear_ref_tensor(idx, out)
        pbar.close()
        if 'use_torch_observer_for_cali' not in self.debug_cmd:
            thresholds_map = self.find_threshold(histogram_data_map, histogram_width_map, 128)
            thresholds_map4 = self.find_threshold(histogram_data_map, histogram_width_map, 8)
            for k, v in self.activations_statistics.items():
                _, _, abs_val = v
                thresholds_map_absmax[k] = abs_val
                if thresholds_map[k] > abs_val:
                    thresholds_map[k] = abs_val
                    thresholds_map4[k] = abs_val
                if 'use_percentile9999' in self.debug_cmd:
                    thresholds_map[k] = abs_val
                    thresholds_map4[k] = abs_val
                elif 'use_max' in self.debug_cmd:
                    thresholds_map[k] = abs_val
                    thresholds_map4[k] = abs_val
        t1 = time.time()
        return thresholds_map, thresholds_map_absmax, thresholds_map_scale, thresholds_map_zp, thresholds_map4, thresholds_map_absmax4, thresholds_map_scale4, thresholds_map_zp4

    def run(self):
        layer_name_list = []
        thresholds_map_list = []
        op_layers = self.parser.get_op_name_list()
        if 'input_calibration_table' in self.debug_cmd:
            assert self.args.tune_num > 0
            input_calibration_table = self.debug_cmd['input_calibration_table']
            if input_calibration_table != '' and os.path.exists(input_calibration_table):
                os.system('cp -f {name} {name}.1'.format(name=input_calibration_table))
                threshold_table = CalibrationTable(input_calibration_table)
                for op_name in op_layers:
                    op_name = split_fuseop(op_name)
                    thresholds_map_list.append(threshold_table.thresholds_map[op_name][0])
            else:
                print('input_calibration_table error')
                exit(1)
        else:
            thresholds_map, thresholds_map_absmax, thresholds_map_scale, thresholds_map_zp, thresholds_map4, thresholds_map_absmax4, thresholds_map_scale4, thresholds_map_zp4 = self.activation_collect_and_calc_th_new()
            self._clean_resource()
            # step 3: dump threshold table of default histogram bins
            cali_table = self.args.calibration_table
            if self.args.tune_num > 0:
                cali_table += ".1"
            with open(cali_table, 'w') as f:
                f.write("# mlir version: {}\n".format(pymlir.__version__))
                f.write("# mlir: {}\n".format(self.args.mlir_file))
                f.write("# genetated time: {}\n".format(datetime.datetime.now()))
                if 'fp8' in self.debug_cmd:
                    f.write("#tpu-mlir-fp8 caliration table\n")
                else:
                    f.write("# histogram number: {}\n".format(self.histogram_bin_num))
                f.write("# sample number: {}\n###\n".format(self.num_samples))
                f.write("# op_name    threshold    min    max\n")
                for i, op_name in enumerate(op_layers):
                    outputs = self.parser.get_outputs_by_op_name(op_name)
                    for out in outputs:
                        if out not in thresholds_map:
                            continue   # possible useless leaf output
                        if 'fp8' in self.debug_cmd:
                            threshold = thresholds_map[out]
                            min_value, max_value = -threshold*128.0/127.0, threshold
                            thresholds_map_list.append(threshold)
                            f.write("{} {:.7f} {:.7f} {:.7f}\n".format(out, threshold, min_value,
                                                               max_value))
                        else:
                            if 'use_torch_observer_for_cali' in self.debug_cmd:
                                qmin, qmax = -128, 127
                                scale = thresholds_map_scale[out]
                                zp = thresholds_map_zp[out]
                                threshold = float(scale * max(-(qmin-zp), qmax-zp))
                                min_value = float(scale * (qmin - zp))
                                max_value = float(scale * (qmax - zp))
                            else:
                                if out in thresholds_map:
                                    threshold = thresholds_map[out]
                                else:
                                    threshold = 1.0
                                if out in self.activations_statistics:
                                    min_value, max_value, _ = self.activations_statistics[out]
                                else:
                                    min_value, max_value = -1,1
                            thresholds_map_list.append(threshold)
                            f.write("{} {:.7f} {:.7f} {:.7f}\n".format(out, threshold, min_value,
                                                                   max_value))
                if 'int4' in self.debug_cmd:
                    f.write("\n")
                    f.write("#int4_th\n")
                    for i, op_name in enumerate(op_layers):
                        outputs = self.parser.get_outputs_by_op_name(op_name)
                        for out in outputs:
                            if out not in thresholds_map:
                                continue
                            threshold = thresholds_map4[out]
                            min_value, max_value = -threshold*8.0/7.0, threshold
                            f.write("{} {:.7f} {:.7f} {:.7f}\n".format(out, threshold, min_value,
                                                                    max_value))

            # if 'use_torch_observer_for_cali' in self.debug_cmd:
            #     exit(0)
        if self.args.tune_num <= 0 or 'int4' in self.debug_cmd or 'use_percentile9999' in self.debug_cmd:
            os.rename(cali_table, self.args.calibration_table)
            return
        # setp 4: tune to get better threshold of each layers.
        self.tunner = SimpleTuner(self.args, self.tune_ds, self.ppa_list, thresholds_map_absmax)
        thresholds = self.tunner.run()

        # step 5: dump threshold table after tuning
        tuned_threshold_list = []
        with open(self.args.calibration_table, 'w') as f:
            f.write("# mlir version: {}\n".format(pymlir.__version__))
            f.write("# mlir: {}\n".format(self.args.mlir_file))
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# histogram number: {}\n".format(self.histogram_bin_num))
            f.write("# sample number: {}\n".format(self.num_samples))
            f.write("# tune number: {}\n###\n".format(self.args.tune_num))
            f.write("# op_name    threshold    min    max\n")
            for i, op_name in enumerate(op_layers):
                op_name = split_fuseop(op_name)
                threshold = thresholds[op_name]
                if threshold <= 1e-5 or np.isnan(threshold):
                    threshold = 1e-5
                    print("WARNING: layer {} threshold is zero. Please check the "
                        "input data correctness.".format(op_name))
                layer_name_list.append('{}_{}'.format(i, op_name))
                tuned_threshold_list.append(threshold)
                if 'input_calibration_table' in self.debug_cmd:
                    min_value = threshold_table.thresholds_map[op_name][1]
                    max_value = threshold_table.thresholds_map[op_name][2]
                else:
                    if 'use_torch_observer_for_cali' in self.debug_cmd:
                        qmin, qmax = -128, 127
                        scale = thresholds_map_scale[op_name]
                        zp = thresholds_map_zp[op_name]
                        min_value = float(scale * (qmin - zp))
                        max_value = float(scale * (qmax - zp))
                    else:
                        min_value, max_value, _ = self.activations_statistics[op_name]
                f.write("{} {:.7f} {:.7f} {:.7f}\n".format(op_name, threshold, min_value,
                                                           max_value))
            if self.args.part_asymmtric:
                f.write("\n")
                f.write("#asym_op\n")
                for i, op_name in enumerate(self.asym_op1):
                    f.write("{}\n".format(op_name))
        os.remove(cali_table)
        if 'print_debug_info' in self.debug_cmd:
            th_before_tuned = np.array(thresholds_map_list)
            th_after_tuned = np.array(tuned_threshold_list)
            file_prefix = './{}_{}pics_{}_times_tuned_th_statistic'.format(
                self.args.mlir_file.split('.')[0], self.tunner.args.tune_num,
                self.tunner.tune_steps)
            save_tensor_diff_subplot(th_before_tuned, th_after_tuned, layer_name_list,
                                     'before_tuned', 'after_tuned', file_prefix)
        if self.args.kurtosis_analysis:
            if '/' in self.args.calibration_table:
                last_index = self.args.calibration_table.rfind('/')
                kurtosis_result = self.args.calibration_table[:last_index + 1] + "kurtosis_analysis"
            else:
                kurtosis_result = "kurtosis_analysis"
            # 按照值降序排序
            sorted_kurtosis_result = dict(sorted(self.kurtosis_result.items(), key=lambda item: item[1], reverse=True))
            with open(kurtosis_result, "w") as f:
                f.write("# op_name   kurtosis   op_type\n")
                for layer,kurtosis in sorted_kurtosis_result.items():
                    op_type =  self.parser.get_op_type_by_op_name(layer)
                    if op_type in ['top.LayerNorm','top.Softmax']:
                        pass
                    else:
                        f.write("{}  {:.7f}  {}\n".format(layer, kurtosis[0], op_type))
