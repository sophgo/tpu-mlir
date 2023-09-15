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
import gc
import time
import copy
import numpy as np
import pymlir
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


class BaseKldCalibrator:

    def __init__(self, math_lib_path='calibration_math.so'):
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

    def kld_threshold(self, hist, width, bin_num):
        threshold = self.calib_lib.kl_diversity_hist(hist.ctypes.data_as(POINTER(c_int)),
                                                     c_float(width), c_longlong(bin_num))
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
        self.tune_steps = 20
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
        for input_op in input_ops:
            data = self.ref_activations[i][input_op][0]
            refcount = self.ref_activations[i][input_op][1]
            if i == 0:
                tmp += '\nits input:{}, refcount:{}'.format(input_op, refcount)

            self.module.set_tensor(input_op, data)
        if len(input_ops) > 0:
            value = self.module.invoke_at(op_name)
            count = self.parser.get_use_count_by_op_name(op_name)
            self.ref_activations[i][op_name] = [value.copy(), count]
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
        op_no = self.parser.get_op_name_list().index(tuned_op)
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
            for n in range(2):
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
                    elif cur_distance < prev_distance:  # and cur_cos_sim > best_cos_sim:
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
                    self.dot.edge(pre_op, evaled_op, label=pre_op)

            for idx in range(self.args.tune_num):
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


class ActivationCalibrator2(BaseKldCalibrator):

    def __init__(self, args, ds: DataSelector, tune_ds: DataSelector):
        super().__init__()
        self.args = copy.deepcopy(args)
        self.start_time = time.time()
        self.tuned_op_list = []
        self.debug_cmd = parse_debug_cmd(args.debug_cmd)
        # if 'input_calibration_table' in self.debug_cmd:
        self.module = pymlir.module()
        self.module.load(args.mlir_file)
        self.torchObserver_dict = {}
        if 'use_torch_observer_for_cali' in self.debug_cmd:
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
        self.module.before_invoke(set_func)
        self.module.after_invoke(get_func)
        if len(self.parser.get_pre_op_by_op_name(op_name)) > 0:
            value = self.module.invoke_at(op_name)
        self.module.clear_hooks()
    def find_threshold(self, histogram_data_map, histogram_width_map):
        thresholds = {}
        num = len(histogram_data_map)
        pbar = tqdm(range(num), total=num, position=0, leave=True)
        for item in histogram_data_map:
            pbar.set_description("[{}] threshold: {}".format(self.histogram_bin_num, item))
            pbar.update(1)
            thresholds[item] = self.kld_threshold(histogram_data_map[item],
                                                  histogram_width_map[item], self.histogram_bin_num)
        pbar.close()
        return thresholds

    def activation_collect_and_calc_th(self):
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
            tensor = self.get_ref_tensor(0, evaled_op)
            if tensor is None:
                continue
            tensor_size = (self.get_ref_tensor(0, evaled_op)).size
            all_data = np.zeros(tensor_size * self.args.input_num, dtype = np.float32)
            num = self.args.input_num
            per = 99.99 + i * step
            res_length = int(num * tensor_size * (1 - per / 100)) + 1
            all_data_test = np.zeros(self.args.input_num * res_length)

            for idx in range(self.args.input_num):
                activation = self.get_ref_tensor(idx, evaled_op)
                if activation is None:
                    continue
                if 'use_torch_observer_for_cali' in self.debug_cmd:
                    from torch import Tensor
                    self.torchObserver_dict[evaled_op](Tensor(activation.astype(np.float32)))
                else:
                    min_value = min(np.min(activation), min_value)
                    max_value = max(np.max(activation), max_value)
                    abs_value = max(abs(min_value), abs(max_value))
                    if 'use_percentile9999' in self.debug_cmd:
                        all_data[idx * tensor_size : (idx + 1) * tensor_size] = activation.flatten()
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
                      "input data correctness.".format(evaled_op))
            self.activations_statistics[evaled_op] = (min_value, max_value, abs_value)

            if 'use_torch_observer_for_cali' not in self.debug_cmd:
                for idx in range(self.args.input_num):
                    activation = self.get_ref_tensor(idx, evaled_op)
                    _, _, abs_value = self.activations_statistics[evaled_op]
                    hist, width = self.histogram(activation, abs_value, self.histogram_bin_num)
                    if evaled_op not in histogram_data_map:
                        histogram_data_map[evaled_op] = hist
                        histogram_width_map[evaled_op] = width
                    else:
                        histogram_data_map[evaled_op] += hist
            else:
                qmin, qmax = -128, 127
                scale, zp = self.torchObserver_dict[evaled_op].calculate_qparams()
                threshold = float(scale * max(-(qmin-zp), (qmax-zp)))
                threshold = 1e-5 if (threshold <= 1e-5) else threshold  # fix me
                thresholds_map[evaled_op] = threshold
                thresholds_map_absmax[evaled_op] = threshold
                thresholds_map_scale[evaled_op] = scale.numpy()[0]
                thresholds_map_zp[evaled_op] = zp.numpy()[0]
                if 'int4' in self.debug_cmd: # give when int4 selected, give symeetric value like in qat, both for int4 and int8
                    qmin, qmax = -128, 127
                    scale, zp = self.torchObserver_dict[evaled_op].calculate_qparams()
                    threshold = float(scale * max(-(qmin-zp), (qmax-zp)))
                    threshold = 1e-5 if (threshold <= 1e-5) else threshold  # fix me
                    thresholds_map[evaled_op] = threshold
                    thresholds_map_absmax[evaled_op] = threshold
                    thresholds_map_scale[evaled_op] = threshold/127.5
                    thresholds_map_zp[evaled_op] = 0
                    qmin, qmax = -8, 7
                    scale, zp = self.torchObserver_dict[evaled_op].calculate_qparams()
                    threshold = float(scale * max(-(qmin-zp), (qmax-zp)))
                    threshold = 1e-5 if (threshold <= 1e-5) else threshold
                    thresholds_map4[evaled_op] = threshold
                    thresholds_map_absmax4[evaled_op] = threshold
                    thresholds_map_scale4[evaled_op] = threshold/127.5
                    thresholds_map_zp4[evaled_op] = 0

            for idx in range(self.args.input_num):
                self.clear_ref_tensor(idx, evaled_op)
        pbar.close()

        if 'use_torch_observer_for_cali' not in self.debug_cmd:
            thresholds_map = self.find_threshold(histogram_data_map, histogram_width_map)
            thresholds_map4 = thresholds_map.copy()
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
                    thresholds_map_list.append(threshold_table.thresholds_map[op_name][0])
            else:
                print('input_calibration_table error')
                exit(1)
        else:
            thresholds_map, thresholds_map_absmax, thresholds_map_scale, thresholds_map_zp, thresholds_map4, thresholds_map_absmax4, thresholds_map_scale4, thresholds_map_zp4 = self.activation_collect_and_calc_th()
            self._clean_resource()
            # step 3: dump threshold table of default histogram bins
            cali_table = self.args.calibration_table
            if self.args.tune_num > 0:
                cali_table += ".1"
            with open(cali_table, 'w') as f:
                f.write("# genetated time: {}\n".format(datetime.datetime.now()))
                f.write("# histogram number: {}\n".format(self.histogram_bin_num))
                f.write("# sample number: {}\n###\n".format(self.num_samples))
                f.write("# op_name    threshold    min    max\n")
                for i, op_name in enumerate(op_layers):
                    if 'int4' in self.debug_cmd:
                        if 'use_torch_observer_for_cali' in self.debug_cmd:
                            qmin, qmax = -128, 127
                            scale = thresholds_map_scale[op_name]
                            zp = thresholds_map_zp[op_name]
                            threshold = float(scale * max(-(qmin-zp), qmax-zp))
                            min_value = -threshold*128.0/127.0
                            max_value = threshold
                        else:
                            threshold = thresholds_map[op_name]
                            min_value, max_value = -threshold*128.0/127.0, threshold
                        thresholds_map_list.append(threshold)
                        f.write("{} {:.7f} {:.7f} {:.7f}\n".format(op_name, threshold, min_value,
                                                               max_value))
                    else:
                        if 'use_torch_observer_for_cali' in self.debug_cmd:
                            qmin, qmax = -128, 127
                            scale = thresholds_map_scale[op_name]
                            zp = thresholds_map_zp[op_name]
                            threshold = float(scale * max(-(qmin-zp), qmax-zp))
                            min_value = float(scale * (qmin - zp))
                            max_value = float(scale * (qmax - zp))
                        else:
                            if op_name in thresholds_map:
                                threshold = thresholds_map[op_name]
                            else:
                                threshold = 1.0
                            if op_name in self.activations_statistics:
                                min_value, max_value, _ = self.activations_statistics[op_name]
                            else:
                                min_value, max_value = -1,1
                        thresholds_map_list.append(threshold)
                        f.write("{} {:.7f} {:.7f} {:.7f}\n".format(op_name, threshold, min_value,
                                                               max_value))
                if 'int4' in self.debug_cmd and ('use_torch_observer_for_cali' in self.debug_cmd or 'use_percentile9999' in self.debug_cmd or 'use_max' in self.debug_cmd):
                    f.write("\n")
                    f.write("#int4_th\n")
                    for i, op_name in enumerate(op_layers):
                        if 'use_torch_observer_for_cali' in self.debug_cmd:
                            qmin, qmax = -8, 7
                            scale = thresholds_map_scale4[op_name]
                            zp = thresholds_map_zp4[op_name]
                            threshold = float(scale * max(-(qmin-zp), qmax-zp))
                            min_value = -threshold*128.0/127.0
                            max_value = threshold
                        else:
                            threshold = thresholds_map4[op_name]
                            min_value, max_value = -threshold*128.0/127.0, threshold
                        f.write("{} {:.7f} {:.7f} {:.7f}\n".format(op_name, threshold, min_value,
                                                                max_value))

            # if 'use_torch_observer_for_cali' in self.debug_cmd:
            #     exit(0)
        if self.args.tune_num <= 0 or 'int4' in self.debug_cmd:
            return

        # setp 4: tune to get better threshold of each layers.
        self.tunner = SimpleTuner(self.args, self.tune_ds, self.ppa_list, thresholds_map_absmax)
        thresholds = self.tunner.run()

        # step 5: dump threshold table after tuning
        tuned_threshold_list = []
        with open(self.args.calibration_table, 'w') as f:
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# histogram number: {}\n".format(self.histogram_bin_num))
            f.write("# sample number: {}\n".format(self.num_samples))
            f.write("# tune number: {}\n###\n".format(self.args.tune_num))
            f.write("# op_name    threshold    min    max\n")
            for i, op_name in enumerate(op_layers):
                threshold = thresholds[op_name]
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
        os.remove(cali_table)
        if 'print_debug_info' in self.debug_cmd:
            th_before_tuned = np.array(thresholds_map_list)
            th_after_tuned = np.array(tuned_threshold_list)
            file_prefix = './{}_{}pics_{}_times_tuned_th_statistic'.format(
                self.args.mlir_file.split('.')[0], self.tunner.args.tune_num,
                self.tunner.tune_steps)
            save_tensor_diff_subplot(th_before_tuned, th_after_tuned, layer_name_list,
                                     'before_tuned', 'after_tuned', file_prefix)

