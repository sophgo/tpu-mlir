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
from utils.mlir_parser import *
from utils.log_setting import setup_logger
from utils.misc import *
from math import *
from calibration.data_selector import DataSelector
from .utils import *

from .cali_math import cosine_sim, math_impl, POOL_THREADS

pymlir.set_mem_mode("force_value_mem")

cur_dir_path = os.path.join(os.path.dirname(__file__))
calibration_math_path = os.path.join("/".join(cur_dir_path.split("/")[:-2]),
                                     "lib/calibration_math.so")
if not os.path.exists(calibration_math_path):
    calibration_math_path = "calibration_math.so"


def import_quant_bias(value, threshold):
    scale = 127 / threshold
    value = np.round(value * scale)
    value[value > 127] = 127
    value[value < -128] = -128
    value /= scale
    return value


class SimpleTuner:

    def __init__(self, args, ds: DataSelector, ppa_list, abs_max_dict):
        self.args = copy.deepcopy(args)
        self.start_time = time.time()
        self.abs_max_dict = abs_max_dict
        self.args.tune_num = min(len(ds.data_list), args.tune_num)
        self.tuned_op_list = []
        self.data_list = ds.data_list  # [:self.args.tune_num]
        self.ppa_list = ppa_list
        self.debug_cmd = args.debug_cmd
        self.fuseop_list = {}
        if 'input_calibration_table' in self.debug_cmd:
            self.threshold_table = CalibrationTable(self.debug_cmd['input_calibration_table'] +
                                                    ".1")
        else:
            self.threshold_table = CalibrationTable(args.calibration_table + ".1")
        self.module = pymlir.module()
        self.module.load(args.mlir_file)
        self.parser = MlirParser(args.mlir_file)
        for op_name in self.parser.get_op_name_list():
            fuseop_list_append(op_name, self.fuseop_list)
        self.batch_size = self.parser.get_batch_size()
        self.input_num = self.parser.get_input_num()
        self.ds = ds
        if ds.all_image or ds.all_yuv:
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

        if self.ds.all_image or self.ds.all_yuv:
            batched_inputs = self.input_num * ['']
        else:
            batched_inputs = {}

        idx, tune_idx = 0, 0
        self.dq_activations[tune_idx] = {}
        self.ref_activations[tune_idx] = {}
        only_one = len(self.module.input_names) == 1
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
        if self.ref_activations[i][evaled_op][1] == 0:  # Clear residual network output
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
            self.module_dq.set_tensor(input_op, data_q, data_q.shape)
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

            self.module.set_tensor(input_op, data, data.shape)
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
                self.module_dq.set_tensor(input, value, value.shape)
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
            if 'find_lower_th' in self.debug_cmd:
                r = 2
            for n in range(r):
                if n == 1 and 'find_lower_th' in self.debug_cmd:
                    th_min = best_threshold - step
                    th_max = best_threshold + step
                    diff = abs(th_max - th_min)
                    times = self.tune_steps // 2
                    step = (th_max - th_min) / times
                    ranges = range(times + 1)[1:-1]
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
                    elif cur_distance < prev_distance:  # and cur_cos_sim > best_cos_sim
                        self.print_dbg(
                            "### tuning i:{}, tuned_op_idx:{}, tuned_op:{}, find a better threshold:"
                            "{:5f}".format(i, op_no, tuned_op, cur_threshold))
                        prev_distance = cur_distance
                        best_threshold = cur_threshold
                        best_cos_sim = cur_cos_sim
        else:
            self.print_dbg("### {} step <=0 or diff < min_tuned_diff, {} {} skip!".format(
                evaled_op, diff, min_tuned_diff))

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

        # If the current tune's best_threshold is greater than the best threshold from the previous th tune, save it; otherwise, keep the previous tune's larger result.
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
                    evaled_op = split_fuseop(evaled_op)[0]
                    self.dot.edge(pre_op, evaled_op, label=pre_op)

            for idx in range(self.args.tune_num):
                evaled_op = split_fuseop(evaled_op)[0]
                self.gen_ref_tensor(idx, evaled_op, node_label)

            # If multiple op inputs are adjusted, select any one (temporarily the first) for adjustment.
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
                # Multiple inputs of op, adjust those that haven't been adjusted;
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
