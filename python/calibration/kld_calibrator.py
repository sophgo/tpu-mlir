#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
#
# Licensed under the Apache License v2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ==============================================================================

import os
import numpy as np
import pymlir
from ctypes import *
from tqdm import tqdm
import datetime
from data.preprocess import preprocess
from utils.mlir_parser import *
import pdb
from utils.log_setting import setup_logger

have_weight_op = [
    'top.Conv',
    'top.MatMul'
]

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
                    print(
                        "Table format should be 'op_name, threshold, min, max'")
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

    def update(self, target_op, new_threshold):
        _, _min, _max = self.thresholds_map[target_op]
        self.thresholds_map[target_op] = [new_threshold, _min, _max]

    def update_to(self, dest_table, target_op, new_threshold):
        with open(dest_table, "w") as f:
            for line in self.headers:
                f.write(line + "\n")
            for k, v in self.thresholds_map.items():
                if k == target_op:
                    f.write("{} {:.7f} {:.7f} {:.7f}\n".format(
                            k, new_threshold, v[1], v[2]))
                else:
                    f.write("{} {:.7f} {:.7f} {:.7f}\n".format(k, *v))

def read_weight(weight_file):
  weight_dict = {}
  data = np.load(weight_file)
  for key in data.files:
    weight_dict[key] = data[key]
  return weight_dict

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

class SimpleTuner:
    def __init__(self, fp32_mlir, calibration_table,
                 images, image_num, ppa_list, debug_cmd):
        image_num = min(len(images), image_num)
        self.tune_iteration = 4
        self.threshold_update_factor = 0.01
        self.images = images[:image_num]
        self.tuned_op = []
        self.fp32_mlir = fp32_mlir
        self.ppa_list = ppa_list
        self.threshold_table = CalibrationTable(calibration_table)
        self.module = pymlir.module()
        self.module.load(fp32_mlir)
        self.module_parsered = MlirParser(fp32_mlir)
        self.num_inputs = self.module_parsered.get_input_num()
        weight_file = self.module_parsered.attrs['module.weight_file'][1:-1]
        self.weight = read_weight(weight_file)
        self.debug_cmd = debug_cmd
        log_level="DEBUG" if debug_cmd is not None and 'debug_log' in debug_cmd else "INFO"
        self.logger = setup_logger('auto_tune', log_level = log_level)

        data_dict = {}
        for op_name in self.module.all_tensor_names:
            op_type = self.module_parsered.get_op_type_by_op_name(op_name)
            if op_type not in have_weight_op:
                continue
            if op_name not in self.threshold_table.thresholds_map:
                continue
            input = self.module_parsered.get_pre_op_by_op_name(op_name)
            assert len(input) == 1
            threshold = self.threshold_table.thresholds_map[input[0]][0]
            opds = self.module_parsered.get_opds_by_op_name(op_name)
            data_dict[opds[1]], data_dict[opds[2]] = self.weight_fake_quant(threshold, opds, op_type)

        os.system('rm -f {}'.format(weight_file))
        np.savez(weight_file, **data_dict)
        self.module_dq = pymlir.module()
        self.module_dq.load(fp32_mlir)
        self.load_net_input()

    def print_dbg(self, *para):
        tmp = [str(item) for item in para]
        tmpStr = ' '.join(tmp)
        self.logger.debug(tmpStr)

    def print_info(self, *para):
        tmp = [str(item) for item in para]
        tmpStr = ' '.join(tmp)
        self.logger.info(tmpStr)

    def weight_fake_quant(self, threshold, opds, op_type):
        bias = None
        if op_type == 'top.Conv':
            weight = self.weight[opds[1]]
            max_per_chan = np.max(weight.reshape(weight.shape[0], -1), axis = 1)
            scale = 127/max_per_chan
            weight = np.round(weight*scale[:,None,None,None])
            weight /= scale[:,None,None,None]
            if len(opds) > 2:
                bias = self.weight[opds[2]]
                bias = np.round(bias*scale*127/threshold)
                bias = bias*threshold/scale/127
        elif op_type == 'top.MatMul':
            weight = self.weight[opds[1]]
            max = np.max(weight.reshape(-1))
            scale = 127/max
            weight = np.round(weight*scale)
            weight /= scale
            if len(opds) > 2:
                bias = self.weight[opds[2]]
                bias = np.round(bias*scale*127/threshold)
                bias = bias*threshold/scale/127
        return weight, bias

    def load_net_input(self):
        self.dq_activations = {}
        self.ref_activations = {}
        count = self.module_parsered.get_user_count_by_op_name(self.ppa_list[0].input_name)
        for i, image in enumerate(self.images):
            if is_npy(image):
                x = np.load(image)
            else:
                x = self.ppa_list[0].run(image)
            self.dq_activations[i] = {self.ppa_list[0].input_name:[x, count]}
            self.ref_activations[i] = {self.ppa_list[0].input_name:[x, count]}

    def get_input_tensor(self, i, op_name):
        if op_name in self.dq_activations[i]:
            return self.dq_activations[i][op_name][0]

    def clear_input_tensor(self, i, op_name):
        input_ops = self.module_parsered.get_pre_op_by_op_name(op_name)
        for input_op in input_ops:
            if input_op in self.dq_activations[i]:
                if self.dq_activations[i][input_op][1] == 1:
                    self.dq_activations[i].pop(input_op)
                else:
                    self.dq_activations[i][input_op][1] -= 1

    def gen_input_tensor(self, i, op_name):
        self.print_dbg('gen {}th pic, {}\'s input_tensor'.format(i, op_name))
        if op_name in self.dq_activations[i]:
            self.print_dbg('exist, return')
            return

        input_ops = self.module_parsered.get_pre_op_by_op_name(op_name)
        if len(input_ops) == 0:
            self.print_dbg('no input_ops')
        threshold = None
        for input_op in input_ops:
            data = self.get_input_tensor(i, input_op)
            threshold = self.threshold_table.thresholds_map[input_op][0]
            data = import_quant_bias(data, threshold)
            self.print_dbg('input is {}, tuned_th:{}'.format(input_op, threshold))
            self.module_dq.set_tensor(input_op, data)

        if len(input_ops) > 0:
            self.bias_fake_quant(op_name, threshold)
            value = self.module_dq.invoke_at(op_name)
            count = self.module_parsered.get_user_count_by_op_name(op_name)
            self.dq_activations[i][op_name] = [value, count]
            self.print_dbg('module_dq.invoke_at({}), ret shape:'.format(op_name), value.shape, ' refcount:', count)

    def gen_ref_tensor(self, i, op_name):
        if op_name in self.ref_activations[i]:
            return

        self.print_dbg('gen {}th pic, {}\'s ref_tensor'.format(i, op_name))
        input_ops = self.module_parsered.get_pre_op_by_op_name(op_name)
        if len(input_ops) == 0:
            self.print_dbg('no input_ops')
        for input_op in input_ops:
            data = self.ref_activations[i][input_op][0]
            if self.ref_activations[i][input_op][1] == 1:
                self.ref_activations[i].pop(input_op)
                self.print_dbg('pop its input:{}'.format(input_op))
            else:
                self.ref_activations[i][input_op][1] -= 1
                tmp = self.ref_activations[i][input_op][1]
                self.print_dbg('dec {}\'s refcount:{} > {}'.format(input_op, tmp+1, tmp))
            self.module.set_tensor(input_op, data)
        if len(input_ops) > 0:
            value = self.module.invoke_at(op_name)
            count = self.module_parsered.get_user_count_by_op_name(op_name)
            self.print_dbg('have {} users as refcount:'.format(count))
            self.ref_activations[i][op_name] = [value, count]
            self.print_dbg('module.invoke_at({}), ret shape:'.format(op_name), value.shape)

    def bias_fake_quant(self, op_name, threshold):
        op_type = self.module_parsered.get_op_type_by_op_name(op_name)
        if op_type in have_weight_op:
            opds = self.module_parsered.get_opds_by_op_name(op_name)
            _, bias = self.weight_fake_quant(threshold, opds, op_type)
            if bias is not None:
                self.module_dq.set_tensor(opds[2], bias)
    def calc_distance(self, op_name, input, threshold):
        distance = 0
        self.bias_fake_quant(op_name, threshold)
        for idx in range(len(self.images)):
            for input in self.module_parsered.get_pre_op_by_op_name(op_name):
                value = self.get_input_tensor(idx, input)
                value = import_quant_bias(value, threshold)
                self.module_dq.set_tensor(input, value)
            target_activations = self.module_dq.invoke_at(op_name)
            target_fp32_activations = self.ref_activations[idx][op_name][0]
            diff = target_fp32_activations.flatten() - target_activations.flatten()
            norm_2 = np.linalg.norm(diff)
            norm_1 = np.linalg.norm(target_fp32_activations.flatten(), ord=1)
            distance += norm_2 / norm_1
        return distance

    def find_better_threshold(self, op_name, input):
        try_cnt = 0
        fail_cnt = 0
        prev_distance = -1
        threshold = self.threshold_table.thresholds_map[input][0]
        abs_max = max(map(abs,self.threshold_table.thresholds_map[input][1:]))
        self.print_dbg('>>>op_name:', op_name,', input:',input, ', threshold:', threshold, 'abs_max:', abs_max)
        cur_threshold = threshold
        best_threshold = threshold

        for idx in range(len(self.images)):
            self.gen_ref_tensor(idx, op_name)
            for pre_op in self.module_parsered.get_pre_op_by_op_name(op_name):
                self.gen_input_tensor(idx, pre_op)

        delta_sum = 0
        while fail_cnt < self.tune_iteration:
            delta_sum += cur_threshold * self.threshold_update_factor
            delta = delta_sum / (try_cnt + 1)
            cur_threshold += delta
            if cur_threshold > abs_max:
                self.print_dbg('meet cur_threshold > abs_max, find_better_threshold break')
                break
            try_cnt += 1

            cur_distance = self.calc_distance(op_name, input, cur_threshold)
            # try:
            #     cur_distance = self.calc_distance(target_op, op_name, cur_threshold)
            # except Exception as e:
            #     self.print_dbg("warning but continue: {}".format(e))
            #     fail_cnt += 1
            #     continue

            self.print_dbg("cur[{}] threshold: {:5f}, distance: {:5f}".format(
                        try_cnt, cur_threshold, cur_distance))

            if prev_distance == -1:
                prev_distance = cur_distance
                prev_threshold = cur_threshold
            elif cur_distance < prev_distance:
                self.print_dbg("### tuning {}, find a better threshold:"
                            "{:5f} -> {:5f}, distance: {:9f} -> {:9f}".format(
                            op_name, prev_threshold, cur_threshold,
                            prev_distance, cur_distance))
                best_threshold = cur_threshold
                prev_distance = cur_distance
                prev_threshold = cur_threshold
                #fail_cnt = 0 #若成功时是否清零失败次数，即连续4次失败才退出循环
            else:
                fail_cnt += 1

        for idx in range(len(self.images)):
            for pre_op in self.module_parsered.get_pre_op_by_op_name(op_name):
                self.clear_input_tensor(idx, pre_op)

        self.threshold_table.update(input, best_threshold)

    def isAllInputTuned(self, op_name):
        pre_ops = self.module_parsered.get_pre_op_by_op_name(op_name)
        for pre_op in pre_ops:
            if pre_op not in self.tuned_op:
                return False
        return True

    def run(self):
        #pdb.set_trace()
        for op_name in self.module.all_tensor_names:
            if self.module_parsered.get_op_type_by_op_name(op_name) == 'top.Input':
                continue

            pre_ops = self.module_parsered.get_pre_op_by_op_name(op_name)
            #若op的多个输入都已调节过，那任挑其中1个来调节，暂定第1个
            if self.isAllInputTuned(op_name):
                self.find_better_threshold(op_name, pre_ops[0])
                self.tuned_op.append(pre_ops[0])
                continue
            for pre_op in pre_ops:
                #op的多个输入，调节那些没有调节过的；
                if pre_op not in self.tuned_op:
                    self.find_better_threshold(op_name, pre_op)
                    self.tuned_op.append(pre_op)
            self.print_dbg(op_name, 'dq keys:', list(self.dq_activations[0].keys()))
            for i in self.dq_activations[0]:
                self.print_dbg('key', i, ' count:', self.dq_activations[0][i][1])
            self.print_dbg(op_name, 'ref keys:', list(self.ref_activations[0].keys()))
            for i in self.ref_activations[0]:
                self.print_dbg('key', i, ' count:', self.ref_activations[0][i][1])

        print('auto tune end')
        return self.threshold_table.thresholds_map


class ActivationCalibrator(BaseKldCalibrator):
    def __init__(self, args, data_list: list):
        super().__init__()
        self.module = pymlir.module()
        self.module.load(args.mlir_file)
        self.fp32_mlir = args.mlir_file
        self.tune_num = args.tune_num
        self.args = args

        self.ppa_list = []
        self.module_parsered = MlirParser(args.mlir_file)
        self.batch_size = self.module_parsered.get_batch_size()
        self.input_num = self.module_parsered.get_input_num()
        for i in range(self.input_num):
            tmp = preprocess()
            tmp.load_config(self.module_parsered.get_input_op_by_idx(i))
            self.ppa_list.append(tmp)

        n = len(data_list) % self.batch_size
        if n != 0:
            for i in range(n):
                data_list.append(data_list[-1])
        self.data_list = data_list
        self.num_samples = len(data_list)

        self.tensor_max = {}
        self.tensor_min = {}
        self.histogram_bin_num = args.histogram_bin_num
        self.buffered_activations = dict()

    def _activations_size(self, tensors):
        size = 0
        for _, v in tensors.items():
            size += v.size
        return size * 4

    def _activations_generator(self):
        idx = 0
        batched_inputs = self.input_num*['']
        for data in self.data_list:
            if data.lower().endswith('.npz'):
                x = np.load(data)
                for k, v in x.items():
                    self.module.set_tensor(k, v)
                self.module.invoke()
            elif data.lower().endswith('.jpg') or data.lower().endswith('.jpeg'):
                inputs = data.split(',')
                inputs = [s.strip() for s in inputs]
                assert(self.input_num == len(inputs))
                idx += 1
                for i in range(self.input_num):
                    batched_inputs[i] += '{},'.format(inputs[i])
                    if idx == self.batch_size:
                        x = self.ppa_list[i].run(batched_inputs[i][:-1])
                        self.module.set_tensor(self.ppa_list[i].input_name, x)
                if idx == self.batch_size:
                    self.module.invoke()
                    idx = 0
                    batched_inputs = self.input_num*['']
            else:
                inputs = data.split(',')
                inputs = [s.strip() for s in inputs]
                assert (self.input_num == len(inputs))
                for name, input in zip(self.module.input_names, inputs):
                    assert (input.lower().endswith('.npy'))
                    x = np.load(input)
                    self.module.set_tensor(name, x)
                self.module.invoke()
            activations = self.module.get_all_tensor()
            self.buffered_activations[data] = activations

    def _clean_resource(self):
        del self.buffered_activations
        self.buffered_activations = None
        del self.module
        self.module = None

    def find_min_max_abs(self):
        activations_statistics = dict()

        pbar = tqdm(self.data_list, total=self.num_samples, position=0, leave=True)
        for file, activations in self.buffered_activations.items():
            file_name = file.split("/")[-1]
            pbar.set_description("Find Min Max *{}".format(file_name))
            pbar.update(1)

            for op_name, activation in activations.items():
                if op_name not in activations_statistics:
                    minimum, maximum = np.min(activation), np.max(activation)
                else:
                    minimum, maximum, _ = activations_statistics[op_name]
                min_value = min(np.min(activation), minimum)
                max_value = max(np.max(activation), maximum)
                abs_value = max(abs(min_value), abs(max_value))
                activations_statistics[op_name] = (min_value, max_value, abs_value)

        # check max is zero
        for k, v in activations_statistics.items():
            _, _, _abs = v
            if _abs == 0:
                # if network outputs are all zero, change it to 1e-5 for them.
                activations_statistics[k] = (-1e-5, 1e-5, 1e-5)
                print("WARNING: layer {} is all zeros. Please check the "
                      "input data correctness.".format(k))
        pbar.close()
        return activations_statistics

    def calc_thresholds(self, activations_statistics, hist_bin_nums):
        print("calculate histogram..")

        histogram_data_map = {}
        histogram_width_map = {}
        for bin_num in hist_bin_nums:
            histogram_data_map[bin_num] = {}
            histogram_width_map[bin_num] = {}

        pbar = tqdm(self.data_list, total=self.num_samples, position=0, leave=True)
        for file, activations in self.buffered_activations.items():
            file_name = file.split("/")[-1]
            pbar.set_description("histogram: {}".format(file_name))
            pbar.update(1)

            for op_name, activation in activations.items():
                _, _, abs_value = activations_statistics[op_name]
                for bin_num in histogram_data_map.keys():
                    hist, width = self.histogram(activation, abs_value, bin_num)
                    if op_name not in histogram_data_map[bin_num]:
                        histogram_data_map[bin_num][op_name] = hist
                        histogram_width_map[bin_num][op_name] = width
                    else:
                        histogram_data_map[bin_num][op_name] += hist
            del activations
        pbar.close()

        thresholds_map = {}
        for bin_num in histogram_data_map.keys():
            thresholds_map[bin_num] = self.find_threshold(histogram_data_map[bin_num],
                                                          histogram_width_map[bin_num], bin_num)
        thresholds_map['abs_max'] = {}
        for k, v in activations_statistics.items():
            _, _, abs_val = v
            thresholds_map['abs_max'][k] = abs_val

        return thresholds_map

    def find_threshold(self, histogram_data_map, histogram_width_map, histogram_bin_num):
        thresholds = {}
        num = len(histogram_data_map)
        pbar = tqdm(range(num), total=num, position=0, leave=True)
        for item in histogram_data_map:
            pbar.set_description("[{}] threshold: {}".format(histogram_bin_num, item))
            pbar.update(1)
            thresholds[item] = self.kld_threshold(histogram_data_map[item],
                                                  histogram_width_map[item], histogram_bin_num)
        pbar.close()
        return thresholds

    def run(self):
        # step 1: find min max
        op_layers = self.module.all_tensor_names
        self._activations_generator()
        activations_statistics = self.find_min_max_abs()

        # step 2: set histogram bins
        hist_bin_nums = [(2**i) * 512 for i in range(7)]
        if self.histogram_bin_num not in hist_bin_nums:
            hist_bin_nums.append(self.histogram_bin_num)

        # step 3: calculate threshold with histogram bins
        thresholds_map = self.calc_thresholds(activations_statistics, hist_bin_nums)
        self._clean_resource()

        # step 6: dump threshold table of default histogram bins
        with open(self.args.calibration_table, 'w') as f:
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# histogram number: {}\n".format(self.histogram_bin_num))
            f.write("# sample number: {}\n###\n".format(self.num_samples))
            f.write("# op_name    threshold    min    max\n")
            for op_name in op_layers:
                threshold = thresholds_map[self.histogram_bin_num][op_name]
                min_value, max_value, _ = activations_statistics[op_name]
                f.write("{} {:.7f} {:.7f} {:.7f}\n".format(op_name, threshold, min_value,
                                                        max_value))
        if self.tune_num <= 0:
                return

        # setp 4: tune to get better threshold of each layers.
        self.module = SimpleTuner(self.fp32_mlir, self.args.calibration_table,
                                 self.data_list,  self.tune_num, self.ppa_list, self.args.debug_cmd)
        thresholds = self.module.run()

        # step 5: dump threshold table after tuning
        with open(self.args.calibration_table + '.tuned', 'w') as f:
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# sample number: {}\n###\n".format(self.num_samples))
            f.write("# op_name    threshold    min    max\n")
            for op_name in op_layers:
                threshold = thresholds[op_name]
                min_value, max_value, _ = activations_statistics[op_name]
                f.write("{} {:.7f} {:.7f} {:.7f}\n".format(op_name, threshold[0],
                                                           min_value, max_value))


        # step 7: dump all thresholds to csv files
        with open(self.args.calibration_table + '.tuned.csv', 'w') as f:
            f.write("name,default({}),best,min,max,abs,{}\n".format(self.histogram_bin_num,
                   ','.join([str(x) for x in hist_bin_nums])))
            for op_name in op_layers:
                default_threshold = thresholds_map[self.histogram_bin_num][op_name]
                best_threshold = thresholds[op_name]
                min_value, max_value, abs_value = activations_statistics[op_name]
                f.write("{},{:.7f},{:.7f},{:.7f},{:.7f},{:5f},".format(
                        op_name, default_threshold,
                        best_threshold[0], min_value, max_value, abs_value))
                _str_thres = []
                for x in hist_bin_nums:
                    _str_thres.append('{:.7f}'.format(thresholds_map[x][op_name]))
                f.write(','.join(_str_thres))
                f.write('\n')
