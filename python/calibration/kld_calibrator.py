#!/usr/bin/env python3
##
# Copyright (C) Cristal Vision Technologies Inc.
# All Rights Reserved.
##

import numpy as np
import pymlir
from ctypes import *
from tqdm import tqdm
import datetime


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


class ThresholdTable:

    def __init__(self, candidate_threshold_map, activations_statistics):
        self.candidate_threshold_map = self.transform_threshold_map(candidate_threshold_map)
        self.activations_statistics = activations_statistics
        self.thresholds_map = {}
        for k, v in self.candidate_threshold_map.items():
            self.thresholds_map[k] = v[0]

    def transform_threshold_map(self, candidate_threshold_map):
        threshold_list_map = {}
        for k, _map in candidate_threshold_map.items():
            for op, threshold in _map.items():
                if op not in threshold_list_map:
                    threshold_list_map[op] = [threshold]
                else:
                    threshold_list_map[op].append(threshold)
        return threshold_list_map

    def update_to(self, dest_table, target_op, new_threshold):
        with open(dest_table, "w") as f:
            for k, v in self.thresholds_map.items():
                _min, _max, _ = self.activations_statistics[k]
                if k == target_op:
                    threshold = new_threshold
                else:
                    threshold = v
                f.write("{} {:.7f} {:.7f} {:.7f}\n".format(k, threshold, _min, _max))

    def update(self, target_op, best_threshold):
        self.thresholds_map[target_op] = best_threshold

    def candidate_thresholds(self, target_op):
        return self.candidate_threshold_map[target_op]


class ActivationCalibrator(BaseKldCalibrator):

    def __init__(self, mlir_file: str, data_list: list, histogram_bin_num: int):
        super().__init__()
        self.data_list = data_list
        self.num_samples = len(data_list)

        self.module = pymlir.module()
        self.module.load(mlir_file)
        self.fp32_mlir = mlir_file

        self.tensor_max = {}
        self.tensor_min = {}
        self.histogram_bin_num = histogram_bin_num
        self.buffered_activations = dict()

    def _activations_size(self, tensors):
        size = 0
        for _, v in tensors.items():
            size += v.size
        return size * 4

    def _activations_generator(self):
        for data in self.data_list:
            if data.endswith('.npz'):
                x = np.load(data)
                for k, v in x.items():
                    self.module.set_tensor(k, v)
                self.module.invoke()
            else:
                inputs = data.split(',')
                inputs = [s.strip() for s in inputs]
                assert (len(self.input_names) == len(inputs))
                for name, input in zip(self.module.input_names, inputs):
                    assert (input.endswith('.npy'))
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

    def run(self, output_calibration_table):
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
        with open(output_calibration_table, 'w') as f:
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# histogram number: {}\n".format(self.histogram_bin_num))
            f.write("# sample number: {}\n###\n".format(self.num_samples))
            f.write("# op_name    threshold    min    max\n")
            for op_name in op_layers:
                threshold = thresholds_map[self.histogram_bin_num][op_name]
                min_value, max_value, _ = activations_statistics[op_name]
                f.write("{} {:.7f} {:.7f} {:.7f}\n".format(op_name, threshold, min_value,
                                                           max_value))
