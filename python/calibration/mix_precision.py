#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import pymlir
import numpy as np
import os
import math
import sys
from tqdm import tqdm
from utils.mlir_shell import mlir_lowering
from utils.mlir_parser import MlirParser
from utils.preprocess import preprocess
from calibration.data_selector import DataSelector

SKIP_OPERATION = ['top.Input', 'top.Reshape', 'top.Softmax']

FLOAT_MAP = {
    "bm1684x": "F16",
    "bm1684": "F32",
    "cv183x": "BF16",
    "cv182x": "BF16",
    "cv181x": "BF16",
}


class MixQuantModel:

    def __init__(self, fp32_mlir, chip: str, calib_table=None, mix_table=None):
        self.fp32_mlir = fp32_mlir
        self.chip = chip
        self.calib_table = None
        self.mix_table = None
        self.stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        if calib_table:
            self.mode = "INT8"
            self.calib_table = calib_table
            self.mix_table = mix_table
        else:
            self.mode = FLOAT_MAP[chip]
        self.quanted_mlir_file = '{}.quanted.tune.mlir'.format(fp32_mlir)
        mlir_lowering(self.fp32_mlir, self.quanted_mlir_file, self.mode, self.chip,
                      self.calib_table, False, self.mix_table)
        self.module = pymlir.module()
        self.module.load(self.quanted_mlir_file)
        parser = MlirParser(self.quanted_mlir_file)
        self.weight_file = parser.module_weight_file

    def __exit__(self):
        try:
            sys.stdout.close()
            sys.stdout = self.stdout
        except:
            pass

    def infer(self, data: list):
        for k, v in zip(self.module.input_names, data):
            self.module.set_tensor(k, v)
        self.module.invoke()
        outputs = {}
        for name in self.module.output_names:
            outputs[name] = self.module.get_tensor(name)
        return outputs

    def clean(self):
        try:
            del self.module
            os.remove(self.quanted_mlir_file)
            os.remove(self.mlir_weight_file)
        except:
            pass


class MixPrecSearcher(object):

    def __init__(self, args):
        self.fp32_mlir = args.mlir_file
        self.calib_table = args.calibration_table
        self.skip_ops = SKIP_OPERATION
        self.chip = args.chip
        self.mix_mode = FLOAT_MAP[self.chip]
        self.parser = MlirParser(args.mlir_file)
        self.batch_size = self.parser.get_batch_size()
        self.input_num = self.parser.get_input_num()
        self.input_data_buffer = []
        self.num_sample = 0
        self._init_inputs(args)

    def _init_inputs(self, args):
        ds = DataSelector(args.dataset, args.input_num, args.data_list)
        ppa_list = []
        if ds.all_image:
            for i in range(self.input_num):
                ppa = preprocess()
                ppa.load_config(self.parser.get_input_op_by_idx(i))
                ppa_list.append(ppa)
            n = len(ds.data_list) % self.batch_size
            if n != 0:
                for i in range(self.batch_size - n):
                    ds.data_list.append(ds.data_list[-1])
            self.num_sample = len(ds.data_list) // self.batch_size
            self.input_data_buffer = [[] for i in range(self.num_sample)]
            batched_idx = 0
            batched_inputs = self.input_num * ['']
            sample_idx = 0
            for data in ds.data_list:
                inputs = data.split(',')
                inputs = [s.strip() for s in inputs]
                assert (len(inputs) == self.input_num)
                batched_idx += 1
                for i in range(self.input_num):
                    batched_inputs[i] += '{},'.format(inputs[i])
                    if batched_idx == self.batch_size:
                        x = ppa_list[i].run(batched_inputs[i][:-1])
                        self.input_data_buffer[sample_idx].append(x)
                if batched_idx == self.batch_size:
                    sample_idx += 1
                    batched_idx = 0
                    batched_inputs = self.input_num * ['']
        elif ds.all_npy:
            self.num_sample = len(ds.data_list)
            self.input_data_buffer = [[] for i in range(self.num_sample)]
            for sample_idx, data in enumerate(ds.data_list):
                inputs = data.split(',')
                inputs = [s.strip() for s in inputs]
                assert (len(inputs) == self.input_num)
                for npy in inputs:
                    x = np.load(npy)
                    self.input_data_buffer[sample_idx].append(x.astype(np.float32))
        elif ds.all_npz:
            self.num_sample = len(ds.data_list)
            self.input_data_buffer = [[] for i in range(self.num_sample)]
            input_names = [op.name for op in self.parser.inputs]
            for data in ds.data_list:
                npz = np.load(data)
                for name in input_names:
                    self.input_data_buffer[sample_idx].append(npz[name].astype(np.float32))
        else:
            raise RuntimeError("dataset is uncorrect")

    def _gen_mix_table(self, target_file, op):
        with open(target_file, 'w') as f:
            f.write("{} {}\n".format(op.name, self.mix_mode))

    def _cal_sqnr(self, signal_raw, signal_dequant, remove_zero=True):
        # SQNR is non-commutative
        # Unlike other distance function
        # Cannot change the order of signal_raw and signal_dequant
        raw = signal_raw.flatten()
        dequant = signal_dequant.flatten()

        if remove_zero is True:
            idx = dequant != 0
            raw = raw[idx]
            dequant = dequant[idx]

        noise = raw - dequant

        avg_raw = np.sum(raw) / raw.size
        avg_noise = np.sum(noise) / noise.size

        raw_zero_mean = raw - avg_raw
        noise_zero_mean = noise - avg_noise

        var_raw_zero_mean = np.sum(np.square(raw_zero_mean))
        var_noise_zero_mean = np.sum(np.square(noise_zero_mean))

        if var_noise_zero_mean == 0.0:
            return math.inf

        sqnr = 10 * np.log10(var_raw_zero_mean / var_noise_zero_mean)

        return sqnr

    def _loss(self, preds, gt_preds):
        ret = 0
        cnt = 0
        for op_name in gt_preds:
            a = gt_preds[op_name]
            b = preds[op_name]
            loss = self._cal_sqnr(a, b)
            if not math.isinf(loss):
                ret += -loss * a.size
                cnt += a.size

        if ret == 0 and cnt == 0:
            return -math.inf
        else:
            return ret / cnt

    def run(self):
        loss_list = list()
        predictions_gt = list()

        # set all layer for float
        bf16_model = MixQuantModel(self.fp32_mlir, self.chip)
        for idx in range(self.num_sample):
            outputs = bf16_model.infer(self.input_data_buffer[idx])
            predictions_gt.append(outputs)

        pbar = tqdm(self.parser.ops)
        for op in pbar:
            pbar.set_description("Processing {}".format(op.name))
            if op.type in self.skip_ops:
                continue

            mix_table = "tmp_mix_table.txt"
            self._gen_mix_table(mix_table, op)
            mix_model = MixQuantModel(self.fp32_mlir, self.chip, self.calib_table, mix_table)
            loss = 0
            for idx in range(self.num_sample):
                outputs = mix_model.infer(self.input_data_buffer[idx])
                loss += self._loss(outputs, predictions_gt[idx])
            mix_model.clean()
            loss_list.append((op.name, loss / self.num_sample))

        bf16_model.clean()
        return sorted(loss_list, key=lambda x: x[1], reverse=True)
