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
import copy
import time
import datetime
from tqdm import tqdm
from utils.mlir_shell import mlir_lowering
from utils.mlir_parser import MlirParser
from utils.misc import parse_debug_cmd
from utils.preprocess import preprocess
from calibration.data_selector import DataSelector

SKIP_OPERATION = [
    'top.Input', 'top.Reshape', 'top.Softmax', 'top.Weight', 'top.MaxPool', 'top.Slice', 'top.Tile',
    'top.Permute', 'top.Upsample'
]

FLOAT_MAP = {
    #"bm1684x": "F16",
    "bm1684x": "F32",
    "bm1684": "F32",
    "cv183x": "BF16",
    "cv182x": "BF16",
    "cv181x": "BF16",
}

def find_all_pre_layers(out_layers, op_name, parser, ref_activations_keys):
    pre_layers = parser.get_pre_op_by_op_name(op_name)
    if len(pre_layers) > 0:
        for pre_layer in pre_layers:
            if pre_layer not in out_layers:
                out_layers.append(pre_layer)
            if pre_layer not in ref_activations_keys:
                find_all_pre_layers(out_layers, pre_layer, parser, ref_activations_keys)
    else:
        if op_name not in out_layers:
            out_layers.append(op_name)

class MixQuantModel:

    def __init__(self, fp32_mlir, chip: str, calib_table: str = None, mix_table: str = None):
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

        self.quanted_mlir_file = '{}.{}.tune.mlir'.format(fp32_mlir, 'mix' if mix_table else 'int')
        mlir_lowering(self.fp32_mlir, self.quanted_mlir_file, self.mode, self.chip,
                      self.calib_table, False, self.mix_table)
        self.module = pymlir.module()
        self.module.load(self.quanted_mlir_file)
        self.parser = MlirParser(self.quanted_mlir_file)
        self.weight_file = self.parser.module_weight_file

    def infer(self, data: list):
        for k, v in zip(self.module.input_names, data):
            self.module.set_tensor(k, v)
        self.module.invoke()
        outputs = {}
        for name in self.module.output_names:
            outputs[name] = self.module.get_tensor(name)
        return outputs

    def infer2(self, top_op_name, input_data_dict: dict, input_data_dict2: dict):
        # print('mix model op list:', self.parser.get_op_name_list())
        for k in input_data_dict:
            self.module.set_tensor_from_int(k, input_data_dict[k])
            print(f'infer2 set_tensor:{k}')
            # print('set value:', input_data_dict[k].flatten()[:32])
            next_ops = self.parser.get_next_op_by_op_name(k)
            print(f'infer2 {k}\'s next_ops:{next_ops}')
            for next_op in next_ops:
                op = self.parser.get_op_by_op_name(next_op)
                if op.type == "tpu.Cast":
                    print(f'invoke_at CastOp:{next_op}')
                    self.module.invoke_at(next_op)
        for k in input_data_dict2:
            self.module.set_tensor_from_int(k, input_data_dict2[k])
        print(f'invoke_from {top_op_name}')
        self.module.invoke_from(top_op_name)
        outputs = {}
        for name in self.module.output_names:
            outputs[name] = self.module.get_tensor(name)
        return outputs

    def clean(self):
        try:
            sys.stdout.close()
            sys.stdout = self.stdout
            del self.module
            os.remove(self.quanted_mlir_file)
            os.remove(self.weight_file)
        except:
            pass


class MixPrecSearcher(object):

    def __init__(self, args):
        self.fp32_mlir = args.mlir_file
        self.calib_table = args.calibration_table
        self.loss_table = args.loss_table
        self.quantize_table = args.quantize_table
        self.chip = args.chip
        self.mix_mode = FLOAT_MAP[self.chip]
        self.parser = MlirParser(args.mlir_file)
        self.batch_size = self.parser.get_batch_size()
        self.input_num = self.parser.get_input_num()
        self.input_data_buffer = []
        self.num_sample = 0
        self.debug_cmd = parse_debug_cmd(args.debug_cmd)
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

    def _cal_sqnr(self, signal_raw, signal_dequant, remove_zero=False):
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
        for name1, name2 in zip(gt_preds, preds):
            a = gt_preds[name1]
            b = preds[name2]
            if a.dtype != b.dtype or a.shape != b.shape:
                raise RuntimeError("Calc loss fail:{} vs {}".format(name1, name2))
            loss = self._cal_sqnr(a, b)
            if not math.isinf(loss):
                ret += -loss * a.size
                cnt += a.size

        if ret == 0 and cnt == 0:
            return -math.inf
        else:
            return ret / cnt

    def run(self):
        t0 = time.time()
        loss_list = list()
        predictions_gt = list()

        # set all layer as float
        print("run float mode: {}".format(self.fp32_mlir))
        float_model = MixQuantModel(self.fp32_mlir, self.chip)
        for idx in range(self.num_sample):
            outputs = float_model.infer(self.input_data_buffer[idx])
            predictions_gt.append(outputs)
        float_model.clean()

        # set all layer as int8
        print("run int8 mode: {}".format(self.fp32_mlir))
        int8_model = MixQuantModel(self.fp32_mlir, self.chip, self.calib_table)
        int8_loss = 0
        for idx in range(self.num_sample):
            outputs = int8_model.infer(self.input_data_buffer[idx])
            int8_loss += self._loss(outputs, predictions_gt[idx])
        int8_model.clean()
        int8_loss = int8_loss / self.num_sample

        # set mix layers
        print("run mix mode: {}".format(self.fp32_mlir))
        pbar = tqdm(self.parser.ops)
        for op in pbar:
            pbar.set_description("Processing {}".format(op.name))
            if op.type in SKIP_OPERATION:
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

        loss_list = sorted(loss_list, key=lambda x: x[1], reverse=False)
        th = float(self.debug_cmd['more_mix_layer_factor']) if 'more_mix_layer_factor' in self.debug_cmd else 0.95
        num_mix_layers = 0
        with open(self.loss_table, "w") as f:
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# sample number: {}\n".format(self.num_sample))
            f.write("# all int8 loss: {}\n".format(int8_loss))
            f.write("# chip: {}  mix_mode: {}\n".format(self.chip, self.mix_mode))
            f.write("###\n")
            for idx, layer in enumerate(loss_list):
                loss_msg = "No.{:<4}: Layer: {:<50}\t\tLoss: {}".format(idx, layer[0], layer[1])
                f.write("{}\n".format(loss_msg))
                print(loss_msg)
                if (int8_loss / (layer[1]+1e-6) > th and num_mix_layers == 0):
                    num_mix_layers = idx
        max_mix_layers = len(loss_list) // 6  # no more than 1/4 layer to be float
        num_mix_layers = min(max_mix_layers, num_mix_layers)
        if num_mix_layers == 0:
            raise RuntimeError(
                "Mix layers loss similar to int8 loss. Try quantize {}".format(mix_model))
        with open(self.quantize_table, "w") as f:
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# sample number: {}\n".format(self.num_sample))
            f.write("# all int8 loss: {}\n".format(int8_loss))
            f.write("# chip: {}  mix_mode: {}\n".format(self.chip, self.mix_mode))
            f.write("###\n")
            f.write("# op_name   quantize_mode\n")
            for idx in range(num_mix_layers):
                name = loss_list[idx][0]
                f.write("{} {}\n".format(name, self.mix_mode))
        print("Output mix quantization table to {}".format(self.quantize_table))
        print("total time:{}".format(time.time() - t0))


class MixPrecSearcher2:
    def __init__(self, args):
        self.fp32_mlir = args.mlir_file
        self.calib_table = args.calibration_table
        self.loss_table = args.loss_table
        self.quantize_table = args.quantize_table
        self.chip = args.chip
        self.mix_mode = FLOAT_MAP[self.chip]
        self.parser = MlirParser(args.mlir_file)
        self.batch_size = self.parser.get_batch_size()
        self.input_num = self.parser.get_input_num()
        self.num_sample = 0
        self.debug_cmd = parse_debug_cmd(args.debug_cmd)
        # log_level = "DEBUG" if 'debug_log' in self.debug_cmd else "INFO"
        # self.logger = setup_logger('MixPrecSearcher2', log_level=log_level)
        self._init_inputs(args)

    def _init_inputs(self, args):
        self.ref_activations = {}
        tune_idx = 0
        self.ref_activations[tune_idx] = {}
        input_names = [op.name for op in self.parser.inputs]
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
            batched_idx = 0
            batched_inputs = self.input_num * ['']
            for data in ds.data_list:
                inputs = data.split(',')
                inputs = [s.strip() for s in inputs]
                assert (len(inputs) == self.input_num)
                batched_idx += 1
                for i, input in enumerate(input_names):
                    batched_inputs[i] += '{},'.format(inputs[i])
                    if batched_idx == self.batch_size:
                        x = ppa_list[i].run(batched_inputs[i][:-1])
                        count = self.parser.get_user_count_by_op_name(input)
                        self.ref_activations[tune_idx][input] = [x, count]
                if batched_idx == self.batch_size:
                    tune_idx += 1
                    batched_idx = 0
                    batched_inputs = self.input_num * ['']
                    self.ref_activations[tune_idx] = {}
        elif ds.all_npy:
            self.num_sample = len(ds.data_list)
            self.input_data_buffer = [[] for i in range(self.num_sample)]
            for data in ds.data_list:
                inputs = data.split(',')
                inputs = [s.strip() for s in inputs]
                assert (len(inputs) == self.input_num)
                for name, npy in zip(input_names, inputs):
                    x = np.load(npy)
                    count = self.parser.get_user_count_by_op_name(name)
                    self.ref_activations[tune_idx][name] = [x, count]
                tune_idx += 1
                self.ref_activations[tune_idx] = {}
        elif ds.all_npz:
            self.num_sample = len(ds.data_list)
            self.input_data_buffer = [[] for i in range(self.num_sample)]
            input_names = [op.name for op in self.parser.inputs]
            for data in ds.data_list:
                npz = np.load(data)
                for name in input_names:
                    count = self.parser.get_user_count_by_op_name(name)
                    self.ref_activations[tune_idx][name] = [npz[name], count]
                tune_idx += 1
                self.ref_activations[tune_idx] = {}
        else:
            raise RuntimeError("dataset is uncorrect")


    def _gen_mix_table(self, target_file, op):
        with open(target_file, 'w') as f:
            f.write("{} {}\n".format(op.name, self.mix_mode))

    def _cal_sqnr(self, signal_raw, signal_dequant, remove_zero=False):
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
        for name1, name2 in zip(gt_preds, preds):
            a = gt_preds[name1]
            b = preds[name2]
            if a.dtype != b.dtype or a.shape != b.shape:
                raise RuntimeError("Calc loss fail:{} vs {}".format(name1, name2))
            loss = self._cal_sqnr(a, b)
            if not math.isinf(loss):
                ret += -loss * a.size
                cnt += a.size

        if ret == 0 and cnt == 0:
            return -math.inf
        else:
            return ret / cnt


    def get_input_int8_tensor(self, i, op_name):
        if op_name in self.ref_activations[i]:
            return self.ref_activations[i][op_name][0]
        print('error, idx:{} op_name:{} not in ref_activations'.format(i, op_name))
        return None

    def layer_maybe_used_by_op_name_next(self, pre_layer, op_name, parser):
        check = False
        for op in parser.ops:
            if op_name == op.name:
                check = True
            if check:
                if pre_layer in op.opds:
                    return True
        return False

    def clear_int8_tensor(self, i, op_name, parser, parser2, only_in_int8):
        pre_layers = []
        find_all_pre_layers(pre_layers, op_name, parser, list(self.ref_activations[0].keys()))
        # print(f'i:{i}, clear_int8_tensor for {op_name}\'s pre_layers:{pre_layers}')
        for layer in pre_layers:
            if not self.layer_maybe_used_by_op_name_next(layer, op_name, parser):
                if layer in self.ref_activations[i]:
                    if self.ref_activations[i][layer][1] <= 1:
                        self.ref_activations[i].pop(layer)
                        # print(f'pop {layer}')
                    else:
                        self.ref_activations[i][layer][1] -= 1
                        # print(f'dec refcount {layer}')

        pre_int8_layers = []
        find_all_pre_layers(pre_int8_layers, op_name, parser2, list(self.ref_activations[0].keys()))
        for layer in pre_int8_layers:
            if not self.layer_maybe_used_by_op_name_next(layer, op_name, parser) and layer in only_in_int8:
                if layer in self.ref_activations[i]:
                    if self.ref_activations[i][layer][1] <= 1:
                        self.ref_activations[i].pop(layer)
                        only_in_int8.remove(layer)
                        # print(f'pop2 {layer}')
                    else:
                        self.ref_activations[i][layer][1] -= 1
                        # print(f'dec2 refcount {layer}')

        # print('ref_activations status:')
        # for i in self.ref_activations:
        #     for layer in self.ref_activations[i]:
        #         print(f'idx:{i}, layer:{layer} exist, refcount:{self.ref_activations[i][layer][1]}')


    def gen_int8_tensor(self, i, op_name, int8_model):
        if op_name in self.ref_activations[i]:
            return
        input_ops = int8_model.parser.get_pre_op_by_op_name(op_name)
        for input_op in input_ops:
            data = self.ref_activations[i][input_op][0]
            # print(f'set_tensor for input tensor:{input_op}, value:', data.flatten()[:32])
            int8_model.module.set_tensor_from_int(input_op, data)

        if len(input_ops) > 0:
            value = int8_model.module.invoke_at(op_name)
            print(f'invoke_at {op_name}')
            # value_fp32 = int8_model.module.get_fp32_tensor(op_name)
            # print(f'ret value:', value_fp32.flatten()[:32])
            count = int8_model.parser.get_user_count_by_op_name(op_name)
            self.ref_activations[i][op_name] = [value, count]

    def set_tensor_to_zero(self, model):
        for op in model.parser.ops:
            model.module.set_tensor(op.name, np.zeros(op.shape))

    def run(self):
        t0 = time.time()
        loss_list = list()
        predictions_gt = list()

        # set all layer as float
        print("run float mode: {}".format(self.fp32_mlir))
        float_model = MixQuantModel(self.fp32_mlir, self.chip)
        for idx in range(self.num_sample):
            net_input = list(self.ref_activations[idx].keys())[0]
            outputs = float_model.infer(self.ref_activations[idx][net_input])
            predictions_gt.append(outputs)
        float_model.clean()

        # set all layer as int8
        print("run int8 mode: {}".format(self.fp32_mlir))
        int8_model = MixQuantModel(self.fp32_mlir, self.chip, self.calib_table)
        int8_loss = 0
        for idx in range(self.num_sample):
            net_input = list(self.ref_activations[idx].keys())[0]
            outputs = int8_model.infer(self.ref_activations[idx][net_input])
            int8_loss += self._loss(outputs, predictions_gt[idx])
        int8_loss = int8_loss / self.num_sample
        # self.set_tensor_to_zero(int8_model)
        print('int8_loss:', int8_loss)

        # set mix layers
        only_in_int8 = []
        top_ops = {op.name:op for op in self.parser.ops}
        top_op_names = self.parser.get_op_name_list()
        print("run mix mode: {}".format(self.fp32_mlir))
        pbar = tqdm(int8_model.parser.ops)
        for op in pbar:
            print(f'gen_int8_tensor for {op.name}')
            for idx in range(self.num_sample):
                self.gen_int8_tensor(idx, op.name, int8_model)
            if op.name not in top_ops:
                only_in_int8.append(op.name)
                print(f'{op.name} only in int8_model')
                continue
            top_op = top_ops[op.name]
            pbar.set_description("Processing {}".format(top_op.name))
            if top_op.type in SKIP_OPERATION:
                print(f'skip {top_op.name} with type:{top_op.type}')
                continue

            mix_table = "tmp_mix_table.txt"
            self._gen_mix_table(mix_table, top_op)
            mix_model = MixQuantModel(self.fp32_mlir, self.chip, self.calib_table, mix_table)
            # self.set_tensor_to_zero(mix_model)
            print(f'MixQuantModel for {top_op.name}')
            first_seg,second_seg,other_tensor = [],[],[]
            for i, name in enumerate(top_op_names):
                if top_op.name == name:
                    first_seg = top_op_names[:i]
                    second_seg = top_op_names[i+1:]
                    # print(top_op.name, ',first_seg:',first_seg,',second_seg:',second_seg)
                    break
            for name in second_seg:
                other_tensor.extend([i for i in self.parser.get_pre_op_by_op_name(name) if i in first_seg])
            other_tensor = set(other_tensor)

            loss = 0
            for idx in range(self.num_sample):
                input_data_dict = {}
                for i, input in enumerate(self.parser.get_pre_op_by_op_name(top_op.name)):
                    # if int8_model.parser.get_op_by_op_name(input).type == "tpu.Cast":
                    #     for j in int8_model.parser.get_pre_op_by_op_name(input):
                    #         if int8_model.parser.get_op_by_op_name(j).type == "top.Input":
                    #             input = j
                    #             print('meet tpu.Cast,new input:',input)
                    #             break
                    print(f'{top_op.name}\'s top input{i}:{input}')
                    input_data_dict[input] = self.get_input_int8_tensor(idx, input)
                    if input in other_tensor:
                        other_tensor.remove(input)
                input_data_dict2 = {}
                for i, input in enumerate(other_tensor):
                    print(f'{top_op.name}\'s other input{i}:{input}')
                    input_data_dict2[input] = self.get_input_int8_tensor(idx, input)
                outputs = mix_model.infer2(top_op.name, input_data_dict, input_data_dict2)
                loss += self._loss(outputs, predictions_gt[idx])
            print(f'{op.name} loss:', loss / self.num_sample)
            loss_list.append((op.name, loss / self.num_sample))

            for idx in range(self.num_sample):
                self.clear_int8_tensor(idx, top_op.name, self.parser, int8_model.parser, only_in_int8)

            # if '693_Relu' == top_op.name:
            #     exit(1)
            mix_model.clean()

        loss_list = sorted(loss_list, key=lambda x: x[1], reverse=False)
        num_mix_layers = 0
        th = float(self.debug_cmd['more_mix_layer_factor']) if 'more_mix_layer_factor' in self.debug_cmd else 0.95
        with open(self.loss_table, "w") as f:
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# sample number: {}\n".format(self.num_sample))
            f.write("# all int8 loss: {}\n".format(int8_loss))
            f.write("# chip: {}  mix_mode: {}\n".format(self.chip, self.mix_mode))
            f.write("###\n")
            for idx, layer in enumerate(loss_list):
                loss_msg = "No.{:<4}: Layer: {:<50}\t\tLoss: {}".format(idx, layer[0], layer[1])
                f.write("{}\n".format(loss_msg))
                print(loss_msg, 'int8_loss/mix_loss:', int8_loss / (layer[1]+1e-6))
                if (int8_loss / (layer[1]+1e-6) > th  and num_mix_layers == 0): #eg.-10/-13, it is int8_loss/layer[1] > 1/1.2=0.83
                    num_mix_layers = idx
        print('num_mix_layers:', num_mix_layers)
        max_mix_layers = len(loss_list) // 4  # no more than 1/4 layer to be float
        num_mix_layers = min(max_mix_layers, num_mix_layers)
        if num_mix_layers == 0:
            raise RuntimeError(
                "Mix layers loss similar to int8 loss. Try quantize {}".format(mix_model))
        with open(self.quantize_table, "w") as f:
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# sample number: {}\n".format(self.num_sample))
            f.write("# all int8 loss: {}\n".format(int8_loss))
            f.write("# chip: {}  mix_mode: {}\n".format(self.chip, self.mix_mode))
            f.write("###\n")
            f.write("# op_name   quantize_mode\n")
            for idx in range(num_mix_layers):
                name = loss_list[idx][0]
                f.write("{} {}\n".format(name, self.mix_mode))
        print("Output mix quantization table to {}".format(self.quantize_table))
        print("total time:{}".format(time.time() - t0))


