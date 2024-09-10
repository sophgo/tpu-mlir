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
pymlir.set_mem_mode("force_value_mem")
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
from utils.misc import cos_sim, seed_all
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.net_dot_log import net_dot_log
from utils.log_setting import logger
import importlib.util

SKIP_OPERATION = [
    'top.Input', 'top.Reshape', 'top.Softmax', 'top.Weight', 'top.MaxPool', 'top.Slice', 'top.Tile',
    'top.Permute', 'top.Upsample'
]

FLOAT_MAP = {
    "bm1684x": "F16",
    "bm1684": "F32",
    "cv183x": "BF16",
    "cv182x": "BF16",
    "cv181x": "BF16",
    "cv180x": "BF16",
    "bm1688": "F16",
    "cv186x": "F16",
    "bm1690": "F16",
    "mars3": "BF16"
}

chip_support_mix_fp_type = {
    "bm1684x": ["F16", "F32"],
    "bm1688": ["F16", "F32"],
    "cv186x": ["F16", "F32"],
    "bm1684": ["F32"],
    "cv183x": ["BF16"],
    "cv182x": ["BF16"],
    "cv181x": ["BF16"],
    "cv180x": ["BF16"],
    "mars3": ["BF16"]
}


def find_all_pre_layers(all_pre_layers, op_name, parser, exist_ref_layers=None):
    pre_layers = parser.get_pre_op_by_op_name(op_name)
    if len(pre_layers) > 0:
        for pre_layer in pre_layers:
            if pre_layer not in all_pre_layers:
                all_pre_layers.append(pre_layer)
            if exist_ref_layers is not None:
                if pre_layer in exist_ref_layers:
                    find_all_pre_layers(all_pre_layers, pre_layer, parser, exist_ref_layers)
            else:
                find_all_pre_layers(all_pre_layers, pre_layer, parser, exist_ref_layers)
    else:
        if op_name not in all_pre_layers:
            all_pre_layers.append(op_name)


class MixQuantModel:
    def __init__(self, fp32_mlir, chip: str, calib_table: str = None, mix_table: str = None, fp_type: str = 'auto'):
        self.fp32_mlir = fp32_mlir
        self.chip = chip
        self.calib_table = None
        self.mix_table = None
        if calib_table:
            self.mode = "INT8"
            self.calib_table = calib_table
            self.mix_table = mix_table
        else:
            if fp_type == 'auto':
                self.mode = FLOAT_MAP[chip]
            else:
                self.mode = fp_type
                if fp_type not in chip_support_mix_fp_type[self.chip]:
                    print('parameter error, fp_type:{fp_type} not support by {chip}')
                    exit(1)

        self.quanted_mlir_file = '{}.{}.tune.mlir'.format(fp32_mlir, 'mix' if mix_table else self.mode)
        mlir_lowering(self.fp32_mlir, self.quanted_mlir_file, self.mode, self.chip, 1, 1,
                      self.calib_table, False, self.mix_table)
        self.module = pymlir.module()
        self.module.load(self.quanted_mlir_file)
        self.parser = MlirParser(self.quanted_mlir_file)
        self.weight_file = self.parser.module_weight_file

    def infer(self, data: list, global_compare_layers: list = None):
        for k, v in zip(self.module.input_names, data):
            self.module.set_tensor(k, v)
        self.module.invoke()
        outputs = {}
        if global_compare_layers is None:
            for name in self.module.output_names:
                outputs[name] = self.module.get_tensor(name).copy()
        else:
            for name in global_compare_layers:
                outputs[name] = self.module.get_tensor(name).copy()
        return outputs

    def infer_from(self, top_op_name, input_data_dict: dict, extra_input_data_dict: dict,
                   global_compare_layers: list = None):
        # print('mix model op list:', self.parser.get_op_name_list())
        for k in input_data_dict:
            self.module.set_tensor_from_int(k, input_data_dict[k])
            print(f'infer_from set_tensor:{k}')
            # print('set value:', input_data_dict[k].flatten()[:32])
            next_ops = self.parser.get_next_op_by_op_name(k)
            print(f'infer_from {k}\'s next_ops:{next_ops}')
            for next_op in next_ops:
                op = self.parser.get_op_by_op_name(next_op)
                if op.type == "tpu.Cast":
                    print(f'invoke_at CastOp:{next_op}')
                    self.module.invoke_at(next_op)
        for k in extra_input_data_dict:
            print(f'infer_from set_extra_tensor:{k}')
            self.module.set_tensor_from_int(k, extra_input_data_dict[k])
            next_ops = self.parser.get_next_op_by_op_name(k)
            print(f'infer_from {k}\'s next_ops:{next_ops}')
            for next_op in next_ops:
                op = self.parser.get_op_by_op_name(next_op)
                if op.type == "tpu.Cast":
                    print(f'invoke_at CastOp:{next_op}')
                    self.module.invoke_at(next_op)
        print(f'invoke_from {top_op_name}')
        self.module.invoke_from(top_op_name)
        outputs = {}
        if global_compare_layers is None:
            for name in self.module.output_names:
                outputs[name] = self.module.get_tensor(name).copy()
        else:
            for name in global_compare_layers:
                outputs[name] = self.module.get_tensor(name).copy()
        return outputs

    def clean(self):
        try:
            del self.module
            os.remove(self.quanted_mlir_file)
            os.remove(self.weight_file)
        except:
            pass


class MixPrecSearcher:
    def __init__(self, args):
        self.args = args
        self.fp32_mlir = args.mlir_file
        self.calib_table = args.calibration_table
        if 'loss_table' in args:
            self.loss_table = args.loss_table
        if 'quantize_table' in args:
            self.quantize_table = args.quantize_table
        self.chip = args.chip
        if args.fp_type == 'auto':
            self.mix_mode = FLOAT_MAP[self.chip]
        else:
            self.mix_mode = args.fp_type
            if args.fp_type not in chip_support_mix_fp_type[self.chip]:
                print('parameter error, fp_type:{args.fp_type} not support by {self.chip}')
                exit(1)

        self.parser = MlirParser(args.mlir_file)
        self.batch_size = self.parser.get_batch_size()
        self.input_num = self.parser.get_input_num()
        self.num_sample = 0
        self.debug_cmd = parse_debug_cmd(args.debug_cmd)
        log_level = "DEBUG" if 'debug_log' in self.debug_cmd else "INFO"
        if '_logger' in args:
            self.logger = args._logger
        else:
            try:
                if args.inference_num > 0:
                    self.logger = logger('SensitiveLayerSearcher', log_level=log_level)
            except:
                self.logger = logger('MixPrecSearcher', log_level=log_level)
        self._init_inputs(args)
        try:
            if args.post_process is not None:
                self.post_process_path = args.post_process
                if self.post_process_path.find('/') != -1:
                    self.post_process_name = self.post_process_path.split('/')[-1]
                else:
                    self.post_process_name = self.post_process_path
                self.post_process_name = self.post_process_name[:-3]
            else:
                self.post_process_path = None
                self.post_process_name = None
        except:
            self.post_process_path = None
            self.post_process_name = None

    def disable_print(self):
        if 'debug_log' not in self.debug_cmd:
            self.stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def enable_print(self):
        if 'debug_log' not in self.debug_cmd:
            sys.stdout.close()
            sys.stdout = self.stdout

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
        self.int8_activations = copy.deepcopy(self.ref_activations)

    def _gen_mix_table(self, mix_ops):
        target_file = "tmp_mix_table.txt"
        with open(target_file, 'w') as f:
            for mix_op in mix_ops:
                f.write("{} {}\n".format(mix_op, self.mix_mode))
        return target_file

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

    def _sqnr_loss(self, preds, gt_preds, layers_rate):
        ret = 0
        cnt, i = 0, 0
        for name1, name2 in zip(gt_preds, preds):
            a = gt_preds[name1]
            b = preds[name2]
            if a.dtype != b.dtype or a.shape != b.shape:
                raise RuntimeError("Calc loss fail:{} vs {}".format(name1, name2))
            loss = self._cal_sqnr(a, b)
            if not math.isinf(loss):
                ret += -loss * a.size * layers_rate[i]
                cnt += a.size
                i += 1
        ret = ret / sum(layers_rate)
        if ret == 0 and cnt == 0:
            return -math.inf
        else:
            return ret / cnt

    def _cos_loss(self, preds, gt_preds, layers_rate):
        cos, i = 0, 0
        for name1, name2 in zip(gt_preds, preds):
            a = gt_preds[name1]
            b = preds[name2]
            if a.dtype != b.dtype or a.shape != b.shape:
                raise RuntimeError("Calc loss fail:{} vs {}".format(name1, name2))
            cos += layers_rate[i] * cos_sim(a.reshape(-1), b.reshape(-1))
            i += 1
        return cos / sum(layers_rate)

    def _snr_loss(self, preds, gt_preds, layers_rate):
        snr,i=0,0
        for name1, name2 in zip(gt_preds, preds):
            a = gt_preds[name1]
            b = preds[name2]
            if a.dtype != b.dtype or a.shape != b.shape:
                raise RuntimeError("Calc loss fail:{} vs {}".format(name1, name2))
            a=a.flatten()
            b=b.flatten()
            noise_power=np.power(b-a,2).sum(axis=-1)
            signal_power=np.power(a,2).sum(axis=-1)
            snr=layers_rate[i]*(noise_power)/(signal_power+1e-7)
            i+=1
        return snr / sum(layers_rate)

    # def _snr_loss(self, preds, gt_preds, layers_rate):
    #     snr = 0
    #     epsilon = 1e-15
    #     for name1, name2 in zip(gt_preds, preds):
    #         a = gt_preds[name1]
    #         b = preds[name2]
    #         if a.dtype != b.dtype or a.shape != b.shape:
    #             raise RuntimeError("Calc loss fail:{} vs {}".format(name1, name2))
    #         b = np.clip(b, epsilon, 1. - epsilon)
    #         ce = -np.sum(a * np.log(b))
    #         snr += ce / a.shape[0]
    #     return snr / sum(layers_rate)

    def _loss(self, preds, gt_preds, layers_rate=None, type='cos'):
        assert type == 'cos' or type == 'sqnr' or type=='snr'
        if layers_rate is None:
            layers_rate = len(preds) * [1]
        if type == 'cos':
            return self._cos_loss(preds, gt_preds, layers_rate)
        elif type == 'sqnr':
            return self._sqnr_loss(preds, gt_preds, layers_rate)
        elif type=='snr':
            return self._snr_loss(preds, gt_preds, layers_rate)

    def get_input_fp32_tensor(self, i, op_name):
        if op_name in self.ref_activations[i]:
            return self.ref_activations[i][op_name][0]
        self.exception_occur('error, idx:{} op_name:{} not in ref_activations'.format(i, op_name))

    def exception_occur(self, info):
        if 'debug_log' in self.debug_cmd:
            print(info)
        else:
            raise Exception(info)

    def get_input_int8_tensor(self, i, op_name, ret_fp32=False):
        if op_name in self.int8_activations[i]:
            if ret_fp32:
                return self.int8_activations[i][op_name][2]
            else:
                return self.int8_activations[i][op_name][0]
        self.exception_occur('error, idx:{} op_name:{} not in int8_activations'.format(i, op_name))

    def layer_used_by_successor_op(self, pre_layer, op_name, parser):
        check = False
        for op in parser.ops:
            if check:
                if pre_layer in op.opds:
                    return True
            if op_name == op.name:
                check = True
        return False

    def clear_tensor(self, i, op_name, data_dict, parser, all_pre_layers):
        for layer in all_pre_layers:
            if not self.layer_used_by_successor_op(layer, op_name, parser):
                if layer in data_dict[i]:
                    if data_dict[i][layer][1] <= 1:
                        data_dict[i].pop(layer)
                        if i == 0:
                            self.logger.print_dbg(f'pop {layer}')
                    else:
                        data_dict[i][layer][1] -= 1
                        if i == 0:
                            self.logger.print_dbg(f'dec {layer} refcount')

        self.logger.print_dbg('data_dict status:')
        for layer in data_dict[0]:
            self.logger.print_dbg(f'layer:{layer} exist, refcount:{data_dict[0][layer][1]}')

    def gen_ref_tensor(self, i, op_name, data_dict, model, is_int8_data=False):
        if op_name in data_dict[i]:
            return False
        input_ops = model.parser.get_pre_op_by_op_name(op_name)
        for input_op in input_ops:
            data = data_dict[i][input_op][0]
            if data is None:
                raise Exception(f"{op_name} \'s input:{input_op} not exist")
            if is_int8_data:
                model.module.set_tensor_from_int(input_op, data)
            else:
                model.module.set_tensor(input_op, data)
        if len(input_ops) > 0:
            value = model.module.invoke_at(op_name).copy()
            self.logger.print_dbg(f'invoke_at {op_name}')
            fp32_v = None
            if is_int8_data:
                fp32_v = model.module.get_fp32_tensor(op_name)
            count = model.parser.get_user_count_by_op_name(op_name)
            if fp32_v is None:
                data_dict[i][op_name] = [value.copy(), count]
            else:
                data_dict[i][op_name] = [value.copy(), count, fp32_v]
        return True

    def visual_tensor_diff(self, name, cos, int8_out, fp32_out):
        data_size = fp32_out.size
        max_sampling = 10000
        if 'max_sampling' in self.debug_cmd:
            max_sampling = int(self.debug_cmd['max_sampling'])
        if data_size > max_sampling:
            step = data_size // max_sampling
        else:
            step = 1
        index = np.arange(data_size)
        index = index[::step]
        rows = 2
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("ref vs target",
                            "ref - target   COS:{}".format(cos)))

        fig.add_trace(go.Scattergl(y=fp32_out.reshape([-1])[::step],
                                   x=index,
                                   name='ref',
                                   mode='lines+markers',
                                   marker={
                                       "size": 6,
                                       "symbol": 300
                                   },
                                   line={"width": 1}),
                      row=1,
                      col=1)
        fig.add_trace(go.Scattergl(y=int8_out.reshape([-1])[::step],
                                   x=index,
                                   name='target',
                                   mode='lines+markers',
                                   marker={
                                       "size": 6,
                                       "symbol": 304,
                                       "opacity": 0.8
                                   },
                                   line={"width": 1}),
                      row=1,
                      col=1)

        fig.add_trace(go.Scattergl(y=(fp32_out - int8_out).reshape([-1])[::step],
                                   x=index,
                                   name='diff',
                                   line={"width": 1}),
                      row=2,
                      col=1)
        fig.update_layout(height=400 * rows, width=900)
        fig.update_layout(margin=dict(l=5, r=10, t=20, b=0))
        fig.update_layout(legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.95))
        tmpstr = './tensor_diff_fp32_vs_int8/{}'.format(name.replace('/', '_'))
        if 'dump_tensor_diff_by_html' in self.debug_cmd:
            fig.write_html(tmpstr + '.html')
            return tmpstr + '.html'
        fig.write_image(tmpstr + '.png')
        return tmpstr + '.png'

    def get_extra_input_tensor(self, op_name, parser):
        first_seg, second_seg, extra_input = [], [], []
        op_names = parser.get_op_name_list()
        for i, name in enumerate(op_names):
            if op_name == name:
                first_seg = op_names[:i]
                second_seg = op_names[i + 1:]
                # print(op_name, ',first_seg:',first_seg,',second_seg:',second_seg)
                break
        for name in second_seg:
            extra_input.extend([i for i in parser.get_pre_op_by_op_name(name) if i in first_seg])
        extra_input = set(extra_input)
        return extra_input

    def clear_ref_tensor(self, op_name, parser, data_dict):
        all_pre_layers = []
        find_all_pre_layers(all_pre_layers, op_name, parser, list(data_dict[0].keys()))
        for idx in range(self.num_sample):
            self.clear_tensor(idx, op_name, data_dict, parser, all_pre_layers)

    def clear_ref_tensor2(self, op_name, top_parser, int8_parser, data_dict):
        all_pre_layers = []
        find_all_pre_layers(all_pre_layers, op_name, top_parser, list(data_dict[0].keys()))
        all_pre_layers2 = []
        find_all_pre_layers(all_pre_layers2, op_name, int8_parser, list(data_dict[0].keys()))
        all_pre_layers.extend(all_pre_layers2)
        all_pre_layers = set(all_pre_layers)
        all_pre_layers_used = len(all_pre_layers) * [False]
        for i, layer in enumerate(all_pre_layers):
            if self.layer_used_by_successor_op(layer, op_name, top_parser):
                all_pre_layers_used[i] = True
            if self.layer_used_by_successor_op(layer, op_name, int8_parser):
                all_pre_layers_used[i] = True
        for i in range(self.num_sample):
            for used, layer in zip(all_pre_layers_used, all_pre_layers):
                if not used:
                    if layer in data_dict[i]:
                        if data_dict[i][layer][1] <= 1:
                            data_dict[i].pop(layer)
                            if i == 0:
                                self.logger.print_dbg(f'pop {layer}')
                        else:
                            data_dict[i][layer][1] -= 1
                            if i == 0:
                                self.logger.print_dbg(f'dec {layer} refcount')
        self.logger.print_dbg('data_dict status:')
        for layer in data_dict[0]:
            self.logger.print_dbg(f'  layer:{layer} exist, refcount:{data_dict[0][layer][1]}')

    def collect_op_input_tensor(self, idx, op_name, extra_input, fp_layer_list=None):
        input_data_dict = {}
        # for i, input in enumerate(self.parser.get_pre_op_by_op_name(op_name)):
        for i, input in enumerate(self.parser.get_pre_op_by_op_name(op_name)):
            self.logger.print_dbg(f'{op_name}\'s top input{i}:{input}')
            # if self.parser.get_op_by_op_name(input).type in SKIP_OPERATION:
            #     fp_layer_list.append(input)
            if fp_layer_list is not None and input in fp_layer_list:
                input_data_dict[input] = self.get_input_fp32_tensor(idx, input)
            else:
                input_data_dict[input] = self.get_input_int8_tensor(idx, input)
            if input in extra_input:
                extra_input.remove(input)
        extra_input_data_dict = {}
        for i, input in enumerate(extra_input):
            self.logger.print_dbg(f'{op_name}\'s other input{i}:{input}')
            if fp_layer_list is not None and input in fp_layer_list:
                extra_input_data_dict[input] = self.get_input_fp32_tensor(idx, input)
            else:
                extra_input_data_dict[input] = self.get_input_int8_tensor(idx, input)
        return input_data_dict, extra_input_data_dict

    def extract_global_layers(self):
        global_compare_layers = None
        layers_rate = None
        all_pre_layers = None
        if self.args.global_compare_layers != '':
            global_compare_layers = [i.strip() for i in self.args.global_compare_layers.split(',')]
            layers_rate = [float(i.split(':')[-1].strip()) for i in global_compare_layers if ':' in i]
            global_compare_layers = [i.split(':')[0].strip() for i in global_compare_layers]
        if global_compare_layers != None:
            all_pre_layers = []
            for layer in global_compare_layers:
                pre_layers = []
                find_all_pre_layers(pre_layers, layer, self.parser)
                all_pre_layers.extend(pre_layers)
            all_pre_layers = set(all_pre_layers)
            if len(layers_rate) > 0:
                if len(global_compare_layers) != len(layers_rate):
                    print('global_compare_layers format error')
                    exit(1)
                if 1 != sum(layers_rate):
                    print('global_compare_layers rate error')
                    exit(1)
            else:
                layers_rate = len(global_compare_layers) * [1]
        return global_compare_layers, layers_rate, all_pre_layers

    def run_model(self, model, float_type, global_compare_layers, layers_rate, predictions_gt):
        outputs_cos = 0
        if float_type:
            self.disable_print()
            self.logger.print_info("run float mode: {}".format(self.fp32_mlir))
        else:
            self.logger.print_info("run int8 mode: {}".format(self.fp32_mlir))
        for idx in range(self.num_sample):
            data = []
            for name in list(self.ref_activations[idx].keys()):
                data.append(self.ref_activations[idx][name][0])
            outputs = model.infer(data, global_compare_layers)
            if self.post_process_path:
                module_path = self.post_process_path
                module_name = self.post_process_name
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                modulevar = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(modulevar)
                outputs = modulevar.PostProcess(outputs)
            if float_type:
                predictions_gt.append(outputs)
            else:
                outputs_cos += self._loss(outputs, predictions_gt[idx], layers_rate)
        outputs_cos = outputs_cos / self.num_sample
        return outputs_cos

    def run_model_loss_snr(self, model, float_type, global_compare_layers, layers_rate, predictions_gt):
        outputs_cos = 0
        if float_type:
            self.disable_print()
            self.logger.print_info("run float mode: {}".format(self.fp32_mlir))
        else:
            self.logger.print_info("run int8 mode: {}".format(self.fp32_mlir))
        for idx in range(self.num_sample):
            data = []
            for name in list(self.ref_activations[idx].keys()):
                data.append(self.ref_activations[idx][name][0])
            outputs = model.infer(data, global_compare_layers)
            if self.post_process_path:
                module_path = self.post_process_path
                module_name = self.post_process_name
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                modulevar = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(modulevar)
                outputs = modulevar.PostProcess(outputs)
            if float_type:
                predictions_gt.append(outputs)
            else:
                type='snr'
                outputs_cos += self._loss(outputs, predictions_gt[idx], layers_rate,type)
        outputs_cos = outputs_cos / self.num_sample
        return outputs_cos

    def get_fixed_float_layers(self,model,global_compare_layers, layers_rate, predictions_gt):
        float_ops = []
        self.logger.print_info("run int8 mode: {}".format(self.fp32_mlir))
        for idx in range(1):
            data = []
            for name in list(self.ref_activations[idx].keys()):
                data.append(self.ref_activations[idx][name][0])
            outputs = model.infer(data, global_compare_layers)
        for op in model.module.all_tensor_names:
            qinfo = model.module.get_tensor_qinfo(op)
            if qinfo.dtype != "I8" and qinfo.dtype != "U8":
                float_ops.append(op)
        return float_ops

    def get_full_op_list(self, float_model, int8_model, fp_op_names, int8_op_names):
        full_op_list, cur_fp_op_idx = {}, 0
        for int8_op in int8_op_names:
            pos = fp_op_names.index(int8_op) if int8_op in fp_op_names else -1
            if -1 == pos:
                full_op_list[int8_op] = int8_model.parser.get_op_by_op_name(int8_op)
            else:
                for i in range(cur_fp_op_idx, pos + 1):
                    full_op_list[fp_op_names[i]] = float_model.parser.get_op_by_op_name(
                        fp_op_names[i])
                cur_fp_op_idx = pos + 1
        print('full_op_list:', list(full_op_list.keys()))
        return full_op_list

    def cal_avg_layer_cos(self, op):
        layer_cos = 0
        for idx in range(self.num_sample):
            int8_out = self.get_input_int8_tensor(idx, op.name, True)
            fp32_out = self.get_input_fp32_tensor(idx, op.name)
            cos = cos_sim(int8_out, fp32_out)
            layer_cos += cos
            if 'visual_tensor' in self.debug_cmd and (
                    op.name in self.debug_cmd['visual_tensor'] or self.debug_cmd['visual_tensor'] == 'all'):
                plot_path = self.visual_tensor_diff(f'int8_model_{op.name}_{idx}', cos, int8_out, fp32_out)
                if idx == 0:
                    self.dot_log.add_node_label(op.name, 'visual_tensor diff enable, please click')
                    self.dot_log.add_node_attr(op.name, 'URL',
                                               plot_path)  # shape='box', fillcolor=next_node_fillcolor , style='filled'
        avg_cos = layer_cos / self.num_sample
        self.dot_log.add_node_label(op.name, f'int8 layer cos:{avg_cos:.6f}')
        return avg_cos

    def gen_int8_tensor(self, op, int8_op_names, int8_model, fp_layer_list):
        have_int8_next = False
        if op.name in int8_op_names:
            for next_int8_op in int8_model.parser.get_next_op_by_op_name(op.name):
                if next_int8_op not in fp_layer_list:
                    have_int8_next = True
                    break
            if have_int8_next:
                for idx in range(self.num_sample):
                    ret = self.gen_ref_tensor(idx, op.name, self.int8_activations, int8_model, True)
                if ret:
                    self.dot_log.add_node_label(op.name, 'gen int8 tensor')
            else:
                self.dot_log.add_node_label(op.name, f'next_layer is {self.mix_mode}, not need to gen int8 tensor')
        return have_int8_next

    def gen_fp_tensor(self, op, fp_op_names, float_model):
        if op.name in fp_op_names:
            for idx in range(self.num_sample):
                ret = self.gen_ref_tensor(idx, op.name, self.ref_activations, float_model)
            if ret:
                self.dot_log.add_node_label(op.name, f'gen {self.mix_mode} tensor')

    def skip_op(self, op, top_ops, have_int8_next, fp_layer_list, all_pre_layers):
        if op.name not in top_ops:
            self.dot_log.add_node_label(op.name, 'op not in top dialect')
            return True
        if op.type in SKIP_OPERATION:
            pre_layers = self.parser.get_pre_op_by_op_name(op.name)
            for pre_layer in pre_layers:
                if pre_layer in fp_layer_list:
                    self.dot_log.add_node_label(op.name, f'add {op.name} to fp_layer_list')
                    fp_layer_list.append(op.name)
                    break
            self.dot_log.add_node_label(op.name, f'meet quant skip op, type:{op.type}, continue')
            return True
        if op.name in fp_layer_list:
            self.dot_log.add_node_label(op.name, f'op is {self.mix_mode} layer, continue')
            return True
        if not have_int8_next:
            self.dot_log.add_node_label(op.name, f'not call gen_ref_tensor, continue')
            return True
        if all_pre_layers is not None and op.name not in all_pre_layers:
            self.dot_log.add_node_label(op.name, f'not in all_pre_layers of compare layer, continue')
            return True
        return False

    def check_input(self,op, next_top_ops, fp_layer_list):
        fp_layer_list_copy=fp_layer_list.copy()
        fp_layer_list_copy.append(op.name)
        tmp = ','.join(next_top_ops)
        fp_layer_list_copy.extend(next_top_ops)
        mix_table = self._gen_mix_table(fp_layer_list_copy)
        mix_model = MixQuantModel(self.fp32_mlir, self.chip, self.calib_table, mix_table, self.args.fp_type)
        for i, input in enumerate(mix_model.parser.get_pre_op_by_op_name(op.name)):
            if input in fp_layer_list_copy:
                if input not in self.ref_activations[0]:
                    return False
            else:
                if input not in self.int8_activations[0]:
                    return False
        return True

    def set_curr_layer_and_the_next_layer_mix(self, op, next_top_ops, fp_layer_list, global_compare_layers, layers_rate,
                                              predictions_gt):
        outputs_cos = 0
        fp_layer_list.append(op.name)
        tmp = ','.join(next_top_ops)
        self.dot_log.add_node_label(op.name, f'cos too low, set {self.mix_mode} to {op.name} and next_top_ops:{tmp}')
        fp_layer_list.extend(next_top_ops)
        mix_table = self._gen_mix_table(fp_layer_list)
        mix_model = MixQuantModel(self.fp32_mlir, self.chip, self.calib_table, mix_table, self.args.fp_type)
        extra_input = self.get_extra_input_tensor(op.name, self.parser)
        for idx in range(self.num_sample):
            input_data_dict, extra_input_data_dict = self.collect_op_input_tensor(idx, op.name, extra_input,
                                                                                  fp_layer_list)
            tmp1 = ','.join(list(input_data_dict.keys()))
            tmp2 = '' if len(extra_input_data_dict) == 0 else ',extra_input_data_dict:' + ','.join(
                list(extra_input_data_dict.keys()))
            if idx == 0:
                self.dot_log.add_node_label(op.name, f'input_data_dict:{tmp1}{tmp2}, call infer_from')
            outputs = mix_model.infer_from(op.name, input_data_dict, extra_input_data_dict, global_compare_layers)
            mix_layer_out = mix_model.module.get_fp32_tensor(op.name)
            fp32_out = self.get_input_fp32_tensor(idx, op.name)
            cos = cos_sim(mix_layer_out.reshape(-1), fp32_out.reshape(-1))
            if idx == 0:
                self.dot_log.add_node_label(op.name, f'first {self.mix_mode} layer:{op.name} cos:{cos:.6f}')
            outputs_cos += self._loss(outputs, predictions_gt[idx], layers_rate)
            for next_top_op in next_top_ops:
                count = self.parser.get_user_count_by_op_name(next_top_op)
                self.int8_activations[idx][next_top_op] = [None, count, None]
                next_ops = mix_model.parser.get_next_op_by_op_name(next_top_op)
                print(f'{next_top_op}\'s next_ops:', ','.join(next_ops))
                for next_op in next_ops:
                    if mix_model.parser.get_op_by_op_name(next_op).type == "tpu.Cast":
                        if idx == 0:
                            self.dot_log.add_node_label(op.name, f'use {next_op} to replace {next_top_op}')
                        self.int8_activations[idx][next_top_op][0] = mix_model.module.get_tensor(next_op).copy()
                        self.int8_activations[idx][next_top_op][2] = mix_model.module.get_fp32_tensor(next_op)
        outputs_cos = outputs_cos / self.num_sample
        return outputs_cos, mix_model

    def search_mix_layer(self, float_model, int8_model, layer_cos_list, predictions_gt, fp_layer_list,
                      global_compare_layers, layers_rate, all_pre_layers, all_int8_cos):
        outputs_cos = 0
        self.dot_log = net_dot_log('mix_prec_result', int8_model.parser, self.logger)
        self.dot_log.add_new_log_region(f'compute cos and set {self.mix_mode} layer')
        top_op_names = self.parser.get_op_name_list()
        max_fp32_layer_num = len(top_op_names) // 4
        fp_op_names = float_model.parser.get_op_name_list()
        top_ops = {op.name: op for op in self.parser.ops}
        int8_model = MixQuantModel(self.fp32_mlir, self.chip, self.calib_table)
        int8_op_names = int8_model.parser.get_op_name_list()
        full_op_list = self.get_full_op_list(float_model, int8_model, fp_op_names, int8_op_names)
        try:
            pbar = tqdm(list(full_op_list.values()))
            for op in pbar:
                pbar.set_description("Processing {}".format(op.name))
                have_int8_next = self.gen_int8_tensor(op, int8_op_names, int8_model, fp_layer_list)
                self.gen_fp_tensor(op, fp_op_names, float_model)
                if self.skip_op(op, top_ops, have_int8_next, fp_layer_list, all_pre_layers):
                    continue
                avg_cos = self.cal_avg_layer_cos(op)
                layer_cos_list.append((op.name, avg_cos))
                if avg_cos < self.args.min_layer_cos:
                    next_top_ops = self.parser.get_next_op_by_op_name(op.name)
                    # have_input=self.check_input(op, next_top_ops, fp_layer_list)
                    # if not have_input:
                    #     continue
                    outputs_cos, mix_model = self.set_curr_layer_and_the_next_layer_mix(op, next_top_ops, fp_layer_list,
                                                                                        global_compare_layers,
                                                                                        layers_rate, predictions_gt)
                    self.dot_log.add_node_label(op.name, f'current output cos:{outputs_cos:.6f}')
                    mix_model.clean()
                    if outputs_cos > self.args.expected_cos:
                        self.dot_log.add_node_label(op.name,
                                                    f'job success, current cos is higher than expected_cos:{self.args.expected_cos}')
                        break
                    if len(fp_layer_list) > max_fp32_layer_num:
                        self.dot_log.add_node_label(op.name,
                                                    f'job fail, the number of layers of {self.mix_mode} exceeded the maximum')
                        break
                    if outputs_cos < all_int8_cos:
                        fp_layer_list.remove(op.name)
                        for i in next_top_ops:
                            fp_layer_list.remove(i)
                        self.dot_log.add_node_label(op.name,
                                                    f'meet outputs_cos:{outputs_cos:.6f} < all_int8_cos:{all_int8_cos:.6f}')
                        for idx in range(self.num_sample):
                            for next_top_op in next_top_ops:
                                self.int8_activations[idx].pop(next_top_op)
                self.clear_ref_tensor(op.name, self.parser, self.ref_activations)
                self.clear_ref_tensor2(op.name, self.parser, int8_model.parser, self.int8_activations)
        except Exception as err:
            self.logger.print_info('An exception happened: ' + str(err))
            pass
        self.dot_log.gen_dot_graph()
        int8_model.clean()
        float_model.clean()
        self.enable_print()
        return outputs_cos

    def print_log_info(self, layer_cos_list, fp_layer_list, all_int8_cos, outputs_cos, t0):
        self.logger.print_info('>>>run result:')
        layer_cos_list = sorted(layer_cos_list, key=lambda x: x[1], reverse=False)
        with open(self.loss_table, "w") as f:
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# sample number: {}\n".format(self.num_sample))
            f.write("# chip: {}  mix_mode: {}\n".format(self.chip, self.mix_mode))
            f.write("###\n")
            for idx, layer in enumerate(layer_cos_list):
                loss_msg = "No.{:<4}: Layer: {:<50}\t\t\tCos: {:.6f}".format(idx, layer[0], layer[1])
                f.write("{}\n".format(loss_msg))
                self.logger.print_info(loss_msg)
        with open(self.quantize_table, "w") as f:
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# sample number: {}\n".format(self.num_sample))
            f.write("# chip: {}  mix_mode: {}\n".format(self.chip, self.mix_mode))
            f.write("# number of {} layer: {}\n".format(self.mix_mode, len(fp_layer_list)))
            f.write("###\n")
            f.write("# op_name   quantize_mode\n")
            for layer in fp_layer_list:
                f.write("{} {}\n".format(layer, self.mix_mode))
        self.logger.print_info(f'int8 outputs_cos:{all_int8_cos:.6f} old')
        self.logger.print_info(f"mix model outputs_cos:{outputs_cos:.6f}")
        self.logger.print_info("Output mix quantization table to {}".format(self.quantize_table))
        self.logger.print_info("total time:{}".format(time.time() - t0))

    def run(self):
        t0 = time.time()
        layer_cos_list, predictions_gt, fp_layer_list = [], [], []
        os.system('rm -rf tensor_diff_fp32_vs_int8;mkdir -p tensor_diff_fp32_vs_int8/')
        global_compare_layers, layers_rate, all_pre_layers = self.extract_global_layers()
        float_model = MixQuantModel(self.fp32_mlir, self.chip)
        _ = self.run_model(float_model, True, global_compare_layers, layers_rate, predictions_gt)

        int8_model = MixQuantModel(self.fp32_mlir, self.chip, self.calib_table)
        outputs_cos = self.run_model(int8_model, False, global_compare_layers, layers_rate, predictions_gt)
        if outputs_cos > self.args.expected_cos:
            float_model.clean()
            int8_model.clean()
            self.enable_print()
            self.logger.print_info(
                f'job success, current int8 cos:{outputs_cos} is higher than expected_cos:{self.args.expected_cos},no need for mix precsion')
            exit(0)
        all_int8_cos = outputs_cos
        outputs_cos = self.search_mix_layer(float_model, int8_model, layer_cos_list, predictions_gt, fp_layer_list,
                                         global_compare_layers, layers_rate, all_pre_layers, all_int8_cos)
        self.print_log_info(layer_cos_list, fp_layer_list, all_int8_cos, outputs_cos, t0)

    def run_bias_correction(self):
        self.logger.print_info("run_bias_correction start")
        t0 = time.time()
        layer_cos_list = list()
        predictions_gt = list()
        weight_file_name = MlirParser(self.fp32_mlir).module_weight_file
        os.system('rm -rf tensor_diff_fp32_vs_int8;mkdir -p tensor_diff_fp32_vs_int8/')

        # set all layer as float
        self.logger.print_info("run float mode: {}".format(self.fp32_mlir))
        self.disable_print()
        float_model = MixQuantModel(self.fp32_mlir, self.chip)
        global_compare_layers = None
        layers_rate = None
        all_pre_layers = None
        if self.args.global_compare_layers != '':
            global_compare_layers = [i.strip() for i in self.args.global_compare_layers.split(',')]
            layers_rate = [float(i.split(':')[-1].strip()) for i in global_compare_layers if ':' in i]
            global_compare_layers = [i.split(':')[0].strip() for i in global_compare_layers]
            all_pre_layers = []
            for layer in global_compare_layers:
                pre_layers = []
                find_all_pre_layers(pre_layers, layer, self.parser)
                all_pre_layers.extend(pre_layers)
            all_pre_layers = set(all_pre_layers)
            if len(layers_rate) > 0:
                if len(global_compare_layers) != len(layers_rate):
                    print('global_compare_layers format error')
                    exit(1)
                if 1 != sum(layers_rate):
                    print('global_compare_layers rate error')
                    exit(1)
            else:
                layers_rate = len(global_compare_layers) * [1]
        for idx in range(self.num_sample):
            data = []
            for name in list(self.ref_activations[idx].keys()):
                data.append(self.ref_activations[idx][name][0])
            outputs = float_model.infer(data, global_compare_layers)
            predictions_gt.append(outputs)

        # set all layer as int8
        self.logger.print_info("run int8 mode: {}".format(self.fp32_mlir))
        outputs_cos = 0
        int8_model = MixQuantModel(self.fp32_mlir, self.chip, self.calib_table)
        for idx in range(self.num_sample):
            data = []
            for name in list(self.ref_activations[idx].keys()):
                data.append(self.ref_activations[idx][name][0])
            outputs = int8_model.infer(data, global_compare_layers)
            outputs_cos += self._loss(outputs, predictions_gt[idx], layers_rate)
        outputs_cos = outputs_cos / self.num_sample
        all_int8_cos = outputs_cos
        if outputs_cos > self.args.expected_cos:
            float_model.clean()
            int8_model.clean()
            self.enable_print()
            self.logger.print_info(
                f'job success, current int8 cos:{outputs_cos:.6f} is higher than expected_cos:{self.args.expected_cos:.6f},no need for mix precsion')
            exit(0)

        max_outputs_cos = all_int8_cos
        max_layer_cos, ret = None, None
        # mixing precision of each layer is automatically selected according to the cosine of each laye
        fp_layer_list = []
        layer_cos_list.clear()
        self.dot_log = net_dot_log('bias_correction', int8_model.parser, self.logger)
        self.dot_log.add_new_log_region('compute cos and run bias correction')
        fp_op_names = float_model.parser.get_op_name_list()
        top_ops = {op.name: op for op in self.parser.ops}
        int8_model = MixQuantModel(self.fp32_mlir, self.chip, self.calib_table)
        int8_op_names = int8_model.parser.get_op_name_list()
        full_op_list, cur_fp_op_idx = {}, 0
        for int8_op in int8_op_names:
            pos = fp_op_names.index(int8_op) if int8_op in fp_op_names else -1
            if -1 == pos:
                full_op_list[int8_op] = int8_model.parser.get_op_by_op_name(int8_op)
            else:
                for i in range(cur_fp_op_idx, pos + 1):
                    full_op_list[fp_op_names[i]] = float_model.parser.get_op_by_op_name(fp_op_names[i])
                cur_fp_op_idx = pos + 1
        print('full_op_list:', list(full_op_list.keys()))
        # try:
        pbar = tqdm(list(full_op_list.values()))
        for op in pbar:
            pbar.set_description("Processing {}".format(op.name))
            have_int8_next = False
            if op.name in int8_op_names:
                for next_int8_op in int8_model.parser.get_next_op_by_op_name(op.name):
                    if next_int8_op not in fp_layer_list:
                        have_int8_next = True
                        break
                if have_int8_next:
                    for idx in range(self.num_sample):
                        ret = self.gen_ref_tensor(idx, op.name, self.int8_activations, int8_model, True)
                    if ret:
                        self.dot_log.add_node_label(op.name, 'gen int8 tensor')
                else:
                    self.dot_log.add_node_label(op.name, f'next_layer is {self.mix_mode}, not need to gen int8 tensor')
            if op.name in fp_op_names:
                for idx in range(self.num_sample):
                    ret = self.gen_ref_tensor(idx, op.name, self.ref_activations, float_model)
                if ret:
                    self.dot_log.add_node_label(op.name, f'gen {self.mix_mode} tensor')
            if op.name not in top_ops:
                self.dot_log.add_node_label(op.name, 'op not in top dialect')
                continue
            if op.type in SKIP_OPERATION:
                pre_layers = self.parser.get_pre_op_by_op_name(op.name)
                for pre_layer in pre_layers:
                    if pre_layer in fp_layer_list:
                        self.dot_log.add_node_label(op.name, f'add {op.name} to fp_layer_list')
                        fp_layer_list.append(op.name)
                        break
                self.dot_log.add_node_label(op.name, f'meet quant skip op, type:{op.type}, continue')
                continue
            if op.name in fp_layer_list:
                self.dot_log.add_node_label(op.name, f'op is {self.mix_mode} layer, continue')
                continue
            if not have_int8_next:
                continue
            if op.name not in top_ops:
                self.dot_log.add_node_label(op.name, 'op only in int8model')
                continue
            if all_pre_layers is not None and op.name not in all_pre_layers:
                self.dot_log.add_node_label(op.name, f'not in all_pre_layers of compare layer, continue')
                continue
            top_op = top_ops[op.name]
            layer_cos, int8_mean, fp32_mean = 0, 0, 0
            for idx in range(self.num_sample):
                int8_out = self.get_input_int8_tensor(idx, top_op.name, True)
                fp32_out = self.get_input_fp32_tensor(idx, top_op.name)
                int8_mean += np.mean(int8_out, axis=(0, 2, 3))
                fp32_mean += np.mean(fp32_out, axis=(0, 2, 3))
                layer_cos += cos_sim(int8_out, fp32_out)
            old_layer_cos = layer_cos / self.num_sample
            print(f'old_layer_cos:{old_layer_cos:.6f}')
            if old_layer_cos < self.args.min_layer_cos and top_op.type == 'top.Conv' and len(top_op.opds) > 2:
                weight_file_dict = {}
                weight_file = np.load(weight_file_name)
                os.system(f'cp -f {weight_file_name} {weight_file_name}_tmpbk')
                for k in weight_file:
                    weight_file_dict[k] = weight_file[k]
                mean_diff = (fp32_mean - int8_mean) / self.num_sample
                # print(f'op:{top_op.name}, mean_diff:', mean_diff[:32])
                bias_float = weight_file_dict[top_op.opds[2]]
                bias_float += mean_diff
                weight_file_dict[top_op.opds[2]] = bias_float
                os.system(f'rm -f {weight_file_name}')
                np.savez(weight_file_name, **weight_file_dict)
                mix_model = MixQuantModel(self.fp32_mlir, self.chip, self.calib_table)
                extra_input = self.get_extra_input_tensor(top_op.name, int8_model.parser)
                layer_cos, outputs_cos = 0, 0
                for idx in range(self.num_sample):
                    input_data_dict, extra_input_data_dict = self.collect_op_input_tensor(idx, top_op.name, extra_input,
                                                                                          fp_layer_list)
                    outputs = mix_model.infer_from(top_op.name, input_data_dict, extra_input_data_dict,
                                                   global_compare_layers)
                    mix_layer_out = mix_model.module.get_fp32_tensor(top_op.name)
                    fp32_out = self.get_input_fp32_tensor(idx, top_op.name)
                    layer_cos += cos_sim(mix_layer_out, fp32_out)
                    outputs_cos += self._loss(outputs, predictions_gt[idx], layers_rate)
                layer_cos = layer_cos / self.num_sample
                outputs_cos = outputs_cos / self.num_sample
                if outputs_cos < max_outputs_cos or (max_layer_cos is not None and layer_cos < max_layer_cos):
                    os.system(f'cp -f {weight_file_name}_tmpbk {weight_file_name}')
                    print(
                        f'op:{top_op.name} bias correction revocation, layer_cos:{layer_cos:.6f}, outputs_cos:{outputs_cos:.6f}')
                else:
                    print(
                        f'op:{top_op.name} bias correction applied, layer_cos:{layer_cos:.6f}, outputs_cos:{outputs_cos:.6f}')
                layer_cos_list.append((top_op.name, old_layer_cos, layer_cos, outputs_cos))
                max_outputs_cos = outputs_cos if outputs_cos > max_outputs_cos else max_outputs_cos
                max_layer_cos = layer_cos if max_layer_cos is None or layer_cos > max_layer_cos else max_layer_cos
                if max_outputs_cos > self.args.expected_cos:
                    print(f'job success, current cos is higher than expected_cos:{self.args.expected_cos:.6f}')
                    break
            self.clear_ref_tensor(top_op.name, self.parser, self.ref_activations)
            self.clear_ref_tensor(top_op.name, int8_model.parser, self.int8_activations)
        # except Exception as err:
        #     self.logger.print_info('An exception happened: ' + str(err))
        #     pass
        self.dot_log.gen_dot_graph()
        int8_model.clean()
        float_model.clean()
        self.enable_print()
        self.logger.print_info('>>>run result:')
        self.logger.print_info(f'int8 outputs_cos:{all_int8_cos:.6f} old')
        self.logger.print_info('bias correction statistic:')
        for item in layer_cos_list:
            self.logger.print_info(
                f'  op:{item[0]}, old layer cos:{item[1]:.6f}, new layer cos:{item[2]:.6f}, new output cos:{item[3]:.6f}')
        self.logger.print_info(f'best mix model outputs_cos:{max_outputs_cos:.6f}')
        self.logger.print_info("total time:{}".format(time.time() - t0))

    def find_successor(self,op):
        # Conv -> Relu -> Conv or Conv -> Conv pattern supported.
        result = []
        nxt_op=self.parser.get_next_op_by_op_name(op)
        for node in nxt_op:
            if self.parser.get_op_type_by_op_name(node) in ['top.Relu', 'top.PRelu']:
                nxt_nxt_op=self.parser.get_next_op_by_op_name(node)
                for _node in nxt_nxt_op:
                    if self.parser.get_op_type_by_op_name(_node) == 'top.Conv':
                        result.append(_node)
                    else:
                        return []
            elif self.parser.get_op_type_by_op_name(node) == 'top.Conv':
                result.append(node)
            else:
                return []
        return result

    def converged(self,cur_weight, prev_weight, threshold=1e-4):
        norm_sum = 0
        norm_sum += np.linalg.norm(cur_weight[0] - prev_weight[0])
        norm_sum += np.linalg.norm(cur_weight[1] - prev_weight[1])
        self.logger.print_info('>>>loss:{}'.format(norm_sum))
        return norm_sum < threshold

    def weight_equalization(self):
        weight_file_name = MlirParser(self.fp32_mlir).module_weight_file
        module = pymlir.module()
        module.load(self.fp32_mlir)
        os.system('rm -rf tensor_diff_fp32_vs_int8;mkdir -p tensor_diff_fp32_vs_int8/')
        all_ops = self.parser.ops
        for op in all_ops:
            if self.parser.get_op_type_by_op_name(op.name) == 'top.Conv':
                succ = self.find_successor(op.name)
                if len(succ) != 1:
                    continue
                iter = 1
                weight_file_dict = {}
                while True:
                    weight_file = np.load(weight_file_name)
                    for k in weight_file:
                        weight_file_dict[k] = weight_file[k]
                    weight_first = weight_file[op.opds[1]]
                    weight_first_shape = module.get_tensor(op.opds[1]).shape
                    if weight_first.shape != weight_first_shape:
                        weight_first = weight_first.reshape(weight_first_shape)
                    new_weight_first = copy.deepcopy(weight_first)
                    if len(op.opds) > 2:
                        bias_first = weight_file[op.opds[2]]
                        new_bias_first = copy.deepcopy(bias_first)
                    next_op_name = succ[0]
                    next_op = self.parser.get_op_by_op_name(next_op_name)
                    weight_second = weight_file[next_op.opds[1]]
                    weight_second_shape = module.get_tensor(next_op.opds[1]).shape
                    if weight_second.shape != weight_second_shape:
                        weight_second = weight_second.reshape(weight_second_shape)
                    new_weight_second = copy.deepcopy(weight_second)
                    num_group = weight_first.shape[0] // weight_second.shape[1]
                    self.logger.print_info('Cross Layer WE: {} --- {} Groups: {} Iter: {}'.format(op.name, next_op.name, num_group, iter))
                    group_channels_i = weight_first.shape[0] // num_group
                    group_channels_o = weight_second.shape[0] // num_group
                    for g in range(num_group):
                        c_start_i = g * group_channels_i
                        c_end_i = (g + 1) * group_channels_i
                        weight_first_group = weight_first[c_start_i:c_end_i]
                        c_start_o = g * group_channels_o
                        c_end_o = (g + 1) * group_channels_o
                        weight_second_group = weight_second[c_start_o:c_end_o]
                        for ii in range(weight_second_group.shape[1]):
                            range_1 = np.abs(weight_first_group)[ii].max()
                            range_2 = np.abs(weight_second_group)[:, ii].max()
                            if range_1 < 1e-6:
                                range_1 = 0.
                            if range_2 < 1e-6:
                                range_2 = 0.
                            s = range_1 / np.sqrt(range_1 * range_2)
                            if np.isinf(s) or np.isnan(s):
                                s = 1.0
                            new_weight_first[c_start_i + ii] /= s
                            new_weight_second[c_start_o:c_end_o, ii] *= s
                            if len(op.opds) > 2:
                                new_bias_first[c_start_i + ii] /= s
                    if self.converged([weight_first, weight_second], [new_weight_first, new_weight_second]):
                        break
                    iter += 1
                    # Update layer.
                    weight_file_dict[op.opds[1]] = new_weight_first
                    weight_file_dict[next_op.opds[1]] = new_weight_second
                    if len(op.opds) > 2:
                        weight_file_dict[op.opds[2]] = new_bias_first
                    os.system(f'rm -f {weight_file_name}')
                    np.savez(weight_file_name, **weight_file_dict)
                    weight_file_dict.clear()
