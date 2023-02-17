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
from utils.misc import cos_sim
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.net_dot_log import net_dot_log

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
    "cv180x": "BF16"
}

def find_all_pre_layers(all_pre_layers, op_name, parser, exist_ref_layers):
    pre_layers = parser.get_pre_op_by_op_name(op_name)
    if len(pre_layers) > 0:
        for pre_layer in pre_layers:
            if pre_layer not in all_pre_layers:
                all_pre_layers.append(pre_layer)
            if pre_layer in exist_ref_layers:
                find_all_pre_layers(all_pre_layers, pre_layer, parser, exist_ref_layers)
    else:
        if op_name not in all_pre_layers:
            all_pre_layers.append(op_name)

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

        self.quanted_mlir_file = '{}.{}.tune.mlir'.format(fp32_mlir, 'mix' if mix_table else self.mode)
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

    def infer_from(self, top_op_name, input_data_dict: dict, extra_input_data_dict: dict):
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

class MixPrecSearcher:
    def __init__(self, args):
        self.args = args
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
        # self.logger = setup_logger('MixPrecSearcher', log_level=log_level)
        self._init_inputs(args)
        self.dot_log = net_dot_log('mix_prec_result')

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

    def _sqnr_loss(self, preds, gt_preds):
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

    def _cos_loss(self, preds, gt_preds):
        cos = 0
        for name1, name2 in zip(gt_preds, preds):
            a = gt_preds[name1]
            b = preds[name2]
            if a.dtype != b.dtype or a.shape != b.shape:
                raise RuntimeError("Calc loss fail:{} vs {}".format(name1, name2))
            cos += cos_sim(a.reshape(-1), b.reshape(-1))
        return cos / len(preds)

    def _loss(self, preds, gt_preds, type='cos'):
        assert type == 'cos' or type == 'sqnr'
        if type == 'cos':
            return self._cos_loss(preds, gt_preds)
        elif type == 'sqnr':
            return self._sqnr_loss(preds, gt_preds)

    def get_input_fp32_tensor(self, i, op_name):
        if op_name in self.ref_activations[i]:
            return self.ref_activations[i][op_name][0]
        raise Exception('error, idx:{} op_name:{} not in ref_activations'.format(i, op_name))


    def get_input_int8_tensor(self, i, op_name, ret_fp32 = False):
        if op_name in self.int8_activations[i]:
            if ret_fp32:
                return self.int8_activations[i][op_name][-1]
            else:
                return self.int8_activations[i][op_name][0]
        raise Exception('error, idx:{} op_name:{} not in int8_activations'.format(i, op_name))

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
                        print(f'pop {layer}')
                    else:
                        data_dict[i][layer][1] -= 1
                        print(f'dec refcount {layer}')
        print('data_dict status:')
        for i in data_dict:
            for layer in data_dict[i]:
                print(f'idx:{i}, layer:{layer} exist, refcount:{data_dict[i][layer][1]}')

    def gen_ref_tensor(self, i, op_name, data_dict, model, is_int8_data = False):
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
            value = model.module.invoke_at(op_name)
            print(f'invoke_at {op_name}')
            fp32_v = None
            if is_int8_data:
                fp32_v = model.module.get_fp32_tensor(op_name)
            count = model.parser.get_user_count_by_op_name(op_name)
            if fp32_v is None:
                data_dict[i][op_name] = [value, count]
            else:
                data_dict[i][op_name] = [value, count, fp32_v]
        return True

    def visual_tensor_diff(self, name, cos, int8_out, fp32_out):
        data_size = fp32_out.size
        max_sampling = 10000
        if 'max_sampling' in self.debug_cmd:
            max_sampling = int(self.debug_cmd['max_sampling'])
        if data_size > max_sampling:
            step = data_size//max_sampling
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
        fig.update_layout(height=400*rows, width=900)
        fig.update_layout(margin=dict(l=5, r=10, t=20, b=0))
        fig.update_layout(legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.95))
        tmpstr = './tensor_diff_fp32_vs_int8/{}'.format(name.replace('/','_'))
        if 'dump_tensor_diff_by_html' in self.debug_cmd:
            fig.write_html(tmpstr+'.html')
            return tmpstr+'.html'
        fig.write_image(tmpstr+'.png')
        return tmpstr+'.png'

    def get_extra_input_tensor(self, op_name, top_op_names):
        first_seg,second_seg,extra_input = [],[],[]
        for i, name in enumerate(top_op_names):
            if op_name == name:
                first_seg = top_op_names[:i]
                second_seg = top_op_names[i+1:]
                # print(op_name, ',first_seg:',first_seg,',second_seg:',second_seg)
                break
        for name in second_seg:
            extra_input.extend([i for i in self.parser.get_pre_op_by_op_name(name) if i in first_seg])
        extra_input = set(extra_input)
        return extra_input

    def clear_ref_tensor(self, op_name, parser, data_dict):
        all_pre_layers = []
        find_all_pre_layers(all_pre_layers, op_name, parser, list(data_dict[0].keys()))
        for idx in range(self.num_sample):
            self.clear_tensor(idx, op_name, data_dict, parser, all_pre_layers)

    def collect_op_input_tensor(self, idx, op_name, extra_input, fp32_layer_list = None):
        input_data_dict = {}
        for i, input in enumerate(self.parser.get_pre_op_by_op_name(op_name)):
            print(f'{op_name}\'s top input{i}:{input}')
            if fp32_layer_list is not None and input in fp32_layer_list:
                input_data_dict[input] = self.get_input_fp32_tensor(idx, input)
            else:
                input_data_dict[input] = self.get_input_int8_tensor(idx, input)
            if input in extra_input:
                extra_input.remove(input)
        extra_input_data_dict = {}
        for i, input in enumerate(extra_input):
            print(f'{op_name}\'s other input{i}:{input}')
            if fp32_layer_list is not None and input in fp32_layer_list:
                extra_input_data_dict[input] = self.get_input_fp32_tensor(idx, input)
            else:
                extra_input_data_dict[input] = self.get_input_int8_tensor(idx, input)
        return input_data_dict, extra_input_data_dict

    def run(self):
        t0 = time.time()
        layer_cos_list = list()
        predictions_gt = list()
        os.system('rm -rf tensor_diff_fp32_vs_int8;mkdir -p tensor_diff_fp32_vs_int8/')

        # set all layer as float
        print("run float mode: {}".format(self.fp32_mlir))
        float_model = MixQuantModel(self.fp32_mlir, self.chip)
        for idx in range(self.num_sample):
            net_input = list(self.ref_activations[idx].keys())[0]
            outputs = float_model.infer(self.ref_activations[idx][net_input])
            predictions_gt.append(outputs)

        #mixing precision of each layer is automatically selected according to the cosine of each laye
        fp32_layer_list = []
        layer_cos_list.clear()
        self.dot_log.add_new_log_region('compute cos and set fp32 layer')
        top_op_names = self.parser.get_op_name_list()
        max_fp32_layer_num = len(top_op_names)//4
        top_ops = {op.name:op for op in self.parser.ops}
        int8_model = MixQuantModel(self.fp32_mlir, self.chip, self.calib_table)
        mix_th = 0.99 if 'mix_th' not in self.debug_cmd else float(self.debug_cmd['mix_th'])
        ret = None
        for op in int8_model.parser.ops:
            pre_ops = int8_model.parser.get_pre_op_by_op_name(op.name)
            op_type = op.type.split('.')[-1]
            if op.type == 'top.Conv' and int(op.attrs['group'].split(':')[0].strip()) > 1:
                op_type = f'{op_type}_depth'
            self.dot_log.append_input_edge_and_node(pre_ops, op.name, op_type)
        try:
            pbar = tqdm(int8_model.parser.ops)
            for op in pbar:
                pbar.set_description("Processing {}".format(op.name))
                have_int8_next = False
                for next_int8_op in int8_model.parser.get_next_op_by_op_name(op.name):
                    if next_int8_op not in fp32_layer_list:
                        have_int8_next = True
                        break
                if have_int8_next:
                    for idx in range(self.num_sample):
                        ret = self.gen_ref_tensor(idx, op.name, self.int8_activations, int8_model, True)
                    if ret:
                        self.dot_log.add_node_label(op.name, 'gen int8 tensor')
                else:
                    self.dot_log.add_node_label(op.name, 'next_layer is fp32, not need to gen int8 tensor')
                if op.name not in top_ops:
                    self.dot_log.add_node_label(op.name, 'op only in int8model')
                    continue
                top_op = top_ops[op.name]
                for idx in range(self.num_sample):
                    ret = self.gen_ref_tensor(idx, top_op.name, self.ref_activations, float_model)
                if ret:
                    self.dot_log.add_node_label(top_op.name, 'gen fp32 tensor')
                if top_op.type in SKIP_OPERATION:
                    pre_layers = self.parser.get_pre_op_by_op_name(top_op.name)
                    for pre_layer in pre_layers:
                        if pre_layer in fp32_layer_list:
                            self.dot_log.add_node_label(top_op.name, f'add {top_op.name} to fp32_layer_list')
                            fp32_layer_list.append(top_op.name)
                            break
                    self.dot_log.add_node_label(top_op.name, 'meet quant skip op, type:{top_op.type}, continue')
                    continue
                if top_op.name in fp32_layer_list:
                    self.dot_log.add_node_label(top_op.name, 'op is fp32layer, continue')
                    continue
                layer_cos = 0
                for idx in range(self.num_sample):
                    int8_out = self.get_input_int8_tensor(idx, top_op.name, True)
                    fp32_out = self.get_input_fp32_tensor(idx, top_op.name)
                    cos = cos_sim(int8_out, fp32_out)
                    layer_cos += cos
                    if 'visual_tensor' in self.debug_cmd and (top_op.name in self.debug_cmd['visual_tensor'] or self.debug_cmd['visual_tensor'] == 'all'):
                        plot_path = self.visual_tensor_diff(f'int8_model_{top_op.name}_{idx}', cos, int8_out, fp32_out)
                        if idx == 0:
                            self.dot_log.add_node_label(top_op.name, 'visual_tensor diff enable, please click')
                            self.dot_log.add_node_attr(top_op.name, 'URL', plot_path) #shape='box', fillcolor=next_node_fillcolor , style='filled'
                avg_cos = layer_cos / self.num_sample
                self.dot_log.add_node_label(top_op.name, f'int8 cos:{avg_cos:.6f}')
                layer_cos_list.append((top_op.name, avg_cos))
                outputs_cos = 0
                if avg_cos < mix_th:
                    fp32_layer_list.append(top_op.name)
                    next_top_ops = self.parser.get_next_op_by_op_name(top_op.name)
                    tmp = ','.join(next_top_ops)
                    self.dot_log.add_node_label(top_op.name, f'cos too low, set fp32 to {top_op.name} and next_top_ops:{tmp}')
                    fp32_layer_list.extend(next_top_ops)
                    mix_table = self._gen_mix_table(fp32_layer_list)
                    mix_model = MixQuantModel(self.fp32_mlir, self.chip, self.calib_table, mix_table)
                    extra_input = self.get_extra_input_tensor(top_op.name, top_op_names)
                    for idx in range(self.num_sample):
                        input_data_dict, extra_input_data_dict = self.collect_op_input_tensor(idx, top_op.name, extra_input, fp32_layer_list)
                        tmp1 = ','.join(list(input_data_dict.keys()))
                        tmp2 = '' if len(extra_input_data_dict) == 0 else ',extra_input_data_dict:' + ','.join(list(extra_input_data_dict.keys()))
                        if idx == 0:
                            self.dot_log.add_node_label(top_op.name, f'input_data_dict:{tmp1}{tmp2}, call infer_from')
                        outputs = mix_model.infer_from(top_op.name, input_data_dict, extra_input_data_dict)
                        mix_layer_out = mix_model.module.get_fp32_tensor(top_op.name)
                        fp32_out = self.get_input_fp32_tensor(idx, top_op.name)
                        cos = cos_sim(mix_layer_out.reshape(-1), fp32_out.reshape(-1))
                        if idx == 0:
                            self.dot_log.add_node_label(top_op.name, f'first fp32 layer:{top_op.name} cos:{cos}')
                        outputs_cos += self._loss(outputs, predictions_gt[idx])
                        for next_top_op in next_top_ops:
                            count = self.parser.get_user_count_by_op_name(next_top_op)
                            self.int8_activations[idx][next_top_op] = [None,count,None]
                            next_ops = mix_model.parser.get_next_op_by_op_name(next_top_op)
                            print(f'{next_top_op}\'s next_ops:',','.join(next_ops))
                            for next_op in next_ops:
                                if mix_model.parser.get_op_by_op_name(next_op).type == "tpu.Cast":
                                    if idx == 0:
                                        self.dot_log.add_node_label(top_op.name, f'use {next_op} to replace {next_top_op}')
                                    self.int8_activations[idx][next_top_op][0] = mix_model.module.get_tensor(next_op)
                                    self.int8_activations[idx][next_top_op][2] = mix_model.module.get_fp32_tensor(next_op)
                    avg_cos = outputs_cos / self.num_sample
                    self.dot_log.add_node_label(top_op.name, f'current output cos:{avg_cos:.6f}')
                    mix_model.clean()
                    if avg_cos > self.args.expected_cos:
                        self.dot_log.add_node_label(top_op.name, f'job success, current cos is higher than expected_cos:{self.args.expected_cos}')
                        break
                    if len(fp32_layer_list) > max_fp32_layer_num:
                        self.dot_log.add_node_label(top_op.name, f'job fail, the number of layers of fp32 exceeded the maximum')
                        break
                self.clear_ref_tensor(top_op.name, self.parser, self.ref_activations)
                self.clear_ref_tensor(top_op.name, int8_model.parser, self.int8_activations)
        except Exception as err:
            print('An exception happened: ' + str(err))
            pass
        self.dot_log.gen_dot_graph()
        int8_model.clean()

        layer_cos_list = sorted(layer_cos_list, key=lambda x: x[1], reverse=False)
        with open(self.loss_table, "w") as f:
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# sample number: {}\n".format(self.num_sample))
            f.write("# chip: {}  mix_mode: {}\n".format(self.chip, self.mix_mode))
            f.write("###\n")
            for idx, layer in enumerate(layer_cos_list):
                loss_msg = "No.{:<4}: Layer: {:<50}\t\tCos: {}".format(idx, layer[0], layer[1])
                f.write("{}\n".format(loss_msg))
                print(loss_msg)
        with open(self.quantize_table, "w") as f:
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# sample number: {}\n".format(self.num_sample))
            f.write("# chip: {}  mix_mode: {}\n".format(self.chip, self.mix_mode))
            f.write("###\n")
            f.write("# op_name   quantize_mode\n")
            for layer in fp32_layer_list:
                f.write("{} {}\n".format(layer, self.mix_mode))
        print("Output mix quantization table to {}".format(self.quantize_table))
        print("total time:{}".format(time.time() - t0))

