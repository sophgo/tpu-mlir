# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from utils.mlir_shell import mlir_lowering
from calibration.data_selector import DataSelector
from utils.preprocess import preprocess
from utils.mlir_parser import MlirParser
import numpy as np
import torch
import os
import sys
import time
import copy
from tqdm import tqdm

from .utils import logging
from .utils import quant_requant_active
from .utils import cal_loss
from .utils import cosine_sim

import pymlir
pymlir.set_mem_mode("force_value_mem")


EASY_QUANT_OPERATION = [
    # 'top.Conv', 'top.MatMul'#
    'top.MatMul'
]


class EasyQuant:
    def __init__(self, args):
        self.scales = None
        self.scales4 = None
        self.orig_scales4 = {}
        self.best_scales4 = {}
        self.finetune_layers = []
        self.finetune_layer_weights = {}  # layers to fine tune, without skipped
        self.finetune_layer_bias = {}
        self.finetune_layer_weight_orig_ths = {}
        self.finetune_layer_weight_best_ths = {}
        self.mlir_file = args.mlir_file
        self.module = pymlir.module()
        self.module.load(self.mlir_file)
        self.parser = MlirParser(args.mlir_file)
        self.batch_size = self.parser.get_batch_size()  # batch size of net
        self.input_num = self.parser.get_input_num()  # number of net inputs
        self.num_sample = 0
        self.search_loop = 2
        self.ref_tensors = {None}
        self.pre_loss = {}
        self.post_loss = {}
        self.loss = {}
        self.chip = args.chip
        self.orig_weights = {}
        self.orig_bias = {}
        self.weight_file = self.parser.module_weight_file
        self.param_back = {}
        self.weights_scales = {}
        self.compare_quanted = True
        self.loger = None
        self.alpha = 0.1
        self.beta = 2
        self.steps = 100
        self.active_only = not args.quantweight
        self.include_layers = self.init_include(args.quant_layers)
        print(f'Search both Weight and Active Threshold')
        self.support_unsigned = False
        self.get_finetune_ops(args.excepts)
        self.backup_weights()
        w = np.load(self.weight_file, allow_pickle=True)
        for k in w:
            self.param_back[k] = w[k]
            for kk in self.finetune_layer_weights:
                if k == self.finetune_layer_weights[kk]:
                    th = np.max(np.abs(w[k]))
                    self.finetune_layer_weight_orig_ths[self.finetune_layer_weights[kk]] = th
                    self.finetune_layer_weight_best_ths[self.finetune_layer_weights[kk]] = th

    def included(self, op):
        if len(self.include_layers) == 0:
            return True
        elif op in self.include_layers:
            return True
        else:
            return False

    def init_include(self, quant_layers):
        if quant_layers == "":
            return []
        f = open(quant_layers, 'r')
        layers = []
        for line in f:
            if len(line) > 0 and (not line.startswith("#")):
                layers.append(line.strip())
        return layers

    def get_finetune_ops(self, excepts):
        top_ops = {op.name: op for op in self.parser.ops}
        exclude = excepts.split(',')
        for op in top_ops:
            if top_ops[op].type in EASY_QUANT_OPERATION and top_ops[op].name not in exclude and self.included(top_ops[op].name):
                if top_ops[op].type == 'top.Conv':
                    if int(top_ops[op].attrs['group'].split(':')[0]) > 1:
                        continue
                if len(top_ops[op].opds) > 1 and top_ops[op].opds[1] in self.module.all_weight_names:
                    self.finetune_layers.append(op)
                    self.finetune_layer_weights[op] = top_ops[op].opds[1]
                if len(top_ops[op].opds) > 2 and top_ops[op].opds[2] in self.module.all_weight_names:
                    self.finetune_layer_bias[op] = top_ops[op].opds[2]

    def search_best_th(self, op, opd):
        if len(self.orig_scales4) == 0:
            for op_ in self.scales4:
                self.orig_scales4[op_] = np.maximum(np.abs(self.scales4[op_][1]), np.abs(self.scales4[op_][2]))
            self.best_scales4 = self.orig_scales4
        opdname = self.parser.get_opds_by_op_name(op)[opd]
        otheropd = self.parser.get_opds_by_op_name(op)[1-opd]
        if opdname in self.finetune_layer_weights[op]:  # handle weight input
            best_sim = np.zeros(self.steps)
            for thidx in np.arange(self.steps):
                for sample_idx in np.arange(self.num_sample):
                    th = self.finetune_layer_weight_orig_ths[opdname]*self.alpha + self.finetune_layer_weight_orig_ths[opdname]*(
                        self.beta-self.alpha)/self.steps * (thidx + 1)
                    self.quant_weight(op, th, 4)
                    self.quant_active(otheropd, self.best_scales4[otheropd], sample_idx, 4)
                    outputs = self.module.invoke_at(op).copy()
                    # get from ref?
                    self.restore_weight(op)
                    refoutputs = self.module.invoke_at(op).copy()
                    best_sim[thidx] += self.calculate_sim(outputs, refoutputs)
            best_th = (np.argmax(best_sim) + 1) * self.finetune_layer_weight_orig_ths[opdname] * (
                self.beta - self.alpha)/self.steps + self.finetune_layer_weight_orig_ths[opdname] * self.alpha
            # print(f'best SIM after searching {opdname} is {best_sim[np.argmax(best_sim)]}')
            # print(f'best th of {opdname} change from {self.finetune_layer_weight_best_ths[opdname]} to {best_th}')
            self.finetune_layer_weight_best_ths[opdname] = best_th

        else:  # handle tensor input, and other input may be tensor or weight
            best_sim = np.zeros(self.steps)
            for thidx in np.arange(self.steps):
                th = self.orig_scales4[opdname]*(self.beta-self.alpha)/self.steps * \
                    (thidx+1) + self.orig_scales4[opdname]*self.alpha
                for sample_idx in np.arange(self.num_sample):
                    if otheropd in self.finetune_layer_weights[op]:
                        otherth = self.finetune_layer_weight_best_ths[otheropd]
                        self.quant_weight(op, otherth, 4)
                    else:
                        self.quant_active(otheropd, self.best_scales4[otheropd], sample_idx, 4)
                    self.quant_active(opdname, th, sample_idx, 4)
                    outputs = self.module.invoke_at(op).copy()
                    # get from ref?
                    if otheropd in self.finetune_layer_weights[op]:
                        self.restore_weight(op)
                    else:
                        self.restore_active(otheropd, sample_idx)
                    self.restore_active(opdname, sample_idx)
                    refoutputs = self.module.invoke_at(op).copy()
                    best_sim[thidx] += self.calculate_sim(outputs, refoutputs)
            best_th = (np.argmax(best_sim)+1) * self.orig_scales4[opdname] / \
                self.steps + self.orig_scales4[opdname]*self.alpha
            print(f'best SIM after searching {op} {opdname} is {best_sim[np.argmax(best_sim)]}')
            # print(f'best th of {opdname} change from {self.best_scales4[opdname]} to {best_th}')
            self.best_scales4[opdname] = best_th

    def bias_correction(self, op):
        tmpweight = self.orig_weights[op].copy()
        weightname = self.finetune_layer_weights[op]
        self.quant_weight(op, self.finetune_layer_weight_best_ths[weightname])
        qweight = self.module.get_tensor(self.finetune_layer_weights[op])
        nweight = qweight - tmpweight
        self.module.set_tensor(self.finetune_layer_weights[op], nweight)
        tmpout = []
        for sample_idx in np.arange(self.num_sample):
            opdname = self.parser.get_opds_by_op_name(op)[0]
            tmpactive = self.ref_tensors.get(opdname, sample_idx)
            self.module.set_tensor(opdname, tmpactive)
            outputs = self.module.invoke_at(op)
            if len(tmpout) == 0:
                tmpout = outputs
            else:
                tmpout += outputs
        tmpout /= self.num_sample
        shape = tmpout.shape
        origbias = self.orig_bias[op].copy()
        tmpbias = np.sum(tmpout.reshape(-1, tmpout.shape[len(shape)-1]),
                         axis=0).reshape(origbias.shape) / (tmpout.size / tmpout.shape[-1])
        self.update_bias(op, tmpbias)

    def learn_one(self, total):
        pbar_detail = tqdm(np.arange(len(self.finetune_layers)))
        pbar_detail.set_description("Searching Weight and Active scales")
        layer_count = 0
        for op in self.finetune_layers:
            layer_count += 1
            pbar_detail.set_postfix_str(
                f"Searching {layer_count} [Total: {len(self.finetune_layers)}]")
            pbar_detail.update()
            left = True
            for loop in np.arange(self.search_loop):
                for opd in [0, 1]:
                    self.search_best_th(op, opd)
            for opd in [0, 1]:
                opdname = self.parser.get_opds_by_op_name(op)[opd]
                isweight = False
                for op_ in self.finetune_layer_weights:
                    if opdname == self.finetune_layer_weights[op_]:
                        isweight = True
                if isweight:
                    self.restore_weight(op)
                # else:
                    # self.ref_tensors.consumed_tensor(opdname)
        for op in self.best_scales4:
            if self.best_scales4[op] != self.scales4[op][0] and op in self.finetune_layers:
                print(f'change {op} from {self.scales4[op][0]} to {self.best_scales4[op]}')
                self.scales4[op][0] = self.best_scales4[op]
        if not self.active_only:
            pbar_detail = tqdm(np.arange(len(self.finetune_layers)))
            pbar_detail.set_description("Bias Correction")
            for op in self.finetune_layers:
                if op not in self.finetune_layer_weights:
                    continue
                weightname = self.finetune_layer_weights[op]
                if self.finetune_layer_weight_best_ths[weightname] != self.finetune_layer_weight_orig_ths[weightname]:
                    print(
                        f'change {weightname} from {self.finetune_layer_weight_orig_ths[weightname]} to {self.finetune_layer_weight_best_ths[weightname]}')
                    pbar_detail.update()
                    weight = self.orig_weights[op].copy()
                    bitwidth = 4
                    qmax = 2**(bitwidth-1)-1
                    qmin = - 2**(bitwidth-1) + 1
                    step = self.finetune_layer_weight_best_ths[weightname] / (2**(bitwidth-1)-1)
                    weight = np.clip(np.round(weight / step), qmin, qmax)
                    weight = weight * step
                    self.update_weight(op, weight)
                    if op in self.finetune_layer_bias:  # when with weight and with bias, can do bias correction
                        pbar_detail.set_postfix_str(
                            f"Correct {op}")
                        pbar_detail.update()
                        self.bias_correction(op)
        else:  # output new weight th to file
            f = open('new_w_th.txt', 'w')
            for op in self.finetune_layers:
                if op not in self.finetune_layer_weights:
                    continue
                weightname = self.finetune_layer_weights[op]
                f.write(f'{weightname} {self.finetune_layer_weight_best_ths[weightname]}\n')
            f.close()

    def backup_weights(self):
        for op in self.finetune_layers:
            self.orig_weights[op] = copy.deepcopy(self.module.get_tensor(self.finetune_layer_weights[op]))
            if op in self.finetune_layer_bias:
                self.orig_bias[op] = copy.deepcopy(self.module.get_tensor(self.finetune_layer_bias[op]))
            oc = self.orig_weights[op].shape[0]
            if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
                self.weights_scales[op] = np.max(np.abs(self.orig_weights[op].reshape(oc, -1)), axis=1)
                self.weights_scales[op] = np.where(self.weights_scales[op] > 1e-8, self.weights_scales[op], 1e-8)
            elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
                self.weights_scales[op] = np.max(np.abs(self.orig_weights[op]))
                self.weights_scales[op] = np.where(self.weights_scales[op] > 1e-8, self.weights_scales[op], 1e-8)
            else:
                print("not support!")
                sys.exit(1)

    def quant_weight(self, op, scale, bitwidth=8):
        tmpweight = self.orig_weights[op].copy()
        step = scale / (2**(bitwidth-1)-1)
        tmpweight = np.round(tmpweight/step)
        tmpweight = np.clip(tmpweight, -2**(bitwidth-1)+1, 2**(bitwidth-1)-1)  # use -7 to 7 for right absmax
        tmpweight = tmpweight*step
        self.module.set_tensor(self.finetune_layer_weights[op], tmpweight)

    def quant_active(self, active_name, scale, sample_idx, bitwidth=8):
        tmpactive = self.ref_tensors.get(active_name, sample_idx).copy()
        step = scale / (2**(bitwidth-1)-1)
        tmpactive = np.round(tmpactive/step)
        tmpactive = np.clip(tmpactive, -2**(bitwidth-1), 2**(bitwidth-1)-1)
        tmpactive = tmpactive*step
        self.module.set_tensor(active_name, tmpactive)

    def restore_weight(self, op):
        self.module.set_tensor(self.finetune_layer_weights[op], self.orig_weights[op])

    def restore_active(self, active_name, sample_idx):
        tmpactive = self.ref_tensors.get(active_name, sample_idx).copy()
        self.module.set_tensor(active_name, tmpactive)

    def l2norm(self, pred, ref):
        return - np.sum((pred.flatten() - ref.flatten())**2)

    def wl2norm(self, pred, ref):
        return - np.sum(np.abs(ref.flatten()) * (pred.flatten() - ref.flatten())**2)

    def wsl2norm(self, pred, ref):
        return - np.sum(ref.flatten() * (ref.flatten() - pred.flatten())**2)

    def hessian(self, pred, ref):
        return None

    def calculate_sim(self, pred, ref):
        COS = 1  # 0.588 0.592 from 0.4 0.658
        L2NORM = 0  # 0.594 0.776
        WL2NORM = 0  # 0.536 0.754
        WSL2NORM = 0  # 0.248 0.484
        if COS:
            return cosine_sim(pred.flatten(), ref.flatten())
        elif L2NORM:
            return self.l2norm(pred.flatten(), ref.flatten())
        elif WL2NORM:
            return self.wl2norm(pred.flatten(), ref.flatten())
        elif WSL2NORM:
            return self.wsl2norm(pred.flatten(), ref.flatten())

    def set_op_inputs(self, op, loop, bitwidth=8, quant=True):
        pre_ops = self.parser.get_pre_op_by_op_name(op)
        for pop in pre_ops:
            shape = self.ref_tensors.get(pop, loop).shape
            if pop not in self.module.all_tensor_names:
                if self.parser.get_op_type_by_op_name(pop) == 'top.Reshape':
                    pre_pre_ops = self.parser.get_pre_op_by_op_name(pop)
                    pop = pre_pre_ops[0]
                else:
                    print(f"{op} input {pop} not in all tensor list")
                    sys.exit(1)
            d = self.ref_tensors.get(pop, loop).reshape(shape)
            if bitwidth == 4:
                scale = self.scales4[pop][0]
                unsigned = self.scales4[op][1] >= 0 and self.scales4[op][2] >= 0
            else:
                scale = self.scales[pop][0]
                unsigned = self.scales[op][1] >= 0 and self.scales[op][2] >= 0
            if scale != 1.0 and quant:
                if self.support_unsigned:
                    d = quant_requant_active(d, scale, unsigned, bits=bitwidth)
                else:
                    d = quant_requant_active(d, scale, False, bits=bitwidth)
            self.module.set_tensor(pop, d)

    def get_op_input0(self, op, loop, bitwidth=8, quanted=False):
        pre_ops = self.parser.get_pre_op_by_op_name(op)
        for pop in pre_ops:
            shape = self.ref_tensors.get(pop, loop).shape
            if pop not in self.module.all_tensor_names:
                if self.parser.get_op_type_by_op_name(pop) == 'top.Reshape':
                    pre_pre_ops = self.parser.get_pre_op_by_op_name(pop)
                    pop = pre_pre_ops[0]
                else:
                    print(f"{op} input {pop} not in all tensor list")
                    sys.exit(1)
            d = self.ref_tensors.get(pop, loop).reshape(shape).copy()
            if bitwidth == 4:
                scale = self.scales4[pop][0]
                unsigned = self.scales4[op][1] >= 0 and self.scales4[op][2] >= 0
            else:
                scale = self.scales[pop][0]
                unsigned = self.scales[op][1] >= 0 and self.scales[op][2] >= 0
            if scale != 1.0 and quanted:
                if self.support_unsigned:
                    d = quant_requant_active(d, scale, unsigned, bits=bitwidth)
                else:
                    d = quant_requant_active(d, scale, False, bits=bitwidth)
            return d

    def update_weight(self, op, weight):
        self.param_back[self.finetune_layer_weights[op]] = weight

    def update_bias(self, op, bias):
        self.param_back[self.finetune_layer_bias[op]] = bias

    def save_weights(self):
        os.rename(self.weight_file, self.weight_file.replace(".npz", ".bak.npz"))
        np.savez(self.weight_file, **self.param_back)

    def learning(self):
        total = len(self.finetune_layers)
        self.learn_one(total)

        # update weight and update threhold of activate
        self.save_weights()
