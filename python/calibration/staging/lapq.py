# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import scipy.optimize as opt
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
import shutil

from .utils import logging
from .utils import quant_requant_active
from .utils import cal_loss
from .utils import cosine_sim
from .utils import lower_and_eval

import pymlir
pymlir.set_mem_mode("force_value_mem")


LAPQ_OPERATION = [
    # 'top.Conv', 'top.MatMul'#
    'top.MatMul'
]


class LossAwareQuant:
    def __init__(self, cali_table, args):
        self.cali_table = copy.deepcopy(cali_table)
        self.scales = cali_table.table.copy()
        self.scales4 = cali_table.table4.copy()
        self.orig_scales4 = {}
        self.trying_scales4 = {}
        self.finetune_layers = []
        self.point_idx = {}
        self.finetune_layer_weights = {}  # layers to fine tune, without skipped
        self.finetune_layer_bias = {}
        self.finetune_layer_weight_orig_ths = {}
        self.finetune_layer_weight_best_ths = {}
        self.mlir_file = args.mlir_file
        self.module = pymlir.module()
        self.module.load(self.mlir_file)
        self.parser = MlirParser(args.mlir_file)
        self.calibration_table = args.calibration_table
        self.qtable = args.qtable
        self.chip = args.chip
        self.batch_size = self.parser.get_batch_size()  # batch size of net
        self.input_num = self.parser.get_input_num()  # number of net inputs
        self.num_sample = 0
        self.search_loop = 2
        self.ref_tensors = {None}
        self.orig_weights = {}
        self.orig_bias = {}
        self.weight_file = self.parser.module_weight_file
        self.param_back = {}
        self.weights_scales = {}
        self.compare_quanted = True
        self.loger = None
        self.include_layers = self.init_include(args.quant_layers)
        print(f'Loss Aware Weight and Active Threshold')
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
            if top_ops[op].type in LAPQ_OPERATION and top_ops[op].name not in exclude and self.included(top_ops[op].name):
                if top_ops[op].type == 'top.Conv':
                    if int(top_ops[op].attrs['group'].split(':')[0]) > 1:
                        continue
                if len(top_ops[op].opds) > 1 and top_ops[op].opds[1] in self.module.all_weight_names:
                    self.finetune_layers.append(op)
                    self.finetune_layer_weights[op] = top_ops[op].opds[1]
                if len(top_ops[op].opds) > 2 and top_ops[op].opds[2] in self.module.all_weight_names:
                    self.finetune_layer_bias[op] = top_ops[op].opds[2]

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

    def quant_tensor(self, tensor, scale, is_weight, bit_width=8):
        tmp = tensor.copy()
        step = scale / (2**(bit_width-1)-1)
        qmax = 2**(bit_width-1) - 1
        qmin = -qmax
        if is_weight:
            qmin = -qmax + 1
        tmp = np.round(tmp/step)
        tmp = np.clip(tmp, qmin, qmax)
        tmp = tmp*step
        return tmp

    def lpnorm_loss(self, pred, ref, p):
        return np.mean(np.abs(pred.flatten()-ref.flatten())**p)

    def lpnorm_quant_loss(self, tensor, scale, p, is_weight, bit_width=8):
        tmp = self.quant_tensor(tensor, scale, is_weight, bit_width)
        return self.lpnorm_loss(tensor, tmp, p)

    def lpnorm_quant(self, tensor, p, is_weight, bit_width=8):
        opt_scale = opt.minimize_scalar(lambda scale: self.lpnorm_quant_loss(tensor, scale, p, is_weight, bit_width),
                                        bounds=(0, np.max(np.abs(tensor)))).x
        return opt_scale

    def calculate_sim(self, pred, ref):
        return cosine_sim(pred.flatten(), ref.flatten())

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

    def backup_weight_file(self):
        shutil.copy(self.weight_file, self.weight_file.replace(".npz", ".orig.npz"))

    def restore_weight_file(self):
        shutil.copy(self.weight_file.replace(".npz", ".orig.npz"), self.weight_file)

    def try_one_q(self, p, cali_samples, eval_samples):
        # backup original weight npz
        # loop the objected active and weight with norm p quant
        # lower and eval and get preds
        # restore weight npz
        self.backup_weight_file()
        self.scales4 = self.cali_table.table4.copy()
        for op in self.finetune_layers:
            for opd in [0, 1]:
                opdname = self.parser.get_opds_by_op_name(op)[opd]
                if opdname in self.finetune_layer_weights[op]:  # handle weight input
                    tmp = self.orig_weights[op].copy()
                    opt_th = self.lpnorm_quant(tmp, p, True, 4)
                    self.finetune_layer_weight_best_ths[opdname] = opt_th
                else:
                    tmpact = self.ref_tensors.get(opdname, cali_samples[0])
                    for s in cali_samples[1:]:
                        tmpact = np.concatenate((tmpact, self.ref_tensors.get(opdname, s)))
                    opt_th = self.lpnorm_quant(tmpact, p, False, 4)
                    self.trying_scales4[opdname] = opt_th
        tmp_cali = self.create_cali_table(False)
        self.create_weights(False)
        preds = lower_and_eval(self.mlir_file, 'INT4', self.chip, tmp_cali.out_table,
                               self.qtable, self.ref_tensors, eval_samples)
        outputname = self.module.output_names[0]
        tmploss = 0
        for s in eval_samples:
            ref = self.ref_tensors.get(outputname, s)
            pred = preds[s-self.num_sample//2]
            loss = 1 - self.calculate_sim(pred, ref)
            tmploss += loss
            print(f'try one q loss of {s} is {loss}, total :{tmploss}')
        tmploss /= len(eval_samples)
        print(f'Loss  of try is {tmploss}')
        del preds
        self.restore_weight_file()
        return tmploss

    def gather_points(self):
        points = []
        idx = 0
        self.point_idx = {}
        for op in self.finetune_layers:
            for opd in [0, 1]:
                opdname = self.parser.get_opds_by_op_name(op)[opd]
                if opdname in self.finetune_layer_weights[op]:  # handle weight input
                    points.append(self.finetune_layer_weight_best_ths[opdname])
                    self.point_idx[opdname] = idx
                    idx += 1
                else:
                    points.append(self.trying_scales4[opdname])
                    self.point_idx[opdname] = idx
                    idx += 1
        return points

    def scatter_points(self, points):
        for opdname in self.point_idx:
            if opdname in self.finetune_layer_weight_best_ths:
                self.finetune_layer_weight_best_ths[opdname] = points[self.point_idx[opdname]]
            elif opdname in self.trying_scales4:
                self.trying_scales4[opdname] = points[self.point_idx[opdname]]

    def create_cali_table(self, ifprint):
        tmp_cali = copy.deepcopy(self.cali_table)
        for op in self.trying_scales4:
            for op_ in tmp_cali.table4:
                if op_ == op:
                    if ifprint:
                        print(f'change active {op} from {tmp_cali.table4[op][0]} to {self.trying_scales4[op]}')
                    tmp_cali.table4[op][0] = self.trying_scales4[op]
        tmp_cali.out_table = tmp_cali.out_table + '.opt'
        tmp_cali.write()
        return tmp_cali

    def create_weights(self, ifprint):
        # quant weight with the th in points
        w = np.load(self.weight_file)
        ww = {}
        for opdname in w.files:
            if opdname in self.finetune_layer_weight_best_ths:
                if ifprint:
                    print(
                        f'quant weight {opdname} orig {self.finetune_layer_weight_orig_ths[opdname]} to {self.finetune_layer_weight_best_ths[opdname]}')
                tmp = self.quant_tensor(w[opdname], self.finetune_layer_weight_best_ths[opdname], True, 4)
                ww[opdname] = tmp
            else:
                ww[opdname] = w[opdname]
        np.savez(self.weight_file, **ww)

    def opt_lower_eval(self, points, cali_samples, eval_samples):
        # backup original weight npz
        # create new cali_table for lowerring
        # create new weight for lowerring with quanted int4 op
        # lower and eval and get preds
        # restore weight npz
        print(points)
        self.backup_weight_file()
        self.scatter_points(points)
        tmp_cali = self.create_cali_table(False)
        self.create_weights(False)
        preds = lower_and_eval(self.mlir_file, 'INT4', self.chip, tmp_cali.out_table,
                               self.qtable, self.ref_tensors, eval_samples)
        outputname = self.module.output_names[0]  # or should get the lowerred module outputname
        tmploss = 0
        for s in eval_samples:
            ref = self.ref_tensors.get(outputname, s)
            pred = preds[s-self.num_sample//2]
            loss = 1 - self.calculate_sim(pred, ref)
            tmploss += loss
            # print(f'Orig loss of {s} is {loss}, total :{orig_loss}')
        tmploss /= len(eval_samples)
        print(f'Loss  of opt is {tmploss}')
        del preds
        self.restore_weight_file()
        return tmploss

    def opt_loop(self, opt_loop):
        print(f'opt loop is {opt_loop}')
        opt_loop += 1

    def learning(self):
        total = len(self.finetune_layers)
        outputname = self.module.output_names[0]
        orig_loss = 0
        cali_samples = range(0, self.num_sample//2)
        eval_samples = range(self.num_sample//2, self.num_sample)
        '''
        preds = lower_and_eval(self.mlir_file, 'INT4', self.chip, self.calibration_table, self.qtable, self.ref_tensors, eval_samples)
        for s in eval_samples:
            ref = self.ref_tensors.get(outputname, s)
            pred = preds[s-self.num_sample//2]
            loss = 1 - self.calculate_sim(pred, ref)
            orig_loss += loss
            print(f'Orig loss of {s} is {loss}, total :{orig_loss}')
        orig_loss /= (self.num_sample//2)
        print(f'Orig Loss is {orig_loss}')
        del preds

        ps = np.linspace(2, 4, 10)
        losses = []
        for p in tqdm(ps):
            loss = self.try_one_q(p, cali_samples, eval_samples)
            losses.append(loss.item())
            print("(p, loss) - ({}, {})".format(p, loss.item()))

        z = np.polyfit(ps, losses, 2)
        y = np.poly1d(z)
        p_intr = y.deriv().roots[0]
        # loss_opt = y(p_intr)
        '''
        # p_intr = calcuated as about 10, got 0, orig is about 0.01
        # p_intr = 2, get 0.112/0.288, fev = 100, iter = 5, input_num = 60

        # p_intr = 1.414  # fev = 500, iter = 5, input_num = 60, get -th,crash
        # p_intr = 3.9  # fev = 100 iter = 2, input_num 60, get 0.02 0.012
        p_intr = 1.9  # fev = 100 iter = 2, input_num 60, get 0.122 0.24

        print("p intr: {:.2f}".format(p_intr))

        lp_loss = self.try_one_q(p_intr, cali_samples, eval_samples)
        points = np.array(self.gather_points())
        min_options = {}
        min_options['maxiter'] = self.search_loop
        min_options['maxfev'] = 100
        # min_options['maxiter'] = args.maxiter
        # min_options['maxfev'] = args.maxfev

        min_method = "Powell"
        # method = coord_descent if min_method == 'CD' else min_method
        method = min_method
        opt_loop = 0
        res = opt.minimize(lambda points: self.opt_lower_eval(points, cali_samples, eval_samples), points,
                           method=method, options=min_options, callback=lambda opt_loop: self.opt_callback(opt_loop))
        print(res)
        self.scatter_points(points)
        tmp_cali = self.create_cali_table(True)
        self.create_weights(True)
        print('done!')

        # update weight and update threhold of activate
        # self.save_weights()
