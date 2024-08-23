# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import tensorboard
import tensorflow as tf
from utils.mlir_shell import mlir_lowering
from calibration.data_selector import DataSelector
from utils.preprocess import preprocess
from utils.mlir_parser import MlirParser
import numpy as np
import os
import sys
import copy
from scipy.special import expit
from tqdm import tqdm

from .utils import quant_requant_active
from .utils import cal_loss

import pymlir
pymlir.set_mem_mode("force_value_mem")

LEARNING_WEIGHT_OPERATION = [
    # 'top.Conv', 'top.MatMul'#
    'top.MatMul'
]

log_dir = "logs/histogram"
writer = tf.summary.create_file_writer(log_dir)


def log_alpha(name, alpha, step=0):
    with writer.as_default():
        tensor = tf.convert_to_tensor(alpha, dtype=tf.float32)
        tf.summary.histogram(name, tensor, step=step)


class SgdWeightOpt:
    def __init__(self, lr, momentum=0.0, nesterov=False, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.dampening = 0.0
        self.v = {}
        self.grd = {}
        self.loss = {}
        print(
            f'Learning Weight SGD, momentum is {self.momentum} nesterov is {self.nesterov} weight_decay is {self.weight_decay}')

    def cal_alpha(self, iter, alpha, loss, grd, unsigned=False):
        alpha = alpha - self.lr.cal_lr(iter) * grd
        return alpha

    def update_alpha(self, iter, op, alpha, mini_batch, unsigned=False):
        self.grd[op] = self.grd[op]/mini_batch
        self.loss[op] = self.loss[op]/mini_batch
        if self.weight_decay != 0.0:
            self.grd[op] = self.grd[op] + alpha*self.weight_decay
        if self.momentum != 0.0:
            if op in self.v:
                self.v[op] = self.v[op]*self.momentum + \
                    self.grd[op]*(1.0-self.dampening)
            else:
                self.v[op] = self.grd[op]
            if self.nesterov:
                self.grd[op] = self.v[op]*self.momentum + self.grd[op]
            else:
                self.grd[op] = self.v[op]

        alpha_new = np.clip(self.cal_alpha(
            iter, alpha, self.loss[op], self.grd[op], unsigned), -np.log(11), -np.log(1.2/1.1-1))
        self.reset_grd_loss(op)
        return alpha_new

    def update_loss(self, op, loss):
        if op in self.loss:
            self.loss[op] = self.loss[op] + loss
        else:
            self.loss[op] = loss

    def update_grd(self, op, grd):
        if op in self.grd:
            self.grd[op] = self.grd[op] + grd
        else:
            self.grd[op] = grd

    def reset_grd_loss(self, op):
        del self.grd[op]
        self.loss[op] = 0.0


class LearningAdaWeight:
    def __init__(self, args):
        self.chip = args.chip
        self.scales = None
        self.scales4 = None
        self.finetune_layers = []
        self.finetune_layer_weights = {}  # layers to fine tune, without skipped
        self.mlir_file = args.mlir_file
        self.module = pymlir.module()
        self.module.load(self.mlir_file)
        self.parser = MlirParser(args.mlir_file)
        self.batch_size = self.parser.get_batch_size()  # batch size of net
        self.input_num = self.parser.get_input_num()  # number of net inputs
        self.mini_batch = args.mini_batch
        self.num_sample = 0
        self.epoch = args.epoch
        self.ref_tensors = None
        self.pre_loss = {}
        self.post_loss = {}
        self.loss = {}
        self.grd = {}
        self.orig_bias = {}
        self.finetune_layer_bias = {}
        self.opt = None
        self.zeta = 1.1
        self.gamma = -0.1
        self.lam = 1.0
        self.beta_warmup = 0.2
        self.beta_start = 20
        self.beta_end = 2
        self.reg_param = 0.01
        self.orig_weights = {}
        self.weight_file = self.parser.module_weight_file
        self.param_back = {}
        self.weights_scales = {}
        self.alpha = {}
        self.momentum = args.momentum
        self.dampening = 0.0
        self.nesterov = args.nesterov
        self.weight_decay = args.weight_decay
        self.include_layers = self.init_include(args.quant_layers)
        self.compare_quanted = False
        self.loger = None
        print(
            f'Learning Weight, momentum is {self.momentum} nesterov is {self.nesterov} weight_decay is {self.weight_decay}')
        self.v = {}
        self.support_unsigned = False
        self.get_finetune_ops(args.excepts)
        self.backup_weights()
        self.update_weight_scale(args.weight_cali_table)
        if self.mini_batch <= self.batch_size:
            self.mini_batch = 1
        else:
            self.mini_batch = self.mini_batch // self.batch_size
        w = np.load(self.weight_file, allow_pickle=True)
        for k in w:
            self.param_back[k] = w[k]

    def init_opt(self, scheduler, momentum, nesterov, weight_decay):
        self.opt = SgdWeightOpt(scheduler, momentum, nesterov, weight_decay)

    def sigmoid(self, x):
        return expit(-x)

    def exp_neg(self, x):
        return (1.0/(expit(x)+1e-8))-1.0

    def dwdalpha(self, a):
        l = np.where(a <= -np.log(11), 0, 1)
        h = np.where(a >= -np.log(1.2/1.1-1), 0, 1)
        dwda = 1.2*(self.sigmoid(a)*(1-self.sigmoid(a)))
        dwda = dwda*l
        dwda = dwda*h+(1-h)
        return dwda

    def rec_sig(self, x):
        return np.clip(self.sigmoid(x)*1.2-0.1, 0, 1)

    def cal_beta(self, iter):
        if iter < self.num_sample*self.epoch*self.beta_warmup:
            return self.beta_start
        else:
            return int(self.beta_end + 0.5 * (self.beta_start - self.beta_end) * (1.0 + np.cos((iter-self.num_sample*self.epoch*self.beta_warmup)/(self.num_sample*self.epoch*(1.0-self.beta_warmup)) * np.pi)))

    def cal_round_loss(self, iter, alpha, beta, reg=0.01):
        if iter < self.num_sample * self.epoch * self.beta_warmup:
            return 0.0
        else:
            rect_alpha = np.clip((self.zeta - self.gamma)*self.sigmoid(alpha) + self.gamma, 0, 1)
            return reg * (1 - np.power(2 * np.abs(rect_alpha-0.5), beta)).sum()

    def cal_grd(self, out, ref):
        return out - ref

    def cal_grdr(self, alpha, beta, iter, dwda, reg=0.01):
        if iter < self.num_sample * self.epoch * self.beta_warmup:
            return np.zeros_like(alpha)
        else:
            dw = np.power(2*np.abs(self.rec_sig(alpha)-0.5), beta-1)
            s = np.where(alpha > 0, 1, -1)
            dw = reg*beta*dw*2*dwda*s
            return dw

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
            if top_ops[op].type in LEARNING_WEIGHT_OPERATION and top_ops[op].name not in exclude and self.included(top_ops[op].name):
                if len(top_ops[op].opds) > 1 and top_ops[op].opds[1] in self.module.all_weight_names:
                    self.finetune_layers.append(op)
                    self.finetune_layer_weights[op] = top_ops[op].opds[1]
                if len(top_ops[op].opds) > 2 and top_ops[op].opds[2] in self.module.all_weight_names:
                    self.finetune_layer_bias[op] = top_ops[op].opds[2]

    def backup_weights(self):
        for op in self.finetune_layers:
            self.orig_weights[op] = copy.deepcopy(self.module.get_tensor(self.finetune_layer_weights[op]))
            oc = self.orig_weights[op].shape[0]
            if op in self.finetune_layer_bias:
                self.orig_bias[op] = copy.deepcopy(self.module.get_tensor(self.finetune_layer_bias[op]))
            if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
                self.weights_scales[op] = np.max(np.abs(self.orig_weights[op].reshape(oc, -1)), axis=1)
                self.weights_scales[op] = np.where(self.weights_scales[op] > 1e-8, self.weights_scales[op], 1e-8)
            elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
                self.weights_scales[op] = np.max(np.abs(self.orig_weights[op]))
                self.weights_scales[op] = np.where(self.weights_scales[op] > 1e-8, self.weights_scales[op], 1e-8)
            else:
                print("not support!")
                sys.exit(1)

    def update_weight_scale(self, weight_cali_table):
        if weight_cali_table != "":
            f = open(weight_cali_table, 'r')
            for line in f:
                if len(line) > 0 and (not line.startswith("#")):
                    s = line.split(" ")
                    s = [x for x in s if x != '']
                    if len(s) != 2:
                        continue
                    for op in self.finetune_layers:
                        if s[0] == self.finetune_layer_weights[op]:
                            self.weights_scales[op] = float(s[1])

    def restore_weight(self, op):
        self.module.set_tensor(self.finetune_layer_weights[op], self.orig_weights[op])

    def quant_requant_weight(self, op, bitwidth=8, hard=False):
        if bitwidth == 8:
            qmax = 127.0
            qmin = -128.0
        elif bitwidth == 4:
            qmax = 7.0
            qmin = -7.0
        weight_tmp = self.orig_weights[op].copy()
        scales = self.weights_scales[op]/qmax
        shape = weight_tmp.shape
        if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
            weight_tmp = (weight_tmp.reshape(shape[0], -1)/scales[:, None]).reshape(shape)
        elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
            weight_tmp = weight_tmp/scales
        else:
            print("not support!")
            sys.exit(1)
        if op not in self.alpha:  # init
            alpha = weight_tmp - np.floor(weight_tmp)
            # this is where alpha is coming, refer to ppx and aimet
            alpha = -np.log((self.zeta-self.gamma)/(alpha - self.gamma)-1)
            self.alpha[op] = alpha
        else:
            alpha = self.alpha[op]
        if not hard:
            if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
                weight = (np.clip(np.floor(weight_tmp)+self.rec_sig(alpha), qmin,
                          qmax).reshape(shape[0], -1)*scales[:, None]).reshape(shape)
            elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
                weight = np.clip(np.floor(weight_tmp)+self.rec_sig(alpha), qmin, qmax)*scales
        else:
            if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
                weight = (np.clip(np.floor(weight_tmp)+(alpha >= 0).astype(np.float32),
                          qmin, qmax).reshape(shape[0], -1)*scales[:, None]).reshape(shape)
            elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
                weight = np.clip(np.floor(weight_tmp)+(alpha >= 0).astype(np.float32), qmin, qmax)*scales
        self.module.set_tensor(self.finetune_layer_weights[op], weight)

    def quant_requant_weight_orig(self, op, bitwidth=8):
        weight_tmp = copy.deepcopy(self.orig_weights[op])
        if bitwidth == 8:
            qmax = 127.0
            qmin = -128.0
        elif bitwidth == 4:
            qmax = 7.0
            qmin = -7.0
        else:
            print('wrong bit width\n')
            sys.exit(1)
        scales = self.weights_scales[op]/qmax
        shape = weight_tmp.shape
        if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
            weight_tmp = (weight_tmp.reshape(shape[0], -1)/scales[:, None]).reshape(shape)
        if self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
            weight_tmp = weight_tmp/scales
        weight = np.clip(np.round(weight_tmp), qmin, qmax)
        if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
            self.module.set_tensor(self.finetune_layer_weights[op], (weight.reshape(
                shape[0], -1)*scales[:, None]).reshape(shape))
        elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
            self.module.set_tensor(self.finetune_layer_weights[op], weight*scales)
        else:
            print("not support!")
            sys.exit(1)

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
            scale = self.scales[pop][0]
            unsigned = self.scales[op][1] >= 0 and self.scales[op][2] >= 0
            if scale != 1.0 and quant:
                if self.support_unsigned:
                    d = quant_requant_active(d, scale, unsigned, bitwidth)
                else:
                    d = quant_requant_active(d, scale, False, bitwidth)
            self.module.set_tensor(pop, d)

    def op_first_input(self, op, loop, bitwidth=8):
        pre_ops = self.parser.get_pre_op_by_op_name(op)
        if len(pre_ops) != 1:
            print(f'input num not 1! {op}')
            sys.exit(1)
        pop = pre_ops[0]
        shape = self.ref_tensors.get(pop, loop).shape
        if pop not in self.module.all_tensor_names:
            if self.parser.get_op_type_by_op_name(pop) == 'top.Reshape':
                pre_pre_ops = self.parser.get_pre_op_by_op_name(pop)
                pop = pre_pre_ops[0]
            else:
                print(f"{op} input {pop} not in all tensor list")
                sys.exit(1)
        d = self.ref_tensors.get(pop, loop).reshape(shape)
        scale = self.scales[pop][0]
        unsigned = self.scales[op][1] >= 0 and self.scales[op][2] >= 0
        if scale != 1.0:
            if self.support_unsigned:
                d = quant_requant_active(d, scale, unsigned, bitwidth)
            else:
                d = quant_requant_active(d, scale, False, bitwidth)
        return d

    def bias_correction(self, op, bitwidth=8):
        tmpweight = self.orig_weights[op].copy()
        weightname = self.finetune_layer_weights[op]
        self.quant_requant_weight(op, bitwidth, True)
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

    def learning_one(self, epoch, op, total):
        self.loger.logging(f"now to learn {op} in epoch {epoch}")
        sub_total = 1
        if epoch == 0:
            sub_total += 1
        if epoch == self.epoch - 1:
            sub_total += 1
        pbar_detail = tqdm(np.arange(self.num_sample*sub_total))
        pbar_detail.set_description("Learning Weight, op %s" % op)
        if self.chip == 'bm1688':
            bitwidth = 4
            qmax = 7.0
        else:
            bitwidth = 8
            qmax = 127.0

        if epoch == 0:
            self.quant_requant_weight_orig(op, bitwidth)
            for loop in np.arange(self.num_sample):
                pbar_detail.set_postfix_str(
                    f"Cal orig loss {epoch}.{loop+1}/{self.epoch}.{self.num_sample} [Total: {total}]")
                pbar_detail.update()
                self.set_op_inputs(op, loop, bitwidth, quant=True)
                outputs = self.module.invoke_at(op)
                scale = self.scales[op][0]
                unsigned = self.scales[op][1] >= 0 and self.scales[op][2] >= 0
                if self.compare_quanted:
                    if self.support_unsigned:
                        outputs = quant_requant_active(outputs, scale, unsigned, bitwidth)
                    else:
                        outputs = quant_requant_active(outputs, scale, False, bitwidth)
                ref = self.ref_tensors.get(op, loop)
                if op in self.pre_loss:
                    pre_loss = self.pre_loss[op] + cal_loss(outputs, ref)
                    self.pre_loss[op] = pre_loss
                else:
                    pre_loss = cal_loss(outputs, ref)
                    self.pre_loss[op] = pre_loss

        for loop in np.arange(self.num_sample):
            pbar_detail.set_postfix_str(
                f"Learning {epoch}.{loop+1}/{self.epoch}.{self.num_sample} [Total: {total}]")
            pbar_detail.update()
            self.set_op_inputs(op, loop, bitwidth, quant=True)
            self.quant_requant_weight(op, bitwidth)
            outputs = self.module.invoke_at(op)
            scale = self.scales[op][0]
            unsigned = self.scales[op][1] >= 0 and self.scales[op][2] >= 0
            if self.support_unsigned:
                outputq = quant_requant_active(outputs, scale, unsigned, bitwidth)
            else:
                outputq = quant_requant_active(outputs, scale, False, bitwidth)
            ref = self.ref_tensors.get(op, loop)
            beta = self.cal_beta(epoch*self.num_sample+loop)
            loss = cal_loss(outputq, ref)
            self.loger.logging(f"loss of {op} in loop {loop} is {loss}")
            loss += self.cal_round_loss(epoch*self.num_sample+loop, self.alpha[op], beta)
            self.loger.logging(f"loss of {op} in loop {loop} plus is {loss}")
            self.opt.update_loss(op, loss)
            unsigned = self.scales[op][1] >= 0 and self.scales[op][2] >= 0
            # grd_dst = self.cal_grd(outputs, scale, bitwidth, unsigned)
            grd_dst = self.cal_grd(outputs, ref)
            if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
                grd_w = self.module.backward_weight_at(op, self.finetune_layer_weights[op], grd_dst)
            elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
                # should consider transpose? matmul with weight has one input
                input = self.op_first_input(op, loop, bitwidth)
                shape = input.shape
                batch = outputs.shape[0]
                if len(input.shape) == 3 and input.shape[0] != batch:
                    print('shape mismatch')
                    sys.exit(1)
                if len(shape) == 2:
                    input = input.reshape(-1, shape[-1]).transpose()
                elif len(shape) == 3:
                    input = input.transpose(0, 2, 1)
                if len(grd_dst.shape) == 1:
                    grd_dst = grd_dst.reshape(1, grd_dst.shape[0])
                grd_w = np.matmul(input, grd_dst)
                if len(shape) == 3:
                    grd_w = np.average(grd_w, axis=0)
            shape = self.alpha[op].shape
            grd_w1 = self.dwdalpha(self.alpha[op])
            if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
                grd_w1 = (grd_w1.reshape(shape[0], -1)*((self.weights_scales[op]/qmax)[:, None])).reshape(shape)
            elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
                grd_w1 = grd_w1*(self.weights_scales[op]/qmax)
            else:
                print("not support!")
                sys.exit(1)
            grd_w = grd_w * grd_w1
            grd_r = self.cal_grdr(self.alpha[op], beta, epoch*self.num_sample+loop, grd_w1)
            grd = grd_w + grd_r
            self.opt.update_grd(op, grd)
            if (epoch*self.num_sample+loop+1) % self.mini_batch == 0:
                self.alpha[op] = self.opt.update_alpha(epoch*self.num_sample+loop, op, self.alpha[op], self.mini_batch)
                log_alpha(op, self.alpha[op].copy(), loop)

        if epoch == self.epoch-1:
            self.quant_requant_weight(op, bitwidth, True)
            for loop in np.arange(self.num_sample):
                pbar_detail.set_postfix_str(
                    f"Comparing {epoch}.{loop+1}/{self.epoch}.{self.num_sample} [Total: {total}]")
                pbar_detail.update()
                self.set_op_inputs(op, loop, bitwidth, quant=True)
                self.module.invoke_at(op)
                outputs = self.module.get_tensor(op)
                scale = self.scales[op][0]
                unsigned = self.scales[op][1] >= 0 and self.scales[op][2] >= 0
                if self.compare_quanted:
                    if self.support_unsigned:
                        outputs = quant_requant_active(outputs, scale, unsigned, bitwidth)
                    else:
                        outputs = quant_requant_active(outputs, scale, False, bitwidth)
                ref = self.ref_tensors.get(op, loop)
                if op in self.post_loss:
                    post_loss = self.post_loss[op] + cal_loss(outputs, ref)
                    self.post_loss[op] = post_loss
                else:
                    post_loss = cal_loss(outputs, ref)
                    self.post_loss[op] = post_loss
            self.restore_weight(op)

            # if self.post_loss[op] <= self.pre_loss[op]:
            if True:
                self.loger.logging(f'{op} use trained weight {self.post_loss[op]} vs {self.pre_loss[op]}')
                print(f'{op} use trained weight {self.post_loss[op]} vs {self.pre_loss[op]}')
                self.update_weight(op, bitwidth)
                '''
                if op in self.finetune_layer_bias: # when with weight and with bias, can do bias correction
                    pbar_detail.set_postfix_str(
                        f"Correct {op}")
                    pbar_detail.update()
                    self.bias_correction(op, bitwidth)
                '''

            else:
                self.loger.logging(f'{op} do not use learned weight {self.post_loss[op]} vs {self.pre_loss[op]}')
                print(f'{op} do not use learned weight {self.post_loss[op]} vs {self.pre_loss[op]}')

    def adjust_weight(self, op, bitwidth=8):
        if bitwidth == 8:
            qmax = 127.0
            qmin = -128.0
        elif bitwidth == 4:
            qmax = 7.0
            qmin = -7.0
        shape = self.orig_weights[op].shape
        w_reshape = self.orig_weights[op].reshape(shape[0], -1)
        p = self.alpha[op].reshape(shape[0], -1)
        '''
        adj = np.argmax(self.orig_weights[op].reshape(shape[0],-1),axis=1)
        for i in np.arange(shape[0]):
            if p[i][adj[i]] >= 0 and w_reshape[i][adj[i]] <= 0:
                p[i][adj[i]] = -p[i][adj[i]]
            elif p[i][adj[i]] < 0 and w_reshape[i][adj[i]] > 0:
                p[i][adj[i]] = -p[i][adj[i]]
        '''
        if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
            weight = (np.clip((np.floor(w_reshape/((self.weights_scales[op]/qmax)[:, None])) + np.where(
                p >= 0, 1.0, 0.0).astype(np.float32)), qmin, qmax) * ((self.weights_scales[op]/qmax)[:, None])).reshape(shape)
        elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
            weight = (np.clip((np.floor(w_reshape/(self.weights_scales[op]/qmax)) + np.where(
                p >= 0, 1.0, 0.0).astype(np.float32)), qmin, qmax) * (self.weights_scales[op]/qmax)).reshape(shape)
        else:
            print("not support!")
            sys.exit(1)
        return weight

    def update_weight(self, op, bitwidth=8):
        self.param_back[self.finetune_layer_weights[op]] = self.adjust_weight(op, bitwidth)

    def update_bias(self, op, bias):
        self.param_back[self.finetune_layer_bias[op]] = bias

    def save_weights(self):
        os.rename(self.weight_file, self.weight_file.replace(".npz", ".bak.npz"))
        np.savez(self.weight_file, **self.param_back)

    def learning(self):
        total = len(self.finetune_layers)
        '''
        can_parallel = True
        for l in self.finetune_layers:
            if self.parser.get_op_type_by_op_name(l) != 'top.MatMul':
                can_parallel = False
                break
        if can_parallel:
            groups = into_groups(self.parser, self.finetune_layers)
            for epoch in np.arange(self.epoch):
                for layers in groups:
                    reqs = [(self, epoch, x, total) for x in groups[layers]]
                    learned = 0
                    for result in pool.map(learning_adaweight_wrap, reqs):
                        learned += 1
                print("")
                print("=================================================")
                print(f"  End epoch {epoch}, learned {learned} layers")
                print("=================================================")
                print("")
        else:
            for epoch in np.arange(self.epoch):
                for l in self.finetune_layers:
                    self.learning_one(epoch, l, total)
                print("")
                print("=================================================")
                print(f"  End epoch {epoch}, learned {total} layers")
                print("=================================================")
                print("")
        '''
        for epoch in np.arange(self.epoch):
            for l in self.finetune_layers:
                self.learning_one(epoch, l, total)
            print("")
            print("=================================================")
            print(f"  End epoch {epoch}, learned {total} layers")
            print("=================================================")
            print("")

        self.save_weights()
