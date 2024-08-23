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
import sys
import os
from tqdm import tqdm

from .utils import quant_requant_active
from .utils import cal_loss

import pymlir
pymlir.set_mem_mode("force_value_mem")

SKIP_OPERATION = [
    'top.Input', 'top.Reshape', 'top.Softmax', 'top.Weight', 'top.MaxPool', 'top.Slice', 'top.Tile',
    'top.Permute', 'top.Upsample'
]


class SgdScaleOpt:
    def __init__(self, lr, momentum=0.0, nesterov=False, weight_decay=0.0, loger=None, support_unsigned=False):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.dampening = 0.0
        self.v = {}
        self.grd = {}
        self.loss = {}
        self.loger = loger
        self.support_unsigned = support_unsigned
        print(
            f'Learning Scale SGD, momentum is {self.momentum} nesterov is {self.nesterov} weight_decay is {self.weight_decay}')

    def cal_scale(self, iter, scale, loss, grd, unsigned=False):
        if unsigned:
            step = np.abs(scale)/255.0
        else:
            step = np.abs(scale)/127.0

        step = step - self.lr.cal_lr(iter) * grd
        self.loger.logging(
            f"update scale step {step} grd {grd} loss {loss}")

        if unsigned:
            scale = step*255.0
        else:
            scale = step*127.0
        return scale

    def update_scale(self, iter, op, scale, mini_batch, unsigned=False):
        self.grd[op] = self.grd[op]/mini_batch
        self.loss[op] = self.loss[op]/mini_batch
        if self.weight_decay != 0.0:
            if unsigned:
                self.grd[op] = self.grd[op] + np.abs(scale)/255.0*self.weight_decay
            else:
                self.grd[op] = self.grd[op] + np.abs(scale)/127.0*self.weight_decay
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

        if self.support_unsigned:
            scale = self.cal_scale(iter, scale, self.loss[op], self.grd[op], unsigned)
        else:
            scale = self.cal_scale(iter, scale, self.loss[op], self.grd[op], False)
        self.reset_grd_loss(op)
        return scale

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
        self.grd[op] = 0.0
        self.loss[op] = 0.0


class AdamScaleOpt:
    def __init__(self, lr, beta1=0.9, beta2=0.999, weight_decay=0.0, loger=None, support_unsigned=False):
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.amsgrad = False
        self.steps = {}
        self.eps = 1e-8
        self.exp_avgs = {}
        self.exp_avgs_sqs = {}
        self.max_exp_avgs_sqs = {}
        self.grd = {}
        self.loss = {}
        self.ams_grads = {}
        self.loger = loger
        print(
            f'Learning Scale Adam, weight_decay is {self.weight_decay}')

    def cal_scale(self, scale, delta, unsigned=False):
        if unsigned:
            step = np.abs(scale)/255.0
        else:
            step = np.abs(scale)/127.0

        step = step - delta
        self.loger.logging(
            f"update scale step {step} step {step}")

        if unsigned:
            scale = step*255.0
        else:
            scale = step*127.0
        return scale

    def update_scale(self, iter, op, scale, mini_batch, unsigned=False):
        self.grd[op] = self.grd[op]/mini_batch
        self.loss[op] = self.loss[op]/mini_batch
        if op in self.steps:
            self.steps[op] = self.steps[op] + 1
        else:
            self.steps[op] = 1
        if self.weight_decay != 0.0:
            if unsigned:
                self.grd[op] = scale/255.0*self.weight_decay + self.grd[op]
            else:
                self.grd[op] = scale/127.0*self.weight_decay + self.grd[op]
        bias_correction1 = 1 - self.beta1 ** self.steps[op]
        bias_correction2 = 1 - self.beta2 ** self.steps[op]

        if op in self.exp_avgs:
            self.exp_avgs[op] = self.exp_avgs[op]*self.beta1+self.grd[op]*(1-self.beta1)
            self.exp_avgs_sqs[op] = self.exp_avgs_sqs[op]*self.beta2+(self.grd[op]**2)*(1-self.beta2)
        else:
            self.exp_avgs[op] = 0
            self.exp_avgs_sqs[op] = 0
            self.max_exp_avgs_sqs[op] = 0
        if self.amsgrad:
            self.max_exp_avgs_sqs[op] = np.maximum(self.max_exp_avgs_sqs[op], self.exp_avgs_sqs[op])
            denorm = np.sqrt(self.max_exp_avgs_sqs[op])/np.sqrt(bias_correction2)+self.eps
        else:
            denorm = np.sqrt(self.exp_avgs_sqs[op])/np.sqrt(bias_correction2)+self.eps
        step_size = self.lr.cal_lr(iter) / bias_correction1
        delta = step_size * self.exp_avgs[op]/denorm
        if unsigned:
            scale = (np.abs(scale)/255.0 - delta)*255.0
            self.loger.logging(
                f"update scale {scale/255.0} grd {self.grd[op]} delta {delta}")
        else:
            scale = (np.abs(scale)/127.0 - delta)*127.0
            self.loger.logging(
                f"update scale {scale/127.0} grd {self.grd[op]} delta {delta}")
        self.reset_grd_loss(op)
        return scale

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
        self.grd[op] = 0.0
        self.loss[op] = 0.0


class LearningScale:
    def __init__(self, args):
        self.args = args
        self.mlir_file = args.mlir_file
        self.chip = args.chip
        self.module = pymlir.module()
        self.module.load(self.mlir_file)
        self.parser = MlirParser(args.mlir_file)
        self.batch_size = self.parser.get_batch_size()  # batch size of net
        self.input_num = self.parser.get_input_num()  # number of net inputs
        self.mini_batch = args.mini_batch
        self.epoch = args.epoch
        self.ref_tensors = None
        self.num_sample = 0
        self.orig_scales8 = {}
        self.orig_scales4 = {}
        self.new_scales = {}
        self.pre_loss = {}
        self.post_loss = {}
        self.loger = None
        self.support_unsigned = False
        self.opt = None
        self.finetune_layers = []  # layers to fine tune, without skipped
        self.get_finetune_ops(args.excepts)
        if self.mini_batch <= self.batch_size:
            self.mini_batch = 1
        else:
            self.mini_batch = self.mini_batch // self.batch_size

    def init_sgd(self, scheduler, momentum, nesterov, weight_decay):
        self.opt = SgdScaleOpt(scheduler, momentum, nesterov, weight_decay, self.loger)

    def init_adam(self, scheduler, momentum, nesterov, weight_decay):
        self.opt = AdamScaleOpt(scheduler, momentum, nesterov, weight_decay, self.loger)

    def get_finetune_ops(self, excepts):
        top_ops = {op.name: op for op in self.parser.ops}
        exclude = excepts.split(',')
        for n in top_ops:
            if top_ops[n].type not in SKIP_OPERATION and top_ops[n].name not in exclude:
                self.finetune_layers.append(n)

    def set_op_inputs(self, op, loop, quant=False):
        pre_ops = self.parser.get_pre_op_by_op_name(op)
        for pop in pre_ops:
            shape = self.ref_tensors.get(pop, loop).shape
            if pop not in self.module.all_tensor_names:
                if self.parser.get_op_type_by_op_name(pop) == 'top.Reshape' or self.parser.get_op_type_by_op_name(pop) == 'top.Squeeze' or self.parser.get_op_type_by_op_name(pop) == 'top.Unsqueeze':
                    pre_pre_ops = self.parser.get_pre_op_by_op_name(pop)
                    pop = pre_pre_ops[0]
                else:
                    print(f"{op} input {pop} not in all tensor list")
                    sys.exit(1)
            d = self.ref_tensors.get(pop, loop).copy().reshape(shape)
            scale = 1.0
            if pop in self.new_scales:
                scale = self.new_scales[pop][0]
            elif pop in self.orig_scales8:
                scale = self.orig_scales8[pop][0]
            unsigned = self.orig_scales8[op][1] >= 0 and self.orig_scales8[op][2] >= 0
            if scale != 1.0 and not quant:
                if self.support_unsigned:
                    d = quant_requant_active(d, scale, unsigned)
                else:
                    d = quant_requant_active(d, scale, False)
            self.module.set_tensor(pop, d)

    def cal_grdscale_unsigned(self, out):
        return 1.0 / (out.size * 255.0) ** 0.5

    def cal_grdscale_signed(self, out):
        return 1.0 / (out.size * 127.0) ** 0.5

    def cal_grd_unsigned(self, out, scale):
        step = np.abs(scale)/255.0
        grd = 0
        m = np.round(out/step)
        qmin = np.zeros_like(m)
        qmax = np.ones_like(m)*(255)
        g = (np.minimum(np.maximum(m, qmin), qmax)*step - out)*2
        grd = grd + (np.where(m <= 0, 0, 0)*g).sum()
        grd = grd + (np.where(m >= 255, 255, 0)*g).sum()
        left = np.where(m > 0, 1, 0) & np.where(m < 255, 1, 0)
        m = m-out/step
        m = m*left * g
        grd = grd + m.sum()
        return grd

    def cal_grd_signed(self, out, scale):
        step = np.abs(scale)/127.0
        grd = 0
        m = np.round(out/step)
        qmin = np.ones_like(m)*(-128)
        qmax = np.ones_like(m)*(127)
        g = (np.minimum(np.maximum(m, qmin), qmax)*step - out)*2
        grd = grd + (np.where(m <= -128, -128, 0)*g).sum()
        grd = grd + (np.where(m >= 127, 127, 0)*g).sum()
        left = np.where(m > -128, 1, 0) & np.where(m < 127, 1, 0)
        m = m-out/step
        m = m*left * g
        grd = grd + m.sum()
        return grd

    def cal_grd(self, out, scale, use_grdscale=True, unsigned=False):
        grd_funcs = {'unsigned': self.cal_grd_unsigned,
                     'signed': self.cal_grd_signed}
        grdscale_funcs = {'unsigned': self.cal_grdscale_unsigned,
                          'signed': self.cal_grdscale_signed}
        if unsigned:
            grdfunc = grd_funcs['unsigned']
            grdscalefunc = grdscale_funcs['unsigned']
        else:
            grdfunc = grd_funcs['signed']
            grdscalefunc = grdscale_funcs['signed']

        grd = grdfunc(out, scale)
        grd_scale = grdscalefunc(out)

        if use_grdscale:
            return grd*grd_scale
        else:
            return grd

    def learning_one(self, op, total):
        self.loger.logging(f"now to learn {op} scale")
        pbar_detail = tqdm(np.arange(self.num_sample*3))
        pbar_detail.set_description("Learning Scale, op %s" % op)
        for loop in np.arange(self.num_sample):
            pbar_detail.set_postfix_str(
                f"Cal orig loss {loop} [Total Progress: {total}]")
            pbar_detail.update()
            self.set_op_inputs(op, loop, quant=True)
            outputs = self.module.invoke_at(op).copy()
            scale = self.orig_scales8[op][0]
            unsigned = self.orig_scales8[op][1] >= 0 and self.orig_scales8[op][2] >= 0
            if self.support_unsigned:
                outputs[0] = quant_requant_active(outputs[0], scale, unsigned)
            else:
                outputs[0] = quant_requant_active(outputs[0], scale, False)
            ref = self.ref_tensors.get(op, loop)
            if op in self.pre_loss:
                pre_loss = self.pre_loss[op] + cal_loss(outputs, ref)
                self.pre_loss[op] = pre_loss
            else:
                pre_loss = cal_loss(outputs, ref)
                self.pre_loss[op] = pre_loss

        for loop in np.arange(self.num_sample):
            pbar_detail.set_postfix_str(
                f"Learning {loop} [Total Progress: {total}]")
            pbar_detail.update()
            self.set_op_inputs(op, loop, quant=True)
            outputs = self.module.invoke_at(op).copy()
            scale = 1.0
            if op in self.new_scales:
                scale = self.new_scales[op][0]
            elif op in self.orig_scales8:
                scale = self.orig_scales8[op][0]
            unsigned = self.orig_scales8[op][1] >= 0 and self.orig_scales8[op][2] >= 0
            if scale != 1.0:
                if self.support_unsigned:
                    outputq = quant_requant_active(outputs[0], scale, unsigned)
                else:
                    outputq = quant_requant_active(outputs[0], scale, False)
            else:
                outputq = outputs[0]
            ref = self.ref_tensors.get(op, loop)
            loss = cal_loss(outputq, ref)
            self.opt.update_loss(op, loss)
            unsigned = self.orig_scales8[op][1] >= 0 and self.orig_scales8[op][2] >= 0
            grd = self.cal_grd(outputs[0], scale, True, unsigned)
            self.opt.update_grd(op, grd)
            if (loop+1) % self.mini_batch == 0:
                scale = self.opt.update_scale(loop, op, scale, self.mini_batch, unsigned)
                if op in self.new_scales:
                    self.new_scales[op][0] = scale
                else:
                    self.new_scales[op] = [scale, 0, 0]
                self.loger.logging("{} new scale is {:.16f} iter {} batch {}".format(
                    op, scale, loop+1, self.mini_batch))

        for loop in np.arange(self.num_sample):
            pbar_detail.set_postfix_str(
                f"Comparing {loop} [Total Progress: {total}]")
            pbar_detail.update()
            self.set_op_inputs(op, loop, quant=True)
            self.module.invoke_at(op)
            outputs = self.module.get_tensor(op).copy()
            scale = self.new_scales[op][0]
            unsigned = self.orig_scales8[op][1] >= 0 and self.orig_scales8[op][2] >= 0
            if self.support_unsigned:
                outputs[0] = quant_requant_active(outputs[0], scale, unsigned)
            else:
                outputs[0] = quant_requant_active(outputs[0], scale, False)
            ref = self.ref_tensors.get(op, loop)
            if op in self.post_loss:
                post_loss = self.post_loss[op] + cal_loss(outputs, ref)
                self.post_loss[op] = post_loss
            else:
                post_loss = cal_loss(outputs, ref)
                self.post_loss[op] = post_loss

        for in_ in self.parser.get_pre_op_by_op_name(op):
            self.ref_tensors.consumed_tensor(in_)

        if self.post_loss[op] >= self.pre_loss[op] or self.new_scales[op][0] < 0 or self.new_scales[op][0]/self.orig_scales8[op][0] > 1.5:
            self.loger.logging(
                f'abandon backward tune of {op}, old loss: {self.pre_loss[op]}, new loss: {self.post_loss[op]}, old scale {self.orig_scales8[op][0]} new scale {self.new_scales[op][0]}')
            del self.new_scales[op]
        else:
            self.loger.logging(
                f'use tune of {op}, old loss: {self.pre_loss[op]}, new loss: {self.post_loss[op]}, old scale {self.orig_scales8[op][0]} new scale {self.new_scales[op][0]}')

    def learning(self):
        total = len(self.finetune_layers)
        '''
        groups = into_groups(self.parser, self.finetune_layers)
        can_parallel = True
        for l in self.finetune_layers:
            if self.parser.get_op_type_by_op_name(l) != 'top.MatMul':
                can_parallel = False
                break
        if can_parallel:
            for layers in groups:
                reqs = [(self, x, total) for x in groups[layers]]
                learned = 0
                for result in pool.map(learning_scale_wrap, reqs):
                    learned += 1
        else:
        '''
        for e in range(self.epoch):
            for l in self.finetune_layers:
                self.learning_one(l, total)
