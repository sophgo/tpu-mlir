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

import pymlir
pymlir.set_mem_mode("force_value_mem")


LEARNING_WEIGHT_OPERATION = [
    # 'top.Conv', 'top.MatMul'#
    'top.MatMul'
]


class GptqQuantizer():
    def __init__(self, shape, bits, perchannel=False, sym=True,
                 mse=False, norm=2.4, grid=100, maxshrink=.8):
        self.maxq = []
        self.scale = np.zeros(shape[0])
        self.zero = np.zeros(shape[0])
        self.maxq = 2 ** bits - 1
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink

    def find_params(self, x, weight=False):
        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.reshape(shape[0], -1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.reshape(shape[0], -1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0])
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = np.ones_like(self.scale)*((self.maxq + 1) / 2)
        else:
            self.zero = np.round(-xmin / self.scale)

        if self.mse:
            best = np.full([x.shape[0]], float('inf'))
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = np.round(-xmin1 / scale1) if not self.sym else self.zero
                q = self.quantize(x)
                q = np.power(np.abs(q-x, self.norm))
                err = np.sum(q, axis=1)
                tmp = err < best
                if np.any(tmp):
                    best = np.where(tmp, err, best)
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):  # fixme , check perchannel
        q = np.clip(np.round(x / self.scale) + self.zero, 0, self.maxq)
        return self.scale * (q - self.zero)


class LearningGptqWeight:
    def __init__(self, args):
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
        self.samples = {}
        self.epoch = args.epoch
        self.ref_tensors = None
        self.pre_loss = {}
        self.post_loss = {}
        self.loss = {}
        self.chip = args.chip
        self.orig_weights = {}
        self.weight_file = self.parser.module_weight_file
        self.param_back = {}
        self.weights_scales = {}
        self.compare_quanted = True
        self.loger = None
        print(f'Learning GptqWeight')
        self.support_unsigned = False
        self.get_finetune_ops(args.excepts)
        self.backup_weights()
        self.H = {}
        if self.mini_batch <= self.batch_size:
            self.mini_batch = 1
        else:
            self.mini_batch = self.mini_batch // self.batch_size
        w = np.load(self.weight_file, allow_pickle=True)
        for k in w:
            self.param_back[k] = w[k]

    def get_finetune_ops(self, excepts):
        top_ops = {op.name: op for op in self.parser.ops}
        exclude = excepts.split(',')
        for op in top_ops:
            if top_ops[op].type in LEARNING_WEIGHT_OPERATION and top_ops[op].name not in exclude:
                if top_ops[op].type == 'top.Conv':
                    if int(top_ops[op].attrs['group'].split(':')[0]) > 1:
                        continue
                if len(top_ops[op].opds) > 1 and top_ops[op].opds[1] in self.module.all_weight_names:
                    self.finetune_layers.append(op)
                    self.finetune_layer_weights[op] = top_ops[op].opds[1]

    def filter_fixed_floats(self, except_layers):
        for l in self.finetune_layers:
            if l in except_layers:
                self.finetune_layers.remove(l)

    def backup_weights(self):
        for op in self.finetune_layers:
            self.orig_weights[op] = copy.deepcopy(self.module.get_tensor(self.finetune_layer_weights[op]))
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

    def restore_weight(self, op):
        self.module.set_tensor(self.finetune_layer_weights[op], self.orig_weights[op])

    def quant_requant_weight(self, op, blocksize=128, percdamp=0.01, groupsize=-1, bitwidth=8, actorder=False):
        W = torch.Tensor(self.orig_weights[op].copy())

        tick = time.time()

        if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
            shape = W.shape
            W = W.reshape(shape[0], -1)
            shape = W.shape
            rows = W.reshape(shape[0], -1).shape[0]
            columns = W.reshape(shape[0], -1).shape[1]
            quanter = GptqQuantizer(shape, bits=bitwidth, perchannel=True, sym=True, mse=False)
        elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
            shape_org = W.shape
            W = W.t()
            shape = W.shape
            rows = shape[0]
            columns = shape[1]
            quanter = GptqQuantizer(shape, bits=bitwidth, perchannel=False, sym=True, mse=False)
        quanter.find_params(W, weight=True)

        H = torch.Tensor(self.H[op])
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, columns, blocksize):
            i2 = min(i1 + blocksize, columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        quanter.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                q = quanter.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]

        if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
            self.update_weight(op, Q.numpy().reshape(shape))
            self.module.set_tensor(self.finetune_layer_weights[op], Q.numpy().reshape(shape))
        elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
            self.update_weight(op, Q.numpy().transpose().reshape(shape))
            self.module.set_tensor(self.finetune_layer_weights[op], Q.numpy().transpose().reshape(shape_org))

    def quant_requant_weight_orig(self, op, bits=8):
        weight_tmp = copy.deepcopy(self.orig_weights[op])
        scales = self.weights_scales[op]/(2**(bits-1)-1)
        shape = weight_tmp.shape
        if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
            weight_tmp = (weight_tmp.reshape(shape[0], -1)/scales[:, None]).reshape(shape)
        if self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
            weight_tmp = weight_tmp/scales
        weight = np.clip(np.round(weight_tmp), -(2**(bits-1)), (2**(bits-1)-1))
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

    def shape_str_to_list(self, shape):
        s = shape.replace('[', '').replace(']', '').replace(' ', '').split(',')
        s = [int(x) for x in s]
        return s

    def update_H(self, op, input, output):
        shape = input.shape
        if len(shape) == 2:
            in_num = 1
        elif len(shape) == 3:
            in_num = shape[0]
        else:
            print(f"input shape not support {shape}")
            sys.exit(1)
        if op not in self.H:
            weight_shape = self.orig_weights[op].shape
            print(f'input shape {shape} weight shape {weight_shape}')
            if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
                weight_shape = self.orig_weights[op].reshape(weight_shape[0], -1).shape
            elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
                weight_shape = self.orig_weights[op].transpose().shape
            else:
                print('not support!')
                sys.exit(1)
            self.H[op] = np.zeros((weight_shape[1], weight_shape[1]))
            self.samples[op] = 0
        else:
            # do the add batch update to H
            if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
                op_ = self.parser.get_op_by_op_name(op)
                if op_ == None:
                    print(f'error find op {op}')
                    sys.exit(1)
                k_shape = self.shape_str_to_list(op_.attrs['kernel_shape'])
                dia = self.shape_str_to_list(op_.attrs['dilations'])
                pads = self.shape_str_to_list(op_.attrs['pads'])
                # group = int(op_.attrs['group'].split(':')[0])
                if len(pads) == 4:
                    pads = [pads[0], pads[2]]
                strides = self.shape_str_to_list(op_.attrs['strides'])
                ufold = torch.nn.Unfold(tuple(k_shape), dilation=dia, padding=pads, stride=strides)
                inp = ufold(torch.Tensor(input)).permute([1, 0, 2]).numpy()
                # inp = inp.reshape(inp.shape[0]//group,-1)
                inp = inp.reshape(inp.shape[0], -1)
            elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
                inp = input.reshape(-1, shape[-1]).transpose()
            else:
                print("not support!")
                sys.exit(1)
            self.H[op] *= self.samples[op]/(self.samples[op]+in_num)
            self.samples[op] = self.samples[op]+in_num
            inp = np.sqrt(2/self.samples[op])*inp
            self.H[op] += np.matmul(inp, inp.transpose())

    def learning_one(self, epoch, op, total):
        self.loger.logging(f"now to learn {op} in epoch {epoch}")
        sub_total = 1
        if epoch == 0:
            sub_total += 1
        if epoch == self.epoch - 1:
            sub_total += 1
        pbar_detail = tqdm(np.arange(self.num_sample*sub_total))
        pbar_detail.set_description("Learning Gptq Weight, op %s" % op)

        if self.chip == 'bm1688':
            input_bw = 4
            weight_bw = 4
            output_bw = 8
        else:
            input_bw = 8
            weight_bw = 8
            output_bw = 8
        if epoch == 0:
            self.quant_requant_weight_orig(op, bits=weight_bw)
            for loop in np.arange(self.num_sample):
                pbar_detail.set_postfix_str(
                    f"Cal orig loss {epoch}.{loop+1}/{self.epoch}.{self.num_sample} [Total: {total}]")
                pbar_detail.update()
                self.set_op_inputs(op, loop, bitwidth=input_bw, quant=False)
                outputs = self.module.invoke_at(op)
                if output_bw == 4:
                    scale = self.scales4[op][0]
                    unsigned = self.scales4[op][1] >= 0 and self.scales4[op][2] >= 0
                else:
                    scale = self.scales[op][0]
                    unsigned = self.scales[op][1] >= 0 and self.scales[op][2] >= 0
                if self.compare_quanted:
                    if self.support_unsigned:
                        outputs[0] = quant_requant_active(outputs[0], scale, unsigned, bits=output_bw)
                    else:
                        outputs[0] = quant_requant_active(outputs[0], scale, False, bits=output_bw)
                ref = self.ref_tensors.get(op, loop)
                if op in self.pre_loss:
                    pre_loss = self.pre_loss[op] + cal_loss(outputs, ref)
                    self.pre_loss[op] = pre_loss
                else:
                    pre_loss = cal_loss(outputs, ref)
                    self.pre_loss[op] = pre_loss
            self.restore_weight(op)

        for loop in np.arange(self.num_sample):
            pbar_detail.set_postfix_str(
                f"Learning {epoch}.{loop+1}/{self.epoch}.{self.num_sample} [Total: {total}]")
            pbar_detail.update()
            input = self.get_op_input0(op, loop, bitwidth=input_bw, quanted=False)
            if epoch == 0 and loop == 0:
                self.update_H(op, input, None)  # init H
            self.update_H(op, input, None)

        if epoch == self.epoch-1:
            self.quant_requant_weight(op, bitwidth=weight_bw)
            for loop in np.arange(self.num_sample):
                pbar_detail.set_postfix_str(
                    f"Comparing {epoch}.{loop+1}/{self.epoch}.{self.num_sample} [Total: {total}]")
                pbar_detail.update()
                self.set_op_inputs(op, loop, bitwidth=input_bw, quant=False)
                self.module.invoke_at(op)
                outputs = self.module.get_tensor(op)
                if output_bw == 4:
                    scale = self.scales4[op][0]
                    unsigned = self.scales4[op][1] >= 0 and self.scales4[op][2] >= 0
                else:
                    scale = self.scales[op][0]
                    unsigned = self.scales[op][1] >= 0 and self.scales[op][2] >= 0
                if self.compare_quanted:
                    if self.support_unsigned:
                        outputs[0] = quant_requant_active(outputs[0], scale, unsigned, bits=output_bw)
                    else:
                        outputs[0] = quant_requant_active(outputs[0], scale, False, bits=output_bw)
                ref = self.ref_tensors.get(op, loop)
                if op in self.post_loss:
                    post_loss = self.post_loss[op] + cal_loss(outputs, ref)
                    self.post_loss[op] = post_loss
                else:
                    post_loss = cal_loss(outputs, ref)
                    self.post_loss[op] = post_loss

            if self.post_loss[op] <= self.pre_loss[op]:
                self.loger.logging(f'{op} use trained weight {self.post_loss[op]} vs {self.pre_loss[op]}')
            else:
                self.loger.logging(f'{op} do not use learned weight {self.post_loss[op]} vs {self.pre_loss[op]}')
            self.restore_weight(op)

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
                    for result in pool.map(learning_gptweight_wrap, reqs):
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
