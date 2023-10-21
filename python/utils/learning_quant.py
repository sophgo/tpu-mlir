#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import math
from tqdm import tqdm
import gc
import copy
from scipy.special import expit

from datetime import datetime
import time
from multiprocessing.pool import ThreadPool
from multiprocessing import Lock


import pymlir
pymlir.set_mem_mode("value_mem")
from utils.mlir_parser import MlirParser
from utils.preprocess import preprocess
from calibration.data_selector import DataSelector

import torch

SKIP_OPERATION = [
    'top.Input', 'top.Reshape', 'top.Softmax', 'top.Weight', 'top.MaxPool', 'top.Slice', 'top.Tile',
    'top.Permute', 'top.Upsample'
]

LEARNING_WEIGHT_OPERATION = [
    'top.Conv', 'top.MatMul'#
]

def r_show(op, alpha, weight):
    '''
    import matplotlib.pyplot as plt
    shape = alpha.shape
    bins=200
    min_ = np.min(alpha)
    max_ = np.max(alpha)
    w = copy.deepcopy(alpha.reshape(shape[0],-1).clip(min_, max_))
    h_ = np.histogram(w,bins=bins, range=(min_,max_))
    hist = h_[0][:]
    f=plt.figure()
    f.set_figwidth(10)
    f.set_figheight(6)
    plt.plot(hist)
    plt.savefig('./dist/'+op.replace('/','_')+'_weight')
    plt.close()
    h_ = np.histogram(weight,bins=bins, range=(0,1))
    hist = h_[0][:]
    f=plt.figure()
    f.set_figwidth(10)
    f.set_figheight(6)
    plt.plot(hist)
    plt.savefig('./dist/'+op.replace('/','_')+'_alpha')
    plt.close()
    '''
    return

def remote_show(op, alpha, iter):
    '''
    import matplotlib.pyplot as plt
    shape = alpha.shape
    bins=200
    min_,max_ = -3,3
    w = copy.deepcopy(alpha.reshape(shape[0],-1).clip(min_, max_))
    h_ = np.histogram(w,bins=bins, range=(min_,max_))
    hist = h_[0][:]
    f=plt.figure()
    f.set_figwidth(10)
    f.set_figheight(6)
    plt.plot(hist)
    plt.savefig('./dist/'+op.replace('/','_')+'_'+str(int(iter)))
    plt.close()
    rec = np.clip(1.2*(1/(1+np.exp(-1*w))) - 0.1, 0, 1)
    min_,max_ = 0,1
    h_ = np.histogram(rec,bins=bins, range=(min_,max_))
    hist = h_[0][:]
    #hist = h_[0][1:bins-1]
    f=plt.figure()
    f.set_figwidth(10)
    f.set_figheight(6)
    plt.plot(hist)
    plt.savefig('./dist/'+op.replace('/','_')+'_rect_'+str(int(iter)))
    plt.close()
    '''
    return

def quant_requant_active(data, scale, unsigned=False,bits=8):
    if unsigned:
        d = data/scale*(2**bits-1)
        dout = np.round(d)
        return np.clip(dout, 0, 2**bits)/(2**bits) * scale
    else:
        d = data/scale*(2**(bits-1))
        dout = np.round(d)
        return np.clip(dout, -(2**(bits-1)), 2**(bits-1)-1)/(2**(bits-1)) * scale

def cal_loss(target, ref):
    mse_diff = ((target - ref)**2).mean()
    return mse_diff

def learning_adaweight_wrap(reqs):
    cls, epoch, op, total = reqs
    return cls.learning_one(epoch, op, total)

def learning_gptweight_wrap(reqs):
    cls, epoch, op, total = reqs
    return cls.learning_one(epoch, op, total)

def learning_scale_wrap(reqs):
    cls, op, total = reqs
    return cls.learning_one(op, total)

def into_groups(parser, layers):
    groups = {}
    group_idx = 0
    for l in layers:
        allocated = False
        if group_idx == 0 and group_idx not in groups:
            groups[0] = [l]
            group_idx += 1
            continue
        for i in np.arange(group_idx):
            notin = True
            for out_op in parser.get_next_op_by_op_name(l):
                for layer in groups[i]:
                    if out_op not in parser.get_pre_op_by_op_name(layer) and out_op not in parser.get_next_op_by_op_name(layer):
                        continue
                    else:
                        notin = False
                        break
            if notin:
                for in_op in parser.get_pre_op_by_op_name(l):
                    for layer in groups[i]:
                        if in_op not in parser.get_next_op_by_op_name(layer) and in_op not in parser.get_pre_op_by_op_name(layer):
                            continue
                        else:
                            notin = False
                            break
                if notin:
                    groups[i].append(l)
                    allocated = True
                    break
            if allocated:
                break
        if allocated:
            continue
        else:
            groups[group_idx] = [l]
            group_idx += 1
    return groups

class logging:
    def __init__(self, filename = "logging"):
        self.file_name = filename
        self.log_file = open(self.file_name,'w')

    def logging(self, info):
        print(info, file=self.log_file)

    def end(self):
        if self.log_file is not None:
            self.log_file.close()


class learning_inputs:
    def __init__(self, parser, args):
        self.dataset = args.dataset
        self.data_list = args.data_list
        self.batch_size = parser.get_batch_size()
        self.input_num = parser.get_input_num()
        self.num_sample = 0
        self.parser = parser
        self.ref_activations = {}

    def prepare(self, input_num):
        tune_idx = 0
        self.ref_activations[tune_idx] = {}
        input_names = [op.name for op in self.parser.inputs]
        ds = DataSelector(self.dataset, input_num, self.data_list)
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
            raise RuntimeError("dataset is incorrect")
        return self.num_sample

class ref_tensors:
    def __init__(self, learner, inputs):
        self.parser = learner.parser
        self.module = learner.module
        self.net_inputs = inputs
        self.epoch_samples = inputs.num_sample
        self.ops = {}
        self.ops_cnt = {}
        self.ops_buffer = {}
        self.init()

    def add_name(self, ops):
        for op in ops:
            if op in self.ops:
                continue
            else:
                self.ops[op] = 0
                self.ops_cnt[op] = 0

    def init(self):
        self.add_name(self.module.all_tensor_names)
        self.add_name(self.parser.get_op_name_list())
        for op in self.ops:
            cnt = 0
            for op_ in self.parser.ops:
                for op__ in op_.opds:
                    if op__ in self.parser.get_op_name_list() and op__ == op:
                        cnt += 1
            self.ops[op] = cnt
            if op in self.module.output_names:
                self.ops[op] += 1

    def get(self, op, idx, quant=False, symetric=True):
        if idx > self.epoch_samples:
            print(f"requested idx out of range {idx} vs {self.epoch_samples}")
        #if already buffered return the buffer
        if op in self.ops_buffer:
            return self.ops_buffer[op][idx]
        elif op in self.net_inputs.ref_activations[0]:
            return self.net_inputs.ref_activations[idx][op][0]

        #if not bufferred, generate all samples
        inputs = self.parser.get_pre_op_by_op_name(op)
        for in_ in inputs:
            if in_ not in self.ops_buffer and in_ not in self.net_inputs.ref_activations[0]:
                loger.logging(f'recursive get {in_}')
                self.get(in_, idx, quant, symetric)
                self.ops_cnt[in_] = self.ops[in_]
        for l in range(self.epoch_samples):
            for in_ in inputs:
                if in_ in self.ops_buffer:
                    self.module.set_tensor(in_, self.ops_buffer[in_][l])
                elif in_ in self.net_inputs.ref_activations[0]:
                    self.module.set_tensor(in_, self.net_inputs.ref_activations[l][in_][0])
                outputs = self.module.invoke_at(op)
                for out_ in self.parser.get_outputs_by_op_name(op):
                    if out_ in self.ops and self.ops[out_] > 0:
                        if out_ in self.ops_buffer:
                            self.ops_buffer[out_].append(self.module.get_tensor(out_).copy())
                        else:
                            self.ops_buffer[out_] = []
                            self.ops_buffer[out_].append(self.module.get_tensor(out_).copy())
                    if l == 0:
                        loger.logging(f'adding {out_} cnt {self.ops[out_]}')
        for out_ in self.parser.get_outputs_by_op_name(op):
            if out_ in self.ops and self.ops[out_] > 0:
                loger.logging(f'setting {out_} {self.ops[out_]}')
                self.ops_cnt[out_] = self.ops[out_]
        for in_ in inputs:
            if in_ in self.ops_buffer:
                self.ops_cnt[in_] -= 1
                if self.ops_cnt[in_] == 0:
                    del self.ops_buffer[in_]
                    loger.logging(f'del in {in_}')
        if op not in self.ops_buffer:
            print(f'{op} not in ref tensors!')
            sys.exit(1)
        return self.ops_buffer[op][idx]

    def consumed_tensor(self, op):  # must call when loop over epoch of using the tensor is done
        if op in self.net_inputs.ref_activations[0]:
            return
        if op not in self.ops_buffer:
            print(f"{op} not in buffer when mark used!")
            #sys.exit(1)
        else:
            loger.logging(f'used, del 1 {op}')
            self.ops_cnt[op] -= 1
            if self.ops_cnt[op] == 0:
                loger.logging(f'end used, del {op}')
                del self.ops_buffer[op]

class LrScheduler:
    def __init__(self, lr, max_iter, mode):
        self.lr = lr
        self.min_lr = lr/10
        self.mode = mode
        self.max_iter = max_iter
        self.scale = 0.1
        self.warm_up = 0.2

    def cal_lr(self, iter):
        if self.mode == 'Fixed':
            return self.lr
        elif self.mode == 'Cosine':
            if iter <= self.max_iter * self.warm_up:
                return self.lr
            else:
                return self.min_lr + 0.5* (self.lr-self.min_lr)*(1.0+np.cos((iter-self.max_iter*self.warm_up)/(self.max_iter-self.max_iter*self.warm_up)*np.pi))
        elif self.mode == 'MultiStep':
            if iter <= self.max_iter * 0.5:
                return self.lr
            elif iter <= self.max_iter*2/3:
                return self.lr * self.scale
            else:
                return self.lr * self.scale * self.scale

class CaliTable:
    def __init__(self, in_table, out_table):
        self.in_table = in_table
        self.out_table = out_table
        self.table = {}
        self.read()

    def read(self):
        textfile = open(self.in_table, 'r')
        for line in textfile:
            if len(line) > 0 and (not line.startswith("#")):
                s = line.split(" ")
                s = [x for x in s if x != '']
                if len(s) != 4:
                    continue
                self.table[s[0]] = [
                    float(s[1]), float(s[2]), float(s[3])]
            if line.startswith("#int4_th"):
                break

    def update(self, new_table):
        for op in new_table:
            for op_ in self.table:
                if op_ == op:
                    self.table[op][0] = new_table[op][0]

    def write(self):
        f = open(self.out_table, 'w')
        for op in self.table:
            f.write(
                f'{op}  {self.table[op][0]}  {self.table[op][1]}  {self.table[op][2]}\n')
        f.close()


DEBUG=False
class LearningGptqWeight:

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
                    x = x.reshape(shape[0],-1)
                else:
                    if len(shape) == 4:
                        x = x.permute([1, 0, 2, 3])
                        x = x.reshape(shape[0],-1)
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
                    q = np.power(np.abs(q-x,self.norm))
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

        def quantize(self, x):  #fixme , check perchannel
            q = np.clip(np.round(x / self.scale) +self.zero, 0, self.maxq)
            return self.scale * (q - self.zero)

    def __init__(self, args):
        self.scales = None
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
        loger.logging(f'Learning Gptq Weight running on layers and weights: {self.finetune_layers}')

    def backup_weights(self):
        for op in self.finetune_layers:
            self.orig_weights[op] = copy.deepcopy(self.module.get_tensor(self.finetune_layer_weights[op]))
            oc = self.orig_weights[op].shape[0]
            if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
                self.weights_scales[op] = np.max(np.abs(self.orig_weights[op].reshape(oc,-1)), axis=1)
                self.weights_scales[op] = np.where(self.weights_scales[op]>1e-8, self.weights_scales[op], 1e-8)
            elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
                self.weights_scales[op] = np.max(np.abs(self.orig_weights[op]))
                self.weights_scales[op] = np.where(self.weights_scales[op]>1e-8, self.weights_scales[op], 1e-8)
            else:
                print("not support!")
                sys.exit(1)
    def restore_weight(self, op):
        self.module.set_tensor(self.finetune_layer_weights[op], self.orig_weights[op])

    def quant_requant_weight(self, op, blocksize=128, percdamp=0.01, groupsize=-1, bitwidth=8, actorder=False):
        W = torch.Tensor(self.orig_weights[op].copy())

        tick  = time.time()

        if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
            W = W.reshape(shape[0],-1)
            shape = W.shape
            rows = W.reshape(shape[0],-1).shape[0]
            columns = W.reshape(shape[0],-1).shape[1]
            quanter = self.GptqQuantizer(shape, bits=bitwidth, perchannel=True, sym=True, mse=False)
        elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
            W = W.t()
            shape = W.shape
            rows = W.shape[0]
            columns = W.reshape(W.shape[0],-1).shape[1]
            quanter = self.GptqQuantizer(shape, bits=bitwidth, perchannel=False, sym=True, mse=False)
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
            self.module.set_tensor(self.finetune_layer_weights[op], Q.numpy().transpose().reshape(shape))

    def quant_requant_weight_orig(self, op, bits=8):
        weight_tmp = copy.deepcopy(self.orig_weights[op])
        scales = self.weights_scales[op]/(2**(bits-1)-1)
        shape=weight_tmp.shape
        if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
            weight_tmp = (weight_tmp.reshape(shape[0],-1)/scales[:,None]).reshape(shape)
        if self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
            weight_tmp = weight_tmp/scales
        weight = np.clip(np.round(weight_tmp), -(2**(bits-1)), (2**(bits-1)-1))
        if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
            self.module.set_tensor(self.finetune_layer_weights[op], (weight.reshape(shape[0],-1)*scales[:,None]).reshape(shape))
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
                    pop=pre_pre_ops[0]
                else:
                    print(f"{op} input {pop} not in all tensor list")
                    sys.exit(1)
            d = self.ref_tensors.get(pop, loop).reshape(shape)
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
                    pop=pre_pre_ops[0]
                else:
                    print(f"{op} input {pop} not in all tensor list")
                    sys.exit(1)
            d = self.ref_tensors.get(pop, loop).reshape(shape).copy()
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
        s = shape.replace('[','').replace(']','').replace(' ','').split(',')
        s = [int(x) for x in s]
        return s

    def update_H(self, op, input, output):
        shape = input.shape
        in_num = shape[0]
        if op not in self.H:
            weight_shape = self.orig_weights[op].shape
            if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
                weight_shape = self.orig_weights[op].reshape(weight_shape[0],-1).shape
            elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
                weight_shape = self.orig_weights[op].transpose().shape
            else:
                print('not support!')
                sys.exit(1)
            self.H[op] = np.zeros((weight_shape[1],weight_shape[1]))
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
                #group = int(op_.attrs['group'].split(':')[0])
                if len(pads) == 4:
                    pads = [pads[0], pads[2]]
                strides = self.shape_str_to_list(op_.attrs['strides'])
                ufold = torch.nn.Unfold(tuple(k_shape), dilation=dia, padding = pads, stride=strides)
                inp = ufold(torch.Tensor(input)).permute([1,0,2]).numpy()
                #inp = inp.reshape(inp.shape[0]//group,-1)
                inp = inp.reshape(inp.shape[0],-1)
            elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
                inp = input.reshape(-1,shape[-1]).transpose()
            else:
                print("not support!")
                sys.exit(1)
            self.H[op] *= self.samples[op]/(self.samples[op]+in_num)
            self.samples[op] = self.samples[op]+in_num
            inp = np.sqrt(2/self.samples[op])*inp
            self.H[op] += np.matmul(inp, inp.transpose())

    def learning_one(self, epoch, op, total):
        loger.logging(f"now to learn {op} in epoch {epoch}")
        sub_total = 1
        if epoch == 0:
            sub_total += 1
        if epoch == self.epoch - 1:
            sub_total += 1
        pbar_detail = tqdm(np.arange(self.num_sample*sub_total))
        pbar_detail.set_description("Learning Gptq Weight, op %s" % op)

        if self.chip == 'bm1686':
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
                self.set_op_inputs(op, loop, bitwidth = input_bw, quant=False)
                outputs = self.module.invoke_at(op)
                scale = self.scales[op][0]
                unsigned = self.scales[op][1] >= 0 and self.scales[op][2] >= 0
                if self.compare_quanted:
                    if self.support_unsigned:
                        outputs[0] = quant_requant_active(outputs[0], scale, unsigned, bits = output_bw)
                    else:
                        outputs[0] = quant_requant_active(outputs[0], scale, False, bits = output_bw)
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
            input = self.get_op_input0(op, loop, bitwidth=input_bw,quanted=False)
            if epoch == 0 and loop == 0:
                self.update_H(op, input, None) # init H
            self.update_H(op, input, None)

        if epoch == self.epoch-1:
            self.quant_requant_weight(op, bitwidth = weight_bw)
            for loop in np.arange(self.num_sample):
                pbar_detail.set_postfix_str(
                    f"Comparing {epoch}.{loop+1}/{self.epoch}.{self.num_sample} [Total: {total}]")
                pbar_detail.update()
                self.set_op_inputs(op, loop, bitwidth=input_bw, quant=False)
                self.module.invoke_at(op)
                outputs = self.module.get_tensor(op)
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
                loger.logging(f'{op} use trained weight {self.post_loss[op]} vs {self.pre_loss[op]}')
            else:
                loger.logging(f'{op} do not use learned weight {self.post_loss[op]} vs {self.pre_loss[op]}')
            self.restore_weight(op)

    def save_weights(self):
        os.rename(self.weight_file, self.weight_file.replace(".npz",".bak.npz"))
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

class LearningAdaWeight:
    class SgdWeightOpt:
        def __init__(self,lr, momentum=0.0,nesterov=False, weight_decay=0.0, support_unsigned = False):
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

            alpha_new = self.cal_alpha(iter, alpha, self.loss[op], self.grd[op], unsigned)
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

    def __init__(self, args):
        self.scales = None
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
        self.compare_quanted = True
        print(
            f'Learning Weight, momentum is {self.momentum} nesterov is {self.nesterov} weight_decay is {self.weight_decay}')
        self.v = {}
        self.support_unsigned = False
        self.get_finetune_ops(args.excepts)
        self.backup_weights()
        if self.mini_batch <= self.batch_size:
            self.mini_batch = 1
        else:
            self.mini_batch = self.mini_batch // self.batch_size
        w = np.load(self.weight_file, allow_pickle=True)
        for k in w:
            self.param_back[k] = w[k]

    def sigmoid(self, x):
        return expit(x)

    def exp_neg(self, x):
        return (1.0/(expit(x)+1e-8))-1.0

    def rec_sig(self, x):
        return np.clip(self.sigmoid(x)*1.2-0.1,0,1)

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

    def cal_grd_signed(self, out, scale):
        # calculate output mse grd with quant grad
        step = np.abs(scale)/127.0
        grd = np.zeros_like(out)
        m = np.round(out/step)
        qmin = np.ones_like(m)*(-128)
        qmax = np.ones_like(m)*(127)
        g = (np.minimum(np.maximum(m, qmin), qmax)*step - out)*2
        grd_u = np.where(m <= -128, -128, 0)*g
        grd_l = np.where(m >= 127, 127, 0)*g
        left = np.where(m > -128, 1, 0) & np.where(m < 127, 1, 0)
        grd_m = (m - out/step)*left
        return grd_m + grd_u + grd_l

    def cal_grd(self, out, scale, unsigned = False):
        return self.cal_grd_signed(out, scale)

    def cal_grdr_signed(self, alpha, beta, iter, reg=0.01):
        if iter < self.num_sample * self.epoch * self.beta_warmup:
            return np.zeros_like(alpha)
        else:
            sig_a = self.sigmoid(alpha)
            rect_a = 2*np.clip(sig_a*1.2-0.1, 0, 1)-1
            pos = np.where(alpha>=0, 1, 0)
            neg = np.where(alpha<0, 1, 0)
            p_eff = np.where((sig_a*1.2-0.1)>1.0, 1, 0)
            n_eff = np.where((sig_a*1.2-0.1)<0, 1, 0)
            pos = pos - p_eff
            neg = neg - n_eff
            grdp = -2.4*beta*np.power(rect_a, beta-1)*sig_a*(1-sig_a)*pos
            grdn = 2.4*beta*np.power(-rect_a, beta-1)*sig_a*(1-sig_a)*neg
            grd = (grdp+grdn)*reg
            return grd

    def cal_grdr(self, alpha, beta, iter, unsigned = False):
        return self.cal_grdr_signed(alpha, beta, iter)

    def get_finetune_ops(self, excepts):
        top_ops = {op.name: op for op in self.parser.ops}
        exclude = excepts.split(',')
        for op in top_ops:
            if top_ops[op].type in LEARNING_WEIGHT_OPERATION and top_ops[op].name not in exclude:
                if len(top_ops[op].opds) > 1 and top_ops[op].opds[1] in self.module.all_weight_names:
                    self.finetune_layers.append(op)
                    self.finetune_layer_weights[op] = top_ops[op].opds[1]
        loger.logging(f'Learning Weight running on layers and weights: {self.finetune_layers}')

    def backup_weights(self):
        for op in self.finetune_layers:
            self.orig_weights[op] = copy.deepcopy(self.module.get_tensor(self.finetune_layer_weights[op]))
            oc = self.orig_weights[op].shape[0]
            if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
                self.weights_scales[op] = np.max(np.abs(self.orig_weights[op].reshape(oc,-1)), axis=1)
                self.weights_scales[op] = np.where(self.weights_scales[op]>1e-8, self.weights_scales[op], 1e-8)
            elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
                self.weights_scales[op] = np.max(np.abs(self.orig_weights[op]))
                self.weights_scales[op] = np.where(self.weights_scales[op]>1e-8, self.weights_scales[op], 1e-8)
            else:
                print("not support!")
                sys.exit(1)

    def restore_weight(self, op):
        self.module.set_tensor(self.finetune_layer_weights[op], self.orig_weights[op])

    def quant_requant_weight(self, op, hard=False):
        weight_tmp = self.orig_weights[op].copy()
        scales = self.weights_scales[op]/127.0
        shape=weight_tmp.shape
        if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
            weight_tmp = (weight_tmp.reshape(shape[0],-1)/scales[:,None]).reshape(shape)
        elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
            weight_tmp = weight_tmp/scales
        else:
            print("not support!")
            sys.exit(1)
        if op not in self.alpha:  # init
            r_show(op, weight_tmp, weight_tmp-np.floor(weight_tmp))
            alpha = weight_tmp - np.floor(weight_tmp)
            alpha = -np.log((self.zeta-self.gamma)/(alpha - self.gamma)-1) # this is where alpha is coming, refer to ppx and aimet
            self.alpha[op] = alpha
        else:
            alpha = self.alpha[op]
        if not hard:
            if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
                weight = (np.clip(np.floor(weight_tmp)+self.rec_sig(alpha),-128,127).reshape(shape[0],-1)*scales[:,None]).reshape(shape)
            elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
                weight = np.clip(np.floor(weight_tmp)+self.rec_sig(alpha),-128,127)*scales
        else:
            if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
                weight = (np.clip(np.floor(weight_tmp)+(alpha>=0).astype(np.float32),-128,127).reshape(shape[0],-1)*scales[:,None]).reshape(shape)
            elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
                weight = np.clip(np.floor(weight_tmp)+(alpha>=0).astype(np.float32),-128,127)*scales
        self.module.set_tensor(self.finetune_layer_weights[op], weight)

    def quant_requant_weight_orig(self, op):
        weight_tmp = copy.deepcopy(self.orig_weights[op])
        scales = self.weights_scales[op]/127.0
        shape=weight_tmp.shape
        if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
            weight_tmp = (weight_tmp.reshape(shape[0],-1)/scales[:,None]).reshape(shape)
        if self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
            weight_tmp = weight_tmp/scales
        weight = np.clip(np.round(weight_tmp), -128.0, 127.0)
        if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
            self.module.set_tensor(self.finetune_layer_weights[op], (weight.reshape(shape[0],-1)*scales[:,None]).reshape(shape))
        elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
            self.module.set_tensor(self.finetune_layer_weights[op], weight*scales)
        else:
            print("not support!")
            sys.exit(1)

    def set_op_inputs(self, op, loop, quant=True):
        pre_ops = self.parser.get_pre_op_by_op_name(op)
        for pop in pre_ops:
            shape = self.ref_tensors.get(pop, loop).shape
            if pop not in self.module.all_tensor_names:
                if self.parser.get_op_type_by_op_name(pop) == 'top.Reshape':
                    pre_pre_ops = self.parser.get_pre_op_by_op_name(pop)
                    pop=pre_pre_ops[0]
                else:
                    print(f"{op} input {pop} not in all tensor list")
                    sys.exit(1)
            d = self.ref_tensors.get(pop, loop).reshape(shape)
            scale = self.scales[pop][0]
            unsigned = self.scales[op][1] >= 0 and self.scales[op][2] >= 0
            if scale != 1.0 and quant:
                if self.support_unsigned:
                    d = quant_requant_active(d, scale, unsigned)
                else:
                    d = quant_requant_active(d, scale, False)
            self.module.set_tensor(pop, d)

    def op_first_input(self, op, loop):
        pre_ops = self.parser.get_pre_op_by_op_name(op)
        if len(pre_ops) != 1:
            print(f'input num not 1! {op}')
            sys.exit(1)
        pop = pre_ops[0]
        shape = self.ref_tensors.get(pop, loop).shape
        if pop not in self.module.all_tensor_names:
            if self.parser.get_op_type_by_op_name(pop) == 'top.Reshape':
                pre_pre_ops = self.parser.get_pre_op_by_op_name(pop)
                pop=pre_pre_ops[0]
            else:
                print(f"{op} input {pop} not in all tensor list")
                sys.exit(1)
        d = self.ref_tensors.get(pop, loop).reshape(shape)
        scale = self.scales[pop][0]
        unsigned = self.scales[op][1] >= 0 and self.scales[op][2] >= 0
        if scale != 1.0:
            if self.support_unsigned:
                d = quant_requant_active(d, scale, unsigned)
            else:
                d = quant_requant_active(d, scale, False)
        return d

    def learning_one(self, epoch, op, total):
        loger.logging(f"now to learn {op} in epoch {epoch}")
        sub_total = 1
        if epoch == 0:
            sub_total += 1
        if epoch == self.epoch - 1:
            sub_total += 1
        pbar_detail = tqdm(np.arange(self.num_sample*sub_total))
        pbar_detail.set_description("Learning Weight, op %s" % op)

        if epoch == 0:
            self.quant_requant_weight_orig(op)
            for loop in np.arange(self.num_sample):
                pbar_detail.set_postfix_str(
                    f"Cal orig loss {epoch}.{loop+1}/{self.epoch}.{self.num_sample} [Total: {total}]")
                pbar_detail.update()
                self.set_op_inputs(op, loop, quant=True)
                outputs = self.module.invoke_at(op)
                scale = self.scales[op][0]
                unsigned = self.scales[op][1] >= 0 and self.scales[op][2] >= 0
                if self.compare_quanted:
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
                f"Learning {epoch}.{loop+1}/{self.epoch}.{self.num_sample} [Total: {total}]")
            pbar_detail.update()
            self.set_op_inputs(op, loop, quant=True)
            self.quant_requant_weight(op)
            outputs = self.module.invoke_at(op)
            scale = self.scales[op][0]
            unsigned = self.scales[op][1] >= 0 and self.scales[op][2] >= 0
            if self.support_unsigned:
                outputq = quant_requant_active(outputs[0], scale, unsigned)
            else:
                outputq = quant_requant_active(outputs[0], scale, False)
            ref = self.ref_tensors.get(op, loop)
            beta = self.cal_beta(epoch*self.num_sample+loop)
            loss = cal_loss(outputq, ref)
            loger.logging(f"loss of {op} in loop {loop} is {loss}")
            loss += self.cal_round_loss(epoch*self.num_sample+loop, self.alpha[op], beta)
            loger.logging(f"loss of {op} in loop {loop} plus is {loss}")
            self.opt.update_loss(op, loss)
            unsigned = self.scales[op][1] >= 0 and self.scales[op][2] >= 0
            grd_dst = self.cal_grd(outputs[0], scale, unsigned)
            if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
                grd_w = self.module.backward_weight_at(op, self.finetune_layer_weights[op], grd_dst)
            elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
                #should consider transpose? matmul with weight has one input
                input = self.op_first_input(op, loop)
                shape = input.shape
                input = input.reshape(-1,shape[-1]).transpose()
                if len(grd_dst.shape) == 1:
                    grd_dst=grd_dst.reshape(1,grd_dst.shape[0])
                shape = grd_dst.shape
                grd_d = grd_dst.reshape(-1,shape[-1])
                grd_w = np.matmul(input, grd_d)
                grd_w = grd_w/(np.prod(shape)/(shape[-1]*shape[-2]))
            shape = self.alpha[op].shape
            exp_alpha = self.exp_neg(self.alpha[op])
            grd_w1 =1.2*(exp_alpha/((exp_alpha+1)*(exp_alpha+1)))
            if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
                grd_w1 = (grd_w1.reshape(shape[0],-1)*((self.weights_scales[op]/127.0)[:,None])).reshape(shape)
            elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
                grd_w1 =1.2*(exp_alpha/((exp_alpha+1)*(exp_alpha+1)))*(self.weights_scales[op]/127.0)
            else:
                print("not support!")
                sys.exit(1)
            grd_w = grd_w * grd_w1
            grd_r = self.cal_grdr(self.alpha[op], beta, epoch*self.num_sample+loop, unsigned)
            grd = grd_w + grd_r
            self.opt.update_grd(op, grd)
            if (epoch*self.num_sample+loop+1) % self.mini_batch == 0:
                self.alpha[op] = self.opt.update_alpha(epoch*self.num_sample+loop, op, self.alpha[op], self.mini_batch)
                if (loop+1) % 4 == 0:
                    remote_show(op, self.alpha[op], loop)

        if epoch == self.epoch-1:
            self.quant_requant_weight(op, True)
            for loop in np.arange(self.num_sample):
                pbar_detail.set_postfix_str(
                    f"Comparing {epoch}.{loop+1}/{self.epoch}.{self.num_sample} [Total: {total}]")
                pbar_detail.update()
                self.set_op_inputs(op, loop, quant=True)
                self.module.invoke_at(op)
                outputs = self.module.get_tensor(op)
                scale = self.scales[op][0]
                unsigned = self.scales[op][1] >= 0 and self.scales[op][2] >= 0
                if self.compare_quanted:
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
            self.restore_weight(op)

            if self.post_loss[op] <= self.pre_loss[op]:
                loger.logging(f'{op} use trained weight {self.post_loss[op]} vs {self.pre_loss[op]}')
                print(f'{op} use trained weight {self.post_loss[op]} vs {self.pre_loss[op]}')
                self.update_weight(op)
            else:
                loger.logging(f'{op} do not use learned weight {self.post_loss[op]} vs {self.pre_loss[op]}')
                print(f'{op} do not use learned weight {self.post_loss[op]} vs {self.pre_loss[op]}')

    def adjust_weight(self, op):
        shape = self.orig_weights[op].shape
        w_reshape = self.orig_weights[op].reshape(shape[0],-1)
        adj = np.argmax(self.orig_weights[op].reshape(shape[0],-1),axis=1)
        p = self.alpha[op].reshape(shape[0],-1)
        for i in np.arange(shape[0]):
            if p[i][adj[i]] >= 0 and w_reshape[i][adj[i]] <= 0:
                p[i][adj[i]] = -p[i][adj[i]]
            elif p[i][adj[i]] < 0 and w_reshape[i][adj[i]] > 0:
                p[i][adj[i]] = -p[i][adj[i]]
        if self.parser.get_op_type_by_op_name(op) == 'top.Conv':
            weight = ((np.floor(w_reshape/((self.weights_scales[op]/127.0)[:,None])) + np.where(p>=0, 1.0, 0.0).astype(np.float32)) * ((self.weights_scales[op]/127.0)[:,None])).reshape(shape)
        elif self.parser.get_op_type_by_op_name(op) == 'top.MatMul':
            weight = ((np.floor(w_reshape/(self.weights_scales[op]/127.0)) + np.where(p>=0, 1.0, 0.0).astype(np.float32)) * (self.weights_scales[op]/127.0)).reshape(shape)
        else:
            print("not support!")
            sys.exit(1)
        return weight

    def update_weight(self, op):
        self.param_back[self.finetune_layer_weights[op]] = self.adjust_weight(op)

    def save_weights(self):
        os.rename(self.weight_file, self.weight_file.replace(".npz",".bak.npz"))
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

class LearningScale:
    class SgdScaleOpt:
        def __init__(self,lr, momentum=0.0,nesterov=False, weight_decay=0.0, support_unsigned=False):
            self.lr = lr
            self.momentum = momentum
            self.nesterov = nesterov
            self.weight_decay = weight_decay
            self.dampening = 0.0
            self.v = {}
            self.grd = {}
            self.loss = {}
            self.support_unsigned = support_unsigned
            print(
                f'Learning Scale SGD, momentum is {self.momentum} nesterov is {self.nesterov} weight_decay is {self.weight_decay}')
        def cal_scale(self, iter, scale, loss, grd, unsigned=False):
            if unsigned:
                step = np.abs(scale)/255.0
            else:
                step = np.abs(scale)/127.0

            step = step - self.lr.cal_lr(iter) * grd
            loger.logging(
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
        def __init__(self, lr, beta1=0.9, beta2=0.999, weight_decay=0.0, support_unsigned=False):
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
            print(
                f'Learning Scale Adam, weight_decay is {self.weight_decay}')

        def cal_scale(self, scale, delta, unsigned=False):
            if unsigned:
                step = np.abs(scale)/255.0
            else:
                step = np.abs(scale)/127.0

            step = step - delta
            loger.logging(
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
                loger.logging(
                    f"update scale {scale/255.0} grd {self.grd[op]} delta {delta}")
            else:
                scale = (np.abs(scale)/127.0 - delta)*127.0
                loger.logging(
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
        self.orig_scales = {}
        self.new_scales = {}
        self.pre_loss = {}
        self.post_loss = {}
        self.support_unsigned = False
        self.opt = None
        self.finetune_layers = []  # layers to fine tune, without skipped
        self.get_finetune_ops(args.excepts)
        if self.mini_batch <= self.batch_size:
            self.mini_batch = 1
        else:
            self.mini_batch = self.mini_batch // self.batch_size

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
                    pop=pre_pre_ops[0]
                else:
                    print(f"{op} input {pop} not in all tensor list")
                    sys.exit(1)
            d = self.ref_tensors.get(pop, loop).copy().reshape(shape)
            scale = 1.0
            if pop in self.new_scales:
                scale = self.new_scales[pop][0]
            elif pop in self.orig_scales:
                scale = self.orig_scales[pop][0]
            unsigned = self.orig_scales[op][1] >= 0 and self.orig_scales[op][2] >= 0
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
        loger.logging(f"now to learn {op} scale")
        pbar_detail = tqdm(np.arange(self.num_sample*3))
        pbar_detail.set_description("Learning Scale, op %s" % op)
        for loop in np.arange(self.num_sample):
            pbar_detail.set_postfix_str(
                f"Cal orig loss {loop} [Total Progress: {total}]")
            pbar_detail.update()
            self.set_op_inputs(op, loop, quant=True)
            outputs = self.module.invoke_at(op).copy()
            scale = self.orig_scales[op][0]
            unsigned = self.orig_scales[op][1] >= 0 and self.orig_scales[op][2] >= 0
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
            elif op in self.orig_scales:
                scale = self.orig_scales[op][0]
            unsigned = self.orig_scales[op][1] >= 0 and self.orig_scales[op][2] >= 0
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
            unsigned = self.orig_scales[op][1] >= 0 and self.orig_scales[op][2] >= 0
            grd = self.cal_grd(outputs[0], scale, True, unsigned)
            self.opt.update_grd(op, grd)
            if (loop+1) % self.mini_batch == 0:
                scale = self.opt.update_scale(loop, op, scale, self.mini_batch, unsigned)
                if op in self.new_scales:
                    self.new_scales[op][0] = scale
                else:
                    self.new_scales[op] = [scale, 0, 0]
                loger.logging("{} new scale is {:.16f} iter {} batch {}".format(
                    op, scale, loop+1, self.mini_batch))

        for loop in np.arange(self.num_sample):
            pbar_detail.set_postfix_str(
                f"Comparing {loop} [Total Progress: {total}]")
            pbar_detail.update()
            self.set_op_inputs(op, loop, quant=True)
            self.module.invoke_at(op)
            outputs = self.module.get_tensor(op).copy()
            scale = self.new_scales[op][0]
            unsigned = self.orig_scales[op][1] >= 0 and self.orig_scales[op][2] >= 0
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


        if self.post_loss[op] >= self.pre_loss[op] or self.new_scales[op][0] < 0 or self.new_scales[op][0]/self.orig_scales[op][0] > 1.5:
            loger.logging(
                f'abandon backward tune of {op}, old loss: {self.pre_loss[op]}, new loss: {self.post_loss[op]}, old scale {self.orig_scales[op][0]} new scale {self.new_scales[op][0]}')
            del self.new_scales[op]
        else:
            loger.logging(
                f'use tune of {op}, old loss: {self.pre_loss[op]}, new loss: {self.post_loss[op]}, old scale {self.orig_scales[op][0]} new scale {self.new_scales[op][0]}')

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

if __name__ == '__main__':
    print("SOPHGO Toolchain {}".format(pymlir.module().version))
    # yapf: disable
    parser = argparse.ArgumentParser(
        description="Learning the scale for quantization, run after basic quant table")
    parser.add_argument('mlir_file', help='fp32 mlir file')
    parser.add_argument(
        '--dataset', required=True, type=str, help='dataset path for mix precision searching')
    parser.add_argument(
        "--data_list", required=False, type=str, help="specify a file with inputs's absolute path for mix precision searching")
    parser.add_argument('--input_num', required=True, type=int, default=1000,
                        help='num of input samples for quantization searching')
    parser.add_argument('--data_seg', required=False, type=int, default=1000,
                        help='num of samples to buffer data on disk, they will be re-aranged after gather all samples')
    parser.add_argument('--epoch', required=False, type=int, default=1,
                        help='num of repeat times of input_num samples for weight learning')
    parser.add_argument('--mini_batch', required=False, type=int, default=4,
                        help='batch size for learning')
    parser.add_argument('--threads', required=False, type=int, default=4,
                        help='number of working threads')
    parser.add_argument('--momentum', required=False, type=float, default=0.9,
                        help='momentum of learning')
    parser.add_argument('--nesterov', required=False, action='store_true', dest='nesterov',
                        help='use nesterov in learning')
    parser.add_argument('--weight_decay', required=False, type=float, default=0.001,
                        help='weight decay in learning')
    parser.add_argument('--lr', required=False, type=float, default=0.001,
                        help='learning rate in learning')
    parser.add_argument('--lr_scheduler', required=False, type=str,default='Cosine',
                        choices=['Fixed','Cosine','MultiStep'],
                        help='lr scheduler')
    parser.add_argument('--calibration_table', required=True,
                        help='calibration table generated by calibration or tune tool')
    parser.add_argument('--chip', required=False, type=str,default='bm1684x',
                        choices=['bm1684x', 'bm1686', 'cv183x',
                                 'cv182x', 'cv181x', 'cv180x'],
                        help='chip platform name')
    parser.add_argument('--opt', required=False, type=str,default='SGD',
                        choices=['SGD','ADAM'],
                        help='Optimizer')
    parser.add_argument('--target', type=str,default='Scale',
                        choices=['Scale','AdaWeight', 'GptWeight'],
                        help='to learn scale or weight or both')
    parser.add_argument('-o', '--output_calibration_table', required=False, default="./new_cali",
                        help='output of calibration table after learning')
    parser.add_argument('-excepts', '--excepts', required=False, default="",
                        help='learning excepts these layers, split with comma')

    args = parser.parse_args()
    if args.chip != "bm1684x" and args.chip != "bm1686":
        print("only support bm1684x and bm1686 till now!")
        sys.exit(1)
    if args.data_seg > args.input_num:
        args.data_seg = args.input_num
    loger = logging()
    pool = ThreadPool(args.threads)
    scale_searcher = LearningScale(args)
    cali_table = CaliTable(args.calibration_table, args.output_calibration_table)
    scale_searcher.orig_scales = cali_table.table
    all_inputs = learning_inputs(scale_searcher.parser, args)
    num_sample = all_inputs.prepare(args.input_num)

    learn_scale = args.target == "Scale"
    learn_adaweight = args.target == "AdaWeight"
    learn_gptweight = args.target == "GptWeight"

    print(f'Learning Scale: {learn_scale}; Learning AdaWeight: {learn_adaweight}; Learning GptWeight: {learn_gptweight}')
    if learn_scale:
        scale_searcher.num_sample = num_sample
        scale_searcher.ref_tensors = ref_tensors(scale_searcher, all_inputs)
        scheduler = LrScheduler(args.lr, scale_searcher.num_sample, args.lr_scheduler)
        if args.opt == 'SGD':
            scale_searcher.opt = scale_searcher.SgdScaleOpt(scheduler, args.momentum, args.nesterov, args.weight_decay)
        else:
            scale_searcher.opt = scale_searcher.AdamScaleOpt(scheduler, 0.9, 0.999, args.weight_decay)
        scale_searcher.learning()
        cali_table.update(scale_searcher.new_scales)
        cali_table.write()
        del scale_searcher.ref_tensors
        del scale_searcher
    if learn_adaweight:
        scheduler = LrScheduler(args.lr, num_sample*args.epoch, args.lr_scheduler)
        weight_searcher = LearningAdaWeight(args)
        weight_searcher.scales = cali_table.table
        weight_searcher.num_sample = num_sample
        weight_searcher.ref_tensors = ref_tensors(weight_searcher, all_inputs)

        weight_searcher.opt = weight_searcher.SgdWeightOpt(scheduler, args.momentum, args.nesterov, args.weight_decay, args.epoch)
        weight_searcher.learning()
        del weight_searcher.ref_tensors
        del weight_searcher
    if learn_gptweight:
        weight_searcher = LearningGptqWeight(args)
        weight_searcher.scales = cali_table.table
        weight_searcher.num_sample = num_sample
        weight_searcher.ref_tensors = ref_tensors(weight_searcher, all_inputs)
        weight_searcher.learning()
        del weight_searcher

    loger.end()

