# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import math
import torch
import os
import sys
import numpy as np
from scipy import spatial
import datetime


from utils.preprocess import preprocess
from calibration.data_selector import DataSelector

from utils.mlir_shell import mlir_lowering
import pymlir
pymlir.set_mem_mode("force_value_mem")


def quant_requant_active(data, scale, unsigned=False, bits=8):
    if unsigned:
        d = data/scale*(2**bits-1)
        dout = np.round(d)
        return np.clip(dout, 0, 2**bits-1)/(2**bits-1) * scale
    else:
        d = data/scale*(2**(bits-1)-1)
        dout = np.round(d)
        return np.clip(dout, -(2**(bits-1)), 2**(bits-1)-1)/(2**(bits-1)-1) * scale


def cal_loss(target, ref):
    mse_diff = ((target - ref)**2).mean()
    return mse_diff


def cosine_sim(x, y):
    x[np.isnan(x)] = 0.0
    y[np.isnan(y)] = 0.0
    cosine_similarity = 1 - spatial.distance.cosine(x.flatten().astype(np.float32),
                                                    y.flatten().astype(np.float32))
    return cosine_similarity


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


def get_fixed_float_layers(mlir, mode, chip, cali_table, q_table, ref_tensor):
    quanted_mlir_file = '{}_int8.quant_test.mlir'.format(mlir)
    float_tensors = []
    mlir_lowering(mlir, quanted_mlir_file, mode, chip, 1, 1, cali_table, False, q_table)
    module = pymlir.module()
    module.load(quanted_mlir_file)
    for input_name in module.input_names:
        data = ref_tensor.get(input_name, 0)
        module.set_tensor(input_name, data)
    module.invoke()
    # loop over op to check if its output tensor is int8
    for op in module.all_tensor_names:
        qinfo = module.get_tensor_qinfo(op)
        if qinfo.dtype != "I8" and qinfo.dtype != "U8":
            float_tensors.append(op)
    return float_tensors


def learning_lower(mlir, mode, chip, cali_table, q_table):
    quanted_mlir_file = '{}_int8.quant_eval.mlir'.format(mlir)
    mlir_lowering(mlir, quanted_mlir_file, mode, chip, 1, 1, cali_table, False, q_table)
    module = pymlir.module()
    module.load(quanted_mlir_file)
    return module


def lower_and_eval(mlir, mode, chip, cali_table, q_table, ref_tensor, sample_idxs):
    module = learning_lower(mlir, mode, chip, cali_table, q_table)
    outputs = []
    for s in sample_idxs:
        for input_name in module.input_names:
            data = ref_tensor.get(input_name, s)
            module.set_tensor(input_name, data)
        module.invoke()
        output = module.get_tensor(module.output_names[0]).copy()
        outputs.append(output)
    del module
    return outputs


class logging:
    def __init__(self, filename="logging"):
        self.file_name = filename
        self.log_file = open(self.file_name, 'w')

    def logging(self, info):
        print(info, file=self.log_file)

    def end(self):
        if self.log_file is not None:
            self.log_file.close()


class imagenet_dataset:
    def __init__(self, path, input_size, mean, scale):
        self.path = path
        self.input_size = input_size
        self.mean = mean
        self.scale = scale
        self.format = 'bgr'
        self.shuffle = True
        self.val_batchsize = 1
        self.num_workers = 4

    def build_dataset(self):
        mean = tuple([m/255.0 for m in self.mean])
        std = tuple([(1.0/x)/255.0 for x in self.scale])
        crop_pct = 0.875
        train_transform = self.build_transform(input_size=self.input_size, mean=mean, std=std, crop_pct=crop_pct)
        val_transform = self.build_transform(input_size=self.input_size, mean=mean, std=std, crop_pct=crop_pct)

        # Data
        traindir = os.path.join(self.path, 'train')
        valdir = os.path.join(self.path, 'val')

        val_dataset = datasets.ImageFolder(valdir, val_transform)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.val_batchsize,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        '''
        train_dataset = datasets.ImageFolder(traindir, train_transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.val_batchsize,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        '''
        return val_loader

    def build_transform(self, input_size=224, interpolation="bicubic",
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                        crop_pct=0.875):
        def _pil_interp(method):
            if method == "bicubic":
                return Image.BICUBIC
            elif method == "lanczos":
                return Image.LANCZOS
            elif method == "hamming":
                return Image.HAMMING
            else:
                return Image.BILINEAR
        resize_im = input_size > 32
        t = []
        if resize_im:
            size = int(math.floor(input_size / crop_pct))
            ip = _pil_interp(interpolation)
            t.append(
                transforms.Resize(
                    size, interpolation=ip
                ),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(input_size))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)


class learning_inputs:
    def __init__(self, parser, args):
        self.dataset = args.dataset
        self.data_list = args.data_list
        self.imagenet = args.imagenet
        self.batch_size = parser.get_batch_size()
        self.input_num = parser.get_input_num()
        self.num_sample = 0
        self.parser = parser
        self.ref_activations = {}
        self.mean = []
        self.scale = []
        self.resize_dims = []
        self.pixel_format = 'rgb'

    def prepare(self, input_num):
        tune_idx = 0
        self.ref_activations[tune_idx] = {}
        input_names = [op.name for op in self.parser.inputs]
        if self.imagenet:
            self.num_sample = input_num
            if self.input_num > 1 or self.batch_size > 1:
                print('support one image input net only!')
                sys.exit(1)
            for i in range(self.input_num):
                ppa = preprocess()
                ppa.load_config(self.parser.get_input_op_by_idx(i))
                self.mean = [x for x in ppa.mean.reshape(-1)]
                self.scale = [x for x in ppa.scale.reshape(-1)]
                self.resize_dims = ppa.net_input_dims
                self.pixel_format = ppa.pixel_format
            imgnet_ds = imagenet_dataset(self.dataset, self.resize_dims[0], self.mean, self.scale)  # ugly input dim 0
            val_loader = imgnet_ds.build_dataset()
            cnt = 0
            batched_idx = 0
            for data, target in val_loader:
                inputs = [s.strip() for s in input_names]
                assert (len(inputs) == 1)
                batched_idx += 1
                batched_inputs = self.input_num * ['']
                for i, input in enumerate(input_names):
                    batched_inputs[i] += '{},'.format(inputs[i])
                    if batched_idx == self.batch_size:
                        x = data
                        count = self.parser.get_user_count_by_op_name(input)
                        self.ref_activations[tune_idx][input] = [x, count]
                if batched_idx == self.batch_size:
                    tune_idx += 1
                    batched_idx = 0
                    batched_inputs = self.input_num * ['']
                    self.ref_activations[tune_idx] = {}
                cnt += 1
                if cnt >= self.num_sample:
                    break
        else:
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
    def __init__(self, learner, inputs, loger=None):
        self.parser = learner.parser
        self.module = learner.module
        self.net_inputs = inputs
        self.epoch_samples = inputs.num_sample
        self.ops = {}
        self.ops_cnt = {}
        self.ops_buffer = {}
        self.loger = loger
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
        # if already buffered return the buffer
        if op in self.ops_buffer:
            self.loger.logging(f'return from ops_buffer {op}')
            return self.ops_buffer[op][idx]
        elif op in self.net_inputs.ref_activations[0]:
            self.loger.logging(f'return from ref_activation {op}')
            return self.net_inputs.ref_activations[idx][op][0]

        # if not bufferred, generate all samples
        inputs = self.parser.get_pre_op_by_op_name(op)
        for in_ in inputs:
            if in_ not in self.ops_buffer and in_ not in self.net_inputs.ref_activations[0]:
                self.loger.logging(f'recursive get {in_}')
                self.get(in_, idx, quant, symetric)
                self.ops_cnt[in_] = self.ops[in_]
        for l in range(self.epoch_samples):
            for in_ in inputs:
                if in_ in self.ops_buffer:
                    self.module.set_tensor(in_, self.ops_buffer[in_][l])
                elif in_ in self.net_inputs.ref_activations[0]:
                    self.module.set_tensor(in_, self.net_inputs.ref_activations[l][in_][0])
                else:
                    import os
                    print("error, no input found in buffer and input")
                    os.exit(1)
            outputs = self.module.invoke_at(op)
            for out_ in self.parser.get_outputs_by_op_name(op):
                if out_ in self.ops and self.ops[out_] > 0:
                    if out_ in self.ops_buffer:
                        self.ops_buffer[out_].append(self.module.get_tensor(out_).copy())
                    else:
                        self.ops_buffer[out_] = []
                        self.ops_buffer[out_].append(self.module.get_tensor(out_).copy())
                if l == 0:
                    self.loger.logging(f'adding {out_} cnt {self.ops[out_]}')
        for out_ in self.parser.get_outputs_by_op_name(op):
            if out_ in self.ops and self.ops[out_] > 0:
                self.loger.logging(f'setting {out_} {self.ops[out_]}')
                self.ops_cnt[out_] = self.ops[out_]
        for in_ in inputs:
            if in_ in self.ops_buffer:
                self.ops_cnt[in_] -= 1
                if self.ops_cnt[in_] == 0:
                    del self.ops_buffer[in_]
                    self.loger.logging(f'del in {in_}')
        if op not in self.ops_buffer:
            print(f'{op} not in ref tensors!')
            sys.exit(1)
        return self.ops_buffer[op][idx]

    def consumed_tensor(self, op):  # must call when loop over epoch of using the tensor is done
        if op in self.net_inputs.ref_activations[0]:
            return
        if op not in self.ops_buffer:
            print(f"{op} not in buffer when mark used!")
        else:
            self.loger.logging(f'used, del 1 {op}')
            self.ops_cnt[op] -= 1
            if self.ops_cnt[op] == 0:
                self.loger.logging(f'end used, del {op}')
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
                return self.min_lr + 0.5 * (self.lr-self.min_lr)*(1.0+np.cos((iter-self.max_iter*self.warm_up)/(self.max_iter-self.max_iter*self.warm_up)*np.pi))
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
        self.table4 = {}
        self.read()

    def read(self):
        int4 = False
        textfile = open(self.in_table, 'r')
        for line in textfile:
            if len(line) > 0 and (not line.startswith("#")):
                s = line.split(" ")
                s = [x for x in s if x != '']
                if len(s) != 4:
                    continue
                if int4:
                    self.table4[s[0]] = [
                        float(s[1]), float(s[2]), float(s[3])]
                else:
                    self.table[s[0]] = [
                        float(s[1]), float(s[2]), float(s[3])]
            if line.startswith("#int4_th"):
                int4 = True

    def update(self, new_table, int4=False):
        for op in new_table:
            if int4:
                for op_ in self.table4:
                    if op_ == op:
                        self.table4[op][0] = new_table[op][0]
            else:
                for op_ in self.table:
                    if op_ == op:
                        self.table[op][0] = new_table[op][0]

    def write(self):
        f = open(self.out_table, 'w')
        f.write("# Learning genetated time: {}\n".format(datetime.datetime.now()))
        f.write("# op_name    threshold    min    max\n")
        for op in self.table:
            f.write(
                f'{op}  {self.table[op][0]}  {self.table[op][1]}  {self.table[op][2]}\n')
        if len(self.table4) > 0:
            f.write('\n')
            f.write('#int4_th\n')
            for op in self.table4:
                f.write(
                    f'{op}  {self.table4[op][0]}  {self.table4[op][1]}  {self.table4[op][2]}\n')
        f.close()
