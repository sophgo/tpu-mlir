#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import numpy as np
import time
import argparse
import pymlir
pymlir.set_mem_mode("value_mem")
import onnx
import onnxruntime
from utils.mlir_shell import *
from utils.mlir_parser import *
from tools.model_runner import mlir_inference, model_inference
from utils.preprocess import preprocess
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from enum import Enum

def mlir_validate(val_loader, module, ppa_list, count=-1):
    """https://github.com/pytorch/examples/blob/main/imagenet/main.py"""

    criterion = nn.CrossEntropyLoss()

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [top1, top5, losses, batch_time],
        prefix='Test: ')

    end = time.time()
    for i, item in enumerate(val_loader):
        (images,target), (path,_) = item

        assert(input_num == 1)
        x = ppa_list[0].run(path[0])
        module.set_tensor(ppa_list[0].input_name, x)
        module.invoke()
        tensors = module.get_all_tensor()
        assert(len(module.output_names) == 1)
        output = torch.from_numpy(tensors[module.output_names[0]])

        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            progress.display(i + 1)

        if i == count:
            break


class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(MyImageFolder, self).__getitem__(index), self.imgs[index]


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        total = torch.FloatTensor([self.sum, self.count])
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="mlir file.")
    parser.add_argument("--dataset", required=True, help="imagenet dataset")
    parser.add_argument('--count', type=int, required=False, default=-1,
                        help='num of images for eval')
    args = parser.parse_args()

    val_dataset = MyImageFolder(
        args.dataset,
        transforms.Compose([
            transforms.PILToTensor(),
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=True,
        num_workers=1, pin_memory=True, sampler=None)

    if args.model.endswith('.onnx'):
        raise RuntimeError("ONNX not supported yet, modle file:{}".format(args.model))
    elif args.model.endswith('.mlir'):
        print("Running eval on imagenet with Modle file:{}".format(args.model))

        # construct ppa_list from mlir
        ppa_list = []
        module_parsered = MlirParser(args.model)
        input_num = module_parsered.get_input_num()
        for i in range(input_num):
            tmp = preprocess()
            tmp.load_config(module_parsered.get_input_op_by_idx(i))
            ppa_list.append(tmp)
        print(ppa_list)
        print(ppa_list[0].input_name)

        # validate
        module = pymlir.module()
        module.load(args.model)

        mlir_validate(val_loader, module, ppa_list, args.count)

    elif args.model.endswith(".tflite"):
        raise RuntimeError("TFLite not supported yet, modle file:{}".format(args.model))
    elif args.model.endswith(".bmodel") or args.model.endswith(".cvimodel"):
        raise RuntimeError("bmodel not supported yet, modle file:{}".format(args.model))
    else:
        raise RuntimeError("not support modle file:{}".format(args.model))
