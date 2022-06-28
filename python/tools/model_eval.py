#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
#
# Licensed under the Apache License v2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ==============================================================================
# use pytorch for dataloader
# https://github.com/pytorch/examples/blob/master/imagenet/main.py

import argparse
from eval.model_inference import *
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from calibration.data_selector import DataSelector

parser = argparse.ArgumentParser(
    description="Evaluation on ImageNet Dataset.")
parser.add_argument("--mlir_file", type=str, required=True, help="mlir file.")
parser.add_argument("--label_file", type=str, default = '', help="label file.")
parser.add_argument("--data_list", type=str, default = '', help="image list file.")
parser.add_argument("--dataset", type=str, help="The root directory of dataset.")
parser.add_argument("--dataset_type", type=str, default = 'imagenet', choices=['imagenet', 'coco', 'voc', 'user_define'],
                    help="The root directory of dataset.")
parser.add_argument("--postprocess_type", type=str, default = 'topx',
                    help="the postprocess type.")
parser.add_argument("--count", type=int, default=50000)
args = parser.parse_args()

class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        return self.imgs[index][0], super(MyImageFolder, self).__getitem__(index)[1]

if __name__ == '__main__':
  engine = mlir_inference(args)
  if args.dataset_type == 'imagenet':
    val_loader = torch.utils.data.DataLoader(
        MyImageFolder(args.dataset),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True, sampler=None)

    for i, item in enumerate(val_loader):
        path, target = item
        engine.run(i, path[0], int(target[0]))
        if (i + 1) >= args.count:
          break
  elif args.dataset_type == 'user_define':
    selector = DataSelector(args.dataset, args.count, args.data_list)
    for i, img in enumerate(selector.data_list):
        engine.run(i, img)
        if (i + 1) >= args.count:
          break
  engine.get_result()
