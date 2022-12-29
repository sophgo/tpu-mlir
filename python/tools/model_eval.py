#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
# use pytorch for dataloader
# https://github.com/pytorch/examples/blob/master/imagenet/main.py

import argparse
from tqdm import tqdm
from eval.model_inference import *
from utils.misc import *
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from calibration.data_selector import DataSelector

parser = argparse.ArgumentParser(
    description="Evaluation on ImageNet Dataset.")
parser.add_argument("--model_file", type=str, required=True, help="model file.")
parser.add_argument("--label_file", type=str, default = '', help="label file.")
parser.add_argument("--data_list", type=str, default = '', help="image list file.")
parser.add_argument("--dataset", type=str, help="The root directory of dataset.")
parser.add_argument("--dataset_type", type=str, default = 'imagenet', choices=['imagenet', 'coco', 'voc', 'user_define'],
                    help="The root directory of dataset.")
parser.add_argument("--postprocess_type", type=str, required=True,
                    help="the postprocess type.")
parser.add_argument("--count", type=int, default=0)
parser.add_argument('--debug_cmd', type=str, default='', help='debug cmd')
args,_ = parser.parse_known_args()

class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        return self.imgs[index][0], super(MyImageFolder, self).__getitem__(index)[1]

if __name__ == '__main__':
  if not os.path.exists(args.dataset):
      raise ValueError ("Dataset path doesn't exist.")
  engine = model_inference(parser)
  if args.dataset_type == 'imagenet':
    val_loader = torch.utils.data.DataLoader(
        MyImageFolder(args.dataset),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True, sampler=None)
    count = args.count if args.count > 0 else len(val_loader)
    for i, item in enumerate(val_loader):
        path, target = item
        engine.run(i, path[0], int(target[0]))
        if (i + 1) >= count:
          break
  elif args.dataset_type == 'coco':
    image_list = get_image_list(args.dataset, args.count)
    with tqdm(total = len(image_list)) as pbar:
        for i, image_path in enumerate(image_list):
            engine.run(i, image_path)
            pbar.update(1)
  elif args.dataset_type == 'user_define':
    selector = DataSelector(args.dataset, args.count, args.data_list)
    for i, img in enumerate(selector.data_list):
        engine.run(i, img)
        if (i + 1) >= args.count:
          break
  engine.get_result()
