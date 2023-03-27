#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import random
import pathlib

# input may be these cases
# case 0: one input, and jpg
# a.jpg
# b.jpg
# case 1: one input, and npy
# a.npy
# b.npy
# case 2: one or more inputs, and npz
# a.npz
# b.npz
# case 3: more inputs, and npy
# a.npy,b.npy
# c.npy,d.npy
# case 4: more inputs, and jpg
# a.jpg,b.jpg
# c.jpg,d.jpg


class DataSelector:

    def __init__(self, dataset: str, num: int = 0, data_list_file: str = None):
        self.data_list = []
        if data_list_file:
            with open(data_list_file, 'r') as f:
                for line in f.readlines():
                    self.data_list.append(line.strip())
            if num > 0 and num < len(self.data_list):
                self.data_list = self.data_list[:num]
        elif dataset:
            self.data_list = self._random_select(dataset, num)
        else:
            raise RuntimeError("Please specific dataset path by --dataset")
        if len(self.data_list) == 0:
            raise RuntimeError("There is no inputs")
        self.all_npz, self.all_npy, self.all_image = False, False, False
        self._check_data_list()

    def _check_data_list(self):
        for file in self.data_list:
            if self.is_npz(file):
                self.all_npz = True
            else:
                inputs = file.split(',')
                inputs = [s.strip() for s in inputs]
                for i in inputs:
                    if self.is_npy(i):
                        self.all_npy = True
                    elif self.is_image(i):
                        self.all_image = True
                    else:
                        raise RuntimeError("File illegal:{}".format(file))
            num_type = self.all_npz + self.all_image + self.all_npy
            if num_type != 1:
                raise RuntimeError("Only support one input type: npy/npz/image")

    def _random_select(self, dataset_path, num):
        full_list = []
        for file in pathlib.Path(dataset_path).glob('**/*'):
            name = str(file)
            if self.is_npz(name) or self.is_npy(name) or self.is_image(name):
                full_list.append(name)
        full_list = sorted(full_list)
        random.seed(1684)
        random.shuffle(full_list)
        num = num if len(full_list) > num else len(full_list)
        if num == 0:
            num = len(full_list)
        return full_list[:num]

    def is_npz(self, filename: str):
        return True if filename.lower().split('.')[-1] == 'npz' else False

    def is_npy(self, filename: str):
        return True if filename.lower().split('.')[-1] == 'npy' else False

    def is_image(self, filename: str):
        type = filename.lower().split('.')[-1]
        return type in ['jpg', 'bmp', 'png', 'jpeg', 'jfif']

    def _print(self):
        for i, img in enumerate(self.image_list):
            print(" <{}> {}".format(i, img))

    def dump(self, file):
        with open(file, 'w') as f:
            for input in self.data_list:
                f.write(input + '\n')
