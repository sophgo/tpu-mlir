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

import random
import pathlib

class DataSelector:
    def __init__(self, dataset:str, num:int=0, data_list_file:str=None):
        self.data_list = []
        if data_list_file:
            with open(data_list_file, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if len(line) != 0:
                        self.data_list.append(line.strip())
        elif dataset:
            for file in pathlib.Path(dataset).glob('**/*'):
                if file.is_file() and self._is_cali_file(file.name):
                    self.data_list.append(str(file))
        else:
            raise RuntimeError("Please specific dataset path by --dataset")
        if len(self.data_list) == 0:
            raise RuntimeError("There is no inputs")
        if num == 0 or num >= len(self.data_list):
            return
        random.shuffle(self.data_list)
        self.data_list = self.data_list[:num]


    @staticmethod
    def _is_cali_file(filename:str):
        support_list={'.jpg','.bmp','.png','.jpeg','.jfif','.npy','.npz'}
        for type in support_list:
            if filename.endswith(type):
                return True
        return False

    def _print(self):
        for i, img in enumerate(self.image_list):
            print(" <{}> {}".format(i, img))

    def dump(self, file):
        with open(file, 'w') as f:
            for input in self.data_list:
                f.write(input + '\n')
