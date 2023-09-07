#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import abc
import numpy as np
import argparse
from utils.misc import *

def create_and_save_tensors_to_npz(tensor_names, shapes, data_types):
    tensors = []
    for shape, data_type in zip(shapes, data_types):
        if data_type.startswith('int'):
            low, high = 1, 100
            tensor = np.random.randint(low, high + 1, size=shape, dtype=data_type)
        else:
            tensor = np.random.rand(*shape).astype(data_type)
        tensors.append(tensor)

    tensors_dict = {name: tensor for name, tensor in zip(tensor_names, tensors)}
    np.savez('input.npz', **tensors_dict)
    print("Tensors saved to 'input.npz'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_names", type=str2list, default=list(),
                        help="if set, will find names in model and set as real outputs")
    parser.add_argument("--input_shapes", type=str2shape, default=list(),
                        help="list of input shapes, like:[[1,3,224,224],[10],[16]]")
    parser.add_argument("--input_types", type=str2list, default=list(),
                        help="list of input types, like:float32,int32. if not set, float32 as default")
    arg = parser.parse_args()
    create_and_save_tensors_to_npz(arg.input_names, arg.input_shapes, arg.input_types)
