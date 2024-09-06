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
from utils.mlir_parser import *
from PIL import Image

TypeMap = {
    'f32': np.float32,
    'si32': np.int32,
    'f16': np.float16,
}


def create_and_save_tensors_to_npz(tensor_names, shapes, data_types, ranges, output_file):
    tensors = []
    for shape, data_type, range_ids in zip(shapes, data_types, ranges):
        low, high = range_ids
        if data_type == 'si32':
            tensor = np.random.randint(low, high + 1, size=shape, dtype=TypeMap[data_type])
        else:
            # tensor = np.random.rand(*shape).astype(TypeMap[data_type])
            tensor = np.random.uniform(low, high + 1, shape).astype(TypeMap[data_type])
        tensors.append(tensor)

    tensors_dict = {name: tensor for name, tensor in zip(tensor_names, tensors)}
    np.savez(output_file, **tensors_dict)
    print(f"Tensors saved to {output_file}.")


def generate_random_img(shapes, output_file):
    n, c, h, w = shapes[0]
    fake_image = np.random.randint(0, 256, (h, w, c), dtype=np.uint8)
    random_image = Image.fromarray(fake_image)
    random_image.save(output_file)
    print(f"Image saved to {output_file}.")


def run_from_mlir(args):
    module_parsered = MlirParser(args.mlir)
    mlir_shapes = module_parsered.get_input_shapes()
    mlir_types = args.input_types if args.input_types else module_parsered.get_input_types()
    input_num = module_parsered.get_input_num()
    tensor_names = []
    for i in range(input_num):
        tensor_names.append(module_parsered.inputs[i].name)
    if args.img:
        generate_random_img(mlir_shapes, args.output)
        return
    ranges = args.ranges
    if len(ranges) == 0:
        for i in range(input_num):
            ranges.append([-1.0, 1.0])

    create_and_save_tensors_to_npz(tensor_names, mlir_shapes, mlir_types, ranges, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlir", type=str, required=True, help="output mlir model file")
    parser.add_argument("--ranges",
                        type=str2shape,
                        default=list(),
                        help="list of input ranges, like:[[0,1],[-10,10]]")
    parser.add_argument(
        "--input_types",
        type=str2list,
        default=list(),
        help="list of input types, like:f32,si32. if not set, it will be read from mlir")
    parser.add_argument("--img", action='store_true', help="generate fake image for CV tasks")
    parser.add_argument("--output", type=str, default='input.npz', help="output npz/img file")
    arg = parser.parse_args()
    run_from_mlir(arg)
