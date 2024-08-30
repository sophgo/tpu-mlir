#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import os
from ctypes import *

cur_dir_path = os.path.join(os.path.dirname(__file__))
bmodel_combine_path = os.path.join("/".join(cur_dir_path.split("/")[:-2]), "lib/libmodel_combine.so")
if not os.path.exists(bmodel_combine_path):
    bmodel_combine_path = "libmodel_combine.so"

class BmodelCombine:

    def __init__(self, lib_path=bmodel_combine_path):
        self.lib = CDLL(lib_path)
        self.lib.tpu_model_create.restype = c_void_p
        self.lib.tpu_model_add.argtypes = [c_void_p, c_void_p, c_uint64]
        self.lib.tpu_model_combine.argtypes = [c_void_p]
        self.lib.tpu_model_combine.restype = c_uint64
        self.lib.tpu_model_save.argtypes = [c_void_p, c_void_p]
        self.lib.tpu_model_destroy.argtypes = [c_void_p]

    def combine_bmodel(self, inputs: list, sizes):
        bmodel = self.lib.tpu_model_create()
        for i in range(len(inputs)):
            input = inputs[i]
            size = sizes[i]
            self.lib.tpu_model_add(bmodel, input, size)
        size = self.lib.tpu_model_combine(bmodel)
        output = (c_int8 * size)()
        self.lib.tpu_model_save(bmodel, output)
        self.lib.tpu_model_destroy(bmodel)
        return output

def combine_bmodel(out_dir, dirs: list[str], is_dir=False):
    cmd_str = "model_tool --combine_coeff"
    if is_dir:
        cmd_str += "_dir"
    for d in dirs:
        cmd_str += " " + d
    if out_dir != None:
        cmd_str += " -o " + out_dir
    print(cmd_str)
    import subprocess
    subprocess.call(cmd_str, shell=True)

# combine bmodel to one, coeff will combine
# combine bmodels to one bmodel, all models' coeff is same
# mode = 0: input bmodel
# mode = 1: input bmodel path
# mode = 2: input bmodel data, size
def combine(inputs: list, sizes: list[int] = None, output = None, mode = 0):
    if mode == 0:
        assert isinstance(inputs[0], str)
        assert inputs[0].split('.')[-1] == "bmodel"
        combine_bmodel(output, inputs)
    elif mode == 1:
        assert isinstance(inputs[0], str)
        combine_bmodel(output, inputs, True)
    elif mode == 2:
        assert len(sizes) != 0
        assert isinstance(sizes[0], int)
        combine = BmodelCombine()
        output_data = combine.combine_bmodel(inputs, sizes)
        return output_data
    else:
        print("mode: {} do not support".format(mode))
        assert(0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--output", default="bm_combined", type=str, help="output bmodel name")
    parser.add_argument("--inputs", default="", type=str, help="input bmodel names")
    parser.add_argument("--model_data", default=False, type=bool, help="inputs is model data")
   # yapf: enable
    args = parser.parse_args()
    if args.model_data:
        inputs = [inp for inp in args.inputs]
        input_datas = []
        sizes = []
        for inp in args.inputs.split(' '):
            file = open(inp, 'rb')
            data = file.read()
            input_datas.append(data)
            sizes.append(len(data))
            file.close()
        output_data = combine(input_datas, sizes, mode=2)
        file = open(args.output, 'wb')
        file.write(output_data)
        file.close()
    else:
        inputs = args.inputs.split(" ")
        combine(inputs, output=args.output)
