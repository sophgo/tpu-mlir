#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import argparse
import numpy as np
from utils.mlir_shell import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input npz file")
    parser.add_argument("--model", type=str, required=True,
                        help="CPU object file , .o")
    # TODO: Requiring users to specify this argument is quite unreasonable
    #parser.add_argument("--output_size", required=True, help="size of the output tensor")
    parser.add_argument("--output_shape", required=True, help="shape of the output tensor")
    args = parser.parse_args()

    # prepare input for c_interface
    data = np.load(args.input)
    f = open("data_for_capi.txt", "w")
    # TODO: Currently, only cases with one input of 4-D considered 
    data_input = data[data.files[0]]
    input_shape = data_input.shape
    print("Shape of input data:", input_shape)
    print("Generating data_for_capi.txt ...")
    f.write("%ld\n" %input_shape[0])
    f.write("%ld\n" %input_shape[1])
    f.write("%ld\n" %input_shape[2])
    f.write("%ld\n" %input_shape[3])
    data_size = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
    f.write("%ld\n" %data_size)
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            for k in range(input_shape[2]):
                for t in range(input_shape[3]):
                    f.write("%f\n" %data_input[i][j][k][t])
    f.close()
    print("Successfully generate data_for_capi.txt!")
    # inference_result.txt will be generated
    output_shape = args.output_shape.replace("x", "*")
    #output_size = 1
    #for i in output_shape:
    #    output_size = output_size * eval(i)
    model_inference_cpu(args.model, eval(output_shape))



    