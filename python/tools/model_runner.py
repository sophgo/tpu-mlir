#!/usr/bin/env python3
import numpy as np
import argparse
import pymlir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input npz file")
    parser.add_argument("--model", required=True, help="mlir file.")
    parser.add_argument("--output", default='_output.npz', help="output npz file")
    args = parser.parse_args()
    module = pymlir.module()
    module.load(args.model)
    inputs = module.get_input_names()
    data = np.load(args.input)
    assert(len(inputs) == len(data))
    for input in inputs:
       assert(input in data)
       module.set_tensor(input, data[input])
    module.invoke()
    output = module.get_all_tensor()
    np.savez(args.output, **output)
