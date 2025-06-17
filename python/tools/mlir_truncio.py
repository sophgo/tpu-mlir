#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================


def mlir_truncio(compare_result_file: str, input_mlir_file: str, output_mlir_file: str):
    import pandas as pd
    df = pd.read_csv(compare_result_file, sep=', ', header=0, engine='python')
    names = df['name']
    passes = df['pass']
    inputs = list(names[passes == True])
    outputs = list(names[passes == False])
    if len(inputs) == len(passes):
        print("[WARNING] no error")
        return
    cmd = f"tpuc-tool {input_mlir_file} --trunc-io='inputs="
    for i in range(len(inputs)):
        cmd += inputs[i]
        if i != len(inputs) - 1:
            cmd += ","
    cmd += " outputs="
    for i in range(len(outputs)):
        cmd += outputs[i]
        if i != len(outputs) - 1:
            cmd += ","
    cmd += f"' -o {output_mlir_file}"
    import os
    os.system(cmd)


if __name__ == '__main__':
    # yapf: disable
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp_file", "-c", required=True, help="compare result file")
    parser.add_argument("--input_mlir_file", "-i", type=str, required=True, help="input mlir file")
    parser.add_argument("--output_mlir_file", "-o", type=str, required=True, help="output mlir file")
    # yapf: enable
    args = parser.parse_args()
    mlir_truncio(args.comp_file, args.input_mlir_file, args.output_mlir_file)
