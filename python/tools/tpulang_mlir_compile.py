#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import json
import numpy as np
import os, sys
import transform.TpuLang as tpul
from typing import List
import math
import argparse
import time


def main():
    parser = argparse.ArgumentParser(description='TpuLang MLIR compile Script')
    parser.add_argument('-p',
                        '--model_path',
                        type=str,
                        default="./tmp_model_origin.mlir",
                        help='Path to the model file')
    parser.add_argument('-c',
                        '--chip',
                        type=str,
                        default="bm1688",
                        choices=["bm1688", "bm1684x"],
                        help='bm1688 or bm1684x')
    parser.add_argument(
        '-t',
        '--tag',
        type=str,
        default="",
        help="compile tag, useful for compiling same mlir with different tpu-mlir version")
    parser.add_argument('-m',
                        '--mode',
                        type=str,
                        default="int8",
                        choices=["f32", "f16", "int8"],
                        help="compile mode")
    parser.add_argument('-l',
                        '--layer_group_config',
                        type=str,
                        default="./",
                        help="layer_group_config_path")
    parser.add_argument('-d',
                        '--dynamic',
                        action='store_true',
                        default=False,
                        help='set if the model is dynamic, default is false')
    parser.add_argument('-n',
                        '--num-core',
                        type=int,
                        choices=[1, 2],
                        default=1,
                        help='core num for currnet model, default is 1')
    args = parser.parse_args()

    print(f"Model path: {args.model_path}")
    start_time = time.perf_counter()
    tpul.init(args.chip)
    if args.tag:
        name = f"tmp_model_{args.tag}_{args.chip}"
    else:
        name = "tmp_model_{args.chip}"
    if args.mode == "f32":
        tpul.mlir_compile_f32(name,
                              args.model_path,
                              layer_group_config=args.layer_group_config,
                              dynamic=args.dynamic,
                              num_core=args.num_core)
    elif args.mode == "f16":
        tpul.mlir_compile_f32(name,
                              args.model_path,
                              mode='f16',
                              layer_group_config=args.layer_group_config,
                              dynamic=args.dynamic,
                              num_core=args.num_core)
    elif args.mode == "int8":
        tpul.mlir_compile(name,
                          args.model_path,
                          layer_group_config=args.layer_group_config,
                          dynamic=args.dynamic,
                          num_core=args.num_core)
    tpul.deinit()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Model path: {args.model_path}")
    print(f"chip: {args.chip}")
    print(f"tag: {args.tag}")
    print(f"mode: {args.mode}")
    print(f"layer_group_config: {args.layer_group_config}")
    print(f"dynamic: {args.dynamic}")
    print(f"num_core: {args.num_core}")
    print(f" Processing completed!")
    print(f" Total time: {elapsed_time:.4f} seconds")


if __name__ == "__main__":
    main()
