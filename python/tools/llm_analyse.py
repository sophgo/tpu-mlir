#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import argparse
from llm.LlmProfile import LlmProfiler

if __name__ == '__main__':
    # yapf: disable
    parser = argparse.ArgumentParser(description='llm_analyse')
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='original weight, like ./Qwen2-7B-Instruct')
    parser.add_argument('-s', '--seq_length', type=int, required=True,
                        help="sequence length")
    parser.add_argument('-q', '--quantize', type=str, required=True,
                        choices=["bf16", "w8bf16", "w4bf16", "f16", "w8f16", "w4f16"],
                        help="quantize type for bmodel")
    parser.add_argument('-g', "--group_size", default=64, type=int,
                        help="group size for per-group quant, only used in W4A16 quant mode")
    parser.add_argument('-c', '--chip', type=str, default="bm1684x",
                        choices=["bm1684x", "bm1688", "cv186x"],
                        help="chip type for bmodel")
    parser.add_argument('--tpu_freq', type=int,
                        help="tpu frequency")
    parser.add_argument('--num_device', type=int, default=1,
                        help="num device for bmodel")
    parser.add_argument('--num_core', type=int, default=1,
                        help="num cores for bmodel")
    parser.add_argument('--quant_lmhead', action='store_true',
                        help="quantize lmhead based on quant type")
    args = parser.parse_args()
    # yapf: enable

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    profiler = LlmProfiler(args, config)
    profiler.analyze()
