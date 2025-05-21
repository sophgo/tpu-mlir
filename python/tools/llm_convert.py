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


def parse_max_pixels(value):
    """
    If the input is a single number, convert it to an integer.
    If it contains a comma, parse it as a tuple (or list) of two integers, e.g., "128,124".
    """
    if ',' in value:
        parts = value.split(',')
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "The input must be two integers separated by a comma, e.g., 128,124")
        try:
            width = int(parts[0].strip())
            height = int(parts[1].strip())
        except ValueError:
            raise argparse.ArgumentTypeError("The input values must be integers, e.g., 128,124")
        return int(width * height)
    else:
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "The input must be an integer or two integers separated by a comma, e.g., 128,124")


if __name__ == '__main__':
    # yapf: disable
    parser = argparse.ArgumentParser(description='llm_exporter', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='original weight, like ./Qwen2-7B-Instruct')
    parser.add_argument('-s', '--seq_length', type=int, required=True,
                        help="sequence length")
    parser.add_argument('-q', '--quantize', type=str, required=True,
                        choices=["bf16", "w8bf16", "w4bf16", "f16", "w8f16", "w4f16"],
                        help="quantize type for bmodel")
    parser.add_argument('-g', "--q_group_size", default=64, type=int,
                        help="group size for per-group quant, only used in W4A16 quant mode")
    parser.add_argument('-c', '--chip', type=str, default="bm1684x",
                        choices=["bm1684x", "bm1688", "cv186ah"],
                        help="chip type for bmodel")
    parser.add_argument('--num_device', type=int, default=1,
                        help="num device for bmodel")
    parser.add_argument('--num_core', type=int, default=0, help = "num cores for bmodel")
    parser.add_argument('--symmetric', action='store_true', help='do symmetric quantize')
    parser.add_argument('--embedding_disk', action='store_true',
                        help='export embedding as bin file and inference by cpu')
    parser.add_argument('--penalty_sample', action='store_true',
                        help='Add penalty sample head and separate greedy head from lmhead')
    parser.add_argument('--max_pixels', type=parse_max_pixels, default=0,
                        help="max pixels for vit, for example: 240,420 or 100800")
    parser.add_argument('--dynamic', action='store_true',
                        help='enable dynamic compiling for prefill, not recommended')
    parser.add_argument('--debug', action='store_true',
                        help='enable debug mode, temp files will not be deleted')
    parser.add_argument('-o', '--out_dir', type=str, default='./tmp',
                        help='output mlir/bmodel path, default `./tmp`')
    args = parser.parse_args()
    # yapf: enable
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if config.model_type in ["qwen3", "qwen2", "llama"]:
        from llm.LlmConverter import LlmConverter
        converter = LlmConverter(args, config)
    elif config.model_type in ['qwen2_vl', 'qwen2_5_vl']:
        from llm.Qwen2VLConverter import Qwen2VLConverter
        converter = Qwen2VLConverter(args, config)
    else:
        raise RuntimeError("Unsupported model type: {}".format(config.model_type))
    converter.run()
