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
import pymlir


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
                        help="group size for per-group quant, only used in quant mode")
    parser.add_argument('-c', '--chip', type=str, default="bm1684x",
                        choices=["bm1684x", "bm1688", "cv186x"],
                        help="chip type for bmodel")
    parser.add_argument('--num_device', type=int, default=1,
                        help="num device for bmodel")
    parser.add_argument('--num_core', type=int, default=0, help="num cores for bmodel")
    parser.add_argument('--symmetric', action='store_true', help='do symmetric quantize')
    parser.add_argument('--embedding_disk', action='store_true',
                        help='export embedding as bin file and inference by cpu')
    parser.add_argument('--do_sample', action='store_true',
                        help='Add sample head and separate greedy head from lmhead')
    parser.add_argument('--use_block_with_kv', action='store_true',
                        help='use history kv for prefill, default is False')
    parser.add_argument('--max_input_length', type=int, default=0,
                        help='max input length for prefill, default 0 means the same as seq_length')
    parser.add_argument('--max_prefill_kv_length', type=int, default=0,
                        help='max prefill kv length, default 0 means the same as seq_length')
    parser.add_argument('--max_pixels', type=parse_max_pixels, default=0,
                        help="max pixels for vit, for example: 240,420 or 100800")
    parser.add_argument('--dynamic', action='store_true',
                        help='enable dynamic compiling for prefill, not recommended')
    parser.add_argument('--debug', action='store_true',
                        help='enable debug mode, temp files will not be deleted')
    parser.add_argument("-V", "--version", action='version', version='%(prog)s ' + pymlir.__version__)
    parser.add_argument('-o', '--out_dir', type=str, default='./tmp',
                        help='output mlir/bmodel path, default `./tmp`')
    args = parser.parse_args()
    # yapf: enable
    if args.use_block_with_kv:
        if args.max_input_length <= 0:
            args.max_input_length = args.seq_length // 4
            print("Warning: max_input_length is not set, use seq_length // 4 as default value: {}".
                  format(args.max_input_length))
        elif args.max_input_length > args.seq_length // 2:
            raise ValueError(
                "max_input_length should not be larger than seq_length // 2, got: {}".format(
                    args.max_input_length))
    if args.max_prefill_kv_length <= 0:
        args.max_prefill_kv_length = args.seq_length
    elif args.max_prefill_kv_length > args.seq_length:
        raise ValueError(
            "max_prefill_kv_length should not be larger than seq_length, got: {}".format(
                args.max_prefill_kv_length))

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if config.model_type in ["qwen3", "qwen2", "llama", "minicpm"]:
        from llm.LlmConverter import LlmConverter
        converter = LlmConverter(args, config)
    elif config.model_type in ["chatglm"]:
        from llm.Chatglm3Converter import Chatglm3Converter
        converter = Chatglm3Converter(args, config)
    elif config.model_type in ["phi3"]:
        from llm.Phi3Converter import Phi3Converter
        converter = Phi3Converter(args, config)
    elif config.model_type in ['qwen2_vl']:
        from llm.Qwen2VLConverter import Qwen2VLConverter
        converter = Qwen2VLConverter(args, config)
    elif config.model_type in ['qwen2_5_vl']:
        from llm.Qwen2_5VLConverter import Qwen2_5VLConverter
        converter = Qwen2_5VLConverter(args, config)
    elif config.model_type in ['internvl_chat']:
        from llm.InternVL3Converter import InternVL3Converter
        converter = InternVL3Converter(args, config)
    elif config.model_type in ['gemma3']:
        from llm.Gemma3Converter import Gemma3Converter
        converter = Gemma3Converter(args, config)
    else:
        raise RuntimeError("Unsupported model type: {}".format(config.model_type))
    converter.run()
