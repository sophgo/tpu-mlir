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
    parser.add_argument('--debug', action='store_true',
                        help='enable debug mode, temp files will not be deleted')
    parser.add_argument('-o', '--out_dir', type=str, default='./tmp',
                        help='output mlir/bmodel path, default `./tmp`')
    args = parser.parse_args()
    # yapf: enable
    from transform.LlmConverter import LlmConverter
    converter = LlmConverter(args)
    converter.run()
