#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def dump(file_path):
    with safe_open(file_path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            print(f"{key}: shape={list(tensor.shape)}, dtype={tensor.dtype}")


def convert(file_path, output_path, target_dtype):
    tensors = {}
    with safe_open(file_path, framework="pt") as f:
        metadata = f.metadata()
        for key in f.keys():
            tensor = f.get_tensor(key)
            if tensor.dtype == torch.float32:
                tensor = tensor.to(target_dtype)
            tensors[key] = tensor
    save_file(tensors, output_path, metadata=metadata)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Safetensors tool")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dump", metavar="FILE", help="Dump tensor info from a safetensors file")
    group.add_argument("--f32_to_f16", metavar="FILE", help="Convert fp32 tensors to fp16")
    group.add_argument("--f32_to_bf16", metavar="FILE", help="Convert fp32 tensors to bf16")
    parser.add_argument("-o",
                        "--output",
                        metavar="FILE",
                        help="Output file path (required for conversion)")
    args = parser.parse_args()

    if args.dump:
        dump(args.dump)
    elif args.f32_to_f16:
        if not args.output:
            parser.error("--f32_to_f16 requires -o/--output")
        convert(args.f32_to_f16, args.output, torch.float16)
    elif args.f32_to_bf16:
        if not args.output:
            parser.error("--f32_to_bf16 requires -o/--output")
        convert(args.f32_to_bf16, args.output, torch.bfloat16)


if __name__ == "__main__":
    main()
