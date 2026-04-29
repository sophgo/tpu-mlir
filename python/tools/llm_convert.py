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
import importlib
import os
import pymlir


def parse_max_pixels(value):
    """
    Parse a "width,height" string and return [width, height].
    """
    if ',' not in value:
        raise argparse.ArgumentTypeError(
            "The input must be two integers separated by a comma, e.g., 672,896")
    parts = value.split(',')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "The input must be two integers separated by a comma, e.g., 672,896")
    try:
        width = int(parts[0].strip())
        height = int(parts[1].strip())
    except ValueError:
        raise argparse.ArgumentTypeError("The input values must be integers, e.g., 672,896")
    return [width, height]


def deprecated_option(cond, msg):
    if cond:
        raise RuntimeError(msg)


# Dispatch table: model_type(s) -> (module, class, options).
# Adding a new model = one line here; no if/elif edits.
#   default_max_shape: per-model fallback when --max_pixels is omitted.
#   force_dynamic    : flip args.dynamic on (with a notice).
#   pixel_multiple   : require max_pixels % (m*m) == 0 (early validation).
LLM_CONVERTERS = [
    (("qwen3", "qwen2", "llama", "minicpm", "qwen2_moe"), "llm.LlmConverter", "LlmConverter", {}),
    (("mllama", ), "llm.Llama3_2VConverter", "Llama3_2VConverter", {}),
    (("chatglm", ), "llm.Chatglm3Converter", "Chatglm3Converter", {}),
    (("phi3", ), "llm.Phi3Converter", "Phi3Converter", {}),
    (("qwen2_vl", ), "llm.Qwen2VLConverter", "Qwen2VLConverter", {
        "pixel_multiple": 28
    }),
    (("qwen2_5_vl", ), "llm.Qwen2_5VLConverter", "Qwen2_5VLConverter", {
        "default_max_shape": (672, 896),
        "pixel_multiple": 28
    }),
    (("qwen3_vl", ), "llm.Qwen3VLConverter", "Qwen3VLConverter", {
        "pixel_multiple": 32
    }),
    (("qwen3_5", ), "llm.Qwen3_5Converter", "Qwen3_5Converter", {
        "force_dynamic": True,
        "pixel_multiple": 32
    }),
    (("qwen2_5_omni", ), "llm.Qwen2_5OConverter", "Qwen2_5OConverter", {}),
    (("qwen3_asr", ), "llm.Qwen3AsrConverter", "Qwen3AsrConverter", {}),
    (("internvl_chat", ), "llm.InternVL3Converter", "InternVL3Converter", {}),
    (("gemma3", ), "llm.Gemma3Converter", "Gemma3Converter", {}),
    (("glm4v", ), "llm.GLM4VConverter", "GLM4VConverter", {
        "pixel_multiple": 28
    }),
    (("minicpmv", ), "llm.MiniCPMV4Converter", "MiniCPMV4Converter", {
        "default_max_shape": (980, 980),
        "pixel_multiple": 28
    }),
    (("janus", ), "llm.JanusConverter", "JanusConverter", {}),
    (("paddleocr_vl", ), "llm.PaddleOCRVLConverter", "PaddleOCRVLConverter", {}),
    (("lfm2_vl", ), "llm.LFM2VLConverter", "LFM2VLConverter", {}),
]


def find_converter_spec(model_type):
    for types, module, cls, opts in LLM_CONVERTERS:
        if model_type in types:
            return module, cls, opts
    raise RuntimeError("Unsupported model type: {}".format(model_type))


def auto_out_dir(model_path, chip, quantize):
    base = os.path.basename(os.path.normpath(model_path)).lower()
    return "./{}_{}_{}".format(base, chip, quantize)


if __name__ == '__main__':
    # yapf: disable
    parser = argparse.ArgumentParser(description='llm_exporter', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='original weight, like ./Qwen2-7B-Instruct')
    parser.add_argument('-s', '--seq_length', type=int, required=True,
                        help="sequence length")
    parser.add_argument('-q', '--quantize', type=str, default="auto",
                        choices=["auto", "bf16", "w8bf16", "w4bf16", "f16", "w8f16", "w4f16"],
                        help="quantize type for bmodel")
    parser.add_argument('-g', "--q_group_size", default=64, type=int,
                        help="group size for per-group quant, only used in quant mode")
    parser.add_argument('-c', '--chip', "--processor", type=str, default="bm1684x",
                        choices=["bm1684x", "bm1688", "cv186x", "bm1690", "bm1684x2"],
                        help="chip type for bmodel")
    parser.add_argument('--num_device', type=int, default=1,
                        help="num device for bmodel")
    parser.add_argument('--distribute_strategy', type=str, default="tp",
                        choices=["tp", "pp"],
                        help="distribute strategy for bmodel, only used when num_device > 1")
    parser.add_argument('--num_core', type=int, default=0, help="num cores for bmodel")
    parser.add_argument('-b', '--batch', type=int, default=1,
                        help='batch size for bmodel')
    parser.add_argument('--lora_max_rank', type=int, default=0, help="lora rank, default is 0 means no lora")
    parser.add_argument('--symmetric', action='store_true', help='do symmetric quantize')
    parser.add_argument('--embedding_disk', action='store_true',
                        help='export embedding as bin file and inference by cpu')
    parser.add_argument('--do_sample', action='store_true',
                        help='Add sample head and separate greedy head from lmhead')
    parser.add_argument('--use_block_with_kv', action='store_true',
                        help='use history kv for prefill, default is False')
    parser.add_argument('--share_prompt', action='store_true',
                        help='share the same prompt for multi dialog, default is False')
    parser.add_argument('--max_input_length', type=int, default=0,
                        help='max input length for prefill, default 0 means the same as seq_length')
    parser.add_argument('--max_prefill_kv_length', type=int, default=0,
                        help='max prefill kv length, default 0 means the same as seq_length')
    parser.add_argument('--max_pixels', type=parse_max_pixels, default=None,
                        help="max pixels for vit as 'width,height', e.g. 672,896. "
                             "If unset, defaults are picked by model_type: "
                             "qwen2_5_vl -> 672,896, minicpmv -> 980,980, others -> 768,768.")
    parser.add_argument('--dynamic', action='store_true',
                        help='enable dynamic compiling for llm prefill')
    parser.add_argument('--debug', action='store_true',
                        help='enable debug mode, temp files will not be deleted')
    parser.add_argument('--only_mlir', action='store_true', help='only export mlir file, do not convert to bmodel')
    parser.add_argument('--rvti', action='store_true',
                        help='enable rvti, only for bm1684x2 and bm1690e')
    parser.add_argument("--again", action='store_true',
                        help='resume an interrupted conversion: skip stages whose '
                             'outputs already exist in --out_dir.')
    parser.add_argument('--dry_run', action='store_true',
                        help='resolve the configuration, print it, and exit without '
                             'invoking the converter')
    parser.add_argument("-V", "--version", action='version', version='%(prog)s ' + pymlir.__version__)
    parser.add_argument('-o', '--out_dir', type=str, default=None,
                        help='output mlir/bmodel path. If unset, defaults to '
                             './<model_name>_<chip>_<quantize>')
    #========== DEPRECATED Options ==============
    parser.add_argument("--dynamic_vit", action='store_true',
                        help='enable dynamic compiling for vit')
    parser.add_argument('--input_length_list', action="store_true",
                        help="a list of input lengths separated by '+', each input length can be a single integer")
    args = parser.parse_args()
    deprecated_option(args.dynamic_vit, "DEPRECATED,default is dynamic compiling")
    deprecated_option(args.input_length_list, "DEPRECATED, please use --dynamic to enable dynamic compiling")
    # yapf: enable
    if args.share_prompt:
        args.use_block_with_kv = True
    if args.input_length_list and args.max_input_length > 0:
        raise ValueError("Cannot set both input_length_list and max_input_length.")
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

    # Resolve out_dir before loading the model so --dry_run is fast.
    if args.out_dir is None:
        args.out_dir = auto_out_dir(args.model_path, args.chip, args.quantize)
        print("Info: --out_dir not set, using '{}'".format(args.out_dir))

    from transformers import AutoConfig

    # Qwen-ASR uses a custom model type; importing qwen_asr registers it with
    # transformers so AutoConfig can resolve it.
    _path_lower = args.model_path.lower()
    if "qwen" in _path_lower and "asr" in _path_lower:
        import qwen_asr  # noqa: F401
    try:
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError("Failed to load model config from '{}': {}\n"
                           "Hint: your transformers/torchvision may be outdated. "
                           "Try updating them and run again:\n"
                           "    pip3 install transformers torchvision -U".format(
                               args.model_path, e)) from e

    module_name, class_name, opts = find_converter_spec(config.model_type)

    # Resolve max_pixels: user value > per-model default > generic 768x768.
    if args.max_pixels is None:
        default_shape = opts.get("default_max_shape", (768, 768))
        args.max_shape = [int(default_shape[0]), int(default_shape[1])]
    else:
        args.max_shape = list(args.max_pixels)
    args.max_pixels = args.max_shape[0] * args.max_shape[1]

    pixel_multiple = opts.get("pixel_multiple")
    if pixel_multiple and args.max_pixels % (pixel_multiple * pixel_multiple) != 0:
        raise ValueError(
            "max_pixels (={}, from {}x{}) must be a multiple of {}*{} for model_type '{}'.".format(
                args.max_pixels, args.max_shape[0], args.max_shape[1], pixel_multiple,
                pixel_multiple, config.model_type))

    if opts.get("force_dynamic") and not args.dynamic:
        print("Info: forcing --dynamic for model_type '{}'".format(config.model_type))
        args.dynamic = True

    if args.dry_run:
        print("=== llm_convert dry-run ===")
        for k in ("model_path", "model_type:" + config.model_type, "out_dir", "chip", "quantize",
                  "seq_length", "max_input_length", "max_prefill_kv_length", "max_shape",
                  "max_pixels", "num_device", "distribute_strategy", "num_core", "batch", "dynamic",
                  "embedding_disk", "do_sample", "use_block_with_kv", "share_prompt",
                  "lora_max_rank", "symmetric"):
            if ":" in k:
                key, value = k.split(":", 1)
                print("  {:<22} = {}".format(key, value))
            else:
                print("  {:<22} = {}".format(k, getattr(args, k)))
        print("  converter             = {}.{}".format(module_name, class_name))
        import sys
        sys.exit(0)

    module = importlib.import_module(module_name)
    converter_cls = getattr(module, class_name)
    converter = converter_cls(args, config)
    converter.run()
