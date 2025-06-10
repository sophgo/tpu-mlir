#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import os
import json
import argparse
from typing import List, Dict, Any


def generate_rewriter_rules(pattern_name: str, **params_kwargs: Any) -> List[Dict[str, Any]]:
    params_defaults = {}
    params_dtypes = {}
    for key, value in params_kwargs.items():
        params_defaults[key] = value[0]
        params_dtypes[key] = value[1]
    return [{
        "pattern_name": pattern_name,
        "params_dtype": params_dtypes,
        "params": params_defaults,
    }]


def create_config(*rules: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    all_rules = []
    for rule_list in rules:
        all_rules.extend(rule_list)

    return {"rewriter_rules": all_rules}


def save_config(config: Dict[str, Any],
                filename: str = "rewriter_rules.json",
                silence: bool = False) -> None:
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[Success] Create config file \'{filename}\'.")
    # print(f"Config filename: {filename}")
    if silence:
        return
    print("Content:")
    print(json.dumps(config, indent=2))


def gen_rewriter_config(model_name: str = "",
                        dialect: str = "tpu",
                        chip: str = "",
                        quantize: str = "",
                        overwrite: bool = False,
                        output: str = "",
                        silence: bool = False) -> None:
    config_filename = ""
    if not output:
        if not model_name or not dialect or not chip or not quantize:
            print(
                "[ERROR] \'--model_name\', \'--dialect\', \'--chip\', \'--quantize\' must be set if \'--output\' or \'-o\' is not set."
            )
            exit(1)
    if output:
        config_filename = output
    else:
        if dialect == "tpu":
            assert chip, "chip must be set when dialect is tpu"
            assert quantize, "quantize must be set when dialect is tpu"
            config_filename = "{}_{}_{}.tpu_processor_optimize.json".format(
                model_name, chip.lower(), quantize.lower())
        else:
            # not implemented
            assert False, "[ERROR] Only tpu dialect is supported in this script."

    # check if file exists
    if os.path.exists(config_filename) and not overwrite:
        print(
            f"[ERROR] Config file \'{config_filename}\' already exists, do nothing to avoid overwriting. "
            "If you want to overwrite it, please use \'--overwrite\' option.")
        exit(1)

    # Add rewrite rule template
    #######################################################################
    # SplitQuantizedMLP2Pattern
    # ------------
    rule_SplitQuantizedMLP2Pattern_template = generate_rewriter_rules(
        "SplitQuantizedMLP2Pattern",
        shape_input=([], "vector<int>"),
        shape_w0=([], "vector<int>"),
        shape_w1=([], "vector<int>"),
        split_num=(1, "int"),
    )
    rule_SplitQuantizedMLP2Pattern_1 = generate_rewriter_rules(
        "SplitQuantizedMLP2Pattern",
        shape_input=([1, 256, 1024], "vector<int>"),
        shape_w0=([1024, 4096], "vector<int>"),
        shape_w1=([4096, 1024], "vector<int>"),
        split_num=(8, "int"),
    )
    rule_SplitQuantizedMLP2Pattern_2 = generate_rewriter_rules(
        "SplitQuantizedMLP2Pattern",
        shape_input=([4, 256, 1024], "vector<int>"),
        shape_w0=([1024, 4096], "vector<int>"),
        shape_w1=([4096, 1024], "vector<int>"),
        split_num=(8, "int"),
    )
    rule_SplitQuantizedMLP2Pattern_3 = generate_rewriter_rules(
        "SplitQuantizedMLP2Pattern",
        shape_input=([8, 256, 1024], "vector<int>"),
        shape_w0=([1024, 4096], "vector<int>"),
        shape_w1=([4096, 1024], "vector<int>"),
        split_num=(8, "int"),
    )

    # Create config with the rewrite rules
    config = create_config(
        # rule_SplitQuantizedMLP2Pattern_template,
        rule_SplitQuantizedMLP2Pattern_1,
        rule_SplitQuantizedMLP2Pattern_2,
        rule_SplitQuantizedMLP2Pattern_3,
    )

    # Save the config to a file
    save_config(config, config_filename, silence=silence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="model name form \'model_transform\' step")
    parser.add_argument("--dialect",
                        type=str,
                        default="tpu",
                        choices=['tpu', 'top'],
                        help="dialect needs rewriter config")
    parser.add_argument("--chip",
                        type=str,
                        default="",
                        choices=[
                            'bm1688', 'bm1684x', 'bm1684', 'bm1690', 'mars3', 'sgtpuv8', 'sg2380',
                            'cv183x', 'cv182x', 'cv181x', 'cv180x', 'cv186x', 'cpu'
                        ],
                        help="if dialect is \'tpu\', set quantize type as \'model_deploy\' step")
    parser.add_argument("--quantize",
                        type=str,
                        default="",
                        choices=[
                            'F32', 'BF16', 'F16', 'INT8', 'INT4', 'W8F16', 'W8BF16', 'W4F16',
                            'W4BF16', "F8E4M3", "F8E5M2", 'QDQ'
                        ],
                        help="if dialect is \'tpu\', set quantize type as \'model_deploy\' step")
    parser.add_argument("--overwrite",
                        action='store_true',
                        help="overwrite the config file if it exists")
    parser.add_argument('-o',
                        "--output",
                        type=str,
                        default="",
                        help="output file name, default is empty")
    args = parser.parse_args()

    gen_rewriter_config(args.model_name, args.dialect, args.chip, args.quantize, args.overwrite,
                        args.output)
