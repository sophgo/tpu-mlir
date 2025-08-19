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


def gen_layer_group_config(model_name: str = "",
                           chip: str = "",
                           quantize: str = "",
                           overwrite: bool = False,
                           output: str = "",
                           silence: bool = False,
                           strategy: int = 0) -> None:
    config_filename = "layer_group_config.json"
    if not output:
        if not model_name or not chip or not quantize:
            missing_param = ""
            if not model_name:
                missing_param += "[--model_name] "
            if not chip:
                missing_param += "[--chip] "
            if not quantize:
                missing_param += "[--quantize] "
            print(
                f"[WARNING] Using default config filename \'layer_group_config.json\' for missing {missing_param}."
            )
        else:
            config_filename = "{}_{}_{}.layer_group_config.json".format(
                model_name, chip.lower(), quantize.lower())
    else:
        config_filename = output

    # check if file exists
    if os.path.exists(config_filename) and not overwrite:
        print(
            f"[WARNING] Config file \'{config_filename}\' already exists, do nothing to avoid overwriting. "
            "If you want to overwrite it, please use \'--overwrite\' option.")
        return

    # Add layer group config template
    #######################################################################
    def get_default_config(chip: str, strategy: int) -> Dict[str, Any]:
        chip = chip.lower()
        if chip == "cv184x":
            return {
                "shape_secs_search_strategy":
                strategy,
                "sc_method_configs": [
                    {
                        "sc_method": "sc_method_quick_search",
                        "MAX_TRY_NUM": 60
                    },
                    {
                        "sc_method": "sc_method_search_better_v1",
                        "NSECS_SEARCH_RECORD_THRESHOLD": 3,
                        "CSECS_SEARCH_RECORD_THRESHOLD": 3,
                        "DSECS_SEARCH_RECORD_THRESHOLD": 3,
                        "HSECS_SEARCH_RECORD_THRESHOLD": 3,
                        "WSECS_SEARCH_RECORD_THRESHOLD": 3,
                    },
                    {
                        "sc_method": "sc_method_search_better_v2",
                        "MAX_NSECS": 32,
                        "MAX_CSECS": 32,
                        "MAX_DSECS": 32,
                        "MAX_HSECS": 64,
                        "MAX_WSECS": 32,
                        "NSECS_SEARCH_RECORD_THRESHOLD": 2,
                        "CSECS_SEARCH_RECORD_THRESHOLD": 2,
                        "DSECS_SEARCH_RECORD_THRESHOLD": 2,
                        "HSECS_SEARCH_RECORD_THRESHOLD": 2,
                        "WSECS_SEARCH_RECORD_THRESHOLD": 3,
                    },
                ],
            }
        else:
            return {
                "shape_secs_search_strategy":
                strategy,
                "sc_method_configs": [
                    {
                        "sc_method": "sc_method_quick_search",
                        "MAX_TRY_NUM": 20
                    },
                    {
                        "sc_method": "sc_method_search_better_v1",
                        "NSECS_SEARCH_RECORD_THRESHOLD": 3,
                        "CSECS_SEARCH_RECORD_THRESHOLD": 3,
                        "DSECS_SEARCH_RECORD_THRESHOLD": 3,
                        "HSECS_SEARCH_RECORD_THRESHOLD": 3,
                        "WSECS_SEARCH_RECORD_THRESHOLD": 3,
                    },
                    {
                        "sc_method": "sc_method_search_better_v2",
                        "MAX_NSECS": 32,
                        "MAX_CSECS": 32,
                        "MAX_DSECS": 32,
                        "MAX_HSECS": 32,
                        "MAX_WSECS": 32,
                        "NSECS_SEARCH_RECORD_THRESHOLD": 2,
                        "CSECS_SEARCH_RECORD_THRESHOLD": 2,
                        "DSECS_SEARCH_RECORD_THRESHOLD": 2,
                        "HSECS_SEARCH_RECORD_THRESHOLD": 2,
                        "WSECS_SEARCH_RECORD_THRESHOLD": 2,
                    },
                ],
            }

    config = get_default_config(chip, strategy)
    # Save the config to a file
    save_config(config, config_filename, silence=silence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="model name form \'model_transform\' step")
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
    parser.add_argument("--strategy",
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help="strategy for layer group search, 0: SEARCH_QUICK, 1: SEARCH_BETTER")
    parser.add_argument("--overwrite",
                        action='store_true',
                        help="overwrite the config file if it exists")
    parser.add_argument("--silence",
                        action='store_true',
                        help="silence the output, only print success message")
    parser.add_argument('-o',
                        "--output",
                        type=str,
                        default="",
                        help="output file name, default is empty")
    args = parser.parse_args()

    gen_layer_group_config(args.model_name, args.chip, args.quantize, args.overwrite, args.output,
                           False, args.strategy)
