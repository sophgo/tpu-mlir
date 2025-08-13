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


def gen_struct_optimize_config(model_name: str = "",
                               pass_type: str = "struct_optimize",
                               chip: str = "",
                               quantize: str = "",
                               overwrite: bool = False,
                               output: str = "",
                               silence: bool = False,
                               struct_optimize_id: int = 0) -> None:
    """Generate struct_optimize specific configuration

    struct_optimize_id mapping:
      0 -> no rules (do nothing)
      1 -> CLIP model rules
      2,3... -> reserved for future model family
    """
    config_filename = ""

    if not output:
        if not model_name:
            print("[ERROR] '--model_name' must be set if '--output' or '-o' is not set.")
            exit(1)

    if output:
        config_filename = output
    else:
        # TOP pass allows using default values
        if not chip or not quantize:
            print(
                "[WARNING] chip or quantize not set for struct_optimize pass, using default values")
            chip = chip or "ALL"
            quantize = quantize or "f32"
        config_filename = "{}_{}_{}.struct_optimize.json".format(model_name, chip.lower(),
                                                                 quantize.lower())

    # check if file exists
    if os.path.exists(config_filename) and not overwrite:
        print(
            f"[ERROR] Config file \'{config_filename}\' already exists, do nothing to avoid overwriting. "
            "If you want to overwrite it, please use \'--overwrite\' option.")
        exit(1)

    # Build rules based on struct_optimize_id
    # ------------
    if struct_optimize_id == 1:
        # CLIP model specific rules
        rule_ConvertRemovePermuteBeforeLayerNormPattern = generate_rewriter_rules(
            "ConvertRemovePermuteBeforeLayerNormPattern", enable=(True, "bool"))
        rule_ConvertRemovePermuteBetweenAddGatherPattern = generate_rewriter_rules(
            "ConvertRemovePermuteBetweenAddGatherPattern", enable=(True, "bool"))
        rule_ConvertFuseAttentionSlicePattern = generate_rewriter_rules(
            "ConvertFuseAttentionSlicePattern", enable=(True, "bool"))
        rule_ConvertOptimizeReshapePermuteChainPattern = generate_rewriter_rules(
            "ConvertOptimizeReshapePermuteChainPattern", enable=(True, "bool"))
        rule_ConvertPermuteReshapeChainFixPattern = generate_rewriter_rules(
            "ConvertPermuteReshapeChainFixPattern", enable=(True, "bool"))

        struct_optimize_id_1_rules = (rule_ConvertRemovePermuteBeforeLayerNormPattern +
                                      rule_ConvertRemovePermuteBetweenAddGatherPattern +
                                      rule_ConvertFuseAttentionSlicePattern +
                                      rule_ConvertOptimizeReshapePermuteChainPattern +
                                      rule_ConvertPermuteReshapeChainFixPattern)
    else:
        struct_optimize_id_1_rules = []

    # Generate config based on optimize_id
    # ------------
    if struct_optimize_id == 0:
        if not silence:
            print("[Info] struct_optimize_id=0: no struct optimization rules (no-op).")
        config = create_config([])
    elif struct_optimize_id == 1:
        if not silence:
            print("[Info] struct_optimize_id=1: applying CLIP-specific rules.")
        config = create_config(struct_optimize_id_1_rules)
    else:
        if not silence:
            print(f"[Info] struct_optimize_id={struct_optimize_id}: no rules yet.")
        config = create_config([])

    # Save the config to a file
    save_config(config, config_filename, silence=silence)


def gen_rewriter_config(model_name: str = "",
                        pass_type: str = "tpu_processor_optimize",
                        chip: str = "",
                        quantize: str = "",
                        overwrite: bool = False,
                        output: str = "",
                        silence: bool = False) -> None:

    config_filename = ""

    if not output:
        if not model_name or not pass_type or not chip or not quantize:
            print(
                "[ERROR] \'--model_name\', \'--pass_type\', \'--chip\', \'--quantize\' must be set if \'--output\' or \'-o\' is not set."
            )
            exit(1)
    if output:
        config_filename = output
    else:
        if pass_type == "tpu_processor_optimize":
            assert chip, "chip must be set when pass_type is tpu_processor_optimize"
            assert quantize, "quantize must be set when pass_type is tpu_processor_optimize"
            config_filename = "{}_{}_{}.tpu_processor_optimize.json".format(
                model_name, chip.lower(), quantize.lower())
        else:
            # not implemented
            assert False, "[ERROR] Only tpu_processor_optimize pass type is supported in this script."

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

    # SelfAttnTileHeadPattern (tile Head)
    # ------------
    rule_SelfAttnTileHeadPattern_1 = generate_rewriter_rules(
        "SelfAttnTileHeadPattern",
        out_feature_shape=([1, 257, 16, 257], "vector<int>"),
        head_per_tile_attn=(2, "int"),
        head_per_tile_mlp=(16, "int"),
    )
    rule_SelfAttnTileHeadPattern_2 = generate_rewriter_rules(
        "SelfAttnTileHeadPattern",
        out_feature_shape=([-1, 257, 16, 257], "vector<int>"),
        head_per_tile_attn=(1, "int"),
        head_per_tile_mlp=(16, "int"),
    )
    # TileTrVPattern (tile N)
    # ------------
    rule_TileTrVPattern = generate_rewriter_rules(
        "TileTrVPattern",
        out_feature_shape=([-1, 257, 2730], "vector<int>"),
        mm_weight_shape=([1024, 2730], "vector<int>"),
        tile_len=(512, "int"),
    )
    # MatmulTileKPattern (tile K)
    # ------------
    rule_MatmulTileKPattern = generate_rewriter_rules(
        "MatmulTileKPattern",
        out_feature_shape=([-1, 257, 1024], "vector<int>"),
        mm_weight_shape=([2730, 1024], "vector<int>"),
        tile_len=(512, "int"),
    )
    # TileLayerNormPattern (tile C)
    # ------------
    rule_TileLayerNormPattern_1 = generate_rewriter_rules(
        "TileLayerNormPattern",
        out_feature_shape=([-1, 257, 1024], "vector<int>"),
        tile_len=(64, "int"),
    )
    rule_TileLayerNormPattern_2 = generate_rewriter_rules(
        "TileLayerNormPattern",
        out_feature_shape=([-1, 257, 2730], "vector<int>"),
        tile_len=(64, "int"),
    )

    # Create config with the rewrite rules
    config = create_config(
        # rule_SplitQuantizedMLP2Pattern_template,
        rule_SplitQuantizedMLP2Pattern_1,
        rule_SplitQuantizedMLP2Pattern_2,
        rule_SplitQuantizedMLP2Pattern_3,
        # rule for eva-02
        rule_SelfAttnTileHeadPattern_1,
        rule_SelfAttnTileHeadPattern_2,
        rule_TileTrVPattern,
        # rule_MatmulTileKPattern,
        rule_TileLayerNormPattern_1,
        rule_TileLayerNormPattern_2)

    # Save the config to a file
    save_config(config, config_filename, silence=silence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="model name form \'model_transform\' step")
    parser.add_argument("--pass_type",
                        type=str,
                        default="tpu_processor_optimize",
                        choices=['tpu_processor_optimize', 'struct_optimize'],
                        help="pass type needs rewriter config")

    parser.add_argument(
        "--chip",
        type=str,
        default="",
        choices=[
            'bm1688', 'bm1684x', 'bm1684', 'bm1690', 'mars3', 'sgtpuv8', 'sg2380', 'cv183x',
            'cv182x', 'cv181x', 'cv180x', 'cv186x', 'cpu'
        ],
        help="if pass_type is \'tpu_processor_optimize\', set chip type as \'model_deploy\' step")
    parser.add_argument(
        "--quantize",
        type=str,
        default="",
        choices=[
            'F32', 'BF16', 'F16', 'INT8', 'INT4', 'W8F16', 'W8BF16', 'W4F16', 'W4BF16', "F8E4M3",
            "F8E5M2", 'QDQ'
        ],
        help="if pass_type is \'tpu_processor_optimize\', set quantize type as \'model_deploy\' step"
    )
    parser.add_argument("--overwrite",
                        action='store_true',
                        help="overwrite the config file if it exists")
    parser.add_argument('-o',
                        "--output",
                        type=str,
                        default="",
                        help="output file name, default is empty")
    parser.add_argument("--struct_optimize_id",
                        type=int,
                        default=0,
                        help="struct_optimize_id for struct_optimize pass")
    args = parser.parse_args()

    if args.pass_type == "struct_optimize":
        gen_struct_optimize_config(args.model_name,
                                   args.pass_type,
                                   args.chip,
                                   args.quantize,
                                   args.overwrite,
                                   args.output,
                                   silence=False,
                                   struct_optimize_id=args.struct_optimize_id)
    else:
        gen_rewriter_config(args.model_name, args.pass_type, args.chip, args.quantize,
                            args.overwrite, args.output)
