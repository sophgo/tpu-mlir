#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import collections
import json
import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Locate op in final.mlir according to the given core_id, subnet_id, tiu/gdma cmd_id ."
    )
    parser.add_argument(
        "tensor_location",
        type=str,
        nargs="?",
        help="The file path of tensor_location.json.",
    )
    parser.add_argument(
        "subnet_id",
        type=int,
        default=0,
    )
    parser.add_argument(
        "core_id",
        type=int,
        default=0,
    )
    parser.add_argument(
        "cmd_type",
        type=int,
        default=0,
        help=" 'tiu' as 0 , 'gdma' as 1.",
    )
    parser.add_argument(
        "cmd_id",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = main()
    tensor_loc_file = args.tensor_location
    assert os.path.exists(tensor_loc_file)
    with open(tensor_loc_file, "r+", encoding="utf-8") as f:
        op_list = json.load(f)

    file_line_result = None
    local_op_file_line_buffer = collections.defaultdict(int)
    for op_idx, op_dict in enumerate(op_list):
        if op_dict["subnet_id"] != args.subnet_id:
            continue
        if op_dict["core_id"] != args.core_id:
            continue
        if op_dict["is_local"]:
            local_op_file_line_buffer[op_dict["file-line"]] += 1

        if (
            op_dict["tiu_dma_id(before)"][args.cmd_type]
            < args.cmd_id
            <= op_dict["tiu_dma_id(after)"][args.cmd_type]
        ):
            file_line_result = op_dict["file-line"]
            repeat_count = local_op_file_line_buffer[
                op_dict["file-line"]
            ]  # if op is in layer group, it may appears multiple times. Otherwise, it only appears once.
            break
    if file_line_result:
        if args.cmd_type == 0:
            cmd_type = "Tiu"
        elif args.cmd_type == 1:
            cmd_type = "Gdma"
        print(
            f"The {repeat_count} times, file_line:{file_line_result} CONTAINS, subnet_id:{args.subnet_id}, core_id:{args.core_id}, {cmd_type}_{args.cmd_id}"
        )
    else:
        print("Not Found! Please check your input.")
