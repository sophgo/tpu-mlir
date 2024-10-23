#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from numpy_helper import *
from numpy_helper.npz_compare import npz_compare
from numpy_helper.npz_visualize_diff import npz_visualize_diff
from numpy_helper.npz_dump import npz_dump
from numpy_helper.npz_statistic import npz_statistic
from numpy_helper.npz_cali_test import npz_cali_test

npz_tool_func = {
    "compare": npz_compare,
    "visualize_diff": npz_visualize_diff,
    'dump':npz_dump,
    "extract": npz_extract,
    "remove": npz_remove,
    "insert": npz_insert,
    "merge": npz_merge,
    "rename": npz_rename,
    "reshape": npz_reshape,
    "bf16_to_fp32": npz_bf16_to_fp32,
    "fp16_to_fp32": npz_fp16_to_fp32,
    "permute": npz_permute,
    "get_shape": get_npz_shape,
    "to_bin": npz_to_bin,
    "to_dat": npz_to_dat,
    "to_npy": npz_to_npy,
    "cali_test": npz_cali_test,
    "statistic": npz_statistic,
}

def main():
    args_list = sys.argv
    if len(args_list) < 2:
        funcs = npz_tool_func.keys()
        funcs_str = "["
        for idx, key in enumerate(npz_tool_func.keys()):
            funcs_str += key + ("|" if idx != len(funcs)-1 else '')
        funcs_str += "]"
        print(f"Usage: {args_list[0]} " + funcs_str + " ...")
        exit(-1)

    def NoneAndRaise(func):
        raise RuntimeError("No support {} Method".format(func))

    npz_tool_func.get(args_list[1], lambda x: NoneAndRaise(args_list[1]))(args_list[2:])


if __name__ == "__main__":
    main()
