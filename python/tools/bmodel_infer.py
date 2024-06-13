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
import argparse
from transform.TpuLang import bmodel_inference_combine


def main():
    parser = argparse.ArgumentParser(
        description="Verify the correctness of BModel using reference data."
    )
    parser.add_argument(
        "context_dir",
        default="./",
        help="The folder should contain the BModel, its input_data, and tensor_location files.",
    )
    parser.add_argument(
        "reference_data",
        help="The reference data used for checking this BModel.",
    )
    parser.add_argument(
        "--out_fixed", action="store_true", help="Whether to output data as fixed num."
    )

    # config only when using SOC mode
    parser.add_argument(
        "--is_soc", action="store_true", help="Whether to output data as fixed num."
    )
    parser.add_argument(
        "--tmp_path",
        default="/tmp",
        type=str,
        help="Dir to store tmp file on device when using soc mode.",
    )
    parser.add_argument(
        "--trans_tools",
        action="store_true",
        help="Whether to transfer soc infer tools automatically. Required on device when using soc mode.",
    )
    parser.add_argument(
        "--tools_path",
        default="/soc_infer",
        type=str,
        help="Dir to store soc infer tools on device when using soc mode. Do not transfer tools to same dir more than once.",
    )
    parser.add_argument(
        "--hostname",
        default="",
        type=str,
        help="IP for device when using soc mode.",
    )
    parser.add_argument(
        "--port", default=22, type=int, help="Port for device when using soc mode."
    )
    parser.add_argument(
        "--username",
        default="",
        type=str,
        help="Username for device when using soc mode.",
    )
    parser.add_argument(
        "--password",
        default="",
        type=str,
        help="Password for device when using soc mode.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = main()
    context_dir = args.context_dir
    assert os.path.isdir(context_dir)
    bmodel_file = os.path.join(context_dir, "compilation.bmodel")
    final_mlir_fn = os.path.join(context_dir, "final.mlir")
    tensor_loc_file = os.path.join(context_dir, "tensor_location.json")

    assert all(
        [os.path.exists(i) for i in [bmodel_file, final_mlir_fn, tensor_loc_file]]
    )

    input_data_fn = os.path.join(context_dir, "input_ref_data.dat")

    reference_data_fn = args.reference_data

    bmodel_inference_combine(
        bmodel_file,
        final_mlir_fn,
        input_data_fn,
        tensor_loc_file,
        reference_data_fn,
        dump_file=True,
        save_path=context_dir,
        out_fixed=args.out_fixed,
        is_soc=args.is_soc,
        tmp_path=args.tmp_path,
        trans_tools=args.trans_tools,
        tools_path=args.tools_path,
        hostname=args.hostname,
        port=args.port,
        username=args.username,
        password=args.password,
    )
