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
from debugger.plugins.data_checker import DataCheck
from tdb import TdbInterface
import argparse


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
        "--tolerance", default="0.99,0.90", help="tolerance for compare."
    )
    parser.add_argument(
        "--report", type=str, help="The report file for saving state and internal data."
    )
    parser.add_argument(
        "--fail_fast", action="store_true", help="Stop if there is a check failure."
    )
    parser.add_argument(
        "--excepts", type=str, help="List of tensors except from comparing"
    )
    parser.add_argument(
        "--ddr_size",
        type=int,
        nargs="?",
        default=2**32,
        help="The inputs data of the BModel.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=str,
        nargs="?",
        const="",
        help="Control the report information.",
    )
    parser.add_argument("--no_interactive", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = main()
    if args.excepts:
        excepts = [str(s) for s in args.excepts.split(",")]
    else:
        excepts = []

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

    cos_t, euc_t = eval(args.tolerance)

    extra_plugins = []
    if args.verbose is not None:
        extra_plugins.append("progress")

    tdb = TdbInterface(
        bmodel_file=bmodel_file,
        final_mlir_fn=final_mlir_fn,
        tensor_loc_file=tensor_loc_file,
        input_data_fn=input_data_fn,
        reference_data_fn=reference_data_fn,
        extra_plugins=extra_plugins,
        extra_check=[],
        ddr_size=args.ddr_size,
    )
    plugin: DataCheck = tdb.get_plugin(DataCheck)
    if args.fail_fast:
        plugin.break_when_fail = True
    plugin.set_tol(cosine_similarity_tol=cos_t, euclidean_similarity_tol=euc_t)

    tdb.do_run("")
    tdb.message("(<file-line>:[operands]|[results])")
    plugin.do_summary("table")
    msg = """
    type `check` to start analysis and dump data.
    - `check summary table|reduce` to view report of check results.
    - `check data <file-line> <index>` to review value details
    - `check dump` or `check dump <file-name>` to dump tensor that failed in comparison into npz files
    """
    tdb.message(msg)
    if args.no_interactive:
        if args.report is None:
            args.report = os.path.join(context_dir, "failed_bmodel_outputs.npz")
        plugin.do_dump(args.report)
    else:
        tdb.cmdloop()
