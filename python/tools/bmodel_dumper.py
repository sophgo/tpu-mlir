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
from debugger.plugins.data_dump import DataDump
from tdb import TdbInterface
import argparse

def __parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "context_dir",
        help="The folder should contain the BModel and tensor_location files.",
    )
    parser.add_argument(
        "input",
        help="Input data",
    )
    parser.add_argument(
        "output",
        help="Output data",
    )
    parser.add_argument(
        "--excepts",
        type=str,
        help="List of tensors except from comparing"
    )
    parser.add_argument(
        "--ddr_size",
        type=int,
        nargs="?",
        default=2**32,
        help="The inputs data of the BModel.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = __parse_args()
    context_dir = args.context_dir
    assert os.path.isdir(context_dir)
    bmodel_file = os.path.join(context_dir, "compilation.bmodel")
    final_mlir_fn = os.path.join(context_dir, "final.mlir")
    tensor_loc_file = os.path.join(context_dir, "tensor_location.json")

    assert os.path.exists(bmodel_file)
    assert os.path.exists(final_mlir_fn)
    assert os.path.exists(tensor_loc_file)

    extra_plugins = ["data-dump"]

    tdb = TdbInterface(
        bmodel_file=bmodel_file,
        final_mlir_fn=final_mlir_fn,
        tensor_loc_file=tensor_loc_file,
        input_data_fn=args.input,
        ddr_size=args.ddr_size,
        extra_plugins = extra_plugins,
    )

    if args.excepts:
        excepts = [str(s) for s in args.excepts.split(",")]
    else:
        excepts = []

    plugin: DataDump = tdb.get_plugin(DataDump)
    plugin.output = args.output
    plugin.excepts.update(excepts)

    tdb.do_run("")
    tdb.cmdloop()
