# Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
#
# Licensed under the Apache License v2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ==============================================================================

import os

def _os_system(cmd: list):
    cmd_str = ""
    for s in cmd:
        cmd_str += s + " "
    print("[Running]: {}".format(cmd_str))
    ret = os.system(cmd_str)
    if ret == 0:
        print("[Success]: {}".format(cmd_str))
    else:
        raise RuntimeError("[!Error]: {}".format(cmd_str))

def mlir_opt_for_top(mlirfile, opt_mlirfile):
    cmd = ([
        "tpuc-opt", "--canonicalize", "--mark-FLOPs", "--save-weight", mlirfile, "-o",
        opt_mlirfile
    ])
    _os_system(cmd)

def mlir_lowering(top_mlir: str,
                  tpu_mlir: str,
                  mode: str,
                  chip: str,
                  cali_table=None,
                  asymmetric: bool = False,
                  quantize_table=None):
    cmd = ["tpuc-opt", top_mlir]
    if cali_table != None:
        cali_param = "--import-calibration-table=\"file={} asymmetric={}\"".format(
            cali_table, asymmetric)
        cmd.extend([cali_param])
    lower_param = "--lowering=\"mode={} asymmetric={} chip={}\"".format(
        mode.upper(), asymmetric, chip.lower())
    cmd.extend([lower_param, "--canonicalize", "--save-weight", "-o", tpu_mlir])
    _os_system(cmd)


def mlir_to_model(tpu_mlir: str,
                  model: str,
                  final_mlir: str,
                  quant_input: bool = False,
                  quant_output: bool = False):
    codegen_param = '--codegen="model_file={}"'.format(model)
    strip_io_cast_param = '--strip-io-cast="quant_input={} quant_output={}"'.format(
        quant_input, quant_output
    )
    cmd = [
        "tpuc-opt",
        tpu_mlir,
        strip_io_cast_param,
        "--weight-reorder",
        "--subnet-divide",
        "--layer-group",
        "--address-assign",
        "--save-weight",
        codegen_param,
        "-o",
        final_mlir,
    ]

    _os_system(cmd)
    _os_system(["mv compiler_profile_0.txt", model + ".compiler_profile_0.txt"])


def f32_blobs_compare(a_npz: str, b_npz: str, tolerance: str, excepts=None, show_detail=True):
    cmd = ["npz_tool.py", "compare", a_npz, b_npz, "--tolerance", tolerance]
    if excepts:
        cmd.extend(["--except", excepts])
    if show_detail:
        cmd.append('-vv')
    _os_system(cmd)
