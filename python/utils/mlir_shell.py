# Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
#
# Licensed under the Apache License v2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ==============================================================================

import subprocess
import os


def checkReturnValue(ret, func: str):
    if ret.returncode == 0:
        print("{} run success".format(func))
    else:
        print("[!Error]cmd: {}".format(" ".join(ret.args)))
        print("error occured: {}, func: {}\nmsg: {}".format(ret.returncode, func, ret))


def mlir_opt_for_top(mlirfile, opt_mlirfile):
    ret = subprocess.run([
        "tpuc-opt", "--canonicalize", "--mark-FLOPs", "--save-weight", mlirfile, "-o",
        opt_mlirfile
    ])
    checkReturnValue(ret, "tpuc-opt")
    return ret.returncode


def _os_system(cmd: list):
    cmd_str = ""
    for s in cmd:
        cmd_str += s + " "
    ret = os.system(cmd_str)
    if ret == 0:
        print("{} run success".format(cmd_str))
    else:
        print("[!Error]cmd: {}".format(cmd_str))
    return ret


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
    cmd.extend([lower_param, "--save-weight", "-o", tpu_mlir])
    return _os_system(cmd)
    #ret = subprocess.run(cmd)
    #checkReturnValue(ret, "tpuc-opt")
    #return ret


def mlir_to_model(tpu_mlir: str, model: str, final_mlir: str):
    codegen_param = "--codegen=\"model_file={}\"".format(model)
    cmd = [
        "tpuc-opt", tpu_mlir, "--weight-reorder", "--subnet-divide", "--layer-group",
        "--address-asign", "--save-weight", codegen_param, "-o", final_mlir
    ]
    return _os_system(cmd)
    #ret = subprocess.run(cmd)
    #checkReturnValue(ret, "tpuc-opt")
    # return ret


def f32_blobs_compare(a_npz: str, b_npz: str, tolerance: str, excepts=None, show_detail=True):
    cmd = ["npz_tool.py", "compare", a_npz, b_npz, "--tolerance", tolerance]
    if excepts:
        cmd.extend(["--except", excepts])
    if show_detail:
        cmd.append('-vv')
    ret = subprocess.run(cmd)
    checkReturnValue(ret, "compare")
    return ret.returncode
