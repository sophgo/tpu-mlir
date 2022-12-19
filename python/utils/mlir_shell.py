# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
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
    cmd = [
        "tpuc-opt",
        "--canonicalize",
        "--mark-FLOPs",
        "--save-weight",
        "--mlir-print-debuginfo",
        mlirfile,
        "-o",
        opt_mlirfile,
    ]
    _os_system(cmd)


def mlir_lowering(top_mlir: str,
                  tpu_mlir: str,
                  mode: str,
                  chip: str,
                  cali_table: str = None,
                  asymmetric: bool = False,
                  quantize_table: str = None,
                  qdq: bool = False):
    cmd = ["tpuc-opt", top_mlir]
    if qdq:
        assert cali_table == None, "qdq cannot work with cali_table"
        assert quantize_table == None, "qdq cannot work with quantize_table"
        cmd.extend(["--convert-qdq-to-calibrated-dialect"])
        mode = 'int8'
    if cali_table != None:
        cali_param = "--import-calibration-table=\"file={} asymmetric={}\"".format(
            cali_table, asymmetric)
        cmd.extend([cali_param])
    qtable = ""
    if quantize_table:
        qtable = "qtable={}".format(quantize_table)
    lower_param = "--convert-top-to-tpu=\"mode={} {} asymmetric={} chip={}\"".format(
        mode.upper(), qtable, asymmetric, chip.lower())
    cmd.extend([
        lower_param,
        "--canonicalize",
        "--save-weight",
        "--mlir-print-debuginfo",
        "-o",
        tpu_mlir,
    ])
    _os_system(cmd)


def mlir_to_model(tpu_mlir: str,
                  model: str,
                  final_mlir: str,
                  quant_input: bool = False,
                  quant_output: bool = False):
    codegen_param = '--codegen="model_file={}"'.format(model)
    strip_io_quant_param = '--strip-io-quant="quant_input={} quant_output={}"'.format(
        quant_input, quant_output)
    cmd = [
        "tpuc-opt",
        tpu_mlir,
        strip_io_quant_param,
        "--weight-reorder",
        "--subnet-divide",
        "--layer-group",
        "--address-assign",
        "--save-weight",
        codegen_param,
        "--mlir-print-debuginfo",
        "-o",
        final_mlir,
    ]

    _os_system(cmd)
    try:
        _os_system(["mv compiler_profile_0.txt", model + ".compiler_profile_0.txt"])
    except RuntimeError:
        pass


# tmp for cvitek, remove in the future
def mlir_to_cvi_model(tpu_mlir: str,
                      model: str,
                      final_mlir: str,
                      quant_input: bool = False,
                      quant_output: bool = False):
    codegen_param = '--cv-codegen="model_file={}"'.format(model)
    strip_io_quant_param = '--strip-io-quant="quant_input={} quant_output={}"'.format(
        quant_input, quant_output)
    cmd = [
        "tpuc-opt",
        tpu_mlir,
        "--convert-relu-limit",
        strip_io_quant_param,
        "--weight-reorder",
        "--subnet-divide",
        "--address-assign",
        codegen_param,
        "--save-weight",
        "--mlir-print-debuginfo",
        "-o",
        final_mlir,
    ]

    _os_system(cmd)


def f32_blobs_compare(a_npz: str, b_npz: str, tolerance: str, excepts=None, show_detail=True):
    cmd = ["npz_tool.py", "compare", a_npz, b_npz, "--tolerance", tolerance]
    if excepts:
        cmd.extend(["--except", excepts])
    if show_detail:
        cmd.append('-vv')
    _os_system(cmd)
