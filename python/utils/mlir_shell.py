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
        "--init",
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
                  qdq: bool = False,
                  customization_format: str = None,
                  fuse_preprocess: bool = False,
                  aligned_input: bool = False):
    cmd = ["tpuc-opt", top_mlir, "--init"]
    mode = mode.upper()
    if mode == 'QDQ':
        assert cali_table == None, "qdq cannot work with cali_table"
        assert quantize_table == None, "qdq cannot work with quantize_table"
        cmd.extend(["--convert-qdq-to-calibrated-dialect"])
        mode = 'INT8'
    if mode != 'INT8':
        asymmetric = True
    if cali_table != None:
        cali_param = "--import-calibration-table=\"file={} asymmetric={}\"".format(
            cali_table, asymmetric)
        cmd.extend([cali_param])
    if fuse_preprocess:
        fuse_pre_param = "--fuse-preprocess=\"mode={} customization_format={}\"".format(
            mode, customization_format)
        cmd.extend([fuse_pre_param])
    if aligned_input:
        aligned_param = "--align-input=\"chip={} customization_format={}\"".format(
            chip.lower(), customization_format)
        cmd.extend([aligned_param])
    qtable = ""
    if quantize_table:
        qtable = "qtable={}".format(quantize_table)
    lower_param = "--convert-top-to-tpu=\"mode={} {} asymmetric={} chip={}\"".format(
        mode, qtable, asymmetric, chip.lower())
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
                  dynamic: bool = False,
                  quant_input: bool = False,
                  quant_output: bool = False,
                  disable_layer_group: bool = False):
    # generate final mlir
    strip_io_quant_param = '--strip-io-quant="quant_input={} quant_output={}"'.format(
        quant_input, quant_output)
    lg_param = '--layer-group="opt=2"'
    if model.endswith(".cvimodel"):
        lg_param = '--layer-group="opt=1"'
    if disable_layer_group:
        lg_param = ''
    cmd = [
        "tpuc-opt",
        tpu_mlir,
        "--init",
        "--mlir-disable-threading",
        "--do-extra-opt",
        strip_io_quant_param,
        "--weight-reorder",
        "--subnet-divide",
        lg_param,
        "--address-assign",
        #"--address-assign=\"reuse_addr=false\"",
        "--save-weight",
        "--mlir-print-debuginfo",
        "-o",
        final_mlir,
    ]

    _os_system(cmd)

    # codegen based on final mlir
    if model.endswith(".bmodel"):
        codegen_param = '--codegen="model_file={} dynamic={}"'.format(model, dynamic)
    elif model.endswith(".cvimodel"):
        codegen_param = '--cv-codegen="model_file={}"'.format(model)
    cmd = [
        "tpuc-opt",
        final_mlir,
        "--init",
        codegen_param,
        "-o /dev/null",
    ]
    _os_system(cmd)

    try:
        _os_system(["mv compiler_profile_0.txt", model + ".compiler_profile_0.txt"])
    except RuntimeError:
        pass


def f32_blobs_compare(a_npz: str, b_npz: str, tolerance: str, excepts=None, show_detail=True):
    cmd = ["npz_tool.py", "compare", a_npz, b_npz, "--tolerance", tolerance]
    if excepts:
        cmd.extend(["--except", excepts])
    if show_detail:
        cmd.append('-vv')
    _os_system(cmd)
