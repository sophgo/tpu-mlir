# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import os
import subprocess
import logging


def _os_system_log(cmd_str):
    # use subprocess to redirect the output stream
    # the file for saving the output stream should be set if using this function
    logging.info("[Running]: %s", cmd_str)

    process = subprocess.Popen(cmd_str,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True)

    while True:
        output = process.stdout.readline().strip()
        if output == '' and process.poll() is not None:
            break
        if output:
            logging.info(output)

    process.wait()
    ret = process.returncode

    if ret == 0:
        logging.info("[Success]: %s", cmd_str)
    else:
        raise RuntimeError("[!Error]: {}".format(cmd_str))


def _os_system(cmd: list, save_log: bool = False):
    cmd_str = ""
    for s in cmd:
        cmd_str += str(s) + " "
    if not save_log:
        print("[Running]: {}".format(cmd_str))
        ret = os.system(cmd_str)
        if ret == 0:
            print("[Success]: {}".format(cmd_str))
        else:
            raise RuntimeError("[!Error]: {}".format(cmd_str))
    else:
        _os_system_log(cmd_str)


def mlir_opt_for_top(mlirfile, opt_mlirfile, add_postprocess=""):
    cmd = ["tpuc-opt", mlirfile, "--shape-infer"]
    if len(add_postprocess) > 0:
        cmd.extend([f"--add-postprocess=\"type={add_postprocess}\""])
    cmd.extend(["--canonicalize", "--extra-optimize", "-o", opt_mlirfile])
    _os_system(cmd)


def mlir_lowering(top_mlir: str,
                  tpu_mlir: str,
                  mode: str,
                  chip: str,
                  cali_table: str = None,
                  asymmetric: bool = False,
                  quantize_table: str = None,
                  customization_format: str = None,
                  fuse_preprocess: bool = False,
                  aligned_input: bool = False,
                  ignore_f16_overflow: bool = False,
                  linear_quant_mode: str = None,
                  do_winograd:bool = False):
    cmd = ["tpuc-opt", top_mlir, "--chip-assign=\"chip={}\"".format(chip.lower())]
    mode = mode.upper()
    asymmetric = False # TODO: always using symmetric, as asymmetric not good
    if cali_table != None:
        cali_param = "--import-calibration-table=\"file={} asymmetric={}\"".format(
            cali_table, asymmetric)
        cmd.extend([cali_param])
    #do extra conversion for differnet chips
    cmd.extend(["--chip-top-optimize"])
    if fuse_preprocess:
        fuse_pre_param = "--fuse-preprocess=\"mode={} customization_format={} align={}\"".format(
            mode, customization_format, aligned_input)
        cmd.extend([fuse_pre_param])
    qtable = ""
    if quantize_table:
        assert (tpu_mlir.endswith(".mlir"))
        weight_name = tpu_mlir[:-len(".mlir")] + "_qtable_weights.npz"
        qtable = "qtable={} weightFileName={}".format(quantize_table, weight_name)
    lower_param = "--convert-top-to-tpu=\"mode={} {} asymmetric={} linear_quant_mode={} doWinograd={} ignore_f16_overflow={}\"".format(
        mode, qtable, asymmetric, linear_quant_mode, do_winograd, ignore_f16_overflow)
    cmd.extend([
        lower_param,
        "--canonicalize",
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
                  disable_layer_group: bool = False,
                  merge_weight: bool = False,
                  op_divide: bool = False,
                  num_device: int = 1,
                  num_core: int = 1,
                  embed_debug_info: bool = False,
                  model_version: str = ""):
    # generate final mlir
    strip_io_quant_param = '--strip-io-quant="quant_input={} quant_output={}"'.format(
        quant_input, quant_output)
    lg_param = ''
    if not disable_layer_group:
        lg_param = '--layer-group="opt=2"'
    subnet_param = '--subnet-divide="dynamic={}"'.format(dynamic)
    address_assign_param = '--address-assign'
    #address_assign_param = '--address-assign="reuse_addr=false"'
    if merge_weight:
        address_assign_param = '--address-assign="merge_weight=true weight_map_file=_weight_map.csv"'
    distribute_param = f"--distribute='num_device={num_device}'"
    parallel_param = f"--parallel='num_core={num_core}'"

    op_divide_param = ""
    if op_divide:
        op_divide_param = "--op-divide"
    cmd = [
        "tpuc-opt",
        tpu_mlir,
        "--mlir-disable-threading",
        strip_io_quant_param,
        "--chip-tpu-optimize",
        distribute_param,
        "--weight-reorder",
        op_divide_param,
        subnet_param,
        "--op-reorder",
        lg_param,
        parallel_param,
        address_assign_param,
        "-o",
        final_mlir,
    ]

    _os_system(cmd)

    # codegen based on final mlir
    codegen_param = (
        f'--codegen="model_file={model} embed_debug_info={str(embed_debug_info).lower()} model_version={str(model_version).lower()}"'
    )
    cmd = [
        "tpuc-opt",
        final_mlir,
        codegen_param,
        "-o /dev/null",
    ]
    _os_system(cmd)

    try:
        if model.endswith(".bmodel"):
            # The suffix of the profile file is not consistent.
            # bm1684 uses ".dat", bm1684x uses ".txt".
            _os_system(["mv compiler_profile_0.[td][xa]t", model + ".compiler_profile_0.txt"])
            _os_system(["mv net_0.profile", model + ".net_0.profile"])
    except RuntimeError:
        pass


def f32_blobs_compare(a_npz: str, b_npz: str, tolerance: str, excepts=None, show_detail=True):
    cmd = ["npz_tool.py", "compare", a_npz, b_npz, "--tolerance", tolerance]
    if excepts:
        cmd.extend(["--except", excepts])
    if show_detail:
        cmd.append('-vv')
    _os_system(cmd)


# TOPTOTOSA
def top_to_tosa(top_mlir: str,
                tosa_mlir: str,
                includeWeight: bool = False):
    cmd = ["tpuc-opt", top_mlir]
    lower_param = "--convert-top-to-tosa=\"includeWeight="
    if includeWeight:
        lower_param += "True\""
    else:
        lower_param += "False\""
    cmd.extend([
        lower_param,
        "--canonicalize",
        "-o",
        tosa_mlir
    ])
    _os_system(cmd)

# TOSATOObj
def tosa_to_llvm(tosa_mlir: str,
                 objfile: str):
    cmd = ["mlir-opt", tosa_mlir]
    lower_param = ("--pass-pipeline=\"builtin.module("
                   "func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-arith, tosa-to-tensor, tosa-to-scf), "
                   "convert-tensor-to-linalg, "
                   "func.func(canonicalize, linalg-bufferize, convert-linalg-to-affine-loops, affine-loop-fusion, affine-simplify-structures, lower-affine), "
                   "func-bufferize, "
                   "func.func(tensor-bufferize, llvm-request-c-wrappers), "
                   "arith-expand, arith-bufferize, normalize-memrefs, convert-scf-to-cf, "
                   "convert-math-to-llvm, convert-arith-to-llvm, convert-func-to-llvm, convert-cf-to-llvm, "
                   "convert-bufferization-to-memref, memref-expand, expand-strided-metadata, finalize-memref-to-llvm, "
                   "canonicalize, llvm-legalize-for-export, reconcile-unrealized-casts)\""
                   "| mlir-translate --mlir-to-llvmir "
                   "| llc -mtriple=x86_64-unknown-linux-gnu --filetype=obj")
    cmd.extend([
        lower_param,
        "-o",
        objfile
    ])
    _os_system(cmd)

# Model inference on CPU
def model_inference_cpu(objfile: str,
                        output_size: str):
    # generate executable file: a.out
    print("Generating executable file a.out ...")
    ccompiler = "clang"
    cfile = "/workspace/tpu-mlir/capi/runtime_cpu.c"
    model = objfile
    lib1 = "/workspace/tpu-mlir/capi/lib/libmlir_c_runner_utils.so.17git"
    lib2 = "/workspace/tpu-mlir/capi/lib/libmlir_runner_utils.so.17git"
    lib3 = "/workspace/tpu-mlir/capi/lib/libmlir_float16_utils.so.17git"
    lib4 = "-lm"
    cflag = "-fPIC"
    cmd = [ccompiler, cfile, model, lib1, lib2, lib3, lib4, cflag]
    _os_system(cmd)
    print("Successfully generate executable file a.out!")
    # execute model inference
    print("Runing ...")
    cmd1 = ["./a.out", output_size]
    _os_system(cmd1)
    print("Inference ends successfully! Results are saved in inference_result.txt.")

# Extra tool: delete file in current directory
def delete_file(file: str):
    cmd = ["rm -f", file]
    _os_system(cmd)
