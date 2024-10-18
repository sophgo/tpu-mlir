# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import shutil
import os
import subprocess
import logging
import utils.pattern_counter
import multiprocessing

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


def _os_system(cmd: list, save_log: bool = False, mute: bool = False, log_level: str = "normal"):
    cmd_str = " ".join(cmd)
    if mute:
        ret = subprocess.call(cmd_str,
                              shell=True,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        assert ret == 0
        return
    if save_log:
        _os_system_log(cmd_str)
    elif log_level == "quiet":
        ret = os.system(cmd_str)
        if ret != 0:
            print("[Failed]: {}".format(cmd_str))
    else:
        print("[Running]: {}".format(cmd_str))
        ret = os.system(cmd_str)
        if ret == 0:
            print("[Success]: {}".format(cmd_str))
        else:
            raise RuntimeError("[!Error]: {}".format(cmd_str))


def get_matched_patterns(log_file: str = ""):
    if log_file:
        matcher = utils.pattern_counter.PatternCounter(log_file)
        matcher.count_matched_patterns()
        return matcher.success_counter
    return {}


def top_opt_options(add_postprocess: str = ""):
    options = ["--shape-infer"]
    if len(add_postprocess) > 0:
        options.extend([f"--add-postprocess=\"type={add_postprocess}\""])
    options.extend(["--canonicalize", "--extra-optimize"])
    return options

def mlir_opt_for_top(mlirfile: str,
                     opt_mlirfile: str,
                     add_postprocess: str = "",
                     count_patterns: bool = False, log_level:str="normal"):
    cmd = ["tpuc-opt", mlirfile]
    options = top_opt_options(add_postprocess)
    cmd.extend(options)
    cmd.extend(["-o", opt_mlirfile])
    log_file = ""
    if count_patterns:
        log_file = "top_patterns.log"
        cmd.extend(["-debug-only=pattern-application,dialect-conversion,greedy-rewriter", "> {} 2>&1".format(log_file)])
    if log_level == "quiet":
        cmd.extend(["> /dev/null"])
    elif log_level == "simple":
        cmd.insert(2, '--init="level=1"')
    _os_system(cmd,log_level=log_level)
    return get_matched_patterns(log_file)

def lowering_options(mode: str,
                     chip: str,
                     num_device: int = 1,
                     num_core: int = 1,
                     cali_table: str = None,
                     asymmetric: bool = False,
                     quantize_table: str = None,
                     weight_name: str = None,
                     customization_format: str = None,
                     fuse_preprocess: bool = False,
                     aligned_input: bool = False,
                     ignore_f16_overflow: bool = False,
                     do_winograd: bool = False,
                     q_group_size: int = 0,
                     addr_mode: str = "auto",
                     matmul_perchannel: bool = False):
    mode = mode.upper()
    options = [
        "--processor-assign=\"chip={} mode={} num_device={} num_core={} addr_mode={}\"".format(
            chip.lower(), mode, num_device, num_core, addr_mode)
    ]

    # asymmetric = False  # TODO: always using symmetric, as asymmetric not good
    if cali_table != None:
        cali_param = "--import-calibration-table=\"file={} asymmetric={}\"".format(
            cali_table, asymmetric)
        options.extend([cali_param])
    #do extra conversion for differnet chips
    options.extend(["--processor-top-optimize"])
    if fuse_preprocess:
        fuse_pre_param = "--fuse-preprocess=\"mode={} customization_format={} align={}\"".format(
            mode, customization_format, aligned_input)
        options.extend([fuse_pre_param])
    qw = "qtable={} weightFileName={}".format(quantize_table, weight_name) if quantize_table else ""
    lower_param = "--convert-top-to-tpu=\"{} asymmetric={} doWinograd={} ignore_f16_overflow={} q_group_size={} matmul_perchannel={}\"".format(
        qw, asymmetric, do_winograd, ignore_f16_overflow, q_group_size, matmul_perchannel)
    options.extend([
        lower_param,
        "--canonicalize",
        "--weight-fold",
    ])
    return options

def mlir_lowering(top_mlir: str,
                  tpu_mlir: str,
                  mode: str,
                  chip: str,
                  num_device: int = 1,
                  num_core: int = 1,
                  cali_table: str = None,
                  asymmetric: bool = False,
                  quantize_table: str = None,
                  customization_format: str = None,
                  fuse_preprocess: bool = False,
                  aligned_input: bool = False,
                  ignore_f16_overflow: bool = False,
                  do_winograd: bool = False,
                  q_group_size: int = 0,
                  count_patterns: bool = False,
                  addr_mode: str = "auto",
                  mute: bool = False,
                  log_level: str = "normal",
                  matmul_perchannel: bool = False):
    mode = mode.upper()
    cmd = ["tpuc-opt", top_mlir]
    weight_name = ""
    if quantize_table:
        assert (tpu_mlir.endswith(".mlir"))
        weight_name = tpu_mlir[:-len(".mlir")] + "_qtable_weights.npz"
    options =  lowering_options(mode,
                                chip,
                                num_device,
                                num_core,
                                cali_table,
                                asymmetric,
                                quantize_table,
                                weight_name,
                                customization_format,
                                fuse_preprocess,
                                aligned_input,
                                ignore_f16_overflow,
                                do_winograd,
                                q_group_size,
                                addr_mode,
                                matmul_perchannel)
    cmd.extend(options)
    cmd.extend(["-o", tpu_mlir])
    log_file = ""
    if count_patterns:
        log_file = "tpu_patterns.log"
        cmd.extend(["--debug-only=pattern-application,dialect-conversion,greedy-rewriter", "> {} 2>&1".format(log_file)])
    if log_level == "quiet":
        cmd.extend(["> /dev/null"])
    elif log_level == "only-layer-group":
        cmd.extend(["--debug-only=layer-group,LayerGroupUtil"])
        cmd.insert(2, '--init="level=2"')
    elif log_level == "simple":
        cmd.insert(2, '--init="level=1"')
    _os_system(cmd, mute=mute,log_level=log_level)
    return get_matched_patterns(log_file)

def tpu_opt_options(quant_input: bool = False,
                    quant_output: bool = False,
                    quant_input_list: str = "",
                    quant_output_list: str = "",
                    mlir_disable_threading: bool = True) :
    # generate final mlir
    strip_io_quant_param = '--strip-io-quant="quant_input={} quant_output={} quant_input_list={} quant_output_list={}"'.format(
        quant_input, quant_output, quant_input_list, quant_output_list)
    # yapf: disable
    options = []
    if mlir_disable_threading:
        options.extend(["--mlir-disable-threading"])
    options.extend([
        strip_io_quant_param,
        "--processor-tpu-optimize",
    ])
    return options

def tpu_ada_options(dynamic: bool = False,
                    disable_layer_group: bool = False,
                    opt: int = 2,
                    merge_weight: bool = False,
                    op_divide: bool = False,
                    group_by_cores: str = "auto",
                    compress_mode: str = "none",
                    future_update_rank: int = 0,
                    future_update_list: str = ""):
    lg_param = ''
    if not disable_layer_group:
        lg_param = '--layer-group="opt={} group_by_cores={} compress_mode={}"'.format(
            opt, group_by_cores, compress_mode)
    subnet_param = '--subnet-divide="dynamic={}"'.format(dynamic)
    address_assign_param = '--address-assign'
    if merge_weight:
        address_assign_param = '--address-assign="merge_weight=true weight_map_file=_weight_map.csv"'
    distribute_param = f"--dev-parallel"
    parallel_param = f"--core-parallel"
    future_update_param = '--future-update="rank={} weight_list={}"'.format(future_update_rank, future_update_list)

    op_divide_param = ""
    if op_divide:
        op_divide_param = "--op-divide"
    options = [
        distribute_param,
        "--weight-reorder",
        op_divide_param,
        subnet_param,
        "--op-reorder",
        future_update_param,
        lg_param,
        parallel_param,
        address_assign_param
    ]
    return options

def codegen_options(model: str,
                    embed_debug_info: bool = False,
                    model_version: str = "",
                    bmodel_only: bool = False):
    options = [
        '--codegen="model_file={} embed_debug_info={} model_version={} bmodel_only={}"'.format(
          model, str(embed_debug_info).capitalize(), str(model_version).lower(), str(bmodel_only).capitalize())
    ]
    return options

## ========================================
## build ppl src code
ppl_lock = multiprocessing.Lock()

def get_latest_file_mtime(directory):
    latest_mtime = 0

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            mtime = os.path.getmtime(file_path)
            if mtime > latest_mtime:
                latest_mtime = mtime

    return latest_mtime

def build_ppl(
    mute: bool = False,
    log_level: str = "normal",
):
    with ppl_lock:
        ori_dir = os.getcwd()
        tpu_path = os.environ.get("TPUC_ROOT")
        if tpu_path is None:
            raise RuntimeError("[!Error]: can't find TPUC_ROOT")
        ppl_path = os.path.join(tpu_path, 'PplBackend')
        ppl_so_path = os.path.join(tpu_path, 'lib/libppl_host.so')
        if os.path.isfile(ppl_so_path):
            # if no modify, return directly
            ppl_so_mtime = os.path.getmtime(ppl_so_path)
            ppl_src_dir = os.path.join(ppl_path, 'src')
            ppl_src_mtime = get_latest_file_mtime(ppl_src_dir)
            if ppl_src_mtime < ppl_so_mtime:
                return
        try:
            cmd = ["bash", os.path.join(ppl_path, "build.sh")]
            _os_system(cmd, mute=mute, log_level=log_level)
        except RuntimeError:
            raise RuntimeError("[!Error]: build ppl src code failed")
        finally:
            os.chdir(ori_dir)


## ========================================

def mlir_to_model(tpu_mlir: str,
                  model: str,
                  final_mlir: str,
                  dynamic: bool = False,
                  quant_input: bool = False,
                  quant_output: bool = False,
                  quant_input_list: str = "",
                  quant_output_list: str = "",
                  disable_layer_group: bool = False,
                  opt: int = 2,
                  merge_weight: bool = False,
                  op_divide: bool = False,
                  embed_debug_info: bool = False,
                  group_by_cores: str = "auto",
                  model_version: str = "",
                  count_patterns: bool = False,
                  compress_mode: str = "none",
                  future_update_rank: int = 0,
                  future_update_list: str = "",
                  debug_cmd: str = "",
                  log_level:str = "normal"):
    cmd = ["tpuc-opt", tpu_mlir]
    options = tpu_opt_options(quant_input,
                              quant_output,
                              quant_input_list,
                              quant_output_list)
    cmd.extend(options)

    # yapf: enable

    if embed_debug_info:
        tpu_opt_mlir = final_mlir[:-10] + "tpu_opt.mlir"
        # save the optimized tpu.mlir
        cmd.extend([
            "-o",
            tpu_opt_mlir,
            debug_cmd
        ])
        _os_system(cmd,log_level=log_level)
        cmd = [
            "tpuc-opt",
            tpu_opt_mlir
        ]

    options = tpu_ada_options(dynamic,
                              disable_layer_group,
                              opt,
                              merge_weight,
                              op_divide,
                              group_by_cores,
                              compress_mode,
                              future_update_rank,
                              future_update_list)
    cmd.extend(options)
    cmd.extend([
        "-o",
        final_mlir,
        debug_cmd
    ])
    log_file = ""
    if count_patterns:
        log_file = "tpu_patterns.log"
        cmd.extend(["-debug-only=pattern-application,dialect-conversion,greedy-rewriter", "> {} 2>&1".format(log_file)])
    if log_level == "quiet":
        cmd.extend(["> /dev/null"])
    elif log_level == "only-layer-group":
        cmd.extend(["--debug-only=layer-group,LayerGroupUtil"])
        cmd.insert(2, '--init="level=2"')
    elif log_level == "simple":
        cmd.insert(2, '--init="level=1"')
    _os_system(cmd,log_level=log_level)

    # compile ppl code
    build_ppl()

    # codegen based on final mlir
    cmd = ["tpuc-opt", final_mlir]
    options = codegen_options(model,
                              embed_debug_info,
                              model_version)
    cmd.extend(options)
    cmd.extend(["-o /dev/null"])
    _os_system(cmd,log_level=log_level)

    out_dir = model.rsplit(".", maxsplit=1)[0]
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(final_mlir, os.path.join(out_dir, 'final.mlir'))
    try:
        if model.endswith(".bmodel") and not dynamic:
            # The suffix of the profile file is not consistent.
            # bm1684 uses ".dat", bm1684x uses ".txt".
            _os_system(["mv compiler_profile_0.[td][xa]t", model + ".compiler_profile_0.txt"], log_level=log_level)
            _os_system(["mv net_0.profile", model + ".net_0.profile"],log_level=log_level)
    except RuntimeError:
        pass

    return get_matched_patterns(log_file)


def origin_mlir_txt_to_bmodel(converter,
                              model_name: str,
                              mode: str,
                              chip: str,
                              add_postprocess: str = "",
                              num_device: int = 1,
                              num_core: int = 1,
                              cali_table: str = None,
                              asymmetric: bool = False,
                              quantize_table: str = None,
                              customization_format: str = None,
                              fuse_preprocess: bool = False,
                              aligned_input: bool = False,
                              ignore_f16_overflow: bool = False,
                              do_winograd: bool = False,
                              q_group_size: int = 0,
                              dynamic: bool = False,
                              quant_input: bool = False,
                              quant_output: bool = False,
                              quant_input_list: str = "",
                              quant_output_list: str = "",
                              disable_layer_group: bool = False,
                              opt: int = 2,
                              merge_weight: bool = False,
                              op_divide: bool = False,
                              embed_debug_info: bool = False,
                              addr_mode: str = "auto",
                              group_by_cores: str = "auto",
                              model_version: str = "",
                              count_patterns: bool = False,
                              compress_mode: str = "none",
                              log_level: str = 'normal',
                              future_update_rank: int = 0,
                              future_update_list: str = "",
                              matmul_perchannel: bool = False):

    options = []
    new_options = top_opt_options(add_postprocess)
    options.extend(new_options)
    new_options =  lowering_options(mode,
                                    chip,
                                    num_device,
                                    num_core,
                                    cali_table,
                                    asymmetric,
                                    quantize_table,
                                    None,
                                    customization_format,
                                    fuse_preprocess,
                                    aligned_input,
                                    ignore_f16_overflow,
                                    do_winograd,
                                    q_group_size,
                                    addr_mode,
                                    matmul_perchannel)
    options.extend(new_options)
    new_options =   tpu_opt_options(quant_input,
                                    quant_output,
                                    quant_input_list,
                                    quant_output_list,
                                    False)
    options.extend(new_options)
    new_options =   tpu_ada_options(dynamic,
                                    disable_layer_group,
                                    opt,
                                    merge_weight,
                                    op_divide,
                                    group_by_cores,
                                    compress_mode,
                                    future_update_rank,
                                    future_update_list)
    options.extend(new_options)
    new_options = codegen_options(f"{model_name}_{mode}.bmodel",
                                  embed_debug_info,
                                  model_version,
                                  True)
    options.extend(new_options)
    options.extend(['--deinit="no_save_weight=True"'])

    log_file = ""
    if count_patterns:
        log_file = "tpu_patterns.log"
        options.extend(["-debug-only=pattern-application,dialect-conversion,greedy-rewriter"])

    import pymlir
    import sys
    mlir_txt = converter.get_mlir_txt()
    weight_option = "weight_in_mem=True"
    if log_level == "quiet":
        options.insert(0, f'--init="{weight_option}"')
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
        try:
            pymlir.run_pass_pipeline(mlir_txt, options)
        finally:
            os.dup2(sys.__stdout__.fileno(), sys.stdout.fileno())
            os.dup2(sys.__stderr__.fileno(), sys.stderr.fileno())
    else:
        if log_level == "simple":
            options = [opt for opt in options if not opt.startswith('--init')]
            options.insert(0, f'--init="{weight_option} level=1"')
        elif log_level == "only-layer-group":
            options = [opt for opt in options if not opt.startswith('--init')]
            options.insert(0, f'--init="{weight_option} level=2"')
            # pymlir.debug(["layer-group","LayerGroupUtil"]) #todo
        else:
            options.insert(0, f'--init="{weight_option}"')
        print("options: ", options)
        pymlir.run_pass_pipeline(mlir_txt, options)
    return get_matched_patterns(log_file)


def f32_blobs_compare(a_npz: str, b_npz: str, tolerance: str, excepts=None, show_detail=True, fuzzy_match=False, log_level="normal"):
    cmd = ["npz_tool.py", "compare", a_npz, b_npz, "--tolerance", tolerance]
    if excepts:
        cmd.extend(["--except", excepts])
    if show_detail:
        cmd.append('-vv')
    if fuzzy_match:
        cmd.append('--fuzzy_match')
    _os_system(cmd, log_level=log_level)


# TOPTOTOSA
def top_to_tosa(top_mlir: str, tosa_mlir: str, includeWeight: bool = False):
    cmd = ["tpuc-opt", top_mlir]
    lower_param = "--convert-top-to-tosa=\"includeWeight="
    if includeWeight:
        lower_param += "True\""
    else:
        lower_param += "False\""
    cmd.extend([lower_param, "--canonicalize", "-o", tosa_mlir])
    _os_system(cmd)


# TOSATOObj
def tosa_to_llvm(tosa_mlir: str, objfile: str):
    cmd = ["mlir-opt", tosa_mlir]
    lower_param = (
        "--pass-pipeline=\"builtin.module("
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
    cmd.extend([lower_param, "-o", objfile])
    _os_system(cmd)


# Model inference on CPU
def model_inference_cpu(objfile: str, output_size: str, log_level:str = "normal"):
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
    _os_system(cmd,log_level=log_level)
    print("Successfully generate executable file a.out!")
    # execute model inference
    print("Runing ...")
    cmd1 = ["./a.out", output_size]
    _os_system(cmd1,log_level=log_level)
    print("Inference ends successfully! Results are saved in inference_result.txt.")


# Extra tool: delete file in current directory
def delete_file(file: str):
    cmd = ["rm -f", file]
    _os_system(cmd)

def mlir2onnx(mlir_file: str, onnx_file: str):
    cmd = ['mlir2onnx.py -m', mlir_file, '-o', onnx_file]
    _os_system(cmd)
