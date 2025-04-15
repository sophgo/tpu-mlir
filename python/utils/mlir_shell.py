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
import traceback
import sys

stdout = sys.stdout
stderr = sys.stderr
import json
import textwrap
import shutil
import tempfile
from typing import Tuple
from subprocess import run, CalledProcessError, TimeoutExpired
import os
import sys
# from typing import TextIO
# from utils.tpuc_cmd_builder import TpucCommandBuilder


def env2bashstr(env: dict, auto_pick=True):
    ret = []

    def pick():
        if not auto_pick:
            return True

        if kk.endswith("PATH"):
            return True
        if kk.endswith("ROOT"):
            return True

        return kk in {
            "PWD",
            "GIT_SSH_COMMAND",
            "UPGRADE_BRANCH",
            "HOST_IP",
            "REDIS_OM_URL",
        }

    for kk, vv in env.items():
        if not pick():
            continue

        if kk.startswith("#"):
            ret.append(f"# {kk}={vv}")
        else:
            ret.append(f'export {kk}="{vv}"')
    return "\n".join(ret)


from io import StringIO


def is_logger_configured(logger_name):
    logger = logging.getLogger(logger_name)
    return len(logger.handlers) > 0


def getstatusoutput_v2(
    cmd,
    cwd=None,
    env=None,
    timeout=None,
    check=False,
    shell=False,
    print_output=False,
    **kwargs,
) -> Tuple[int, str]:
    if is_logger_configured("logger_to_file") and is_logger_configured("logger_to_file.console"):
        # for handler in logging.getLogger("logger_to_file").handlers:
        #     if isinstance(handler, logging.FileHandler):
        #         print(f"FileHandler found: {handler.baseFilename}")
        #         print(f"Level: {handler.level}")
        #         print(f"Formatter: {handler.formatter._fmt}")
        #         print(f"Mode: {handler.mode}")
        #         print(f"Encoding: {handler.encoding}")

        return getstatusoutput_v2_split_log(
            cmd,
            cwd=cwd,
            env=env,
            timeout=timeout,
            check=check,
            shell=shell,
            **kwargs,
        )
    else:
        return getstatusoutput_v2_without_split_log(
            cmd,
            cwd=cwd,
            env=env,
            timeout=timeout,
            check=check,
            shell=shell,
            print_output=print_output,
            **kwargs,
        )


def getstatusoutput_v2_without_split_log(
    cmd,
    cwd=None,
    env=None,
    timeout=None,
    check=False,
    shell=False,
    redirect_output=None,
    redirect_error=None,
    print_output=False,
    **kwargs,
) -> Tuple[int, str]:
    """Return (exitcode, output) of executing cmd in a shell."""
    if env is None:
        env = os.environ.copy()
    # kwargs = {}
    if cwd is not None:
        assert os.path.isdir(cwd), cwd
        kwargs["cwd"] = cwd
    if env is not None:
        assert isinstance(env, dict)
        if "PWD" in env:
            env.pop("PWD")
        kwargs["env"] = env

    logf = StringIO()
    # logfn = get_logfilename()
    logger = logging.root
    ex = None

    if redirect_output:
        temp_output = redirect_output
    else:
        temp_output = tempfile.NamedTemporaryFile("w+", delete=False)

    if redirect_error:
        temp_error = redirect_error
    else:
        temp_error = tempfile.NamedTemporaryFile("w+", delete=False)

    logger.debug(f"running command: {cmd}")
    logger.debug(f"you can review middle output in {temp_output.name}")
    # if env
    try:
        cmd_str = cmd if shell else " ".join(cmd)
        logf.write(f" cwd: {cwd}\n")
        logf.write(f" -> Executing {cmd_str}\n")
        logf.write(f"========Command Output Start=========\n")
        logf.flush()

        run(
            cmd,
            shell=shell,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE if print_output else temp_output,
            stderr=subprocess.PIPE if print_output else temp_error,
            timeout=timeout,
            check=True,
            # universal_newlines=True,
            bufsize=0,
            text=True,
            **kwargs,
        )
        exitcode = 0
        if redirect_output is None:
            temp_output.seek(0)
            shutil.copyfileobj(temp_output, logf)

        logf.write(f"========Command Output End=========\n")
        logf.write(f" -> [Success]\n")

    except (CalledProcessError, TimeoutExpired) as e:
        temp_output.seek(0)
        logf.write(textwrap.indent(temp_output.read(), f"{str(type(e).__name__).upper()}!! "))
        logf.write(f"========Command Output End=========\n")
        logf.write(f" -> [Failed in Command]\n")
        ex = e
        if isinstance(e, TimeoutExpired):
            exitcode = 110
        else:
            exitcode = e.returncode
        logf.write(f"# cwd\n")
        logf.write(f"cd {cwd}\n\n")

        logf.write(f"# Environments: \n")
        logf.write(textwrap.indent(json.dumps(env, indent=2), "# ") + "\n")
        logf.write(env2bashstr(env, auto_pick=False))

        logf.write("# command: \n")
        cmd_opt = " ".join(cmd) if isinstance(cmd, list) else cmd
        logf.write(f"{cmd_opt}\n")
        logf.write(traceback.format_exc())
        logf.write(f"======traceback info end======\n")
    except Exception as e:
        if redirect_output is None:
            temp_output.seek(0)
            logf.write(textwrap.indent(temp_output.read(), f"{str(type(e).__name__).upper()}!! "))
        logf.write(f"========Command Output End=========\n")
        logf.write(f" -> [Failed Outside Command]\n")
        ex = e
        exitcode = -256
        logf.write(f"cmd={cmd}\n")
        logf.write(f"cwd={cwd}\n")
        logf.write(f"env={env}\n")
        logf.write(traceback.format_exc())
        logf.write(f"======traceback info end======\n")
        traceback.print_exc()

    output = ""
    if redirect_output is None:
        temp_output.seek(0)
        output = temp_output.read()
        temp_output.close()

    if print_output:
        logger.info(output)

    try:
        if redirect_output is None:
            os.remove(temp_output.name)
    except Exception as e:
        logger.error(f"remove temp log file {temp_output.name} failed: {e}")

    if check and ex:
        raise RuntimeError(f"{output}\n[Failed] {cmd}")
    elif ex:
        logger.error(output)
        logger.error(f"[Failed] {cmd}")
    else:
        logger.debug(f"[Success]: {cmd}")
    return exitcode, output


def getstatusoutput_v2_split_log(
    cmd,
    cwd=None,
    env=None,
    timeout=None,
    check=False,
    shell=False,
    **kwargs,
) -> Tuple[int, str]:
    """Return (exitcode, output) of executing cmd in a shell."""
    if env is None:
        env = os.environ.copy()
    # kwargs = {}
    if cwd is not None:
        assert os.path.isdir(cwd), cwd
        kwargs["cwd"] = cwd
    if env is not None:
        assert isinstance(env, dict)
        if "PWD" in env:
            env.pop("PWD")
        kwargs["env"] = env

    file_logger = logging.getLogger("logger_to_file")
    console_logger = logging.getLogger("logger_to_file.console")
    file_logger.setLevel(logging.DEBUG)  # do not delete this
    ex = None

    temp_logf = tempfile.NamedTemporaryFile("w+", delete=False)

    # if env
    try:
        cmd_str = cmd if shell else " ".join(cmd)
        console_logger.info(f" cwd: {cwd}\n")
        console_logger.info(f" -> Executing {cmd_str}\n")
        file_logger.debug(f"========Command Output Start=========\n")
        run(
            cmd,
            shell=shell,
            stdout=temp_logf,
            stderr=temp_logf,
            timeout=timeout,
            check=True,
            # universal_newlines=True,
            bufsize=0,
            text=True,
            **kwargs,
        )
        exitcode = 0
        temp_logf.seek(0)
        file_logger.debug(temp_logf.read())
        file_logger.debug(f"========Command Output End=========\n")
        console_logger.info(f" -> [Success]\n")

    except (CalledProcessError, TimeoutExpired) as e:
        temp_logf.seek(0)
        file_logger.debug(textwrap.indent(temp_logf.read(), f"{str(type(e).__name__).upper()}!!"))
        file_logger.debug(f"========Command Output End=========\n")
        console_logger.info(f" -> [Failed in Command]\n")
        ex = e
        if isinstance(e, TimeoutExpired):
            exitcode = 110
        else:
            exitcode = e.returncode
        file_logger.debug(f"# cwd\n")
        file_logger.debug(f"cd {cwd}\n\n")

        file_logger.debug(f"# Environments: \n")
        file_logger.debug(textwrap.indent(json.dumps(env, indent=2), "# ") + "\n")
        file_logger.debug(env2bashstr(env, auto_pick=False))

        file_logger.debug("# command: \n")
        cmd_opt = " ".join(cmd) if isinstance(cmd, list) else cmd
        file_logger.debug(f"{cmd_opt}\n")
        console_logger.info(traceback.format_exc())
        console_logger.info(f"======traceback info end======\n")
    except Exception as e:
        temp_logf.seek(0)
        file_logger.debug(textwrap.indent(temp_logf.read(), f"{str(type(e).__name__).upper()}!! "))
        file_logger.debug(f"========Command Output End=========\n")
        console_logger.info(f" -> [Failed Outside Command]\n")
        ex = e
        exitcode = -256
        file_logger.debug(f"cmd={cmd}\n")
        file_logger.debug(f"cwd={cwd}\n")
        file_logger.debug(f"env={env}\n")
        console_logger.info(traceback.format_exc())
        console_logger.info(f"======traceback info end======\n")
    temp_logf.seek(0)
    output = temp_logf.read()
    temp_logf.close()
    try:
        os.remove(temp_logf.name)
    except Exception as e:
        console_logger.error(f"remove temp log file {temp_logf.name} failed: {e}")

    if check and ex:
        raise RuntimeError(f"{output}\n[Failed] {cmd}")
    elif ex:
        console_logger.error(output)
        console_logger.error(f"[Failed] {cmd}")
    else:
        console_logger.debug(f"[Success]: {cmd}")
    return exitcode, output


def _os_system_log(cmd_str, cwd=None):
    # use subprocess to redirect the output stream
    # the file for saving the output stream should be set if using this function
    logging.info("[Running]: %s", cmd_str)

    return getstatusoutput_v2(cmd_str, cwd=cwd, shell=True, check=True, print_output=True)
    subprocess.run(
        cmd_str,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=cwd,
    )

    # while True:
    #     output = process.stdout.readline().strip()

    #     if output:
    #         logging.info(output)

    #     if output == '' and process.poll() is not None:
    #         break

    # while True:
    #     error = process.stderr.readline().strip()
    #     if error:
    #         logging.error(error)
    #     else:
    #         break

    # process.wait()
    # ret = process.returncode

    if ret == 0:
        logging.info("[Success]: %s", cmd_str)
    else:
        logging.error(cmd_str)
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


def top_opt_options(add_postprocess: str = "", pruning: str = ""):
    options = ["--shape-infer"]
    if len(pruning) > 0:
        options = [f"--pruning=\"config={pruning}\"", "--shape-infer"]
    if len(add_postprocess) > 0:
        options.extend([f"--add-postprocess=\"type={add_postprocess}\""])
    options.extend(["--canonicalize", "--extra-optimize"])
    return options


def mlir_opt_for_top(mlirfile: str,
                     opt_mlirfile: str,
                     add_postprocess: str = "",
                     count_patterns: bool = False,
                     log_level: str = "normal",
                     pruning: str = ""):
    cmd = ["tpuc-opt", mlirfile]
    options = top_opt_options(add_postprocess, pruning)
    cmd.extend(options)
    cmd.extend(["-o", opt_mlirfile])
    log_file = ""
    if count_patterns:
        log_file = "top_patterns.log"
        cmd.extend([
            "-debug-only=pattern-application,dialect-conversion,greedy-rewriter",
            "> {} 2>&1".format(log_file)
        ])
    if log_level == "quiet":
        cmd.extend(["> /dev/null"])
    elif log_level == "simple":
        cmd.insert(2, '--init="level=1"')
    _os_system(cmd, log_level=log_level)
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
                     high_precision: bool = False,
                     do_winograd: bool = False,
                     q_group_size: int = 0,
                     q_symmetric: bool = False,
                     addr_mode: str = "auto",
                     matmul_perchannel: bool = False,
                     gelu_mode: str = "normal"):
    mode = mode.upper()
    options = [
        "--processor-assign=\"chip={} mode={} num_device={} num_core={} addr_mode={} high_precision={}\""
        .format(chip.lower(), mode, num_device, num_core, addr_mode, high_precision)
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
    lower_param = "--convert-top-to-tpu=\"{} asymmetric={} doWinograd={}" \
                  " q_group_size={} q_symmetric={} matmul_perchannel={} gelu_mode={}\"".format(
        qw, asymmetric, do_winograd, q_group_size, q_symmetric, matmul_perchannel, gelu_mode)
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
                  high_precision: bool = False,
                  do_winograd: bool = False,
                  q_group_size: int = 0,
                  q_symmetric: bool = False,
                  count_patterns: bool = False,
                  addr_mode: str = "auto",
                  mute: bool = False,
                  log_level: str = "normal",
                  matmul_perchannel: bool = False,
                  gelu_mode: str = "normal"):
    mode = mode.upper()
    cmd = ["tpuc-opt", top_mlir]
    weight_name = ""
    if quantize_table:
        assert (tpu_mlir.endswith(".mlir"))
        weight_name = tpu_mlir[:-len(".mlir")] + "_qtable_weights.npz"
    options = lowering_options(mode, chip, num_device, num_core, cali_table, asymmetric,
                               quantize_table, weight_name, customization_format, fuse_preprocess,
                               aligned_input, high_precision, do_winograd, q_group_size,
                               q_symmetric, addr_mode, matmul_perchannel, gelu_mode)
    cmd.extend(options)
    cmd.extend(["-o", tpu_mlir])
    log_file = ""
    if count_patterns:
        log_file = "tpu_patterns.log"
        cmd.extend([
            "--debug-only=pattern-application,dialect-conversion,greedy-rewriter",
            "> {} 2>&1".format(log_file)
        ])
    if log_level == "quiet":
        cmd.extend(["> /dev/null"])
    elif log_level == "only-layer-group":
        cmd.extend(["--debug-only=layer-group,LayerGroupUtil"])
        cmd.insert(2, '--init="level=2"')
    elif log_level == "simple":
        cmd.insert(2, '--init="level=1"')
    _os_system(cmd, mute=mute, log_level=log_level)
    return get_matched_patterns(log_file)


def tpu_opt_options(quant_input: bool = False,
                    quant_output: bool = False,
                    quant_input_list: str = "",
                    quant_output_list: str = "",
                    mlir_disable_threading: bool = True,
                    quant_output_bf16: bool = False):
    # generate final mlir
    strip_io_quant_param = '--strip-io-quant="quant_input={} quant_output={} quant_input_list={} quant_output_list={} quant_output_bf16={}"'.format(
        quant_input, quant_output, quant_input_list, quant_output_list, quant_output_bf16)
    # yapf: disable
    options = []
    if mlir_disable_threading:
        options.extend(["--mlir-disable-threading"])
    options.extend([
        strip_io_quant_param,
        "--processor-tpu-optimize",
    ])
    return options


def tpu_ada_options(
    *,
    dynamic: bool = False,
    disable_layer_group: bool = False,
    opt: int = 2,
    merge_weight: bool = False,
    op_divide: bool = False,
    group_by_cores: str = "auto",
    compress_mode: str = "none",
    future_update_rank: int = 0,
    future_update_list: str = "",
    trunc_final: list = None,
    opt_post_processor: bool = False,
    lg_debugger: bool = False,
    disable_group_overlap: bool = False,
):
    lg_param = ''
    disable_group_overlap = "true" if disable_group_overlap else "false"
    if not disable_layer_group:
        debugger = 1 if lg_debugger else 0
        lg_param = '--layer-group="opt={} group_by_cores={} compress_mode={} debugger={} disable_group_overlap={}"'.format(
            opt, group_by_cores, compress_mode, debugger, disable_group_overlap)
    subnet_param = '--subnet-divide="dynamic={}"'.format(dynamic)
    address_assign_param = '--address-assign'
    if merge_weight:
        address_assign_param = '--address-assign="merge_weight=true weight_map_file=_weight_map.csv"'

    trunc_param = ""
    if trunc_final:
        loc_param = ','.join(trunc_final)
        trunc_param += f"--trunc-layer=\"cutLocs={loc_param}\""

    distribute_param = f"--dev-parallel"
    parallel_param = f"--core-parallel"
    future_update_param = '--future-update="rank={} weight_list={}"'.format(future_update_rank, future_update_list)

    op_divide_param = ""
    if op_divide:
        op_divide_param = "--op-divide"
    opt_post_processor_param = "--opt-post-processor" if opt_post_processor else ""
    options = [
        distribute_param,
        "--weight-reorder",
        op_divide_param,
        subnet_param,
        "--op-reorder",
        future_update_param,
        lg_param,
        trunc_param,
        parallel_param,
        opt_post_processor_param,
        address_assign_param
    ]
    return options

def time_fixed_subnet_options(time_fixed_subnet, subnet_params, layer_group_cache):
    all_layers = []
    layer_group_cache_path = layer_group_cache
    with open(layer_group_cache_path, 'r') as f:
        layer_group_cache = json.load(f)
        for group in (*layer_group_cache["GroupLayer"], *layer_group_cache["GlobalLayer"]):
            all_layers.append({
                "index": group["index"],
                "group_cost": group["group_cost"],
                "locs": group.get("locs", group.get("loc")),
            })

    all_layers = sorted(all_layers, key=lambda x: x['index'])
    subnets = {}
    subnet_index = 0
    if time_fixed_subnet == "normal":
        max_exceed = max(layer['group_cost'] for layer in all_layers if layer['group_cost'] > 0)
        current_subnet = []
        current_sum = 0
        for element in all_layers:
            element_cost = element['group_cost']
            if current_sum + element_cost <= max_exceed:
                current_subnet.append(element)
                current_sum += element_cost
            else:
                _save_subnet(subnets, current_subnet, subnet_index)
                subnet_index += 1
                current_subnet = [element]
                current_sum = element_cost
        if current_subnet:
            _save_subnet(subnets, current_subnet, subnet_index)
    elif time_fixed_subnet == "limit":
        i = 0
        n = len(all_layers)
        while i < n:
            if all_layers[i]['group_cost'] > 0:
                _save_subnet(subnets, [all_layers[i]], subnet_index)
                subnet_index += 1
                i += 1
            else:
                zero_start = i
                while i < n and all_layers[i]['group_cost'] == 0:
                    i += 1
                if i < n:
                    subnet_layers = all_layers[zero_start:i+1]
                    _save_subnet(subnets, subnet_layers, subnet_index)
                    subnet_index += 1
                    i += 1
                else:
                    if subnets:
                        last_key = f'subfunc_{subnet_index-1}'
                        last_subnet = subnets[last_key]
                        last_subnet['end_index'] = all_layers[-1]['index']
                        if all_layers[-1]['locs']:
                            locs = all_layers[-1]['locs']
                            current = locs
                            while isinstance(current, list) and len(current) > 0:
                                current = current[-1]
                            last_subnet['last_loc'] = str(current) if not isinstance(current, list) else current
                    else:
                        _save_subnet(subnets, all_layers[zero_start:], subnet_index)
    elif time_fixed_subnet == "custom":
        try:
            params = subnet_params.split(',')
            if len(params) != 2:
                raise ValueError
            frequency, duration = map(float, params)
            if frequency <= 0 or duration <= 0:
                raise ValueError
        except ValueError:
            raise ValueError(f"Invalid subnet_params format: {subnet_params}. "
                             "Expected 'frequency,duration' (e.g. '1000,30')") from None
        cost_threshold = frequency * duration * 1e3
        current_subnet = []
        current_cost = 0
        for layer in all_layers:
            layer_cost = layer['group_cost']
            if layer_cost > cost_threshold:
                if current_subnet:
                    _save_subnet(subnets, current_subnet, subnet_index)
                    subnet_index += 1
                    current_subnet = []
                    current_cost = 0
                _save_subnet(subnets, [layer], subnet_index)
                subnet_index += 1
            elif current_cost + layer_cost <= cost_threshold:
                current_subnet.append(layer)
                current_cost += layer_cost
            else:
                if current_subnet:
                    _save_subnet(subnets, current_subnet, subnet_index)
                    subnet_index += 1
                current_subnet = [layer]
                current_cost = layer_cost
        if current_subnet:
            _save_subnet(subnets, current_subnet, subnet_index)
    else:
        raise ValueError(f"Unsupported time_fixed_subnet mode: {time_fixed_subnet}")

    output_json = {
        "subfuncs": [
            {
                "subfunc_id": key,
                "start_index": value["start_index"],
                "end_index": value["end_index"],
                "group_cost": value["group_cost"],
                "last_loc": value["last_loc"]
            }
            for key, value in subnets.items()
        ]
    }

    time_fixed_subnet_path = layer_group_cache_path.replace('.layer_group_cache.json', '.subnets.json')
    with open(time_fixed_subnet_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
    return [
        '--time-fixed-subnet="json_file={}"'.format(time_fixed_subnet_path)
    ]

def _save_subnet(subnets, current_subnet, subnet_index):
    if not current_subnet:
        return
    last_loc = None
    if current_subnet and current_subnet[-1]['locs']:
        locs = current_subnet[-1]['locs']
        current = locs
        while isinstance(current, list) and len(current) > 0:
            current = current[-1]
        last_loc = str(current) if not isinstance(current, list) else current

    subnets[f'subfunc_{subnet_index}'] = {
        'start_index': current_subnet[0]['index'],
        'end_index': current_subnet[-1]['index'],
        'group_cost': sum(layer['group_cost'] for layer in current_subnet),
        'last_loc': last_loc
    }

def codegen_options(model: str,
                    embed_debug_info: bool = False,
                    model_version: str = "",
                    bmodel_only: bool = False,
                    gdma_check: bool = True):
    options = [
        '--codegen="model_file={} embed_debug_info={} model_version={} bmodel_only={} gdma_check={}"'.format(
          model, str(embed_debug_info).capitalize(), str(model_version).lower(), str(bmodel_only).capitalize(), str(gdma_check).capitalize())
    ]
    return options

## ========================================
## build ppl src code
## ========================================
''' if build ppl
import multiprocessing
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

'''
## ========================================


def mlir_to_model(
    *,
    tpu_mlir: str,
    bmodel_path: str,
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
    debug_info: str = "",
    log_level: str = "normal",
    trunc_final: list = None,
    command_mem: dict = None,
    quant_output_bf16: bool = False,
    opt_post_processor: bool = False,
    gdma_check: bool = True,
    lg_debugger: bool = False,
    time_fixed_subnet: str = None,
    subnet_params: str = None,
    layer_group_cache: str = ""
):
    if command_mem is None:
        command_mem = {}
    cmd = ["tpuc-opt", tpu_mlir]
    debug_cmd = f"--debug_cmd={debug_info}"
    options = tpu_opt_options(
        quant_input, quant_output, quant_input_list, quant_output_list, quant_output_bf16=quant_output_bf16
    )
    cmd.extend(options)

    # yapf: enable

    if embed_debug_info:
        tpu_opt_mlir = final_mlir[:-10] + "tpu_opt.mlir"
        # save the optimized tpu.mlir
        cmd.extend(["-o", tpu_opt_mlir, debug_cmd])
        _os_system(cmd, log_level=log_level)
        command_mem["tpu_opt"] = " ".join(cmd)
        cmd = ["tpuc-opt", tpu_opt_mlir]

    options = tpu_ada_options(
        dynamic=dynamic,
        disable_layer_group=disable_layer_group,
        opt=opt,
        merge_weight=merge_weight,
        op_divide=op_divide,
        group_by_cores=group_by_cores,
        compress_mode=compress_mode,
        future_update_rank=future_update_rank,
        future_update_list=future_update_list,
        trunc_final=trunc_final,
        opt_post_processor=opt_post_processor,
        lg_debugger=lg_debugger,
        disable_group_overlap=(time_fixed_subnet!=None)
    )
    cmd.extend(options)

    cmd.extend(["-o", final_mlir])
    log_file = ""
    if count_patterns:
        assert not debug_info and "patterns_count is not allowed to be used with debug_cmd"
        log_file = "tpu_patterns.log"
        cmd.extend([
            "-debug-only=pattern-application,dialect-conversion,greedy-rewriter",
            "> {} 2>&1".format(log_file)
        ])
    else:
        cmd.extend([debug_cmd])

    if log_level == "quiet":
        cmd.extend(["> /dev/null"])
    elif log_level == "only-layer-group":
        cmd.extend(["--debug-only=layer-group,LayerGroupUtil"])
        cmd.insert(2, '--init="level=2"')
    elif log_level == "simple":
        cmd.insert(2, '--init="level=1"')

    _os_system(cmd, log_level=log_level)

    if time_fixed_subnet:
        command_mem["final_cut"] = " ".join(cmd)
        cmd = ["tpuc-opt", final_mlir]
        options = time_fixed_subnet_options(time_fixed_subnet, subnet_params, layer_group_cache)
        cmd.extend(options)
        final_cut_mlir = final_mlir.replace("final.mlir", "final_cut.mlir")
        cmd.extend(["-o", final_cut_mlir])
        _os_system(cmd)
        final_mlir = final_cut_mlir

    command_mem["final"] = " ".join(cmd)
    # compile ppl code
    # build_ppl()

    # codegen based on final mlir
    cmd = ["tpuc-opt", final_mlir]
    options = codegen_options(bmodel_path, embed_debug_info, model_version, gdma_check=gdma_check)
    cmd.extend(options)
    cmd.extend(["-o /dev/null"])
    _os_system(cmd, log_level=log_level)
    command_mem["bmodel"] = " ".join(cmd)

    context_dir = os.path.splitext(bmodel_path)[0]
    os.makedirs(context_dir, exist_ok=True)
    shutil.copy(final_mlir, os.path.join(context_dir, "final.mlir"))
    try:
        if bmodel_path.endswith(".bmodel") and not dynamic:
            # The suffix of the profile file is not consistent.
            # bm1684 uses ".dat", bm1684x uses ".txt".
            if os.path.exists("compiler_profile_0.[td][xa]t"):
                _os_system(
                    [
                        "mv compiler_profile_0.[td][xa]t",
                        bmodel_path + ".compiler_profile_0.txt",
                    ],
                    log_level=log_level,
                )
            if os.path.exists("net_0.profile"):
                _os_system(["mv net_0.profile", bmodel_path + ".net_0.profile"],
                           log_level=log_level)
            tensor_loc = bmodel_path + ".json"
            if os.path.exists(tensor_loc):
                shutil.copy(tensor_loc, os.path.join(context_dir, "tensor_location.json"))
    except RuntimeError:
        pass

    return get_matched_patterns(log_file)


def origin_mlir_txt_to_bmodel(
    *,
    converter,
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
    high_precision: bool = False,
    do_winograd: bool = False,
    q_group_size: int = 0,
    q_symmetric: bool = False,
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
    log_level: str = "normal",
    future_update_rank: int = 0,
    future_update_list: str = "",
    matmul_perchannel: bool = False,
    gelu_mode: str = "normal",
    quant_output_bf16: bool = False,
):

    options = []
    new_options = top_opt_options(add_postprocess)
    options.extend(new_options)
    new_options = lowering_options(mode, chip, num_device, num_core, cali_table, asymmetric,
                                   quantize_table, None, customization_format, fuse_preprocess,
                                   aligned_input, high_precision, do_winograd, q_group_size,
                                   q_symmetric, addr_mode, matmul_perchannel, gelu_mode)
    options.extend(new_options)
    new_options = tpu_opt_options(quant_input, quant_output, quant_input_list, quant_output_list,
                                  False, quant_output_bf16)
    options.extend(new_options)
    new_options = tpu_ada_options(
        dynamic=dynamic,
        disable_layer_group=disable_layer_group,
        opt=opt,
        merge_weight=merge_weight,
        op_divide=op_divide,
        group_by_cores=group_by_cores,
        compress_mode=compress_mode,
        future_update_rank=future_update_rank,
        future_update_list=future_update_list,
    )
    options.extend(new_options)
    new_options = codegen_options(f"{model_name}_{mode}.bmodel", embed_debug_info, model_version,
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


def get_final_mlir_name(bmodel: str):
    old_suffix = ".bmodel"
    new_suffix = "_final.mlir"
    filename = bmodel[:-len(old_suffix)] + new_suffix
    return filename


def origin_mlir_to_bmodel(
    origin_mlir: str,
    bmodel_name: str,
    mode: str,
    chip: str,
    num_device: int = 1,
    num_core: int = 1,
    high_precision: bool = False,
    q_group_size: int = 0,
    q_symmetric: bool = False,
    addr_mode: str = "auto",
    quant_input: bool = False,
    quant_output: bool = False,
    quant_input_list: str = "",
    quant_output_list: str = "",
    quant_output_bf16: bool = False,
    dynamic: bool = False,
    debug: bool = False,
    skip_top: bool = False,
):
    if not bmodel_name.endswith(".bmodel"):
        raise RuntimeError(f"bmodel_name should endswith .bmodel:{bmodel_name}")
    options = []
    if not skip_top:
        new_options = top_opt_options()
        options.extend(new_options)
    new_options = lowering_options(mode=mode,
                                   chip=chip,
                                   num_device=num_device,
                                   num_core=num_core,
                                   high_precision=high_precision,
                                   q_group_size=q_group_size,
                                   q_symmetric=q_symmetric,
                                   addr_mode=addr_mode)
    options.extend(new_options)
    new_options = tpu_opt_options(quant_input=quant_input,
                                  quant_output=quant_output,
                                  quant_input_list=quant_input_list,
                                  quant_output_list=quant_output_list,
                                  quant_output_bf16=quant_output_bf16)
    options.extend(new_options)
    new_options = tpu_ada_options(dynamic=dynamic)
    options.extend(new_options)
    if not debug:
        # no final mlir
        new_options = codegen_options(model=bmodel_name)
        options.extend(new_options)
        cmd = ["tpuc-opt", origin_mlir]
        cmd.extend(options)
        cmd.extend(["-o /dev/null"])
        _os_system(cmd)
    else:
        # final mlir
        final_mlir = get_final_mlir_name(bmodel_name)
        cmd = ["tpuc-opt", origin_mlir]
        cmd.extend(options)
        cmd.extend([f"-o {final_mlir}"])
        _os_system(cmd)
        # bmodel
        cmd = ["tpuc-opt", final_mlir]
        new_options = codegen_options(model=bmodel_name)
        cmd.extend(new_options)
        cmd.extend(["-o /dev/null"])
        _os_system(cmd)


def f32_blobs_compare(a_npz: str,
                      b_npz: str,
                      tolerance: str,
                      excepts=None,
                      show_detail=True,
                      fuzzy_match=False,
                      log_level="normal"):
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
def model_inference_cpu(objfile: str, output_size: str, log_level: str = "normal"):
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
    _os_system(cmd, log_level=log_level)
    print("Successfully generate executable file a.out!")
    # execute model inference
    print("Runing ...")
    cmd1 = ["./a.out", output_size]
    _os_system(cmd1, log_level=log_level)
    print("Inference ends successfully! Results are saved in inference_result.txt.")


# Extra tool: delete file in current directory
def delete_file(file: str):
    cmd = ["rm -f", file]
    _os_system(cmd)


def mlir2onnx(mlir_file: str, onnx_file: str):
    cmd = ['mlir2onnx.py -m', mlir_file, '-o', onnx_file]
    _os_system(cmd)
