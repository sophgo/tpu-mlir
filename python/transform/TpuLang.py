# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import getpass
import time
from typing import List, Union, Tuple, Optional

from .TpuLangConverter import TpuLangConverter, Graph, Tensor, Operator, Scalar, to_scalar, annotation_check, generate_name, auto_name, assert_with_out_name
from tools.model_runner import mlir_inference, model_inference, show_fake_cmd
from tools.model_deploy import getCustomFormat
# from deprecated.sphinx import deprecated
from utils.mlir_shell import *
from utils.auto_remove import file_mark
from tools.npz_tool import npz_compare
from tools.tdb import TdbInterface
from debugger.plugins.data_checker import DataCheck, DumpMode
import pymlir

import numpy as np
import math
import logging
from copy import deepcopy

logger = logging.getLogger("root")

device_list = ['cpu', 'bm1684x', 'bm1688', 'cv183x']


class TpuLang:
    graph: Graph = None
    device = None
    chip = None

    def __init__(
        self,
        device: str,
    ):
        if device.lower() in device_list:
            TpuLang.chip = device.lower()
        else:
            KeyError('TpuLang: unsupported device.')
        # self.model_name = model_name
        reset_default_graph()

    @staticmethod
    def insert_op(op_name: str, inputs: List[Tensor], outputs: List[Tensor], params: dict = {}):
        op = Operator(op_name, params=params, inputs=inputs, outputs=outputs)
        TpuLang.graph.insert_op(op)


def init(device: str):
    TpuLang(device=device)


def deinit():
    if isinstance(TpuLang.graph, Tensor):
        TpuLang.graph.reset()
    TpuLang.graph = None
    TpuLang.device = None


def reset_default_graph(graph=None):
    TpuLang.graph = Graph() if graph is None else graph


def get_default_graph():
    return TpuLang.graph


def reset_graph(graph: Graph = None):
    if graph is not None:
        graph.reset()
    else:
        TpuLang.graph.reset()


def compile(
        name: str,
        inputs: List[Tensor],
        outputs: List[Tensor],
        cmp=True,
        refs=None,
        mode='f32',  # unused
        dynamic=False,
        asymmetric=False,
        no_save=False,
        opt=2,
        mlir_inference=True,
        bmodel_inference=True,
        log_level: str = 'normal',
        embed_debug_info=False,
        addr_mode='auto',
        gdma_check=False):
    supported_log_levels = ["normal", "simple", "only-layer-group", "quiet"]
    if log_level not in supported_log_levels:
        raise ValueError(
            f"Invalid log_level: '{log_level}'. Supported values are {supported_log_levels}.")
    if log_level != 'quiet':
        logger.info("TPU-MLIR {}".format(pymlir.__version__))
    assert addr_mode in ['auto', 'io_reloc']
    TpuLang.graph.inputs = inputs
    TpuLang.graph.outputs = outputs
    TpuLang.graph.quantized_type_inference()
    converter = TpuLangConverter(name=name, graph=TpuLang.graph, mode="quantized", no_save=no_save)
    ctm_format = None
    fuse = False
    for input in TpuLang.graph.inputs:
        if input.is_preprocess:
            ctm_format = getCustomFormat(input.pixel_format, input.channel_format)
            fuse = True
            break
    # [NOTE] Please sync options for no_save !!!
    if not no_save:
        save_input_reference(model_name=name, refs=refs)
        model_transform(name, converter, log_level=log_level)
        compare = cmp and refs != None
        model_lowering_and_inference(model_name=name,
                                     quant_mode="int8",
                                     inference=mlir_inference,
                                     cmp=compare,
                                     log_level=log_level,
                                     chip=TpuLang.chip,
                                     asymmetric=asymmetric,
                                     ctm_format=ctm_format,
                                     fuse=fuse,
                                     addr_mode=addr_mode)
        bmodel_generate_and_inference(model_name=name,
                                      quant_mode="int8",
                                      inference=bmodel_inference,
                                      log_level=log_level,
                                      dynamic=dynamic,
                                      opt=opt,
                                      embed_debug_info=embed_debug_info,
                                      gdma_check=gdma_check)
    else:
        origin_mlir_txt_to_bmodel(converter=converter,
                                  model_name=name,
                                  mode="int8",
                                  log_level=log_level,
                                  chip=TpuLang.chip,
                                  asymmetric=asymmetric,
                                  customization_format=ctm_format,
                                  fuse_preprocess=fuse,
                                  dynamic=dynamic,
                                  opt=opt,
                                  embed_debug_info=embed_debug_info,
                                  addr_mode=addr_mode)


def compile_f32(name: str,
                inputs: List[Tensor],
                outputs: List[Tensor],
                cmp=True,
                refs=None,
                mode='f32',
                dynamic=False,
                opt=2,
                no_save=False,
                mlir_inference=True,
                bmodel_inference=True,
                top_mlir_inference=True,
                tpu_mlir_inference=True,
                log_level: str = 'normal',
                embed_debug_info=False,
                addr_mode='auto',
                gdma_check=False):
    TpuLang.graph.inputs = inputs
    TpuLang.graph.outputs = outputs
    TpuLang.graph.quantized_type_inference()
    assert mode in ['f32', 'f16', 'bf16', 'int8', 'all', 'none']
    assert addr_mode in ['auto', 'io_reloc']
    supported_log_levels = ["normal", "simple", "only-layer-group", "quiet"]
    if log_level not in supported_log_levels:
        raise ValueError(
            f"Invalid log_level: '{log_level}'. Supported values are {supported_log_levels}.")
    mode_list = [mode]
    if mode == 'all':
        mode_list = ['f32', 'f16', 'bf16']
    converter = TpuLangConverter(name=name, graph=TpuLang.graph, mode="f32", no_save=no_save)
    # [NOTE] Please sync options for no_save !!!
    if not no_save:
        save_input_reference(model_name=name, refs=refs)
        model_transform(name, converter, log_level=log_level)
        compare = cmp and refs != None
        top_mlir_inference = top_mlir_inference and mlir_inference
        tpu_mlir_inference = tpu_mlir_inference and mlir_inference
        if top_mlir_inference:
            model_top_inference(model_name=name, cmp=compare, log_level=log_level)
        for m in mode_list:
            tpu_mlir_compare = cmp and top_mlir_inference
            model_lowering_and_inference(model_name=name,
                                         quant_mode=m,
                                         inference=tpu_mlir_inference,
                                         cmp=tpu_mlir_compare,
                                         log_level=log_level,
                                         chip=TpuLang.chip,
                                         addr_mode=addr_mode)
            bmodel_generate_and_inference(model_name=name,
                                          quant_mode=m,
                                          inference=bmodel_inference,
                                          log_level=log_level,
                                          dynamic=dynamic,
                                          opt=opt,
                                          embed_debug_info=embed_debug_info,
                                          gdma_check=gdma_check)
    else:
        for m in mode_list:
            origin_mlir_txt_to_bmodel(converter=converter,
                                      model_name=name,
                                      mode=m,
                                      log_level=log_level,
                                      chip=TpuLang.chip,
                                      dynamic=dynamic,
                                      opt=opt,
                                      embed_debug_info=embed_debug_info,
                                      addr_mode=addr_mode)


def model_transform(model_name, converter: TpuLangConverter, log_level: str = 'normal'):
    mlir_file = model_name + '.mlir'
    mlir_origin = model_name + '_origin.mlir'
    converter.generate_mlir(mlir_origin, log_level=log_level)
    mlir_opt_for_top(mlir_origin, mlir_file, log_level=log_level)
    if log_level != "quiet":
        print("Mlir file generated:{}".format(mlir_file))


def save_input_reference(model_name, refs: dict):
    in_f32_npz = model_name + '_in_f32.npz'
    ref_inputs = dict()
    if refs is not None:
        ref_npz = model_name + '_ref_output.npz'
        np.savez(ref_npz, **refs)
    for tensor in TpuLang.graph.inputs:
        if refs is not None and tensor.name in refs.keys():
            ref_inputs[tensor.name] = refs[tensor.name]
        else:
            ref_inputs[tensor.name] = tensor.buffer
    np.savez(in_f32_npz, **ref_inputs)


def model_top_inference(model_name, cmp=False, log_level: str = 'normal'):
    in_f32_npz = model_name + '_in_f32.npz'
    mlir_file = model_name + '.mlir'
    input_data = np.load(in_f32_npz)
    top_npz = model_name + '_top_outputs.npz'
    if log_level != "quiet":
        show_fake_cmd(in_f32_npz, mlir_file, top_npz)
    f32_outputs = mlir_inference(dict(input_data), mlir_file, log_level=log_level)
    np.savez(top_npz, **f32_outputs)
    if cmp:
        ref_npz = model_name + '_ref_output.npz'
        f32_blobs_compare(top_npz, ref_npz, '0.99,0.99', log_level=log_level)

def model_lowering_and_inference(model_name: str, quant_mode: str, chip: str, asymmetric: bool = False, \
                                 inference: bool = True, cmp: bool = False, ctm_format = "BGR_PLANAR", \
                                 fuse=False,log_level : str = 'normal', addr_mode:str = 'auto'):
    top_mlir = "{}.mlir".format(model_name)
    tpu_mlir = "{}_{}.mlir".format(model_name, quant_mode)

    mlir_lowering(top_mlir, tpu_mlir, mode=quant_mode, chip=chip, asymmetric=asymmetric, \
                  customization_format=ctm_format, fuse_preprocess=fuse, addr_mode=addr_mode, \
                  log_level=log_level)
    if inference:
        in_f32_npz = model_name + '_in_f32.npz'
        tpu_npz = tpu_mlir.replace(".mlir", "_tpu_out.npz")
        input_data = np.load(in_f32_npz)
        file_mark(tpu_npz)
        if log_level != "quiet":
            show_fake_cmd(in_f32_npz, tpu_mlir, tpu_npz)
        tpu_mlir_outs = mlir_inference(dict(input_data),
                                       tpu_mlir,
                                       dump_all=True,
                                       log_level=log_level)
        np.savez(tpu_npz, **tpu_mlir_outs)
        if cmp:
            if quant_mode == 'int8':
                ref_npz = model_name + '_ref_output.npz'
                npz_compare([ref_npz, tpu_npz, "--tolerance", "0.95,0.80", "-v"],
                            log_level=log_level)
            else:
                top_npz = model_name + '_top_outputs.npz'
                npz_compare([top_npz, tpu_npz, "--tolerance", "0.95,0.80", "-v"],
                            log_level=log_level)


def bmodel_generate_and_inference(model_name: str,
                                  quant_mode: str,
                                  inference: bool = True,
                                  dynamic: bool = False,
                                  opt: int = 2,
                                  log_level: str = 'normal',
                                  embed_debug_info=False,
                                  gdma_check=False):
    # generate bmodel
    tpu_mlir = "{}_{}".format(model_name, quant_mode)
    tpu_final = tpu_mlir + "_final.mlir"
    bmodel = tpu_mlir + ".bmodel"
    disable_layer_group = opt == 0
    assert opt in [0, 1, 2]
    mlir_to_model(tpu_mlir=tpu_mlir + ".mlir",
                  bmodel_path=bmodel,
                  final_mlir=tpu_final,
                  dynamic=dynamic,
                  opt=opt,
                  disable_layer_group=disable_layer_group,
                  log_level=log_level,
                  embed_debug_info=embed_debug_info,
                  gdma_check=gdma_check)

    if False:
        bmodel_file = tpu_mlir + ".bmodel"
        final_file = tpu_mlir + "_final.mlir"
        input_file = model_name + '_in_f32.npz'
        location_file = bmodel_file + ".json"
        output_file = tpu_mlir + "_tpu_out.npz"
        bmodel_inference_combine(bmodel_file,
                                 final_file,
                                 input_file,
                                 location_file,
                                 output_file,
                                 dump_file=True,
                                 save_path=tpu_mlir + "_model_output.npz",
                                 out_fixed=False)

    if inference:
        # inference
        in_f32_npz = model_name + '_in_f32.npz'
        tpu_npz = tpu_mlir + "_tpu_out.npz"
        input_data = np.load(in_f32_npz)
        model_npz = bmodel.replace("." + bmodel.split(".")[-1], "_model_out.npz")
        file_mark(model_npz)
        if log_level != 'quiet':
            show_fake_cmd(in_f32_npz, bmodel, model_npz)
        model_outs = model_inference(dict(input_data), bmodel, log_level=log_level)
        np.savez(model_npz, **model_outs)
        npz_compare([tpu_npz, model_npz, "--tolerance", "0.95,0.80", "-v"], log_level=log_level)


def bmodel_inference_combine(
        bmodel_file: str,
        final_mlir_fn: str,
        input_data_fn: Union[str, dict],
        tensor_loc_file: str,
        reference_data_fn: str,
        dump_file: bool = True,
        save_path: str = "",
        out_fixed: bool = False,
        dump_cmd_info: bool = True,
        skip_check: bool = True,  # disable data_check to increase processing speed
        run_by_op:
    bool = False,  # enable to run_by_op, may cause timeout error when some OPs contain too many atomic cmds
        desire_op:
    list = [],  # set ["A","B","C"] to only dump tensor A/B/C, dump all tensor as defalt
        is_soc: bool = False,  # soc mode ONLY support {reference_data_fn=xxx.npz, dump_file=True}
        using_memory_opt: bool = False,  # required when is_soc=True
        enable_soc_log: bool = False,  # required when is_soc=True
        soc_tmp_path: str = "/tmp",  # required when is_soc=True
        hostname: str = None,  # required when is_soc=True
        port: int = None,  # required when is_soc=True
        username: str = None,  # required when is_soc=True
        password: str = None,  # required when is_soc=True
):
    tdb = TdbInterface(
        bmodel_file=bmodel_file,
        final_mlir_fn=final_mlir_fn,
        tensor_loc_file=tensor_loc_file,
        input_data_fn=input_data_fn,
        reference_data_fn=reference_data_fn,
        extra_plugins=["progress"],
        extra_check=[],
        auto_soc=is_soc,
        args={
            "run_by_atomic": not run_by_op,
            "ddr_size": 2**32,
        },
    )
    plugin: DataCheck = tdb.get_plugin(DataCheck)
    plugin.__init__(tdb)
    plugin.set_tol(cosine_similarity_tol=0.99, euclidean_similarity_tol=0.99)
    plugin.dump_mode = getattr(DumpMode, "TPULANG", DumpMode.TPULANG)
    plugin.out_fixed = out_fixed
    plugin.is_soc = is_soc
    plugin.skip_check = skip_check
    plugin.desire_op = desire_op

    tdb.message(f"dump mode = {plugin.dump_mode}")
    tdb.do_run("")
    if not is_soc and not reference_data_fn.endswith(".mlir") and not skip_check:
        plugin.do_summary("table")
    if dump_file or is_soc:
        os.makedirs(save_path, exist_ok=True)
        if dump_cmd_info:
            plugin.dump_dataframe(save_path)

    if is_soc:
        import pickle
        import debugger.soc_tools.utils as soc_tools

        # prepare the cmds and values_info
        os.makedirs(os.path.join(save_path, "soc_tmp"), exist_ok=True)
        with open(os.path.join(save_path, "soc_tmp", "cmds.pkl"), "wb") as cmds_pkl:
            pickle.dump(tdb.runner.soc_structs, cmds_pkl)
        with open(os.path.join(save_path, "soc_tmp", "values_in.pkl"), "wb") as values_in_pkl:
            pickle.dump(DataCheck.soc_values_in, values_in_pkl)
        with open(os.path.join(save_path, "soc_tmp", "values_out.pkl"), "wb") as values_out_pkl:
            pickle.dump(DataCheck.soc_values_out, values_out_pkl)

        assert len(DataCheck.soc_values_in) == len(
            DataCheck.soc_values_out
        ), f"len_value_in: {len(DataCheck.soc_values_in)}, len_value_out: {len(DataCheck.soc_values_out)}"

        try:
            # connect remote ssh server
            client, sftp = soc_tools.soc_connect(hostname, port, username, password)

            # collect dependences
            soc_tools.collect_files()

            # transfer cache files and soc_infer tools
            soc_tools.soc_trans_files(
                client,
                sftp,
                save_path,
                soc_tmp_path,
                bmodel_file,
                input_data_fn,
                reference_data_fn,
            )

            # raise KeyError
            # execute soc_bmodel_infer
            remote_bmodel = os.path.basename(bmodel_file)
            remote_input = os.path.basename(input_data_fn)
            remote_ref = os.path.basename(reference_data_fn)
            soc_tools_path = os.path.join(soc_tmp_path, "soc_tools")
            exec_command = f"cd {soc_tools_path} && source envsetup.sh && nohup python3 soc_bmodel_infer.py --path {soc_tmp_path} --bmodel {remote_bmodel} --input {remote_input} --ref {remote_ref} --tool_path {soc_tools_path}"
            if desire_op:
                exec_command += f" --desire_op {','.join(desire_op)}"
            if out_fixed:
                exec_command += " --out_fixed"
            if enable_soc_log:
                exec_command += " --enable_log"
            if using_memory_opt:
                exec_command += " --using_memory_opt"
            if not run_by_op:
                exec_command += " --run_by_atomic"
            exec_command += " &"

            # check wheth program on soc is finished
            soc_tools.soc_check_end_status(exec_command, client, sftp, soc_tools_path)

            # fetch log and infer result
            soc_tools.soc_fetch_log_and_npz(client, sftp, save_path, soc_tools_path, remote_ref,
                                            enable_soc_log)
            soc_tools._soc_rm_dir(client, soc_tmp_path)
        finally:
            soc_tools.clean_collected_files()
            client.close()

        return

    if dump_file:  # plugin.ref_data_from_inference -> [np_dict]
        file_name = os.path.basename(reference_data_fn).split(".")[0]
        out_path = os.path.join(save_path, f"bmodel_infer_{file_name}.npz")
        np.savez(out_path, **plugin.ref_data_from_inference)
    else:
        return plugin.ref_data_from_inference


def init(device: str):
    TpuLang(device=device)


def deinit():
    for op in TpuLang.graph.operators:
        for tensor in op.inputs + op.outputs:
            if isinstance(tensor, Tensor):
                tensor.reset()
    TpuLang.graph = None
    TpuLang.device = None


def ArrayAttr(data: list, data_type: str = 'int64'):
    _data = deepcopy(data)
    return [_data, data_type, False]


# data_type must be in ["float32", "float16", "int64" "int32", "int16". "int8", "uint8", "bool", "string", "dict"]
def Attr(data, data_type: str = 'int64'):
    assert data_type.find("int") >= 0 or data_type in [
        "float32", "float64", "bool", "string", "dict"
    ]
    return [data, data_type, True]


'''
  def operand(inputs, **params):
    attr = {
      param_name : ArrayAttr(param, param_dtype) or Attr(param, param_dtype)
      ...
    }
    output = Tensor(shape, dtype, name)
    TpuLang.insert_op(op_type, inputs=[inputs...], outputs=[output], params=attr)
    return output
'''


@annotation_check
@assert_with_out_name
def custom(tensors_in: List[Tensor],
           op_name: str,
           out_dtypes: List[str],
           out_names: List[str] = None,
           params: dict = None):
    '''
        The custom op
        Arguments:
            tensors_in: list of input tensors (including weight tensors).
            op_name: name of the custom operator.
            out_dtypes: list of data type of outputs.
            out_names: list of name of outputs.
            params: parameters of the custom op.

        Return:
            tensors_out: list of output tensors
    '''

    out_num = len(out_dtypes)
    if out_names:
        if out_num > len(out_names):
            out_names.extend(
                [generate_name(f"custom_{op_name}_{i}") for i in range(len(out_names), out_num)])
    else:
        out_names = [generate_name(f"custom_{op_name}_{i}") for i in range(out_num)]

    tensors_out = []
    for i, out_dtype in enumerate(out_dtypes):
        tensor_out = Tensor(dtype=out_dtype, name=out_names[i])
        tensors_out.append(tensor_out)

    attrs = {}

    attrs["name"] = Attr(op_name, "string")
    dict_array = []
    for key, value in params.items():
        params_new = {}
        if isinstance(value, int):
            value_new = Attr(value)
        elif isinstance(value, bool):
            value_new = Attr(value, "bool")
        elif isinstance(value, list):
            if all(isinstance(x, int) for x in value):
                value_new = ArrayAttr(value)
            elif all(isinstance(x, float) for x in value):
                value_new = ArrayAttr(value, "float32")
            else:
                raise ValueError(f"Elements in the list of {key} must be int-only or float-only")
        else:
            value_new = Attr(value, "float32")
        params_new[key] = value_new
        dict_array.append(Attr(params_new, "dict"))

    attrs["params"] = ArrayAttr(dict_array, "dict")

    TpuLang.insert_op("top.Custom", tensors_in, tensors_out, params=attrs)

    return tensors_out


def binary_dtype_check(tensor_i0: Union[Tensor, Scalar],
                       tensor_i1: Union[Tensor, Scalar],
                       out_dtype: str = None,
                       sign: bool = False,
                       is_shift: bool = False,
                       is_compare: bool = False):
    assert isinstance(tensor_i0, Tensor) or isinstance(tensor_i1, Tensor)
    in0_dtype = tensor_i0.dtype if isinstance(tensor_i0, Tensor) else tensor_i1.dtype
    in1_dtype = tensor_i1.dtype if isinstance(tensor_i1, Tensor) else tensor_i0.dtype
    support_dtype = ["float32", "float16", "int8", "uint8"]
    if is_shift or is_compare:
        start_idx = 0 if is_compare else 2
        support_dtype = support_dtype[start_idx:] + ["int16", "uint16", "int32", "uint32"
                                                     ]  # no float type for shift op
    assert not out_dtype or out_dtype in support_dtype
    if in0_dtype in ["float32", "float16"]:
        assert in0_dtype == in1_dtype
        out_dtype = in0_dtype if out_dtype == None else out_dtype
        assert in0_dtype == out_dtype
    elif in0_dtype.find("int") >= 0:
        assert in1_dtype.find("int") >= 0
        out_dtype = (in0_dtype if in0_dtype.startswith("int") else
                     in1_dtype) if out_dtype == None else out_dtype
        assert out_dtype.find("int") >= 0
        if sign and out_dtype.startswith("uint"):
            out_dtype = out_dtype[1:]  # has to be signed output
    return out_dtype


def same_dtype_check(in0_dtype: str, in1_dtype: str = None, out_dtype: str = None):
    if in1_dtype is not None:
        assert in0_dtype == in1_dtype
    if out_dtype is not None:
        assert in0_dtype == out_dtype
    return in0_dtype


def _conv_float_check(input: Tensor, weight: Tensor, bias: Tensor = None):
    assert input.dtype in ["float32", "float16"]
    assert weight.dtype == input.dtype
    assert input.is_quantized is False
    assert weight.is_quantized is False
    if bias is not None and bias.dtype == "float16":
        bias.dtype = "float32"
        bias.buffer = bias.buffer.astype("float32")


@auto_name()
@annotation_check
@assert_with_out_name
def conv(input: Tensor,
         weight: Tensor,
         bias: Tensor = None,
         stride: List[int] = None,
         dilation: List[int] = None,
         pad: List[int] = None,
         group: int = 1,
         out_dtype: str = None,
         out_name: str = None):
    dilation = [1, 1] if dilation is None else dilation
    stride = [1, 1] if stride is None else stride
    pad = [0, 0, 0, 0] if pad is None else pad
    assert len(dilation) == 2 and len(stride) == 2 and len(pad) == 4
    o_dtype = input.dtype if out_dtype is None else out_dtype
    assert input.dtype == o_dtype
    _conv_float_check(input, weight, bias)

    attr = {
        "kernel_shape": ArrayAttr(weight.shape[2:]),
        "strides": ArrayAttr(stride),
        "dilations": ArrayAttr(dilation),
        "pads": ArrayAttr(pad),
        "do_relu": Attr(False, "bool"),
        "group": Attr(group)
    }

    output = Tensor(dtype=o_dtype, name=out_name)
    inputs = [input, weight, bias]
    TpuLang.insert_op("top.Conv", inputs=inputs, outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def conv_int(input: Tensor,
             weight: Tensor,
             bias: Tensor = None,
             stride: List[int] = None,
             dilation: List[int] = None,
             pad: List[int] = None,
             group: int = 1,
             input_zp: Union[int, List[int]] = None,
             weight_zp: Union[int, List[int]] = None,
             out_dtype: str = None,
             out_name: str = None):
    assert not isinstance(input_zp, list), "not supported yet"
    assert input.dtype in ["int8", "uint8"] and weight.dtype in ["int8", "uint8"]
    if bias:
        assert bias.dtype == "int32"
    o_dtype = "int32" if out_dtype is None else out_dtype
    assert o_dtype in ["int32", "uint32"]
    dilation = [1, 1] if dilation is None else dilation
    stride = [1, 1] if stride is None else stride
    pad = [0, 0, 0, 0] if pad is None else pad
    assert len(dilation) == 2 and len(stride) == 2 and len(pad) == 4
    assert input.is_quantized is True or input_zp is not None
    assert weight.is_quantized is True or weight_zp is not None

    attr = {
        "kernel_shape": ArrayAttr(weight.shape[2:]),
        "strides": ArrayAttr(stride),
        "dilations": ArrayAttr(dilation),
        "pads": ArrayAttr(pad),
        "do_relu": Attr(False, "bool"),
        "group": Attr(group)
    }
    output = Tensor(dtype=o_dtype, name=out_name)
    input.quantization(zero_point=input_zp)
    weight.quantization(zero_point=weight_zp)
    inputs = [input, weight, bias]
    TpuLang.insert_op("top.Conv", inputs=inputs, outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def conv_quant(input: Tensor,
               weight: Tensor,
               bias: Tensor = None,
               stride: List[int] = None,
               dilation: List[int] = None,
               pad: List[int] = None,
               group: int = 1,
               input_scale: Union[float, List[float]] = None,
               weight_scale: Union[float, List[float]] = None,
               output_scale: Union[float, List[float]] = None,
               input_zp: Union[int, List[int]] = None,
               weight_zp: Union[int, List[int]] = None,
               output_zp: Union[int, List[int]] = None,
               out_dtype: str = None,
               out_name: str = None):
    assert not isinstance(input_zp, list), "not supported yet"
    assert not isinstance(output_zp, list), "not supported yet"
    assert not isinstance(input_scale, list), "not supported yet"
    assert not isinstance(output_scale, list), "not supported yet"
    dilation = [1, 1] if dilation is None else dilation
    stride = [1, 1] if stride is None else stride
    pad = [0, 0, 0, 0] if pad is None else pad
    assert len(dilation) == 2 and len(stride) == 2 and len(pad) == 4
    o_dtype = out_dtype if out_dtype is not None else input.dtype
    assert input.dtype in ["int8", "uint8"] and weight.dtype in ["int8", "uint8"
                                                                 ] and o_dtype in ["int8", "uint8"]
    if bias:
        bias.dtype == "int32"
    assert input.is_quantized is True or input_scale is not None
    assert weight.is_quantized is True or weight_scale is not None

    attr = {
        "kernel_shape": ArrayAttr(weight.shape[2:]),
        "strides": ArrayAttr(stride),
        "dilations": ArrayAttr(dilation),
        "pads": ArrayAttr(pad),
        "do_relu": Attr(False, "bool"),
        "group": Attr(group)
    }
    output = Tensor(dtype=o_dtype, name=out_name, scale=output_scale, zero_point=output_zp)
    input.quantization(scale=input_scale, zero_point=input_zp)
    weight.quantization(scale=weight_scale, zero_point=weight_zp)
    inputs = [input, weight, bias]
    TpuLang.insert_op("top.Conv", inputs=inputs, outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def conv3d(input: Tensor,
           weight: Tensor,
           bias: Tensor = None,
           stride: List[int] = None,
           dilation: List[int] = None,
           pad: List[int] = None,
           group: int = 1,
           out_dtype: str = None,
           out_name: str = None):
    dilation = [1, 1, 1] if dilation is None else dilation
    stride = [1, 1, 1] if stride is None else stride
    pad = [0, 0, 0, 0, 0, 0] if pad is None else pad
    assert len(dilation) == 3 and len(stride) == 3 and len(pad) == 6
    o_dtype = input.dtype if out_dtype is None else out_dtype
    assert input.dtype == o_dtype
    _conv_float_check(input, weight, bias)

    attr = {
        "kernel_shape": ArrayAttr(weight.shape[2:]),
        "strides": ArrayAttr(stride),
        "dilations": ArrayAttr(dilation),
        "pads": ArrayAttr(pad),
        "do_relu": Attr(False, "bool"),
        "group": Attr(group)
    }
    output = Tensor(dtype=o_dtype, name=out_name)
    inputs = [input, weight, bias]
    TpuLang.insert_op("top.Conv", inputs=inputs, outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def conv3d_int(input: Tensor,
               weight: Tensor,
               bias: Tensor = None,
               stride: List[int] = None,
               dilation: List[int] = None,
               pad: List[int] = None,
               group: int = 1,
               input_zp: Union[int, List[int]] = None,
               weight_zp: Union[int, List[int]] = None,
               out_dtype: str = None,
               out_name: str = None):
    assert not isinstance(input_zp, list), "not supported yet"
    assert input.dtype in ["int8", "uint8"] and weight.dtype in ["int8", "uint8"]
    dilation = [1, 1, 1] if dilation is None else dilation
    stride = [1, 1, 1] if stride is None else stride
    pad = [0, 0, 0, 0, 0, 0] if pad is None else pad
    assert len(dilation) == 3 and len(stride) == 3 and len(pad) == 6
    o_dtype = "int32" if out_dtype is None else out_dtype
    assert o_dtype in ["int32", "uint32"]
    assert input.is_quantized is True or input_zp is not None
    assert weight.is_quantized is True or weight_zp is not None

    attr = {
        "kernel_shape": ArrayAttr(weight.shape[2:]),
        "strides": ArrayAttr(stride),
        "dilations": ArrayAttr(dilation),
        "pads": ArrayAttr(pad),
        "do_relu": Attr(False, "bool"),
        "group": Attr(group)
    }
    output = Tensor([], dtype=o_dtype, name=out_name)
    input.quantization(zero_point=input_zp)
    weight.quantization(zero_point=weight_zp)
    inputs = [input, weight, bias]
    TpuLang.insert_op("top.Conv", inputs=inputs, outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def conv3d_quant(input: Tensor,
                 weight: Tensor,
                 bias: Tensor = None,
                 stride: List[int] = None,
                 dilation: List[int] = None,
                 pad: List[int] = None,
                 group: int = 1,
                 input_scale: Union[float, List[float]] = None,
                 weight_scale: Union[float, List[float]] = None,
                 output_scale: Union[float, List[float]] = None,
                 input_zp: Union[int, List[int]] = None,
                 weight_zp: Union[int, List[int]] = None,
                 output_zp: Union[int, List[int]] = None,
                 out_dtype: str = None,
                 out_name: str = None):
    assert not isinstance(input_zp, list), "not supported yet"
    assert not isinstance(output_zp, list), "not supported yet"
    assert not isinstance(input_scale, list), "not supported yet"
    assert not isinstance(output_scale, list), "not supported yet"
    assert input.dtype in ["int8", "uint8"] and weight.dtype in ["int8", "uint8"]

    dilation = [1, 1, 1] if dilation is None else dilation
    stride = [1, 1, 1] if stride is None else stride
    pad = [0, 0, 0, 0, 0, 0] if pad is None else pad
    assert len(dilation) == 3 and len(stride) == 3 and len(pad) == 6
    o_dtype = out_dtype if out_dtype is not None else input.dtype
    assert o_dtype in ["int8", "uint8"]
    assert input.is_quantized is True or input_scale is not None
    assert weight.is_quantized is True or weight_scale is not None

    attr = {
        "kernel_shape": ArrayAttr(weight.shape[2:]),
        "strides": ArrayAttr(stride),
        "dilations": ArrayAttr(dilation),
        "pads": ArrayAttr(pad),
        "do_relu": Attr(False, "bool"),
        "group": Attr(group)
    }
    output = Tensor(dtype=o_dtype, name=out_name, scale=output_scale, zero_point=output_zp)
    input.quantization(scale=input_scale, zero_point=input_zp)
    weight.quantization(scale=weight_scale, zero_point=weight_zp)
    inputs = [input, weight, bias]
    TpuLang.insert_op("top.Conv", inputs=inputs, outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def deconv(input: Tensor,
           weight: Tensor,
           bias: Tensor = None,
           stride: List[int] = None,
           dilation: List[int] = None,
           pad: List[int] = None,
           output_padding: List[int] = None,
           group: int = 1,
           out_dtype: str = None,
           out_name: str = None):
    dilation = [1, 1] if dilation is None else dilation
    stride = [1, 1] if stride is None else stride
    pad = [0, 0, 0, 0] if pad is None else pad
    output_padding = [0, 0] if output_padding is None else output_padding
    assert len(dilation) == 2 and len(stride) == 2 and len(pad) == 4 and len(output_padding) == 2
    o_dtype = input.dtype if out_dtype is None else out_dtype
    assert input.dtype == o_dtype
    _conv_float_check(input, weight, bias)

    attr = {
        "kernel_shape": ArrayAttr(weight.shape[2:]),
        "strides": ArrayAttr(stride),
        "dilations": ArrayAttr(dilation),
        "pads": ArrayAttr(pad),
        "output_padding": ArrayAttr(output_padding),
        "do_relu": Attr(False, "bool"),
        "group": Attr(group)
    }
    output = Tensor([], dtype=o_dtype, name=out_name)
    inputs = [input, weight, bias]
    TpuLang.insert_op("top.Deconv", inputs=inputs, outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def deconv_int(input: Tensor,
               weight: Tensor,
               bias: Tensor = None,
               stride: List[int] = None,
               dilation: List[int] = None,
               pad: List[int] = None,
               output_padding: List[int] = None,
               group: int = 1,
               input_zp: Union[int, List[int]] = None,
               weight_zp: Union[int, List[int]] = None,
               out_dtype: str = None,
               out_name: str = None):
    assert not isinstance(input_zp, list), "not supported yet"
    assert input.dtype in ["int8", "uint8"] and weight.dtype in ["int8", "uint8"]
    if bias:
        assert bias.dtype == "int32"
    dilation = [1, 1] if dilation is None else dilation
    stride = [1, 1] if stride is None else stride
    pad = [0, 0, 0, 0] if pad is None else pad
    output_padding = [0, 0] if output_padding is None else output_padding
    assert len(dilation) == 2 and len(stride) == 2 and len(pad) == 4 and len(output_padding) == 2
    o_dtype = "int32" if out_dtype is None else out_dtype
    assert o_dtype in ["int32", "uint32"]
    assert input.is_quantized is True or input_zp is not None
    assert weight.is_quantized is True or weight_zp is not None

    attr = {
        "kernel_shape": ArrayAttr(weight.shape[2:]),
        "strides": ArrayAttr(stride),
        "dilations": ArrayAttr(dilation),
        "pads": ArrayAttr(pad),
        "output_padding": ArrayAttr(output_padding),
        "do_relu": Attr(False, "bool"),
        "group": Attr(group)
    }
    output = Tensor(dtype=o_dtype, name=out_name)
    input.quantization(zero_point=input_zp)
    weight.quantization(zero_point=weight_zp)
    inputs = [input, weight, bias]
    TpuLang.insert_op("top.Deconv", inputs=inputs, outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def deconv3d(input: Tensor,
             weight: Tensor,
             bias: Tensor = None,
             stride: List[int] = None,
             dilation: List[int] = None,
             pad: List[int] = None,
             output_padding: List[int] = None,
             group: int = 1,
             out_dtype: str = None,
             out_name: str = None):
    dilation = [1, 1, 1] if dilation is None else dilation
    stride = [1, 1, 1] if stride is None else stride
    pad = [0, 0, 0, 0, 0, 0] if pad is None else pad
    output_padding = [0, 0, 0] if output_padding is None else output_padding
    assert len(dilation) == 3 and len(stride) == 3 and len(pad) == 6 and len(output_padding) == 3
    o_dtype = input.dtype if out_dtype is None else out_dtype
    assert input.dtype == o_dtype
    _conv_float_check(input, weight, bias)

    attr = {
        "kernel_shape": ArrayAttr(weight.shape[2:]),
        "strides": ArrayAttr(stride),
        "dilations": ArrayAttr(dilation),
        "pads": ArrayAttr(pad),
        "output_padding": ArrayAttr(output_padding),
        "do_relu": Attr(False, "bool"),
        "group": Attr(group)
    }
    output = Tensor([], dtype=o_dtype, name=out_name)
    inputs = [input, weight, bias]
    TpuLang.insert_op("top.Deconv", inputs=inputs, outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def matmul(input: Tensor,
           right: Tensor,
           bias: Tensor = None,
           input_transpose: bool = False,
           right_transpose: bool = False,
           output_transpose: bool = False,
           keep_dims: bool = True,
           out_dtype: str = None,
           out_name: str = None):
    o_dtype = input.dtype if out_dtype is None else out_dtype
    assert input.dtype == o_dtype
    assert input.dtype in ["float32", "float16"] and input.dtype == right.dtype
    if bias:
        assert bias.dtype in ["float32", "float16"]
    assert input.is_quantized is False
    assert right.is_quantized is False

    attr = {
        "right_transpose": Attr(right_transpose, "bool"),
        "left_transpose": Attr(input_transpose, "bool"),
        "output_transpose": Attr(output_transpose, "bool"),
        "hdim_is_batch": Attr(False, "bool"),
        "keep_dims": Attr(keep_dims, "bool"),
        "do_relu": Attr(False, "bool"),
        "relu_limit": Attr(-1.0, "float64")
    }

    output = Tensor(dtype=o_dtype, name=out_name)
    inputs = [input, right, bias]
    TpuLang.insert_op("top.MatMul", inputs=inputs, outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def matmul_int(input: Tensor,
               right: Tensor,
               bias: Tensor = None,
               input_transpose: bool = False,
               right_transpose: bool = False,
               output_transpose: bool = False,
               keep_dims: bool = True,
               input_zp: Union[int, List[int]] = None,
               right_zp: Union[int, List[int]] = None,
               out_dtype: str = None,
               out_name: str = None):
    assert not isinstance(input_zp, list), "not supported yet"
    assert not isinstance(right_zp, list), "not supported yet"
    assert input.dtype in ["int8", "uint8"] and right.dtype in ["int8", "uint8"]
    if bias:
        assert bias.dtype == "int32"
    o_dtype = "int32" if out_dtype is None else out_dtype
    assert o_dtype in ["int32", "uint32"]
    assert input.is_quantized is True or input_zp is not None
    assert right.is_quantized is True or right_zp is not None

    attr = {
        "right_transpose": Attr(right_transpose, "bool"),
        "left_transpose": Attr(input_transpose, "bool"),
        "output_transpose": Attr(output_transpose, "bool"),
        "hdim_is_batch": Attr(False, "bool"),
        "keep_dims": Attr(keep_dims, "bool"),
        "do_relu": Attr(False, "bool"),
        "relu_limit": Attr(-1.0, "float64")
    }

    output = Tensor(dtype=o_dtype, name=out_name)
    input.quantization(zero_point=input_zp)
    right.quantization(zero_point=right_zp)
    inputs = [input, right, bias]
    TpuLang.insert_op("top.MatMul", inputs=inputs, outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def matmul_quant(input: Tensor,
                 right: Tensor,
                 bias: Tensor = None,
                 right_transpose=False,
                 keep_dims=True,
                 input_scale: Union[float, List[float]] = None,
                 right_scale: Union[float, List[float]] = None,
                 output_scale: Union[float, List[float]] = None,
                 input_zp: Union[int, List[int]] = None,
                 right_zp: Union[int, List[int]] = None,
                 output_zp: Union[int, List[int]] = None,
                 out_dtype: str = 'int8',
                 out_name: str = None):
    assert not isinstance(input_zp, list), "not supported yet"
    assert not isinstance(right_zp, list), "not supported yet"
    assert not isinstance(output_zp, list), "not supported yet"
    assert not isinstance(input_scale, list), "not supported yet"
    assert not isinstance(right_scale, list), "not supported yet"
    assert not isinstance(output_scale, list), "not supported yet"

    attr = {
        "right_transpose": Attr(right_transpose, "bool"),
        "left_transpose": Attr(False, "bool"),
        "output_transpose": Attr(False, "bool"),
        "hdim_is_batch": Attr(False, "bool"),
        "keep_dims": Attr(keep_dims, "bool"),
        "do_relu": Attr(False, "bool"),
        "relu_limit": Attr(-1.0, "float64")
    }
    assert input.dtype in ["int8", "uint8"] and right.dtype in ["int8", "uint8"] and out_dtype in [
        "int8", "uint8"
    ]
    if bias:
        bias.dtype == "int32"
    assert input.is_quantized is True or input_scale is not None
    assert right.is_quantized is True or right_scale is not None
    assert output_scale is not None
    output = Tensor(dtype=out_dtype, name=out_name, scale=output_scale, zero_point=output_zp)
    input.quantization(scale=input_scale, zero_point=input_zp)
    right.quantization(scale=right_scale, zero_point=right_zp)
    inputs = [input, right, bias]
    TpuLang.insert_op("top.MatMul", inputs=inputs, outputs=[output], params=attr)
    return output


############## Base Element Operator ###############
def _base_binary(tensor_i0: Union[Tensor, Scalar],
                 tensor_i1: Union[Tensor, Scalar],
                 op_type: str,
                 scale: List[float] = None,
                 zero_point: List[int] = None,
                 is_reverse: bool = None,
                 out_dtype: str = None,
                 out_name: str = None):
    o_dtype = binary_dtype_check(tensor_i0,
                                 tensor_i1,
                                 out_dtype,
                                 sign=(op_type == "top.Sub"),
                                 is_compare=op_type in ["top.Max", "top.Min"])
    output = Tensor(dtype=o_dtype, name=out_name)
    if scale is not None:
        assert len(scale) == 3
        zero_point = zero_point if zero_point is not None else [0, 0, 0]
        assert len(zero_point) == 3
        if isinstance(tensor_i0, Tensor) or op_type != "top.Mul":
            tensor_i0 = tensor_i0 if isinstance(tensor_i0, Tensor) else Tensor(
                dtype=tensor_i0.dtype,
                shape=[1],
                data=np.array([tensor_i0.value]).astype(tensor_i0.dtype),
                ttype="coeff")
            tensor_i0.quantization(scale=scale[0], zero_point=zero_point[0])
        if isinstance(tensor_i1, Tensor) or op_type != "top.Mul":
            tensor_i1 = tensor_i1 if isinstance(tensor_i1, Tensor) else Tensor(
                dtype=tensor_i1.dtype,
                shape=[1],
                data=np.array([tensor_i1.value]).astype(tensor_i1.dtype),
                ttype="coeff")
            tensor_i1.quantization(scale=scale[1], zero_point=zero_point[1])
        output.quantization(scale=scale[2], zero_point=zero_point[2])

    if isinstance(tensor_i0, Tensor) and isinstance(tensor_i1, Tensor):
        TpuLang.insert_op(op_type, [tensor_i0, tensor_i1], [output])
    else:
        tensor = tensor_i0 if isinstance(tensor_i0, Tensor) else tensor_i1
        scalar = tensor_i0 if isinstance(tensor_i1, Tensor) else tensor_i1

        if tensor == scalar:
            raise "input must be have Tensor"
        attr = {
            "const_val": Attr(scalar.value, 'float64'),
        }
        if is_reverse is not None:
            attr["is_reverse"] = Attr(is_reverse, 'bool')
        TpuLang.insert_op(op_type + "Const", [tensor], [output], params=attr)
    return output


@to_scalar(2)
@auto_name()
@annotation_check
@assert_with_out_name
def add(tensor_i0: Union[Tensor, Scalar, int, float],
        tensor_i1: Union[Tensor, Scalar, int, float],
        scale: List[float] = None,
        zero_point: List[int] = None,
        out_dtype: str = None,
        out_name: str = None):
    return _base_binary(tensor_i0,
                        tensor_i1,
                        "top.Add",
                        scale,
                        zero_point,
                        out_dtype=out_dtype,
                        out_name=out_name)


@to_scalar(2)
@auto_name()
@annotation_check
@assert_with_out_name
def mul(tensor_i0: Union[Tensor, Scalar, int, float],
        tensor_i1: Union[Tensor, Scalar, int, float],
        scale: List[float] = None,
        zero_point: List[int] = None,
        out_dtype: str = None,
        out_name: str = None):
    return _base_binary(tensor_i0,
                        tensor_i1,
                        "top.Mul",
                        scale,
                        zero_point,
                        out_dtype=out_dtype,
                        out_name=out_name)


@to_scalar(2)
@auto_name()
@annotation_check
@assert_with_out_name
def sub(tensor_i0: Union[Tensor, Scalar, int, float],
        tensor_i1: Union[Tensor, Scalar, int, float],
        scale: List[float] = None,
        zero_point: List[int] = None,
        out_dtype: str = None,
        out_name: str = None):
    is_reverse = None if isinstance(tensor_i0, Tensor) else True
    assert out_dtype != "uint8"
    return _base_binary(tensor_i0,
                        tensor_i1,
                        "top.Sub",
                        scale,
                        zero_point,
                        is_reverse=is_reverse,
                        out_dtype=out_dtype,
                        out_name=out_name)


@to_scalar(2)
@auto_name()
@annotation_check
@assert_with_out_name
def div(tensor_i0: Union[Tensor, Scalar], tensor_i1: Union[Tensor, Scalar], out_name: str = None):
    assert tensor_i0.dtype in ["float32", "float16"]
    assert tensor_i1.dtype in ["float32", "float16"]
    o_dtype = same_dtype_check(tensor_i0.dtype, tensor_i1.dtype)
    output = Tensor([], dtype=o_dtype, name=out_name)
    assert isinstance(tensor_i0, Tensor) or isinstance(tensor_i1, Tensor)
    if isinstance(tensor_i0, Tensor) and isinstance(tensor_i1, Tensor):
        TpuLang.insert_op("top.Div", [tensor_i0, tensor_i1], [output])
    else:
        if isinstance(tensor_i0, Tensor):
            output.reset()
            tensor_i1.value = 1 / tensor_i1.value
            return _base_binary(tensor_i0,
                                tensor_i1,
                                "top.Mul",
                                out_dtype=o_dtype,
                                out_name=out_name)
        else:
            tensor_i0 = Tensor(dtype=tensor_i0.dtype,
                               shape=[1],
                               data=np.array([tensor_i0.value]).astype(tensor_i0.dtype),
                               ttype="coeff")
            TpuLang.insert_op("top.Div", [tensor_i0, tensor_i1], [output])
    return output
    # is_reverse = None if isinstance(tensor_i0, Tensor) else True
    # return _base_binary(tensor_i0, tensor_i1, "top.Div", is_reverse=is_reverse, out_dtype=o_dtype, out_name=out_name)


@to_scalar(2)
@auto_name()
@annotation_check
@assert_with_out_name
def max(tensor_i0: Union[Tensor, Scalar, int, float],
        tensor_i1: Union[Tensor, Scalar, int, float],
        scale: List[float] = None,
        zero_point: List[int] = None,
        out_dtype: str = None,
        out_name: str = None):
    return _base_binary(tensor_i0,
                        tensor_i1,
                        "top.Max",
                        scale,
                        zero_point,
                        out_dtype=out_dtype,
                        out_name=out_name)


@to_scalar(2)
@auto_name()
@annotation_check
@assert_with_out_name
def min(tensor_i0: Union[Tensor, Scalar, int, float],
        tensor_i1: Union[Tensor, Scalar, int, float],
        scale: List[float] = None,
        zero_point: List[int] = None,
        out_dtype: str = None,
        out_name: str = None):
    return _base_binary(tensor_i0,
                        tensor_i1,
                        "top.Min",
                        scale,
                        zero_point,
                        out_dtype=out_dtype,
                        out_name=out_name)


def __binary_shift(tensor_i0: Tensor,
                   tensor_i1: Tensor,
                   type: str,
                   shift: int,
                   out_dtype: str,
                   is_reverse: bool = None,
                   saturation: bool = True,
                   round_mode: str = 'half_away_from_zero',
                   out_name: str = None):
    assert type in ["Add", "Sub", "Mul"]
    o_dtype = binary_dtype_check(tensor_i0,
                                 tensor_i1,
                                 out_dtype=out_dtype,
                                 sign=(type == "Sub"),
                                 is_shift=True)
    if out_name is None:
        out_name = generate_name(type)
    attr = {
        "mode": Attr(type, "string"),
        "shift": Attr(shift, "int32"),
        "saturation": Attr(saturation, "bool"),
        "round_mode": Attr(round_mode_convert(round_mode), "string"),
    }
    output = Tensor(dtype=o_dtype, name=out_name)
    if isinstance(tensor_i0, Tensor) and isinstance(tensor_i1, Tensor):
        TpuLang.insert_op("top.BinaryShift", [tensor_i0, tensor_i1], [output], params=attr)
    else:
        tensor = tensor_i0 if isinstance(tensor_i0, Tensor) else tensor_i1
        scalar = tensor_i0 if isinstance(tensor_i1, Tensor) else tensor_i1

        if tensor == scalar:
            raise "input must be have Tensor"
        attr["scale"] = Attr(scalar.value, 'int32')
        if is_reverse is not None:
            attr["is_reverse"] = Attr(is_reverse, 'bool')
        TpuLang.insert_op("top.BinaryConstShift", inputs=[tensor], outputs=[output], params=attr)
    return output


@to_scalar(2)
@auto_name()
@annotation_check
@assert_with_out_name
def add_shift(tensor_i0: Union[Tensor, Scalar, int],
              tensor_i1: Union[Tensor, Scalar, int],
              shift: int,
              out_dtype: str,
              round_mode: str = 'half_away_from_zero',
              is_saturate: bool = True,
              out_name: str = None):
    return __binary_shift(tensor_i0,
                          tensor_i1,
                          "Add",
                          shift,
                          out_dtype,
                          saturation=is_saturate,
                          round_mode=round_mode,
                          out_name=out_name)


@to_scalar(2)
@auto_name()
@annotation_check
@assert_with_out_name
def sub_shift(tensor_i0: Union[Tensor, Scalar, int],
              tensor_i1: Union[Tensor, Scalar, int],
              shift: int,
              out_dtype: str,
              round_mode: str = 'half_away_from_zero',
              is_saturate: bool = True,
              out_name: str = None):
    is_reverse = None if isinstance(tensor_i0, Tensor) else True
    return __binary_shift(tensor_i0,
                          tensor_i1,
                          "Sub",
                          shift,
                          out_dtype,
                          saturation=is_saturate,
                          is_reverse=is_reverse,
                          round_mode=round_mode,
                          out_name=out_name)


@to_scalar(2)
@auto_name()
@annotation_check
@assert_with_out_name
def mul_shift(tensor_i0: Union[Tensor, Scalar, int],
              tensor_i1: Union[Tensor, Scalar, int],
              shift: int,
              out_dtype: str,
              round_mode: str = 'half_away_from_zero',
              is_saturate: bool = True,
              out_name: str = None):
    return __binary_shift(tensor_i0,
                          tensor_i1,
                          "Mul",
                          shift,
                          out_dtype,
                          saturation=is_saturate,
                          round_mode=round_mode,
                          out_name=out_name)


# def add_shift(tensor_i0: Tensor,
#               tensor_i1: Tensor,
#               shift: int,
#               out_dtype: str = None,
#               out_name: str = None,
#               round_mode: str = 'half_up',
#               is_saturate: bool = True,):
#     o_dtype = "uint32"
#     if out_dtype is None:
#         o_dtype = tensor_i0.dtype
#     else:
#         o_dtype = out_dtype

#     output = Tensor(shape, dtype=o_dtype, name=out_name)

#     TpuLang.insert_op("top.Min", [tensor_i0, tensor_i1], [output])

#     return output


@auto_name()
@annotation_check
@assert_with_out_name
def copy(input: Tensor, out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    if out_name is None:
        out_name = generate_name("copy")
    attr = {
        "shape": ArrayAttr(input.shape),
        "input_stride": ArrayAttr([1] * (len(input.shape))),
        "output_stride": ArrayAttr([1] * (len(input.shape))),
    }
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Copy", [input], [output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def cast(tensor_i: Tensor,
         out_dtype: str = 'float32',
         out_name: str = None,
         round_mode: str = 'half_away_from_zero'):
    shape = tensor_i.shape
    attr = {
        "round_mode": Attr(round_mode_convert(round_mode), data_type="string"),
    }
    assert round_mode not in ["half_up", "half_down"], \
            f"cast does not suooprt round mode {round_mode}"
    output = Tensor(shape, dtype=out_dtype, name=out_name)
    TpuLang.insert_op("top.Cast", [tensor_i], [output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def clamp(input: Tensor, min: float, max: float, out_name: str = None):
    assert input.dtype in ["float32", "float16"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    assert max >= min
    attr = {
        "min": Attr(min, data_type="float64"),
        "max": Attr(max, data_type="float64"),
    }
    TpuLang.insert_op("top.Clip", [input], [output], params=attr)
    return output


###### quant operator ########
@auto_name()
@annotation_check
@assert_with_out_name
def requant_fp_to_int(tensor_i: Tensor,
                      scale: Union[float, List[float]],
                      offset: Union[int, List[int]],
                      requant_mode: int,
                      out_dtype: str,
                      out_name: str = None,
                      round_mode: str = 'half_away_from_zero'):
    output = Tensor(tensor_i.shape, name=out_name, dtype=out_dtype, scale=scale, zero_point=offset)
    attr = {
        "round_mode": Attr(round_mode_convert(round_mode), data_type="string"),
    }
    if isinstance(scale, List) and len(scale) > 1:
        assert round_mode not in ["half_up", "half_down"], \
                f"requant_fp_to_int per channel does not suooprt round mode {round_mode}"
    TpuLang.insert_op("top.Cast", inputs=[tensor_i], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def dequant_int_to_fp(tensor_i: Tensor,
                      scale: Union[float, List[float]],
                      offset: Union[int, List[int], float, List[float]],
                      out_dtype: str = "float32",
                      out_name: str = None,
                      round_mode: str = 'half_away_from_zero'):
    assert out_dtype in ["float32", "float16"]
    output = Tensor(tensor_i.shape, name=out_name, dtype=out_dtype)
    tensor_i.quantization(scale=scale, zero_point=offset)
    attr = {
        "round_mode": Attr(round_mode_convert(round_mode), data_type="string"),
    }
    if out_dtype == "float16":
        assert round_mode not in ["half_up", "half_down"], \
                f"dequant_int_to_fp float16 does not suooprt round mode {round_mode}"
    TpuLang.insert_op("top.Cast", inputs=[tensor_i], outputs=[output], params=attr)
    return output


def round_mode_convert(rmode: str):
    round_mode = "".join([r.capitalize() for r in rmode.split('_')])
    assert round_mode in [
        "HalfAwayFromZero", "HalfUp", "HalfDown", "HalfToEven", "TowardsZero", "Up", "Down"
    ], f"do not support round mode {rmode}"
    return round_mode


@auto_name()
@annotation_check
@assert_with_out_name
def requant_int(tensor_i: Tensor,
                mul: Union[int, List[int]],
                shift: Union[int, List[int]],
                offset: Union[int, List[int]],
                requant_mode: int,
                out_dtype: str = "int8",
                out_name=None,
                round_mode='half_away_from_zero',
                rq_axis: int = 1,
                fuse_rq_to_matmul: bool = False):
    assert requant_mode < 3
    q_mode = ["TFLite_LShift", "TFLite", "MultiplierShift"]
    output = Tensor(tensor_i.shape, name=out_name, dtype=out_dtype, zero_point=offset)
    mul = mul if isinstance(mul, List) else [mul]
    shift = shift if isinstance(shift, List) else [shift]
    shift = [-sft for sft in shift]
    attr = {
        "multiplier": ArrayAttr(mul, "int64"),
        "rshift": ArrayAttr(shift, "int64"),
        "quant_mode": Attr(q_mode[requant_mode], "string"),
        "round_mode": Attr(round_mode_convert(round_mode), "string"),
        "rq_axis": Attr(rq_axis, "int32"),
        "fuse_rq": Attr(fuse_rq_to_matmul, "bool")
    }
    TpuLang.insert_op("top.RequantInt", inputs=[tensor_i], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def dequant_int(tensor_i: Tensor,
                mul: Union[int, List[int]],
                shift: Union[int, List[int]],
                offset: Union[int, List[int]],
                lshift: int,
                requant_mode: int,
                out_dtype: str = "int8",
                out_name=None,
                round_mode='half_up'):
    assert requant_mode < 2
    q_mode = ["Normal", "TFLite"]
    output = Tensor(tensor_i.shape, name=out_name, dtype=out_dtype, zero_point=offset)
    mul = mul if isinstance(mul, List) else [mul]
    shift = shift if isinstance(shift, List) else [shift]
    attr = {
        "multiplier": ArrayAttr(mul, "int64"),
        "shift": ArrayAttr(shift, "int64"),
        "lshift": Attr(lshift, "int64"),
        "quant_mode": Attr(q_mode[requant_mode], "string"),
        "round_mode": Attr(round_mode_convert(round_mode), "string")
    }
    TpuLang.insert_op("top.DequantInt", inputs=[tensor_i], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def requant_fp(tensor_i: Tensor,
               scale: Union[float, List[float]],
               offset: Union[float, List[float]],
               out_dtype: str,
               out_name: str = None,
               round_mode: str = 'half_away_from_zero',
               first_round_mode: str = 'half_away_from_zero'):
    scale = scale if isinstance(scale, List) else [scale]
    offset = offset if isinstance(offset, List) else [offset]
    output = Tensor(tensor_i.shape, name=out_name, dtype=out_dtype, zero_point=(int)(offset[0]))
    if round_mode == "half_up":  # half_up(x) = down(x + 0.5)
        round_mode = "down"
        offset = [of + 0.5 for of in offset]
    elif round_mode == "half_down":  # half_down(x) = up(x - 0.5)
        round_mode = "up"
        offset = [of - 0.5 for of in offset]
    attr = {
        "scale": ArrayAttr(scale, "float64"),
        "offset": ArrayAttr(offset, "float64"),
        "quant_mode": Attr("MultiplierShift", "string"),
        "round_mode": Attr(round_mode_convert(round_mode), "string"),
        "first_round_mode": Attr(round_mode_convert(first_round_mode), "string"),
    }
    TpuLang.insert_op("top.RequantFp", inputs=[tensor_i], outputs=[output], params=attr)
    return output


def unpack_weights(qweight, qzeros, bits):
    dtype = np.int32
    compress_ratio = 32 // bits
    mask = 0xF if bits == 4 else 0xFF

    K, N = qweight.shape
    unpacked_weights = np.zeros((K * compress_ratio, N), dtype=dtype)
    pack_int8_weights = np.zeros((K * compress_ratio // 2, N),
                                 dtype=np.uint8)  # dtype=int8 to fit tpu.a16matmul

    Kz, Nz = qzeros.shape
    unpacked_zeros = np.zeros((Kz, Nz * compress_ratio),
                              dtype=np.uint8)  # dtype=int8 to fit tpu.a16matmul

    for row in range(unpacked_weights.shape[0]):
        i = row % compress_ratio
        unpacked_weights[row, :] = (qweight[row // compress_ratio, :] >> (bits * i)) & mask

        if bits == 4:
            if row % 2 == 0:
                pack_int8_weights[row // 2, :] = unpacked_weights[row, :]
            else:
                pack_int8_weights[row //
                                  2, :] = unpacked_weights[row, :] << 4 | pack_int8_weights[row //
                                                                                            2, :]

    for col in range(unpacked_zeros.shape[1]):
        i = col % compress_ratio
        unpacked_zeros[:, col] = (qzeros[:, col // compress_ratio] >> (bits * i)) & mask

    if bits == 8:
        pack_int8_weights = unpacked_weights.astype("uint8")
    return unpacked_weights, pack_int8_weights, unpacked_zeros + 1


def dequantize_weight(qweight, qzeros, scales, bits, group_size):
    unpacked_qweight, pack_int8_weights, unpacked_qzeros = unpack_weights(qweight, qzeros, bits)

    assert group_size == unpacked_qweight.shape[0] // scales.shape[0], (
        "group_size does not match the shape of unpacked weights and scales.")

    scales_expanded = np.repeat(scales, group_size, axis=0)
    zeros_expanded = np.repeat(unpacked_qzeros, group_size, axis=0)

    dequantized = (unpacked_qweight - zeros_expanded) * scales_expanded

    return dequantized.T, zeros_expanded, pack_int8_weights, unpacked_qzeros


@auto_name()
@annotation_check
@assert_with_out_name
def a16matmul(
        input: Tensor,
        weight: Tensor,
        scale: Tensor,
        zp: Tensor,
        bias: Tensor = None,
        right_transpose: bool = True,
        out_dtype: str = 'float16',
        out_name: str = None,
        group_size: int = 128,
        bits: int = 4,
        g_idx: Tensor = None,  # TODO: formatted as [0, ..., n] in GPTQ, do not support shuffled
):
    attr = {
        "right_transpose": Attr(right_transpose, "bool"),
        "q_group_size": Attr(group_size, "int64"),
        "weight_bits": Attr(bits, "int64")
    }
    assert input.dtype in [
        "float32", "float16"
    ] and weight.dtype == "int32" and zp.dtype == "int32" and out_dtype in ["float32", "float16"]
    assert bits in [4, 8]

    # weight.buffer shape = [K,N]
    qweight_expanded, zeros_expanded, pack_int8_weights, unpacked_qzeros = dequantize_weight(
        weight.buffer, zp.buffer, scale.buffer, bits, group_size)

    # golden = input.buffer @ qweight_expanded.T + bias.buffer
    # np.savez("/workspace/tpu-mlir/qwen2_vl/final/tpulang_test_bm1684x/A16Matmul/output.npz", output=golden)

    # fit tpu.mlir shape and dtype
    weight.update(pack_int8_weights.T, pack_int8_weights.T.shape)
    scale.update(scale.buffer.T, scale.buffer.T.shape)
    zp.update(unpacked_qzeros.T, unpacked_qzeros.T.shape)
    weight.dtype = "uint8"
    zp.dtype = "uint8"

    output = Tensor(dtype=out_dtype, name=out_name)
    inputs = [input, weight, scale, zp, bias]
    TpuLang.insert_op("top.A16MatMul", inputs=inputs, outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def qwen2_block(hidden_states: Tensor,
                position_ids: Tensor,
                attention_mask: Tensor,
                q_proj_weights: Tensor,
                q_proj_scales: Tensor,
                q_proj_zps: Tensor,
                q_proj_bias: Tensor,
                k_proj_weights: Tensor,
                k_proj_scales: Tensor,
                k_proj_zps: Tensor,
                k_proj_bias: Tensor,
                v_proj_weights: Tensor,
                v_proj_scales: Tensor,
                v_proj_zps: Tensor,
                v_proj_bias: Tensor,
                o_proj_weights: Tensor,
                o_proj_scales: Tensor,
                o_proj_zps: Tensor,
                o_proj_bias: Tensor,
                down_proj_weights: Tensor,
                down_proj_scales: Tensor,
                down_proj_zps: Tensor,
                gate_proj_weights: Tensor,
                gate_proj_scales: Tensor,
                gate_proj_zps: Tensor,
                up_proj_weights: Tensor,
                up_proj_scales: Tensor,
                up_proj_zps: Tensor,
                input_layernorm_weight: Tensor,
                post_attention_layernorm_weight: Tensor,
                cos: List[Tensor],
                sin: List[Tensor],
                out_dtype: str = 'float16',
                group_size: int = 128,
                weight_bits: int = 4,
                hidden_size: int = 3584,
                rms_norm_eps: float = 1e-06,
                num_attention_heads: int = 28,
                num_key_value_heads: int = 4,
                mrope_section: List[int] = [16, 24, 24],
                quant_method: str = "gptq",
                out_name: str = None):
    if out_name is None:
        out_name = generate_name("qwen2_block")

    # Only support gptq for now
    assert quant_method == "gptq"

    head_dim = hidden_size // num_attention_heads
    num_key_value_groups = num_attention_heads // num_key_value_heads
    bsz, q_len, _ = hidden_states.shape

    assert hidden_states.dtype in ["float32", "float16"]
    assert out_dtype in ["float32", "float16"]
    assert hidden_states.dtype == input_layernorm_weight.dtype
    assert hidden_states.shape[-1] == input_layernorm_weight.shape[0] and len(
        input_layernorm_weight.shape) == 1

    # input layernorm
    input_layernorm_attr = {"eps": Attr(rms_norm_eps, "float64")}
    input_layernorm_output = Tensor(dtype=out_dtype, name=out_name + "_input_layernorm_output")
    TpuLang.insert_op("top.RMSNorm",
                      inputs=[hidden_states, input_layernorm_weight],
                      outputs=[input_layernorm_output],
                      params=input_layernorm_attr)

    ############################## attention block ##############################
    # q/k/v proj matmul
    q_proj = a16matmul(input_layernorm_output,
                       q_proj_weights,
                       q_proj_scales,
                       q_proj_zps,
                       q_proj_bias,
                       right_transpose=True,
                       out_dtype=out_dtype,
                       out_name=out_name + "_q_proj",
                       group_size=group_size,
                       bits=weight_bits)
    k_proj = a16matmul(input_layernorm_output,
                       k_proj_weights,
                       k_proj_scales,
                       k_proj_zps,
                       k_proj_bias,
                       right_transpose=True,
                       out_dtype=out_dtype,
                       out_name=out_name + "_k_proj",
                       group_size=group_size,
                       bits=weight_bits)
    v_proj = a16matmul(input_layernorm_output,
                       v_proj_weights,
                       v_proj_scales,
                       v_proj_zps,
                       v_proj_bias,
                       right_transpose=True,
                       out_dtype=out_dtype,
                       out_name=out_name + "_v_proj",
                       group_size=group_size,
                       bits=weight_bits)

    # reshape q/k/v
    reshape_q_attr = {
        "shape": ArrayAttr([bsz, q_len, num_attention_heads, head_dim]),
    }
    reshape_kv_attr = {
        "shape": ArrayAttr([bsz, q_len, num_key_value_heads, head_dim]),
    }
    q_proj_reshape = Tensor(dtype=out_dtype, name=out_name + "_q_proj_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[q_proj],
                      outputs=[q_proj_reshape],
                      params=reshape_q_attr)
    k_proj_reshape = Tensor(dtype=out_dtype, name=out_name + "_k_proj_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[k_proj],
                      outputs=[k_proj_reshape],
                      params=reshape_kv_attr)
    v_proj_reshape = Tensor(dtype=out_dtype, name=out_name + "_v_proj_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[v_proj],
                      outputs=[v_proj_reshape],
                      params=reshape_kv_attr)

    # rotary pos emb
    mrope_section = 2 * mrope_section
    rope_offset = [0] * 3
    int64_max = np.iinfo(np.int64).max
    int32_max = np.iinfo(np.int32).max
    ends = [int64_max] * 3
    steps = [1] * 3
    cos_gather_out_list = []
    sin_gather_out_list = []
    for i in range(len(mrope_section) // 2):
        rope_offset[0] = i
        ends[0] = i + 1
        rope_slice_attr = {
            "offset": ArrayAttr(rope_offset),
            "ends": ArrayAttr(ends),
            "steps": ArrayAttr(steps),
            "hasparamConvert_axes": ArrayAttr([0])
        }
        rope_slice_out = Tensor(dtype=out_dtype, name=out_name + "_rope_slice_{}".format(i))
        TpuLang.insert_op("top.Slice",
                          inputs=[position_ids, None, None, None],
                          outputs=[rope_slice_out],
                          params=rope_slice_attr)

        rope_slice_attr = {"axes": ArrayAttr([0])}
        rope_slice_squeeze_out = Tensor(dtype=out_dtype,
                                        name=out_name + "_rope_slice_squeeze_{}".format(i))
        TpuLang.insert_op("top.Squeeze",
                          inputs=[rope_slice_out],
                          outputs=[rope_slice_squeeze_out],
                          params=rope_slice_attr)

        gather_attr = {
            "axis": Attr(0, "int32"),
        }
        cos_gather_out = Tensor(dtype=out_dtype, name=out_name + "_cos_gather_{}".format(i))
        TpuLang.insert_op("top.Gather",
                          inputs=[cos[i], rope_slice_squeeze_out],
                          outputs=[cos_gather_out],
                          params=gather_attr)
        sin_gather_out = Tensor(dtype=out_dtype, name=out_name + "_sin_gather_{}".format(i))
        TpuLang.insert_op("top.Gather",
                          inputs=[sin[i], rope_slice_squeeze_out],
                          outputs=[sin_gather_out],
                          params=gather_attr)

        cos_gather_out_list.append(cos_gather_out)
        sin_gather_out_list.append(sin_gather_out)

    concat_attr = {
        "axis": Attr(3, "int32"),
    }
    cos_concat = Tensor(dtype=out_dtype, name=out_name + "_cos_concat")
    TpuLang.insert_op("top.Concat",
                      inputs=cos_gather_out_list * 2,
                      outputs=[cos_concat],
                      params=concat_attr)
    sin_concat = Tensor(dtype=out_dtype, name=out_name + "_sin_concat")
    TpuLang.insert_op("top.Concat",
                      inputs=sin_gather_out_list * 2,
                      outputs=[sin_concat],
                      params=concat_attr)

    # rotate_half k/q
    # q
    q_slice_attr_1 = {
        "offset": ArrayAttr([0, 0, 0, 0]),
        "ends": ArrayAttr([1, q_len, num_attention_heads, head_dim // 2]),
        "steps": ArrayAttr([1, 1, 1, 1]),
        "hasparamConvert_axes": ArrayAttr([3])
    }
    q_slice_out_1 = Tensor(dtype=out_dtype, name=out_name + "_q_slice_1")
    TpuLang.insert_op("top.Slice",
                      inputs=[q_proj_reshape, None, None, None],
                      outputs=[q_slice_out_1],
                      params=q_slice_attr_1)

    q_slice_attr_2 = {
        "offset": ArrayAttr([0, 0, 0, head_dim // 2]),
        "ends": ArrayAttr([1, q_len, num_attention_heads, int32_max - 1024]),
        "steps": ArrayAttr([1, 1, 1, 1]),
        "hasparamConvert_axes": ArrayAttr([3])
    }
    q_slice_out_2 = Tensor(dtype=out_dtype, name=out_name + "_q_slice_2")
    TpuLang.insert_op("top.Slice",
                      inputs=[q_proj_reshape, None, None, None],
                      outputs=[q_slice_out_2],
                      params=q_slice_attr_2)

    mulconst_attr = {
        "const_val": Attr(-1.0, "float64"),
    }
    q_slice_out_2_mulconst = Tensor(dtype=out_dtype, name=out_name + "_q_slice_2_mulconst")
    TpuLang.insert_op("top.MulConst",
                      inputs=[q_slice_out_2],
                      outputs=[q_slice_out_2_mulconst],
                      params=mulconst_attr)

    concat_attr = {
        "axis": Attr(3, "int32"),
    }
    q_rotate_half_out = Tensor(dtype=out_dtype, name=out_name + "_q_rotate_half_out")
    TpuLang.insert_op("top.Concat",
                      inputs=[q_slice_out_2_mulconst, q_slice_out_1],
                      outputs=[q_rotate_half_out],
                      params=concat_attr)

    # k
    k_slice_attr_1 = {
        "offset": ArrayAttr([0, 0, 0, 0]),
        "ends": ArrayAttr([1, q_len, num_key_value_heads, head_dim // 2]),
        "steps": ArrayAttr([1, 1, 1, 1]),
        "hasparamConvert_axes": ArrayAttr([3])
    }
    k_slice_out_1 = Tensor(dtype=out_dtype, name=out_name + "_k_slice_1")
    TpuLang.insert_op("top.Slice",
                      inputs=[k_proj_reshape, None, None, None],
                      outputs=[k_slice_out_1],
                      params=k_slice_attr_1)

    k_slice_attr_2 = {
        "offset": ArrayAttr([0, 0, 0, head_dim // 2]),
        "ends": ArrayAttr([1, q_len, num_key_value_heads, int32_max - 1024]),
        "steps": ArrayAttr([1, 1, 1, 1]),
        "hasparamConvert_axes": ArrayAttr([3])
    }
    k_slice_out_2 = Tensor(dtype=out_dtype, name=out_name + "_k_slice_2")
    TpuLang.insert_op("top.Slice",
                      inputs=[k_proj_reshape, None, None, None],
                      outputs=[k_slice_out_2],
                      params=k_slice_attr_2)

    mulconst_attr = {
        "const_val": Attr(-1.0, "float64"),
    }
    k_slice_out_2_mulconst = Tensor(dtype=out_dtype, name=out_name + "_k_slice_2_mulconst")
    TpuLang.insert_op("top.MulConst",
                      inputs=[k_slice_out_2],
                      outputs=[k_slice_out_2_mulconst],
                      params=mulconst_attr)

    concat_attr = {
        "axis": Attr(3, "int32"),
    }
    k_rotate_half_out = Tensor(dtype=out_dtype, name=out_name + "_k_rotate_half_out")
    TpuLang.insert_op("top.Concat",
                      inputs=[k_slice_out_2_mulconst, k_slice_out_1],
                      outputs=[k_rotate_half_out],
                      params=concat_attr)

    # q/k embed
    # q
    q_cos_mul = Tensor(dtype=out_dtype, name=out_name + "_q_cos_mul")
    TpuLang.insert_op("top.Mul", inputs=[q_proj_reshape, cos_concat], outputs=[q_cos_mul])

    q_rotate_sin_mul = Tensor(dtype=out_dtype, name=out_name + "_q_rotate_sin_mul")
    TpuLang.insert_op("top.Mul", inputs=[q_rotate_half_out, sin_concat], outputs=[q_rotate_sin_mul])

    q_embed = Tensor(dtype=out_dtype, name=out_name + "_q_embed")
    TpuLang.insert_op("top.Add", inputs=[q_cos_mul, q_rotate_sin_mul], outputs=[q_embed])

    # k
    k_cos_mul = Tensor(dtype=out_dtype, name=out_name + "_k_cos_mul")
    TpuLang.insert_op("top.Mul", inputs=[k_proj_reshape, cos_concat], outputs=[k_cos_mul])

    k_rotate_sin_mul = Tensor(dtype=out_dtype, name=out_name + "_k_rotate_sin_mul")
    TpuLang.insert_op("top.Mul", inputs=[k_rotate_half_out, sin_concat], outputs=[k_rotate_sin_mul])

    k_embed = Tensor(dtype=out_dtype, name=out_name + "_k_embed")
    TpuLang.insert_op("top.Add", inputs=[k_cos_mul, k_rotate_sin_mul], outputs=[k_embed])

    # repeat_kv
    # k
    k_permute = permute(k_embed, [0, 2, 1, 3], out_name=out_name + "_k_permute")
    k_unsqueeze = unsqueeze(k_permute, [2], out_name=out_name + "_k_unsqueeze")

    attr_bc = {"shape": ArrayAttr([1, num_key_value_heads, num_key_value_groups, q_len, head_dim])}
    k_expand = Tensor(dtype=out_dtype, name=out_name + "_k_expand")
    TpuLang.insert_op("top.Expand", inputs=[k_unsqueeze], outputs=[k_expand], params=attr_bc)

    reshape_attr = {
        "shape": ArrayAttr([1, num_key_value_heads * num_key_value_groups, q_len, head_dim]),
    }
    k_expand_reshape = Tensor(dtype=out_dtype, name=out_name + "_k_expand_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[k_expand],
                      outputs=[k_expand_reshape],
                      params=reshape_attr)

    # v
    v_permute = permute(v_proj_reshape, [0, 2, 1, 3], out_name=out_name + "_v_permute")
    v_unsqueeze = unsqueeze(v_permute, [2], out_name=out_name + "_v_unsqueeze")

    v_expand = Tensor(dtype=out_dtype, name=out_name + "_v_expand")
    TpuLang.insert_op("top.Expand", inputs=[v_unsqueeze], outputs=[v_expand], params=attr_bc)

    reshape_attr = {
        "shape": ArrayAttr([1, num_key_value_heads * num_key_value_groups, q_len, head_dim]),
    }
    v_expand_reshape = Tensor(dtype=out_dtype, name=out_name + "_v_expand_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[v_expand],
                      outputs=[v_expand_reshape],
                      params=reshape_attr)

    # q * k
    q_permute = permute(q_embed, [0, 2, 1, 3], out_name=out_name + "_q_permute")
    k_expand_permute = permute(k_expand_reshape, [0, 1, 3, 2],
                               out_name=out_name + "_k_expand_permute")
    attention_weight = matmul(q_permute,
                              k_expand_permute,
                              out_dtype=out_dtype,
                              out_name=out_name + "_attention_weight")

    mulconst_attr = {"const_val": Attr(1 / math.sqrt(head_dim), "float64")}
    attention_weight_mulconst = Tensor(dtype=out_dtype,
                                       name=out_name + "_attention_weight_mulconst")
    TpuLang.insert_op("top.MulConst",
                      inputs=[attention_weight],
                      outputs=[attention_weight_mulconst],
                      params=mulconst_attr)

    # add attn mask
    attention_weight_mask = Tensor(dtype=out_dtype, name=out_name + "_attention_weight_mask")
    TpuLang.insert_op("top.Add",
                      inputs=[attention_weight_mulconst, attention_mask],
                      outputs=[attention_weight_mask])

    # softmax
    softmax_out = softmax(attention_weight_mask, 3, out_name=out_name + "_softmax")

    # softmax * v
    softmax_v = matmul(softmax_out,
                       v_expand_reshape,
                       out_dtype=out_dtype,
                       out_name=out_name + "_softmax_v")

    # o_proj
    softmax_v_permute = permute(softmax_v, [0, 2, 1, 3], out_name=out_name + "_softmax_v_permute")

    softmax_reshape_attr = {
        "shape": ArrayAttr([bsz, q_len, -1]),
    }
    softmax_v_reshape = Tensor(dtype=out_dtype, name=out_name + "_softmax_v_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[softmax_v_permute],
                      outputs=[softmax_v_reshape],
                      params=softmax_reshape_attr)

    attention_out = a16matmul(softmax_v_reshape,
                              o_proj_weights,
                              o_proj_scales,
                              o_proj_zps,
                              o_proj_bias,
                              right_transpose=True,
                              out_dtype=out_dtype,
                              out_name=out_name + "_o_proj",
                              group_size=group_size,
                              bits=weight_bits)

    ############################## mlp block ##############################
    # add residual
    attention_out_add = Tensor(dtype=out_dtype, name=out_name + "_attention_out_add")
    TpuLang.insert_op("top.Add", inputs=[hidden_states, attention_out], outputs=[attention_out_add])

    # post layernorm
    post_layernorm_attr = {"eps": Attr(rms_norm_eps, "float64")}
    post_layernorm_output = Tensor(dtype=out_dtype, name=out_name + "_post_layernorm_output")
    TpuLang.insert_op("top.RMSNorm",
                      inputs=[attention_out_add, post_attention_layernorm_weight],
                      outputs=[post_layernorm_output],
                      params=post_layernorm_attr)

    # mlp
    gate_proj = a16matmul(post_layernorm_output,
                          gate_proj_weights,
                          gate_proj_scales,
                          gate_proj_zps,
                          right_transpose=True,
                          out_dtype=out_dtype,
                          out_name=out_name + "_gate_proj",
                          group_size=group_size,
                          bits=weight_bits)
    silu_out = silu(gate_proj, out_name="_silu_out")
    up_proj = a16matmul(post_layernorm_output,
                        up_proj_weights,
                        up_proj_scales,
                        up_proj_zps,
                        right_transpose=True,
                        out_dtype=out_dtype,
                        out_name=out_name + "_up_proj",
                        group_size=group_size,
                        bits=weight_bits)

    gate_up_mul = Tensor(dtype=out_dtype, name=out_name + "_gate_up_mul")
    TpuLang.insert_op("top.Mul", inputs=[silu_out, up_proj], outputs=[gate_up_mul])

    down_proj = a16matmul(gate_up_mul,
                          down_proj_weights,
                          down_proj_scales,
                          down_proj_zps,
                          right_transpose=True,
                          out_dtype=out_dtype,
                          out_name=out_name + "_down_proj",
                          group_size=group_size,
                          bits=weight_bits)

    # add residual
    mlp_out_add = Tensor(dtype=out_dtype, name=out_name + "_mlp_out_add")
    TpuLang.insert_op("top.Add", inputs=[attention_out_add, down_proj], outputs=[mlp_out_add])

    return mlp_out_add, k_embed, v_proj_reshape


@auto_name()
@annotation_check
@assert_with_out_name
def qwen2_block_cache(hidden_states: Tensor,
                      position_ids: Tensor,
                      attention_mask: Tensor,
                      k_cache: Tensor,
                      v_cache: Tensor,
                      q_proj_weights: Tensor,
                      q_proj_scales: Tensor,
                      q_proj_zps: Tensor,
                      q_proj_bias: Tensor,
                      k_proj_weights: Tensor,
                      k_proj_scales: Tensor,
                      k_proj_zps: Tensor,
                      k_proj_bias: Tensor,
                      v_proj_weights: Tensor,
                      v_proj_scales: Tensor,
                      v_proj_zps: Tensor,
                      v_proj_bias: Tensor,
                      o_proj_weights: Tensor,
                      o_proj_scales: Tensor,
                      o_proj_zps: Tensor,
                      o_proj_bias: Tensor,
                      down_proj_weights: Tensor,
                      down_proj_scales: Tensor,
                      down_proj_zps: Tensor,
                      gate_proj_weights: Tensor,
                      gate_proj_scales: Tensor,
                      gate_proj_zps: Tensor,
                      up_proj_weights: Tensor,
                      up_proj_scales: Tensor,
                      up_proj_zps: Tensor,
                      input_layernorm_weight: Tensor,
                      post_attention_layernorm_weight: Tensor,
                      cos: List[Tensor],
                      sin: List[Tensor],
                      out_dtype: str = 'float16',
                      group_size: int = 128,
                      weight_bits: int = 4,
                      hidden_size: int = 3584,
                      rms_norm_eps: float = 1e-06,
                      num_attention_heads: int = 28,
                      num_key_value_heads: int = 4,
                      mrope_section: List[int] = [16, 24, 24],
                      quant_method: str = "gptq",
                      out_name: str = None):
    if out_name is None:
        out_name = generate_name("qwen2_block_cache")

    # Only support gptq for now
    assert quant_method == "gptq"

    head_dim = hidden_size // num_attention_heads
    num_key_value_groups = num_attention_heads // num_key_value_heads
    bsz, q_len, _ = hidden_states.shape
    _, cache_len, _, _ = k_cache.shape

    concat_len = q_len + cache_len

    assert hidden_states.dtype in ["float32", "float16"]
    assert hidden_states.dtype == input_layernorm_weight.dtype
    assert hidden_states.shape[-1] == input_layernorm_weight.shape[0] and len(
        input_layernorm_weight.shape) == 1

    # input layernorm
    input_layernorm_attr = {"eps": Attr(rms_norm_eps, "float64")}
    input_layernorm_output = Tensor(dtype=out_dtype, name=out_name + "_input_layernorm_output")
    TpuLang.insert_op("top.RMSNorm",
                      inputs=[hidden_states, input_layernorm_weight],
                      outputs=[input_layernorm_output],
                      params=input_layernorm_attr)

    ############################## attention block ##############################
    # q/k/v proj matmul
    q_proj = a16matmul(input_layernorm_output,
                       q_proj_weights,
                       q_proj_scales,
                       q_proj_zps,
                       q_proj_bias,
                       right_transpose=True,
                       out_dtype=out_dtype,
                       out_name=out_name + "_q_proj",
                       group_size=group_size,
                       bits=weight_bits)
    k_proj = a16matmul(input_layernorm_output,
                       k_proj_weights,
                       k_proj_scales,
                       k_proj_zps,
                       k_proj_bias,
                       right_transpose=True,
                       out_dtype=out_dtype,
                       out_name=out_name + "_k_proj",
                       group_size=group_size,
                       bits=weight_bits)
    v_proj = a16matmul(input_layernorm_output,
                       v_proj_weights,
                       v_proj_scales,
                       v_proj_zps,
                       v_proj_bias,
                       right_transpose=True,
                       out_dtype=out_dtype,
                       out_name=out_name + "_v_proj",
                       group_size=group_size,
                       bits=weight_bits)

    # reshape q/k/v
    reshape_q_attr = {
        "shape": ArrayAttr([bsz, q_len, num_attention_heads, head_dim]),
    }
    reshape_kv_attr = {
        "shape": ArrayAttr([bsz, q_len, num_key_value_heads, head_dim]),
    }
    q_proj_reshape = Tensor(dtype=out_dtype, name=out_name + "_q_proj_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[q_proj],
                      outputs=[q_proj_reshape],
                      params=reshape_q_attr)
    k_proj_reshape = Tensor(dtype=out_dtype, name=out_name + "_k_proj_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[k_proj],
                      outputs=[k_proj_reshape],
                      params=reshape_kv_attr)
    v_proj_reshape = Tensor(dtype=out_dtype, name=out_name + "_v_proj_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[v_proj],
                      outputs=[v_proj_reshape],
                      params=reshape_kv_attr)

    # rotary pos emb
    mrope_section = 2 * mrope_section
    rope_offset = [0] * 3
    int64_max = np.iinfo(np.int64).max
    int32_max = np.iinfo(np.int32).max
    ends = [int64_max] * 3
    steps = [1] * 3
    cos_gather_out_list = []
    sin_gather_out_list = []
    for i in range(len(mrope_section) // 2):
        rope_offset[0] = i
        ends[0] = i + 1
        rope_slice_attr = {
            "offset": ArrayAttr(rope_offset),
            "ends": ArrayAttr(ends),
            "steps": ArrayAttr(steps),
            "hasparamConvert_axes": ArrayAttr([0])
        }
        rope_slice_out = Tensor(dtype=out_dtype, name=out_name + "_rope_slice_{}".format(i))
        TpuLang.insert_op("top.Slice",
                          inputs=[position_ids, None, None, None],
                          outputs=[rope_slice_out],
                          params=rope_slice_attr)

        rope_slice_attr = {"axes": ArrayAttr([0])}
        rope_slice_squeeze_out = Tensor(dtype=out_dtype,
                                        name=out_name + "_rope_slice_squeeze_{}".format(i))
        TpuLang.insert_op("top.Squeeze",
                          inputs=[rope_slice_out],
                          outputs=[rope_slice_squeeze_out],
                          params=rope_slice_attr)

        gather_attr = {
            "axis": Attr(0, "int32"),
        }
        cos_gather_out = Tensor(dtype=out_dtype, name=out_name + "_cos_gather_{}".format(i))
        TpuLang.insert_op("top.Gather",
                          inputs=[cos[i], rope_slice_squeeze_out],
                          outputs=[cos_gather_out],
                          params=gather_attr)
        sin_gather_out = Tensor(dtype=out_dtype, name=out_name + "_sin_gather_{}".format(i))
        TpuLang.insert_op("top.Gather",
                          inputs=[sin[i], rope_slice_squeeze_out],
                          outputs=[sin_gather_out],
                          params=gather_attr)

        cos_gather_out_list.append(cos_gather_out)
        sin_gather_out_list.append(sin_gather_out)

    concat_attr = {
        "axis": Attr(3, "int32"),
    }
    cos_concat = Tensor(dtype=out_dtype, name=out_name + "_cos_concat")
    TpuLang.insert_op("top.Concat",
                      inputs=cos_gather_out_list * 2,
                      outputs=[cos_concat],
                      params=concat_attr)
    sin_concat = Tensor(dtype=out_dtype, name=out_name + "_sin_concat")
    TpuLang.insert_op("top.Concat",
                      inputs=sin_gather_out_list * 2,
                      outputs=[sin_concat],
                      params=concat_attr)

    # rotate_half k/q
    # q
    q_slice_attr_1 = {
        "offset": ArrayAttr([0, 0, 0, 0]),
        "ends": ArrayAttr([1, q_len, num_attention_heads, head_dim // 2]),
        "steps": ArrayAttr([1, 1, 1, 1]),
        "hasparamConvert_axes": ArrayAttr([3])
    }
    q_slice_out_1 = Tensor(dtype=out_dtype, name=out_name + "_q_slice_1")
    TpuLang.insert_op("top.Slice",
                      inputs=[q_proj_reshape, None, None, None],
                      outputs=[q_slice_out_1],
                      params=q_slice_attr_1)

    q_slice_attr_2 = {
        "offset": ArrayAttr([0, 0, 0, head_dim // 2]),
        "ends": ArrayAttr([1, q_len, num_attention_heads, int32_max - 1024]),
        "steps": ArrayAttr([1, 1, 1, 1]),
        "hasparamConvert_axes": ArrayAttr([3])
    }
    q_slice_out_2 = Tensor(dtype=out_dtype, name=out_name + "_q_slice_2")
    TpuLang.insert_op("top.Slice",
                      inputs=[q_proj_reshape, None, None, None],
                      outputs=[q_slice_out_2],
                      params=q_slice_attr_2)

    mulconst_attr = {
        "const_val": Attr(-1.0, "float64"),
    }
    q_slice_out_2_mulconst = Tensor(dtype=out_dtype, name=out_name + "_q_slice_2_mulconst")
    TpuLang.insert_op("top.MulConst",
                      inputs=[q_slice_out_2],
                      outputs=[q_slice_out_2_mulconst],
                      params=mulconst_attr)

    concat_attr = {
        "axis": Attr(3, "int32"),
    }
    q_rotate_half_out = Tensor(dtype=out_dtype, name=out_name + "_q_rotate_half_out")
    TpuLang.insert_op("top.Concat",
                      inputs=[q_slice_out_2_mulconst, q_slice_out_1],
                      outputs=[q_rotate_half_out],
                      params=concat_attr)

    # k
    k_slice_attr_1 = {
        "offset": ArrayAttr([0, 0, 0, 0]),
        "ends": ArrayAttr([1, q_len, num_key_value_heads, head_dim // 2]),
        "steps": ArrayAttr([1, 1, 1, 1]),
        "hasparamConvert_axes": ArrayAttr([3])
    }
    k_slice_out_1 = Tensor(dtype=out_dtype, name=out_name + "_k_slice_1")
    TpuLang.insert_op("top.Slice",
                      inputs=[k_proj_reshape, None, None, None],
                      outputs=[k_slice_out_1],
                      params=k_slice_attr_1)

    k_slice_attr_2 = {
        "offset": ArrayAttr([0, 0, 0, head_dim // 2]),
        "ends": ArrayAttr([1, q_len, num_key_value_heads, int32_max - 1024]),
        "steps": ArrayAttr([1, 1, 1, 1]),
        "hasparamConvert_axes": ArrayAttr([3])
    }
    k_slice_out_2 = Tensor(dtype=out_dtype, name=out_name + "_k_slice_2")
    TpuLang.insert_op("top.Slice",
                      inputs=[k_proj_reshape, None, None, None],
                      outputs=[k_slice_out_2],
                      params=k_slice_attr_2)

    mulconst_attr = {
        "const_val": Attr(-1.0, "float64"),
    }
    k_slice_out_2_mulconst = Tensor(dtype=out_dtype, name=out_name + "_k_slice_2_mulconst")
    TpuLang.insert_op("top.MulConst",
                      inputs=[k_slice_out_2],
                      outputs=[k_slice_out_2_mulconst],
                      params=mulconst_attr)

    concat_attr = {
        "axis": Attr(3, "int32"),
    }
    k_rotate_half_out = Tensor(dtype=out_dtype, name=out_name + "_k_rotate_half_out")
    TpuLang.insert_op("top.Concat",
                      inputs=[k_slice_out_2_mulconst, k_slice_out_1],
                      outputs=[k_rotate_half_out],
                      params=concat_attr)

    # q/k embed
    # q
    q_cos_mul = Tensor(dtype=out_dtype, name=out_name + "_q_cos_mul")
    TpuLang.insert_op("top.Mul", inputs=[q_proj_reshape, cos_concat], outputs=[q_cos_mul])

    q_rotate_sin_mul = Tensor(dtype=out_dtype, name=out_name + "_q_rotate_sin_mul")
    TpuLang.insert_op("top.Mul", inputs=[q_rotate_half_out, sin_concat], outputs=[q_rotate_sin_mul])

    q_embed = Tensor(dtype=out_dtype, name=out_name + "_q_embed")
    TpuLang.insert_op("top.Add", inputs=[q_cos_mul, q_rotate_sin_mul], outputs=[q_embed])

    # k
    k_cos_mul = Tensor(dtype=out_dtype, name=out_name + "_k_cos_mul")
    TpuLang.insert_op("top.Mul", inputs=[k_proj_reshape, cos_concat], outputs=[k_cos_mul])

    k_rotate_sin_mul = Tensor(dtype=out_dtype, name=out_name + "_k_rotate_sin_mul")
    TpuLang.insert_op("top.Mul", inputs=[k_rotate_half_out, sin_concat], outputs=[k_rotate_sin_mul])

    k_embed = Tensor(dtype=out_dtype, name=out_name + "_k_embed")
    TpuLang.insert_op("top.Add", inputs=[k_cos_mul, k_rotate_sin_mul], outputs=[k_embed])

    # concate kv_cache
    kv_concat_attr = {
        "axis": Attr(1, "int32"),
    }
    k_cache_concat = Tensor(dtype=out_dtype, name=out_name + "_k_cache_concat")
    TpuLang.insert_op("top.Concat",
                      inputs=[k_cache, k_embed],
                      outputs=[k_cache_concat],
                      params=kv_concat_attr)

    v_cache_concat = Tensor(dtype=out_dtype, name=out_name + "_v_cache_concat")
    TpuLang.insert_op("top.Concat",
                      inputs=[v_cache, v_proj_reshape],
                      outputs=[v_cache_concat],
                      params=kv_concat_attr)

    # repeat_kv
    # k
    k_permute = permute(k_cache_concat, [0, 2, 1, 3], out_name=out_name + "_k_permute")
    k_unsqueeze = unsqueeze(k_permute, [2], out_name=out_name + "_k_unsqueeze")

    attr_bc = {
        "shape": ArrayAttr([1, num_key_value_heads, num_key_value_groups, concat_len, head_dim])
    }
    k_expand = Tensor(dtype=out_dtype, name=out_name + "_k_expand")
    TpuLang.insert_op("top.Expand", inputs=[k_unsqueeze], outputs=[k_expand], params=attr_bc)

    reshape_attr = {
        "shape": ArrayAttr([1, num_key_value_heads * num_key_value_groups, concat_len, head_dim]),
    }
    k_expand_reshape = Tensor(dtype=out_dtype, name=out_name + "_k_expand_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[k_expand],
                      outputs=[k_expand_reshape],
                      params=reshape_attr)

    # v
    v_permute = permute(v_cache_concat, [0, 2, 1, 3], out_name=out_name + "_v_permute")
    v_unsqueeze = unsqueeze(v_permute, [2], out_name=out_name + "_v_unsqueeze")

    v_expand = Tensor(dtype=out_dtype, name=out_name + "_v_expand")
    TpuLang.insert_op("top.Expand", inputs=[v_unsqueeze], outputs=[v_expand], params=attr_bc)

    reshape_attr = {
        "shape": ArrayAttr([1, num_key_value_heads * num_key_value_groups, concat_len, head_dim]),
    }
    v_expand_reshape = Tensor(dtype=out_dtype, name=out_name + "_v_expand_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[v_expand],
                      outputs=[v_expand_reshape],
                      params=reshape_attr)

    # q * k
    q_permute = permute(q_embed, [0, 2, 1, 3], out_name=out_name + "_q_permute")
    k_expand_permute = permute(k_expand_reshape, [0, 1, 3, 2],
                               out_name=out_name + "_k_expand_permute")
    attention_weight = matmul(q_permute,
                              k_expand_permute,
                              out_dtype=out_dtype,
                              out_name=out_name + "_attention_weight")

    mulconst_attr = {"const_val": Attr(1 / math.sqrt(head_dim), "float64")}
    attention_weight_mulconst = Tensor(dtype=out_dtype,
                                       name=out_name + "_attention_weight_mulconst")
    TpuLang.insert_op("top.MulConst",
                      inputs=[attention_weight],
                      outputs=[attention_weight_mulconst],
                      params=mulconst_attr)

    # add attn mask
    attention_weight_mask = Tensor(dtype=out_dtype, name=out_name + "_attention_weight_mask")
    TpuLang.insert_op("top.Add",
                      inputs=[attention_weight_mulconst, attention_mask],
                      outputs=[attention_weight_mask])

    # softmax
    softmax_out = softmax(attention_weight_mask, 3, out_name=out_name + "_softmax")

    # softmax * v
    softmax_v = matmul(softmax_out,
                       v_expand_reshape,
                       out_dtype=out_dtype,
                       out_name=out_name + "_softmax_v")

    # o_proj
    softmax_v_permute = permute(softmax_v, [0, 2, 1, 3], out_name=out_name + "_softmax_v_permute")

    softmax_reshape_attr = {
        "shape": ArrayAttr([bsz, q_len, -1]),
    }
    softmax_v_reshape = Tensor(dtype=out_dtype, name=out_name + "_softmax_v_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[softmax_v_permute],
                      outputs=[softmax_v_reshape],
                      params=softmax_reshape_attr)

    attention_out = a16matmul(softmax_v_reshape,
                              o_proj_weights,
                              o_proj_scales,
                              o_proj_zps,
                              o_proj_bias,
                              right_transpose=True,
                              out_dtype=out_dtype,
                              out_name=out_name + "_o_proj",
                              group_size=group_size,
                              bits=weight_bits)

    ############################## mlp block ##############################
    # add residual
    attention_out_add = Tensor(dtype=out_dtype, name=out_name + "_attention_out_add")
    TpuLang.insert_op("top.Add", inputs=[hidden_states, attention_out], outputs=[attention_out_add])

    # post layernorm
    post_layernorm_attr = {"eps": Attr(rms_norm_eps, "float64")}
    post_layernorm_output = Tensor(dtype=out_dtype, name=out_name + "_post_layernorm_output")
    TpuLang.insert_op("top.RMSNorm",
                      inputs=[attention_out_add, post_attention_layernorm_weight],
                      outputs=[post_layernorm_output],
                      params=post_layernorm_attr)

    # mlp
    gate_proj = a16matmul(post_layernorm_output,
                          gate_proj_weights,
                          gate_proj_scales,
                          gate_proj_zps,
                          right_transpose=True,
                          out_dtype=out_dtype,
                          out_name=out_name + "_gate_proj",
                          group_size=group_size,
                          bits=weight_bits)
    silu_out = silu(gate_proj, out_name="_silu_out")
    up_proj = a16matmul(post_layernorm_output,
                        up_proj_weights,
                        up_proj_scales,
                        up_proj_zps,
                        right_transpose=True,
                        out_dtype=out_dtype,
                        out_name=out_name + "_up_proj",
                        group_size=group_size,
                        bits=weight_bits)

    gate_up_mul = Tensor(dtype=out_dtype, name=out_name + "_gate_up_mul")
    TpuLang.insert_op("top.Mul", inputs=[silu_out, up_proj], outputs=[gate_up_mul])

    down_proj = a16matmul(gate_up_mul,
                          down_proj_weights,
                          down_proj_scales,
                          down_proj_zps,
                          right_transpose=True,
                          out_dtype=out_dtype,
                          out_name=out_name + "_down_proj",
                          group_size=group_size,
                          bits=weight_bits)

    # add residual
    mlp_out_add = Tensor(dtype=out_dtype, name=out_name + "_mlp_out_add")
    TpuLang.insert_op("top.Add", inputs=[attention_out_add, down_proj], outputs=[mlp_out_add])

    return mlp_out_add, k_embed, v_proj_reshape


######## Up / Down Scaling Operator #########
@auto_name()
@annotation_check
@assert_with_out_name
def maxpool2d(input: Tensor,
              kernel: Union[List[int], Tuple[int], None] = None,
              stride: Union[List[int], Tuple[int], None] = None,
              pad: Union[List[int], Tuple[int], None] = None,
              ceil_mode: bool = False,
              scale: List[float] = None,
              zero_point: List[int] = None,
              out_name: str = None,
              round_mode: str = "half_away_from_zero"):
    assert (not kernel or (len(kernel) == 2))
    assert (not stride or len(stride) == 2)
    assert (not pad or len(pad) == 4)
    kernel = [] if kernel is None else kernel
    stride = [1, 1] if stride is None else stride
    pad = [0, 0, 0, 0] if pad is None else pad
    if isinstance(kernel, Tuple): kernel = list(kernel)
    if isinstance(stride, Tuple): stride = list(stride)
    if isinstance(pad, Tuple): pad = list(pad)
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    o_dtype = input.dtype

    attr = {
        "kernel_shape": ArrayAttr(kernel),
        "strides": ArrayAttr(stride),
        "pads": ArrayAttr(pad),
        "ceil_mode": Attr(ceil_mode, "bool"),
        "keepdims": Attr(True, "bool"),
        "pad_value": Attr(0),
        "count_include_pad": Attr(False, "bool"),
        "do_relu": Attr(False, "bool"),
        "relu_limit": Attr(-1.0, "float64"),
        "round_mode": Attr(round_mode_convert(round_mode), data_type="string"),
    }

    output = Tensor(dtype=o_dtype, name=out_name)
    if scale is not None:
        zero_point = zero_point if zero_point is not None else [0, 0]
        input.quantization(scale=scale[0], zero_point=zero_point[0])
        output.quantization(scale=scale[1], zero_point=zero_point[1])
    TpuLang.insert_op("top.MaxPool", inputs=[input], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def maxpool3d(input: Tensor,
              kernel: Union[List[int], int, Tuple[int, ...]] = None,
              stride: Union[List[int], int, Tuple[int, ...]] = None,
              pad: Union[List[int], int, Tuple[int, ...]] = None,
              ceil_mode: bool = False,
              scale: List[float] = None,
              zero_point: List[int] = None,
              out_name: str = None,
              round_mode: str = "half_away_from_zero"):
    kernel = [] if kernel is None else kernel
    if isinstance(kernel, int):
        kernel = [kernel] * 3
    if stride is None:
        stride = [1, 1, 1]
    if isinstance(stride, int):
        stride = [stride] * 3
    if isinstance(pad, int):
        pad = [pad] * 6
    if isinstance(kernel, Tuple): kernel = list(kernel)
    if isinstance(stride, Tuple): stride = list(stride)
    if isinstance(pad, Tuple): pad = list(pad)
    pad = [0, 0, 0, 0, 0, 0] if pad is None else pad
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    o_dtype = input.dtype
    attr = {
        "kernel_shape": ArrayAttr(kernel),
        "strides": ArrayAttr(stride),
        "pads": ArrayAttr(pad),
        "ceil_mode": Attr(ceil_mode, "bool"),
        "keepdims": Attr(True, "bool"),
        "pad_value": Attr(0),
        "count_include_pad": Attr(False, "bool"),
        "do_relu": Attr(False, "bool"),
        "relu_limit": Attr(-1.0, "float64"),
        "round_mode": Attr(round_mode_convert(round_mode), data_type="string"),
    }
    output = Tensor(dtype=o_dtype, name=out_name)
    if scale is not None:
        zero_point = zero_point if zero_point is not None else [0, 0]
        input.quantization(scale=scale[0], zero_point=zero_point[0])
        output.quantization(scale=scale[1], zero_point=zero_point[1])
    TpuLang.insert_op("top.MaxPool", inputs=[input], outputs=[output], params=attr)
    return output


@auto_name('out_name')
@auto_name('mask_name')
@annotation_check
@assert_with_out_name
def maxpool2d_with_mask(input: Tensor,
                        kernel: Union[List[int], Tuple[int], None] = None,
                        stride: Union[List[int], Tuple[int], None] = None,
                        pad: Union[List[int], Tuple[int], None] = None,
                        ceil_mode: bool = False,
                        out_name: str = None,
                        mask_name: str = None):
    assert (not kernel or (len(kernel) == 2))
    assert (not stride or len(stride) == 2)
    assert (not pad or len(pad) == 4)
    kernel = [] if kernel is None else kernel
    stride = [1, 1] if stride is None else stride
    pad = [0, 0, 0, 0] if pad is None else pad
    if isinstance(kernel, Tuple): kernel = list(kernel)
    if isinstance(stride, Tuple): stride = list(stride)
    if isinstance(pad, Tuple): pad = list(pad)
    assert input.dtype in ["float32"]
    o_dtype = input.dtype

    attr = {
        "kernel_shape": ArrayAttr(kernel),
        "strides": ArrayAttr(stride),
        "pads": ArrayAttr(pad),
        "ceil_mode": Attr(ceil_mode, "bool"),
        "keepdims": Attr(True, "bool"),
        "pad_value": Attr(0),
        "count_include_pad": Attr(False, "bool"),
        "do_relu": Attr(False, "bool"),
        "relu_limit": Attr(-1.0, "float64")
    }

    output = Tensor(dtype=o_dtype, name=out_name)
    mask = Tensor(dtype="int32", name=mask_name)
    TpuLang.insert_op("top.MaxPoolWithMask", inputs=[input], outputs=[output, mask], params=attr)
    return output, mask


@auto_name()
@annotation_check
@assert_with_out_name
def avgpool2d(input: Tensor,
              kernel: Union[List[int], Tuple[int], None] = None,
              stride: Union[List[int], Tuple[int], None] = None,
              pad: Union[List[int], Tuple[int], None] = None,
              ceil_mode: bool = False,
              scale: List[float] = None,
              zero_point: List[int] = None,
              out_name: str = None,
              count_include_pad: bool = False,
              round_mode: str = "half_away_from_zero",
              first_round_mode: str = "half_away_from_zero"):
    kernel = [] if kernel is None else kernel
    stride = [1, 1] if stride is None else stride
    pad = [0, 0, 0, 0] if pad is None else pad
    if isinstance(kernel, Tuple): kernel = list(kernel)
    if isinstance(stride, Tuple): stride = list(stride)
    if isinstance(pad, Tuple): pad = list(pad)
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    o_dtype = input.dtype

    attr = {
        "kernel_shape": ArrayAttr(kernel),
        "strides": ArrayAttr(stride),
        "pads": ArrayAttr(pad),
        "ceil_mode": Attr(ceil_mode, "bool"),
        "keepdims": Attr(True, "bool"),
        "pad_value": Attr(0),
        "count_include_pad": Attr(count_include_pad, "bool"),
        "do_relu": Attr(False, "bool"),
        "relu_limit": Attr(-1.0, "float64"),
        "round_mode": Attr(round_mode_convert(round_mode), data_type="string"),
        "first_round_mode": Attr(round_mode_convert(first_round_mode), data_type="string"),
    }

    output = Tensor(dtype=o_dtype, name=out_name)
    if scale is not None:
        zero_point = zero_point if zero_point is not None else [0, 0]
        input.quantization(scale=scale[0], zero_point=zero_point[0])
        output.quantization(scale=scale[1], zero_point=zero_point[1])
    TpuLang.insert_op("top.AvgPool", inputs=[input], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def avgpool3d(input: Tensor,
              kernel: Union[List[int], int, Tuple[int, ...]] = None,
              stride: Union[List[int], int, Tuple[int, ...]] = None,
              pad: Union[List[int], int, Tuple[int, ...]] = None,
              ceil_mode: bool = False,
              scale: List[float] = None,
              zero_point: List[int] = None,
              out_name: str = None,
              count_include_pad: bool = False,
              round_mode: str = "half_away_from_zero",
              first_round_mode: str = "half_away_from_zero"):
    kernel = [] if kernel is None else kernel
    if isinstance(kernel, int):
        kernel = [kernel] * 3
    if stride is None:
        stride = [1, 1, 1]
    if isinstance(stride, int):
        stride = [stride] * 3
    if isinstance(pad, int):
        pad = [pad] * 6
    pad = [0, 0, 0, 0, 0, 0] if pad is None else pad
    if isinstance(kernel, Tuple): kernel = list(kernel)
    if isinstance(stride, Tuple): stride = list(stride)
    if isinstance(pad, Tuple): pad = list(pad)
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    o_dtype = input.dtype
    attr = {
        "kernel_shape": ArrayAttr(kernel),
        "strides": ArrayAttr(stride),
        "pads": ArrayAttr(pad),
        "ceil_mode": Attr(ceil_mode, "bool"),
        "keepdims": Attr(True, "bool"),
        "pad_value": Attr(0),
        "count_include_pad": Attr(count_include_pad, "bool"),
        "do_relu": Attr(False, "bool"),
        "relu_limit": Attr(-1.0, "float64"),
        "round_mode": Attr(round_mode_convert(round_mode), data_type="string"),
        "first_round_mode": Attr(round_mode_convert(first_round_mode), data_type="string"),
    }
    output = Tensor(dtype=o_dtype, name=out_name)
    if scale is not None:
        zero_point = zero_point if zero_point is not None else [0, 0]
        input.quantization(scale=scale[0], zero_point=zero_point[0])
        output.quantization(scale=scale[1], zero_point=zero_point[1])
    TpuLang.insert_op("top.AvgPool", inputs=[input], outputs=[output], params=attr)
    return output


######### Activation Operator ###############
@auto_name()
@annotation_check
@assert_with_out_name
def relu(input: Tensor, out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Relu", inputs=[input], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def prelu(input: Tensor, slope: Tensor, out_name: str = None):
    assert input.dtype in ["float32", "float16"]
    assert slope.ttype == "coeff"
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.PRelu", inputs=[input, slope], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def leaky_relu(
    input: Tensor,
    negative_slope: float = 0.01,
    out_name: str = None,
    round_mode: str = "half_away_from_zero",
):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    attr = {
        "alpha": Attr(negative_slope, data_type="float64"),
        "round_mode": Attr(round_mode_convert(round_mode), data_type="string"),
    }
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.LeakyRelu", inputs=[input], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def abs(input: Tensor, out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Abs", inputs=[input], outputs=[output])
    return output


def _active_scale(input: Tensor,
                  output: Tensor,
                  scale: List[float] = None,
                  zero_point: List[int] = None):
    if scale != None:
        zero_point = zero_point if zero_point is not None else [0, 0]
        assert len(scale) == 2 and len(zero_point) == 2
        output.quantization(scale=scale[1], zero_point=zero_point[1])
        input.quantization(scale=scale[0], zero_point=zero_point[0])


@auto_name()
@annotation_check
@assert_with_out_name
def ceil(input: Tensor,
         scale: List[float] = None,
         zero_point: List[int] = None,
         out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Ceil", inputs=[input], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def floor(input: Tensor,
          scale: List[float] = None,
          zero_point: List[int] = None,
          out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Floor", inputs=[input], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def round(input: Tensor, out_name: str = None):
    assert input.dtype in ["float32", "float16"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Round", inputs=[input], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def sin(input: Tensor,
        scale: List[float] = None,
        zero_point: List[int] = None,
        out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Sin", inputs=[input], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def cos(input: Tensor,
        scale: List[float] = None,
        zero_point: List[int] = None,
        out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Cos", inputs=[input], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def exp(input: Tensor,
        scale: List[float] = None,
        zero_point: List[int] = None,
        out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Exp", inputs=[input], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def ln(input: Tensor,
       scale: List[float] = None,
       zero_point: List[int] = None,
       out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Log", inputs=[input], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def tanh(input: Tensor,
         scale: List[float] = None,
         zero_point: List[int] = None,
         out_name: str = None,
         round_mode: str = "half_away_from_zero"):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    attr = {
        "round_mode": Attr(round_mode_convert(round_mode), data_type="string"),
    }
    TpuLang.insert_op("top.Tanh", inputs=[input], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def sigmoid(input: Tensor,
            scale: List[float] = None,
            zero_point: List[int] = None,
            out_name: str = None,
            round_mode: str = "half_away_from_zero"):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    attr = {
        "round_mode": Attr(round_mode_convert(round_mode), "string"),
    }
    TpuLang.insert_op("top.Sigmoid", inputs=[input], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def elu(input: Tensor,
        scale: List[float] = None,
        zero_point: List[int] = None,
        out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    attr = {
        "alpha": Attr(1.0, "float64"),
    }
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Elu", inputs=[input], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def square(input: Tensor,
           scale: List[float] = None,
           zero_point: List[int] = None,
           out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    assert not scale or len(scale) == 2
    scale = [scale[0], scale[0], scale[1]] if scale != None else scale
    assert not zero_point or len(zero_point) == 2
    zero_point = [zero_point[0], zero_point[0], zero_point[1]] if zero_point != None else zero_point
    return mul(input,
               input,
               scale=scale,
               zero_point=zero_point,
               out_dtype=input.dtype,
               out_name=out_name)


@auto_name()
@annotation_check
@assert_with_out_name
def sqrt(input: Tensor,
         scale: List[float] = None,
         zero_point: List[int] = None,
         out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Sqrt", inputs=[input], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def rsqrt(input: Tensor,
          scale: List[float] = None,
          zero_point: List[int] = None,
          out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Rsqrt", inputs=[input], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def silu(input: Tensor,
         scale: List[float] = None,
         zero_point: List[int] = None,
         out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.SiLU", inputs=[input], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def swish(input: Tensor,
          beta: float,
          scale: List[float] = None,
          zero_point: List[int] = None,
          round_mode: str = "half_away_from_zero",
          out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    attrs = {
        "beta": Attr(beta, "float64"),
        "round_mode": Attr(round_mode_convert(round_mode), "string"),
    }
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Swish", inputs=[input], outputs=[output], params=attrs)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def erf(input: Tensor,
        scale: List[float] = None,
        zero_point: List[int] = None,
        out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Erf", inputs=[input], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def log_sigmoid(input: Tensor,
                scale: List[float] = None,
                zero_point: List[int] = None,
                out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Sigmoid",
                      inputs=[input],
                      outputs=[output],
                      params={"log", Attr(True, bool)})
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def tan(input: Tensor, out_name: str = None):
    assert input.dtype in ["float32", "float16"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Tan", inputs=[input], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def softmax(input: Tensor, axis: int, out_name: str = None):
    assert input.dtype in ["float32", "float16"]
    attr = {
        "axis": Attr(axis, data_type="int32"),
    }
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Softmax", inputs=[input], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def softmax_int(input: Tensor,
                axis: int,
                scale: List[float],
                zero_point: List[int] = None,
                out_name: str = None,
                round_mode: str = "half_away_from_zero"):
    assert input.dtype in ["int8", "uint8"]
    attr = {
        "axis": Attr(axis, data_type="int32"),
        "round_mode": Attr(round_mode_convert(round_mode), "string"),
    }
    zero_point = zero_point if zero_point is not None else [0, 0]
    assert len(scale) == 2 and len(zero_point) == 2
    output = Tensor(input.shape,
                    dtype=input.dtype,
                    name=out_name,
                    scale=scale[1],
                    zero_point=zero_point[1])
    input.quantization(scale=scale[0], zero_point=zero_point[0])
    TpuLang.insert_op("top.Softmax", inputs=[input], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def mish(input: Tensor,
         scale: List[float] = None,
         zero_point: List[int] = None,
         out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Mish", inputs=[input], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def hswish(input: Tensor,
           scale: List[float] = None,
           zero_point: List[int] = None,
           out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.HardSwish", inputs=[input], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def arccos(input: Tensor, out_name: str = None):
    assert input.dtype in ["float32", "float16"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Arccos", inputs=[input], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def arctanh(input: Tensor, out_name: str = None):
    input.dtype in ["float32", "float16"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Arctanh", inputs=[input], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def sinh(input: Tensor,
         scale: List[float] = None,
         zero_point: List[int] = None,
         out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Sinh", inputs=[input], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def cosh(input: Tensor,
         scale: List[float] = None,
         zero_point: List[int] = None,
         out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Cosh", inputs=[input], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def sign(input: Tensor,
         scale: List[float] = None,
         zero_point: List[int] = None,
         out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Sign", inputs=[input], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def gelu(input: Tensor,
         scale: List[float] = None,
         zero_point: List[int] = None,
         out_name: str = None,
         round_mode: str = "half_away_from_zero"):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    output = Tensor(dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    attr = {
        "round_mode": Attr(round_mode_convert(round_mode), data_type="string"),
    }
    TpuLang.insert_op("top.GELU", inputs=[input], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def hsigmoid(input: Tensor,
             scale: List[float] = None,
             zero_point: List[int] = None,
             out_name: str = None):
    assert input.dtype in ["float32", "float16", "int8", "uint8"]
    attr = {
        "alpha": Attr(1 / 6, data_type="float64"),
        "beta": Attr(0.5, data_type="float64"),
    }
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.HardSigmoid", inputs=[input], outputs=[output], params=attr)
    return output


######### Sort Operator ############
@auto_name()
@annotation_check
@assert_with_out_name
def arg(input: Tensor,
        method: str = "max",
        axis: int = 0,
        keep_dims: bool = True,
        out_name: str = None):
    dims = len(input.shape)
    if input.dtype == "float32":
        o_dtype = input.dtype
    else:
        o_dtype = "int32"
    if method == 'max':
        method = 'ArgMax'
    elif method == 'min':
        method = 'ArgMin'
    attr = {
        "axis": Attr(axis),
        "keepdims": Attr(keep_dims, "bool"),
        "mode": Attr(method, "string"),
    }
    output1 = Tensor(dtype=o_dtype, name=f"{out_name}_ind")
    output2 = Tensor(dtype=input.dtype, name=f"{out_name}_val")
    TpuLang.insert_op("top.Arg", inputs=[input], outputs=[output1, output2], params=attr)
    return output1, output2


@auto_name()
@annotation_check
@assert_with_out_name
def sort(input: Tensor, axis: int = -1, descending: bool = True, out_name: str = None):
    dims = len(input.shape)
    attr = {
        "axis": Attr(axis),
        "descending": Attr(descending, "bool"),
    }
    output1 = Tensor(dtype=input.dtype, name=f"{out_name}_val")
    output2 = Tensor(dtype='int32', name=f"{out_name}_ind")
    TpuLang.insert_op("top.Sort", inputs=[input], outputs=[output1, output2], params=attr)
    return output1, output2


@auto_name()
@annotation_check
@assert_with_out_name
def argsort(input: Tensor, axis: int = -1, descending: bool = True, out_name: str = None):
    dims = len(input.shape)
    attr = {
        "axis": Attr(axis),
        "descending": Attr(descending, "bool"),
    }
    output = Tensor(dtype='int32', name=out_name)
    TpuLang.insert_op("top.Sort", inputs=[input], outputs=[None, output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def sort_by_key(input: Tensor,
                key: Tensor,
                axis: int = -1,
                descending: bool = True,
                out_name: str = None):
    dims = len(input.shape)
    assert len(key.shape) == 1, "dims of key should be 1"
    assert key.shape[0] == input.shape[axis], "number of keys should be same as input.shape[axis]"
    attr = {
        "axis": Attr(0),
        "descending": Attr(descending, "bool"),
    }
    sorted_key = Tensor(dtype=key.dtype, name=f"{out_name}_sorted_key")
    ind = Tensor(dtype='int32', name=f"{out_name}_interm")
    TpuLang.insert_op("top.Sort", inputs=[key], outputs=[sorted_key, ind], params=attr)
    attr = {
        "axis": Attr(axis, "int32"),
    }
    sorted_out = Tensor(dtype=input.dtype, name=f"{out_name}_sorted_out")
    TpuLang.insert_op("top.Gather", inputs=[input, ind], outputs=[sorted_out], params=attr)
    return sorted_out, sorted_key


######### Data Arrange Operator ############
@auto_name()
@annotation_check
@assert_with_out_name
def permute(input: Tensor, order: Union[Tuple[int], List[int]], out_name: str = None):
    if isinstance(order, Tuple): order = list(order)
    attr = {
        "order": ArrayAttr(order),
    }
    output = Tensor(dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Permute", inputs=[input], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def tile(input: Tensor, reps: Union[Tuple[int], List[int]], out_name: str = None):
    if isinstance(reps, Tuple): reps = list(reps)
    attr = {
        "tile": ArrayAttr(reps),
    }
    assert input.dtype in ["float32", "float16", "int8", "uint8", "int16", "uint16"]
    output = Tensor(dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Tile", inputs=[input], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def concat(inputs: List[Tensor],
           scales: Optional[Union[List[float], List[int]]] = None,
           zero_points: Optional[List[int]] = None,
           axis: int = 0,
           out_name: str = None,
           dtype: str = "float32",
           round_mode: str = "half_away_from_zero"):
    if scales is None:
        scales = [None] * (len(inputs) + 1)
    if isinstance(scales, Tuple): scales = list(scales)
    if zero_points is None:
        zero_points = [None] * (len(inputs) + 1)
    assert len(inputs) > 1, "concat should have more than one input"
    attr = {
        "axis": Attr(axis, "int32"),
        "round_mode": Attr(round_mode_convert(round_mode), data_type="string"),
    }
    input_list_ = []
    for index, i_tensor in enumerate(inputs):
        i_tensor.quantization(scale=scales[index], zero_point=zero_points[index])
        input_list_.append(i_tensor)
    output = Tensor(dtype=inputs[0].dtype, name=out_name)
    if dtype not in ["float32", "float16"]:
        output.quantization(scale=scales[len(scales) - 1],
                            zero_point=zero_points[len(zero_points) - 1])
    TpuLang.insert_op("top.Concat", inputs=input_list_, outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def broadcast(input: Tensor, reps: Union[Tuple[int], List[int]], out_name: str = None):
    output = Tensor(dtype=input.dtype, name=out_name)
    if isinstance(reps, Tuple): reps = list(reps)
    assert input.dtype in ["float32", "float16", "int8", "uint8", "int16", "uint16"]
    TpuLang.insert_op("top.Expand",
                      inputs=[input],
                      outputs=[output],
                      params={"shape": ArrayAttr(reps)})
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def nonzero(input: Tensor, dtype="int32", out_name: str = None):
    attr = {
        "order": Attr("ColMajor", "string"),
    }
    assert input.dtype in ["float32", "float16"]
    output = Tensor(dtype=dtype, name=out_name)
    TpuLang.insert_op("top.NonZero", inputs=[input], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def upsample(input: Tensor, scale: int = 2, out_name: str = None):
    attr = {
        "scale_h": Attr(scale, data_type="int64"),
        "scale_w": Attr(scale, data_type="int64"),
    }
    assert input.dtype in ["float32", "float16", "int8"]
    output = Tensor(dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Upsample", inputs=[input], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def reduce(input: Tensor,
           method: str = "ReduceSum",
           axes: Union[List[int], int] = [1, 2],
           keep_dims: bool = True,
           out_name: str = None):
    assert (method in [
        "ReduceMin", "ReduceMax", "ReduceMean", "ReduceProd", "ReduceL2", "ReduceL1", "ReduceSum"
    ])
    if isinstance(axes, int):
        axes = [axes]
    assert input.dtype in ["float32", "float16"]
    attr = {
        "axes": ArrayAttr(axes, "int64"),
        "keepdims": Attr(keep_dims, "bool"),
        "mode": Attr(method, "string"),
    }
    output = Tensor(dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Reduce", inputs=[input], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def unsqueeze(input: Tensor, axes: List[int] = [1, 2], out_name: str = None):
    attr = {
        "axes": ArrayAttr(axes, "int64"),
    }
    output = Tensor(dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Unsqueeze", inputs=[input], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def yuv2rgb(
    inputs: Tensor,
    src_format: int,
    dst_format: int,
    ImageOutFormatAttr: str,
    formula_mode: str,
    round_mode: str,
    out_name: str = None,
):
    # src_format/dst_format:
    # {
    #     FORMAT_MAPPING_YUV420P_YU12, 0
    #     FORMAT_MAPPING_YUV420P_YV12, 1
    #     FORMAT_MAPPING_NV12,         2
    #     FORMAT_MAPPING_NV21,         3
    #     FORMAT_MAPPING_RGB,          4
    #     FORMAT_MAPPING_BGR,          5
    # }
    assert ImageOutFormatAttr in [
        # "FLOAT32", bug remains in 'bmodel_inference' stage
        "UINT8",
    ]
    assert formula_mode in ["_601_limited", "_601_full"]
    assert round_mode in ["HalfAwayFromZero", "HalfToEven"]
    attr = {
        "src_format": Attr(src_format, "uint32"),
        "dst_format": Attr(dst_format, "uint32"),
        "image_format": Attr(ImageOutFormatAttr, "string"),
        "formula_mode": Attr(formula_mode, "string"),
        "round_mode": Attr(round_mode, "string"),
    }

    output = Tensor(dtype=ImageOutFormatAttr, name=out_name)
    TpuLang.insert_op("top.Yuv2rgbFormula", inputs=[inputs], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def split(input: Tensor,
          axis: int = 0,
          num: int = 1,
          size: Union[Tuple[int], List[int]] = (),
          out_name: str = None) -> List[Tensor]:
    assert (num > 1 and "number of split output should be more than 1")
    # if not size:
    #     assert(input.shape[axis] % num == 0 and "invalid split size")
    #     size = [int(input.shape[axis] / num)] * num
    # else:
    #     assert(num == len(size) and "size should be the same as num")
    #     assert(sum(size) == input.shape[axis] and "invalid size")
    if isinstance(size, Tuple): size = list(size)
    attr = {
        "axis": Attr(axis, "int32"),
        "num": Attr(num),
    }
    assert input.dtype in ["float32", "float16", "int8", "uint8", "int16", "uint16"]
    if len(size) != 0:
        attr["split_size"] = ArrayAttr(size)

    outputs = []
    for i in range(num):
        outputs.append(Tensor(dtype=input.dtype, name=f"{out_name}_{i}"))

    TpuLang.insert_op("top.Split", inputs=[input], outputs=outputs, params=attr)
    return outputs


@auto_name()
@annotation_check
@assert_with_out_name
def slice(input: Tensor,
          starts: Union[int, List[int]],
          ends: Union[int, List[int]],
          steps: Union[int, List[int]] = None,
          axes: Union[int, List[int]] = None,
          out_name: str = None) -> Tensor:
    starts = [starts] if isinstance(starts, int) else starts
    ends = [ends] if isinstance(ends, int) else ends
    length = len(starts)
    if steps is None:
        steps = [1] * length
    steps = [steps] if isinstance(steps, int) else steps
    if axes is not None:
        axes = [axes] if isinstance(axes, int) else axes
        assert length == len(axes)
    assert length == len(ends) and length == len(steps)
    attr = {
        "offset": ArrayAttr(starts, "int64"),
        "steps": ArrayAttr(steps, "int64"),
        "ends": ArrayAttr(ends, "int64"),
    }
    if axes is not None:
        attr["axes"] = ArrayAttr(axes, "int64")
    output = Tensor(dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Slice", inputs=[input, None, None, None], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def pad(input: Tensor,
        method: str = "constant",
        value: Scalar = None,
        padding: Union[Tuple[int], List[int]] = None,
        out_name: str = None):
    assert (method in ["constant", "reflect", "symmetric", "edge"] and "Not supported pad type")
    if padding is None:
        padding = [0] * (len(input.shape) * 2)
    if isinstance(padding, Tuple): padding = list(padding)
    assert (not padding or len(padding) == 2 * len(input.shape) and "Invalid padding length")
    assert input.dtype in ["float32", "float16", "int8", "uint8", "int16", "uint16"]
    attr = {
        "paddings": ArrayAttr(padding, "int64"),
        "val": Attr(value.value if value is not None else 0.0, "float64"),
        "mode": Attr(method, "string"),
    }
    output = Tensor(dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Pad", inputs=[input], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def repeat(input: Tensor, reps: Tensor, out_name: str = None):
    # reps = Tensor(data = reps, shape = input.shape)
    output = Tensor(dtype=input.dtype, name=out_name)
    assert input.dtype in ["float32", "float16", "int8", "uint8", "int16", "uint16"]
    TpuLang.insert_op("top.Repeat", inputs=[input, reps], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def extract(input: Tensor,
            start: Union[List[int], Tuple[int]] = None,
            end: Union[List[int], Tuple[int]] = None,
            stride: Union[List[int], Tuple[int]] = None,
            out_name: str = None):
    dims = len(input.shape)
    if isinstance(start, Tuple): start = list(start)
    if isinstance(end, Tuple): end = list(end)
    if isinstance(stride, Tuple): stride = list(stride)
    if start:
        assert (dims == len(start)), f"length of `start` should be {dims}"
    else:
        start = [0] * dims
    if end:
        assert (end is None or dims == len(end)), f"length of `end` should be {dims}"
    else:
        end = input.shape
    if stride:
        assert (stride is None or dims == len(stride)), f"length of `stride` should be {dims}"
    else:
        stride = [1] * dims
    if out_name is None:
        out_name = generate_name("extract")
    assert input.dtype in ["float32", "float16", "int8", "uint8", "int16", "uint16"]
    output = Tensor(dtype=input.dtype, name=out_name)
    attr = {
        "offset": ArrayAttr(start),
        "ends": ArrayAttr(end),
        "steps": ArrayAttr(stride),
    }
    TpuLang.insert_op("top.Slice", inputs=[input, None, None, None], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def multi_scale_deformable_attention(
    query: Tensor,
    value: Tensor,
    key_padding_mask: Tensor,
    reference_points: Tensor,
    sampling_offsets_weight: Tensor,
    sampling_offsets_bias_ori: Tensor,
    attention_weights_weight: Tensor,
    attention_weights_bias_ori: Tensor,
    value_proj_weight: Tensor,
    value_proj_bias_ori: Tensor,
    output_proj_weight: Tensor,
    output_proj_bias_ori: Tensor,
    spatial_shapes: List[List[int]],
    embed_dims: int,
    num_heads: int = 8,
    num_levels: int = 4,
    num_points: int = 4,
    out_name: str = None,
):
    assert query.shape[0] == 1
    assert value.shape[0] == 1
    bs = 1
    assert embed_dims % (num_heads * num_levels) == 0
    assert TpuLang.chip in ["bm1684x", "bm1688"]
    npu_num = 64 if TpuLang.chip == "bm1684x" else 32
    _, num_query, _ = query.shape
    _, num_value, _ = value.shape
    _, _, num_levels, _ = reference_points.shape
    spatial_shapes_np = np.array(spatial_shapes)
    assert (spatial_shapes_np[:, 0] * spatial_shapes_np[:, 1]).sum() == num_value
    import math
    num_padding = math.ceil(num_query / npu_num) * npu_num - num_query
    num_query_padding = num_padding + num_query

    if out_name is None:
        out_name = generate_name("multi_scale_deformable_attention")
    o_dtype = query.dtype
    int64_max = np.iinfo(np.int64).max

    # reshape: reshape all bias to [1, 1, bias_len]
    bias_reshape_attr = {
        "shape": ArrayAttr([1, 1, -1]),
    }
    sampling_offsets_bias = Tensor(dtype=o_dtype, name=out_name + "_sampling_offsets_bias_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[sampling_offsets_bias_ori],
                      outputs=[sampling_offsets_bias],
                      params=bias_reshape_attr)
    attention_weights_bias = Tensor(dtype=o_dtype,
                                    name=out_name + "_attention_weights_bias_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[attention_weights_bias_ori],
                      outputs=[attention_weights_bias],
                      params=bias_reshape_attr)
    value_proj_bias = Tensor(dtype=o_dtype, name=out_name + "_value_proj_bias_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[value_proj_bias_ori],
                      outputs=[value_proj_bias],
                      params=bias_reshape_attr)
    output_proj_bias = Tensor(dtype=o_dtype, name=out_name + "_output_proj_bias_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[output_proj_bias_ori],
                      outputs=[output_proj_bias],
                      params=bias_reshape_attr)

    #  concat: padding query to align with npu_num
    query_padding_weight_np = np.zeros((bs, num_padding, embed_dims), dtype=o_dtype)
    query_padding_weight = Tensor(dtype=o_dtype,
                                  shape=[bs, num_padding, embed_dims],
                                  data=query_padding_weight_np,
                                  ttype="coeff")
    query_concat_input_list = [query, query_padding_weight]
    query_padding_out = Tensor(dtype=o_dtype, name=out_name + "_padding_query_with_concat")
    concat_attr = {
        "axis": Attr(1, "int32"),
    }
    TpuLang.insert_op("top.Concat",
                      inputs=query_concat_input_list,
                      outputs=[query_padding_out],
                      params=concat_attr)

    # concat: padding reference_points to align with npu_num
    reference_points_padding_weight_np = np.zeros((bs, num_padding, num_levels, 2), dtype=o_dtype)
    reference_points_padding_weight = Tensor(dtype=o_dtype,
                                             shape=[bs, num_padding, num_levels, 2],
                                             data=reference_points_padding_weight_np,
                                             ttype="coeff")
    reference_points_concat_input_list = [reference_points, reference_points_padding_weight]
    reference_points_padding_out = Tensor(dtype=o_dtype,
                                          name=out_name + "_padding_reference_points_with_concat")
    concat_attr = {
        "axis": Attr(1, "int32"),
    }
    TpuLang.insert_op("top.Concat",
                      inputs=reference_points_concat_input_list,
                      outputs=[reference_points_padding_out],
                      params=concat_attr)

    # reshape: convert reference_points from [1, align_up(num_query, npu_num), num_levels, 2] to [align_up(num_query, npu_num), num_levels*2]
    reference_points_reshape_attr = {
        "shape": ArrayAttr([num_query_padding, num_levels * 2]),
    }
    reference_points_reshape_out = Tensor(dtype=o_dtype,
                                          name=out_name + "_reference_points_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[reference_points_padding_out],
                      outputs=[reference_points_reshape_out],
                      params=reference_points_reshape_attr)

    # permute: permute reference_points to [num_levels*2, align_up(num_query, npu_num)], order=[1, 0]
    reference_points_permute_attr = {
        "order": ArrayAttr([1, 0]),
    }
    reference_points_permute_out = Tensor(dtype=o_dtype,
                                          name=out_name + "_reference_points_permute")
    TpuLang.insert_op("top.Permute",
                      inputs=[reference_points_reshape_out],
                      outputs=[reference_points_permute_out],
                      params=reference_points_permute_attr)

    # loop for num_levels*2 times
    reference_points_list = []
    reference_points_shape_len = 2
    reference_points_slice_offset = [0] * reference_points_shape_len
    reference_points_slice_ends = [int64_max] * reference_points_shape_len
    reference_points_slice_steps = [1] * reference_points_shape_len
    for i in range(num_levels):
        for k in range(2):
            idx = i * 2 + k
            # slice: split reference_points
            reference_points_slice_offset[0] = idx
            reference_points_slice_ends[0] = idx + 1
            reference_points_slice_attr = {
                "offset": ArrayAttr(reference_points_slice_offset),
                "ends": ArrayAttr(reference_points_slice_ends),
                "steps": ArrayAttr(reference_points_slice_steps),
                "axes": ArrayAttr([]),
            }
            reference_points_slice_out = Tensor(dtype=o_dtype,
                                                name=out_name +
                                                "_reference_points_slice_{}".format(idx))
            TpuLang.insert_op("top.Slice",
                              inputs=[reference_points_permute_out, None, None, None],
                              outputs=[reference_points_slice_out],
                              params=reference_points_slice_attr)
            # reshape: reshape reference_points to convert dimension c to npu_num
            reference_points_reshape_attr = {
                "shape": ArrayAttr([1, npu_num, -1]),
            }
            reference_points_reshape_out = Tensor(dtype=o_dtype,
                                                  name=out_name +
                                                  "_reference_points_reshape_{}".format(idx))
            TpuLang.insert_op("top.Reshape",
                              inputs=[reference_points_slice_out],
                              outputs=[reference_points_reshape_out],
                              params=reference_points_reshape_attr)
            reference_points_list.append(reference_points_reshape_out)

    # reorder weight & bias of linear layer which is used to compute sampling_offsets
    # reorder weight
    # reshape
    sampling_offsets_weight_reshape_0_attr = {
        "shape": ArrayAttr([embed_dims, num_heads, num_levels, num_points, 2]),
    }
    sampling_offsets_weight_reshape_0_out = Tensor(dtype=o_dtype,
                                                   name=out_name +
                                                   "_sampling_offsets_weight_reshape_0")
    TpuLang.insert_op("top.Reshape",
                      inputs=[sampling_offsets_weight],
                      outputs=[sampling_offsets_weight_reshape_0_out],
                      params=sampling_offsets_weight_reshape_0_attr)
    # permute
    sampling_offsets_weight_permute_attr = {
        "order": ArrayAttr([0, 2, 1, 4, 3]),
    }
    sampling_offsets_weight_permute_out = Tensor(dtype=o_dtype,
                                                 name=out_name + "_sampling_offsets_weight_permute")
    TpuLang.insert_op("top.Permute",
                      inputs=[sampling_offsets_weight_reshape_0_out],
                      outputs=[sampling_offsets_weight_permute_out],
                      params=sampling_offsets_weight_permute_attr)
    # reshape
    sampling_offsets_weight_reshape_1_attr = {
        "shape": ArrayAttr([embed_dims, -1]),
    }
    sampling_offsets_weight_reshape_1_out = Tensor(dtype=o_dtype,
                                                   name=out_name +
                                                   "_sampling_offsets_weight_reshape_1")
    TpuLang.insert_op("top.Reshape",
                      inputs=[sampling_offsets_weight_permute_out],
                      outputs=[sampling_offsets_weight_reshape_1_out],
                      params=sampling_offsets_weight_reshape_1_attr)
    # reorder bias
    # reshape
    sampling_offsets_bias_reshape_attr = {
        "shape": ArrayAttr([num_heads, num_levels, num_points, 2]),
    }
    sampling_offsets_bias_reshape_0_out = Tensor(dtype=o_dtype,
                                                 name=out_name + "_sampling_offsets_bias_reshape_0")
    TpuLang.insert_op("top.Reshape",
                      inputs=[sampling_offsets_bias],
                      outputs=[sampling_offsets_bias_reshape_0_out],
                      params=sampling_offsets_bias_reshape_attr)
    # permute
    sampling_offsets_bias_permute_attr = {
        "order": ArrayAttr([1, 0, 3, 2]),
    }
    sampling_offsets_bias_permute_out = Tensor(dtype=o_dtype,
                                               name=out_name + "_sampling_offsets_bias_permute")
    TpuLang.insert_op("top.Permute",
                      inputs=[sampling_offsets_bias_reshape_0_out],
                      outputs=[sampling_offsets_bias_permute_out],
                      params=sampling_offsets_bias_permute_attr)
    # reshape
    sampling_offsets_bias_reshape_attr = {
        "shape": ArrayAttr([1, 1, -1]),
    }
    sampling_offsets_bias_reshape_1_out = Tensor(dtype=o_dtype,
                                                 name=out_name + "_sampling_offsets_bias_reshape_1")
    TpuLang.insert_op("top.Reshape",
                      inputs=[sampling_offsets_bias_permute_out],
                      outputs=[sampling_offsets_bias_reshape_1_out],
                      params=sampling_offsets_bias_reshape_attr)

    # matmul with weight & bias: compute sampling_offsets
    sampling_offsets_matmul_attr = {
        "right_transpose": Attr(False, "bool"),
        "left_transpose": Attr(False, "bool"),
        "output_transpose": Attr(False, "bool"),
        "hdim_is_batch": Attr(False, "bool"),
        "keep_dims": Attr(True, "bool"),
        "do_relu": Attr(False, "bool"),
        "relu_limit": Attr(-1.0, "float64")
    }
    sampling_offsets_matmul_out = Tensor(dtype=o_dtype, name=out_name + "_sampling_offsets_matmul")
    TpuLang.insert_op("top.MatMul",
                      inputs=[
                          query_padding_out, sampling_offsets_weight_reshape_1_out,
                          sampling_offsets_bias_reshape_1_out
                      ],
                      outputs=[sampling_offsets_matmul_out],
                      params=sampling_offsets_matmul_attr)

    # permute: permute sampling_offsets to [1, num_levels*num_heads*2*num_points, align_up(num_query, npu_num)], order=[0, 2, 1]
    sampling_offsets_permute_attr = {
        "order": ArrayAttr([0, 2, 1]),
    }
    sampling_offsets_permute_out = Tensor(dtype=o_dtype,
                                          name=out_name + "_sampling_offsets_permute")
    TpuLang.insert_op("top.Permute",
                      inputs=[sampling_offsets_matmul_out],
                      outputs=[sampling_offsets_permute_out],
                      params=sampling_offsets_permute_attr)

    # reshape: reshape sampling_offsets to [num_heads, num_levels, num_points, 2, align_up(num_query, npu_num)]
    sampling_offsets_reshape_0_attr = {
        "shape": ArrayAttr([num_levels * num_heads * 2, num_points, -1]),
    }
    sampling_offsets_reshape_0_out = Tensor(dtype=o_dtype,
                                            name=out_name + "_sampling_offsets_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[sampling_offsets_permute_out],
                      outputs=[sampling_offsets_reshape_0_out],
                      params=sampling_offsets_reshape_0_attr)

    # loop for num_levels*num_heads*2 times
    sampling_offsets_list = []
    sampling_offsets_shape_len = 3
    sampling_offsets_slice_offset = [0] * sampling_offsets_shape_len
    sampling_offsets_slice_ends = [int64_max] * sampling_offsets_shape_len
    sampling_offsets_slice_steps = [1] * sampling_offsets_shape_len
    for i in range(num_levels * num_heads * 2):
        # slice: split sampling_offsets
        sampling_offsets_slice_offset[0] = i
        sampling_offsets_slice_ends[0] = i + 1
        sampling_offsets_slice_attr = {
            "offset": ArrayAttr(sampling_offsets_slice_offset),
            "ends": ArrayAttr(sampling_offsets_slice_ends),
            "steps": ArrayAttr(sampling_offsets_slice_steps),
            "axes": ArrayAttr([]),
        }
        sampling_offsets_slice_out = Tensor(dtype=o_dtype,
                                            name=out_name + "_sampling_offsets_slice_{}".format(i))
        TpuLang.insert_op("top.Slice",
                          inputs=[sampling_offsets_reshape_0_out, None, None, None],
                          outputs=[sampling_offsets_slice_out],
                          params=sampling_offsets_slice_attr)
        # reshape: reshape sampling_offsets to convert dimension c to npu_num
        sampling_offsets_reshape_1_attr = {
            "shape": ArrayAttr([num_points, npu_num, -1]),
        }
        sampling_offsets_reshape_1_out = Tensor(dtype=o_dtype,
                                                name=out_name +
                                                "_sampling_offsets_reshape_{}".format(i))
        TpuLang.insert_op("top.Reshape",
                          inputs=[sampling_offsets_slice_out],
                          outputs=[sampling_offsets_reshape_1_out],
                          params=sampling_offsets_reshape_1_attr)
        sampling_offsets_list.append(sampling_offsets_reshape_1_out)

    # loop for num_levels*num_heads*2 times
    offset_normalizer = np.stack([spatial_shapes_np[..., 1], spatial_shapes_np[..., 0]],
                                 -1).flatten()
    sampling_locations_list = []
    for i in range(num_levels):
        for j in range(num_heads):
            for k in range(2):
                ik = i * 2 + k
                ijk = i * num_heads * 2 + j * 2 + k
                # calculate 2 *(reference_points_list[ik] + sampling_offsets_list[ijk] / offset_normalizer_list[ik]) - 1
                # mulconst: sampling_offsets_list[ijk] * (1/offset_normalizer_list[ik])
                sampling_locations_mulconst_0_attr = {
                    "const_val": Attr(1.0 / offset_normalizer[ik], "float64"),
                }
                sampling_locations_mulconst_0_out = Tensor(
                    dtype=o_dtype, name=out_name + "_sampling_locations_mulconst_0_{}".format(ijk))
                TpuLang.insert_op("top.MulConst",
                                  inputs=[sampling_offsets_list[ijk]],
                                  outputs=[sampling_locations_mulconst_0_out],
                                  params=sampling_locations_mulconst_0_attr)
                # add: reference_points_list[ik] + sampling_locations_mulconst_0_out[ijk]
                sampling_locations_add_out = Tensor(dtype=o_dtype,
                                                    name=out_name +
                                                    "_sampling_locations_add_{}".format(ijk))
                TpuLang.insert_op(
                    "top.Add",
                    inputs=[reference_points_list[ik], sampling_locations_mulconst_0_out],
                    outputs=[sampling_locations_add_out])
                # mulconst: 2 * sampling_locations_add_out[ijk]
                sampling_locations_mulconst_1_attr = {
                    "const_val": Attr(2.0, "float64"),
                }
                sampling_locations_mulconst_1_out = Tensor(
                    dtype=o_dtype, name=out_name + "_sampling_locations_mulconst_1_{}".format(ijk))
                TpuLang.insert_op("top.MulConst",
                                  inputs=[sampling_locations_add_out],
                                  outputs=[sampling_locations_mulconst_1_out],
                                  params=sampling_locations_mulconst_1_attr)
                # addconst: 2 * sampling_locations_add_out[ijk] - 1
                sampling_locations_addconst_attr = {
                    "const_val": Attr(-1.0, "float64"),
                }
                sampling_locations_addconst_out = Tensor(
                    dtype=o_dtype, name=out_name + "_sampling_locations_addconst_{}".format(ijk))
                TpuLang.insert_op("top.AddConst",
                                  inputs=[sampling_locations_mulconst_1_out],
                                  outputs=[sampling_locations_addconst_out],
                                  params=sampling_locations_addconst_attr)
                sampling_locations_list.append(sampling_locations_addconst_out)
    # concat: concat sampling_locations
    sampling_locations_concat_attr = {
        "axis": Attr(0, "int32"),
    }
    sampling_locations_concat_out = Tensor(dtype=o_dtype,
                                           name=out_name + "_sampling_locations_concat")
    TpuLang.insert_op("top.Concat",
                      inputs=sampling_locations_list,
                      outputs=[sampling_locations_concat_out],
                      params=sampling_locations_concat_attr)
    # reshape: reshape sampling_locations to [num_levels, num_heads, 2, num_points, align_up(num_query, npu_num)]
    sampling_locations_reshape_attr = {
        "shape": ArrayAttr([num_levels, num_heads, 2, num_points, -1]),
    }
    sampling_grids_reshape_out = Tensor(dtype=o_dtype,
                                        name=out_name + "_sampling_locations_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[sampling_locations_concat_out],
                      outputs=[sampling_grids_reshape_out],
                      params=sampling_locations_reshape_attr)

    # matmul with weight & bias: compute attention_weights
    attention_weights_matmul_attr = {
        "right_transpose": Attr(False, "bool"),
        "left_transpose": Attr(False, "bool"),
        "output_transpose": Attr(False, "bool"),
        "hdim_is_batch": Attr(False, "bool"),
        "keep_dims": Attr(True, "bool"),
        "do_relu": Attr(False, "bool"),
        "relu_limit": Attr(-1.0, "float64")
    }
    attention_weights_matmul_out = Tensor(dtype=o_dtype,
                                          name=out_name + "_attention_weights_matmul")
    if len(attention_weights_bias.shape) == 1:
        attention_weights_bias.shape = [1, 1, attention_weights_bias.shape[0]]
        attention_weights_bias.buffer = attention_weights_bias.buffer.reshape(
            attention_weights_bias.shape)
    TpuLang.insert_op("top.MatMul",
                      inputs=[query_padding_out, attention_weights_weight, attention_weights_bias],
                      outputs=[attention_weights_matmul_out],
                      params=attention_weights_matmul_attr)
    # permute: permute attention_weights to [1, num_heads*num_levels*num_points, align_up(num_query, npu_num)], order=[0, 2, 1]
    attention_weights_permute_attr = {
        "order": ArrayAttr([0, 2, 1]),
    }
    attention_weights_permute_out = Tensor(dtype=o_dtype,
                                           name=out_name + "_attention_weights_permute")
    TpuLang.insert_op("top.Permute",
                      inputs=[attention_weights_matmul_out],
                      outputs=[attention_weights_permute_out],
                      params=attention_weights_permute_attr)
    # reshape: reshape attention_weights to [num_heads, num_levels*num_points, align_up(num_query, npu_num)]
    attention_weights_reshape_attr = {
        "shape": ArrayAttr([num_heads, num_levels * num_points, -1]),
    }
    attention_weights_reshape_out = Tensor(dtype=o_dtype,
                                           name=out_name + "_attention_weights_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[attention_weights_permute_out],
                      outputs=[attention_weights_reshape_out],
                      params=attention_weights_reshape_attr)
    # softmax: compute softmax of attention_weights at axis=-2
    attention_weights_softmax_attr = {
        "axis": Attr(-2, "int32"),
    }
    attention_weights_softmax_out = Tensor(dtype=o_dtype,
                                           name=out_name + "_attention_weights_softmax")
    TpuLang.insert_op("top.Softmax",
                      inputs=[attention_weights_reshape_out],
                      outputs=[attention_weights_softmax_out],
                      params=attention_weights_softmax_attr)
    # loop for num_levels times
    attention_weights_list = []
    attention_weights_shape_len = 3
    attention_weights_slice_offset = [0] * attention_weights_shape_len
    attention_weights_slice_ends = [int64_max] * attention_weights_shape_len
    attention_weights_slice_steps = [1] * attention_weights_shape_len
    for i in range(num_levels):
        # slice: split attention_weights
        attention_weights_slice_offset[1] = i * num_points
        attention_weights_slice_ends[1] = (i + 1) * num_points
        attention_weights_slice_attr = {
            "offset": ArrayAttr(attention_weights_slice_offset),
            "ends": ArrayAttr(attention_weights_slice_ends),
            "steps": ArrayAttr(attention_weights_slice_steps),
            "axes": ArrayAttr([]),
        }
        attention_weights_slice_out = Tensor(dtype=o_dtype,
                                             name=out_name +
                                             "_attention_weights_slice_{}".format(i))
        TpuLang.insert_op("top.Slice",
                          inputs=[attention_weights_softmax_out, None, None, None],
                          outputs=[attention_weights_slice_out],
                          params=attention_weights_slice_attr)
        # reshape: reshape to [num_heads, 1, num_points, align_up(num_query, npu_num)]
        attention_weights_reshape_attr = {
            "shape": ArrayAttr([num_heads, 1, num_points, -1]),
        }
        attention_weights_reshape_out = Tensor(dtype=o_dtype,
                                               name=out_name +
                                               "_attention_weights_reshape_{}".format(i))
        TpuLang.insert_op("top.Reshape",
                          inputs=[attention_weights_slice_out],
                          outputs=[attention_weights_reshape_out],
                          params=attention_weights_reshape_attr)
        attention_weights_list.append(attention_weights_reshape_out)

    # subconst: 1 - key_padding_mask, instead of where key_padding_mask == 0, set it to 1, otherwise set it to 0
    key_padding_mask_subconst_attr = {
        "const_val": Attr(1.0, "float64"),
        "is_reverse": Attr(True, "bool"),
    }
    key_padding_mask_subconst_out = Tensor(dtype=o_dtype,
                                           name=out_name + "_key_padding_mask_subconst")
    TpuLang.insert_op("top.SubConst",
                      inputs=[key_padding_mask],
                      outputs=[key_padding_mask_subconst_out],
                      params=key_padding_mask_subconst_attr)

    # reshape: key_padding_mask_subconst_out to [1, 1, num_value]
    key_padding_mask_reshape_attr = {
        "shape": ArrayAttr([1, 1, num_value]),
    }
    key_padding_mask_reshape_out = Tensor(dtype=o_dtype,
                                          name=out_name + "_key_padding_mask_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[key_padding_mask_subconst_out],
                      outputs=[key_padding_mask_reshape_out],
                      params=key_padding_mask_reshape_attr)
    # loop for num_levels times
    value_shape_len = 3
    value_slice_offset = [0] * value_shape_len
    value_slice_ends = [int64_max] * value_shape_len
    value_slice_ends[1] = 0
    value_slice_steps = [1] * value_shape_len
    key_padding_mask_shape_len = 3
    key_padding_mask_slice_offset = [0] * key_padding_mask_shape_len
    key_padding_mask_slice_ends = [int64_max] * key_padding_mask_shape_len
    key_padding_mask_slice_ends[2] = 0
    key_padding_mask_slice_steps = [1] * key_padding_mask_shape_len
    sampling_grid_shape_len = 5
    sampling_grid_slice_offset = [0] * sampling_grid_shape_len
    sampling_grid_slice_ends = [int64_max] * sampling_grid_shape_len
    sampling_grid_slice_steps = [1] * sampling_grid_shape_len
    dim_per_head = embed_dims // num_heads
    sampling_value_list = []
    if len(value_proj_bias.shape) == 1:
        value_proj_bias.shape = [1, 1, value_proj_bias.shape[0]]
        value_proj_bias.buffer = value_proj_bias.buffer.reshape(value_proj_bias.shape)
    for i, (H_, W_) in enumerate(spatial_shapes):
        HW_ = H_ * W_
        # slice: split value to merge matmul + permute
        value_slice_ends[1] += HW_
        value_slice_attr = {
            "offset": ArrayAttr(value_slice_offset),
            "ends": ArrayAttr(value_slice_ends),
            "steps": ArrayAttr(value_slice_steps),
            "axes": ArrayAttr([]),
        }
        value_slice_offset[1] += HW_
        value_slice_out = Tensor(dtype=o_dtype, name=out_name + "_value_slice_{}".format(i))
        TpuLang.insert_op("top.Slice",
                          inputs=[value, None, None, None],
                          outputs=[value_slice_out],
                          params=value_slice_attr)
        # matmul with weight & bias: compute value_proj
        value_proj_matmul_attr = {
            "right_transpose": Attr(False, "bool"),
            "left_transpose": Attr(False, "bool"),
            "output_transpose": Attr(False, "bool"),
            "hdim_is_batch": Attr(False, "bool"),
            "keep_dims": Attr(True, "bool"),
            "do_relu": Attr(False, "bool"),
            "relu_limit": Attr(-1.0, "float64")
        }
        value_proj_matmul_out = Tensor(dtype=o_dtype,
                                       name=out_name + "_value_proj_matmul_{}".format(i))
        TpuLang.insert_op("top.MatMul",
                          inputs=[value_slice_out, value_proj_weight, value_proj_bias],
                          outputs=[value_proj_matmul_out],
                          params=value_proj_matmul_attr)
        # permute: permute value_proj to [1, embed_dims, align_up(num_query, npu_num)], order=[0, 2, 1]
        value_proj_permute_attr = {
            "order": ArrayAttr([0, 2, 1]),
        }
        value_proj_permute_out = Tensor(dtype=o_dtype,
                                        name=out_name + "_value_proj_permute_{}".format(i))
        TpuLang.insert_op("top.Permute",
                          inputs=[value_proj_matmul_out],
                          outputs=[value_proj_permute_out],
                          params=value_proj_permute_attr)
        # slice: split key_padding_mask
        key_padding_mask_slice_ends[2] += HW_
        key_padding_mask_slice_attr = {
            "offset": ArrayAttr(key_padding_mask_slice_offset),
            "ends": ArrayAttr(key_padding_mask_slice_ends),
            "steps": ArrayAttr(key_padding_mask_slice_steps),
            "axes": ArrayAttr([]),
        }
        key_padding_mask_slice_offset[2] += HW_
        key_padding_mask_slice_out = Tensor(dtype=o_dtype,
                                            name=out_name + "_key_padding_mask_slice_{}".format(i))
        TpuLang.insert_op("top.Slice",
                          inputs=[key_padding_mask_reshape_out, None, None, None],
                          outputs=[key_padding_mask_slice_out],
                          params=key_padding_mask_slice_attr)
        # # tile: tile key_padding_mask to [1, embed_dims, num_value]
        # key_padding_mask_tile_attr = {
        #     "tile": ArrayAttr([1, embed_dims, 1]),
        # }
        # key_padding_mask_tile_out = Tensor(dtype=o_dtype, name=out_name + "_key_padding_mask_tile_{}".format(i))
        # TpuLang.insert_op("top.Tile", inputs=[key_padding_mask_slice_out], outputs=[key_padding_mask_tile_out], params=key_padding_mask_tile_attr)
        # # where
        # value_l_where_attr = {
        #     "x_const_val": Attr(0.0, "float64"),
        #     "x_is_const": Attr(True, "bool"),
        #     "y_const_val": Attr(0.0, "float64"),
        #     "y_is_const": Attr(False, "bool"),
        # }
        # value_l_out = Tensor(dtype=o_dtype, name=out_name + "_value_l_where_{}".format(i))
        # TpuLang.insert_op("top.Where", inputs=[key_padding_mask_tile_out, None, value_proj_permute_out], outputs=[value_l_out], params=value_l_where_attr)
        # reshape: reshape value_l_where to [num_heads, dim_per_head, H_, W_]

        # mul: key_padding_mask_slice_out * value_proj_permute_out instead of where
        value_l_out = Tensor(dtype=o_dtype, name=out_name + "_key_padding_mask_mul_{}".format(i))
        TpuLang.insert_op("top.Mul",
                          inputs=[key_padding_mask_slice_out, value_proj_permute_out],
                          outputs=[value_l_out])
        value_l_reshape_attr = {
            "shape": ArrayAttr([num_heads, dim_per_head, H_, W_]),
        }
        value_l_reshape_out = Tensor(dtype=o_dtype, name=out_name + "_value_l_reshape_{}".format(i))
        TpuLang.insert_op("top.Reshape",
                          inputs=[value_l_out],
                          outputs=[value_l_reshape_out],
                          params=value_l_reshape_attr)
        # slice: sampling_grids_reshape_out
        sampling_grid_slice_offset[0] = i
        sampling_grid_slice_ends[0] = i + 1
        sampling_grid_slice_attr = {
            "offset": ArrayAttr(sampling_grid_slice_offset),
            "ends": ArrayAttr(sampling_grid_slice_ends),
            "steps": ArrayAttr(sampling_grid_slice_steps),
            "axes": ArrayAttr([]),
        }
        sampling_grids_slice_out = Tensor(dtype=o_dtype,
                                          name=out_name + "_sampling_grids_slice_{}".format(i))
        TpuLang.insert_op("top.Slice",
                          inputs=[sampling_grids_reshape_out, None, None, None],
                          outputs=[sampling_grids_slice_out],
                          params=sampling_grid_slice_attr)
        # squeeze
        sampling_grids_squeeze_attr = {
            "axes": ArrayAttr([0]),
        }
        sampling_grids_squeeze_out = Tensor(dtype=o_dtype,
                                            name=out_name + "_sampling_grids_squeeze_{}".format(i))
        TpuLang.insert_op("top.Squeeze",
                          inputs=[sampling_grids_slice_out],
                          outputs=[sampling_grids_squeeze_out],
                          params=sampling_grids_squeeze_attr)
        # permute
        sampling_grids_permute_attr = {
            "order": ArrayAttr([0, 2, 3, 1]),
        }
        sampling_grids_permute_out = Tensor(dtype=o_dtype,
                                            name=out_name + "_sampling_grids_permute_{}".format(i))
        TpuLang.insert_op("top.Permute",
                          inputs=[sampling_grids_squeeze_out],
                          outputs=[sampling_grids_permute_out],
                          params=sampling_grids_permute_attr)
        # grid_sampler
        sampling_grid_l_grid_sampler_attr = {
            "mode": Attr(0, "int64"),
            "padding_mode": Attr(0, "int64"),
            "align_corners": Attr(False, 'bool'),
        }
        grid_sampler_out = Tensor(dtype=o_dtype,
                                  name=out_name + "_sampling_grid_l_grid_sampler_{}".format(i))
        TpuLang.insert_op("top.GridSampler",
                          inputs=[value_l_reshape_out, sampling_grids_permute_out],
                          outputs=[grid_sampler_out],
                          params=sampling_grid_l_grid_sampler_attr)
        sampling_value_list.append(grid_sampler_out)
    sampling_value_l_reduce_out_buffer = None
    for i in range(num_levels):
        # mul
        sampling_value_l_mul_out = Tensor(dtype=o_dtype,
                                          name=out_name + "_sampling_value_l_mul_{}".format(i))
        TpuLang.insert_op("top.Mul",
                          inputs=[attention_weights_list[i], sampling_value_list[i]],
                          outputs=[sampling_value_l_mul_out])
        if sampling_value_l_reduce_out_buffer is None:
            sampling_value_l_reduce_out_buffer = sampling_value_l_mul_out
        else:
            # add
            sampling_value_l_reduce_out = Tensor(dtype=o_dtype,
                                                 name=out_name +
                                                 "_sampling_value_l_reduce_{}".format(i))
            TpuLang.insert_op("top.Add",
                              inputs=[sampling_value_l_reduce_out_buffer, sampling_value_l_mul_out],
                              outputs=[sampling_value_l_reduce_out])
            sampling_value_l_reduce_out_buffer = sampling_value_l_reduce_out
    # reduce_sum
    sampling_value_l_reduce_all_attr = {
        "axes": ArrayAttr([-2], "int64"),
        "keepdims": Attr(False, "bool"),
        "mode": Attr("ReduceSum", "string"),
    }
    sampling_value_l_reduce_all_out = Tensor(dtype=o_dtype,
                                             name=out_name + "_sampling_value_l_reduce_all")
    TpuLang.insert_op("top.Reduce",
                      inputs=[sampling_value_l_reduce_out],
                      outputs=[sampling_value_l_reduce_all_out],
                      params=sampling_value_l_reduce_all_attr)
    # reshape
    sampling_value_l_reshape_attr = {
        "shape": ArrayAttr([1, embed_dims, -1]),
    }
    sampling_value_l_reshape_out = Tensor(dtype=o_dtype,
                                          name=out_name + "_sampling_value_l_reshape")
    TpuLang.insert_op("top.Reshape",
                      inputs=[sampling_value_l_reduce_all_out],
                      outputs=[sampling_value_l_reshape_out],
                      params=sampling_value_l_reshape_attr)
    # permute
    sampling_value_l_permute_attr = {
        "order": ArrayAttr([0, 2, 1]),
    }
    output_padding = Tensor(dtype=o_dtype, name=out_name + "_sampling_value_l_permute")
    TpuLang.insert_op("top.Permute",
                      inputs=[sampling_value_l_reshape_out],
                      outputs=[output_padding],
                      params=sampling_value_l_permute_attr)
    # slice: form align_up(num_query, npu_num) back to num_query
    output_padding_shape_len = 3
    output_padding_slice_offset = [0] * output_padding_shape_len
    output_padding_slice_ends = [int64_max] * output_padding_shape_len
    output_padding_slice_ends[1] = num_query
    output_padding_slice_steps = [1] * output_padding_shape_len
    output_padding_slice_attr = {
        "offset": ArrayAttr(output_padding_slice_offset),
        "ends": ArrayAttr(output_padding_slice_ends),
        "steps": ArrayAttr(output_padding_slice_steps),
        "axes": ArrayAttr([]),
    }
    output_padding_slice_out = Tensor(dtype=o_dtype, name=out_name + "_output_padding_slice")
    TpuLang.insert_op("top.Slice",
                      inputs=[output_padding, None, None, None],
                      outputs=[output_padding_slice_out],
                      params=output_padding_slice_attr)
    # matmul with weight & bias: compute output_proj
    output_proj_matmul_attr = {
        "right_transpose": Attr(False, "bool"),
        "left_transpose": Attr(False, "bool"),
        "output_transpose": Attr(False, "bool"),
        "hdim_is_batch": Attr(False, "bool"),
        "keep_dims": Attr(True, "bool"),
        "do_relu": Attr(False, "bool"),
        "relu_limit": Attr(-1.0, "float64")
    }
    output_proj_matmul_out = Tensor(dtype=o_dtype, name=out_name)
    TpuLang.insert_op("top.MatMul",
                      inputs=[output_padding_slice_out, output_proj_weight, output_proj_bias],
                      outputs=[output_proj_matmul_out],
                      params=output_proj_matmul_attr)
    return output_proj_matmul_out


@auto_name()
@annotation_check
@assert_with_out_name
def roll(input: Tensor,
         shifts: Union[int, List[int], Tuple[int]],
         dims: Union[int, List[int], Tuple[int]] = None,
         out_name: str = None):
    #
    # ====== case1 (dims is None): =========
    #
    #       roll => flatten -> slice0 -> concat -> reshape
    #                        \        /
    #                          slice1
    #
    #
    #
    # ====== case2 (dims is not None): ========
    #
    #    for i in dims:
    #       roll => slice1_i -> concat_i
    #             \          /
    #               slice2_i
    #
    #    concat(concat_0, ···， concat_dims)
    #
    if isinstance(shifts, Tuple): shifts = list(shifts)
    if isinstance(dims, Tuple): dims = list(dims)
    assert input.dtype in ["float32", "float16", "int8", "uint8", "int16", "uint16"]
    o_dtype = input.dtype
    in_shape = input.shape
    if dims is None:
        assert isinstance(shifts, int), "invalid dims/shifts"
        start_dim = 0
        end_dim = -1
        length = 1
        for i in in_shape:
            length *= i
        flatten_attr = {"start_dim": Attr(start_dim), "end_dim": Attr(end_dim)}
        flatten_out = Tensor(dtype=o_dtype, name=out_name + "_flatten")
        TpuLang.insert_op("top.Flatten", inputs=[input], outputs=[flatten_out], params=flatten_attr)

        slice0_attr = {
            "offset": ArrayAttr([length - (shifts % length)]),
            "steps": ArrayAttr([1]),
            "ends": ArrayAttr([length]),
            "axes": ArrayAttr([0]),
        }
        slice0_out = Tensor(dtype=o_dtype, name=out_name + "_slice0")
        TpuLang.insert_op("top.Slice",
                          inputs=[flatten_out, None, None, None],
                          outputs=[slice0_out],
                          params=slice0_attr)

        slice1_attr = {
            "offset": ArrayAttr([0]),
            "steps": ArrayAttr([1]),
            "ends": ArrayAttr([length - (shifts % length)]),
            "axes": ArrayAttr([0]),
        }
        slice1_out = Tensor(dtype=o_dtype, name=out_name + "_slice1")
        TpuLang.insert_op("top.Slice",
                          inputs=[flatten_out, None, None, None],
                          outputs=[slice1_out],
                          params=slice1_attr)

        concat_attr = {"axis": Attr(0, "int32")}
        concat_out = Tensor(dtype=o_dtype, name=out_name + "_concat")
        TpuLang.insert_op("top.Concat",
                          inputs=[slice0_out, slice1_out],
                          outputs=[concat_out],
                          params=concat_attr)

        reshape_attr = {"shape": ArrayAttr(in_shape)}
        final_out = Tensor(dtype=o_dtype, name=out_name)  # out_name should keep the same
        TpuLang.insert_op("top.Reshape",
                          inputs=[concat_out],
                          outputs=[final_out],
                          params=reshape_attr)
        return final_out
    else:
        len_shape = len(in_shape)
        if isinstance(shifts, int):
            assert isinstance(dims, int) and 0 <= dims < len_shape, "invalid dims/shifts"
            dims = [dims]
            shifts = [shifts]
        elif isinstance(shifts, Tuple) or isinstance(shifts, List):
            assert (isinstance(dims, Tuple) or isinstance(dims, List)) and len(shifts) == len(dims) and len(shifts) <= len_shape, \
                        "invalid dims/shifts"
            for dim in dims:
                assert 0 <= dim < len_shape, "invalid dims"

        final_out = None
        cur_in = input
        for i, (dim, shift) in enumerate(zip(dims, shifts)):
            offset_0 = [0] * len_shape
            offset_0[dim] = in_shape[dim] - (shift % in_shape[dim])
            slice0_attr = {
                "offset": ArrayAttr(offset_0),
                "steps": ArrayAttr([1] * len_shape),
                "ends": ArrayAttr(in_shape),
                "axes": ArrayAttr(list(range(0, len_shape, 1)))
            }
            slice0_out = Tensor(dtype=o_dtype, name=out_name + "_{}_slice0".format(i))
            TpuLang.insert_op("top.Slice",
                              inputs=[cur_in, None, None, None],
                              outputs=[slice0_out],
                              params=slice0_attr)

            ends_1 = in_shape.copy()
            ends_1[dim] = ends_1[dim] - (shift % ends_1[dim])
            slice1_attr = {
                "offset": ArrayAttr([0] * len_shape),
                "steps": ArrayAttr([1] * len_shape),
                "ends": ArrayAttr(ends_1),
                "axes": ArrayAttr(list(range(0, len_shape, 1)))
            }
            slice1_out = Tensor(dtype=o_dtype, name=out_name + "_{}_slice1".format(i))
            TpuLang.insert_op("top.Slice",
                              inputs=[cur_in, None, None, None],
                              outputs=[slice1_out],
                              params=slice1_attr)

            concat_out_name = out_name
            concat_out_name += "" if i == (len(dims) - 1) else "_{}_concat".format(i)
            concat_out = Tensor(dtype=o_dtype, name=concat_out_name)
            TpuLang.insert_op("top.Concat",
                              inputs=[slice0_out, slice1_out],
                              outputs=[concat_out],
                              params={"axis": Attr(dim, "int32")})
            cur_in = concat_out
            final_out = concat_out
        return final_out


######### Vision Operator ############
'''
[User-Guide]
please find DynamicFuse Example at branch <elif mode == "DynFuse":>, several tips to notice:
for 7-len rois,
1) rois only sliced at [2:6] for [x0,y0,x1,y1], thus please ensure your model interface is also sth like [a,b,x0,y0,x1,y1], where a,b,c are not concerned here
2) to generate a [batch_id,x0,y0,x1,y1] for batch-1 case, a zero tensor concat with sliced rois[2:6]
'''


@auto_name()
@annotation_check
@assert_with_out_name
def roiExtractor(rois: Tensor,
                 target_lvls: Tensor,
                 feats: List[Tensor],
                 PH: int,
                 PW: int,
                 sampling_ratio: int,
                 list_spatial_scale: Union[int, List[int], Tuple[int]],
                 num_layer: int,
                 mode: str = None,
                 out_name: str = None):
    if isinstance(list_spatial_scale, Tuple): list_spatial_scale = list(list_spatial_scale)
    list_spatial_scale = [1.0 / x for x in list_spatial_scale]
    for i in range(len(feats) - 1):
        assert feats[i].shape[1] == feats[i + 1].shape[1]

    roi_num = rois.shape[0]
    o_dtype = rois.dtype
    assert mode in ["DynFuse", "DynNormal", "StaticFuse"]
    outFusion = None
    if mode == "DynNormal":

        pad = Tensor(dtype=o_dtype,
                     shape=[roi_num, 1],
                     data=np.zeros([roi_num, 1], dtype=np.float32))

        Slice_attr = {
            "offset": ArrayAttr([0, 2]),
            "steps": ArrayAttr([1, 1]),
            "ends": ArrayAttr([roi_num, 6]),
            "axes": ArrayAttr([]),
            "hasparamConvert_axes": ArrayAttr([1]),
        }
        Slice_out = Tensor(dtype=o_dtype, name=out_name + "_Slice")
        TpuLang.insert_op("top.Slice",
                          inputs=[rois, None, None, None],
                          outputs=[Slice_out],
                          params=Slice_attr)

        Concat_attr = {"axis": Attr(1, "int32")}
        Concat_out = Tensor(dtype=o_dtype, name=out_name + "_Concat")
        TpuLang.insert_op("top.Concat",
                          inputs=[pad, Slice_out],
                          outputs=[Concat_out],
                          params=Concat_attr)

        layers = num_layer
        out_shape = [roi_num, feats[0].shape[1], PH, PW]
        roi_feats = Tensor(dtype=o_dtype,
                           shape=list(out_shape),
                           data=np.zeros(out_shape, dtype=np.float32))

        ScatterND_outputs = [
            Tensor(dtype=o_dtype, name=out_name + "_ScatterND_{}".format(i)) for i in range(layers)
        ]

        for i in range(layers):
            CompareConst_attr = {
                "mode": Attr("Equal", "string"),
                "const_val": Attr(float(i), "float64"),
                "inversed": Attr(False, "bool")
            }
            CompareConst_out = Tensor(dtype=o_dtype, name=out_name + "_CompareConst_{}".format(i))
            TpuLang.insert_op("top.CompareConst",
                              inputs=[target_lvls],
                              outputs=[CompareConst_out],
                              params=CompareConst_attr)

            NonZero_attr = {
                "order": Attr("ColMajor", "string"),
            }
            NonZero_out = Tensor(dtype=o_dtype, name=out_name + "_NonZero_{}".format(i))
            TpuLang.insert_op("top.NonZero",
                              inputs=[CompareConst_out],
                              outputs=[NonZero_out],
                              params=NonZero_attr)

            Gather_attr = {
                "axis": Attr(0, "int32"),
            }
            Gather_out = Tensor(dtype=o_dtype, name=out_name + "_Gather_{}".format(i))
            TpuLang.insert_op("top.Gather",
                              inputs=[Concat_out, NonZero_out],
                              outputs=[Gather_out],
                              params=Gather_attr)

            Squeeze_attr = {
                "axes": ArrayAttr([1]),
            }
            Squeeze_out = Tensor(dtype=o_dtype, name=out_name + "_Squeeze_{}".format(i))
            TpuLang.insert_op("top.Squeeze",
                              inputs=[Gather_out],
                              outputs=[Squeeze_out],
                              params=Squeeze_attr)

            RoiAlign_attr = {
                "mode": Attr("Avg", "string"),
                "output_height": Attr(PH, "int64"),
                "output_width": Attr(PW, "int64"),
                "sampling_ratio": Attr(sampling_ratio, "int64"),
                "spatial_scale": Attr(list_spatial_scale[i], "float64"),
                "align_corners": Attr(False, "bool"),
            }
            RoiAlign_out = Tensor(dtype=o_dtype, name=out_name + "_RoiAlign_{}".format(i))
            TpuLang.insert_op("top.RoiAlign",
                              inputs=[feats[i], Squeeze_out],
                              outputs=[RoiAlign_out],
                              params=RoiAlign_attr)

            if i == 0:
                TpuLang.insert_op("top.ScatterND",
                                  inputs=[roi_feats, NonZero_out, RoiAlign_out],
                                  outputs=[ScatterND_outputs[i]])
            else:
                TpuLang.insert_op("top.ScatterND",
                                  inputs=[ScatterND_outputs[i - 1], NonZero_out, RoiAlign_out],
                                  outputs=[ScatterND_outputs[i]])

        return ScatterND_outputs[-1]
    elif mode == "DynFuse":
        RoiExtractor_out = Tensor(dtype=o_dtype, name=out_name + "_RoiExtractor_{}".format(0))
        assert (sampling_ratio > 0)
        assert (num_layer <= 5 and num_layer >= 1)
        assert (rois.shape[1] == 5 or rois.shape[1] == 7)
        Slice_out = None
        if (rois.shape[1] == 7):
            #check coordinates of rois satisfied with [a, batch_id, x0, y0, x1, y1, b]
            batch = feats[0].shape[0]
            for i in range(num_layer):
                assert batch == feats[i].shape[0]
            valid_rois_batch_id = [j for j in range(batch)]
            for i in rois.buffer[:, 1]:
                assert i in valid_rois_batch_id
            print("[DynFuse-RoiExtractor] valid_rois_batch_id checked")
            assert batch == 1, "[Error] RoiExtractor support batch = 1"
            pad = Tensor(dtype=o_dtype,
                         shape=[roi_num, 1],
                         data=np.zeros([roi_num, 1], dtype=np.float32))
            Slice_attr = {
                "offset": ArrayAttr([0, 2]),
                "steps": ArrayAttr([1, 1]),
                "ends": ArrayAttr([roi_num, 6]),
                "axes": ArrayAttr([]),
                "hasparamConvert_axes": ArrayAttr([1]),
            }
            Slice_out = Tensor(dtype=o_dtype, name=out_name + "_Slice")
            TpuLang.insert_op("top.Slice",
                              inputs=[rois, None, None, None],
                              outputs=[Slice_out],
                              params=Slice_attr)

            Concat_attr = {"axis": Attr(1, "int32")}
            Concat_out = Tensor(dtype=o_dtype, name=out_name + "_Concat")
            TpuLang.insert_op("top.Concat",
                              inputs=[pad, Slice_out],
                              outputs=[Concat_out],
                              params=Concat_attr)

        RoiExtractor_attr = {
            "num_levels": Attr(num_layer, "int64"),
            "mode": Attr("Avg", "string"),
            "output_height": Attr(PH, "int64"),
            "output_width": Attr(PW, "int64"),
            "sampling_ratio": Attr(sampling_ratio, "int64"),
            "spatial_scales": ArrayAttr(list_spatial_scale, "float64"),
            "align_corners": Attr(False, "bool"),
            "is_static": Attr(False, "bool"),
        }
        # inputs =  [feats[i] for i in range(num_layer)] + [rois, target_lvls]
        rois_revise = rois if rois.shape[1] == 5 else Concat_out
        assert rois_revise is not None
        inputs = [rois_revise, target_lvls] + [feats[i] for i in range(num_layer)]
        TpuLang.insert_op("top.RoiExtractor",
                          inputs=inputs,
                          outputs=[RoiExtractor_out],
                          params=RoiExtractor_attr)
        return RoiExtractor_out
    else:
        assert 0

    return outFusion


@auto_name()
@annotation_check
@assert_with_out_name
def topk(input: Tensor, axis: int, k: int, out_name: str = None):
    dims = len(input.shape)
    assert input.dtype in ["float32", "int32"]
    assert k > 0, f"k:{k} is not valid"
    attr = {
        "axis": Attr(axis),
        "K": Attr(k),
    }
    output1 = Tensor(dtype=input.dtype, name=f'{out_name}_val')
    output2 = Tensor(dtype='int32', name=f'{out_name}_ind')
    TpuLang.insert_op("top.TopK", inputs=[input], outputs=[output1, output2], params=attr)
    return output1, output2


@auto_name()
@annotation_check
@assert_with_out_name
def nms(boxes: Tensor,
        scores: Tensor,
        box_format: str = 'PYTORCH',
        max_box_num_per_class: int = 1,
        out_name: str = None):
    boxes_dims = len(boxes.shape)
    scores_dims = len(scores.shape)
    assert boxes_dims == 3, f"dims of boxes expect 3 but get {boxes_dims}"
    assert scores_dims == 3, f"dims of boxes expect 3 but get {scores_dims}"
    assert box_format in ['TENSORFLOW', 'PYTORCH'], f"box_format:{box_format} is not supported"
    assert max_box_num_per_class > 0, f"max_box_num_per_class:{max_box_num_per_class} is not valid"
    if box_format == 'PYTORCH':
        center_point_box = 1
    else:
        center_point_box = 0
    attr = {
        "center_point_box": Attr(center_point_box),
        "max_output_size": Attr(max_box_num_per_class),
    }
    output = Tensor(dtype=boxes.dtype, name=out_name)
    TpuLang.insert_op("top.Nms", inputs=[boxes, scores], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def rope(input: Tensor,
         weight0: Tensor,
         weight1: Tensor,
         is_permute_optimize: bool = False,
         mul1_round_mode: str = 'half_up',
         mul2_round_mode: str = 'half_up',
         add_round_mode: str = 'half_up',
         mul1_shift: int = None,
         mul2_shift: int = None,
         add_shift: int = None,
         mul1_saturation: bool = True,
         mul2_saturation: bool = True,
         add_saturation: bool = True,
         out_name: str = None):

    attr = {
        "is_permute_optimize": Attr(is_permute_optimize, "bool"),
        "mul1_round_mode": Attr(round_mode_convert(mul1_round_mode), data_type="string"),
        "mul2_round_mode": Attr(round_mode_convert(mul2_round_mode), data_type="string"),
        "add_round_mode": Attr(round_mode_convert(add_round_mode), data_type="string"),
        "mul1_saturation": Attr(mul1_saturation, "bool"),
        "mul2_saturation": Attr(mul2_saturation, "bool"),
        "add_saturation": Attr(add_saturation, "bool")
    }

    if input.dtype not in ["float32", "float16"]:
        attr.update({
            "mul1_shift": Attr(mul1_shift, data_type="int32"),
            "mul2_shift": Attr(mul2_shift, data_type="int32"),
            "add_shift": Attr(add_shift, data_type="int32")
        })

    output = Tensor(dtype=input.dtype, name=out_name)
    if len(input.shape) == 4:
        if not is_permute_optimize:
            assert len(input.shape) == 4 and len(weight0.shape) == 2 and len(weight1.shape) == 2
            assert input.shape[2] == weight0.shape[0] and input.shape[3] == weight0.shape[1]
            assert input.shape[2] == weight1.shape[0] and input.shape[3] == weight1.shape[1]
            weight0.shape = [1, 1, weight0.shape[0], weight0.shape[1]]
            weight0.buffer = weight0.buffer.reshape(weight0.shape)
            weight1.shape = [1, 1, weight1.shape[0], weight1.shape[1]]
            weight1.buffer = weight1.buffer.reshape(weight1.shape)
        else:
            assert len(input.shape) == 4 and len(weight0.shape) == 2 and len(weight1.shape) == 2
            assert input.shape[1] == weight0.shape[0] and input.shape[3] == weight0.shape[1]
            assert input.shape[1] == weight1.shape[0] and input.shape[3] == weight1.shape[1]
            weight0.shape = [1, weight0.shape[0], 1, weight0.shape[1]]
            weight0.buffer = weight0.buffer.reshape(weight0.shape)
            weight1.shape = [1, weight1.shape[0], 1, weight1.shape[1]]
            weight1.buffer = weight1.buffer.reshape(weight1.shape)
    if len(input.shape) == 3:
        assert len(input.shape) == 3 and len(weight0.shape) == 2 and len(weight1.shape) == 2
        assert input.shape[1] == weight0.shape[0] and input.shape[2] == weight0.shape[1]
        assert input.shape[1] == weight1.shape[0] and input.shape[2] == weight1.shape[1]
        weight0.shape = [1, weight0.shape[0], weight0.shape[1]]
        weight0.buffer = weight0.buffer.reshape(weight0.shape)
        weight1.shape = [1, weight1.shape[0], weight1.shape[1]]
        weight1.buffer = weight1.buffer.reshape(weight1.shape)
    if len(input.shape) == 5:
        assert len(input.shape) == 5 and len(weight0.shape) == 2 and len(weight1.shape) == 2
        assert input.shape[3] == weight0.shape[0] and input.shape[4] == weight0.shape[1]
        assert input.shape[3] == weight1.shape[0] and input.shape[4] == weight1.shape[1]
        weight0.shape = [1, 1, 1, weight0.shape[0], weight0.shape[1]]
        weight0.buffer = weight0.buffer.reshape(weight0.shape)
        weight1.shape = [1, 1, 1, weight1.shape[0], weight1.shape[1]]
        weight1.buffer = weight1.buffer.reshape(weight1.shape)

    TpuLang.insert_op("top.Rope", inputs=[input, weight0, weight1], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def interpolate(input: Tensor,
                scale_h: float,
                scale_w: float,
                method: str = 'nearest',
                coord_mode: str = "pytorch_half_pixel",
                out_name: str = None):
    input_dims = len(input.shape)
    assert input_dims >= 2, f"input dims expect >=2 but get {input_dims}"
    assert scale_h > 0, f"scale_h:{scale_h} is not valid"
    assert scale_w > 0, f"scale_w:{scale_w} is not valid"
    assert method in ['nearest', 'linear'], f"method:{method} is not supported"
    assert coord_mode in ["align_corners", "pytorch_half_pixel", "half_pixel",
                          "asymmetric"], f"coord_mode:{coord_mode} is not supported"
    return interpolate_v2(input,
                          None,
                          None, [scale_h, scale_w],
                          method=method,
                          coord_mode=coord_mode,
                          out_name=out_name)
    # attr = {
    #     "scale_h": Attr(scale_h, 'float64'),
    #     "scale_w": Attr(scale_w, 'float64'),
    #     "mode": Attr(method, 'string'),
    #     "coord_mode": Attr(coord_mode, 'string'),
    # }
    # output = Tensor(dtype=input.dtype, name=out_name)
    # TpuLang.insert_op("top.Interp", inputs=[input, None], outputs=[output], params=attr)
    # return output


@auto_name()
@annotation_check
@assert_with_out_name
def interpolate_v2(input: Tensor,
                   roi: Tensor = None,
                   sizes: Tensor = None,
                   scale: List[float] = None,
                   method: str = 'nearest',
                   coord_mode: str = "pytorch_half_pixel",
                   antialias: int = 0,
                   axes: List[int] = None,
                   cubic_coeff_a: float = -0.75,
                   exclude_outside: int = 0,
                   extrapolation_value: float = 0.0,
                   keep_aspect_ratio_policy: str = "stretch",
                   nearest_mode: str = "round_prefer_floor",
                   out_name: str = None):

    scale_h = scale[0] if scale != None else 0.0
    scale_w = scale[1] if scale != None else 0.0
    if sizes == None:
        assert scale is not None, f"sizes and scale is not supported"
    assert method in ['nearest', 'linear'], f"method:{method} is not supported"
    assert coord_mode in ["align_corners", "pytorch_half_pixel", "half_pixel",
                          "asymmetric"], f"coord_mode:{coord_mode} is not supported"
    assert roi == None
    assert antialias == 0
    assert axes == None
    assert exclude_outside == 0
    if method == "cubic":
        assert cubic_coeff_a == -0.75
    assert extrapolation_value == 0.0
    assert exclude_outside == 0
    assert keep_aspect_ratio_policy == "stretch"
    if method == 'nearest':
        assert nearest_mode == "round_prefer_floor"
    attr = {
        "scale_h": Attr(scale_h, 'float64'),
        "scale_w": Attr(scale_w, 'float64'),
        "mode": Attr(method, 'string'),
        "coord_mode": Attr(coord_mode, 'string'),
    }
    output = Tensor(dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Interp", inputs=[input, sizes], outputs=[output], params=attr)
    return output


######### Element-wise Compare Operator ############
def __compare(tensor_i0: Tensor,
              tensor_i1: Tensor,
              type: str,
              scale: List[float] = None,
              zero_point: List[int] = None,
              out_name: str = None):
    assert type in [
        "Greater", "Less", "GreaterOrEqual", "LessOrEqual", "Equal", "NotEqual", "And", "Xor"
    ]
    o_dtype = same_dtype_check(tensor_i0.dtype, tensor_i1.dtype)
    assert tensor_i0.dtype in ["float32", "float16", "int8", "uint8"]
    if out_name is None:
        out_name = generate_name(type)
    attr = {
        "mode": Attr(type, "string"),
    }
    output = Tensor(dtype=o_dtype, name=out_name)
    if scale != None:
        assert len(scale) == 3
        zero_point = zero_point if zero_point is not None else [0, 0, 0]
        assert len(zero_point) == 3
        assert len(scale) == 3 and len(zero_point) == 3
        assert scale[0] == scale[1] and zero_point[0] == zero_point[1]
        output.quantization(scale=scale[2], zero_point=zero_point[2])
        tensor_i0.quantization(scale=scale[0], zero_point=zero_point[0])
        tensor_i1.quantization(scale=scale[1], zero_point=zero_point[1])
    TpuLang.insert_op("top.Compare", inputs=[tensor_i0, tensor_i1], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def gt(tensor_i0: Tensor,
       tensor_i1: Tensor,
       scale: List[float] = None,
       zero_point: List[int] = None,
       out_name: str = None):
    return __compare(tensor_i0, tensor_i1, "Greater", scale, zero_point, out_name)


@auto_name()
@annotation_check
@assert_with_out_name
def lt(tensor_i0: Tensor,
       tensor_i1: Tensor,
       scale: List[float] = None,
       zero_point: List[int] = None,
       out_name: str = None):
    return __compare(tensor_i0, tensor_i1, "Less", scale, zero_point, out_name)


@auto_name()
@annotation_check
@assert_with_out_name
def ge(tensor_i0: Tensor,
       tensor_i1: Tensor,
       scale: List[float] = None,
       zero_point: List[int] = None,
       out_name: str = None):
    return __compare(tensor_i0, tensor_i1, "GreaterOrEqual", scale, zero_point, out_name)


@auto_name()
@annotation_check
@assert_with_out_name
def le(tensor_i0: Tensor,
       tensor_i1: Tensor,
       scale: List[float] = None,
       zero_point: List[int] = None,
       out_name: str = None):
    return __compare(tensor_i0, tensor_i1, "LessOrEqual", scale, zero_point, out_name)


@auto_name()
@annotation_check
@assert_with_out_name
def eq(tensor_i0: Tensor,
       tensor_i1: Tensor,
       scale: List[float] = None,
       zero_point: List[int] = None,
       out_name: str = None):
    return __compare(tensor_i0, tensor_i1, "Equal", scale, zero_point, out_name)


@auto_name()
@annotation_check
@assert_with_out_name
def ne(tensor_i0: Tensor,
       tensor_i1: Tensor,
       scale: List[float] = None,
       zero_point: List[int] = None,
       out_name: str = None):
    return __compare(tensor_i0, tensor_i1, "NotEqual", scale, zero_point, out_name)


@auto_name()
@annotation_check
@assert_with_out_name
def and_op(tensor_i0: Tensor,
           tensor_i1: Tensor,
           scale: List[float] = None,
           zero_point: List[int] = None,
           out_name: str = None):
    assert (np.all((tensor_i0.buffer == 0) | (tensor_i0.buffer == 1)))
    assert (np.all((tensor_i1.buffer == 0) | (tensor_i1.buffer == 1)))
    if scale:
        assert (len(scale) == 3)
        assert (scale[0] == 1 and scale[1] == 1 and scale[2] == 1)
    return __compare(tensor_i0, tensor_i1, "And", scale, zero_point, out_name)


@auto_name()
@annotation_check
@assert_with_out_name
def xor_op(tensor_i0: Tensor,
           tensor_i1: Tensor,
           scale: List[float] = None,
           zero_point: List[int] = None,
           out_name: str = None):
    return __compare(tensor_i0, tensor_i1, "Xor", scale, zero_point, out_name)


@auto_name()
@annotation_check
@assert_with_out_name
def not_op(tensor_i0: Tensor,
           scale: List[float] = None,
           zero_point: List[int] = None,
           out_name: str = None):
    return __compare_const(tensor_i0, 0, "Not", scale, zero_point, out_name)


@auto_name()
@annotation_check
@assert_with_out_name
def or_op(tensor_i0: Tensor,
          tensor_i1: Tensor,
          scale: List[float] = None,
          zero_point: List[int] = None,
          out_name: str = None):
    o_dtype = same_dtype_check(tensor_i0.dtype, tensor_i1.dtype)
    assert tensor_i0.dtype in ["float32", "float16", "int8", "uint8"]
    if out_name is None:
        out_name = generate_name("or")
    tensor_tmp = _base_binary(tensor_i0,
                              tensor_i1,
                              "top.Add",
                              scale,
                              zero_point,
                              out_dtype=o_dtype,
                              out_name=out_name + "_add")
    if scale:
        return __compare_const(tensor_tmp, 0, "Greater", [scale[2], scale[2]], zero_point, out_name)
    else:
        return __compare_const(tensor_tmp, 0, "Greater", scale, zero_point, out_name)


@to_scalar(2)
def __compare_const(tensor_i0: Tensor,
                    scalar_i1: Scalar,
                    type: str,
                    scale: List[float] = None,
                    zero_point: List[int] = None,
                    out_name: str = None):
    assert tensor_i0.dtype in ["float32", "float16", "int8", "uint8"]
    if out_name is None:
        out_name = generate_name(type)
    attr = {
        "mode": Attr(type, "string"),
        "const_val": Attr(scalar_i1.value, "float64"),
        'inversed': Attr(False, "bool")
    }
    output = Tensor([], dtype=tensor_i0.dtype, name=out_name)
    if scale != None:
        zero_point = zero_point if zero_point is not None else [0, 0]
        assert len(scale) == 2 and len(zero_point) == 2
        output.quantization(scale=scale[1], zero_point=zero_point[1])
        tensor_i0.quantization(scale=scale[0], zero_point=zero_point[0])
    TpuLang.insert_op("top.CompareConst", inputs=[tensor_i0], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def gts(tensor_i0: Tensor,
        scalar_i1: Union[Scalar, int, float],
        scale: List[float] = None,
        zero_point: List[int] = None,
        out_name: str = None):
    return __compare_const(tensor_i0, scalar_i1, "Greater", scale, zero_point, out_name)


@auto_name()
@annotation_check
@assert_with_out_name
def lts(tensor_i0: Tensor,
        scalar_i1: Union[Scalar, int, float],
        scale: List[float] = None,
        zero_point: List[int] = None,
        out_name: str = None):
    return __compare_const(tensor_i0, scalar_i1, "Less", scale, zero_point, out_name)


@auto_name()
@annotation_check
@assert_with_out_name
def ges(tensor_i0: Tensor,
        scalar_i1: Union[Scalar, int, float],
        scale: List[float] = None,
        zero_point: List[int] = None,
        out_name: str = None):
    return __compare_const(tensor_i0, scalar_i1, "GreaterOrEqual", scale, zero_point, out_name)


@auto_name()
@annotation_check
@assert_with_out_name
def les(tensor_i0: Tensor,
        scalar_i1: Union[Scalar, int, float],
        scale: List[float] = None,
        zero_point: List[int] = None,
        out_name: str = None):
    return __compare_const(tensor_i0, scalar_i1, "LessOrEqual", scale, zero_point, out_name)


@auto_name()
@annotation_check
@assert_with_out_name
def eqs(tensor_i0: Tensor,
        scalar_i1: Union[Scalar, int, float],
        scale: List[float] = None,
        zero_point: List[int] = None,
        out_name: str = None):
    return __compare_const(tensor_i0, scalar_i1, "Equal", scale, zero_point, out_name)


@auto_name()
@annotation_check
@assert_with_out_name
def nes(tensor_i0: Tensor,
        scalar_i1: Union[Scalar, int, float],
        scale: List[float] = None,
        zero_point: List[int] = None,
        out_name: str = None):
    return __compare_const(tensor_i0, scalar_i1, "NotEqual", scale, zero_point, out_name)


@to_scalar(2)
@auto_name()
@annotation_check
@assert_with_out_name
def ands(tensor_i0: Tensor,
         scalar_i1: Union[Scalar, int, float],
         scale: List[float] = None,
         zero_point: List[int] = None,
         out_name: str = None):
    assert (np.all((tensor_i0.buffer == 0) | (tensor_i0.buffer == 1)))
    assert ((scalar_i1.value == 0) | (scalar_i1.value == 1))
    if scale:
        assert (len(scale) == 2)
        assert (scale[0] == 1 and scale[1] == 1)
    return __compare_const(tensor_i0, scalar_i1, "And", scale, zero_point, out_name)


@auto_name()
@annotation_check
@assert_with_out_name
def xors(tensor_i0: Tensor,
         scalar_i1: Union[Scalar, int, float],
         scale: List[float] = None,
         zero_point: List[int] = None,
         out_name: str = None):
    return __compare_const(tensor_i0, scalar_i1, "Xor", scale, zero_point, out_name)


@to_scalar(2)
@auto_name()
@annotation_check
@assert_with_out_name
def ors(tensor_i0: Tensor,
        scalar_i1: Union[Scalar, int, float],
        scale: List[float] = None,
        zero_point: List[int] = None,
        out_name: str = None):
    assert tensor_i0.dtype in ["float32", "float16", "int8", "uint8"]
    if out_name is None:
        out_name = generate_name("ors")
    add_output = Tensor(dtype=tensor_i0.dtype, name=out_name + "_add")
    final_output = Tensor(dtype=tensor_i0.dtype, name=out_name)

    if scale != None:
        zero_point = zero_point if zero_point is not None else [0, 0]
        assert len(scale) == 2 and len(zero_point) == 2
        add_output.quantization(scale=scale[1], zero_point=zero_point[1])
        final_output.quantization(scale=scale[1], zero_point=zero_point[1])
        tensor_i0.quantization(scale=scale[0], zero_point=zero_point[0])

    attr = {
        "const_val": Attr(scalar_i1.value, "float64"),
    }
    TpuLang.insert_op("top.AddConst", inputs=[tensor_i0], outputs=[add_output], params=attr)

    attr = {
        "mode": Attr("Greater", "string"),
        "const_val": Attr(float(0), 'float64'),
        "inversed": Attr(False, "bool"),
    }
    TpuLang.insert_op("top.CompareConst", inputs=[add_output], outputs=[final_output], params=attr)
    return final_output


######### Shape-Related Operator ############
@auto_name()
@annotation_check
@assert_with_out_name
def squeeze(tensor_i: Tensor, axis: Union[Tuple[int], List[int]], out_name: str = None):
    if isinstance(axis, Tuple): axis = list(axis)
    if out_name is None:
        out_name = generate_name("squeeze")
    attr = {
        "axes": ArrayAttr(axis),
    }
    output = Tensor(dtype=tensor_i.dtype, name=out_name)
    TpuLang.insert_op("top.Squeeze", inputs=[tensor_i], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def reshape(tensor: Tensor, new_shape: Union[Tuple[int], List[int], Tensor], out_name: str = None):
    output = Tensor(dtype=tensor.dtype, name=out_name)
    inputs = [tensor]
    attr = {}
    if isinstance(new_shape, Tuple): new_shape = list(new_shape)
    if not isinstance(new_shape, Tensor):
        attr["shape"] = ArrayAttr(new_shape)
    else:
        inputs.append(new_shape)
    TpuLang.insert_op("top.Reshape", inputs=inputs, outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def shape_fetch(tensor_i: Tensor,
                begin_axis: int = None,
                end_axis: int = None,
                step: int = 1,
                out_name: str = None):
    attr = {"step": Attr(step)}
    if begin_axis is not None:
        attr["start"] = Attr(begin_axis)
    if end_axis is not None:
        attr["end"] = Attr(end_axis)
    output = Tensor(dtype=tensor_i.dtype, name=out_name)
    TpuLang.insert_op("top.Shape", inputs=[tensor_i], outputs=[output], params=attr)
    return output


############## Normalization Operator ###############
@auto_name()
@annotation_check
@assert_with_out_name
def batch_norm(input: Tensor,
               mean: Tensor,
               variance: Tensor,
               gamma: Tensor = None,
               beta: Tensor = None,
               epsilon: float = 1e-5,
               out_name: str = None):
    assert epsilon >= 0
    assert input.dtype in ["float32", "float16"]
    assert mean.shape == variance.shape
    output = Tensor(dtype=input.dtype, name=out_name)
    attr = {"epsilon": Attr(epsilon, 'float64')}
    TpuLang.insert_op("top.BatchNorm",
                      inputs=[input, mean, variance, gamma, beta],
                      outputs=[output],
                      params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def layer_norm(input: Tensor,
               gamma: Tensor = None,
               beta: Tensor = None,
               epsilon: float = 1e-5,
               axis: int = 2,
               out_name: str = None):
    assert epsilon >= 0, "invalid epsilon"
    assert gamma is None or beta is None or gamma.shape == beta.shape
    assert input.dtype in ["float32", "float16"]
    output = Tensor(dtype=input.dtype, name=out_name)
    attr = {
        "eps": Attr(epsilon, 'float64'),
        "axis": Attr(axis, 'int32'),
        "normalized_shape": ArrayAttr([], "int64")
    }
    TpuLang.insert_op("top.LayerNorm", inputs=[input, gamma, beta], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def group_norm(input: Tensor,
               gamma: Tensor = None,
               beta: Tensor = None,
               epsilon: float = 1e-5,
               num_groups: int = 1,
               out_name: str = None):
    output = Tensor(dtype=input.dtype, name=out_name)
    assert input.dtype in ["float32", "float16"]
    attr = {"eps": Attr(epsilon, 'float64'), "num_groups": Attr(num_groups, 'int64')}
    TpuLang.insert_op("top.GroupNorm", inputs=[input, gamma, beta], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def rms_norm(input: Tensor,
             gamma: Tensor = None,
             epsilon: float = 1e-5,
             axis: int = -1,
             out_name: str = None):
    output = Tensor(dtype=input.dtype, name=out_name)
    assert input.dtype in ["float32", "float16"], "invalid input dtype"
    assert axis == -1 or axis == len(input.shape) - 1, "axis={} not supported yet".format(axis)
    if gamma:
        assert input.dtype == gamma.dtype, "invalid input and gamma dtype"
        assert input.shape[-1] == gamma.shape[0] and len(gamma.shape) == 1, \
            "invalid input/gamma shape"
    attr = {"eps": Attr(epsilon, 'float64')}
    TpuLang.insert_op("top.RMSNorm", inputs=[input, gamma], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def normalize(input: Tensor,
              p: float = 2.0,
              axes: Union[List[int], int] = 1,
              eps: float = 1e-12,
              out_name: str = None):
    assert input.dtype in ["float32", "float16"], "invalid input dtype"
    assert axes is not None, "axes is None"

    # axes in range [-l, l), l = len(input.shape)
    if isinstance(axes, list):
        axes_sort = sorted(axes)
        b = None
        for a in axes_sort:
            assert a < len(
                input.shape) or a >= -len(input.shape), "axes={} is out of range [-{}, {}).".format(
                    a, len(input.shape), len(input.shape))
            # in axes list, axis must be continous
            if b is None:
                b = a
            else:
                assert a == b + 1, "axes are not continuous"
                b = a

        assert not (-1 in axes and 0 in axes), "axes are not continuous"

    if isinstance(axes, int):
        assert axes < len(
            input.shape) or axes >= -len(input.shape), "axes={} is out of range [-{}, {}).".format(
                axis, len(input.shape), len(input.shape))
        axes = [axes]

    abs_tensor = Tensor(input.shape, dtype=input.dtype, name=out_name + "/Abs_output_0_Abs")
    TpuLang.insert_op("top.Abs", inputs=[input], outputs=[abs_tensor])

    pow_tensor = Tensor(input.shape, dtype=input.dtype, name=out_name + "/Pow_output_0_Pow")
    attr_pow = {"exponent": Attr(p, 'float64')}

    # pow = pow(abs , p)
    if p == 1.0:
        attr_copy = {
            "shape": ArrayAttr(abs_tensor.shape),
            "input_stride": ArrayAttr([1] * (len(abs_tensor.shape))),
            "output_stride": ArrayAttr([1] * (len(abs_tensor.shape))),
        }
        TpuLang.insert_op("top.Copy", [abs_tensor], [pow_tensor], params=attr_copy)
    if p == 2.0:
        TpuLang.insert_op("top.Mul", [abs_tensor, abs_tensor], [pow_tensor])
    else:
        TpuLang.insert_op("top.Pow", inputs=[abs_tensor], outputs=[pow_tensor], params=attr_pow)

    # sum = sum(exp_p)
    attr_reduce = {
        "axes": ArrayAttr(axes, "int64"),
        "keepdims": Attr(True, "bool"),
        "mode": Attr("ReduceSum", "string"),
    }
    sum_tensor = Tensor(dtype=input.dtype, name=out_name + "/ReduceSum_output_0_ReduceSum")
    TpuLang.insert_op("top.Reduce", inputs=[pow_tensor], outputs=[sum_tensor], params=attr_reduce)
    # TpuLang.insert_op("top.Reduce", inputs=[exp_p_tensor], outputs=[sum_tensor], params=attr_reduce)

    # norm = pow(sum, 1/p)
    norm_tensor = Tensor(sum_tensor.shape, dtype=input.dtype, name=out_name + '/Pow_1_output_0_Pow')
    attr_root = {"exponent": Attr(1 / p, 'float64')}
    TpuLang.insert_op("top.Pow", inputs=[sum_tensor], outputs=[norm_tensor], params=attr_root)

    # clip the norm with epsilon to aviod division by zero
    norm_clip_tensor = Tensor(norm_tensor.shape, dtype=input.dtype, name=out_name + "_clipNorm")
    attr_clip = {
        "min": Attr(eps, data_type="float64"),
        "max": Attr(float('inf'), data_type="float64")
    }
    TpuLang.insert_op("top.Clip", [norm_tensor], [norm_clip_tensor], params=attr_clip)

    # broadcast clipped norm tensor to input shape
    broadcast_tensor = Tensor(input.shape, dtype=input.dtype, name=out_name + "+_bcNorm")
    attr_bc = {"shape": ArrayAttr(input.shape)}
    TpuLang.insert_op("top.Expand", [norm_clip_tensor], [broadcast_tensor], params=attr_bc)

    # Normalize the tensor
    # output = input / norm
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Div", [input, broadcast_tensor], [output])

    return output


######### Select Operator ############


@auto_name()
@annotation_check
@assert_with_out_name
def lut(input: Tensor, table: Tensor, out_name: str = None):
    assert input.dtype == 'int8' or input.dtype == 'uint8'
    assert table.dtype == 'int8' or table.dtype == 'uint8'
    output = Tensor(input.shape, dtype=table.dtype, name=out_name)
    TpuLang.insert_op("top.Lut", inputs=[input, table], outputs=[output])
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def cond_select(cond: Tensor,
                tbrn: Union[Tensor, Scalar],
                fbrn: Union[Tensor, Scalar],
                out_name: str = None):
    assert tbrn.dtype == fbrn.dtype
    out_dtype = tbrn.dtype
    assert cond.dtype == "float32"
    params = {}
    if isinstance(tbrn, Tensor):
        assert tbrn.shape == cond.shape, "shape of `tbrn` and `cond` should be the same"
        params['y_is_const'] = Attr(False, 'bool')
    else:
        params['x_is_const'] = Attr(True, 'bool')
        params['x_const_val'] = Attr(float(tbrn.value), 'float64')
        tbrn = None
    if isinstance(fbrn, Tensor):
        assert fbrn.shape == cond.shape, "shape of `fbrn` and `cond` should be the same"
        params['y_is_const'] = Attr(False, 'bool')
    else:
        params['y_is_const'] = Attr(True, 'bool')
        params['y_const_val'] = Attr(float(fbrn.value), 'float64')
        fbrn = None
    output = Tensor(cond.shape, dtype=out_dtype, name=out_name)
    TpuLang.insert_op("top.Where", inputs=[cond, tbrn, fbrn], outputs=[output], params=params)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def select(lhs: Tensor, rhs: Tensor, tbrn: Tensor, fbrn: Tensor, type: str, out_name: str = None):
    assert lhs.shape == rhs.shape
    assert lhs.dtype == "float32"
    cond = __compare(lhs, rhs, type, out_name=f"{out_name}_compare")
    cond.shape = lhs.shape
    return cond_select(cond, tbrn, fbrn, out_name=f"{out_name}_cond_select")


@auto_name()
@annotation_check
@assert_with_out_name
def index_select(input: Tensor,
                 index: Tensor,
                 axis: int = -1,
                 out_name: str = None,
                 keep_dims: bool = True):
    attr = {
        "axis": Attr(axis, "int32"),
        "keepdims": Attr(keep_dims, "bool"),
    }
    output = Tensor(dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Gather", inputs=[input, index], outputs=[output], params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def mean_std_scale(input: Tensor,
                   std: List[float],
                   mean: List[float],
                   scale: Optional[Union[List[float], List[int]]] = None,
                   zero_points: Optional[List[int]] = None,
                   out_name: str = None,
                   odtype="float16",
                   round_mode: str = "half_away_from_zero"):
    idtype = input.dtype
    if idtype in ["float32", "uint8", "int8"] and odtype == "float16":
        #Align with IEEE 754 standard
        round_mode = "HALF_TO_EVEN"

    assert len(std) == len(mean) == input.shape[1]
    assert input.shape[1] <= 9
    assert len(input.shape) == 4 or len(input.shape) == 5

    h = input.shape[2]
    w = input.shape[3]
    op_params = {
        "channel_order": Attr("nchw", "string"),
        "resize_dims": ArrayAttr([h, w], "int64"),
        "std": ArrayAttr(std, "float64"),
        "mean": ArrayAttr(mean, "float64"),
        "scale": ArrayAttr(scale, "float64"),
        "zero_points": ArrayAttr(zero_points, "float64"),
        "rounding_mode": Attr(round_mode_convert(round_mode), "string"),
        "customization_format": Attr("RGB", "string"),
        "quant_mode": Attr("MultiplierShift", "string")
    }

    output = Tensor(input.shape, dtype=odtype, name=out_name)
    TpuLang.insert_op("top.MeanStdScale", [input], [output], params=op_params)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def scatter(input: Tensor, index: Tensor, updates: Tensor, axis: int = 0, out_name: str = None):
    o_dtype = input.dtype
    if axis < 0:
        assert axis <= len(input.shape), "axis is invalid"
        axis = len(input.shape) + axis
    else:
        assert axis < len(input.shape), "axis is invalid"
    attr = {
        "axis": Attr(axis, "int64"),
    }
    assert input.shape[:axis] == index.shape[:axis] == updates.shape[:axis], "input should have not less than shape of index and updates"
    assert len(index.shape) == len(updates.shape), "index and updates should have the same shape"
    output = Tensor(dtype=o_dtype, name=out_name)
    TpuLang.insert_op("top.ScatterElements",
                      inputs=[input, index, updates],
                      outputs=[output],
                      params=attr)
    return output


@auto_name()
@annotation_check
@assert_with_out_name
def matmulrq_int_op(
        input: Tensor,
        right: Tensor,
        bias=None,
        input_transpose: bool = False,
        right_transpose: bool = False,
        output_transpose: bool = False,
        keep_dims: bool = True,
        out_dtype: str = "int8",
        out_name: str = None,
        multiplier: Union[int, List[int]] = None,
        shift: Union[int, List[int]] = None,
        offset: Union[int, List[int]] = None,
        requant_mode: int = 2,  # Default to "MultiplierShift"
        round_mode: str = 'half_away_from_zero',
        rq_axis=-1):
    assert input_transpose == False and output_transpose == False
    assert out_dtype == "int8" or out_dtype == "int16"
    assert rq_axis == -1 or (rq_axis == len(input.shape) - 1)
    matmul_out_name = out_name + "_matmul" if out_name else "matmul_output"
    matmul_output = matmul_int(input,
                               right,
                               bias,
                               input_transpose=False,
                               right_transpose=False,
                               output_transpose=False,
                               keep_dims=True,
                               input_zp=0,
                               right_zp=0,
                               out_dtype="int32",
                               out_name=matmul_out_name)
    shift = shift if isinstance(shift, List) else [shift]
    shift = [-sft for sft in shift]
    requantized_output = requant_int(matmul_output,
                                     multiplier,
                                     shift,
                                     offset,
                                     requant_mode,
                                     out_dtype=out_dtype,
                                     out_name=out_name,
                                     round_mode=round_mode,
                                     rq_axis=rq_axis,
                                     fuse_rq_to_matmul=True)
    return requantized_output


@auto_name()
@annotation_check
@assert_with_out_name
def scatterND(input: Tensor, indices: Tensor, updates: Tensor, out_name: str = None):
    o_dtype = input.dtype

    assert len(input.shape) + len(indices.shape) - indices.shape[-1] - 1 == len(
        updates.shape), "The shapes of inputs are not correct."
    assert indices.dtype == 'uint32', "Wrong indices type"
    output = Tensor(dtype=o_dtype, name=out_name)

    TpuLang.insert_op("top.ScatterND", inputs=[input, indices, updates], outputs=[output])
    return output
