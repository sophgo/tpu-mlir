# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from typing import List, Union, Tuple, Optional
from .TpuLangConverter import TpuLangConverter, Graph, Tensor, Operator, Scalar, to_scalar, annotation_check, generate_name, auto_name
from tools.model_runner import mlir_inference, model_inference, show_fake_cmd
from tools.model_deploy import getCustomFormat
# from deprecated.sphinx import deprecated
from utils.mlir_shell import *
from utils.auto_remove import file_mark
from tools.npz_tool import npz_compare
import pymlir

import numpy as np
import logging

logger = logging.getLogger("root")

device_list = ['cpu', 'bm1684x', 'bm1688', 'cv183x']

class TpuLang:
    graph = None
    device = None
    chip = None

    def __init__(
        self,
        device: str,
    ):
        if device.lower() in device_list:
            TpuLang.chip = device
        else:
            KeyError('TpuLang: unsupported device.')
        # self.model_name = model_name
        TpuLang.graph = Graph()

    @staticmethod
    def insert_op(op_name: str, inputs: List[Tensor], outputs: List[Tensor], params: dict = {}):
        op = Operator(op_name, params=params, inputs=inputs, outputs=outputs)
        TpuLang.graph.operators.append(op)


def compile(name: str,
            inputs: List[Tensor],
            outputs: List[Tensor],
            cmp=True,
            refs=None,
            mode='f32',         # unused
            dynamic=False,
            asymmetric=False,
            no_save=False):
    logger.info("TPU-MLIR {}".format(pymlir.module().version))
    TpuLang.graph.inputs = inputs
    TpuLang.graph.outputs = outputs
    TpuLang.graph.quantized_type_inference()
    # convert to mlir
    converter = TpuLangConverter(name=name, graph=TpuLang.graph, mode="quantized")
    ctm_format = None
    fuse = False
    for input in TpuLang.graph.inputs:
        if input.is_preprocess:
            ctm_format = getCustomFormat(input.pixel_format, input.channel_format)
            fuse = True
            break
    if not no_save:
        save_input_reference(model_name=name, refs=refs)
        model_transform(name, converter)
        compare = cmp and refs != None
        model_lowering_and_inference(model_name=name, quant_mode="int8", chip=TpuLang.chip, cmp=compare, \
                                     asymmetric=asymmetric, ctm_format=ctm_format, fuse=fuse)
        bmodel_generate_and_inference(model_name=name, quant_mode="int8", dynamic=dynamic)
    else:
        originMlir_to_Model_without_quantize(converter=converter, model_name=name, mode="int8", chip=TpuLang.chip, asymmetric=asymmetric, dynamic=dynamic)


def compile_f32(name: str,
            inputs: List[Tensor],
            outputs: List[Tensor],
            cmp=True,
            refs=None,
            mode='f32',
            dynamic=False):
    TpuLang.graph.inputs = inputs
    TpuLang.graph.outputs = outputs
    TpuLang.graph.quantized_type_inference()
    # convert to mlir
    converter = TpuLangConverter(name=name, graph=TpuLang.graph, mode="f32")
    save_input_reference(model_name=name, refs=refs)
    model_transform(name, converter)
    compare = cmp and refs != None
    model_top_inference(model_name=name, cmp=compare)
    assert mode in ['f32', 'f16', 'bf16', 'int8', 'all', 'none']
    if mode == 'all':
        for m in ['f32', 'f16', 'bf16']:
            model_lowering_and_inference(model_name=name, quant_mode=m, chip=TpuLang.chip, cmp=cmp)
            bmodel_generate_and_inference(model_name=name, quant_mode=m, dynamic=dynamic)
    else:
        model_lowering_and_inference(model_name=name, quant_mode=mode, chip=TpuLang.chip, cmp=cmp)
        bmodel_generate_and_inference(model_name=name, quant_mode=mode, dynamic=dynamic)


def model_transform(model_name, converter: TpuLangConverter):
    mlir_file = model_name + '.mlir'
    mlir_origin = model_name + '_origin.mlir'
    converter.generate_mlir(mlir_origin)
    mlir_opt_for_top(mlir_origin, mlir_file)
    print("Mlir file generated:{}".format(mlir_file))


def save_input_reference(model_name, refs:dict):
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


def model_top_inference(model_name, cmp=False):
    in_f32_npz = model_name + '_in_f32.npz'
    mlir_file = model_name + '.mlir'
    input_data = np.load(in_f32_npz)
    top_npz = model_name + '_top_outputs.npz'
    show_fake_cmd(in_f32_npz, mlir_file, top_npz)
    f32_outputs = mlir_inference(input_data, mlir_file)
    np.savez(top_npz, **f32_outputs)
    if cmp:
        ref_npz = model_name + '_ref_output.npz'
        f32_blobs_compare(top_npz, ref_npz, '0.99,0.99')

def model_lowering_and_inference(model_name: str, quant_mode: str, chip: str, asymmetric: bool = False, \
                                 inference: bool = True, cmp: bool = False, ctm_format = "BGR_PLANAR", \
                                 fuse=False):
    top_mlir = "{}.mlir".format(model_name)
    tpu_mlir = "{}_{}.mlir".format(model_name, quant_mode)

    mlir_lowering(top_mlir, tpu_mlir, mode=quant_mode, chip=chip, asymmetric=asymmetric, \
                  customization_format=ctm_format, fuse_preprocess=fuse)
    if inference:
        in_f32_npz = model_name + '_in_f32.npz'
        tpu_npz = tpu_mlir.replace(".mlir", "_tpu_out.npz")
        input_data = np.load(in_f32_npz)
        file_mark(tpu_npz)
        show_fake_cmd(in_f32_npz, tpu_mlir, tpu_npz)
        tpu_mlir_outs = mlir_inference(input_data, tpu_mlir, dump_all=True)
        np.savez(tpu_npz, **tpu_mlir_outs)
        if cmp:
            if quant_mode == 'int8':
                ref_npz = model_name + '_ref_output.npz'
                npz_compare([ref_npz, tpu_npz, "--tolerance", "0.95,0.80", "-v"])
            else:
                top_npz = model_name + '_top_outputs.npz'
                npz_compare([top_npz, tpu_npz, "--tolerance", "0.95,0.80", "-v"])

def bmodel_generate_and_inference(model_name: str, quant_mode: str, inference: bool = True, dynamic: bool = False):
    # generate bmodel
    tpu_mlir = "{}_{}".format(model_name, quant_mode)
    tpu_final = tpu_mlir + "_final.mlir"
    bmodel = tpu_mlir + ".bmodel"
    mlir_to_model(tpu_mlir + ".mlir", bmodel, tpu_final, dynamic=dynamic)

    if inference:
        #inference
        in_f32_npz = model_name + '_in_f32.npz'
        tpu_npz = tpu_mlir + "_tpu_out.npz"
        input_data = np.load(in_f32_npz)
        model_npz = bmodel.replace("." + bmodel.split(".")[-1], "_model_out.npz")
        file_mark(model_npz)
        show_fake_cmd(in_f32_npz, bmodel, model_npz)
        model_outs = model_inference(input_data, bmodel)
        np.savez(model_npz, **model_outs)
        npz_compare([tpu_npz, model_npz, "--tolerance", "0.95,0.80", "-v"])


def init(device: str):
    TpuLang(device=device)


def deinit():
    TpuLang.graph = None
    TpuLang.device = None


def ArrayAttr(data: list, data_type: str = 'int64'):
    return [data, data_type, False]


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
            out_names.extend([generate_name(f"custom_{op_name}_{i}")
                     for i in range(len(out_names), out_num)])
    else:
        out_names = [generate_name(f"custom_{op_name}_{i}")
                     for i in range(out_num)]

    tensors_out = []
    for i, out_dtype in enumerate(out_dtypes):
        tensor_out = Tensor(dtype=out_dtype,
                            name=out_names[i])
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

def binary_dtype_check(tensor_i0: Union[Tensor, Scalar], tensor_i1: Union[Tensor, Scalar], out_dtype: str = None, sign: bool = False):
    in0_dtype = tensor_i0.dtype if isinstance(tensor_i0, Tensor) else tensor_i1.dtype
    in1_dtype = tensor_i1.dtype if isinstance(tensor_i1, Tensor) else tensor_i0.dtype
    if in0_dtype in ["float32", "float16"]:
        assert in0_dtype == in1_dtype
        out_dtype = in0_dtype if out_dtype == None else out_dtype
        assert in0_dtype == out_dtype
    elif in0_dtype.find("int") >= 0:
        assert in1_dtype.find("int") >= 0
        out_dtype = "int32" if out_dtype == None else out_dtype
        assert out_dtype.find("int") >= 0
        if sign:
            assert out_dtype.find("uint") < 0
    return out_dtype

def same_dtype_check(in0_dtype: str, in1_dtype: str = None, out_dtype: str = None):
    if in1_dtype is not None:
        assert in0_dtype == in1_dtype
    if out_dtype is not None:
        assert in0_dtype == out_dtype
    return in0_dtype

@auto_name()
@annotation_check
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
    o_dtype = input.dtype if out_dtype is None else out_dtype
    assert input.dtype in ["float32", "float16"]
    assert input.is_quantized is False
    assert weight.is_quantized is False

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
    dilation = [1, 1] if dilation is None else dilation
    stride = [1, 1] if stride is None else stride
    pad = [0, 0, 0, 0] if pad is None else pad
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
    output = Tensor(dtype=o_dtype, name=out_name)
    input.quantization(zero_point=input_zp)
    weight.quantization(zero_point=weight_zp)
    inputs = [input, weight, bias]
    TpuLang.insert_op("top.Conv", inputs=inputs, outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
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
    dilation = [1, 1] if dilation is None else dilation
    stride = [1, 1] if stride is None else stride
    pad = [0, 0, 0, 0] if pad is None else pad
    o_dtype = out_dtype if out_dtype is not None else input.dtype
    assert o_dtype in ["int8", "uint8"]
    assert input.is_quantized is True or input_scale is not None
    assert weight.is_quantized is True  or weight_scale is not None

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
    o_dtype = "float32" if out_dtype is None else out_dtype
    assert input.dtype in ["float32", "float16"]
    assert input.is_quantized is False
    assert weight.is_quantized is False

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
    dilation = [1, 1, 1] if dilation is None else dilation
    stride = [1, 1, 1] if stride is None else stride
    pad = [0, 0, 0, 0, 0, 0] if pad is None else pad
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
    dilation = [1, 1, 1] if dilation is None else dilation
    stride = [1, 1, 1] if stride is None else stride
    pad = [0, 0, 0, 0, 0, 0] if pad is None else pad
    o_dtype = out_dtype if out_dtype is not None else input.dtype
    assert o_dtype in ["int8", "uint8"]
    assert input.is_quantized is True or input_scale is not None
    assert weight.is_quantized is True  or weight_scale is not None

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
    o_dtype = "float32" if out_dtype is None else out_dtype
    assert input.dtype in ["float32", "float16"]
    assert input.is_quantized is False
    assert weight.is_quantized is False

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
    dilation = [1, 1] if dilation is None else dilation
    stride = [1, 1] if stride is None else stride
    pad = [0, 0, 0, 0] if pad is None else pad
    output_padding = [0, 0] if output_padding is None else output_padding
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
    o_dtype = "float32" if out_dtype is None else out_dtype
    assert input.dtype in ["float32", "float16"]
    assert input.is_quantized is False
    assert weight.is_quantized is False

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
    assert input.dtype in ["float32", "float16"]
    assert input.is_quantized is False
    assert right.is_quantized is False

    attr = {
        "right_transpose": Attr(right_transpose, "bool"),
        "left_transpose": Attr(input_transpose, "bool"),
        "output_transpose": Attr(output_transpose, "bool"),
        "hdim_is_batch": Attr(False, "bool"),
        "keep_dims": Attr(keep_dims, "bool"),
        "do_relu":  Attr(False, "bool"),
        "relu_limit":  Attr(-1.0, "float64")
    }

    output = Tensor(dtype=o_dtype, name=out_name)
    inputs = [input, right, bias]
    TpuLang.insert_op("top.MatMul", inputs=inputs, outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
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
        "do_relu":  Attr(False, "bool"),
        "relu_limit":  Attr(-1.0, "float64")
    }

    output = Tensor(dtype=o_dtype, name=out_name)
    input.quantization(zero_point=input_zp)
    right.quantization(zero_point=right_zp)
    inputs = [input, right, bias]
    TpuLang.insert_op("top.MatMul", inputs=inputs, outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
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

    attr = {
        "right_transpose": Attr(right_transpose, "bool"),
        "left_transpose": Attr(False, "bool"),
        "output_transpose": Attr(False, "bool"),
        "hdim_is_batch": Attr(False, "bool"),
        "keep_dims": Attr(keep_dims, "bool"),
        "do_relu":  Attr(False, "bool"),
        "relu_limit":  Attr(-1.0, "float64")
    }

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
def _base_binary(tensor_i0: Union[Tensor, Scalar], tensor_i1: Union[Tensor, Scalar], op_type: str,
        scale: List[float]=None, zero_point: List[int]=None, is_reverse : bool = None, out_dtype: str = None, out_name: str = None):
    o_dtype = binary_dtype_check(tensor_i0, tensor_i1, out_dtype)
    if scale is not None:
        zero_point = zero_point if zero_point is not None else [0, 0, 0]
        tensor0 = tensor_i0 if isinstance(tensor_i0, Tensor) else Tensor(dtype=tensor_i0.dtype, shape=[1], data=np.array([tensor_i0.value]).astype(tensor_i0.dtype), ttype="coeff")
        tensor1 = tensor_i1 if isinstance(tensor_i1, Tensor) else Tensor(dtype=tensor_i1.dtype, shape=[1], data=np.array([tensor_i1.value]).astype(tensor_i1.dtype), ttype="coeff")
        output = Tensor(dtype=o_dtype, name=out_name, scale=scale[2], zero_point=zero_point[2])
        tensor0.quantization(scale=scale[0], zero_point=zero_point[0])
        tensor1.quantization(scale=scale[1], zero_point=zero_point[1])
        TpuLang.insert_op(op_type, [tensor0, tensor1], [output])
        return output
    else:
        output = Tensor(dtype=o_dtype, name=out_name)
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
            TpuLang.insert_op(op_type+"Const", [tensor], [output], params = attr)
        return output

@to_scalar(2)
@auto_name()
@annotation_check
def add(tensor_i0: Union[Tensor, Scalar, int, float], tensor_i1: Union[Tensor, Scalar, int, float],
        scale: List[float]=None, zero_point: List[int]=None, out_dtype: str = None, out_name: str = None):
    return _base_binary(tensor_i0, tensor_i1, "top.Add", scale, zero_point, out_dtype=out_dtype, out_name=out_name)

@to_scalar(2)
@auto_name()
@annotation_check
def mul(tensor_i0: Union[Tensor, Scalar, int, float], tensor_i1: Union[Tensor, Scalar, int, float],
        scale: List[float]=None, zero_point: List[int]=None, out_dtype: str = None, out_name: str = None):
    return _base_binary(tensor_i0, tensor_i1, "top.Mul", scale, zero_point, out_dtype=out_dtype, out_name=out_name)

@to_scalar(2)
@auto_name()
@annotation_check
def sub(tensor_i0: Union[Tensor, Scalar, int, float], tensor_i1: Union[Tensor, Scalar, int, float],
        scale: List[float]=None, zero_point: List[int]=None, out_dtype: str = None, out_name: str = None):
    is_reverse = None if isinstance(tensor_i0, Tensor) else True
    return _base_binary(tensor_i0, tensor_i1, "top.Sub", scale, zero_point, is_reverse=is_reverse, out_dtype=out_dtype, out_name=out_name)

@to_scalar(2)
@auto_name()
@annotation_check
def div(tensor_i0: Union[Tensor, Scalar], tensor_i1: Union[Tensor, Scalar], out_name: str = None):
    o_dtype = same_dtype_check(tensor_i0.dtype, tensor_i1.dtype)
    output = Tensor([], dtype=o_dtype, name=out_name)
    if isinstance(tensor_i0, Tensor) and isinstance(tensor_i1, Tensor):
        TpuLang.insert_op("top.Div", [tensor_i0, tensor_i1], [output])
    else:
        if isinstance(tensor_i0, Tensor):
            del output
            tensor_i1.value = 1 / tensor_i1.value
            return _base_binary(tensor_i0, tensor_i1, "top.Mul", out_dtype=o_dtype, out_name=out_name)
        else:
            tensor_i0 = Tensor(dtype=tensor_i0.dtype, shape=[1], data=np.array([tensor_i0.value]).astype(tensor_i0.dtype), ttype="coeff")
            TpuLang.insert_op("top.Div", [tensor_i0, tensor_i1], [output])
    return output
    # is_reverse = None if isinstance(tensor_i0, Tensor) else True
    # return _base_binary(tensor_i0, tensor_i1, "top.Div", is_reverse=is_reverse, out_dtype=o_dtype, out_name=out_name)

@to_scalar(2)
@auto_name()
@annotation_check
def max(tensor_i0: Union[Tensor, Scalar, int, float], tensor_i1: Union[Tensor, Scalar, int, float],
        scale: List[float]=None, zero_point: List[int]=None, out_dtype: str = None, out_name: str = None):
    return _base_binary(tensor_i0, tensor_i1, "top.Max", scale, zero_point, out_dtype=out_dtype, out_name=out_name)

@to_scalar(2)
@auto_name()
@annotation_check
def min(tensor_i0: Union[Tensor, Scalar, int, float], tensor_i1: Union[Tensor, Scalar, int, float],
        scale: List[float]=None, zero_point: List[int]=None, out_dtype: str = None, out_name: str = None):
    return _base_binary(tensor_i0, tensor_i1, "top.Min", scale, zero_point, out_dtype=out_dtype, out_name=out_name)

def __binary_shift(tensor_i0: Tensor, tensor_i1: Tensor, type: str, shift: int,
                   out_dtype: str, is_reverse: bool=None, saturation: bool=True,
                   round_mode: str='half_away_from_zero', out_name: str = None):
    assert type in ["Add", "Sub", "Mul"]
    o_dtype = binary_dtype_check(tensor_i0, tensor_i1, out_dtype=out_dtype)
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
        attr["scale"] =  Attr(scalar.value, 'int32')
        if is_reverse is not None:
            attr["is_reverse"] = Attr(is_reverse, 'bool')
        TpuLang.insert_op("top.BinaryConstShift", inputs=[tensor], outputs=[output], params=attr)
    return output

@to_scalar(2)
@auto_name()
@annotation_check
def add_shift(tensor_i0: Union[Tensor, Scalar, int], tensor_i1: Union[Tensor, Scalar, int],
              shift: int, out_dtype: str, round_mode: str='half_away_from_zero', is_saturate: bool=True,
              out_name: str = None):
    return __binary_shift(tensor_i0, tensor_i1, "Add", shift, out_dtype, saturation=is_saturate,
                          round_mode=round_mode, out_name=out_name)

@to_scalar(2)
@auto_name()
@annotation_check
def sub_shift(tensor_i0: Union[Tensor, Scalar, int], tensor_i1: Union[Tensor, Scalar, int],
              shift: int, out_dtype: str, round_mode: str='half_away_from_zero', is_saturate: bool=True,
              out_name: str = None):
    is_reverse = None if isinstance(tensor_i0, Tensor) else True
    return __binary_shift(tensor_i0, tensor_i1, "Sub", shift, out_dtype, saturation=is_saturate,
                          is_reverse=is_reverse, round_mode=round_mode, out_name=out_name)

@to_scalar(2)
@auto_name()
@annotation_check
def mul_shift(tensor_i0: Union[Tensor, Scalar, int], tensor_i1: Union[Tensor, Scalar, int],
              shift: int, out_dtype: str, round_mode: str='half_away_from_zero', is_saturate: bool=True,
              out_name: str = None):
    return __binary_shift(tensor_i0, tensor_i1, "Mul", shift, out_dtype, saturation=is_saturate,
                          round_mode=round_mode, out_name=out_name)

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
def copy(input: Tensor, out_name: str = None):
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
def cast(tensor_i: Tensor,
         out_dtype: str = 'float32',
         out_name: str = None,
         round_mode: str = 'half_away_from_zero'):
    shape = tensor_i.shape
    attr = {
        "round_mode": Attr(round_mode_convert(round_mode), data_type="string"),
    }
    output = Tensor(shape, dtype=out_dtype, name=out_name)
    TpuLang.insert_op("top.Cast", [tensor_i], [output], params=attr)
    return output

@auto_name()
@annotation_check
def clamp(input: Tensor, min:float, max:float, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    attr = {
        "min": Attr(min, data_type="float64"),
        "max": Attr(max, data_type="float64"),
    }
    TpuLang.insert_op("top.Clip", [input], [output], params=attr)
    return output


###### quant operator ########
@auto_name()
@annotation_check
def requant_fp_to_int(tensor_i: Tensor,
                      scale: Union[float, List[float]],
                      offset: Union[int, List[int]],
                      requant_mode: int,
                      out_dtype: str,
                      out_name: str=None,
                      round_mode: str='half_away_from_zero'):
    output = Tensor(tensor_i.shape, name=out_name, dtype=out_dtype, scale=scale, zero_point=offset)
    attr = {
        "round_mode": Attr(round_mode_convert(round_mode), data_type="string"),
    }
    TpuLang.insert_op("top.Cast", inputs=[tensor_i], outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
def dequant_int_to_fp(tensor_i: Tensor,
                      scale: Union[float, List[float]],
                      offset: Union[int, List[int], float, List[float]],
                      out_dtype: str="float32",
                      out_name: str=None,
                      round_mode: str='half_away_from_zero'):
    assert out_dtype in ["float32", "float16"]
    output = Tensor(tensor_i.shape, name=out_name, dtype=out_dtype)
    tensor_i.quantization(scale=scale, zero_point=offset)
    attr = {
        "round_mode": Attr(round_mode_convert(round_mode), data_type="string"),
    }
    TpuLang.insert_op("top.Cast", inputs=[tensor_i], outputs=[output], params=attr)
    return output

def round_mode_convert(rmode: str):
    round_mode = "".join([r.capitalize() for r in rmode.split('_')])
    assert round_mode in ["HalfAwayFromZero", "HalfUp", "HalfDown", "HalfToEven", "HalfToOdd", "HalfTowardsZero", "TowardsZero", "Up", "Down"]
    return round_mode

@auto_name()
@annotation_check
def requant_int(tensor_i: Tensor,
                mul: Union[int, List[int]],
                shift: Union[int, List[int]],
                offset: Union[int, List[int]],
                requant_mode: int,
                out_dtype: str="int8",
                out_name=None,
                round_mode='half_away_from_zero'):
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
        "round_mode": Attr(round_mode_convert(round_mode), "string")
    }
    TpuLang.insert_op("top.RequantInt", inputs=[tensor_i], outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
def dequant_int(tensor_i: Tensor,
                mul: Union[int, List[int]],
                shift: Union[int, List[int]],
                offset: Union[int, List[int]],
                lshift: int,
                requant_mode: int,
                out_dtype: str="int8",
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
def requant_fp(tensor_i: Tensor,
               scale: Union[float, List[float]],
               offset: Union[float, List[float]],
               out_dtype: str,
               out_name: str=None,
               round_mode: str='half_away_from_zero',
               first_round_mode: str='half_away_from_zero'):
    scale = scale if isinstance(scale, List) else [scale]
    offset = offset if isinstance(offset, List) else [offset]
    output = Tensor(tensor_i.shape, name=out_name, dtype=out_dtype, zero_point=(int)(offset[0]))
    attr = {
        "scale": ArrayAttr(scale, "float64"),
        "offset": ArrayAttr(offset, "float64"),
        "quant_mode": Attr("MultiplierShift", "string"),
        "round_mode": Attr(round_mode_convert(round_mode), "string"),
        "first_round_mode": Attr(round_mode_convert(first_round_mode), "string"),
    }
    TpuLang.insert_op("top.RequantFp", inputs=[tensor_i], outputs=[output], params=attr)
    return output

######## Up / Down Scaling Operator #########
@auto_name()
@annotation_check
def maxpool2d(input: Tensor,
            kernel: List[int]=None,
            stride: List[int] = None,
            pad: List[int] = None,
            ceil_mode: bool = False,
            scale: List[float] = None,
            zero_point: List[int] = None,
            out_name: str = None,
            round_mode : str="half_away_from_zero"):
    assert(not kernel or (len(kernel) == 2))
    assert(not stride or len(stride) == 2)
    assert(not pad or len(pad) == 4)
    kernel = [] if kernel is None else kernel
    stride = [1, 1] if stride is None else stride
    pad = [0, 0, 0, 0] if pad is None else pad
    o_dtype = input.dtype

    attr = {
        "kernel_shape": ArrayAttr(kernel),
        "strides": ArrayAttr(stride),
        "pads": ArrayAttr(pad),
        "ceil_mode": Attr(ceil_mode, "bool"),
        "keepdims": Attr(True, "bool"),
        "pad_value": Attr(0),
        "count_include_pad": Attr(False, "bool"),
        "do_relu":  Attr(False, "bool"),
        "relu_limit":  Attr(-1.0, "float64"),
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
def maxpool2d_with_mask(input: Tensor,
                        kernel: List[int]=None,
                        stride: List[int] = None,
                        pad: List[int] = None,
                        ceil_mode: bool = False,
                        out_name: str = None,
                        mask_name: str = None):
    assert(not kernel or (len(kernel) == 2))
    assert(not stride or len(stride) == 2)
    assert(not pad or len(pad) == 4)
    kernel = [] if kernel is None else kernel
    stride = [1, 1] if stride is None else stride
    pad = [0, 0, 0, 0] if pad is None else pad
    o_dtype = input.dtype

    attr = {
        "kernel_shape": ArrayAttr(kernel),
        "strides": ArrayAttr(stride),
        "pads": ArrayAttr(pad),
        "ceil_mode": Attr(ceil_mode, "bool"),
        "keepdims": Attr(True, "bool"),
        "pad_value": Attr(0),
        "count_include_pad": Attr(False, "bool"),
        "do_relu":  Attr(False, "bool"),
        "relu_limit":  Attr(-1.0, "float64")
    }

    output = Tensor(dtype=o_dtype, name=out_name)
    mask = Tensor(dtype="int32", name=mask_name)
    TpuLang.insert_op("top.MaxPoolWithMask", inputs=[input], outputs=[output, mask], params=attr)
    return output, mask

@auto_name()
@annotation_check
def avgpool2d(input: Tensor,
            kernel: List[int]=None,
            stride: List[int] = None,
            pad: List[int] = None,
            ceil_mode: bool = False,
            scale: List[float] = None,
            zero_point: List[int] = None,
            out_name: str = None,
            count_include_pad : bool = False,
            round_mode : str="half_away_from_zero",
            first_round_mode : str="half_away_from_zero"):
    kernel = [] if kernel is None else kernel
    stride = [1, 1] if stride is None else stride
    pad = [0, 0, 0, 0] if pad is None else pad
    o_dtype = input.dtype

    attr = {
        "kernel_shape": ArrayAttr(kernel),
        "strides": ArrayAttr(stride),
        "pads": ArrayAttr(pad),
        "ceil_mode": Attr(ceil_mode, "bool"),
        "keepdims": Attr(True, "bool"),
        "pad_value": Attr(0),
        "count_include_pad": Attr(count_include_pad, "bool"),
        "do_relu":  Attr(False, "bool"),
        "relu_limit":  Attr(-1.0, "float64"),
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
def relu(input: Tensor, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Relu", inputs=[input], outputs=[output])
    return output

@auto_name()
@annotation_check
def prelu(input: Tensor, slope : Tensor, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.PRelu", inputs=[input, slope], outputs=[output])
    return output

@auto_name()
@annotation_check
def leaky_relu(input: Tensor, negative_slope: float = 0.01, out_name: str = None, round_mode : str="half_away_from_zero",):
    attr = {
        "alpha": Attr(negative_slope, data_type="float64"),
        "round_mode": Attr(round_mode_convert(round_mode), data_type="string"),
    }
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.LeakyRelu", inputs=[input], outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
def abs(input: Tensor, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Abs", inputs=[input], outputs=[output])
    return output

def _active_scale(input:Tensor, output: Tensor, scale: List[float]=None, zero_point: List[int]=None):
    if scale != None:
        zero_point = zero_point if zero_point is not None else [0, 0]
        assert len(scale) == 2 and len(zero_point) == 2
        output.quantization(scale=scale[1], zero_point=zero_point[1])
        input.quantization(scale=scale[0], zero_point=zero_point[0])

@auto_name()
@annotation_check
def ceil(input: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Ceil", inputs=[input], outputs=[output])
    return output

@auto_name()
@annotation_check
def floor(input: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Floor", inputs=[input], outputs=[output])
    return output

@auto_name()
@annotation_check
def round(input: Tensor, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Round", inputs=[input], outputs=[output])
    return output

@auto_name()
@annotation_check
def sin(input: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Sin", inputs=[input], outputs=[output])
    return output

@auto_name()
@annotation_check
def cos(input: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Cos", inputs=[input], outputs=[output])
    return output

@auto_name()
@annotation_check
def exp(input: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Exp", inputs=[input], outputs=[output])
    return output

@auto_name()
@annotation_check
def ln(input: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Log", inputs=[input], outputs=[output])
    return output

@auto_name()
@annotation_check
def tanh(input: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None, round_mode : str="half_away_from_zero",):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    attr = {
        "round_mode": Attr(round_mode_convert(round_mode), data_type="string"),
    }
    TpuLang.insert_op("top.Tanh", inputs=[input], outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
def sigmoid(input: Tensor, scale: List[float]=None, zero_point: List[int]=None,
            out_name: str = None, round_mode : str="half_away_from_zero"):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    attr = {
        "round_mode": Attr(round_mode_convert(round_mode), "string"),
    }
    TpuLang.insert_op("top.Sigmoid", inputs=[input], outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
def elu(input: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    attr = {
        "alpha": Attr(1.0, "float64"),
    }
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Elu", inputs=[input], outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
def square(input: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    scale = [scale[0], scale[0], scale[1]] if scale != None else scale
    zero_point = [zero_point[0], zero_point[0], zero_point[1]] if zero_point != None else zero_point
    return mul(input, input, scale=scale, zero_point=zero_point, out_dtype=input.dtype, out_name=out_name)

@auto_name()
@annotation_check
def sqrt(input: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Sqrt", inputs=[input], outputs=[output])
    return output

@auto_name()
@annotation_check
def rsqrt(input: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Rsqrt", inputs=[input], outputs=[output])
    return output

@auto_name()
@annotation_check
def silu(input: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.SiLU", inputs=[input], outputs=[output])
    return output

@auto_name()
@annotation_check
def erf(input: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Erf", inputs=[input], outputs=[output])
    return output

@auto_name()
@annotation_check
def log_sigmoid(input: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Sigmoid", inputs=[input], outputs=[output], params={"log", Attr(True, bool)})
    return output

@auto_name()
@annotation_check
def tan(input: Tensor, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Tan", inputs=[input], outputs=[output])
    return output

@auto_name()
@annotation_check
def softmax(input: Tensor, axis: int, out_name: str = None):
    attr = {
        "axis": Attr(axis, data_type="int32"),
    }
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Softmax", inputs=[input], outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
def softmax_int(input: Tensor, axis: int, scale: List[float], zero_point: List[int] = None,
                out_name: str = None, round_mode : str="half_away_from_zero"):
    attr = {
        "axis": Attr(axis, data_type="int32"),
        "round_mode": Attr(round_mode_convert(round_mode), "string"),
    }
    zero_point = zero_point if zero_point is not None else [0, 0]
    assert len(scale) == 2 and len(zero_point) == 2
    output = Tensor(input.shape, dtype=input.dtype, name=out_name, scale=scale[1], zero_point=zero_point[1])
    input.quantization(scale=scale[0], zero_point=zero_point[0])
    TpuLang.insert_op("top.Softmax", inputs=[input], outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
def mish(input: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Mish", inputs=[input], outputs=[output])
    return output

@auto_name()
@annotation_check
def hswish(input: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.HardSwish", inputs=[input], outputs=[output])
    return output

@auto_name()
@annotation_check
def arccos(input: Tensor, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Arccos", inputs=[input], outputs=[output])
    return output

@auto_name()
@annotation_check
def arctanh(input: Tensor, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Arctanh", inputs=[input], outputs=[output])
    return output

@auto_name()
@annotation_check
def sinh(input: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Sinh", inputs=[input], outputs=[output])
    return output

@auto_name()
@annotation_check
def cosh(input: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Cosh", inputs=[input], outputs=[output])
    return output

@auto_name()
@annotation_check
def sign(input: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.Sign", inputs=[input], outputs=[output])
    return output

@auto_name()
@annotation_check
def gelu(input: Tensor, scale: List[float]=None, zero_point: List[int]=None,
         out_name: str = None, round_mode : str="half_away_from_zero"):
    output = Tensor(dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    attr = {
        "round_mode": Attr(round_mode_convert(round_mode), data_type="string"),
    }
    TpuLang.insert_op("top.GELU", inputs=[input], outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
def hsigmoid(input: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    attr = {
        "alpha": Attr(1/6, data_type="float64"),
        "beta": Attr(0.5, data_type="float64"),
    }
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    _active_scale(input, output, scale, zero_point)
    TpuLang.insert_op("top.HardSigmoid", inputs=[input], outputs=[output], params=attr)
    return output


######### Sort Operator ############
@auto_name()
@annotation_check
def arg(input: Tensor,
        method: str = "max",
        axis: int = 0,
        keep_dims: bool = True,
        out_name: str = None):
    dims = len(input.shape)
    assert -dims <= axis and axis < dims, "axis is invalid"
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
def sort(input: Tensor,
         axis: int = -1,
         descending: bool = True,
         out_name: str = None):
    dims = len(input.shape)
    assert -dims <= axis and axis < dims, "axis is invalid"
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
def argsort(input: Tensor,
            axis: int = -1,
            descending: bool = True,
            out_name: str = None):
    dims = len(input.shape)
    assert -dims <= axis and axis < dims, "axis is invalid"
    attr = {
        "axis": Attr(axis),
        "descending": Attr(descending, "bool"),
    }
    output = Tensor(dtype='int32', name=out_name)
    TpuLang.insert_op("top.Sort", inputs=[input], outputs=[None, output], params=attr)
    return output

@auto_name()
@annotation_check
def sort_by_key(input: Tensor,
                key: Tensor,
                axis: int = -1,
                descending: bool = True,
                out_name: str = None):
    dims = len(input.shape)
    assert -dims <= axis and axis < dims, "axis is invalid"
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
def permute(input: Tensor, order: Union[Tuple[int], List[int]], out_name: str = None):
    attr = {
        "order": ArrayAttr(order),
    }
    output = Tensor(dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Permute", inputs=[input], outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
def tile(input: Tensor, reps: Union[Tuple[int], List[int]], out_name: str = None):
    attr = {
        "tile": ArrayAttr(reps),
    }
    output = Tensor(dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Tile", inputs=[input], outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
def concat(inputs: List[Tensor], scales: Optional[Union[List[float],List[int]]] = None,
           zero_points: Optional[List[int]] = None, axis: int = 0, out_name: str = None,
           dtype: str="float32", round_mode: str="half_away_from_zero"):
    if scales is None:
        scales = [None] * (len(inputs) + 1)
    if zero_points is None:
        zero_points = [None] * (len(inputs) + 1)
    assert len(inputs) > 1, "concat should have more than one input"
    attr = {
        "axis": Attr(axis, "int32"),
        "round_mode": Attr(round_mode_convert(round_mode), data_type="string"),
    }
    input_list_ = []
    for index, i_tensor in  enumerate(inputs):
        i_tensor.quantization(scale=scales[index], zero_point=zero_points[index])
        input_list_.append(i_tensor)
    output = Tensor(dtype=inputs[0].dtype, name=out_name)
    if dtype != "float32":
        output.quantization(scale=scales[len(scales) - 1], zero_point=zero_points[len(zero_points) - 1])
    TpuLang.insert_op("top.Concat", inputs=input_list_, outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
def broadcast(input: Tensor, reps: Union[Tuple[int], List[int]], out_name: str = None):
    output = Tensor(dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Expand", inputs=[input], outputs=[output], params={"shape": ArrayAttr(reps)})
    return output

@auto_name()
@annotation_check
def nonzero(inputs: Tensor, dtype = "int32", out_name: str = None):
    attr = {
        "order": Attr("ColMajor", "string"),
        }
    output = Tensor(dtype=dtype, name=out_name)
    TpuLang.insert_op("top.NonZero", inputs=[inputs], outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
def upsample(input: Tensor, scale: int = 2, out_name: str = None):
    attr = {
        "scale_h": Attr(scale, data_type="int64"),
        "scale_w": Attr(scale, data_type="int64"),
    }
    output = Tensor(dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Upsample", inputs=[input], outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
def reduce(input: Tensor, method: str = "ReduceSum", axes: Union[List[int], int] = [1,2], keep_dims: bool = True, out_name: str = None):
    assert(method in ["ReduceMin", "ReduceMax", "ReduceMean", "ReduceProd", "ReduceL2", "ReduceL1","ReduceSum"])
    if isinstance(axes, int):
        axes = [axes]
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
def unsqueeze(input: Tensor, axes: List[int] = [1,2], out_name: str = None):
    attr = {
        "axes": ArrayAttr(axes, "int64"),
    }
    output = Tensor(dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Unsqueeze", inputs=[input], outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
def split(input: Tensor,
          axis: int = 0,
          num: int = 1,
          size: Union[Tuple[int], List[int]] = (),
          out_name: str = None) -> List[Tensor]:
    assert(num > 1 and "number of split output should be more than 1")
    # if not size:
    #     assert(input.shape[axis] % num == 0 and "invalid split size")
    #     size = [int(input.shape[axis] / num)] * num
    # else:
    #     assert(num == len(size) and "size should be the same as num")
    #     assert(sum(size) == input.shape[axis] and "invalid size")

    attr = {
        "axis": Attr(axis, "int32"),
        "num": Attr(num),
    }
    if len(size) != 0:
        attr["split_size"] = ArrayAttr(size)

    outputs = []
    for i in range(num):
        outputs.append(Tensor(dtype=input.dtype, name=f"{out_name}_{i}"))

    TpuLang.insert_op("top.Split", inputs=[input], outputs=outputs, params=attr)
    return outputs

@auto_name()
@annotation_check
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
def pad(input: Tensor,
        method: str = "constant",
        value: Scalar = None,
        padding: Union[Tuple[int], List[int]] = None,
        out_name: str = None):
    assert(method in ["constant","reflect","symmetric","edge"] and "Not supported pad type")
    assert(not padding or len(padding) == 2 * len(input.shape) and "Invalid padding length")
    if out_name is None:
        out_name = generate_name("pad")
    if padding is None:
        padding = [0] * (len(input.shape)*2)
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
def repeat(input: Tensor, reps: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("repeat")
    # reps = Tensor(data = reps, shape = input.shape)
    output = Tensor(dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Repeat", inputs=[input, reps], outputs=[output])
    return output

@auto_name()
@annotation_check
def extract(input: Tensor, start: Union[List[int], Tuple[int]] = None, end: Union[List[int], Tuple[int]] = None, stride: Union[List[int], Tuple[int]] = None, out_name: str = None):
    dims = len(input.shape)
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
    output = Tensor(dtype=input.dtype, name=out_name)
    attr = {
        "offset": ArrayAttr(start),
        "ends": ArrayAttr(end),
        "steps": ArrayAttr(stride),
    }
    TpuLang.insert_op("top.Slice", inputs=[input, None, None, None], outputs=[output], params=attr)
    return output


######### Vision Operator ############
@auto_name()
@annotation_check
def topk(input: Tensor,
         axis: int,
         k: int,
         out_name: str = None):
    dims = len(input.shape)
    assert -dims <= axis and axis < dims, "axis is invalid"
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
def nms(boxes: Tensor,
        scores: Tensor,
        box_format: str = 'PYTORCH',
        max_box_num_per_class: int = 0,
        out_name: str = None):
    boxes_dims = len(boxes.shape)
    scores_dims = len(scores.shape)
    assert boxes_dims == 3, f"dims of boxes expect 3 but get {boxes_dims}"
    assert scores_dims == 3, f"dims of boxes expect 3 but get {scores_dims}"
    assert box_format in ['TENSORFLOW', 'PYTORCH'], f"box_format:{box_format} is not supported"
    assert max_box_num_per_class >= 0, f"max_box_num_per_class:{max_box_num_per_class} is not valid"
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
    assert coord_mode in ["align_corners", "pytorch_half_pixel", "half_pixel", "asymmetric"], f"coord_mode:{coord_mode} is not supported"
    attr = {
        "scale_h": Attr(scale_h, 'float64'),
        "scale_w": Attr(scale_w, 'float64'),
        "mode": Attr(method, 'string'),
        "coord_mode": Attr(coord_mode, 'string'),
    }
    output = Tensor(dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Interp", inputs=[input, None], outputs=[output], params=attr)
    return output


######### Element-wise Compare Operator ############
def __compare(tensor_i0: Tensor, tensor_i1: Tensor, type: str, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    assert type in ["Greater", "Less", "GreaterOrEqual", "LessOrEqual", "Equal", "NotEqual"]
    o_dtype = same_dtype_check(tensor_i0.dtype, tensor_i1.dtype)
    if out_name is None:
        out_name = generate_name(type)
    attr = {
        "mode": Attr(type, "string"),
    }
    output = Tensor(dtype=o_dtype, name=out_name)
    if scale != None:
        zero_point = zero_point if zero_point is not None else [0, 0, 0]
        assert len(scale) == 3 and len(zero_point) == 3
        assert scale[0] == scale[1] and zero_point[0] == zero_point[1]
        output.quantization(scale=scale[2], zero_point=zero_point[2])
        tensor_i0.quantization(scale=scale[0], zero_point=zero_point[0])
        tensor_i1.quantization(scale=scale[1], zero_point=zero_point[1])
    TpuLang.insert_op("top.Compare", inputs=[tensor_i0, tensor_i1], outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
def gt(tensor_i0: Tensor, tensor_i1: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    return __compare(tensor_i0, tensor_i1, "Greater", scale, zero_point, out_name)

@auto_name()
@annotation_check
def lt(tensor_i0: Tensor, tensor_i1: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    return __compare(tensor_i0, tensor_i1, "Less", scale, zero_point, out_name)

@auto_name()
@annotation_check
def ge(tensor_i0: Tensor, tensor_i1: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    return __compare(tensor_i0, tensor_i1, "GreaterOrEqual", scale, zero_point, out_name)

@auto_name()
@annotation_check
def le(tensor_i0: Tensor, tensor_i1: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    return __compare(tensor_i0, tensor_i1, "LessOrEqual", scale, zero_point, out_name)

@auto_name()
@annotation_check
def eq(tensor_i0: Tensor, tensor_i1: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    return __compare(tensor_i0, tensor_i1, "Equal", scale, zero_point, out_name)

@auto_name()
@annotation_check
def ne(tensor_i0: Tensor, tensor_i1: Tensor, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    return __compare(tensor_i0, tensor_i1, "NotEqual", scale, zero_point, out_name)

@to_scalar(2)
def __compare_const(tensor_i0: Tensor, scalar_i1: Scalar, type: str, scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
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
def gts(tensor_i0: Tensor, scalar_i1: Union[Scalar, int, float], scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    return __compare_const(tensor_i0, scalar_i1, "Greater", scale, zero_point, out_name)

@auto_name()
@annotation_check
def lts(tensor_i0: Tensor, scalar_i1: Union[Scalar, int, float], scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    return __compare_const(tensor_i0, scalar_i1, "Less", scale, zero_point, out_name)

@auto_name()
@annotation_check
def ges(tensor_i0: Tensor, scalar_i1: Union[Scalar, int, float], scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    return __compare_const(tensor_i0, scalar_i1, "GreaterOrEqual", scale, zero_point, out_name)

@auto_name()
@annotation_check
def les(tensor_i0: Tensor, scalar_i1: Union[Scalar, int, float], scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    return __compare_const(tensor_i0, scalar_i1, "LessOrEqual", scale, zero_point, out_name)

@auto_name()
@annotation_check
def eqs(tensor_i0: Tensor, scalar_i1: Union[Scalar, int, float], scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    return __compare_const(tensor_i0, scalar_i1, "Equal", scale, zero_point, out_name)

@auto_name()
@annotation_check
def nes(tensor_i0: Tensor, scalar_i1: Union[Scalar, int, float], scale: List[float]=None, zero_point: List[int]=None, out_name: str = None):
    return __compare_const(tensor_i0, scalar_i1, "NotEqual", scale, zero_point, out_name)


######### Shape-Related Operator ############
@auto_name()
@annotation_check
def squeeze(tensor_i: Tensor, axis: Union[Tuple[int], List[int]], out_name: str = None):
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
def reshape(tensor: Tensor, new_shape: Union[Tuple[int], List[int], Tensor], out_name: str = None):
    output = Tensor(dtype=tensor.dtype, name=out_name)
    inputs = [tensor]
    attr = {}
    if not isinstance(new_shape, Tensor):
        attr["shape"] = ArrayAttr(new_shape)
    else:
        inputs.append(new_shape)
    TpuLang.insert_op("top.Reshape", inputs=inputs, outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
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
def batch_norm(input: Tensor, mean: Tensor, variance: Tensor,
               gamma: Tensor = None, beta: Tensor = None,
               epsilon: float = 1e-5, out_name: str = None):
    assert epsilon >= 0
    output = Tensor(dtype=input.dtype, name=out_name)
    attr = {"epsilon": Attr(epsilon, 'float64')}
    TpuLang.insert_op("top.BatchNorm", inputs=[input, mean, variance, gamma, beta], outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
def layer_norm(input: Tensor, gamma: Tensor = None, beta: Tensor = None,
               epsilon: float = 1e-5, axis: int = 2, out_name: str = None):
    assert epsilon >= 0
    output = Tensor(dtype=input.dtype, name=out_name)
    attr = {"eps": Attr(epsilon, 'float64'), "axis":  Attr(axis, 'int32'), "normalized_shape":  ArrayAttr([], "int64")}
    TpuLang.insert_op("top.LayerNorm", inputs=[input, gamma, beta], outputs=[output], params=attr)
    return output

@auto_name()
@annotation_check
def group_norm(input: Tensor, gamma: Tensor = None, beta: Tensor = None,
               epsilon: float = 1e-5, num_groups: int = 1, out_name: str = None):
    output = Tensor(dtype=input.dtype, name=out_name)
    attr = {"eps": Attr(epsilon, 'float64'), "num_groups":  Attr(num_groups, 'int64')}
    TpuLang.insert_op("top.GroupNorm", inputs=[input, gamma, beta], outputs=[output], params=attr)
    return output

######### Select Operator ############

@auto_name()
@annotation_check
def lut(input: Tensor, table: Tensor, out_name: str = None):
    assert input.dtype == 'int8' or input.dtype == 'uint8'
    assert table.dtype == 'int8' or table.dtype == 'uint8'
    output = Tensor(input.shape, dtype=table.dtype, name=out_name)
    TpuLang.insert_op("top.Lut", inputs=[input, table], outputs=[output])
    return output

@auto_name()
@annotation_check
def cond_select(cond: Tensor, tbrn: Union[Tensor, Scalar], fbrn: Union[Tensor, Scalar], out_name: str = None):
    assert tbrn.dtype == fbrn.dtype
    out_dtype = tbrn.dtype
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
def select(lhs: Tensor, rhs: Tensor, tbrn: Tensor, fbrn: Tensor, type: str, out_name: str = None):
    assert lhs.shape == rhs.shape
    cond = __compare(lhs, rhs, type, f"{out_name}_compare")
    cond.shape = lhs.shape
    return cond_select(cond, tbrn, fbrn, f"{out_name}_cond_select")

@auto_name()
@annotation_check
def index_select(input: Tensor, index : Tensor, axis : int = -1, out_name: str = None, keep_dims: bool = True):
    attr = {
        "axis": Attr(axis, "int32"),
        "keepdims": Attr(keep_dims, "bool"),
    }
    output = Tensor(dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Gather", inputs=[input, index], outputs=[output], params=attr)
    return output
