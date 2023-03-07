# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from typing import List, Union
from .MLIRImporter import Top
from .TpuLangConverter import TpuLangConverter, Graph, Tensor, Operator
from deprecated.sphinx import deprecated
from utils.mlir_shell import *

import numpy as np


class TpuLang:
    graph = None
    device = None

    def __init__(
        self,
        device: str,
    ):
        device_list = ['cpu', 'bm1684x', 'bm1686', 'cv183x']
        if device.lower in device_list:
            self.chip = device
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
            opt=2,
            dyn=False,
            profile=False,
            refs=None):
    TpuLang.graph.inputs = inputs
    TpuLang.graph.outputs = outputs
    # convert to mlir
    converter = TpuLangConverter(name=name, graph=TpuLang.graph)
    model_transform(name, converter)
    model_inference(model_name=name, inputs=inputs)
    if cmp and refs is not None:
        model_validate(model_name=name, refs=refs)


def model_transform(model_name, converter: TpuLangConverter):
    mlir_file = model_name + '.mlir'
    mlir_origin = model_name + '_origin.mlir'
    converter.generate_mlir(mlir_origin)
    mlir_opt_for_top(mlir_origin, mlir_file)
    print("Mlir file generated:{}".format(mlir_file))


def model_inference(model_name, inputs):
    in_f32_npz = model_name + '_in_f32.npz'
    mlir_file = model_name + '.mlir'
    ref_inputs = dict()
    for tensor in TpuLang.graph.inputs:
        ref_inputs[tensor.name] = tensor.buffer
    np.savez(in_f32_npz, **ref_inputs)
    res_npz = model_name + '_top_outputs.npz'
    # inference of mlir model
    from tools.model_runner import mlir_inference, show_fake_cmd
    show_fake_cmd(in_f32_npz, mlir_file, res_npz)
    f32_outputs = mlir_inference(ref_inputs, mlir_file)
    np.savez(res_npz, **f32_outputs)


def model_validate(model_name, refs: dict):
    ref_outputs = dict()
    for tensor in TpuLang.graph.outputs:
        ref_outputs[tensor.name] = refs[tensor.name]
    ref_npz = model_name + '_ref_outputs.npz'
    np.savez(ref_npz, **ref_outputs)
    res_npz = model_name + '_top_outputs.npz'
    # compare all blobs layer by layers
    f32_blobs_compare(res_npz, ref_npz, '0.99,0.99')


def init(device: str, assist=False, outdir=''):
    TpuLang(device=device)


def deinit():
    TpuLang.graph = None
    TpuLang.device = None


def ArrayAttr(data: list, data_type: str = 'int64'):
    return [data, data_type, False]


# data_type must be in ["float32", "float16", "int64" "int32", "int16". "int8", "uint8", "bool"]
def Attr(data, data_type: str = 'int64'):
    assert data_type.find("int") >= 0 or data_type in ["float32", "float64", "bool"]
    return [data, data_type, True]


'''
  def operand(in_tensor, **params):
    def _shape_inference():
      ...
      return output_shape
    attr = {
      param_name : ArrayAttr(param, param_dtype) or Attr(param, param_dtype)
      ...
    }
    output = Tensor(_shape_inference(), dtype, name)
    TpuLang.insert_op(op_type, inputs=[in_tensor], outputs=[output], params=attr)
    return output
'''


def conv_v2(input: Tensor,
            weight: Tensor,
            bias: Tensor = None,
            stride: List[int] = None,
            dilation: List[int] = None,
            pad: List[int] = None,
            group=1,
            input_zp: Union[int, List[int]] = None,
            weight_zp: Union[int, List[int]] = None,
            out_dtype: str = None,
            out_name: str = None):
    dilation = [1, 1] if dilation is None else dilation
    stride = [1, 1] if stride is None else stride
    pad = [0, 0, 0, 0] if pad is None else pad
    o_dtype = "int32"
    if out_dtype is not None:
        o_dtype = out_dtype
    elif input.dtype == "float32" or input.dtype == "float16":
        o_dtype = input.dtype

    def _shape_inference():
        kh_ext = dilation[0] * (weight.shape[2] - 1) + 1
        kw_ext = dilation[1] * (weight.shape[3] - 1) + 1
        oh = (input.shape[2] + pad[0] + pad[1] - kh_ext) // stride[0] + 1
        ow = (input.shape[3] + pad[2] + pad[3] - kw_ext) // stride[1] + 1
        return [input.shape[0], weight.shape[0], oh, ow]

    attr = {
        "kernel_shape": ArrayAttr(weight.shape[2:]),
        "strides": ArrayAttr(stride),
        "dilations": ArrayAttr(dilation),
        "pads": ArrayAttr(pad),
        "do_relu": Attr(False, "bool"),
        "group": Attr(group)
    }
    input.quantization(zero_point=input_zp)
    weight.quantization(zero_point=weight_zp)
    output = Tensor(_shape_inference(), dtype=o_dtype, name=out_name)
    inputs = [input, weight, bias]
    TpuLang.insert_op(Top.ConvOp, inputs=inputs, outputs=[output], params=attr)
    return output


@deprecated(version=1.0, reason="This function will be removed soon")
def conv(input: Tensor,
         weight: Tensor,
         bias: Tensor = None,
         kernel=None,
         dilation: List[int] = None,
         pad: List[int] = None,
         stride: List[int] = None,
         group: int = 1,
         out_name: str = None):
    return conv_v2(input=input,
                   weight=weight,
                   bias=bias,
                   stride=stride,
                   dilation=dilation,
                   pad=pad,
                   group=group,
                   out_dtype=input.dtype,
                   out_name=out_name)


def requant_fp_to_int(tensor_i: Tensor,
                      scale,
                      offset,
                      requant_mode,
                      out_dtype,
                      out_name=None,
                      round_mode='half_away_from_zero'):
    output = Tensor(tensor_i.shape, name=out_name, dtype=out_dtype)
    output.quantization(scale=scale, zero_point=offset)
    TpuLang.insert_op("top.Cast", inputs=[tensor_i], outputs=[output])
    return output
