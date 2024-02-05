# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from typing import List, Union, Tuple
from .TpuLangConverter import TpuLangConverter, Graph, Tensor, Operator, Scalar, to_scalar, annotayion_check
# from deprecated.sphinx import deprecated
from utils.mlir_shell import *

import numpy as np
import uuid


class TpuLang:
    graph = None
    device = None

    def __init__(
        self,
        device: str,
    ):
        device_list = ['cpu', 'bm1684x', 'bm1688', 'cv183x']
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
            has_custom=False,
            refs=None):
    TpuLang.graph.inputs = inputs
    TpuLang.graph.outputs = outputs
    # convert to mlir
    converter = TpuLangConverter(name=name, graph=TpuLang.graph)
    model_transform(name, converter)
    model_inference(model_name=name, inputs=inputs, has_custom=has_custom)
    if cmp and refs is not None:
        model_validate(model_name=name, refs=refs)


def model_transform(model_name, converter: TpuLangConverter):
    mlir_file = model_name + '.mlir'
    mlir_origin = model_name + '_origin.mlir'
    converter.generate_mlir(mlir_origin)
    mlir_opt_for_top(mlir_origin, mlir_file)
    print("Mlir file generated:{}".format(mlir_file))


def model_inference(model_name, inputs, has_custom=False):
    in_f32_npz = model_name + '_in_f32.npz'
    mlir_file = model_name + '.mlir'
    ref_inputs = dict()
    for tensor in TpuLang.graph.inputs:
        ref_inputs[tensor.name] = tensor.buffer
    np.savez(in_f32_npz, **ref_inputs)
    # inference of mlir model, no inference performed when there is custom op
    if not has_custom:
        res_npz = model_name + '_top_outputs.npz'
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


def broadcast_shape_inference(ops: list):
    assert len(ops) > 0
    op: Tensor = ops[0]
    out_shape = op.shape
    for i in ops[1:]:
        hs_shape = i.shape
        tmp_shape = []
        for idx in range(len(hs_shape) - 1 if len(hs_shape) > len(out_shape) else len(out_shape) - 1, -1, -1):
            try:
                if out_shape[idx] != 1:
                    tmp_shape.append(out_shape[idx])
                else:
                    raise
            except:
                if idx < len(hs_shape):
                    tmp_shape.append(hs_shape[idx])
                else:
                    tmp_shape.append(out_shape[idx])
        out_shape = [i for i in reversed(tmp_shape)]
    return out_shape


# data_type must be in ["float32", "float16", "int64" "int32", "int16". "int8", "uint8", "bool", "string", "dict"]
def Attr(data, data_type: str = 'int64'):
    assert data_type.find("int") >= 0 or data_type in [
        "float32", "float64", "bool", "string", "dict"
    ]
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


def custom(tensors_in: list,
           shape_func,
           op_name: str,
           out_dtypes: list,
           out_names: list = None,
           params: dict = None):
    '''
        The custom op
        Arguments:
            tensors_in: list of input tensors (including weight tensors)
            shape_func: function for doing shape inference, taking tensors_in as the parameter
            op_name: name of the custom operator
            out_dtypes: list of outputs' data type
            out_name: list of output names
            params: parameters of the custom op

        Return:
            tensors_out: list of output tensors
    '''

    out_shapes = shape_func(tensors_in)

    tensors_out = []
    for i, out_dtype in enumerate(out_dtypes):
        tensor_out = Tensor(out_shapes[i],
                            dtype=out_dtype,
                            name=out_names[i] if out_names else None)
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
        # not support string params for now
        # elif isinstance(value, str):
        #     value_new = Attr(value, "string")
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


@annotayion_check
@to_scalar
def add(tensor_i0: Union[Tensor, Scalar, int, float], tensor_i1: Union[Tensor, Scalar, int, float], out_dtype: str = None, out_name: str = None):
    o_dtype = binary_dtype_check(tensor_i0, tensor_i1, out_dtype)
    # shape = broadcast_shape_inference([tensor_i0, tensor_i1])
    output = Tensor([], dtype=o_dtype, name=out_name)
    if isinstance(tensor_i0, Tensor) and isinstance(tensor_i1, Tensor):
        TpuLang.insert_op("top.Add", [tensor_i0, tensor_i1], [output])
    else:
        tensor = tensor_i0 if isinstance(tensor_i0, Tensor) else tensor_i1
        scalar = tensor_i0 if isinstance(tensor_i1, Tensor) else tensor_i1

        if tensor == scalar:
            raise "input muat be have Tensor"
        attr = {
            "const_val": Attr(scalar.value, 'float64'),
        }
        TpuLang.insert_op("top.AddConst", [tensor], [output], params = attr)
    return output

@annotayion_check
def conv_v2(input: Tensor,
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
    o_dtype = "int32"
    if out_dtype is not None:
        o_dtype = out_dtype
    elif input.dtype == "float32" or input.dtype == "float16":
        o_dtype = input.dtype

    assert(input.shape[1] == weight.shape[1] * group and "ic != kic * group")

    def _shape_inference():
        kh_ext = dilation[0] * (weight.shape[2] - 1) + 1
        kw_ext = dilation[1] * (weight.shape[3] - 1) + 1
        oh = (input.shape[2] + pad[0] + pad[2] - kh_ext) // stride[0] + 1
        ow = (input.shape[3] + pad[1] + pad[3] - kw_ext) // stride[1] + 1
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
    TpuLang.insert_op("top.Conv", inputs=inputs, outputs=[output], params=attr)
    return output

@annotayion_check
def conv3d_v2(input: Tensor,
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
    dilation = [1, 1, 1] if dilation is None else dilation
    stride = [1, 1, 1] if stride is None else stride
    pad = [0, 0, 0, 0, 0, 0] if pad is None else pad
    o_dtype = "int32"
    if out_dtype is not None:
        o_dtype = out_dtype
    else:
        o_dtype = input.dtype

    assert(input.shape[1] == weight.shape[1] * group and "ic != kic * group")

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
    output = Tensor([], dtype=o_dtype, name=out_name)
    inputs = [input, weight, bias]
    TpuLang.insert_op("top.Conv", inputs=inputs, outputs=[output], params=attr)
    return output

@annotayion_check
def deconv_v2(input: Tensor,
            weight: Tensor,
            bias: Tensor = None,
            stride: List[int] = None,
            dilation: List[int] = None,
            pad: List[int] = None,
            output_padding: List[int] = None,
            group=1,
            input_zp: Union[int, List[int]] = None,
            weight_zp: Union[int, List[int]] = None,
            out_dtype: str = None,
            out_name: str = None):
    dilation = [1, 1] if dilation is None else dilation
    stride = [1, 1] if stride is None else stride
    pad = [0, 0, 0, 0] if pad is None else pad
    output_padding = [0, 0] if output_padding is None else output_padding

    o_dtype = "int32"
    if out_dtype is not None:
        o_dtype = out_dtype
    else:
        o_dtype = input.dtype

    attr = {
        "kernel_shape": ArrayAttr(weight.shape[2:]),
        "strides": ArrayAttr(stride),
        "dilations": ArrayAttr(dilation),
        "pads": ArrayAttr(pad),
        "output_padding": ArrayAttr(output_padding),
        "do_relu": Attr(False, "bool"),
        "group": Attr(group)
    }
    input.quantization(zero_point=input_zp)
    weight.quantization(zero_point=weight_zp)
    output = Tensor([], dtype=o_dtype, name=out_name)
    inputs = [input, weight, bias]
    TpuLang.insert_op("top.Deconv", inputs=inputs, outputs=[output], params=attr)
    return output

@annotayion_check
def deconv3d_v2(input: Tensor,
                weight: Tensor,
                bias: Tensor = None,
                stride: List[int] = None,
                dilation: List[int] = None,
                pad: List[int] = None,
                output_padding: List[int] = None,
                group=1,
                input_zp: Union[int, List[int]] = None,
                weight_zp: Union[int, List[int]] = None,
                out_dtype: str = None,
                out_name: str = None):
    dilation = [1, 1, 1] if dilation is None else dilation
    stride = [1, 1, 1] if stride is None else stride
    pad = [0, 0, 0, 0, 0, 0] if pad is None else pad
    output_padding = [0, 0, 0] if output_padding is None else output_padding
    o_dtype = "int32"
    if out_dtype is not None:
        o_dtype = out_dtype
    else:
        o_dtype = input.dtype

    attr = {
        "kernel_shape": ArrayAttr(weight.shape[2:]),
        "strides": ArrayAttr(stride),
        "dilations": ArrayAttr(dilation),
        "pads": ArrayAttr(pad),
        "output_padding": ArrayAttr(output_padding),
        "do_relu": Attr(False, "bool"),
        "group": Attr(group)
    }
    input.quantization(zero_point=input_zp)
    weight.quantization(zero_point=weight_zp)
    output = Tensor([], dtype=o_dtype, name=out_name)
    inputs = [input, weight, bias]
    TpuLang.insert_op("top.Deconv", inputs=inputs, outputs=[output], params=attr)
    return output

@annotayion_check
def mul(tensor_i0: Tensor, tensor_i1: Tensor, out_dtype: str = None, out_name: str = None):
    o_dtype = binary_dtype_check(tensor_i0, tensor_i1, out_dtype, True)
    # if tensor_i0.dtype == "float32" or tensor_i0.dtype == "float16" or out_dtype is None:
    #     o_dtype = tensor_i0.dtype
    # elif out_dtype is not None:
    #     o_dtype = out_dtype
    # shape = broadcast_shape_inference([tensor_i0, tensor_i1])
    output = Tensor([], dtype=o_dtype, name=out_name)
    TpuLang.insert_op("top.Mul", [tensor_i0, tensor_i1], [output])
    return output

@annotayion_check
def sub(tensor_i0: Tensor, tensor_i1: Tensor, out_dtype: str = None, out_name: str = None):
    o_dtype = binary_dtype_check(tensor_i0, tensor_i1, out_dtype, True)
    # if tensor_i0.dtype == "float32" or tensor_i0.dtype == "float16" or out_dtype is None:
    #     o_dtype = tensor_i0.dtype
    # elif out_dtype is not None:
    #     o_dtype = out_dtype
    # shape = broadcast_shape_inference([tensor_i0, tensor_i1])
    output = Tensor([], dtype=o_dtype, name=out_name)
    TpuLang.insert_op("top.Sub", [tensor_i0, tensor_i1], [output])
    return output

@annotayion_check
def div(tensor_i0: Tensor, tensor_i1: Tensor, out_name: str = None):
    o_dtype = same_dtype_check(tensor_i0.dtype, tensor_i1.dtype, "float32")
    # shape = broadcast_shape_inference([tensor_i0, tensor_i1])
    output = Tensor([], dtype=o_dtype, name=out_name)
    TpuLang.insert_op("top.Div", [tensor_i0, tensor_i1], [output])
    return output

@annotayion_check
def max(tensor_i0: Tensor, tensor_i1: Tensor, out_name: str = None):
    o_dtype = same_dtype_check(tensor_i0.dtype, tensor_i1.dtype)
    # if tensor_i0.dtype == "float32" or tensor_i0.dtype == "float16" or out_dtype is None:
    #     o_dtype = tensor_i0.dtype
    # elif out_dtype is not None:
    #     o_dtype = out_dtype
    # shape = broadcast_shape_inference([tensor_i0, tensor_i1])
    output = Tensor([], dtype=o_dtype, name=out_name)
    TpuLang.insert_op("top.Max", [tensor_i0, tensor_i1], [output])
    return output

@annotayion_check
def min(tensor_i0: Tensor, tensor_i1: Tensor, out_name: str = None):
    o_dtype = same_dtype_check(tensor_i0.dtype, tensor_i1.dtype)
    # if tensor_i0.dtype == "float32" or tensor_i0.dtype == "float16" or out_dtype is None:
    #     o_dtype = tensor_i0.dtype
    # elif out_dtype is not None:
    #     o_dtype = out_dtype
    # shape = broadcast_shape_inference([tensor_i0, tensor_i1])
    output = Tensor([], dtype=o_dtype, name=out_name)
    TpuLang.insert_op("top.Min", [tensor_i0, tensor_i1], [output])
    return output

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

#     shape = broadcast_shape_inference([tensor_i0, tensor_i1])

#     output = Tensor(shape, dtype=o_dtype, name=out_name)

#     TpuLang.insert_op("top.Min", [tensor_i0, tensor_i1], [output])

#     return output
def generate_name(op):
    unique_name = str(uuid.uuid4())
    return f"{op}_{unique_name}"

@annotayion_check
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

# def cast(tensor_i: Tensor,
#          out_dtype: str = 'float32',
#          out_name: str = None,
#          round_mode: str = 'half_away_from_zero'):
#     shape = tensor_i.shape
#     if out_name is None:
#         out_name = generate_name("cast")
#     output = Tensor(shape, dtype=out_dtype, name=out_name)
#     TpuLang.insert_op("top.Cast", [tensor_i], [output])

#     return output

@annotayion_check
def clamp(input: Tensor, min:float, max:float, out_name: str = None):
    if out_name is None:
        out_name = generate_name("clamp")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    attr = {
        "min": Attr(min, data_type="float64"),
        "max": Attr(max, data_type="float64"),
    }
    TpuLang.insert_op("top.Clip", [input], [output], params=attr)
    return output

# @deprecated(version=1.0, reason="This function will be removed soon")
@annotayion_check
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

# @deprecated(version=1.0, reason="Th  is function will be removed soon")
@annotayion_check
def conv3d(input: Tensor,
         weight: Tensor,
         bias: Tensor = None,
         kernel=None,
         dilation: List[int] = None,
         pad: List[int] = None,
         stride: List[int] = None,
         group: int = 1,
         out_name: str = None):
    return conv3d_v2(input=input,
                   weight=weight,
                   bias=bias,
                   stride=stride,
                   dilation=dilation,
                   pad=pad,
                   group=group,
                   out_dtype=input.dtype,
                   out_name=out_name)

# @deprecated(version=1.0, reason="This function will be removed soon")
@annotayion_check
def deconv(input: Tensor,
           weight: Tensor,
           bias: Tensor = None,
           kernel=None,
           dilation: List[int] = None,
           pad: List[int] = None,
           output_padding: List[int] = None,
           stride: List[int] = None,
           group: int = 1,
           out_name: str = None):
    return deconv_v2(input=input,
                     weight=weight,
                     bias=bias,
                     stride=stride,
                     dilation=dilation,
                     pad=pad,
                     output_padding=output_padding,
                     group=group,
                     out_dtype=input.dtype,
                     out_name=out_name)

@annotayion_check
def requant_fp_to_int(tensor_i: Tensor,
                      scale: Union[float, List[float]],
                      offset: Union[int, List[int], float, List[float]],
                      requant_mode: int,
                      out_dtype: str,
                      out_name=None,
                      round_mode='half_away_from_zero'):
    output = Tensor(tensor_i.shape, name=out_name, dtype=out_dtype)
    output.quantization(scale=scale, zero_point=offset)
    TpuLang.insert_op("top.Cast", inputs=[tensor_i], outputs=[output])
    return output

@annotayion_check
def matmul(input: Tensor,
            right: Tensor,
            bias: Tensor = None,
            right_transpose=False,
            left_transpose=False,
            output_transpose=False,
            hdim_is_batch=False,
            keep_dims=False,
            out_name: str = None):

    o_dtype = input.dtype

    def _shape_inference():
        l_dims = len(input.shape)
        r_dims =len(right.shape)
        k = input.shape[l_dims - 1]
        k_idx = r_dims - (1 if right_transpose else 2)
        n_idx = r_dims - (2 if right_transpose else 1)
        n = right.shape[n_idx]
        import copy
        out_shape = copy.copy(input.shape)
        if r_dims == 1:
            assert(right.shape[0] == k)
            out_shape.pop()
        elif right.shape[k_idx] == k:
            out_shape[-1] = n
        elif r_dims == 2:
            sum = right.shape[k_idx]
            while len(out_shape) > 0 and sum % out_shape[-1] == 0 and sum != 1:
                sum = sum // out_shape.pop()
            if sum != 1:
                raise ValueError("shape is illegal")
            out_shape.append(n)
        else:
            out_shape[-1] = n
        if not keep_dims:
            batch_size = 1
            for s in out_shape[:-1]:
                batch_size *= s
            out_shape = [batch_size, n]
        return out_shape



    attr = {
        "right_transpose": Attr(right_transpose, "bool"),
        "left_transpose": Attr(left_transpose, "bool"),
        "output_transpose": Attr(output_transpose, "bool"),
        "hdim_is_batch": Attr(hdim_is_batch, "bool"),
        "keep_dims": Attr(keep_dims, "bool"),
        "do_relu":  Attr(False, "bool"),
        "relu_limit":  Attr(-1.0, "float64")
    }

    output = Tensor(_shape_inference(), dtype=o_dtype, name=out_name)
    inputs = [input, right, bias]
    TpuLang.insert_op("top.MatMul", inputs=inputs, outputs=[output], params=attr)
    return output

@annotayion_check
def maxpool(input: Tensor,
            kernel: List[int]=None,
            stride: List[int] = None,
            pad: List[int] = None,
            ceil_mode: bool = False,
            out_name: str = None):
    kernel = [1, 1] if kernel is None else kernel
    stride = [1, 1] if stride is None else stride
    pad = [0, 0, 0, 0] if pad is None else pad
    o_dtype = input.dtype

    def _shape_inference():
        assert len(input.shape) > 2
        spacial_rank = len(input.shape) - 2
        assert spacial_rank == len(kernel)
        assert len(pad) == spacial_rank * 2
        out_shape = [input.shape[0], input.shape[1]]
        input_spacial_shape = input.shape[2:]
        for i in range(spacial_rank):
            input_dim_expanded = input_spacial_shape[i] + pad[i] + pad[i + spacial_rank] - kernel[i]
            out_dim = (input_dim_expanded // stride[i]) + 1
            # move ceil_mode to padding
            need_fix_pad = input_dim_expanded % stride[i]
            if ceil_mode and ceil_mode.value and need_fix_pad:
                new_pad = pad[i + spacial_rank] + stride[i] - need_fix_pad
                if new_pad < kernel[i]:
                    pad[i + spacial_rank] = new_pad
                    out_dim += 1
            out_shape.append(out_dim)

        return out_shape

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

    output = Tensor(_shape_inference(), dtype=o_dtype, name=out_name)

    TpuLang.insert_op("top.MaxPool", inputs=[input], outputs=[output], params=attr)
    return output

@annotayion_check
def relu(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("relu")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Relu", inputs=[input], outputs=[output])
    return output

@annotayion_check
def leaky_relu(input: Tensor, negative_slope: float = 0.01, out_name: str = None):
    if out_name is None:
        out_name = generate_name("leaky_relu")
    attr = {
        "alpha": Attr(negative_slope, data_type="float64"),
    }
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.LeakyRelu", inputs=[input], outputs=[output], params=attr)
    return output

@annotayion_check
def abs(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("abs")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Abs", inputs=[input], outputs=[output])
    return output

@annotayion_check
def ceil(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("ceil")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Ceil", inputs=[input], outputs=[output])
    return output

@annotayion_check
def floor(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("floor")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Floor", inputs=[input], outputs=[output])
    return output

@annotayion_check
def round(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("round")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Round", inputs=[input], outputs=[output])
    return output

@annotayion_check
def sin(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("sin")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Sin", inputs=[input], outputs=[output])
    return output

@annotayion_check
def cos(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("cos")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Cos", inputs=[input], outputs=[output])
    return output

@annotayion_check
def exp(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("exp")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Exp", inputs=[input], outputs=[output])
    return output

@annotayion_check
def tanh(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("tanh")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Tanh", inputs=[input], outputs=[output])
    return output

@annotayion_check
def sigmoid(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("sigmoid")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Sigmoid", inputs=[input], outputs=[output])
    return output

@annotayion_check
def elu(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("elu")
    attr = {
        "alpha": Attr(1.0, "float64"),
    }
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Elu", inputs=[input], outputs=[output], params=attr)
    return output

@annotayion_check
def sqrt(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("sqrt")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Sqrt", inputs=[input], outputs=[output])
    return output

# def rsqrt(input: Tensor, out_name: str = None):
#     if out_name is None:
#         out_name = generate_name("rsqrt")
#     output = Tensor(input.shape, dtype=input.dtype, name=out_name)
#     TpuLang.insert_op("top.Rsqrt", inputs=[input], outputs=[output])
#     return output

@annotayion_check
def erf(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("erf")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Erf", inputs=[input], outputs=[output])
    return output

@annotayion_check
def tan(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("tan")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Tan", inputs=[input], outputs=[output])
    return output

@annotayion_check
def softmax(input: Tensor, axis: int, out_name: str = None):
    if out_name is None:
        out_name = generate_name("softmax")
    attr = {
        "axis": Attr(axis, data_type="int32"),
    }
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Softmax", inputs=[input], outputs=[output], params=attr)
    return output

@annotayion_check
def mish(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("mish")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Mish", inputs=[input], outputs=[output])
    return output

@annotayion_check
def hswish(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("hswish")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.HardSwish", inputs=[input], outputs=[output])
    return output

@annotayion_check
def arccos(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("arccos")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Arccos", inputs=[input], outputs=[output])
    return output

@annotayion_check
def arctanh(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("arctanh")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Arctanh", inputs=[input], outputs=[output])
    return output

@annotayion_check
def sinh(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("sinh")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Sinh", inputs=[input], outputs=[output])
    return output

@annotayion_check
def cosh(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("cosh")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Cosh", inputs=[input], outputs=[output])
    return output

@annotayion_check
def sign(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("sign")
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Sign", inputs=[input], outputs=[output])
    return output

@annotayion_check
def gelu(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("gelu")
    output = Tensor([], dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.GELU", inputs=[input], outputs=[output])
    return output

@annotayion_check
def hsigmoid(input: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("hsigmoid")
    attr = {
        "alpha": Attr(1/6, data_type="float64"),
        "beta": Attr(0.5, data_type="float64"),
    }
    output = Tensor(input.shape, dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.HardSigmoid", inputs=[input], outputs=[output], params=attr)
    return output

@annotayion_check
def arg(input: Tensor,
        method: str = "max",
        axis: int = 0,
        keep_dims: bool = True,
        out_name: str = None):
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
    output1 = Tensor([], dtype=o_dtype, name=out_name)
    output2 = Tensor([], dtype=o_dtype, name=out_name)
    TpuLang.insert_op("top.Arg", inputs=[input], outputs=[output1, output2], params=attr)
    return output1, output2

@annotayion_check
def permute(input: Tensor, order: Union[Tuple[int], List[int]], out_name: str = None):
    if out_name is None:
        out_name = generate_name("permute")
    attr = {
        "order": ArrayAttr(order),
    }
    output = Tensor([], dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Permute", inputs=[input], outputs=[output], params=attr)
    return output

@annotayion_check
def tile(input: Tensor, reps: Union[Tuple[int], List[int]], out_name: str = None):
    if out_name is None:
        out_name = generate_name("tile")
    attr = {
        "tile": ArrayAttr(reps),
    }
    output = Tensor([], dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Tile", inputs=[input], outputs=[output], params=attr)
    return output

@annotayion_check
def concat(inputs: List[Tensor], axis: int = 0, out_name: str = None):
    assert(len(inputs) > 1 and "concat should have more than one input")
    if out_name is None:
        out_name = generate_name("concat")
    attr = {
        "axis": Attr(axis, "int32"),
    }
    output = Tensor([], dtype=inputs[0].dtype, name=out_name)
    TpuLang.insert_op("top.Concat", inputs=inputs, outputs=[output], params=attr)
    return output

@annotayion_check
def split(input: Tensor,
          axis: int = 0,
          num: int = 1,
          size: Union[Tuple[int], List[int]] = (),
          out_name: str = None) -> List[Tensor]:
    assert(num > 1 and "number of split output should be more than 1")
    if not size:
        assert(input.shape[axis] % num == 0 and "invalid split size")
        size = [int(input.shape[axis] / num)] * num
    else:
        assert(num == len(size) and "size should be the same as num")
        assert(sum(size) == input.shape[axis] and "invalid size")

    if out_name is None:
        out_name = generate_name("split")

    attr = {
        "axis": Attr(axis, "int32"),
        "num": Attr(num),
        "split_size": ArrayAttr(size),
    }
    outputs = []
    for i in range(num):
        outputs.append(Tensor([], dtype=input.dtype, name=f"{out_name}_{i}"))

    TpuLang.insert_op("top.Split", inputs=[input], outputs=outputs, params=attr)
    return outputs

# def pad(input: Tensor,
#         axis: int = 0,
#         value:
#         padding: Union[Tuple[int], List[int]] = None,
#         out_name: str = None):
#     if out_name is None:
#         out_name = generate_name("split")
#     attr = {
#         "axis": Attr(axis, "int32"),
#         "num": Attr(num),
#         "split_size": ArrayAttr(size),
#     }
#     output = Tensor([], dtype=input.dtype, name=out_name)
#     TpuLang.insert_op("top.Split", inputs=[input], outputs=[output], params=attr)
#     return output

@annotayion_check
def repeat(input: Tensor, reps: Tensor, out_name: str = None):
    if out_name is None:
        out_name = generate_name("repeat")
    # reps = Tensor(data = reps, shape = input.shape)
    output = Tensor([], dtype=input.dtype, name=out_name)
    TpuLang.insert_op("top.Repeat", inputs=[input, reps], outputs=[output])
    return output

# def nms(boxes: Tensor,
#         score: int = 0,
#         num: int = 1,
#         size: Union[Tuple[int], List[int]] = (),
#         out_name: str = None):
#     if out_name is None:
#         out_name = generate_name("split")
#     attr = {
#         "axis": Attr(axis, "int32"),
#         "num": Attr(num),
#         "split_size": ArrayAttr(size),
#     }
#     output = Tensor([], dtype=input.dtype, name=out_name)
#     TpuLang.insert_op("top.Split", inputs=[input], outputs=[output], params=attr)
#     return output

# def interpolate(tensor_i: Tensor,
#                 size: Union[Tuple[int], List[int]],
#                 scale_factor: Union[Tuple[float], List[float]] = (),
#                 platform: str = 'CAFFE',
#                 method: str = 'NEAREST',
#                 align_corners: bool = False,
#                 half_pixel_centers: bool = False,
#                 out_name: str = None):
#     if out_name is None:
#         out_name = generate_name("interpolate")
#     target_shape = tensor_i.shape
#     if size:
#         target_shape[2:] = size[:]
#     if align_corners:
#         coord = "align_corners"
#     elif half_pixel_centers:
#         if platform == 'PYTORCH':
#             coord = "pytorch_half_pixel"
#         else:
#             coord = "half_pixel"
#     else:
#         coord = "asymmetric"
#     attr = {
#         # "target_shape": ArrayAttr(target_shape),
#         "mode": Attr(method, 'string'),
#         "coord_mode": Attr(coord, 'string'),
#     }
#     output = Tensor([], dtype=tensor_i.dtype, name=out_name)
#     TpuLang.insert_op("top.Interp", inputs=[tensor_i, target_shape], outputs=[output], params=attr)
#     return output

def __compare(tensor_i0: Tensor, tensor_i1: Tensor, type: str, out_name: str = None):
    o_dtype = same_dtype_check(tensor_i0.dtype, tensor_i1.dtype)
    if out_name is None:
        out_name = generate_name(type)
    attr = {
        "mode": Attr(type, "string"),
    }
    output = Tensor([], dtype=o_dtype, name=out_name)
    TpuLang.insert_op("top.Compare", inputs=[tensor_i0, tensor_i1], outputs=[output], params=attr)
    return output

@annotayion_check
def gt(tensor_i0: Tensor, tensor_i1: Tensor, out_name: str = None):
    # o_dtype = same_dtype_check(tensor_i0.dtype, tensor_i1.dtype)
    # if out_name is None:
    #     out_name = generate_name("gt")
    # attr = {
    #     "mode": Attr("Greater", "string"),
    # }
    # output = Tensor([], dtype=o_dtype, name=out_name)
    # TpuLang.insert_op("top.Compare", inputs=[tensor_i0, tensor_i1], outputs=[output], params=attr)
    return __compare(tensor_i0, tensor_i1, "Greater", out_name)

@annotayion_check
def lt(tensor_i0: Tensor, tensor_i1: Tensor, out_name: str = None):
    # o_dtype = same_dtype_check(tensor_i0.dtype, tensor_i1.dtype)
    # if out_name is None:
    #     out_name = generate_name("lt")
    # attr = {
    #     "mode": Attr("Less", "string"),
    # }
    # output = Tensor([], dtype=o_dtype, name=out_name)
    # TpuLang.insert_op("top.Compare", inputs=[tensor_i0, tensor_i1], outputs=[output], params=attr)
    # return output
    return __compare(tensor_i0, tensor_i1, "Less", out_name)

@annotayion_check
def ge(tensor_i0: Tensor, tensor_i1: Tensor, out_name: str = None):
    # o_dtype = same_dtype_check(tensor_i0.dtype, tensor_i1.dtype)
    # if out_name is None:
    #     out_name = generate_name("ge")
    # attr = {
    #     "mode": Attr("GreaterOrEqual", "string"),
    # }
    # output = Tensor([], dtype=o_dtype, name=out_name)
    # TpuLang.insert_op("top.Compare", inputs=[tensor_i0, tensor_i1], outputs=[output], params=attr)
    # return output
    return __compare(tensor_i0, tensor_i1, "GreaterOrEqual", out_name)

@annotayion_check
def le(tensor_i0: Tensor, tensor_i1: Tensor, out_name: str = None):
    # o_dtype = same_dtype_check(tensor_i0.dtype, tensor_i1.dtype)
    # if out_name is None:
    #     out_name = generate_name("le")
    # attr = {
    #     "mode": Attr("LessOrEqual", "string"),
    # }
    # output = Tensor([], dtype=tensor_i0.dtype, name=out_name)
    # TpuLang.insert_op("top.Compare", inputs=[tensor_i0, tensor_i1], outputs=[output], params=attr)
    # return output
    return __compare(tensor_i0, tensor_i1, "LessOrEqual", out_name)

@annotayion_check
def eq(tensor_i0: Tensor, tensor_i1: Tensor, out_name: str = None):
    # o_dtype = same_dtype_check(tensor_i0.dtype, tensor_i1.dtype)
    # if out_name is None:
    #     out_name = generate_name("eq")
    # attr = {
    #     "mode": Attr("Equal", "string"),
    # }
    # output = Tensor([], dtype=tensor_i0.dtype, name=out_name)
    # TpuLang.insert_op("top.Compare", inputs=[tensor_i0, tensor_i1], outputs=[output], params=attr)
    # return output
    return __compare(tensor_i0, tensor_i1, "Equal", out_name)

@annotayion_check
def ne(tensor_i0: Tensor, tensor_i1: Tensor, out_name: str = None):
    # o_dtype = same_dtype_check(tensor_i0.dtype, tensor_i1.dtype)
    # if out_name is None:
    #     out_name = generate_name("ne")
    # attr = {
    #     "mode": Attr("NotEqual", "string"),
    # }
    # output = Tensor([], dtype=tensor_i0.dtype, name=out_name)
    # TpuLang.insert_op("top.Compare", inputs=[tensor_i0, tensor_i1], outputs=[output], params=attr)
    # return output
    return __compare(tensor_i0, tensor_i1, "NotEqual", out_name)

@to_scalar
def __compare_const(tensor_i0: Tensor, scalar_i1: Scalar, type: str, out_name: str = None):
    if out_name is None:
        out_name = generate_name(type)
    attr = {
        "mode": Attr(type, "string"),
        "const_val": Attr(scalar_i1.value, "float64"),
        'inversed': Attr(False, "bool")
    }
    output = Tensor([], dtype=tensor_i0.dtype, name=out_name)
    TpuLang.insert_op("top.CompareConst", inputs=[tensor_i0], outputs=[output], params=attr)
    return output

@annotayion_check
def gts(tensor_i0: Tensor, scalar_i1: Union[Scalar, int, float], out_name: str = None):
    # if out_name is None:
    #     out_name = generate_name("gts")
    # attr = {
    #     "mode": Attr("Greater", "string"),
    #     "const_val": Attr(scalar_i1, "float64"),
    #     'inversed': Attr(False, "bool")
    # }
    # output = Tensor([], dtype=tensor_i0.dtype, name=out_name)
    # TpuLang.insert_op("top.CompareConst", inputs=[tensor_i0], outputs=[output], params=attr)
    # return output
    return __compare_const(tensor_i0, scalar_i1, "Greater", out_name)

@annotayion_check
def lts(tensor_i0: Tensor, scalar_i1: Union[Scalar, int, float], out_name: str = None):
    # if out_name is None:
    #     out_name = generate_name("lts")
    # attr = {
    #     "mode": Attr("Less", "string"),
    #     "const_val": Attr(scalar_i1, "float64"),
    #     'inversed': Attr(False, "bool")
    # }
    # output = Tensor([], dtype=tensor_i0.dtype, name=out_name)
    # TpuLang.insert_op("top.CompareConst", inputs=[tensor_i0], outputs=[output], params=attr)
    # return output
    return __compare_const(tensor_i0, scalar_i1, "Less", out_name)

@annotayion_check
def ges(tensor_i0: Tensor, scalar_i1: Union[Scalar, int, float], out_name: str = None):
    # if out_name is None:
    #     out_name = generate_name("ges")
    # attr = {
    #     "mode": Attr("GreaterOrEqual", "string"),
    #     "const_val": Attr(scalar_i1, "float64"),
    #     'inversed': Attr(False, "bool")
    # }
    # output = Tensor([], dtype=tensor_i0.dtype, name=out_name)
    # TpuLang.insert_op("top.CompareConst", inputs=[tensor_i0], outputs=[output], params=attr)
    # return output
    return __compare_const(tensor_i0, scalar_i1, "GreaterOrEqual", out_name)

@annotayion_check
def les(tensor_i0: Tensor, scalar_i1: Union[Scalar, int, float], out_name: str = None):
    # if out_name is None:
    #     out_name = generate_name("les")
    # attr = {
    #     "mode": Attr("LessOrEqual", "string"),
    #     "const_val": Attr(scalar_i1, "float64"),
    #     'inversed': Attr(False, "bool")
    # }
    # output = Tensor([], dtype=tensor_i0.dtype, name=out_name)
    # TpuLang.insert_op("top.CompareConst", inputs=[tensor_i0], outputs=[output], params=attr)
    # return output
    return __compare_const(tensor_i0, scalar_i1, "LessOrEqual", out_name)

@annotayion_check
def eqs(tensor_i0: Tensor, scalar_i1: Union[Scalar, int, float], out_name: str = None):
    # if out_name is None:
    #     out_name = generate_name("eqs")
    # attr = {
    #     "mode": Attr("Equal", "string"),
    #     "const_val": Attr(scalar_i1, "float64"),
    #     'inversed': Attr(False, "bool")
    # }
    # output = Tensor([], dtype=tensor_i0.dtype, name=out_name)
    # TpuLang.insert_op("top.CompareConst", inputs=[tensor_i0], outputs=[output], params=attr)
    # return output
    return __compare_const(tensor_i0, scalar_i1, "Equal", out_name)

@annotayion_check
def nes(tensor_i0: Tensor, scalar_i1: Union[Scalar, int, float], out_name: str = None):
    # if out_name is None:
    #     out_name = generate_name("nes")
    # attr = {
    #     "mode": Attr("NotEqual", "string"),
    #     "const_val": Attr(scalar_i1, "float64"),
    #     'inversed': Attr(False, "bool")
    # }
    # output = Tensor([], dtype=tensor_i0.dtype, name=out_name)
    # TpuLang.insert_op("top.CompareConst", inputs=[tensor_i0], outputs=[output], params=attr)
    # return output
    return __compare_const(tensor_i0, scalar_i1, "NotEqual", out_name)

