#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import argparse
import onnx
from onnx import helper
from onnx import TensorProto
from utils.mlir_parser import MlirParser
import numpy as np
from mlir.dialects import quant
import mlir


def type_map(element_type):
    signed_flag = True
    ret_type = TensorProto.FLOAT
    if quant.UniformQuantizedType.isinstance(element_type):
        quant_type = quant.UniformQuantizedType(element_type)
        storagetype = quant_type.storage_type
        signed_flag = True if quant_type.is_signed else False
        element_type = storagetype
    if quant.CalibratedQuantizedType.isinstance(element_type):
        quant_type = quant.CalibratedQuantizedType(element_type)
        expressed_type = quant_type.expressed_type
        signed_flag = True if quant_type.is_signed else False
        element_type = expressed_type
    if quant.UniformQuantizedPerAxisType.isinstance(element_type):
        quant_type = quant.UniformQuantizedPerAxisType(element_type)
        storagetype = quant_type.storage_type
        signed_flag = True if quant_type.is_signed else False
        element_type = storagetype
    str_element_type = str(element_type)
    if str_element_type == 'f64':
        ret_type = TensorProto.FLOAT64
    elif str_element_type == 'f32':
        ret_type = TensorProto.FLOAT
    elif str_element_type == 'f16':
        ret_type = TensorProto.FLOAT16
    elif str_element_type == 'bf16':
        ret_type = TensorProto.BFLOAT16
    elif str_element_type == 'si8':
        ret_type = TensorProto.INT8
    elif str_element_type == 'si4':
        ret_type = TensorProto.INT8
    elif (str_element_type == 'si16'):
        ret_type = TensorProto.INT16
    elif (str_element_type == 'si32'):
        ret_type = TensorProto.INT32
    elif (str_element_type == 'si64'):
        ret_type = TensorProto.INT64
    elif str_element_type == 'i8':
        ret_type = TensorProto.INT8 if signed_flag else TensorProto.UINT8
    elif str_element_type == 'i4':
        ret_type = TensorProto.INT8 if signed_flag else TensorProto.UINT8
    elif str_element_type == 'i16':
        ret_type = TensorProto.INT16 if signed_flag else TensorProto.UINT16
    elif str_element_type == 'i32':
        ret_type = TensorProto.INT32 if signed_flag else TensorProto.UINT32
    elif str_element_type == 'i64':
        ret_type = TensorProto.INT64 if signed_flag else TensorProto.UINT64
    elif str_element_type == 'ui8':
        ret_type = TensorProto.UINT8
    elif str_element_type == 'ui4':
        ret_type = TensorProto.UINT8
    elif str_element_type == 'ui16':
        ret_type = TensorProto.UINT16
    elif str_element_type == 'ui32':
        ret_type = TensorProto.UINT32
    elif str_element_type == 'ui64':
        ret_type = TensorProto.UINT64
    elif str_element_type == 'f8E4M3FN':
        ret_type = TensorProto.FLOAT8E4M3FN
    elif str_element_type == 'f8E5M2':
        ret_type = TensorProto.FLOAT8E5M2
    else:
        raise TypeError(
            "{} type in mlir file is not supported by mlir2onnx.py".format(str_element_type))
    return ret_type


def create_input_tvis(parser):
    ops = parser.ops
    tvis = []
    for op in ops:
        if op.type == 'top.Input':
            shape_type = mlir.ir.ShapedType(op.op.results[0].type)
            mlir_type = shape_type.element_type
            tvi = helper.make_tensor_value_info(op.name, type_map(mlir_type), op.shape)
            tvis.append(tvi)
    return tvis


def create_output_tvis(parser):
    ops = parser.ops
    outputs = parser.get_output_op_names_n_shapes()
    tvis = []
    for op in ops:
        for i, name in enumerate(op.outputs):
            if name in outputs:
                shape_type = mlir.ir.ShapedType(op.op.results[i].type)
                mlir_type = shape_type.element_type
                tvi = helper.make_tensor_value_info(name, type_map(mlir_type), outputs[name])
                tvis.append(tvi)
    return tvis


def create_middle_tvis(parser):
    midldle_ops = parser.get_middle_op_names_n_shape_type()
    tvis = []
    for op_name in midldle_ops:
        if str(midldle_ops[op_name]) == "none":
            continue
        mlir_type = midldle_ops[op_name].element_type
        shape = [midldle_ops[op_name].get_dim_size(i) for i in range(midldle_ops[op_name].rank)]
        tvi = helper.make_tensor_value_info(op_name, type_map(mlir_type), shape)
        tvis.append(tvi)
    return tvis


def create_initializer_tensors(parser, weight_file):
    tensors = []
    if weight_file == None:
        return tensors
    initializer_ops = parser.get_initializer_op_names_n_shape_type()
    npzfile = np.load(weight_file)
    for op_name in initializer_ops:
        if op_name in npzfile.files:
            mlir_type = initializer_ops[op_name].element_type
            shape = [
                initializer_ops[op_name].get_dim_size(i)
                for i in range(initializer_ops[op_name].rank)
            ]
            weight_data = npzfile[op_name]
            tensor = helper.make_tensor(op_name, type_map(mlir_type), shape, weight_data)
            tensors.append(tensor)
        else:
            raise ValueError("No {} in {} weight file".format(op_name, weight_file))
    return tensors


def to_onnx(parser, onnx_file, weigth_file):
    inputs = create_input_tvis(parser)
    outputs = create_output_tvis(parser)
    others = []
    initializer = []
    others = create_middle_tvis(parser)
    initializer = create_initializer_tensors(parser, weigth_file)

    nodes = []
    for op in parser.ops:
        if op.type == "top.Input":
            continue
        if 'shape' in op.attrs:
            node = helper.make_node(op.type, op.opds, op.outputs, **op.attrs)
        else:
            node = helper.make_node(op.type, op.opds, op.outputs, shape=op.shape, **op.attrs)
        nodes.append(node)

    graph_def = helper.make_graph(nodes,
                                  'mlir',
                                  inputs,
                                  outputs,
                                  initializer=initializer,
                                  value_info=others)
    model_def = helper.make_model(graph_def, producer_name="mlir")
    onnx.save(model_def, onnx_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--mlir", required=True, help="final mlir file to codegen")
    parser.add_argument('-w', "--weightfile", help="weight file")
    parser.add_argument('-o', "--output", required=True, help="output rebuilt pmu csv file, *.pb")
    args = parser.parse_args()

    mlir_parser = MlirParser(args.mlir)
    to_onnx(mlir_parser, args.output, args.weightfile)
