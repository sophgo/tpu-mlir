#!/usr/bin/python3
import argparse
from onnx import onnx
from onnx import helper
from onnx import TensorProto
from utils.mlir_parser import MlirParser


def create_input_tvis(parser):
    ops = parser.ops
    tvis = []
    for op in ops:
        if op.type == 'top.Input':
            tvi = helper.make_tensor_value_info(
                op.name, TensorProto.FLOAT, op.shape)
            tvis.append(tvi)
    return tvis


def create_output_tvis(parser):
    ops = parser.ops
    outputs = parser.get_output_op_names_n_shapes()
    tvis = []
    for op in ops:
        if op.name in outputs:
            tvi = helper.make_tensor_value_info(
                op.name, TensorProto.FLOAT, outputs[op.name])
            tvis.append(tvi)
    return tvis


def to_onnx(parser, onnx_file):
    inputs = create_input_tvis(parser)
    outputs = create_output_tvis(parser)

    nodes = []
    for op in parser.ops:
        if op.type == "top.Input":
            continue
        node = helper.make_node(
            op.type, op.opds, [op.name], shape=op.shape, **op.attrs)
        nodes.append(node)

    graph_def = helper.make_graph(nodes, 'mlir', inputs, outputs)
    model_def = helper.make_model(graph_def, producer_name="mlir")
    onnx.save(model_def, onnx_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--mlir", required=True,
                        help="final mlir file to codegen")
    parser.add_argument('-o', "--output", required=True,
                        help="output rebuilt pmu csv file, *.pb")
    args = parser.parse_args()

    mlir_parser = MlirParser(args.mlir)
    to_onnx(mlir_parser, args.output)
