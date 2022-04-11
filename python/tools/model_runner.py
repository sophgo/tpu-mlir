#!/usr/bin/env python3
import numpy as np
import argparse
import pymlir
import onnx
import onnxruntime
from transform.TFLiteInterpreter import TFLiteInterpreter


def mlir_inference(inputs: dict, mlir_file: str, dump_all:bool = True) -> dict:
    module = pymlir.module()
    module.load(mlir_file)
    for name in module.input_names:
        assert (name in inputs)
        module.set_tensor(name, inputs[name])
    module.invoke()
    tensors = module.get_all_tensor()
    if dump_all:
       return tensors
    outputs = dict()
    for name in module.output_names:
       outputs[name] = tensors[name]
    return outputs


def generate_onnx_with_all(onnx_file: str):
    # for dump all activations
    # plz refre https://github.com/microsoft/onnxruntime/issues/1455
    output_keys = []
    model = onnx.load(onnx_file)
    no_list = [
        "Cast", "Shape", "Unsqueeze", "Gather", "Split", "Constant", "GRU", "Div", "Sqrt", "Add",
        "ReduceMean", "Pow", "Sub", "Mul", "LSTM"
    ]

    # tested commited #c3cea486d https://github.com/microsoft/onnxruntime.git
    for x in model.graph.node:
        if x.op_type in no_list:
            continue
        _intermediate_tensor_name = list(x.output)
        intermediate_tensor_name = ",".join(_intermediate_tensor_name)
        intermediate_layer_value_info = onnx.helper.ValueInfoProto()
        intermediate_layer_value_info.name = intermediate_tensor_name
        model.graph.output.append(intermediate_layer_value_info)
        output_keys.append(intermediate_layer_value_info.name + '_' + x.op_type)
    dump_all_tensors_onnx = onnx_file.replace('.onnx', '_all.onnx', 1)
    onnx.save(model, dump_all_tensors_onnx)
    return output_keys, dump_all_tensors_onnx


def onnx_inference(inputs: dict, onnx_file: str, dump_all:bool=True) -> dict:
    output_keys = []
    if dump_all:
        output_keys, onnx_file = generate_onnx_with_all(onnx_file)
    session = onnxruntime.InferenceSession(onnx_file)
    inodes = session.get_inputs()
    data = {}
    for node in inodes:
        name = node.name
        dtype = np.float32
        if node.type == 'tensor(int64)':
            dtype = np.int64
        elif node.type == 'tensor(bool)':
            dtype = np.bool
        data[name] = inputs[name].astype(dtype)
    outs = session.run(None, data)
    outputs = dict()
    if not dump_all:
        onodes = session.get_outputs()
        for node, out in zip(onodes, outs):
            outputs[node.name] = out.astype(np.float32)
        return outputs
    else:
        output_num = len(outs) - len(output_keys)
        outs = outs[output_num:]
        return dict(zip(output_keys, map(np.ndarray.flatten, outs)))


def tflite_inference(inputs: dict, tflite_file: str, dump_all: bool = True) -> dict:
    session = TFLiteInterpreter(tflite_file)
    data = {}
    for input in session.inputs:
        name = input["name"]
        data[name] = inputs[name].astype(input["dtype"])
    outputs = session.forward(**data)
    if not dump_all:
        return {k: v.astype(np.float32) for k, v in outputs.items()}
    else:
        return {t["name"]: v for t, v in session.get_all_tensors() if v is not None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input npz file")
    parser.add_argument("--model", type=str, required=True, help="mlir file.")
    parser.add_argument("--output", default='_output.npz', help="output npz file")
    parser.add_argument("--dump_all_tensors",
                        action='store_true',
                        help="dump all tensors to output file")
    args = parser.parse_args()
    data = np.load(args.input)
    output = dict()
    if args.model.endswith('.onnx'):
        output = onnx_inference(data, args.model, args.dump_all_tensors)
    elif args.model.endswith('.mlir'):
        output = mlir_inference(data, args.model, args.dump_all_tensors)
    elif args.model.endswith(".tflite"):
        output = tflite_inference(data, args.model, args.dump_all_tensors)
    else:
        raise RuntimeError("not support modle file:{}".format(args.model))
    np.savez(args.output, **output)
    print("Result saved to:{}".format(args.output))
