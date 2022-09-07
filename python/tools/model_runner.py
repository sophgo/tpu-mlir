#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import numpy as np
import argparse
import os


def round_away_from_zero(x):
    a = np.floor(np.abs(x) + 0.5)
    return np.sign(x) * a


def model_inference(inputs: dict, model_file: str) -> dict:
    import pyruntime
    outputs = dict()
    model = pyruntime.Model(model_file)
    net = model.Net(model.networks[0])
    for i in net.inputs:
        assert i.name in inputs
        assert i.data.shape == inputs[i.name].shape
        if i.data.dtype == inputs[i.name].dtype:
            i.data[:] = inputs[i.name]
        elif i.data.dtype == np.int8 and inputs[i.name].dtype == np.float32:
            data = round_away_from_zero(inputs[i.name] * i.qscale + i.qzero_point)
            i.data[:] = np.clip(data, -128, 127).astype(np.int8)
        elif i.data.dtype == np.uint8 and inputs[i.name].dtype == np.float32:
            data = round_away_from_zero(inputs[i.name] * i.qscale + i.qzero_point)
            i.data[:] = np.clip(data, 0, 255).astype(np.uint8)
        else:
            raise ValueError("unknown type: form {inputs[i.name].dtype} to {i.data.dtype}")
    net.forward()
    for i in net.outputs:
        if (i.data.dtype == np.int8 or i.data.dtype == np.uint8) and i.qscale != 0:
            outputs[i.name] = np.array((i.data.astype(np.float32) - i.qzero_point) * i.qscale, dtype=np.float32)
        else:
            outputs[i.name] = np.array(i.data)
    return outputs


def mlir_inference(inputs: dict, mlir_file: str, dump_all: bool = True) -> dict:
    import pymlir
    module = pymlir.module()
    module.load(mlir_file)
    for name in module.input_names:
        assert (name in inputs)
        input = inputs[name]
        is_int = False
        if input.dtype == np.int8 or input.dtype == np.uint8:
            is_int = True
            input = input.astype(np.float32)
        module.set_tensor(name, input, is_int)
    module.invoke()
    tensors = module.get_all_tensor()
    if dump_all:
        return tensors
    outputs = dict()
    for name in module.output_names:
        outputs[name] = tensors[name]
    return outputs


def onnx_inference(inputs: dict, onnx_file: str, dump_all: bool = True) -> dict:
    import onnx
    import onnxruntime

    def generate_onnx_with_all(onnx_file: str):
        # for dump all activations
        # plz refre https://github.com/microsoft/onnxruntime/issues/1455
        output_keys = []
        model = onnx.load(onnx_file)
        no_list = [
            "Cast", "Shape", "Unsqueeze", "Gather", "Split", "Constant", "GRU", "Sqrt",
            "ReduceMean", "Pow", "Sub", "Dropout", "Loop", "TopK"
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
        os.remove(onnx_file)
        return dict(zip(output_keys, map(np.ndarray.flatten, outs)))


def caffe_inference(inputs: dict, prototxt: str, caffemodel: str, dump_all: bool = True) -> dict:
    import caffe
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    for in_ in net.inputs:
        if in_ not in inputs:
            raise RuntimeError("inputs have no name [{}]".format(in_))
        net.blobs[in_].reshape(*inputs[in_].shape)
        net.blobs[in_].data[...] = inputs[in_]
    top_map = {}
    out = net.forward()
    if dump_all:
        for name, layer in net.layer_dict.items():
            if layer.type == "Split":
                continue
            if layer.type == "Slice":
                continue
            top_map[net.top_names[name][0]] = name
        blobs_dict = dict(inputs)
        for top, name in top_map.items():
            blobs_dict[name] = net.blobs[top].data.copy()
        return blobs_dict
    else:
        return out


def tflite_inference(
        inputs: dict,
        tflite_file: str,
        dump_all: bool = True,
        input_is_nchw: bool = True,
        use_represent_type: bool = True,
        tf_layout: bool = False,  # if "True" the the layout is nhwc
) -> dict:
    # TFLiteInterpreter is heavy, only import it when needed.
    from transform.TFLiteInterpreter import TFLiteInterpreter

    session = TFLiteInterpreter(
        tflite_file,
        experimental_preserve_all_tensors=dump_all,
    )

    def out_tensor_process(tensor_with_desc):
        if use_represent_type:
            tensor = session.to_represent_dat(tensor_with_desc)
        else:
            _, tensor = tensor_with_desc

        if not tf_layout:
            if tensor.ndim == 4:
                return tensor.transpose([0, 3, 1, 2])
        return tensor

    data = {}
    for input in session.inputs:
        name = input["name"]
        t = input["dtype"]
        is_quant = 'quantization' in input
        assert (name in inputs)
        d = inputs[name]
        if d.dtype == t:
            data[name] = d
        elif d.dtype == np.float32 and is_quant:
            scale, zp = input["quantization"]
            data[name] = np.clip(round_away_from_zero(d / scale + zp), 0, 255).astype(t)
        else:
            raise RuntimeError("input type:{} not match model type:{}".format(d.dtype, t))
    outputs = session.run(input_is_nchw, **data)

    if dump_all:
        return {
            k["name"]: out_tensor_process((k, v))
            for k, v in session.get_all_tensors() if v is not None
        }

    else:
        return {k["name"]: out_tensor_process((k, v)) for k, v in outputs}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--input", required=True, help="input npz file")
    parser.add_argument("--model", type=str, required=True,
                        help="mlir/onnx/tflie/bmodel/prototxt file.")
    parser.add_argument("--weight", type=str, default="", help="caffemodel for caffe")
    parser.add_argument("--output", default='_output.npz', help="output npz file")
    parser.add_argument("--dump_all_tensors",action='store_true',
                        help="dump all tensors to output file")
    # yapf: enable
    args = parser.parse_args()
    data = np.load(args.input)
    output = dict()
    if args.model.endswith('.mlir'):
        output = mlir_inference(data, args.model, args.dump_all_tensors)
    elif args.model.endswith('.onnx'):
        output = onnx_inference(data, args.model, args.dump_all_tensors)
    elif args.model.endswith(".tflite"):
        output = tflite_inference(data, args.model, args.dump_all_tensors)
    elif args.model.endswith(".prototxt") and args.weight.endswith(".caffemodel"):
        output = caffe_inference(data, args.model, args.weight, args.dump_all_tensors)
    elif args.model.endswith(".bmodel"):
        output = model_inference(data, args.model)
    else:
        raise RuntimeError("not support modle file:{}".format(args.model))
    np.savez(args.output, **output)
    print("Result saved to:{}".format(args.output))
