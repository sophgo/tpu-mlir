#!/usr/bin/env python3
import numpy as np
import argparse
import pymlir
import pyruntime
import onnx
import onnxruntime
from transform.TFLiteInterpreter import TFLiteInterpreter

def model_inference(inputs: dict, model_file: str) -> dict:
    outputs = dict()
    model = pyruntime.Model(model_file)
    net = model.Net(model.networks[0])
    for i in net.inputs:
        assert(i.name in inputs)
        assert(i.data.shape == inputs[i.name].shape)
        assert(i.data.dtype == inputs[i.name].dtype)
        i.data[:] = inputs[i.name]
    net.forward()
    for i in net.outputs:
        outputs[i.name] = i.data
    return outputs

def mlir_inference(inputs: dict, mlir_file: str, dump_all: bool = True) -> dict:
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


def onnx_inference(inputs: dict, onnx_file: str, dump_all: bool = True) -> dict:
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


def tflite_inference(
    inputs: dict,
    tflite_file: str,
    dump_all: bool = True,
    input_is_nchw: bool = True,
    use_represent_type: bool = True,
    tf_layout: bool = False,  # if "True" the the layout is nhwc
) -> dict:
    session = TFLiteInterpreter(
        tflite_file,
        experimental_preserve_all_tensors=dump_all,
    )

    def nhwc2nchw(x):
        if x.ndim == 4:
            return x.transpose([0, 3, 1, 2])
        return x

    def nchw2nhwc(x):
        if x.ndim == 4:
            return x.transpose([0, 2, 3, 1])
        return x

    def to_represent_dat(desc, v):
        quantization_param = desc["quantization_parameters"]
        scales = quantization_param["scales"]
        zeros = quantization_param["zero_points"]
        if not np.issubdtype(v.dtype.type, np.integer):
            return v
        try:
            if len(zeros) == 1:
                return (v.astype(np.int32) - zeros[0]) * scales[0]
            if len(zeros) == 0:
                return v
            if len(zeros) > 1:
                shape = np.ones(v.ndim, dtype=np.int64)
                shape[quantization_param["quantized_dimension"]] = len(zeros)
                zeros = np.reshape(zeros, shape)
                scales = np.reshape(scales, shape)
                return (v.astype(np.int32) - zeros) * scales
        except:
            print(desc)
            raise ValueError()

    data = {}
    for input in session.inputs:
        name = input["name"]
        input = inputs[name].astype(input["dtype"])
        if input_is_nchw:
            data[name] = nchw2nhwc(input)
        else:
            data[name] = input
    outputs = session.run(**data)

    if tf_layout:
        layout_fun = lambda v: v
    else:
        layout_fun = nhwc2nchw

    if use_represent_type:
        rtp_fun = to_represent_dat
    else:
        rtp_fun = lambda k, v: v

    fm_fun = lambda k, v: rtp_fun(k, layout_fun(v))

    if dump_all:
        return {
            k["name"]: fm_fun(k, v)
            for k, v in session.get_all_tensors()
            if v is not None
        }

    else:
        return {k["name"]: fm_fun(k, v) for k, v in outputs.items()}



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
    elif args.model.endswith(".bmodel"):
        output = model_inference(data, args.model)
    else:
        raise RuntimeError("not support modle file:{}".format(args.model))
    np.savez(args.output, **output)
    print("Result saved to:{}".format(args.output))
