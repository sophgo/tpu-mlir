#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import importlib
import numpy as np
import argparse
import os
import struct


def round_away_from_zero(x):
    a = np.floor(np.abs(x) + 0.5)
    return np.sign(x) * a


def bf16_to_fp32(d_bf16):
    s = d_bf16.shape
    d_bf16 = d_bf16.flatten()
    assert d_bf16.dtype == np.uint16
    d_fp32 = np.empty_like(d_bf16, dtype=np.float32)
    for i in range(len(d_bf16)):
        d_fp32[i] = struct.unpack('<f', struct.pack('<HH', 0, d_bf16[i]))[0]
    return d_fp32.reshape(s)


def fp32_to_bf16(d_fp32):
    s = d_fp32.shape
    d_fp32 = d_fp32.flatten()
    assert d_fp32.dtype == np.float32
    d_bf16 = np.empty_like(d_fp32, dtype=np.uint16)
    for i in range(len(d_bf16)):
        bytes = struct.pack('f', d_fp32[i])
        d_bf16[i] = struct.unpack('<H', struct.pack('BB', bytes[2], bytes[3]))[0]
    return d_bf16.reshape(s)


def show_fake_cmd(in_npz: str, model: str, out_npz: str):
    print("[CMD]: model_runner.py --input {} --model {} --output {}".format(in_npz, model, out_npz))


def get_chip_from_model(model_file: str) -> str:
    fd = os.popen("model_tool --chip {}".format(model_file))
    chip = fd.read()
    fd.close()
    return chip

def is_dynamic_model(model_file: str) -> str:
    fd = os.popen("model_tool --is_dynamic {}".format(model_file))
    dynamic = fd.read()
    fd.close()
    if dynamic == 'true':
        return True
    return False

def model_inference(inputs: dict, model_file: str) -> dict:
    pyruntime = "pyruntime_"
    is_cv18xx = False
    is_dynamic = False
    if model_file.endswith(".bmodel"):
        pyruntime = pyruntime + "bm"
        is_dynamic = is_dynamic_model(model_file)
        chip = get_chip_from_model(model_file)
        # trick for runtime link chip cmodel
        lib_so = 'libcmodel_1684x.so'
        if chip == 'BM1686':
            lib_so = 'libcmodel_1686.so'
        elif chip == 'BM1684':
            lib_so = 'libcmodel_1684.so'
        cmd = 'ln -sf $TPUC_ROOT/lib/{} $TPUC_ROOT/lib/libcmodel.so'.format(lib_so)
        os.system(cmd)
    elif model_file.endswith(".cvimodel"):
        pyruntime = pyruntime + "cvi"
        is_cv18xx = True
    else:
        raise RuntimeError("not support modle file:{}".format(model_file))
    pyruntime = importlib.import_module(pyruntime)

    outputs = dict()
    model = pyruntime.Model(model_file)
    try:
        net = model.Net(model.networks[0])
    except AttributeError:
        net = model
    dyn_input_shapes = []
    for i in net.inputs:
        assert i.name in inputs
        input = inputs[i.name]
        overflow = np.prod(i.data.shape) - np.prod(input.shape)
        if is_cv18xx and i.aligned:
            overflow = i.size - np.prod(input.shape)
        if is_dynamic:
            assert(len(i.data.shape) == len(input.shape))
            for max,dim in zip(i.data.shape, input.shape):
                if dim > max:
                    raise RuntimeError("Error shape: form {} to {}".format(i.data.shape, input.shape))
            dyn_input_shapes.append(input.shape)
            input = np.concatenate([input.flatten(), np.zeros([overflow]).astype(input.dtype)]).reshape(i.data.shape)
        elif overflow != 0:
            raise RuntimeError("Error shape: form {} to {}".format(i.data.shape, input.shape))
        zp = i.qzero_point
        if i.data.dtype == input.dtype:
            i.data[:] = input.reshape(i.data.shape)
        elif i.dtype == "i8" and input.dtype == np.float32:
            data = round_away_from_zero(input * i.qscale + zp)
            i.data[:] = np.clip(data, -128, 127).astype(np.int8).reshape(i.data.shape)
        elif i.dtype == "u8" and input.dtype == np.float32:
            data = round_away_from_zero(input * i.qscale + zp)
            i.data[:] = np.clip(data, 0, 255).astype(np.uint8).reshape(i.data.shape)
        elif i.dtype == "f16" and input.dtype == np.float32:
            i.data[:] = input.astype(np.float16)
        elif i.dtype == "bf16" and input.dtype == np.float32:
            i.data[:] = fp32_to_bf16(input)
        elif i.dtype == "i32" and input.dtype == np.float32:
            i.data[:] = input.astype(np.int32)
        else:
            raise ValueError(f"unknown type: form {input.dtype} to {i.data.dtype}")
    if not is_dynamic:
        net.forward()
    else:
        dyn_output_shapes = net.forward_dynamic(dyn_input_shapes)
    dyn_idx = 0
    for i in net.outputs:
        if (i.data.dtype == np.int8 or i.data.dtype == np.uint8) and i.qscale != 0:
            if is_cv18xx and i.name in inputs:
                name = i.name + "_si8" if i.data.dtype == np.int8 else "_ui8"
                outputs[name] = np.array(i.data.astype(np.float32) / np.float32(i.qscale))
            else:
                zp = i.qzero_point
                outputs[i.name] = np.array((i.data.astype(np.float32) - zp) * np.float32(i.qscale),
                                           dtype=np.float32)
        elif (i.dtype == "f16"):
            outputs[i.name] = np.array(i.data.astype(np.float32))
        elif (i.dtype == "bf16"):
            outputs[i.name] = bf16_to_fp32(i.data)
        else:
            outputs[i.name] = np.array(i.data)
        if is_dynamic:
            if outputs[i.name].shape != dyn_output_shapes[dyn_idx]:
                dyn_len = np.prod(dyn_output_shapes[dyn_idx])
                outputs[i.name] = outputs[i.name].flatten()[:dyn_len].reshape(*dyn_output_shapes[dyn_idx])
                dyn_idx += 1
    return outputs


def mlir_inference(inputs: dict, mlir_file: str, dump_all: bool = True) -> dict:
    import pymlir
    module = pymlir.module()
    module.load(mlir_file)
    for name in module.input_names:
        assert (name in inputs)
        input = inputs[name]
        if input.dtype == np.int8 or input.dtype == np.uint8:
            module.set_tensor_from_int(name, input.astype(np.float32))
        else:
            module.set_tensor(name, input.astype(np.float32))
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
        no_list = ["Cast", "Shape", "Unsqueeze", "Constant", "Dropout", "Loop"]

        # tested commited #c3cea486d https://github.com/microsoft/onnxruntime.git
        for x in model.graph.node:
            if x.op_type in no_list:
                continue
            for name in x.output:
                if not name:
                    continue
                intermediate_layer_value_info = onnx.helper.ValueInfoProto()
                intermediate_layer_value_info.name = name
                model.graph.output.append(intermediate_layer_value_info)
                output_keys.append(intermediate_layer_value_info.name + '_' + x.op_type)
        dump_all_tensors_onnx = onnx_file.replace('.onnx', '_all.onnx', 1)
        onnx.save(model, dump_all_tensors_onnx)
        return output_keys, dump_all_tensors_onnx

    output_keys = []
    if dump_all:
        output_keys, onnx_file = generate_onnx_with_all(onnx_file)
    session = onnxruntime.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
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
        blobs_dict = dict(inputs)
        for name, layer in net.layer_dict.items():
            if layer.type == "Split":
                continue
            if layer.type == "Slice":
                continue
            tops = net.top_names[name]
            for t in tops:
                blobs_dict[t] = net.blobs[t].data.copy()
        return blobs_dict
    else:
        return out


def tflite_inference(
        inputs: dict,
        tflite_file: str,
        dump_all: bool = True,
        input_is_nchw: bool = True,
        use_expressed_type: bool = True,
        tf_layout: bool = False,  # if "True" the the layout is nhwc
) -> dict:
    # TFLiteInterpreter is heavy, only import it when needed.
    from transform.TFLiteInterpreter import TFLiteInterpreter
    session = TFLiteInterpreter(
        tflite_file,
        experimental_preserve_all_tensors=dump_all,
    )

    def out_tensor_process(tensor_with_desc):
        if use_expressed_type:
            tensor = session.to_expressed_dat(tensor_with_desc)
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
    # yapf: disable
    parser = argparse.ArgumentParser()
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
    elif args.model.endswith(".bmodel") or args.model.endswith(".cvimodel"):
        output = model_inference(data, args.model)
    else:
        raise RuntimeError("not support modle file:{}".format(args.model))
    np.savez(args.output, **output)
    print("Result saved to:{}".format(args.output))
