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
import sys
import numpy as np
import argparse
import os
import struct
import shutil
from utils.misc import str2bool
from utils.lowering import lowering, round_away_from_zero, bf16_to_fp32


def show_fake_cmd(in_npz: str, model: str, out_npz: str, dump_all_tensors=False, use_cuda=False):
    dump_all = "--dump_all_tensors" if dump_all_tensors else ""
    cuda = "--cuda" if use_cuda else ""
    print("[CMD]: model_runner.py --input {} --model {} --output {} {} {}".format(
        in_npz, model, out_npz, dump_all, cuda))


def get_chip_from_model(model_file: str) -> str:
    fd = os.popen("model_tool --chip {}".format(model_file))
    chip = fd.read()
    fd.close()
    return chip


def pack_bmodel_context_generator(model_file, net):
    out_dir = model_file.rsplit(".", maxsplit=1)[0]
    tensor_loc = model_file + ".json"
    if not os.path.isfile(tensor_loc):
        yield None
        return
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(model_file, os.path.join(out_dir, "compilation.bmodel"))
    shutil.copy(tensor_loc, os.path.join(out_dir, "tensor_location.json"))
    with open(out_dir + "/input_ref_data.dat", "wb") as f:
        for i in net.inputs:
            i.data.tofile(f)
    yield
    with open(out_dir + "/output_ref_data.dat", "wb") as f:
        for o in net.outputs:
            o.data.tofile(f)


def model_inference(inputs: dict, model_file: str, dump_all: bool = True, mute: bool = False, out_fixed: bool = False, decrypt_lib: str = "",log_level:str='normal') -> dict:
    if mute or log_level == "quiet":
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
    try:
        return _model_inference(inputs, model_file, dump_all, out_fixed, decrypt_lib)
    finally:
        if mute or log_level=='quiet' or log_level=='simple' or log_level=='only-layer-group':
            os.dup2(sys.__stdout__.fileno(), sys.stdout.fileno())
            os.dup2(sys.__stderr__.fileno(), sys.stderr.fileno())


def _model_inference(inputs: dict, model_file: str, dump_all=True, out_fixed=False, decrypt_lib: str = "") -> dict:
    pyruntime = "pyruntime_"
    is_cv18xx = False
    if model_file.endswith(".bmodel"):
        pyruntime = pyruntime + "bm"
        chip = get_chip_from_model(model_file)
        # trick for runtime link chip cmodel
        lib_so = 'libcmodel_1684x.so'
        if chip == 'BM1688' or chip == 'CV186X':
            lib_so = 'libcmodel_1688.so'
        elif chip == 'BM1684':
            lib_so = 'libcmodel_1684.so'
        elif chip == "BM1690":
            lib_so = 'libcmodel_bm1690.so'
            pyruntime = "pyruntime_" + "tpuv7"
        elif chip == "MARS3":
            lib_so = 'libcmodel_mars3.so'
        elif chip == "SG2380":
            lib_so = 'libcmodel_sg2380.so'
        assert (os.path.exists("{}/lib/{}".format(os.getenv("TPUC_ROOT"), lib_so)))
        cmd = 'ln -sf $TPUC_ROOT/lib/{} $TPUC_ROOT/lib/libcmodel.so'.format(
            lib_so)
        os.system(cmd)
        # used in custom layer
        lib_so = 'libcmodel_custom_1684x.so'
        if chip == 'BM1688':
            lib_so = 'libcmodel_custom_1688.so'
        # elif chip == 'BM1684':
        #     lib_so = 'libcmodel_custom_1684.so'
        # elif chip == "BM1690":
        #     lib_so = 'libcmodel_custom_bm1690.so'
        # elif chip == "MARS3":
        #     lib_so = 'libcmodel_custom_mars3.so'
        if os.path.exists("{}/lib/{}".format(os.getenv("TPUC_ROOT"), lib_so)):
            cmd = 'ln -sf $TPUC_ROOT/lib/{} $TPUC_ROOT/lib/libcmodel_custom.so'.format(
                lib_so)
            os.system(cmd)
    elif model_file.endswith(".cvimodel"):
        pyruntime = pyruntime + "cvi"
        is_cv18xx = True
    else:
        raise RuntimeError("not support modle file:{}".format(model_file))
    pyruntime = importlib.import_module(pyruntime)

    outputs = dict()
    if not is_cv18xx:
        model = pyruntime.Model(model_file, 0, decrypt_lib)
        net = model.Net(model.networks[0])
    else:
        model = pyruntime.Model(model_file, output_all_tensors=dump_all)
        net = model
    input_shapes = []
    only_one = len(inputs) == 1
    if only_one and len(net.inputs) != 1:
        raise RuntimeError("Input num not the same")
    for i in net.inputs:
        if not only_one:
            assert i.name in inputs
            input = inputs[i.name]
        else:
            input = list(inputs.values())[0]
        if not is_cv18xx:
            overflow = np.prod(i.data.shape) - np.prod(input.shape)
            assert (overflow >= 0)
            if overflow > 0:
                # input = np.concatenate([input.flatten(),
                #                         np.zeros([overflow]).astype(input.dtype)
                #                         ]).reshape(i.data.shape)
                input_shapes.append(input.shape)
            else:
                input_shapes.append(i.data.shape)
        # i.data[:] = lowering(input, pdtype=i.dtype, pshape=i.data.shape,pzero_point=i.qzero_point, pscale=i.qscale)
        i.data.reshape(-1)[:np.prod(input.shape)] = lowering(input, pdtype=i.dtype,
                                                             pshape=input.shape, pzero_point=i.qzero_point, pscale=i.qscale).flatten()

    size = os.path.getsize(model_file)
    pack_bmodel_context = (iter([None]) if is_cv18xx else pack_bmodel_context_generator(
        model_file, net))

    next(pack_bmodel_context)  # save input_data

    if size > 0x10000000:
        print("Warning: {} is too large and will cost a long time. ".format(model_file))
        print(
            'Add the "--skip_validation" argument when using "model_deploy.py"'
            " to bypass running the cmodel if checking the correctness of the BModel is not required."
        )

    if is_cv18xx:
        net.forward()
    else:
        # always use forward_dynamic to get output shape, because some static model may have dynamic output
        # for example: last layer is NonMaxSuppression, NonZero, etc.
        dyn_output_shapes = net.forward_dynamic(input_shapes)
    dyn_idx = 0

    for i in net.outputs:
        if (i.data.dtype == np.int8 or i.data.dtype == np.uint8) and i.qscale != 0 and out_fixed == False:
            if is_cv18xx and i.name in inputs:
                name = i.name + "_si8" if i.data.dtype == np.int8 else "_ui8"
                outputs[name] = np.array(i.data.astype(
                    np.float32) / np.float32(i.qscale))
            else:
                zp = i.qzero_point
                outputs[i.name] = np.array((i.data.astype(np.float32) - zp) * np.float32(i.qscale),
                                           dtype=np.float32)
        elif (i.dtype == 'u16'):
            outputs[i.name] = np.array(i.data.astype(np.float32))
        elif (i.dtype == "f16"):
            outputs[i.name] = np.array(i.data.astype(np.float32))
        elif (i.dtype == "bf16"):
            outputs[i.name] = bf16_to_fp32(i.data)
        else:
            outputs[i.name] = np.array(i.data)
        if not is_cv18xx:
            if list(outputs[i.name].shape) != dyn_output_shapes[dyn_idx]:
                dyn_len = np.prod(dyn_output_shapes[dyn_idx])
                outputs[i.name] = outputs[i.name].flatten()[:dyn_len].reshape(
                    *dyn_output_shapes[dyn_idx])
            dyn_idx += 1
    try:
        next(pack_bmodel_context)  # save output
    except StopIteration:
        pass

    if not is_cv18xx and dump_all:
        if "NEED_DUMP_DYNAMIC_LAYER_OUTPUT_DATA" in os.environ and os.environ["NEED_DUMP_DYNAMIC_LAYER_OUTPUT_DATA"] == "1":
            if "DYNAMIC_LAYER_OUTPUT_DATA_PATH" in os.environ and \
                    "DYNAMIC_LAYER_OUTPUT_ID_DICT_PATH" in os.environ:
                dyn_layer_out_data_path = os.environ["DYNAMIC_LAYER_OUTPUT_DATA_PATH"]
                id_dict_file = os.environ["DYNAMIC_LAYER_OUTPUT_ID_DICT_PATH"]
                if os.path.exists(dyn_layer_out_data_path) and os.path.exists(id_dict_file) and os.path.isfile(id_dict_file):
                    id_dict_str = "{"
                    with open(id_dict_file) as f:
                        id_dict_str += f.readline()
                    id_dict_str += "}"
                    id_dict = eval(id_dict_str)
                    os.remove(id_dict_file)
                    ids = sorted([int(file) for file in os.listdir(
                        dyn_layer_out_data_path) if file.isdecimal() and int(file) in id_dict])
                    for id in ids:
                        name = id_dict[id]
                        filename = os.path.join(
                            dyn_layer_out_data_path, str(id))
                        with open(filename, "r") as f:
                            dict_str = f.readline()
                            dict_ = eval(dict_str)
                            data = dict_["data"]
                            is_fp = dict_["is_fp"]
                            shape = dict_["shape"]
                            data = np.array(data, dtype=(
                                np.float32 if is_fp else np.int32))
                            outputs[name] = np.reshape(data, shape)
                    os.system(f"rm -r {dyn_layer_out_data_path}/*")

    return outputs

g_final_mlir_module = None

def final_mlir_inference(inputs: dict, mlir_file: str, dump_all: bool = True) -> dict:
    import pyfinalmlir
    global g_final_mlir_module
    if g_final_mlir_module != None:
        g_final_mlir_module = None
    g_final_mlir_module = pyfinalmlir.module()
    g_final_mlir_module.load(mlir_file)
    only_one = len(inputs) == 1
    if only_one:
        assert (len(g_final_mlir_module.input_names) == 1)
    for name in g_final_mlir_module.input_names:
        if not only_one:
            assert (name in inputs)
            input = inputs[name]
        else:
            input = list(inputs.values())[0]
        if input.dtype == np.int8 or input.dtype == np.uint8:
            g_final_mlir_module.set_tensor_from_int(name, input.astype(np.float32))
        else:
            g_final_mlir_module.set_tensor(name, input.astype(np.float32))
    tensors = dict()
    g_final_mlir_module.invoke()
    tensors = g_final_mlir_module.get_all_tensor()
    if dump_all:
        return tensors
    from utils.mlir_parser import MlirParser
    parser = MlirParser(mlir_file)
    outputs = dict()
    for name in g_final_mlir_module.output_names:
        outputs[name] = tensors[name]
        # assume output of op has the same name
        op_type = parser.get_op_type_by_op_name(name)
        if op_type == "tpu.Cast":
            pre_op = parser.get_pre_op_by_op_name(name)[0]
            if pre_op in tensors:
                outputs[pre_op] = tensors[pre_op]
    return outputs


g_mlir_module = None
g_mlir_cuda = None


def mlir_inference(inputs: dict, mlir_file: str, dump_all: bool = True, mute: bool = False, out_fixed: bool = False, use_cuda: bool = False, log_level:str = 'normal') -> dict:
    if mute or log_level == "quiet":
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
    try:
        if not use_cuda:
            return _mlir_inference_by_cpu(inputs, mlir_file, dump_all, out_fixed)
        else:
            return _mlir_inference_by_cuda(inputs, mlir_file, dump_all)
    finally:
        if mute or log_level == "quiet":
            os.dup2(sys.__stdout__.fileno(), sys.stdout.fileno())
            os.dup2(sys.__stderr__.fileno(), sys.stderr.fileno())


def _mlir_inference_by_cpu(inputs: dict, mlir_file: str, dump_all: bool = True, out_fixed: bool = False) -> dict:
    import pymlir
    pymlir.set_mem_mode("value_mem")
    from utils.mlir_parser import MlirParser
    global g_mlir_module
    if g_mlir_module != None:
        g_mlir_module = None
    g_mlir_module = pymlir.module()
    g_mlir_module.load(mlir_file)
    parser = MlirParser(mlir_file)
    only_one = len(inputs) == 1
    if only_one:
        assert (len(g_mlir_module.input_names) == 1)
    for name in g_mlir_module.input_names:
        if not only_one:
            assert (name in inputs)
            input = inputs[name]
        else:
            input = list(inputs.values())[0]
        if input.dtype == np.int8 or input.dtype == np.uint8:
            g_mlir_module.set_tensor_from_int(name, input.astype(np.float32))
        else:
            g_mlir_module.set_tensor(name, input.astype(np.float32))
    tensors = dict()
    g_mlir_module.invoke(not out_fixed)
    tensors = g_mlir_module.get_all_tensor()
    if dump_all:
        return tensors
    outputs = dict()
    for name in g_mlir_module.output_names:
        outputs[name] = tensors[name]
        # assume output of op has the same name
        op_type = parser.get_op_type_by_op_name(name)
        if op_type == "tpu.Cast":
            pre_op = parser.get_pre_op_by_op_name(name)[0]
            if pre_op in tensors:
                outputs[pre_op] = tensors[pre_op]
    return outputs


def _mlir_inference_by_cuda(inputs: dict, mlir_file: str, dump_all: bool = False) -> dict:
    import pymlir
    if not pymlir.support_cuda:
        raise RuntimeError("current version not support cuda")
    global g_mlir_cuda
    if g_mlir_cuda != None:
        g_mlir_cuda = None
    g_mlir_cuda = pymlir.cuda()
    g_mlir_cuda.load(mlir_file)
    only_one = len(inputs) == 1
    if only_one:
        assert (len(g_mlir_cuda.input_names) == 1)
    for name in g_mlir_cuda.input_names:
        if not only_one:
            assert (name in inputs)
            input = inputs[name]
        else:
            input = list(inputs.values())[0]
        g_mlir_cuda.set_tensor(name, input.astype(np.float32))
    g_mlir_cuda.invoke(dump_all)
    return g_mlir_cuda.get_all_tensor()


def free_mlir_module():
    global g_mlir_module
    g_mlir_module = None
    global g_mlir_cuda
    g_mlir_cuda = None


def onnx_inference(inputs: dict, onnx_file: str, dump_all: bool = True) -> dict:
    import onnx
    import onnxruntime

    def generate_onnx_with_all(onnx_file: str):
        # for dump all activations
        # plz refre https://github.com/microsoft/onnxruntime/issues/1455
        output_keys = []
        model = onnx.load(onnx_file)
        no_list = ["Cast", "Constant", "Dropout", "Loop"]

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
                output_keys.append(
                    intermediate_layer_value_info.name + '_' + x.op_type)
        dump_all_tensors_onnx = onnx_file.replace('.onnx', '_all.onnx', 1)
        try:
            onnx.save(model, dump_all_tensors_onnx)
        except Exception as E:
            if "The proto size is larger than the 2 GB limit." in str(E):
                print("LOG: Try to save {} by using save_as_external_data to save tensors separately from the model file.".format(
                    dump_all_tensors_onnx))
                onnx.save(model,
                          dump_all_tensors_onnx,
                          save_as_external_data=True,
                          location="model_runner_external_data",
                          convert_attribute=True)
            else:
                raise E
        return output_keys, dump_all_tensors_onnx

    output_keys = []
    if dump_all:
        output_keys, onnx_file = generate_onnx_with_all(onnx_file)
    session = onnxruntime.InferenceSession(
        onnx_file, providers=['CPUExecutionProvider'])
    inodes = session.get_inputs()
    only_one = len(inputs) == 1
    if only_one:
        assert (len(inodes) == 1)
    data = {}
    for node in inodes:
        name = node.name
        dtype = np.float32
        if node.type == 'tensor(int64)':
            dtype = np.int64
        elif node.type == 'tensor(bool)':
            dtype = np.bool_
        elif node.type == 'tensor(int32)':
            dtype = np.int32
        if not only_one:
            assert (name in inputs)
            data[name] = inputs[name].astype(dtype)
        else:
            data[name] = list(inputs.values())[0].astype(dtype)
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
        return dict(filter(lambda x: isinstance(x[1], np.ndarray), zip(output_keys, outs)))


def caffe_inference(inputs: dict, prototxt: str, caffemodel: str, dump_all: bool = True) -> dict:
    import caffe
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    only_one = len(inputs) == 1
    if only_one:
        assert (len(net.inputs) == 1)
    for in_ in net.inputs:
        if not only_one:
            assert (in_ in inputs)
            input = inputs[in_]
        else:
            input = list(inputs.values())[0]
        net.blobs[in_].reshape(*input.shape)
        net.blobs[in_].data[...] = input
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
    only_one = len(inputs) == 1
    if only_one:
        assert (len(session.inputs) == 1)
    for input in session.inputs:
        name = input["name"]
        t = input["dtype"]
        is_quant = 'quantization' in input
        if not only_one:
            assert (name in inputs)
            d = inputs[name]
        else:
            d = list(inputs.values())[0]
        if d.dtype == t:
            data[name] = d
        elif d.dtype == np.float32 and is_quant:
            scale, zp = input["quantization"]
            data[name] = np.clip(round_away_from_zero(
                d / scale + zp), 0, 255).astype(t)
        else:
            raise RuntimeError(
                "input type:{} not match model type:{}".format(d.dtype, t))
    outputs = session.run(input_is_nchw, **data)

    if dump_all:
        return {
            k["name"]: out_tensor_process((k, v))
            for k, v in session.get_all_tensors() if v is not None
        }

    else:
        return {k["name"]: out_tensor_process((k, v)) for k, v in outputs}


def torch_inference(inputs: dict, model: str, dump_all: bool = True) -> dict:
    import torch

    if dump_all:
        from transform.TorchInterpreter import TorchInterpreter
        net = TorchInterpreter(model)
        net.run_model(inputs)
        return net.ref_tensor
    net = torch.jit.load(model, map_location=torch.device('cpu'))
    net.eval()
    in_tensors = [torch.from_numpy(v) for k, v in inputs.items()]
    with torch.no_grad():
        out_tensors = net(*in_tensors)

    names = []
    graph_alive = net.inlined_graph
    for out in graph_alive.outputs():
        if out.node().kind() == 'prim::TupleConstruct' or out.node().kind(
        ) == 'prim::ListConstruct':
            ins = out.node().inputs()
            names.extend([i.debugName() for i in ins])
        else:
            names.append(out.debugName())

    idx = 0

    def torch_outputs(outputs: dict, names: list, tensors):
        nonlocal idx
        if isinstance(tensors, torch.Tensor):
            outputs[names[idx]] = tensors.numpy()
            idx += 1
            return
        if isinstance(tensors, tuple) or isinstance(tensors, list):
            for t in tensors:
                torch_outputs(outputs, names, t)
        else:
            raise RuntimeError("Not Implemented")

    outputs = {}
    torch_outputs(outputs, names, out_tensors)
    return outputs


if __name__ == '__main__':
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input npz file")
    parser.add_argument("--model", type=str, required=True,
                        help="mlir/pytorch/onnx/tflie/bmodel/prototxt file.")
    parser.add_argument("--weight", type=str, default="",
                        help="caffemodel for caffe")
    parser.add_argument("--output", default='_output.npz',
                        help="output npz file")
    parser.add_argument("--dump_all_tensors", action='store_true',
                        help="dump all tensors to output file")
    parser.add_argument("--debug", type=str, nargs="?", const="",
                        help="configure the debugging information.")
    parser.add_argument("--out_fixed", action="store_true",
                        help="no float number transforming, only for int8/uint8.")
    parser.add_argument("--cuda", action="store_true",
                        help="use cuda to do inference")
    parser.add_argument("--decrypt_lib", type=str, default="",
                        help="use decrypt_lib to load encrypted bmodel")
    # yapf: enable
    args = parser.parse_args()
    data = np.load(args.input)
    output = dict()
    if args.model.endswith("final.mlir"):
        output = final_mlir_inference(
            data, args.model, args.dump_all_tensors)
    elif args.model.endswith(".mlir"):
        output = mlir_inference(
            data, args.model, args.dump_all_tensors, args.debug, args.out_fixed, args.cuda)
    elif args.model.endswith('.onnx'):
        output = onnx_inference(data, args.model, args.dump_all_tensors)
    elif args.model.endswith(".tflite"):
        output = tflite_inference(data, args.model, args.dump_all_tensors)
    elif args.model.endswith(".prototxt") and args.weight.endswith(".caffemodel"):
        output = caffe_inference(
            data, args.model, args.weight, args.dump_all_tensors)
    elif args.model.endswith(".pt") or args.model.endswith(".pth"):
        output = torch_inference(data, args.model, args.dump_all_tensors)
    elif args.model.endswith(".bmodel") or args.model.endswith(".cvimodel"):
        output = model_inference(data, args.model, out_fixed=args.out_fixed, decrypt_lib=args.decrypt_lib)
    else:
        raise RuntimeError("not support modle file:{}".format(args.model))
    print("\nSaving ...")
    if output:
        np.savez(args.output, **output)
        print("\nResult saved to:{}".format(args.output))
