#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import abc
import numpy as np
import argparse

from transform.BaseConverter import BaseConverter
from utils.mlir_shell import *
from utils.mlir_parser import *
from utils.preprocess import get_preprocess_parser, preprocess
import pymlir

def show_fake_cmd(in_npz: str, model: str, out_npz: str):
    print("[CMD]: model_runner.py --input {} --model {} --output {}".format(in_npz, model, out_npz))


class ModelTransformer(object):

    def __init__(self, model_name):
        self.model_name = model_name
        self.converter = BaseConverter()
        self.do_mlir_infer = True

    def cleanup(self):
        pass

    def model_transform(self, mlir_file: str):
        self.mlir_file = mlir_file
        mlir_origin = mlir_file.replace('.mlir', '_origin.mlir', 1)
        self.converter.generate_mlir(mlir_origin)
        mlir_opt_for_top(mlir_origin, self.mlir_file)
        print("Mlir file generated:{}".format(mlir_file))

        self.module_parsered = MlirParser(self.mlir_file)
        self.input_num = self.module_parsered.get_input_num()

    def model_validate(self, file_list: str, tolerance, excepts, test_result):
        in_f32_npz = self.model_name + '_in_f32.npz'
        inputs = dict()
        if len(file_list) == 1 and file_list[0].endswith('.npz'):
            npz_in = np.load(file_list[0])
            for name in self.converter.input_names:
                assert (name in npz_in.files)
                inputs[name] = npz_in[name]
        elif file_list[0].endswith('.jpg'):  #todo add isPicture in util
            ppa = preprocess()
            for i in range(self.input_num):
                pic_path = file_list[i] if i < len(file_list) else file_list[-1]
                file = os.path.expanduser(pic_path)
                ppa.load_config(self.module_parsered.get_input_op_by_idx(i))
                inputs[ppa.input_name] = ppa.run(file)
        else:
            assert (len(file_list) == len(self.converter.input_names))
            for name, file in zip(self.converter.input_names, file_list):
                assert (file.endswith('.npy'))
                inputs[name] = np.load(file)
        np.savez(in_f32_npz, **inputs)

        # original model inference to get blobs of all layer
        ref_outputs = self.origin_inference(inputs)
        if self.do_mlir_infer:
            ref_npz = self.model_name + '_ref_outputs.npz'
            np.savez(ref_npz, **ref_outputs)

            # inference of mlir model
            show_fake_cmd(in_f32_npz, self.mlir_file, test_result)
            from tools.model_runner import mlir_inference
            f32_outputs = mlir_inference(inputs, self.mlir_file)
            np.savez(test_result, **f32_outputs)

            # compare all blobs layer by layers
            f32_blobs_compare(test_result, ref_npz, tolerance, excepts=excepts)
        else:
            np.savez(test_result, **ref_outputs)

    @abc.abstractmethod
    def origin_inference(self, inputs: dict) -> dict:
        pass

class OnnxTransformer(ModelTransformer):
    def __init__(self,
                 model_name,
                 model_def,
                 input_shapes: list = [],
                 output_names=[],
                 preprocessor=None):
        super().__init__(model_name)
        self.model_def = model_def
        from transform.OnnxConverter import OnnxConverter
        self.converter = OnnxConverter(self.model_name, self.model_def, input_shapes, output_names,
                                       preprocessor)

    def origin_inference(self, inputs: dict):
        from tools.model_runner import onnx_inference
        return onnx_inference(inputs, self.converter.onnx_file)

class CaffeTransformer(ModelTransformer):
    def __init__(self,
                 model_name,
                 model_def,
                 model_data,
                 input_shapes: list = [],
                 output_names=[],
                 preprocessor=None):
        super().__init__(model_name)
        self.model_def = model_def
        self.model_data = model_data
        from transform.CaffeConverter import CaffeConverter
        self.converter = CaffeConverter(self.model_name, self.model_def, model_data,
                                        input_shapes, output_names, preprocessor)

    def origin_inference(self, inputs: dict):
        from tools.model_runner import caffe_inference
        return caffe_inference(inputs, self.model_def, self.model_data)

class TFLiteTransformer(ModelTransformer):

    def __init__(self, model_name, model_def, input_shapes: list = [], preprocessor=None):
        super().__init__(model_name)
        self.model_def = model_def
        self.do_mlir_infer = False
        from transform.TFLiteConverter import TFLiteConverter
        self.converter = TFLiteConverter(self.model_name, self.model_def, input_shapes,
                                         preprocessor)

    def origin_inference(self, inputs: dict):
        from tools.model_runner import tflite_inference
        return tflite_inference(inputs, self.converter.tflite_file)


def str2shape(v):
    _shape = eval(v)
    if not isinstance(_shape, list):
        raise KeyError("not shape list:{}".format(v))
    if len(_shape) == 0:
        return []
    dim = np.array(_shape).ndim
    if dim == 1:
        return [_shape]
    if dim != 2:
        raise KeyError("not shape list:{}".format(v))
    return _shape


def str2list(v):
    files = v.split(',')
    files = [s.strip() for s in files]
    while files.count('') > 0:
        files.remove('')
    return files


def get_model_transform(args):
    preprocessor = preprocess()
    preprocessor.config(**vars(args))
    if not args.mlir.endswith('.mlir'):
        raise RuntimeError("your mlir file should endswith .mlir, not:{}".format(args.mlir))
    tool = None
    if args.model_def.endswith('.onnx'):
        tool = OnnxTransformer(args.model_name, args.model_def, args.input_shapes,
                                      args.output_names, preprocessor.to_dict())
    elif args.model_def.endswith('.prototxt') and args.model_data.endswith('.caffemodel'):
        tool = CaffeTransformer(args.model_name, args.model_def, args.model_data, args.input_shapes,
                                      args.output_names, preprocessor.to_dict())
    elif args.model_def.endswith('.tflite'):
        tool = TFLiteTransformer(args.model_name, args.model_def, args.input_shapes,
                                        preprocessor.to_dict())
    else:
        # TODO: support more AI model types
        raise RuntimeError("unsupport model:{}".format(args.model_def))
    return tool


if __name__ == '__main__':
    print("SOPHGO Toolchain {}".format(pymlir.module().version))
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="model name")
    parser.add_argument("--model_def", required=True, help="model definition file.")
    parser.add_argument("--model_data", help="caffemodel, only for caffe model")
    parser.add_argument("--input_shapes",
                        type=str2shape,
                        default=list(),
                        help="list of input shapes, like:[[2,3],[1,2]]")
    parser.add_argument("--output_names",
                        type=str2list,
                        default=list(),
                        help="if set, will find names in model and set as real outputs")
    parser.add_argument("--test_input",
                        default="",
                        type=str2list,
                        help="input jpg/npy/npz file for inference, "
                        "if has more than one input, join jpg or npy with semicolon")
    parser.add_argument("--test_result",
                        default="",
                        type=str,
                        help="if input is set, result is mlir inference result")
    parser.add_argument("--tolerance",
                        default='0.99,0.99',
                        help="minimum similarity tolerance to model transform")
    parser.add_argument("--excepts", default='-', help="excepts")
    parser.add_argument("--mlir", type=str, required=True, help="output mlir model file")
    parser = get_preprocess_parser(existed_parser=parser)
    args = parser.parse_args()
    tool = get_model_transform(args)
    tool.model_transform(args.mlir)
    if args.test_input:
        assert (args.test_result)
        tool.model_validate(args.test_input, args.tolerance, args.excepts, args.test_result)
    tool.cleanup()
