#!/usr/bin/env python3
import abc
import onnx, onnxruntime
import numpy as np
import argparse
from transform.OnnxConverter import OnnxConverter

class ModelTransformTool(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.op_info_csv = self.model_name + '_op_info.csv'

    def cleanup(self):
        pass

    def model_transform(self, mlir_file):
        self.mlir_file = mlir_file
        self._transform_(mlir_file)

    @staticmethod
    def _is_npz(image):
        return True if image.split('.')[-1] == 'npz' else False

    @staticmethod
    def _is_npy(image):
        return True if image.split('.')[-1] == 'npy' else False

    def model_validate(self, image, tolerance, excepts):
        # TODO: validate model inference
        raise RuntimeError("validate fail")

    @abc.abstractmethod
    def _inference_(self):
        pass

    @abc.abstractmethod
    def _transoform_(self):
        pass

class OnnxModelTransformTool(ModelTransformTool):
    def __init__(self, model_name, onnx_model, input_shapes:list=[]):
        super().__init__(model_name)
        self.onnx_model = onnx_model
        self.input_shapes = input_shapes

    def _fill_inputs(self, ort_session, inputs):
        inodes = ort_session.get_inputs()
        data = {}
        for i in range(len(inodes)):
            name = inodes[i].name
            dtype = np.float32
            if inodes[i].type == 'tensor(int64)':
                dtype = np.int64
            elif inodes[i].type == 'tensor(bool)':
                dtype = np.bool
            data[name] = inputs[name].astype(dtype)
        return data

    def _inference_(self, inputs):
        # TODO: support onnx inference
        raise RuntimeError("not support now")

    def _transform_(self, mlir_file):
        cvt = OnnxConverter(self.model_name, self.onnx_model, self.input_shapes, mlir_file)
        cvt.run()

def str2shape(v):
    _shape = eval(v)
    if not isinstance(_shape,list):
        raise KeyError("not shape list:{}".format(v))
    if len(_shape) == 0:
        return []
    dim = np.array(_shape).ndim
    if dim == 1:
        return [_shape]
    if dim != 2:
        raise KeyError("not shape list:{}".format(v))
    return _shape

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="model name")
    parser.add_argument("--model_type", required=True, choices=['onnx'], help="model_type")
    parser.add_argument("--model_def", required=True, help="model definition file.")
    parser.add_argument("--model_data", help="caffemodel, only for caffe model")
    parser.add_argument("--input_shapes", type=str2shape, default=list(),
                        help = "list of input shapes, like:[[2,3],[1,2]]")
    parser.add_argument("--input", default=None, help="input image/npz/npy file for inference, "
                       "if has more than one input, join images with semicolon")
    parser.add_argument("--tolerance", default='0.99,0.99,0.98',
                        help="minimum similarity tolerance to model transform")
    parser.add_argument("--excepts", default='-', help="excepts")
    parser.add_argument("--mlir", required=True, help="output mlir model file")
    args = parser.parse_args()
    tool = None
    if args.model_type == 'onnx':
        tool = OnnxModelTransformTool(args.model_name, args.model_def, args.input_shapes)
    else:
        # TODO: support more AI model types
        raise RuntimeError("unsupport model type:{}".format(args.model_type))
    tool.model_transform(args.mlir)
    if args.input:
        tool.model_validate(args.input, args.tolerance, args.excepts)
    tool.cleanup()
