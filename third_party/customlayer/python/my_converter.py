#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from mlir.ir import _f32ArrayAttr, _i64ArrayAttr
from transform.CaffeConverter import CaffeConverter
import argparse
import pymlir
from utils.mlir_shell import mlir_opt_for_top
from utils.misc import *
import mlir.dialects.top as top
from mlir.ir import *
import my_caffe_layer


def dict_attr_convert(param_dict: dict):
    '''
        Transform python dict to mlir DictArrayAttr
    '''
    array_attr = []
    for key, value in param_dict.items():
        sub_dict = {}
        if isinstance(value, int):
            sub_dict[key] = IntegerAttr.get(IntegerType.get_signless(64), value)
        elif isinstance(value, float):
            sub_dict[key] = FloatAttr.get(F32Type.get(), value)
        elif isinstance(value, list):
            if all(isinstance(x, int) for x in value):
                sub_dict[key] = _i64ArrayAttr.get(value)
            elif all(isinstance(x, float) for x in value):
                sub_dict[key] = _f32ArrayAttr.get(value)
            else:
                raise ValueError(f"Elements in the list of {key} must be int-only or float-only")
        array_attr.append(DictAttr.get(sub_dict))
    attrs = ArrayAttr.get(array_attr)

    return attrs


class MyCaffeConverter(CaffeConverter):

    def __init__(self, model_name: str, prototxt: str, caffemodel: str, input_shapes: list,
                 output_names: list):
        super().__init__(model_name, prototxt, caffemodel, input_shapes, output_names)
        self.caffeop_factory["Python"] = lambda layer: self.convert_python_op(layer)

    def model_convert(self):
        mlir_file = self.model_name + ".mlir"
        mlir_origin = self.model_name + "_origin.mlir"
        self.generate_mlir(mlir_origin)
        mlir_opt_for_top(mlir_origin, mlir_file)

    # Implement the convert function for the 'Python' type caffe op
    def convert_python_op(self, layer):
        assert (self.layerType(layer) == "Python")
        in_op = self.getOperand(layer.bottom[0])
        p = layer.python_param

        dict_attr = dict(eval(p.param_str))
        params = dict_attr_convert(dict_attr)

        # p.layer.lower() to keep the consistency with the backend op name
        attrs = {"name": p.layer.lower(), "params": params, 'loc': self.get_loc(layer.top[0])}

        # The output shape is obtained according to the overridden reshape function in my_caffe_layer
        out_shape = self.getShape(layer.top[0])
        outs = top.CustomOp([self.mlir.get_tensor_type(out_shape)], [in_op],
                            **attrs,
                            ip=self.mlir.insert_point).output
        # add the op result to self.operands
        self.addOperand(layer.top[0], outs[0])


if __name__ == '__main__':
    print("TPU-MLIR {}".format(pymlir.__version__))
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--model_name", required=True, help="model name")
    parser.add_argument("--model_def", required=True, help="model definition file.")
    parser.add_argument("--model_data", help="caffemodel, only for caffe model")
    parser.add_argument("--mlir", type=str, required=True, help="output mlir model file")
    parser.add_argument("--input_shapes", type=str2shape, default=list(),
                    help="list of input shapes, like:[[1,3,224,224],[10],[16]]")
    parser.add_argument("--output_names", type=str2list, default=list(),
                        help="if set, will find names in model and set as real outputs")
    args = parser.parse_args()

    converter = MyCaffeConverter(args.model_name, args.model_def, args.model_data, args.input_shapes, args.output_names)
    converter.model_convert()
