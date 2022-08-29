# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .MLIRImporter import MLIRImporter
from .BaseConverter import BaseConverter
import numpy as np

import math
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from termcolor import colored, cprint
from utils.pad_setting import get_TF_SAME_Padding, set_auto_pad


class CaffeTensor():

    def __init__(self, name, value, shape):
        self.name = name
        self.tensor_data = value
        self.shape = shape


class CaffeConverter(BaseConverter):

    def __init__(self,
                 model_name: str,
                 prototxt: str,
                 caffemodel: str,
                 input_shapes: list,
                 output_names: list,
                 preprocess_args=None):
        super().__init__()
        self.model_name = model_name
        self.prototxt = prototxt
        self.caffemodel = caffemodel
        self.net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)
        self.param = caffe_pb2.NetParameter()
        text_format.Merge(open(prototxt).read(), self.param)
        self.layers = self.param.layer if len(self.param.layer) != 0 else self.param.layers
        self.input_names = self.net.inputs
        self.select_outputs = set(output_names) if output_names else set(self.net.outputs)
        self.blobs = self.net.blobs
        self.mlir = None
        self.layer_dict = self.net.layer_dict
        self.weight_file = "{}_top_weight.npz".format(model_name)
        self.get_model_info(input_shapes, output_names)
        self.init_MLIRImporter()
        self.preprocess_args = preprocess_args

        self.caffeop_factory = {
            #pls add the Op according to the Op's alphabetical order as below
            "Add": lambda layer: self.convert_add_op(layer),
            "AveragePool": lambda layer: self.convert_avgpool_op(layer),
            "BatchNorm": lambda layer: self.convert_batchnorm_op(layer),
            "Convolution": lambda layer: self.convert_conv_op(layer),
            "InnerProduct": lambda layer: self.convert_inner_product_op(layer),
            "MaxPool": lambda layer: self.convert_maxpool_op(layer),
            "Mul": lambda layer: self.convert_mul_op(layer),
            "ReLU": lambda layer: self.convert_relu_op(layer),
            "Scale": lambda layer: self.convert_scale_op(layer),
        }

    def __del__(self):
        if self.mlir != None:
            del self.mlir
            self.mlir = None

    def addOperand(self, name, op):
        self.operands[name] = op

    def check_input(self, input_shapes):
        num_input = len(self.input_names)

        def check_shape(l, r):
            if l != r:
                raise KeyError("input shapes error:{}, {} vs {}".format(input_shapes, l, r))

        check_shape(len(input_shapes), num_input)

        for idx, shape in enumerate([self.shapes[name] for name in self.input_names]):
            check_shape(len(shape), len(input_shapes[idx]))
            for _i, _dim in enumerate(shape):
                check_shape(_dim, input_shapes[idx][_i])

    def get_model_info(self, input_shapes: list, output_names: list):
        # Todo: add select outputs func
        for layer_name, blob in self.blobs.items():
            self.addShape(layer_name, blob.data.shape)
        layer_names = list(self.layer_dict.keys())
        for idx, layer_name in enumerate(layer_names):
            if layer_name not in self.shapes and idx != 0:
                self.addShape(layer_name, self.getShape(layer_names[idx - 1]))
        self.check_input(input_shapes)

        pre_name = ""
        for layer_name, param in self.net.params.items():
            layer_type = self.layer_dict[layer_name].type
            if layer_type == "BatchNorm":
                s = param[2].data if param[2].data != 0 else 1
                mean = param[0].data / s
                variance = param[1].data / s

                self.addWeight(layer_name + "_mean", mean)
                self.addWeight(layer_name + "_variance", variance)
                self.addWeight(layer_name + "_gamma", np.ones_like(mean))
                self.addWeight(layer_name + "_beta", np.zeros_like(mean))
            else:
                self.addWeight(layer_name + "_weight", param[0].data)
                if len(param) > 1:
                    self.addWeight(layer_name + "_bias", param[1].data)

    def init_MLIRImporter(self):
        input_shapes = list()
        for _name in self.input_names:
            input_shapes.append(self.getShape(_name))
        output_shapes = list()
        for _name in self.select_outputs:
            output_shapes.append(self.getShape(_name))
        # init importer
        self.mlir = MLIRImporter(input_shapes, output_shapes, self.model_name)
        self.weight_file = self.mlir.weight_file

    def generate_mlir(self, mlir_file: str):
        # add input op
        for idx, _name in enumerate(self.input_names):
            input_shape = self.getShape(_name)
            channel_axis = 1
            if self.preprocess_args and self.preprocess_args['channel_format'] == 'nhwc':
                channel_axis = -1
            image = (len(input_shape) == 4 and input_shape[channel_axis] <=4) or \
                    (len(input_shape) == 3) # gray
            if not self.preprocess_args or not image:
                input_op = self.mlir.create_input_op(_name, idx, **{})
            else:
                preprocess_hint = {
                    'mean': self.preprocess_args['mean'],
                    'scale': self.preprocess_args['scale'],
                    'pixel_format': self.preprocess_args["pixel_format"],
                    'channel_format': self.preprocess_args["channel_format"],
                    'pad_type': self.preprocess_args["pad_type"],
                    'resize_dims': self.preprocess_args['resize_dims'],
                    'keep_aspect_ratio': self.preprocess_args['keep_aspect_ratio'],
                    'pad_value': self.preprocess_args["pad_value"]
                }
                input_op = self.mlir.create_input_op(_name, idx, **preprocess_hint)
            self.addOperand(_name, input_op)

        def NoneAndRaise(layer):
            raise RuntimeError("{} Op not support now".format(layer.type))

        # add mid layer op
        for layer in self.layers:
            if not len(self.select_outputs): break
            layer_type = layer.type
            if layer.type == "DummyData":
                continue
            elif layer.type == "Pooling":
                pool_type = layer.pooling_param.pool
                if pool_type == 0:
                    layer_type = "MaxPool"
                elif pool_type == 1:
                    layer_type = "AveragePool"
                else:
                    raise RuntimeError("{} Pool Op not support now".format(pool_type))
            elif layer.type == "Eltwise":
                eltwise_type = layer.eltwise_param.operation
                if eltwise_type == 0:
                    layer_type = "Mul"
                elif eltwise_type == 1:
                    layer_type = "Add"
                else:
                    # TODO: MaxOp to be implemented
                    raise RuntimeError("{} Op not support now".format(type.captialize()))

            op_name = self.caffeop_factory.get(layer_type, lambda x: NoneAndRaise(x))(layer)
            if layer.name in self.select_outputs:
                self.select_outputs.remove(layer.name)
                self.output_names.append(op_name)

        # add return op
        return_op = list()
        # Set output
        for idx, _name in enumerate(self.output_names):
            op = self.getOperand(_name)
            return_op.append(op)

        self.mlir.create_return_op(return_op)
        mlir_txt = self.mlir.print_module()
        with open(mlir_file, "w") as f:
            f.write(mlir_txt)
        self.WeightToNpz(self.weight_file)
        print("Save mlir file: {}".format(mlir_file))

    def convert_conv_op(self, caffe_layer):
        assert (caffe_layer.type == "Convolution")
        op = self.getOperand(caffe_layer.bottom[0])
        input_shape = self.getShape(caffe_layer.bottom[0])
        attrs = caffe_layer.convolution_param
        dim = len(input_shape) - 2
        kernel_shape = list(attrs.kernel_size) * dim
        dilations = (list(attrs.dilation) * dim) if attrs.dilation else dim * [1]
        strides = (list(attrs.stride) * dim) if attrs.stride else dim * [1]
        pads = (list(attrs.pad) * dim * 2) if attrs.pad else dim * [0] * 2

        operands = list()
        operands.append(op)
        filter_op = self.getWeightOp(caffe_layer.name + "_weight")
        operands.append(filter_op)
        if attrs.bias_term:
            bias_op = self.getWeightOp(caffe_layer.name + "_bias")
        else:
            bias_op = self.mlir.none_op
        operands.append(bias_op)
        p = {
            'name': "{}_{}".format(caffe_layer.name, caffe_layer.type),
            'kernel_shape': kernel_shape,
            'strides': strides,
            'dilations': dilations,
            'pads': pads,
            'group': 1,
            'do_relu': False,
            'ins': [],
        }
        output_shape = self.getShape(caffe_layer.name)
        new_op = self.mlir.create_conv_op(operands, output_shape, **p)
        self.addOperand(caffe_layer.top[0], new_op)
        return caffe_layer.top[0]

    def convert_batchnorm_op(self, caffe_layer):
        assert (caffe_layer.type == "BatchNorm")
        if caffe_layer.HasField('batch_norm_param') and caffe_layer.batch_norm_param.HasField(
                'use_global_stats'):
            assert (caffe_layer.batch_norm_param.use_global_stats == True)
        op = self.getOperand(caffe_layer.bottom[0])
        input_shape = self.getShape(caffe_layer.bottom[0])
        gamma = self.getWeightOp(caffe_layer.name + "_gamma")
        beta = self.getWeightOp(caffe_layer.name + "_beta")
        mean = self.getWeightOp(caffe_layer.name + "_mean")
        variance = self.getWeightOp(caffe_layer.name + "_variance")
        eps = np.array(caffe_layer.batch_norm_param.eps)
        output_shape = input_shape
        p = {
            "name": "{}_{}".format(caffe_layer.name, caffe_layer.type),
            "epsilon": eps,
        }
        new_op = self.mlir.create_batchnorm_op([op, gamma, beta, mean, variance], output_shape, **p)
        self.addOperand(caffe_layer.top[0], new_op)
        return caffe_layer.top[0]

    def convert_scale_op(self, caffe_layer):
        assert (caffe_layer.type == "Scale")
        op = self.getOperand(caffe_layer.bottom[0])
        input_shape = self.getShape(caffe_layer.bottom[0])
        operands = list()
        operands.append(op)
        num_dims = len(input_shape)
        assert (num_dims == 4 or num_dims == 2)
        output_shape = input_shape
        p = {"name": "{}_{}".format(caffe_layer.name, caffe_layer.type)}
        scale = self.getWeightOp(caffe_layer.name + "_weight")
        operands.append(scale)
        if caffe_layer.scale_param.bias_term:
            bias = self.getWeightOp(caffe_layer.name + "_bias")
            operands.append(bias)
        else:
            operands.append(self.mlir.none_op)
        new_op = self.mlir.create_scale_op(operands, output_shape, **p)
        self.addOperand(caffe_layer.top[0], new_op)
        return caffe_layer.top[0]

    def convert_relu_op(self, caffe_layer):
        assert (caffe_layer.type == "ReLU")
        op = self.getOperand(caffe_layer.bottom[0])
        input_shape = self.getShape(caffe_layer.bottom[0])
        output_shape = input_shape
        p = {
            'name': "{}_{}".format(caffe_layer.name, caffe_layer.type),
        }
        new_op = self.mlir.create_relu_op([op], output_shape, **p)
        self.addOperand(caffe_layer.top[0], new_op)
        return caffe_layer.top[0]

    def convert_maxpool_op(self, caffe_layer):
        assert (caffe_layer.type == "Pooling" and caffe_layer.pooling_param.pool == 0)
        input_shape = self.getShape(caffe_layer.bottom[0])
        op = self.getOperand(caffe_layer.bottom[0])
        attrs = caffe_layer.pooling_param
        dim = len(input_shape) - 2
        kernel_shape = [attrs.kernel_size] * dim
        strides = ([attrs.stride] * dim) if attrs.stride else kernel_shape
        output_shape = self.getShape(caffe_layer.name)
        pads = set_auto_pad("SAME_UPPER", input_shape, kernel_shape, strides)
        p = {
            'name': "{}_{}".format(caffe_layer.name, caffe_layer.type),
            'kernel_shape': kernel_shape,
            'strides': strides,
            'pads': pads,
            'count_include_pad': False,
            'do_relu': False,
        }
        new_op = self.mlir.create_maxpool_op([op], output_shape, **p)
        self.addOperand(caffe_layer.top[0], new_op)
        return caffe_layer.top[0]

    def convert_avgpool_op(self, caffe_layer):
        assert (caffe_layer.type == "Pooling" and caffe_layer.pooling_param.pool == 1)
        input_shape = self.getShape(caffe_layer.bottom[0])
        op = self.getOperand(caffe_layer.bottom[0])
        attrs = caffe_layer.pooling_param
        dim = len(input_shape) - 2
        kernel_shape = [attrs.kernel_size] * dim
        strides = ([attrs.stride] * dim) if attrs.stride else kernel_shape
        output_shape = self.getShape(caffe_layer.name)
        pad = (output_shape[-1] - 1) * strides[-1] - input_shape[-1] + attrs.kernel_size
        pads = [pad] * dim * 2
        p = {
            'name': "{}_{}".format(caffe_layer.name, caffe_layer.type),
            'kernel_shape': kernel_shape,
            'strides': strides,
            'pads': pads,
            'count_include_pad': True,
            'do_relu': False,
        }
        new_op = self.mlir.create_avgpool_op([op], output_shape, **p)
        self.addOperand(caffe_layer.top[0], new_op)
        return caffe_layer.top[0]

    def convert_add_op(self, caffe_layer):
        assert (caffe_layer.type == "Eltwise" and caffe_layer.eltwise_param.operation == 1)
        assert (len(caffe_layer.bottom) == 2)
        name = "{}_{}".format(caffe_layer.name, caffe_layer.type)
        if not self.isWeight(caffe_layer.bottom[0]) and self.isWeight(caffe_layer.bottom[1]):
            opd1_num_elem = np.prod(self.getShape(caffe_layer.bottom[1]))
            output_shape = self.getShape(caffe_layer.name)
            channel = output_shape[1]
            if opd1_num_elem == channel:
                op0 = self.getOperand(caffe_layer.bottom[0])
                offset = self.getWeight(caffe_layer.bottom[1])
                weight_data = np.ones_like(offset)
                self.addWeight(name + '_scale', weight_data)
                weight_op = self.getWeightOp(name + '_scale')
                offset_op = self.getWeightOp(caffe_layer.bottom[1])
                p = {'name': name}
                scale_op = self.mlir.create_scale_op([op0, weight_op, offset_op], output_shape, **p)
                self.addOperand(caffe_layer.top[0], scale_op)
                return caffe_layer.top[0]
            else:
                raise RuntimeError("To support adding with weight")
        op0 = self.getOperand(caffe_layer.bottom[0])
        op1 = self.getOperand(caffe_layer.bottom[1])
        p = {'name': "{}_{}".format(caffe_layer.name, caffe_layer.type)}
        output_shape = self.getShape(caffe_layer.name)
        add_op = self.mlir.create_add_op([op0, op1], output_shape, **p)
        self.addOperand(caffe_layer.top[0], add_op)
        return caffe_layer.top[0]

    def convert_mul_op(self, caffe_layer):
        assert (caffe_layer.op_type == "Eltwise" and caffe_layer.eltwise_param.operation == 0)
        assert (len(caffe_layer.bottom) == 2)
        if self.isWeight(caffe_layer.bottom[0]) and not self.isWeight(caffe_layer.bottom[1]):
            caffe_layer.bottom[0], caffe_layer.bottom[1] = caffe_layer.bottom[
                1], caffe_layer.bottom[0]
            return self.convert_mul_op(caffe_layer)
        name = "{}_{}".format(caffe_layer.name, caffe_layer.type)
        if (not self.isWeight(caffe_layer.bottom[0])) and self.isWeight(caffe_layer.bottom[1]):
            op0 = self.getOperand(caffe_layer.bottom[0])
            weight = self.getWeight(caffe_layer.bottom[1])
            output_shape = self.getShape(caffe_layer.name)
            weight_num_elem = np.prod(self.getShape(caffe_layer.bottom[1]))
            first_val = weight.flatten()[0]
            channel = output_shape[1]
            if weight_num_elem == 1 or np.all(weight == first_val):
                p = {'name': name, 'const_val': weight.flatten()[0]}
                mul_const_op = self.mlir.create_mul_const_op([op0], output_shape, **p)
                self.addOperand(caffe_layer.top[0], mul_const_op)
                return caffe_layer.top[0]
            elif weight_num_elem == channel:
                offset_data = np.zeros_like(weight)
                self.addWeight(name + '_bias', offset_data)
                weight_op = self.getWeightOp(caffe_layer.bottom[1])
                offset_op = self.getWeightOp(name + '_bias')
                p = {'name': name}
                scale_op = self.mlir.create_scale_op([op0, weight_op, offset_op], output_shape, **p)
                self.addOperand(caffe_layer.top[0], scale_op)
                return caffe_layer.top[0]
        else:
            op0 = self.getOperand(caffe_layer.bottom[0])
            op1 = self.getOperand(caffe_layer.bottom[1])
            p = {'name': name}
            output_shape = self.getShape(caffe_layer.name)
            mul_op = self.mlir.create_mul_op([op0, op1], output_shape, **p)
            self.addOperand(caffe_layer.top[0], mul_op)
            return caffe_layer.top[0]

    def convert_inner_product_op(self, caffe_layer):
        assert (caffe_layer.type == "InnerProduct")
        #(M, K) * (K, N) => (M, N)
        attrs = caffe_layer.inner_product_param
        op = self.getOperand(caffe_layer.bottom[0])
        # TODO:support more situations
        input_shape = self.getShape(caffe_layer.bottom[0])
        output_shape = self.getShape(caffe_layer.name)
        if len(input_shape) > 2:
            sec_dim = 1
            for dim in input_shape[1:]:
                sec_dim *= dim
            op = self.mlir.create_reshape_op([op], [input_shape[0], sec_dim], **{
                "name":
                "{}_{}".format(caffe_layer.bottom[0], "Reshape")
            })
        operands = list()
        operands.append(op)
        weight = caffe_layer.name + "_weight"
        _tensor = self.getWeight(weight)
        if output_shape[1] != _tensor.shape[-1]:
            _tensor = np.ascontiguousarray(np.transpose(_tensor, (1, 0)))
            weight += "_fix"
            self.addWeight(weight, _tensor)
        operands.append(self.getWeightOp(weight))
        if attrs.bias_term:
            bias = caffe_layer.name + "_bias"
            operands.append(self.getWeightOp(bias))
        else:
            operands.append(self.mlir.none_op)
        p = {'name': "{}_{}".format(caffe_layer.name, caffe_layer.type), 'do_relu': False}

        new_op = self.mlir.create_matmul_op(operands, output_shape, **p)
        self.addOperand(caffe_layer.top[0], new_op)
        return caffe_layer.top[0]
