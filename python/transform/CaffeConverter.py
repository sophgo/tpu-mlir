# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .MLIRImporter import MLIRImporter
from .BaseConverter import BaseConverter
import numpy as np

import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format


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
        self.init_shapes(input_shapes)
        self.init_MLIRImporter()
        self.preprocess_args = preprocess_args

        self.caffeop_factory = {
            #pls add the Op according to the Op's alphabetical order as below
            "BatchNorm": lambda layer: self.convert_batchnorm_op(layer),
            "Convolution": lambda layer: self.convert_convolution_op(layer),
            'ConvolutionDepthwise': lambda layer: self.convert_convolution_op(layer),
            "Eltwise": lambda layer: self.convert_eltwise_op(layer),
            "InnerProduct": lambda layer: self.convert_inner_product_op(layer),
            'Pooling': lambda layer: self.convert_pooling_op(layer),
            "ReLU": lambda layer: self.convert_relu_op(layer),
            "Scale": lambda layer: self.convert_scale_op(layer),
            'Softmax': lambda layer: self.convert_softmax_op(layer),
        }
        # yapf: disable
        # for caffe v1
        self.layer_type = {
            0: 'None', 35: 'Absval', 1: 'Accuracy', 30: 'Argmax', 2: 'Bnll',
            3: 'Concat', 37: 'ContrastiveLoss', 4: 'Convolution', 5: 'Data',
            39: 'Deconvolution', 6: 'Dropout', 32: 'DummyData', 7: 'EuclideanLoss',
            25: 'Eltwise', 38: 'Exp', 8: 'Flatten', 9: 'Hdf5Data', 10: 'Hdf5Output',
            28: 'HingeLoss', 11: 'Im2col', 12: 'ImageData', 13: 'InfogainLoss',
            14: 'InnerProduct', 15: 'LRN', 29: 'MemoryData', 16: 'MultinomialLogisticLoss',
            34: 'MVN', 17: 'Pooling', 26: 'Power', 18: 'ReLU', 19: 'Sigmoid',
            27: 'SigmoidCrossEntropyLoss', 36: 'Silence', 20: 'Softmax', 21: 'SoftmaxLoss',
            22: 'Split', 33: 'Slice', 23: 'Tanh', 24: 'WindowData', 31: 'Threshold', 32: 'Relu6',
        }
        # yapf: enable

    def __del__(self):
        if self.mlir != None:
            del self.mlir
            self.mlir = None

    def addOperand(self, name, op):
        self.operands[name] = op

    def init_shapes(self, input_shapes: list):
        num_input = len(self.input_names)
        if len(input_shapes) > 0:
            assert (num_input == len(input_shapes))
            for name, shape in zip(self.input_names, input_shapes):
                self.blobs[name].reshape(*shape)
        self.net.reshape()
        for name, blob in self.blobs.items():
            self.addShape(name, list(blob.shape))

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

    def layerType(self, layer):
        if type(layer.type) == int:
            return self.layer_type.get(layer.type)
        else:
            return layer.type

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

        for layer in self.layers:
            if not len(self.select_outputs):
                break
            is_test_phase = True
            if len(layer.include) != 0:
                # only test phase convert
                is_test_phase = False
                for include in layer.include:
                    if include.HasField('phase') and include.phase == 1:
                        is_test_phase = True
                        break
            if not is_test_phase:
                continue
            self.caffeop_factory.get(self.layerType(layer), lambda x: NoneAndRaise(x))(layer)
            for out in layer.top:
                if out in self.select_outputs:
                    self.select_outputs.remove(out)
                    self.output_names.append(out)

        # add return op
        return_op = list()
        # Set output
        for name in self.output_names:
            op = self.getOperand(name)
            return_op.append(op)

        self.mlir.create_return_op(return_op)
        mlir_txt = self.mlir.print_module()
        with open(mlir_file, "w") as f:
            f.write(mlir_txt)
        self.WeightToNpz(self.weight_file)
        print("Save mlir file: {}".format(mlir_file))

    def blob_to_weight_op(self, layer, index):
        name = layer.name + "_{}".format(index)
        blob = self.layer_dict[layer.name].blobs[index]
        return self.create_weight_op(name, blob.data)

    def create_weight_op(self, name, data):
        self.addWeight(name, data)
        return self.getWeightOp(name)

    def convert_convolution_op(self, layer):
        layer_type = self.layerType(layer)
        assert (layer_type == "Convolution" or layer_type == "ConvolutionDepthwise")
        input_shape = self.getShape(layer.bottom[0])
        p = layer.convolution_param
        oc = p.num_output
        dim = len(input_shape) - 2
        g = 1
        if layer_type == "ConvolutionDepthwise":
            g = oc
        else:
            g = p.group
        kernel = [0, 0]
        if len(p.kernel_size) != 0:
            kernel[0] = p.kernel_size[1] if len(p.kernel_size) > 1 else p.kernel_size[0]
            kernel[1] = p.kernel_size[0]
        if p.HasField('kernel_h'):
            kernel[0] = p.kernel_h
        if p.HasField('kernel_w'):
            kernel[1] = p.kernel_w
        stride = [1, 1]
        if len(p.stride) != 0:
            stride[0] = p.stride[1] if len(p.stride) > 1 else p.stride[0]
            stride[1] = p.stride[0]
        if p.HasField('stride_h'):
            stride[0] = p.stride_h
        if p.HasField('stride_w'):
            stride[1] = p.stride_w
        padding = [0, 0]
        if len(p.pad) != 0:
            padding[0] = p.pad[1] if len(p.pad) > 1 else p.pad[0]
            padding[1] = p.pad[0]
        if p.HasField('pad_h'):
            padding[0] = p.pad_h
        if p.HasField('pad_w'):
            padding[1] = p.pad_w
        dilation = [1, 1]
        if len(p.dilation) != 0:
            dilation[0] = p.dilation[1] if len(p.dilation) > 1 else p.dilation[0]
            dilation[1] = p.dilation[0]
        in_op = self.getOperand(layer.bottom[0])
        filter_op = self.blob_to_weight_op(layer, 0)
        bias_op = self.mlir.none_op
        if p.bias_term:
            bias_op = self.blob_to_weight_op(layer, 1)
        attrs = {
            'name': layer.name,
            'kernel_shape': kernel,
            'strides': stride,
            'dilations': dilation,
            'pads': padding * dim,
            'group': g,
            'do_relu': False,
            'ins': [],
        }
        output_shape = self.getShape(layer.top[0])
        new_op = self.mlir.create_conv_op([in_op, filter_op, bias_op], output_shape, **attrs)
        self.addOperand(layer.top[0], new_op)

    def convert_batchnorm_op(self, layer):
        assert (self.layerType(layer) == "BatchNorm")
        eps = 1e-5
        if layer.HasField('batch_norm_param'):
            if layer.batch_norm_param.HasField('eps'):
                eps = layer.batch_norm_param.eps
            if layer.batch_norm_param.HasField('use_global_stats'):
                assert (layer.batch_norm_param.use_global_stats == True)
        in_op = self.getOperand(layer.bottom[0])
        blobs = self.layer_dict[layer.name].blobs
        mean = np.array(blobs[0].data)
        variance = np.array(blobs[1].data)
        scale = blobs[2].data
        mean /= scale
        variance /= scale
        mean_op = self.create_weight_op(layer.name + "_mean", mean)
        var_op = self.create_weight_op(layer.name + "_var", variance)
        output_shape = self.getShape(layer.top[0])
        attrs = {
            "name": layer.name,
            "epsilon": eps,
        }
        gamma_op = self.mlir.none_op
        beta_op = self.mlir.none_op
        new_op = self.mlir.create_batchnorm_op([in_op, mean_op, var_op, gamma_op, beta_op],
                                               output_shape, **attrs)
        self.addOperand(layer.top[0], new_op)

    def convert_scale_op(self, layer):
        assert (self.layerType(layer) == "Scale")
        in_op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        num_dims = len(input_shape)
        assert (num_dims == 4 or num_dims == 2)
        if num_dims == 2:
            raise RuntimeError("Not support now, shape {}".format(input_shape))
        output_shape = input_shape
        attrs = {"name": layer.name}
        scale_op = self.blob_to_weight_op(layer, 0)
        bias_op = self.mlir.none_op
        if layer.scale_param.bias_term:
            bias_op = self.blob_to_weight_op(layer, 1)
        new_op = self.mlir.create_scale_op([in_op, scale_op, bias_op], output_shape, **attrs)
        self.addOperand(layer.top[0], new_op)

    def convert_relu_op(self, layer):
        assert (layer.type == "ReLU")
        op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        output_shape = input_shape
        attrs = {'name': layer.name}
        new_op = self.mlir.create_relu_op([op], output_shape, **attrs)
        self.addOperand(layer.top[0], new_op)

    def convert_pooling_op(self, layer):
        assert (layer.type == "Pooling")
        input_shape = self.getShape(layer.bottom[0])
        op = self.getOperand(layer.bottom[0])
        p = layer.pooling_param
        method = p.pool
        kernel = [input_shape[2], input_shape[3]]
        if not p.global_pooling:
            kernel[0] = p.kernel_h if p.HasField('kernel_h') else p.kernel_size
            kernel[1] = p.kernel_w if p.HasField('kernel_w') else p.kernel_size
        stride = [p.stride, p.stride]
        if p.HasField('stride_h'):
            stride[0] = p.stride_h
        if p.HasField('stride_w'):
            stride[1] = p.stride_w
        padding = [p.pad, p.pad]
        if p.HasField('pad_h'):
            padding[0] = p.pad_h
        if p.HasField('pad_w'):
            padding[1] = p.pad_w
        pads = [0, 0, padding[0], padding[1]]
        output_shape = self.getShape(layer.top[0])
        attrs = {
            'name': layer.name,
            'kernel_shape': kernel,
            'strides': stride,
            'pads': pads,
            'count_include_pad': True,
            'do_relu': False,
        }
        if method == 0:  # MAX
            new_op = self.mlir.create_maxpool_op([op], output_shape, **attrs)
        elif method == 1:  # AVE
            new_op = self.mlir.create_avgpool_op([op], output_shape, **attrs)
        else:
            raise RuntimeError("Method {} not support".format(method))
        self.addOperand(layer.top[0], new_op)

    def convert_eltwise_op(self, layer):
        assert (layer.type == "Eltwise")
        operands = list()
        for bottom in layer.bottom:
            op = self.getOperand(bottom)
            operands.append(op)
        num_input = len(layer.bottom)
        p = layer.eltwise_param
        attrs = {"name": layer.name}
        output_shape = self.getShape(layer.top[0])
        if p.operation == 0:  # mul
            new_op = self.mlir.create_mul_op(operands, output_shape, **attrs)
        elif p.operation == 1:  # add
            coeff = None
            if (len(p.coeff) != 0):
                assert (len(p.coeff) == num_input)
                coeff = [c for c in p.coeff]
            else:
                coeff = [1] * num_input
            attrs['coeff'] = coeff
            new_op = self.mlir.create_add_op(operands, output_shape, **attrs)
        elif p.operation == 2:  # max
            raise RuntimeError("Max not support now")
        elif p.operation == 3:  # min
            raise RuntimeError("Min not support now")
        self.addOperand(layer.top[0], new_op)

    def convert_inner_product_op(self, layer):
        #(M, K) * (K, N) => (M, N)
        assert (self.layerType(layer) == 'InnerProduct')
        in_op = self.getOperand(layer.bottom[0])
        p = layer.inner_product_param
        with_bias = p.bias_term
        if p.transpose:
            filter_op = self.blob_to_weight_op(layer, 0)
        else:
            raise RuntimeError("Not support now")
        attrs = {"name": layer.name}
        bias_op = self.mlir.none_op
        if with_bias:
            bias_op = self.blob_to_weight_op(layer, 1)
        output_shape = self.getShape(layer.top[0])
        new_op = self.mlir.create_matmul_op([in_op, filter_op, bias_op], output_shape, **attrs)
        self.addOperand(layer.top[0], new_op)

    def convert_softmax_op(self, layer):
        assert (self.layerType(layer) == 'Softmax')
        in_op = self.getOperand(layer.bottom[0])
        axis = 1
        if layer.HasField('softmax_param') and layer.softmax_param.HasField('axis'):
            axis = layer.softmax_param.axis
        attrs = {'name': layer.name, 'axis': axis}
        output_shape = self.getShape(layer.top[0])
        new_op = self.mlir.create_softmax_op([in_op], output_shape, **attrs)
        self.addOperand(layer.top[0], new_op)
