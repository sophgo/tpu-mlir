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
from utils.pad_setting import set_caffe_pad


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
            #pls add the Op alphabetically
            'BatchNorm': lambda layer: self.convert_batchnorm_op(layer),
            'BN': lambda layer: self.convert_bn_op(layer),
            'Concat': lambda layer: self.convert_concat_op(layer),
            'ContinuationIndicator': lambda layer: self.convert_continuation_indicator_op(layer),
            'Convolution': lambda layer: self.convert_convolution_op(layer),
            'ConvolutionDepthwise': lambda layer: self.convert_convolution_op(layer),
            'Crop': lambda layer: self.convert_crop_op(layer),
            'Deconvolution': lambda layer: self.convert_deconvolution_op(layer),
            'DetectionOutput': lambda layer: self.convert_detection_output_op(layer),
            'Dropout': lambda layer: self.convert_dropout_op(layer),
            'DummyData': lambda layer: self.convert_dummydata_op(layer),
            'Embed': lambda layer: self.convert_embed_op(layer),
            'Eltwise': lambda layer: self.convert_eltwise_op(layer),
            'Flatten': lambda layer: self.convert_flatten_op(layer),
            'FrcnDetection': lambda layer: self.convert_frcn_detection_op(layer),
            'InnerProduct': lambda layer: self.convert_inner_product_op(layer),
            'Input': lambda layer: self.convert_input_op(layer),
            'Interp': lambda layer: self.convert_interp_op(layer),
            'LRN': lambda layer: self.convert_lrn_op(layer),
            'LSTM': lambda layer: self.convert_lstm_op(layer),
            'Lstm': lambda layer: self.convert_lstm_jun_op(layer),
            'Normalize': lambda layer: self.convert_normalize_op(layer),
            'Mish': lambda layer: self.convert_mish_op(layer),
            'Padding': lambda layer: self.convert_padding_op(layer),
            'Permute': lambda layer: self.convert_permute_op(layer),
            'Pooling': lambda layer: self.convert_pooling_op(layer),
            'Power': lambda layer: self.convert_power_op(layer),
            'PReLU': lambda layer: self.convert_prelu_op(layer),
            'PriorBox': lambda layer: self.convert_priorbox_op(layer),
            'Proposal': lambda layer: self.convert_proposal_op(layer),
            'ReLU': lambda layer: self.convert_relu_op(layer),
            'ReLU6': lambda layer: self.convert_relu6_op(layer),
            'Reorg': lambda layer: self.convert_reorg_op(layer),
            'Reshape': lambda layer: self.convert_reshape_op(layer),
            'Reverse': lambda layer: self.convert_reverse_op(layer),
            'RetinaFaceDetection': lambda layer: self.convert_retinaface_detection_op(layer),
            'ROIPooling': lambda layer: self.convert_roipooling_op(layer),
            'Scale': lambda layer: self.convert_scale_op(layer),
            'ShuffleChannel': lambda layer: self.convert_shufflechannel_op(layer),
            'Sigmoid': lambda layer: self.convert_sigmoid_op(layer),
            'Silence': lambda layer: self.convert_silence_op(layer),
            'Slice': lambda layer: self.convert_slice_op(layer),
            'Softmax': lambda layer: self.convert_softmax_op(layer),
            'Split': lambda layer: self.convert_split_op(layer),
            'TanH': lambda layer: self.convert_tanh_op(layer),
            'Tile': lambda layer: self.convert_tile_op(layer),
            'Upsample': lambda layer: self.convert_upsample_op(layer),
            'YoloDetection': lambda layer: self.convert_yolo_detection_op(layer),
            'MatMul': lambda layer: self.convert_matmul_op(layer),
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
        output_shape = input_shape
        attrs = {"name": layer.name}
        assert (num_dims == 4 or num_dims == 2)
        if num_dims == 2:
            raise RuntimeError("Not support now, shape {}".format(input_shape))
        if len(layer.bottom) == 2:
            op1 = self.getOperand(layer.bottom[1])
            input_shape1 = self.getShape(layer.bottom[1])
            if len(input_shape1) < len(input_shape):
                output_shape1 = list(input_shape1) + [1] * (len(input_shape) - len(input_shape1))
                op1 = self.mlir.create_reshape_op([op1], output_shape1, **{'name': layer.bottom[1] + "_reshape"})
            new_op = self.mlir.create_mul_op([in_op, op1], output_shape, **attrs)
            self.addOperand(layer.top[0], new_op)
        else:
            scale_op = self.blob_to_weight_op(layer, 0)
            if layer.scale_param.bias_term:
                bias_op = self.blob_to_weight_op(layer, 1)
            else:
                bias_op = self.create_weight_op(
                    layer.name + "_1", np.zeros(self.getShape(layer.bottom[1]), np.float32))
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
        output_shape = self.getShape(layer.top[0])
        op = self.getOperand(layer.bottom[0])
        p = layer.pooling_param
        method = p.pool
        imap_shape = [input_shape[2], input_shape[3]]
        kernel_shape = list(imap_shape)
        is_global = p.global_pooling
        if not is_global:
            kernel_shape[0] = p.kernel_h if p.HasField('kernel_h') else p.kernel_size
            kernel_shape[1] = p.kernel_w if p.HasField('kernel_w') else p.kernel_size
            if kernel_shape == imap_shape:
                is_global = True
        if not is_global:
            strides = [p.stride, p.stride]
            if p.HasField('stride_h'):
                strides[0] = p.stride_h
            if p.HasField('stride_w'):
                strides[1] = p.stride_w
            pads = [p.pad] * 4 if p.HasField('pad') else set_caffe_pad(
                input_shape, output_shape, kernel_shape, strides)
        else:
            pads = [0] * 4
            strides = [1] * 2
        attrs = {
            'name': layer.name,
            'kernel_shape': kernel_shape,
            'strides': strides,
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
        else:  # do transpose
            filter = self.layer_dict[layer.name].blobs[0].data
            new_filter = np.ascontiguousarray(np.transpose(filter, (1, 0)))
            filter_op = self.create_weight_op(layer.name + "_filter", new_filter)
        attrs = {"name": layer.name, "do_relu": False}
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

    def convert_bn_op(self, layer):
        assert (self.layerType(layer) == 'BN')
        in_op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        operands = list()
        operands.append(in_op)

        p = layer.bn_param
        bn_mode = 0
        if hasattr(p, 'bn_mode'):
            bn_mode = p.bn_mode

        attrs = {'variance_epsilon': 1e-5, 'frozen': False, 'name': layer.name}

        if layer.HasField('bn_param'):
            if layer.bn_param.HasField('eps'):
                attrs['variance_epsilon'] = layer.bn_param.eps

            if layer.bn_param.HasField('frozen'):
                attrs['frozen'] = layer.bn_param.frozen
                assert (attrs['frozen'] == True and "only support frozen = false now")

        blobs = self.layer_dict[layer.name].blobs

        for idx, blob in enumerate(blobs):
            blob_op = self.blob_to_weight_op(layer, idx)
            operands.append(blob_op)

        output_shape = input_shape
        if bn_mode == 1:
            new_op = self.mlir.create_scale_op(operands, output_shape, **attrs)
            self.addOperand(layer.top[0], new_op)
        else:
            new_op = self.mlir.create_batchnorm_op(operands, output_shape, **attrs)
            self.addOperand(layer.top[0], new_op)

    def convert_concat_op(self, layer):
        assert (self.layerType(layer) == 'Concat')
        in_op = self.getOperand(layer.bottom[0])
        if len(layer.bottom) == 1:
            return self.addOperand(layer.top[0], in_op)
        axis = layer.concat_param.axis
        input_shape = self.getShape(layer.bottom[0])
        assert (axis < len(input_shape))
        concat_axis_dim = 0
        operands = list()
        for bottom in layer.bottom:
            bottom_op = self.getOperand(bottom)
            shape = self.getShape(bottom)
            assert (len(shape) == len(input_shape))
            concat_axis_dim += shape[axis]
            operands.append(bottom_op)
        output_shape = list(input_shape)
        output_shape[axis] = concat_axis_dim
        attrs = {'axis': axis, 'name': layer.name}
        new_op = self.mlir.create_concat_op(operands, output_shape, **attrs)
        self.addOperand(layer.top[0], new_op)

    def convert_continuation_indicator_op(self, layer):
        assert (self.layerType(layer) == 'ContinuationIndicator')
        raise RuntimeError("not implemented")

    def convert_crop_op(self, layer):
        assert (self.layerType(layer) == 'Crop')
        raise RuntimeError("not implemented")

    def convert_deconvolution_op(self, layer):
        assert (self.layerType(layer) == "Deconvolution")
        raise RuntimeError("not implemented")

    def convert_detection_output_op(self, layer):
        assert (self.layerType(layer) == "DetectionOutput")
        raise RuntimeError("not implemented")

    def convert_dropout_op(self, layer):
        assert (self.layerType(layer) == 'Dropout')
        op = self.getOperand(layer.bottom[0])
        self.addOperand(layer.top[0], op)

    def convert_dummydata_op(self, layer):
        assert (self.layerType(layer) == 'DummyData')
        # do nothing

    def convert_embed_op(self, layer):
        assert (self.layerType(layer) == 'Embed')
        raise RuntimeError("not implemented")

    def convert_flatten_op(self, layer):
        assert (self.layerType(layer) == 'Flatten')
        raise RuntimeError("not implemented")

    def convert_frcn_detection_op(self, layer):
        assert (self.layerType(layer) == 'FrcnDetection')
        raise RuntimeError("not implemented")

    def convert_input_op(self, layer):
        assert (self.layerType(layer) == 'Input')
        # do nothing

    def convert_interp_op(self, layer):
        assert (self.layerType(layer) == 'Interp')
        raise RuntimeError("not implemented")

    def convert_lrn_op(self, layer):
        assert (self.layerType(layer) == 'LRN')
        raise RuntimeError("not implemented")

    def convert_lstm_op(self, layer):
        assert (self.layerType(layer) == 'LSTM')
        raise RuntimeError("not implemented")

    def convert_lstm_jun_op(self, layer):
        assert (self.layerType(layer) == 'Lstm')
        raise RuntimeError("not implemented")

    def convert_matmul_op(self, layer):
        assert (self.layerType(layer) == 'MatMul')
        raise RuntimeError("not implemented")

    def convert_normalize_op(self, layer):
        assert (self.layerType(layer) == 'Normalize')
        raise RuntimeError("not implemented")

    def convert_mish_op(self, layer):
        assert (self.layerType(layer) == 'Mish')
        raise RuntimeError("not implemented")

    def convert_padding_op(self, layer):
        assert (self.layerType(layer) == 'Padding')
        raise RuntimeError("not implemented")

    def convert_permute_op(self, layer):
        assert (self.layerType(layer) == 'Permute')
        raise RuntimeError("not implemented")

    def convert_power_op(self, layer):
        assert (self.layerType(layer) == 'Power')
        raise RuntimeError("not implemented")

    def convert_prelu_op(self, layer):
        assert (self.layerType(layer) == 'PReLU')
        raise RuntimeError("not implemented")

    def convert_priorbox_op(self, layer):
        assert (self.layerType(layer) == 'PriorBox')
        raise RuntimeError("not implemented")

    def convert_proposal_op(self, layer):
        assert (self.layerType(layer) == 'Proposal')
        raise RuntimeError("not implemented")

    def convert_relu6_op(self, layer):
        assert (self.layerType(layer) == 'ReLU6')
        raise RuntimeError("not implemented")

    def convert_reorg_op(self, layer):
        assert (self.layerType(layer) == 'Reorg')
        in_op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        assert (len(input_shape) == 4)
        reverse = layer.reorg_param.reverse
        assert (reverse == False)
        stride = layer.reorg_param.stride
        output_shape = list(input_shape)
        output_shape[0] = input_shape[0]
        output_shape[1] = int(input_shape[1] * stride * stride)
        output_shape[2] = int(input_shape[2] / stride)
        output_shape[3] = int(input_shape[3] / stride)
        attrs = {'name': layer.name}
        new_op = self.mlir.create_reshape_op([in_op], output_shape, **attrs)
        self.addOperand(layer.top[0], new_op)

    def convert_reshape_op(self, layer):
        assert (self.layerType(layer) == 'Reshape')
        raise RuntimeError("not implemented")

    def convert_reverse_op(self, layer):
        assert (self.layerType(layer) == 'Reverse')
        raise RuntimeError("not implemented")

    def convert_retinaface_detection_op(self, layer):
        assert (self.layerType(layer) == 'RetinaFaceDetection')
        raise RuntimeError("not implemented")

    def convert_roipooling_op(self, layer):
        assert (self.layerType(layer) == 'ROIPooling')
        raise RuntimeError("not implemented")

    def convert_shufflechannel_op(self, layer):
        assert (self.layerType(layer) == 'ShuffleChannel')
        raise RuntimeError("not implemented")

    def convert_sigmoid_op(self, layer):
        assert (self.layerType(layer) == 'Sigmoid')
        in_op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        output_shape = input_shape
        attrs = {'scale': 1, 'bias': 0, 'name': layer.name}
        new_op = self.mlir.create_sigmoid_op([in_op], output_shape, **attrs)
        self.addOperand(layer.top[0], new_op)

    def convert_silence_op(self, layer):
        assert (self.layerType(layer) == 'Silence')
        # do nothing now

    def convert_slice_op(self, layer):
        assert (self.layerType(layer) == 'Slice')
        raise RuntimeError("not implemented")

    def convert_split_op(self, layer):
        assert (self.layerType(layer) == 'Split')
        raise RuntimeError("not implemented")

    def convert_tanh_op(self, layer):
        assert (self.layerType(layer) == 'TanH')
        raise RuntimeError("not implemented")

    def convert_tile_op(self, layer):
        assert (self.layerType(layer) == 'Tile')
        raise RuntimeError("not implemented")

    def convert_upsample_op(self, layer):
        assert (self.layerType(layer) == 'Upsample')
        in_op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        assert (len(input_shape) == 4)
        p = layer.upsample_param
        scale = p.scale
        output_shape = [
            input_shape[0], input_shape[1], scale * input_shape[2], scale * input_shape[3]
        ]
        if p.HasField('upsample_w'):
            output_shape[3] = p.upsample_w
        if p.HasField('upsample_h'):
            output_shape[2] = p.upsample_h
        attrs = {'scale_h': scale, 'scale_w': scale, 'name': layer.name}

        new_op = self.mlir.create_upsample_op([in_op], output_shape, **attrs)
        self.addOperand(layer.top[0], new_op)

    def convert_yolo_detection_op(self, layer):
        assert (self.layerType(layer) == 'YoloDetection')
        raise RuntimeError("not implemented")
