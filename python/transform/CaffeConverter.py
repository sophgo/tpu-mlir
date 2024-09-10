# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .MLIRImporter import MLIRImporter, Platform
from .BaseConverter import BaseConverter
import numpy as np

import math
import caffe
import torch
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from utils.pad_setting import set_caffe_pad
import mlir.dialects.top as top
from mlir.ir import *
import logging
import copy

logger = logging.getLogger("root")

class CaffeConverter(BaseConverter):

    def __init__(self,
                 model_name: str,
                 prototxt: str,
                 caffemodel: str,
                 input_shapes: list,
                 output_names: list,
                 preprocess_args: dict = {},
                 shape_influencing_input_names: list = [],
                 no_save: bool = False):
        super().__init__(no_save=no_save)
        # yapf: disable
        # for caffe v1
        self.layer_type = {
            0: 'None', 35: 'Absval', 1: 'Accuracy', 30: 'ArgMax', 2: 'Bnll',
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
        self.shape_influencing_input_names = shape_influencing_input_names
        self.model_name = model_name
        self.prototxt = prototxt
        self.caffemodel = caffemodel
        self.net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)
        self.param = caffe_pb2.NetParameter()
        text_format.Merge(open(prototxt).read(), self.param)
        self.layers = self.param.layer if len(self.param.layer) != 0 else self.param.layers
        self.input_names = self.net.inputs
        self.origin_output_names = output_names
        self.select_outputs = list(output_names) if output_names else list(self.net.outputs)
        self.blobs = self.net.blobs
        self.mlir = None
        self.layer_dict = self.net.layer_dict
        self.weight_file = "" if self.no_save else "{}_top_origin_weight.npz".format(model_name)
        self.init_shapes(input_shapes)
        self.init_MLIRImporter()
        self.location = self.resolve_alias()
        self.preprocess_args = {}
        if 'preprocess_list' in preprocess_args:
            if preprocess_args['preprocess_list'] is not None:
                for input_index in preprocess_args['preprocess_list']:
                    assert( 0 < input_index <= len(self.input_names)
                        and "Please check --preprocess_list is right input")
            else:
                preprocess_args['preprocess_list'] = [ i + 1 for i in range(len(self.input_names)) ]
        if 'channel_format' in preprocess_args:
            if preprocess_args['channel_format'] != "none":
                self.preprocess_args = preprocess_args
        self.caffeop_factory = {
            #pls add the Op alphabetically
            "AbsVal": lambda layer: self.convert_absval_op(layer),
            'ArgMax': lambda layer: self.convert_argmax_op(layer),
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
            'DummyData': lambda layer: self.convert_nothing(layer),
            'Embed': lambda layer: self.convert_embed_op(layer),
            'Eltwise': lambda layer: self.convert_eltwise_op(layer),
            'Flatten': lambda layer: self.convert_flatten_op(layer),
            'FrcnDetection': lambda layer: self.convert_frcn_detection_op(layer),
            'InnerProduct': lambda layer: self.convert_inner_product_op(layer),
            'Input': lambda layer: self.convert_nothing(layer),
            'Interp': lambda layer: self.convert_interp_op(layer),
            'ImageData': lambda layer: self.convert_nothing(layer),
            'LRN': lambda layer: self.convert_lrn_op(layer),
            'LSTM': lambda layer: self.convert_lstm_op(layer),
            'Lstm': lambda layer: self.convert_lstm_jun_op(layer),
            'MatMul': lambda layer: self.convert_matmul_op(layer),
            'Mish': lambda layer: self.convert_mish_op(layer),
            'Normalize': lambda layer: self.convert_normalize_op(layer),
            'Padding': lambda layer: self.convert_padding_op(layer),
            'Permute': lambda layer: self.convert_permute_op(layer),
            'Pooling': lambda layer: self.convert_pooling_op(layer),
            'Power': lambda layer: self.convert_power_op(layer),
            'PReLU': lambda layer: self.convert_prelu_op(layer),
            'PriorBox': lambda layer: self.convert_priorbox_op(layer),
            'Proposal': lambda layer: self.convert_proposal_op(layer),
            'Reduction': lambda layer: self.convert_reduce_op(layer),
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
            'Silence': lambda layer: self.convert_nothing(layer),
            'Slice': lambda layer: self.convert_slice_op(layer),
            'Softmax': lambda layer: self.convert_softmax_op(layer),
            'Split': lambda layer: self.convert_split_op(layer),
            'TanH': lambda layer: self.convert_tanh_op(layer),
            'Tile': lambda layer: self.convert_tile_op(layer),
            'Upsample': lambda layer: self.convert_upsample_op(layer),
            'YoloDetection': lambda layer: self.convert_yolo_detection_op(layer),

        }

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

    def resolve_alias(self):
        location = {}
        for layer in self.layers:
            if layer.type == "Dropout":
                continue
            for top in layer.top:
                if top in location:
                    location[top].append(f"{top}/{len(location[top])}")
                else:
                    location[top] = [top]
        return location

    def get_loc_name(self, name):
        if name in self.location:
            return self.location[name].pop()
        return name

    def get_loc(self, names):
        if isinstance(names, str):
            n = self.get_loc_name(names)
            return Location.fused([Location.name(n)], context=self.mlir.ctx)
        elif isinstance(names, list):
            return Location.fused([Location.name(self.get_loc_name(n)) for n in names],
                                  context=self.mlir.ctx)
        else:
            raise RuntimeError("Unknown names:{}".format(names))

    def init_MLIRImporter(self):
        input_shapes = list()
        for _name in self.input_names:
            input_shapes.append(self.getShape(_name))
        output_shapes = list()
        for _name in self.select_outputs:
            o_shape = self.getShape(_name)
            for layer in self.layers:
                if layer.name == _name and self.layerType(layer) == "DetectionOutput":
                    o_shape[2] = layer.detection_output_param.keep_top_k
                    break
            output_shapes.append(self.getShape(_name))
        # init importer
        self.mlir = MLIRImporter(input_shapes, output_shapes, self.model_name, platform=Platform.CAFFE, no_save=self.no_save)
        self.weight_file = self.mlir.weight_file

    def layerType(self, layer):
        if type(layer.type) == int:
            return self.layer_type.get(layer.type)
        else:
            return layer.type

    def convert_subgraph(self):
        # add input op
        input_data = None
        if self.shape_influencing_input_names:
            test_file = ''
            if isinstance(self.test_input, list):
                assert self.test_input[0].endswith('.npz')
                test_file = self.test_input[0]
            elif isinstance(self.test_input, str):
                assert self.test_input.endswith('.npz')
                test_file = self.test_input
            else:
                raise ValueError("test_input npz file is necessary when shape_influencing_input_names is set")
            input_data = np.load(test_file)
        for idx, _name in enumerate(self.input_names):
            kwargs = copy.deepcopy(self.preprocess_args)
            if _name in self.shape_influencing_input_names:
                assert input_data[_name].ndim == 1, "input shape tensor should be 1D tensor"
                kwargs['shape_tensor'] = input_data[_name]
            input_ = self.mlir.create_input_op(self.get_loc(_name), idx, kwargs)
            self.addOperand(_name, input_)

        def NoneAndRaise(layer):
            raise RuntimeError("{} Op not support now".format(layer.type))

        for layer in self.layers:
            if len(self.select_outputs) == len(self.output_names):
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
                    self.output_names.append(out)
        # add return op
        return_op = list()
        # Set output
        final_output_names = []
        if self.origin_output_names:
            final_output_names = self.origin_output_names
        else:
            final_output_names = self.select_outputs
        for name in final_output_names:
            op = self.getOperand(name)
            return_op.append(op)
        return return_op

    def get_mlir_txt(self):
        return_op = self.convert_subgraph()
        self.mlir.create_return_op(return_op)
        return self.mlir.print_module()

    def generate_mlir(self, mlir_file: str):
        mlir_txt = self.get_mlir_txt()
        with open(mlir_file, "w") as f:
            f.write(mlir_txt)
        logger.info("Save mlir file: {}".format(mlir_file))
        self.WeightToNpz(self.weight_file)
        logger.info("Save weight file: {}".format(self.weight_file))

    def blob_to_weight_op(self, layer, index, shape: list = [], permute_order=[]):
        name = layer.name + "_weight_{}".format(index)
        blob = self.layer_dict[layer.name].blobs[index]
        if permute_order != None and len(permute_order) != 0:
            value = blob.data
            value_tensor = torch.tensor(value)
            value_tensor = value_tensor.permute(permute_order)
            value = np.array(value_tensor)
            data = value
        else:
            data = blob.data
        if shape:
            assert (np.prod(data.shape) == np.prod(shape))
            data = data.reshape(shape)
        return self.create_weight_op(name, data)

    def create_weight_op(self, name, data):
        self.addWeight(name, data)
        return self.getWeightOp(name)

    def convert_argmax_op(self, layer):
        layer_type = self.layerType(layer)
        assert (layer_type == "ArgMax")
        input_shape = self.getShape(layer.bottom[0])
        out_shape = self.getShape(layer.top[0])
        in_op = self.getOperand(layer.bottom[0])
        p = layer.argmax_param
        out_max_val = p.out_max_val
        top_k = p.top_k
        name = layer.top[0]
        assert (top_k == 1 and "Only support top_k = 1 for now")
        axis = p.axis
        if axis < 0:
            axis += len(input_shape)
        tmp_shape = input_shape
        tmp_shape[axis] = top_k
        assert (tmp_shape == out_shape and "Must provide axis")
        attrs = {'mode': StringAttr.get(layer_type), 'axis': axis, 'keepdims': True}
        attrs['loc'] = self.get_loc([name +
                                     "_indices", name] if out_max_val else [name, name + "_values"])
        output_shapes = [out_shape]
        output_shapes += [out_shape] if out_max_val else [None]
        out_op = top.ArgOp(*self.mlir.get_tensor_type(output_shapes),
                           in_op,
                           **attrs,
                           ip=self.mlir.insert_point)
        out_ops = [out_op.indices, out_op.values]
        self.addOperand(layer.top[0], out_ops[1] if out_max_val else out_ops[0])

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
            'loc': self.get_loc(layer.top[0]),
            'kernel_shape': kernel,
            'strides': stride,
            'dilations': dilation,
            'pads': padding * dim,
            'group': g,
            'do_relu': False,
            # 'ins': [], # unexpected params ignored
        }
        output_shape = self.getShape(layer.top[0])
        new_op = top.ConvOp(self.mlir.get_tensor_type(output_shape),
                            in_op,
                            filter_op,
                            bias_op,
                            **attrs,
                            ip=self.mlir.insert_point).output
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
            'loc': self.get_loc(layer.top[0]),
            "epsilon": eps,
        }
        gamma_op = self.mlir.none_op
        beta_op = self.mlir.none_op
        new_op = top.BatchNormOp(self.mlir.get_tensor_type(output_shape),
                                 in_op,
                                 mean_op,
                                 var_op,
                                 gamma_op,
                                 beta_op,
                                 **attrs,
                                 ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_scale_op(self, layer):
        assert (self.layerType(layer) == "Scale")
        in_op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        num_dims = len(input_shape)
        output_shape = input_shape
        attrs = {'loc': self.get_loc(layer.top[0])}
        assert (num_dims == 4 or num_dims == 2)
        if len(layer.bottom) == 2:
            op1 = self.getOperand(layer.bottom[1])
            input_shape1 = self.getShape(layer.bottom[1])
            if len(input_shape1) < len(input_shape):
                output_shape1 = list(input_shape1) + [1] * (len(input_shape) - len(input_shape1))
                op1 = top.ReshapeOp(self.mlir.get_tensor_type(output_shape1),
                                    op1,
                                    loc=Location.fused(
                                        [Location.name(layer.bottom[1] + "_reshape")],
                                        context=self.mlir.ctx),
                                    ip=self.mlir.insert_point).output
            new_op = top.MulOp(self.mlir.get_tensor_type(output_shape), [in_op, op1],
                               **attrs,
                               ip=self.mlir.insert_point).output
            self.addOperand(layer.top[0], new_op)
        else:
            scale_op = self.blob_to_weight_op(layer, 0)
            if layer.scale_param.bias_term:
                bias_op = self.blob_to_weight_op(layer, 1)
            else:
                if len(layer.bottom) > 1:
                    bias_op = self.create_weight_op(
                        layer.name + "_1", np.zeros(self.getShape(layer.bottom[1]), np.float32))
                else:
                    bias_op = self.create_weight_op(layer.name + "_1",
                                                    np.zeros(input_shape[1], np.float32))

            new_op = top.ScaleOp(self.mlir.get_tensor_type(output_shape),
                                 in_op,
                                 scale_op,
                                 bias_op,
                                 **attrs,
                                 ip=self.mlir.insert_point).output
            self.addOperand(layer.top[0], new_op)

    def convert_relu_op(self, layer):
        assert (self.layerType(layer) == "ReLU")
        op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        output_shape = input_shape
        attrs = {'loc': self.get_loc(layer.top[0])}
        if layer.relu_param.HasField('negative_slope'):
            attrs['alpha'] = layer.relu_param.negative_slope
            new_op = top.LeakyReluOp(self.mlir.get_tensor_type(output_shape),
                                     op,
                                     **attrs,
                                     ip=self.mlir.insert_point).output
        else:
            new_op = top.ReluOp(self.mlir.get_tensor_type(output_shape),
                                op,
                                **attrs,
                                ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_pooling_op(self, layer):
        assert (self.layerType(layer) == "Pooling")
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
            if p.HasField("pad"):
                leading_pad = [p.pad, p.pad]
            else:
                leading_pad = [p.pad_h, p.pad_w]
            pads = set_caffe_pad(input_shape, output_shape, kernel_shape, strides, leading_pad)
        else:
            pads = [0] * 4
            strides = [1] * 2
        attrs = {
            'kernel_shape': kernel_shape,
            'strides': strides,
            'pads': pads,
            'count_include_pad': True,
            'do_relu': False,
        }
        if len(layer.top) == 1:
            attrs["loc"] = self.get_loc(layer.top[0])
        else:
            attrs["loc"] = self.get_loc([x for x in layer.top])

        if method == 0:  # MAX
            if len(layer.top) == 1:
                new_op = top.MaxPoolOp(self.mlir.get_tensor_type(output_shape),
                                       op,
                                       **attrs,
                                       ip=self.mlir.insert_point).output
                self.addOperand(layer.top[0], new_op)
                return
            else:
                output_shape2 = self.getShape(layer.top[1])
                maskp_op = top.MaxPoolWithMaskOp(*self.mlir.get_tensor_type(
                    [output_shape, output_shape2]),
                                                 op,
                                                 **attrs,
                                                 ip=self.mlir.insert_point)
                new_op, mask_op = maskp_op.output, maskp_op.mask
                self.addOperand(layer.top[0], new_op)
                self.addOperand(layer.top[1], mask_op)
                return
        elif method == 1:  # AVE
            attrs['keepdims'] = len(output_shape) == len(input_shape)
            new_op = top.AvgPoolOp(self.mlir.get_tensor_type(output_shape),
                                   op,
                                   **attrs,
                                   ip=self.mlir.insert_point).output
            self.addOperand(layer.top[0], new_op)
            return
        else:
            raise RuntimeError("Method {} not support".format(method))

    def convert_eltwise_op(self, layer):
        assert (self.layerType(layer) == "Eltwise")
        operands = list()
        for bottom in layer.bottom:
            op = self.getOperand(bottom)
            operands.append(op)
        num_input = len(layer.bottom)
        p = layer.eltwise_param
        attrs = {'loc': self.get_loc(layer.top[0])}
        output_shape = self.getShape(layer.top[0])
        if p.operation == 0:  # mul
            new_op = top.MulOp(self.mlir.get_tensor_type(output_shape),
                               operands,
                               **attrs,
                               ip=self.mlir.insert_point).output
        elif p.operation == 1:  # add
            coeff = None
            if (len(p.coeff) != 0):
                assert (len(p.coeff) == num_input)
                coeff = [c for c in p.coeff]
            else:
                coeff = [1] * num_input
            attrs['coeff'] = coeff
            new_op = top.AddOp(self.mlir.get_tensor_type(output_shape),
                               operands,
                               **attrs,
                               ip=self.mlir.insert_point).output
        elif p.operation == 2:  # max
            new_op = top.MaxOp(self.mlir.get_tensor_type(output_shape),
                               operands,
                               **attrs,
                               ip=self.mlir.insert_point).output
        elif p.operation == 3:  # min
            new_op = top.MinOp(self.mlir.get_tensor_type(output_shape),
                               operands,
                               **attrs,
                               ip=self.mlir.insert_point).output
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
        attrs = {'loc': self.get_loc(layer.top[0]), "do_relu": False}
        bias_op = self.mlir.none_op
        if with_bias:
            bias_op = self.blob_to_weight_op(layer, 1)
        output_shape = self.getShape(layer.top[0])
        # reshape
        ori_in_shape = self.getShape(layer.bottom[0])
        in_shape = [ori_in_shape[0], np.prod(ori_in_shape[1:])]
        reshape_attrs = {'loc': self.get_loc(layer.bottom[0] + "_reshape")}
        output_shape = self.getShape(layer.top[0])

        reshape_op = top.ReshapeOp(self.mlir.get_tensor_type([in_shape[0], in_shape[-1]]),
                                       in_op,
                                       **reshape_attrs,
                                       ip=self.mlir.insert_point).output

        new_op = top.MatMulOp(self.mlir.get_tensor_type(output_shape),
                              reshape_op,
                              filter_op,
                              bias_op,
                              **attrs,
                              ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_softmax_op(self, layer):
        assert (self.layerType(layer) == 'Softmax')
        in_op = self.getOperand(layer.bottom[0])
        output_shape = self.getShape(layer.top[0])
        axis = 1
        if layer.HasField('softmax_param') and layer.softmax_param.HasField('axis'):
            axis = layer.softmax_param.axis
        if axis < 0:
            axis += len(output_shape)
        attrs = {'loc': self.get_loc(layer.top[0]), 'axis': axis}
        new_op = top.SoftmaxOp(self.mlir.get_tensor_type(output_shape),
                               in_op,
                               **attrs,
                               ip=self.mlir.insert_point).output
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

        attrs = {
            'variance_epsilon': 1e-5,
            'epsilon': 1e-5,
            'frozen': False,
            'loc': self.get_loc(layer.top[0])
        }

        if layer.HasField('bn_param'):
            if layer.bn_param.HasField('eps'):
                attrs['variance_epsilon'] = layer.bn_param.eps
                attrs['epsilon'] = layer.bn_param.eps

            if layer.bn_param.HasField('frozen'):
                attrs['frozen'] = layer.bn_param.frozen
                assert (attrs['frozen'] == True and "only support frozen = false now")

        blobs = self.layer_dict[layer.name].blobs

        for idx, blob in enumerate(blobs):
            blob_op = self.blob_to_weight_op(layer, idx)
            operands.append(blob_op)

        output_shape = input_shape
        if bn_mode == 1:
            new_op = top.ScaleOp(
                self.mlir.get_tensor_type(output_shape),
                *operands,
                loc=attrs['loc'],  # unexpected params ignored
                ip=self.mlir.insert_point).output
            self.addOperand(layer.top[0], new_op)
        else:
            if len(operands) == 5:
                operands[1], operands[3] = operands[3], operands[1]
                operands[2], operands[4] = operands[4], operands[2]
            new_op = top.BatchNormOp(
                self.mlir.get_tensor_type(output_shape),
                *operands,
                loc=attrs['loc'],
                epsilon=attrs['epsilon'],  # unexpected params ignored
                ip=self.mlir.insert_point).output
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
        attrs = {'axis': axis, 'loc': self.get_loc(layer.top[0])}
        new_op = top.ConcatOp(self.mlir.get_tensor_type(output_shape),
                              operands,
                              **attrs,
                              ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_continuation_indicator_op(self, layer):
        assert (self.layerType(layer) == 'ContinuationIndicator')
        raise RuntimeError("not implemented")

    def convert_crop_op(self, layer):
        assert (self.layerType(layer) == 'Crop')
        in_op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        output_shape = self.getShape(layer.top[0])
        input_dim = len(input_shape)
        p = layer.crop_param
        axis_index = p.axis
        start_axis = axis_index
        offset_size = len(p.offset)
        crop_n = input_dim - axis_index
        crop_offset = [0] * crop_n
        crop_step = [1] * crop_n
        crop_axes = [x + start_axis for x in range(crop_n)]
        if offset_size > 1:
            assert (offset_size + axis_index <= input_dim)
        for i in range(crop_n):
            offset = 0
            if offset_size == 1:
                # If only one offset is given, all crops have the same offset.
                offset = p.offset[0]
            elif offset_size > 1:
                # For several offsets, the number of offsets must be equal to the
                # number of dimensions to crop, that is dimensions after the axis.
                offset = p.offset[i]
            crop_offset[i] = offset
        crop_ends = [a + b for a, b in zip(crop_offset, output_shape[start_axis:])]
        new_op = top.SliceOp(self.mlir.get_tensor_type(output_shape),
                             in_op,
                             self.mlir.none_op,
                             self.mlir.none_op,
                             self.mlir.none_op,
                             offset=list(crop_offset),
                             steps=list(crop_step),
                             ends=list(crop_ends),
                             axes=list(crop_axes),
                             loc=self.get_loc(layer.top[0]),
                             ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_deconvolution_op(self, layer):
        assert (self.layerType(layer) == "Deconvolution")
        input_shape = self.getShape(layer.bottom[0])
        p = layer.convolution_param
        oc = p.num_output
        dim = len(input_shape) - 2
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
        permute_order = [1, 0, 2, 3]
        filter_op = self.blob_to_weight_op(layer, 0, [], permute_order)
        bias_op = self.mlir.none_op
        if p.bias_term:
            bias_op = self.blob_to_weight_op(layer, 1)
        attrs = {
            'loc': self.get_loc(layer.top[0]),
            'kernel_shape': kernel,
            'strides': stride,
            'dilations': dilation,
            'pads': padding * dim,
            'group': g,
            'do_relu': False,
            'output_padding': dim * [0]
        }
        output_shape = self.getShape(layer.top[0])
        new_op = top.DeconvOp(self.mlir.get_tensor_type(output_shape),
                              in_op,
                              filter_op,
                              bias_op,
                              **attrs,
                              ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_detection_output_op(self, layer):
        assert (self.layerType(layer) == "DetectionOutput")
        in_op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        operands = list()
        for bottom in layer.bottom:
            op = self.getOperand(bottom)
            operands.append(op)
        p = layer.detection_output_param
        code_type = StringAttr.get("CORNER")
        if p.code_type == 2:
            code_type = StringAttr.get("CENTER_SIZE")
        elif p.code_type == 3:
            code_type = StringAttr.get("CORNER_SIZE")
        param = {
            'loc': self.get_loc(layer.top[0]),
            'num_classes': p.num_classes,
            'share_location': p.share_location,
            'background_label_id': p.background_label_id,
            'nms_threshold': p.nms_param.nms_threshold,
            'top_k': p.nms_param.top_k,
            'code_type': code_type,
            'keep_top_k': p.keep_top_k,
            'confidence_threshold': p.confidence_threshold
        }
        assert (1.0 == p.nms_param.eta)
        assert (False == p.variance_encoded_in_target)
        output_shape = [input_shape[0], 1, p.keep_top_k, 7]
        new_op = top.DetectionOutputOp(self.mlir.get_tensor_type(output_shape),
                                       operands,
                                       **param,
                                       ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_dropout_op(self, layer):
        assert (self.layerType(layer) == 'Dropout')
        op = self.getOperand(layer.bottom[0])
        self.addOperand(layer.top[0], op)

    def convert_nothing(self, layer):
        # do nothing
        pass

    def convert_embed_op(self, layer):
        assert (self.layerType(layer) == 'Embed')
        raise RuntimeError("not implemented")

    def convert_flatten_op(self, layer):
        assert (self.layerType(layer) == 'Flatten')
        in_op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        num_dims = len(input_shape)
        assert (num_dims > 1)
        output_shape = [input_shape[0], 1]
        for i in range(1, num_dims):
            output_shape[1] *= input_shape[i]
        param = {'loc': self.get_loc(layer.top[0])}
        new_op = top.ReshapeOp(self.mlir.get_tensor_type(output_shape),
                               in_op,
                               **param,
                               ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_frcn_detection_op(self, layer):
        assert (self.layerType(layer) == 'FrcnDetection')
        raise RuntimeError("not implemented")

    def convert_interp_op(self, layer):
        assert (self.layerType(layer) == 'Interp')
        #
        # all settings:
        #
        #height:33 width:65 (1, 1024, 1, 1)->(1, 1024, 33, 65)
        #shrink_factor:2 (1, 3, 1025, 2049)->(1, 3, 513, 1025)
        #zoom_factor: 2(1, 256, 33, 65)->(1, 256, 65, 129)
        #pad_beg:0
        #pad_end:0
        # plz refer \interp_layer.cpp:30 for more parsing priority info

        in_op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        output_shape = list(input_shape)
        p = layer.interp_param

        assert (len(input_shape) == 4 and "current only support 4 dims")
        assert ((p.pad_beg == 0 and p.pad_end == 0) and "only support pad")

        # append pad
        after_h = input_shape[2] + p.pad_beg + p.pad_end
        after_w = input_shape[3] + p.pad_beg + p.pad_end

        # only deal with > case
        def shrink(after_h, after_w, p):
            assert (p.shrink_factor >= 1 and "p.shrink_factor should > 1")
            after_h = math.floor((after_h - 1) / p.shrink_factor) + 1
            after_w = math.floor((after_w - 1) / p.shrink_factor) + 1
            return after_h, after_w

        def zoom(after_h, after_w, p):
            assert (p.zoom_factor >= 1 and "p.zoom_factor should > 1")
            after_h = after_h + math.floor((after_h - 1) * (p.zoom_factor - 1))
            after_w = after_w + math.floor((after_w - 1) * (p.zoom_factor - 1))
            return after_h, after_w

        shrink_factor = 0
        zoom_factor = 0
        height = 0
        width = 0

        if hasattr(p, 'shrink_factor'):
            if p.shrink_factor > 1:
                shrink_factor = p.shrink_factor

        if hasattr(p, 'zoom_factor'):
            if p.zoom_factor > 1:
                zoom_factor = p.zoom_factor
        if hasattr(p, 'height'):
            height = p.height
        if hasattr(p, 'width'):
            width = p.width

        if shrink_factor and not zoom_factor:
            after_h, after_w = shrink(after_h, after_w, p)
        elif zoom_factor and not shrink_factor:
            after_h, after_w = zoom(after_h, after_w, p)
        elif height and width:
            assert ((p.height > 0 and p.width > 0) and "height/width must > 0")
            after_h = p.height
            after_w = p.width
        elif shrink_factor and zoom_factor:
            after_h, after_w = shrink(after_h, after_w, p)
            after_h, after_w = zoom(after_h, after_w, p)
        else:
            logger.info(f"param is {p}")
            assert (0 and "not support interp type")

        output_shape[2] = after_h
        output_shape[3] = after_w

        param = {
            'loc': self.get_loc(layer.top[0]),
            # 'height': p.height,  # unexpected params ignored
            # 'pad_beg': p.pad_beg,
            # 'pad_end': p.pad_end,
            # 'shrink_factor': p.shrink_factor,
            # 'width': p.width,
            # 'zoom_factor': p.zoom_factor,
            'scale_h': float(input_shape[2] - 1) / (output_shape[2] - 1),
            'scale_w': float(input_shape[3] - 1) / (output_shape[3] - 1),
            'coord_mode': StringAttr.get('align_corners'),
            'mode': StringAttr.get('linear'),
        }

        target_shape_op = self.create_weight_op(layer.name + "_shape",
                                                np.array(output_shape[2:], np.int32))
        new_op = top.InterpOp(self.mlir.get_tensor_type(output_shape),
                              in_op,
                              target_shape_op,
                              **param,
                              ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_lrn_op(self, layer):
        assert (self.layerType(layer) == 'LRN')
        in_op = self.getOperand(layer.bottom[0])
        p = layer.lrn_param
        param = {
            'loc': self.get_loc(layer.top[0]),
            'alpha': p.alpha,
            'beta': p.beta,
            'bias': p.k,
            'size': p.local_size,
        }
        output_shape = self.getShape(layer.top[0])
        new_op = top.LRNOp(self.mlir.get_tensor_type(output_shape),
                           in_op,
                           **param,
                           ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_absval_op(self, layer):
        assert (self.layerType(layer) ==  "AbsVal")
        in_op = self.getOperand(layer.bottom[0])
        output_shape = self.getShape(layer.top[0])
        param = {
            'loc': self.get_loc(layer.top[0])
        }
        new_op = top.AbsOp(self.mlir.get_tensor_type(output_shape),
                           in_op,
                           **param,
                           ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_lstm_op(self, layer):
        assert (self.layerType(layer) == 'LSTM')
        op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        seq_length = input_shape[0]
        batch_size = input_shape[1]
        input_size = input_shape[2]
        hidden_size = layer.recurrent_param.num_output
        operands = list()
        operands.append(op)
        # weight
        wname = layer.name + "_weight__0"
        weight = self.layer_dict[layer.name].blobs[0].data
        weight = weight.reshape([4, hidden_size * input_size])
        weight[[1, 2], :] = weight[[2, 1], :]  # ifoc =>iofc
        weight = weight.reshape([1, 4 * hidden_size, input_size])
        weight_op = self.create_weight_op(wname, weight)
        operands.append(weight_op)
        # recurrence
        rname = layer.name + "_weight_1"
        r = self.layer_dict[layer.name].blobs[2].data
        r = r.reshape([4, hidden_size * hidden_size])
        r[[1, 2], :] = r[[2, 1], :]  # ifoc =>iofc
        r = r.reshape([1, 4 * hidden_size, hidden_size])
        recurrence_op = self.create_weight_op(rname, r)
        operands.append(recurrence_op)
        # bias
        bname = layer.name + "_weight_2"
        bias = self.layer_dict[layer.name].blobs[1].data
        bias = bias.reshape([4, hidden_size])
        bias[[1, 2], :] = bias[[2, 1], :]  # ifoc =>iofc
        bias = bias.reshape([1, 4 * hidden_size])
        rbias = np.zeros_like(bias)
        merge_bias = np.concatenate((bias, rbias), 1)
        bias_op = self.create_weight_op(bname, merge_bias)
        operands.append(bias_op)
        operands.append(self.mlir.none_op)  # initial_h
        operands.append(self.mlir.none_op)  # initial_c
        if len(layer.bottom) > 1:
            cont_op = self.getOperand(layer.bottom[1])
            operands.append(cont_op)
        else:
            operands.append(self.mlir.none_op)  # cont

        name = layer.top[0]
        param = {
            "loc": self.get_loc([name + '_lstm', name + '_H', name + '_C']),
            "hidden_size": hidden_size,
            "bidirectional": bool(False),
            "batch_first": bool(False),
        }
        out_shape = [seq_length, 1, batch_size, hidden_size]
        out_shapes = [out_shape, None, None]
        new_op = top.LSTMOp(*self.mlir.get_tensor_type(out_shapes),
                            *operands,
                            **param,
                            ip=self.mlir.insert_point).Y
        # reshape back
        attrs = {'loc': self.get_loc(name)}
        output_shape = self.getShape(layer.top[0])
        new_reshape_op = top.ReshapeOp(self.mlir.get_tensor_type(output_shape),
                                       new_op,
                                       **attrs,
                                       ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_reshape_op)

    def convert_lstm_jun_op(self, layer):
        assert (self.layerType(layer) == 'Lstm')
        op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        seq_length = input_shape[0]
        batch_size = input_shape[1]
        input_size = input_shape[2]
        hidden_size = layer.lstm_param.num_output
        operands = list()
        operands.append(op)
        # weight
        wname = layer.name + "_0"
        weight = self.layer_dict[layer.name].blobs[0].data
        weight = weight.reshape([4, hidden_size * input_size])
        weight[[1, 2], :] = weight[[2, 1], :]  # ifoc =>iofc
        weight = weight.reshape([1, 4 * hidden_size, input_size])
        weight_op = self.create_weight_op(wname, weight)
        operands.append(weight_op)
        # recurrence
        rname = layer.name + "_1"
        r = self.layer_dict[layer.name].blobs[1].data
        r = r.reshape([4, hidden_size * hidden_size])
        r[[1, 2], :] = r[[2, 1], :]  # ifoc =>iofc
        r = r.reshape([1, 4 * hidden_size, hidden_size])
        recurrence_op = self.create_weight_op(rname, r)
        operands.append(recurrence_op)
        # bias
        bname = layer.name + "_2"
        bias = self.layer_dict[layer.name].blobs[2].data
        bias = bias.reshape([4, hidden_size])
        bias[[1, 2], :] = bias[[2, 1], :]  # ifoc =>iofc
        bias = bias.reshape([1, 4 * hidden_size])
        rbias = np.zeros_like(bias)
        merge_bias = np.concatenate((bias, rbias), 1)
        bias_op = self.create_weight_op(bname, merge_bias)
        operands.append(bias_op)
        operands.append(self.mlir.none_op)  # initial_h
        operands.append(self.mlir.none_op)  # initial_c
        operands.append(self.mlir.none_op)  # cont
        name = layer.top[0]
        param = {
            "loc": self.get_loc([name + '_lstm', name + '_H', name + '_C']),
            "hidden_size": hidden_size,
            "bidirectional": bool(False),
            "batch_first": bool(False),
        }
        out_shape = [seq_length, 1, batch_size, hidden_size]
        out_shapes = [out_shape, None, None]
        new_op = top.LSTMOp(*self.mlir.get_tensor_type(out_shapes),
                            *operands,
                            **param,
                            ip=self.mlir.insert_point).Y
        # reshape back
        attrs = {'loc': self.get_loc(name)}
        output_shape = self.getShape(layer.top[0])
        new_reshape_op = top.ReshapeOp(self.mlir.get_tensor_type(output_shape),
                                       new_op,
                                       **attrs,
                                       ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_reshape_op)

    def convert_matmul_op(self, layer):
        assert (self.layerType(layer) == 'MatMul')
        raise RuntimeError("not implemented")

    def convert_normalize_op(self, layer):
        assert (self.layerType(layer) == 'Normalize')
        input_shape = self.getShape(layer.bottom[0])
        in_op = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(in_op)
        p = layer.norm_param
        param = {
            'loc': self.get_loc(layer.top[0]),
            'across_spatial': p.across_spatial,
            'channel_shared': p.channel_shared,
        }
        assert (False == p.across_spatial)
        assert (len(input_shape) > 1)
        c = input_shape[1]
        #scale
        scale_shape = [1, c]
        scale_name = layer.name + "_0"
        blob = self.layer_dict[layer.name].blobs[0]
        scale_data = np.array
        if p.channel_shared:
            assert (blob.count == 1)
            value = blob.data.flatten()[0]
            scale_data = np.array([[value for i in range(c)]], dtype=float)
        else:
            assert (blob.count == c)
            scale_data = blob.data.reshape(scale_shape)
        scale_op = self.create_weight_op(scale_name, scale_data)
        operands.append(scale_op)
        output_shape = self.getShape(layer.top[0])
        new_op = top.NormalizeOp(self.mlir.get_tensor_type(output_shape),
                                 *operands,
                                 **param,
                                 ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_mish_op(self, layer):
        assert (self.layerType(layer) == 'Mish')
        op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        output_shape = input_shape
        attrs = {'loc': self.get_loc(layer.top[0])}
        new_op = top.MishOp(self.mlir.get_tensor_type(output_shape),
                            op,
                            **attrs,
                            ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_padding_op(self, layer):
        assert (self.layerType(layer) == 'Padding')
        raise RuntimeError("not implemented")

    def convert_permute_op(self, layer):
        assert (self.layerType(layer) == 'Permute')
        in_op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        operands = list()
        for bottom in layer.bottom:
            op = self.getOperand(bottom)
            operands.append(op)
        p = layer.permute_param
        output_shape = list(input_shape)
        for i in range(len(p.order)):
            output_shape[i] = input_shape[p.order[i]]
        attrs = {'order': p.order, 'loc': self.get_loc(layer.top[0])}
        new_op = top.PermuteOp(self.mlir.get_tensor_type(output_shape),
                               in_op,
                               **attrs,
                               ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_power_op(self, layer):
        assert (self.layerType(layer) == 'Power')
        raise RuntimeError("not implemented")

    def convert_prelu_op(self, layer):
        assert (self.layerType(layer) == 'PReLU')
        op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        operands = list()
        operands.append(op)
        dim_len = len(input_shape)
        assert (dim_len > 1)
        slope_shape = [1] * dim_len
        slope_shape[1] = input_shape[1]
        # negative_slope
        slope_op = self.blob_to_weight_op(layer, 0, slope_shape)
        operands.append(slope_op)
        param = {
            'loc': self.get_loc(layer.top[0]),
        }
        output_shape = input_shape
        new_op = top.PReluOp(self.mlir.get_tensor_type(output_shape),
                             *operands,
                             **param,
                             ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_priorbox_op(self, layer):
        assert (self.layerType(layer) == 'PriorBox')
        op0 = self.getOperand(layer.bottom[0])
        op1 = self.getOperand(layer.bottom[1])
        input_shape0 = self.getShape(layer.bottom[0])
        input_shape1 = self.getShape(layer.bottom[1])
        operands = list()
        operands.append(op0)
        operands.append(op1)
        assert (len(input_shape0) == 4)
        h = input_shape0[2]
        w = input_shape0[3]
        p = layer.prior_box_param
        min_size = [i for i in p.min_size]
        max_size = [i for i in p.max_size]
        aspect_ratio = [i for i in p.aspect_ratio]
        variance = [i for i in p.variance]
        assert (len(variance) == 4)

        param = {
            'min_size': min_size,
            'max_size': max_size,
            'variance': variance,
            'clip': p.clip,
            'offset': p.offset,
            'loc': self.get_loc(layer.top[0]),
        }

        if p.HasField('step_h') and p.HasField('step_w'):
            param['step_h'] = p.step_h
            param['step_w'] = p.step_w
        elif p.HasField('step'):
            param['step_h'] = p.step
            param['step_w'] = p.step
        else:
            param['step_h'] = 0
            param['step_w'] = 0

        if p.HasField('img_h') and p.HasField('img_w'):
            param['img_h'] = p.img_h
            param['img_w'] = p.img_w
        elif p.HasField('img_size'):
            param['img_h'] = p.img_size
            param['img_w'] = p.img_size
        else:
            param['img_h'] = 0
            param['img_w'] = 0
        aspect_ratios_ = list()
        use_default_aspect_ratio = True
        if p.HasField('use_default_aspect_ratio'):
            use_default_aspect_ratio = p.use_default_aspect_ratio
        if use_default_aspect_ratio:
            aspect_ratios_.append(1.0)
        for ar in aspect_ratio:
            already_exist = False
            for j in aspect_ratios_:
                if math.fabs(ar - j) < 1e-6:
                    already_exist = True
                    break
            if not already_exist:
                aspect_ratios_.append(ar)
                if p.flip:
                    aspect_ratios_.append(1.0 / ar)
        num_priors = len(aspect_ratios_) * len(min_size)
        if len(max_size) > 0:
            assert (len(max_size) == len(min_size))
            num_priors += len(max_size)
        param['num_priors'] = num_priors
        param['aspect_ratios'] = aspect_ratios_
        param['use_default_aspect_ratio'] = use_default_aspect_ratio
        output_shape = [1, 2, int(h * w * num_priors * 4)]
        new_op = top.PriorBoxOp(self.mlir.get_tensor_type(output_shape),
                                operands,
                                **param,
                                ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_proposal_op(self, layer):
        assert (self.layerType(layer) == 'Proposal')
        input_shape = self.getShape(layer.bottom[0])
        operands = list()
        for bottom in layer.bottom:
            in_op = self.getOperand(bottom)
            operands.append(in_op)
        p = layer.proposal_param
        feat_stride = p.feat_stride
        anchor_base_size = p.anchor_base_size
        rpn_obj_threshold = p.rpn_obj_threshold
        rpn_nms_threshold = p.rpn_nms_threshold
        rpn_nms_post_top_n = p.rpn_nms_post_top_n
        net_input_h = p.net_input_h
        net_input_w = p.net_input_w
        param = {
            'loc': self.get_loc(layer.top[0]),
            'net_input_h': net_input_h,
            'net_input_w': net_input_w,
            'feat_stride': feat_stride,
            'anchor_base_size': anchor_base_size,
            'rpn_obj_threshold': rpn_obj_threshold,
            'rpn_nms_threshold': rpn_nms_threshold,
            'rpn_nms_post_top_n': rpn_nms_post_top_n
        }
        output_shape = [input_shape[0], 1, rpn_nms_post_top_n, 5]
        new_op = top.ProposalOp(self.mlir.get_tensor_type(output_shape),
                                operands,
                                **param,
                                ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_reduce_op(self,layer):
        assert (self.layerType(layer) == 'Reduction')
        p = layer.reduction_param
        reduce_dict = {"4":"ReduceMean"}
        reduce_method = str(p.operation)
        assert reduce_method in reduce_dict
        method = reduce_dict[reduce_method]
        keepdims = False
        axis = p.axis
        input_shape = self.getShape(layer.bottom[0])
        if axis<0:
            axis+=len(input_shape)
        ## cal output shape ##
        if isinstance(axis, int):
            axis = [axis]
        output_shape = list(input_shape)
        for a in sorted(axis, reverse=True):
            output_shape.pop(a)  
        op = self.getOperand(layer.bottom[0])
        params = {}
        param = {
                'axes':axis,
                'keepdims':keepdims,
                'mode':StringAttr.get(method),
                'loc':self.get_loc(layer.top[0])
        }
        new_op = top.ReduceOp(self.mlir.get_tensor_type(output_shape),
                              op,
                              **param,
                              ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_relu6_op(self, layer):
        assert (self.layerType(layer) == 'ReLU6')
        op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        output_shape = input_shape
        clip_attrs = {'loc': self.get_loc(layer.top[0]), 'min': 0.0, 'max': 6.0}
        if layer.relu_param.HasField('negative_slope'):
            attrs = {'loc': self.get_loc(layer.top[0] + '_leakyrelu')}
            attrs['alpha'] = layer.relu_param.negative_slope
            leaky_relu_op = top.LeakyReluOp(self.mlir.get_tensor_type(output_shape),
                                            op,
                                            **attrs,
                                            ip=self.mlir.insert_point).output
            new_op = top.ClipOp(self.mlir.get_tensor_type(output_shape),
                                leaky_relu_op,
                                **clip_attrs,
                                ip=self.mlir.insert_point).output
        else:
            new_op = top.ClipOp(self.mlir.get_tensor_type(output_shape),
                                op,
                                **clip_attrs,
                                ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

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
        attrs = {
            "block_h": stride,
            "block_w": stride,
            "is_CRD": False,
            "is_inversed": True,
            "loc": self.get_loc(layer.top[0])
        }
        new_op = top.Depth2SpaceOp(self.mlir.get_tensor_type(output_shape),
                                   in_op,
                                   **attrs,
                                   ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_reshape_op(self, layer):
        assert (self.layerType(layer) == 'Reshape')
        op = self.getOperand(layer.bottom[0])
        attrs = {'loc': self.get_loc(layer.top[0])}
        output_shape = self.getShape(layer.top[0])
        new_op = top.ReshapeOp(self.mlir.get_tensor_type(output_shape),
                               op,
                               **attrs,
                               ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_reverse_op(self, layer):
        assert (self.layerType(layer) == 'Reverse')
        op = self.getOperand(layer.bottom[0])
        axis = layer.reverse_param.axis
        attrs = {'loc': self.get_loc(layer.top[0]), 'axis': axis}
        output_shape = self.getShape(layer.top[0])
        new_op = top.ReverseOp(self.mlir.get_tensor_type(output_shape),
                               op,
                               **attrs,
                               ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_retinaface_detection_op(self, layer):
        assert (self.layerType(layer) == 'RetinaFaceDetection')
        input_shape = self.getShape(layer.bottom[0])
        operands = list()
        #op = self.getOperand(layer.bottom[0])
        for bottom in layer.bottom:
            op = self.getOperand(bottom)
            operands.append(op)
        p = layer.retinaface_detection_param
        nms_threshold = p.nms_threshold
        confidence_threshold = p.confidence_threshold
        keep_topk = p.keep_topk
        output_shape = [input_shape[0], 1, keep_topk, 15]
        param = {
            'loc': self.get_loc(layer.top[0]),
            'nms_threshold': nms_threshold,
            'confidence_threshold': confidence_threshold,
            'keep_topk': keep_topk,
        }
        new_op = top.RetinaFaceDetectionOp(self.mlir.get_tensor_type(output_shape),
                                           operands,
                                           **param,
                                           ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_roipooling_op(self, layer):
        assert (self.layerType(layer) == 'ROIPooling')
        operands = list()
        assert (len(layer.bottom) == 2)
        op0 = self.getOperand(layer.bottom[0])
        op1 = self.getOperand(layer.bottom[1])
        bottom0_shape = self.getShape(layer.bottom[0])
        bottom1_shape = self.getShape(layer.bottom[1])
        operands.append(op0)
        operands.append(op1)
        p = layer.roi_pooling_param
        pooled_h = p.pooled_h
        pooled_w = p.pooled_w
        spatial_scale = p.spatial_scale
        param = {
            'loc': self.get_loc(layer.top[0]),
            'pooled_h': pooled_h,
            'pooled_w': pooled_w,
            'spatial_scale': spatial_scale
        }
        output_shape = [bottom1_shape[0] * bottom1_shape[2], bottom0_shape[1], pooled_h, pooled_w]
        new_op = top.ROIPoolingOp(self.mlir.get_tensor_type(output_shape),
                                  operands,
                                  **param,
                                  ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_frcn_detection_op(self, layer):
        assert (self.layerType(layer) == 'FrcnDetection')
        input_shape = self.getShape(layer.bottom[2])
        operands = list()
        for bottom in layer.bottom:
            op = self.getOperand(bottom)
            operands.append(op)
        p = layer.frcn_detection_param
        class_num = p.class_num
        obj_threshold = p.obj_threshold
        nms_threshold = p.nms_threshold
        keep_topk = p.keep_topk
        param = {
            'loc': self.get_loc(layer.top[0]),
            'class_num': class_num,
            'obj_threshold': obj_threshold,
            'nms_threshold': nms_threshold,
            'keep_topk': keep_topk
        }

        output_shape = [input_shape[0], 1, keep_topk, 6]
        new_op = top.FrcnDetectionOp(self.mlir.get_tensor_type(output_shape),
                                     operands,
                                     **param,
                                     ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_shufflechannel_op(self, layer):
        assert (self.layerType(layer) == 'ShuffleChannel')
        in_op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        group = layer.shuffle_channel_param.group
        operands = list()
        operands.append(in_op)
        output_shape = input_shape
        attrs = {'loc': self.get_loc(layer.top[0]), 'group': group}
        new_op = top.ShuffleChannelOp(self.mlir.get_tensor_type(output_shape),
                                      *operands,
                                      **attrs,
                                      ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_sigmoid_op(self, layer):
        assert (self.layerType(layer) == 'Sigmoid')
        in_op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        output_shape = input_shape
        attrs = {'scale': 1, 'bias': 0, 'loc': self.get_loc(layer.top[0])}
        new_op = top.SigmoidOp(self.mlir.get_tensor_type(output_shape),
                               in_op,
                               **attrs,
                               ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_slice_op(self, layer):
        assert (self.layerType(layer) == 'Slice')
        in_op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        p = layer.slice_param
        axis = p.axis
        bottom_slice_axis = input_shape[axis]
        top_size = len(layer.top)
        slice_num = len(p.slice_point)
        slices = list()
        if slice_num > 0:
            assert (slice_num == top_size - 1)
            assert (top_size <= bottom_slice_axis)
            prev = 0
            for i in range(slice_num):
                assert (p.slice_point[i] > prev)
                slices.append(p.slice_point[i] - prev)
                prev = p.slice_point[i]
            slices.append(bottom_slice_axis - prev)
        else:
            assert (bottom_slice_axis % top_size == 0)
            for i in range(top_size):
                slices.append(int(bottom_slice_axis / top_size))
        offset = 0
        for i in range(top_size):
            output_shape = list(input_shape)
            output_shape[axis] = slices[i]
            crop_offset = [0] * len(input_shape)
            crop_offset[axis] = offset
            steps = [1] * len(input_shape)
            ends = [a + b for a, b in zip(crop_offset, output_shape)]
            new_op = top.SliceOp(self.mlir.get_tensor_type(output_shape),
                                 in_op,
                                 self.mlir.none_op,
                                 self.mlir.none_op,
                                 self.mlir.none_op,
                                 offset=crop_offset,
                                 steps=steps,
                                 ends=ends,
                                 loc=self.get_loc(layer.top[i]),
                                 ip=self.mlir.insert_point).output
            self.addOperand(layer.top[i], new_op)
            offset += slices[i]
        #raise RuntimeError("not implemented")

    def convert_split_op(self, layer):
        assert (self.layerType(layer) == 'Split')
        raise RuntimeError("not implemented")

    def convert_tanh_op(self, layer):
        assert (self.layerType(layer) == 'TanH')
        in_op = self.getOperand(layer.bottom[0])
        input_shape = self.getShape(layer.bottom[0])
        output_shape = input_shape
        attrs = {'loc': self.get_loc(layer.top[0])}
        new_op = top.TanhOp(self.mlir.get_tensor_type(output_shape),
                            in_op,
                            **attrs,
                            ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

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
        attrs = {'scale_h': scale, 'scale_w': scale, 'loc': self.get_loc(layer.top[0])}
        if len(layer.bottom) == 1:
            new_op = top.UpsampleOp(self.mlir.get_tensor_type(output_shape),
                                    in_op,
                                    **attrs,
                                    ip=self.mlir.insert_point).output
        else:
            mask_op = self.getOperand(layer.bottom[1])
            new_op = top.MaxUnpoolOp(self.mlir.get_tensor_type(output_shape),
                                     in_op,
                                     mask_op,
                                     **attrs,
                                     ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)

    def convert_yolo_detection_op(self, layer):
        assert (self.layerType(layer) == 'YoloDetection')
        input_shape = self.getShape(layer.bottom[0])

        operands = list()
        for bottom in layer.bottom:
            op = self.getOperand(bottom)
            operands.append(op)
        p = layer.yolo_detection_param
        anchors = []
        # yapf: disable
        if p.anchors:
            anchors = [int(d) for d in p.anchors.split(",")]
        elif p.yolo_v4:
            anchors = [142, 110, 192, 243, 459, 401, 36, 75, 76, 55, 72, 146, 12, 16, 19, 36, 40, 28]
        elif p.tiny:
            anchors = [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]
        else:
            anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
        # yapf: enable
        assert (len(anchors) == 6 * len(operands))
        if p.spp_net:
            version = "yolov3_spp"
        elif p.tiny:
            version = "yolov3_tiny"
        elif p.yolo_v4:
            version = "yolov4"
        else:
            version = "yolov3"
        param = {
            'loc': self.get_loc(layer.top[0]),
            'net_input_h': p.net_input_h,
            "net_input_w": p.net_input_w,
            "nms_threshold": p.nms_threshold,
            "obj_threshold": p.obj_threshold,
            "keep_topk": p.keep_topk,
            "version": StringAttr.get(version),
            "class_num": p.class_num,
            "anchors": anchors
        }
        output_shape = [input_shape[0], 1, p.keep_topk, 6]
        new_op = top.YoloDetectionOp(self.mlir.get_tensor_type(output_shape),
                                     operands,
                                     **param,
                                     ip=self.mlir.insert_point).output
        self.addOperand(layer.top[0], new_op)
