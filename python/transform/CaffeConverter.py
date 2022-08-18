# Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
#
# Licensed under the Apache License v2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.
# SPDX-License-Identifier: Apache-2.0
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
        self.inputs = self.net.inputs
        self.outputs = self.net.outputs
        self.blobs = self.net.blobs
        self.layer_dict = self.net.layer_dict

        self.mlir = None
        self.mlir_file_path = mlir_file_path
        self.converted_tensors = list()
        self.input_shapes = list()
        self.output_shapes = list()
        self.CVI = None
        self.weight_file = "{}_top_weight.npz".format(model_name)
        self.preprocess_args = preprocess_args

        self.caffeop_factory = {
            #pls add the Op according to the Op's alphabetical order as below
            'BatchNorm': lambda layer: self.convert_batchnorm_op(layer),
            'BN': lambda layer: self.convert_bn_op(layer),
            'Concat': lambda layer: self.convert_concat_op(layer),
            'ContinuationIndicator': lambda layer: self.convert_continuation_indicator_op(layer),
            'Convolution': lambda layer: self.convert_convolution_op(layer),
            'ConvolutionDepthwise': lambda layer: self.convert_convolution_op(layer),
            'Crop': lambda layer: self.convert_crop_op(layer),
            'Deconvolution': lambda layer: self.convert_convolution_op(layer),
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
            22: 'Split', 33: 'Slice', 23: 'Tanh', 24: 'WindowData', 31: 'Threshold', 32: 'Relu6'
        }
        self.init_importer()

    def __del__(self):
        del self.CVI

    def layerType(self, layer):
        if type(layer.type) == int:
            return self.layer_type.get(layer.type)
        else:
            return layer.type

    def align_up(self, x, n):
        if n == 0 or n == 1:
            return x
        return int((x + n - 1)/n) * n

    def init_importer(self):
        self.input_shapes = list()
        for i in self.inputs:
            input_shape = list(self.blobs[i].shape)
            if self.batch_size != 0:
                input_shape[0] = self.batch_size
            elif input_shape[0] <= 0:
                input_shape[0] = 1
                self.batch_size = 1
            self.blobs[i].reshape(*input_shape)
            self.input_shapes.append(input_shape)

        self.net.reshape()
        self.output_shapes = list()
        for o in self.outputs:
            o_shape = list(self.blobs[o].shape)
            for layer in self.layers:
                if layer.name == o and self.layerType(layer) == 'DetectionOutput':
                    if self.batch_size != 0:
                        o_shape[0] = self.batch_size
                    o_shape[2] = layer.detection_output_param.keep_top_k
                    break
            self.output_shapes.append(o_shape)
        self.CVI = MLIRImporter(self.input_shapes, self.output_shapes, "FP32",
                                weight_file=self.weight_file)

    def addTensor(self, op_name, tensor_data, tensor_shape):
        self.converted_tensors.append(CaffeTensor(
            op_name, tensor_data, tensor_shape))

    def TensortoNpz(self):
        tensor_npz = {}
        for i in self.converted_tensors:
            tensor_npz[i.name] = i.tensor_data.astype(np.float32)
        np.savez(self.weight_file, **tensor_npz)

    def getTensor(self, op_name):
        find_tensor = [t for t in self.converted_tensors if t.name == op_name]
        if len(find_tensor) < 1:
            raise KeyError("No {} tensor in prototxt".format(op_name))
        else:
            return find_tensor[0]

    def add_none_op(self):
        return self.mlir.add_none_op()

    def blob_to_weight_op(self, layer, index, shape=None, permute_order=None, channel_idx=0):
        name = layer.name + "_{}".format(index)
        blob = self.layer_dict[layer.name].blobs[index]
        blob_shape = list(blob.shape)
        value = np.array
        new_shape = list(blob_shape)
        if shape != None:
            new_shape = [int(i) for i in shape]
        if new_shape == blob_shape:
            value = blob.data
        elif not blob_shape:
            value = np.array(
                [blob.data for i in range(new_shape[channel_idx])], dtype=np.float32)
        else:
            value = blob.data.reshape(new_shape)
        if permute_order != None:
            value = blob.data
            value_tensor = torch.tensor(value)
            value_tensor = value_tensor.permute(permute_order)
            value = np.array(value_tensor)
            value = value.reshape(new_shape)
        self.addTensor(name, value, new_shape)
        weight_op = self.mlir.add_load_file_op(name, new_shape)
        return weight_op

    def convert_batchnorm_op(self, layer):
        assert(self.layerType(layer) == "BatchNorm")
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        param = {
            'variance_epsilon': 1e-5
        }
        if layer.HasField('batch_norm_param') and layer.batch_norm_param.HasField('eps'):
            param['variance_epsilon'] = layer.batch_norm_param.eps

        if layer.HasField('batch_norm_param') and layer.batch_norm_param.HasField('use_global_stats'):
            assert(layer.batch_norm_param.use_global_stats == True)

        blobs = self.layer_dict[layer.name].blobs
        for idx, blob in enumerate(blobs):
            blob_op = self.blob_to_weight_op(layer, idx)
            operands.append(blob_op)

        output_shape = input_shape
        new_op = self.mlir.add_batchnorm_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,output_shape)

    def convert_bn_op(self, layer):
        assert(self.layerType(layer) == 'BN')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        # default comes from caffe.proto

        p = layer.bn_param
        bn_mode = 0
        if hasattr(p, 'bn_mode'):
            bn_mode = p.bn_mode

        param = {
            'variance_epsilon': 1e-5,
            #'momentum': 0.9,
            'frozen': False
        }

        if layer.HasField('bn_param'):
            if layer.bn_param.HasField('eps'):
                param['variance_epsilon'] = layer.bn_param.eps

            #if layer.bn_param.HasField('momentum'):
            #    param['momentum'] = layer.bn_param.momentum

            if layer.bn_param.HasField('frozen'):
                param['frozen'] = layer.bn_param.frozen
                assert(param['frozen'] == True and "only support frozen = false now")

        blobs = self.layer_dict[layer.name].blobs

        for idx, blob in enumerate(blobs):
            blob_op = self.blob_to_weight_op(layer, idx)
            operands.append(blob_op)

        if bn_mode == 1:
            output_shape = input_shape
            new_op = self.mlir.add_scale_op(
                layer.name, operands, output_shape)

            self.addOperand(layer.top[0], new_op,
                          output_shape)
        else:
            output_shape = input_shape
            new_op = self.mlir.add_batchnorm_op(layer.name, operands, output_shape, **param)
            self.addOperand(layer.top[0], new_op,
                            output_shape)

    def convert_concat_op(self, layer):
        assert(self.layerType(layer) == 'Concat')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        input_num = len(layer.bottom)
        if input_num == 1:
            return self.addOperand(layer.top[0], op, input_shape)
        axis = layer.concat_param.axis
        assert(axis < len(input_shape))
        concat_axis_dim = 0
        operands = list()
        for bottom in layer.bottom:
            bottom_op, shape, _ = self.getOperand(bottom)
            assert(len(shape) == len(input_shape))
            concat_axis_dim += shape[axis]
            operands.append(bottom_op)
        output_shape = list(input_shape)
        output_shape[axis] = concat_axis_dim
        param = {
            'axis': axis
        }
        new_op = self.mlir.add_concat_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_continuation_indicator_op(self, layer):
        assert(self.layerType(layer) == 'ContinuationIndicator')
        # do nothing

    @ staticmethod
    def calcConv2DSpatialOutput(_i_, _k_, _s_, _p_, _d_):
        return int(((_i_) + 2 * (_p_) - (_d_) * ((_k_)-1) - 1) / (_s_) + 1)

    @ staticmethod
    def calcDeConv2DSpatialOutput(_i_, _k_, _s_, _p_, _d_):
        return int((_s_) * (((_i_)) - 1) + (_d_) * ((_k_)-1) - 2 * (_p_) + 1)

    def convert_convolution_op(self, layer):
        assert(self.layerType(layer) == "Convolution" or self.layerType(layer) ==
               "ConvolutionDepthwise" or self.layerType(layer) == 'Deconvolution')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        assert(len(input_shape) == 4)
        operands = list()
        operands.append(op)
        p = layer.convolution_param
        oc = p.num_output
        g = 1
        is_dw = False
        if self.layerType(layer) == 'ConvolutionDepthwise':
            g = oc
            is_dw = True
        else:
            g = p.group
        is_deconv = True if self.layerType(layer) == 'Deconvolution' else False
        with_bias = p.bias_term
        kernel = [0, 0]
        if len(p.kernel_size) != 0:
            kernel[0] = p.kernel_size[1] if len(
                p.kernel_size) > 1 else p.kernel_size[0]
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
        padding_str = 'SAME' if (padding[0] > 0 or padding[1] > 0) else 'VALID'
        dilation = [1, 1]
        if len(p.dilation) != 0:
            dilation[0] = p.dilation[1] if len(
                p.dilation) > 1 else p.dilation[0]
            dilation[1] = p.dilation[0]
        n = input_shape[0]
        ic = input_shape[1]
        ifmap = [input_shape[2], input_shape[3]]
        ofmap = [0, 0]
        if not is_deconv:
            ofmap[0] = self.calcConv2DSpatialOutput(
                ifmap[0], kernel[0], stride[0], padding[0], dilation[0])
            ofmap[1] = self.calcConv2DSpatialOutput(
                ifmap[1], kernel[1], stride[1], padding[1], dilation[1])
        else:
            ofmap[0] = self.calcDeConv2DSpatialOutput(ifmap[0], kernel[0], stride[0], padding[0],
                                                      dilation[0])
            ofmap[1] = self.calcDeConv2DSpatialOutput(ifmap[1], kernel[1], stride[1], padding[1],
                                                      dilation[1])
        if g > 1 and g == oc and g == ic:
            is_dw = True

        # filter op
        filter_shape = [g, int(oc / g), int(ic / g), kernel[0], kernel[1]
                        ] if (g != 1 or is_dw) else [oc, ic, kernel[0], kernel[1]]
        if is_deconv:
            permute_order = [1, 0, 2, 3]
            filter_op = self.blob_to_weight_op(layer, 0, filter_shape, permute_order)
        else:
            filter_op = self.blob_to_weight_op(layer, 0, filter_shape)
        operands.append(filter_op)
        # bias op
        if with_bias:
            bias_op = self.blob_to_weight_op(layer, 1)
            operands.append(bias_op)
        else:
            operands.append(self.add_none_op())

        output_shape = [n, oc, ofmap[0], ofmap[1]]
        conv_param = {
            'kernel_h': kernel[0],
            'kernel_w': kernel[1],
            'dilation_h': dilation[0],
            'dilation_w': dilation[1],
            'stride_h': stride[0],
            'stride_w': stride[1],
            'padding': padding_str,
            'padding_t': padding[0],
            'padding_b': padding[0],
            'padding_l': padding[1],
            'padding_r': padding[1],
            'group': g,
            'is_dw': is_dw,
            'with_bias': with_bias,
            'do_relu': False,
            'ins': [],
        }
        if not is_deconv:
            new_op = self.mlir.add_conv_op(
                layer.name, operands, output_shape, **conv_param)
            self.addOperand(layer.top[0], new_op, output_shape)
        else:
            new_op = self.mlir.add_deconv_op(
                layer.name, operands, output_shape, **conv_param)
            self.addOperand(layer.top[0], new_op, output_shape)

    def convert_crop_op(self, layer):
        assert(self.layerType(layer) == 'Crop')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        crop_op, crop_shape, _ = self.getOperand(layer.bottom[1])
        p = layer.crop_param
        input_dim = len(input_shape)
        axis_index = p.axis
        start_axis = axis_index
        offset_size = len(p.offset)
        if offset_size > 1:
            assert(offset_size + axis_index <= input_dim)
        output_shape = list(input_shape)
        crop_offset = list(input_shape)
        for i in range(input_dim):
            offset = 0
            new_size = input_shape[i]
            if i >= start_axis:
                new_size = crop_shape[i]
                if offset_size == 1:
                    # If only one offset is given, all crops have the same offset.
                    offset = p.offset[0]
                elif offset_size > 1:
                    # For several offsets, the number of offsets must be equal to the
                    # number of dimensions to crop, that is dimensions after the axis.
                    offset = p.offset[i - start_axis]
            output_shape[i] = new_size
            crop_offset[i] = offset
        # TODO(charle.hu):if crop_op is dummy op, need to erase?
        operands = list()
        operands.append(op)
        param = {
            'crop_offset': crop_offset,
        }
        new_op = self.mlir.add_crop_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_detection_output_op(self, layer):
        assert(self.layerType(layer) == "DetectionOutput")
        _, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        for bottom in layer.bottom:
            op, _, _ = self.getOperand(bottom)
            operands.append(op)
        p = layer.detection_output_param
        code_type = "CORNER"
        if p.code_type == 2:
            code_type = "CENTER_SIZE"
        elif p.code_type == 3:
            code_type = "CORNER_SIZE"
        param = {
            'num_classes': p.num_classes,
            'share_location': p.share_location,
            'background_label_id': p.background_label_id,
            'nms_threshold': p.nms_param.nms_threshold,
            'top_k': p.nms_param.top_k,
            'code_type': code_type,
            'keep_top_k': p.keep_top_k,
            'confidence_threshold': p.confidence_threshold
        }
        assert(1.0 == p.nms_param.eta)
        assert(False == p.variance_encoded_in_target)
        output_shape = [input_shape[0], 1, p.keep_top_k, 7]
        new_op = self.mlir.add_detection_output_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_dropout_op(self, layer):
        assert(self.layerType(layer) == 'Dropout')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        self.addOperand(layer.top[0], op, input_shape)

    def convert_dummydata_op(self, layer):
        assert(self.layerType(layer) == 'DummyData')
        operands = list()
        p = layer.dummy_data_param
        assert(len(p.shape) > 0)
        output_shape = list(p.shape[0].dim)
        new_op = self.mlir.add_dummydata_op(layer.name, operands, output_shape)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_embed_op(self, layer):
        assert(self.layerType(layer) == 'Embed')
        operands = list()
        input_op,input_shape,_ = self.getOperand(layer.bottom[0])
        operands.append(input_op)
        weight_op = self.blob_to_weight_op(layer, 0)
        operands.append(weight_op)
        output_shape = list(input_shape)
        N = layer.embed_param.num_output
        output_shape.append(N)
        new_op = self.mlir.add_embedding_op(layer.name, operands, output_shape)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_eltwise_op(self, layer):
        assert(self.layerType(layer) == 'Eltwise')
        operands = list()
        for bottom in layer.bottom:
            op, _, _ = self.getOperand(bottom)
            operands.append(op)
        _, input_shape, _ = self.getOperand(layer.bottom[0])
        p = layer.eltwise_param
        operation = p.operation
        output_shape = input_shape
        input_num = len(layer.bottom)

        if operation == 0:
            assert(len(p.coeff) == 0)
            new_op = self.mlir.add_eltwise_mul_op(
                layer.name, operands, output_shape)
            self.addOperand(layer.top[0], new_op, output_shape)
        elif operation == 1:
            coeff = None
            if (len(p.coeff) != 0):
                assert(len(p.coeff) == input_num)
                coeff = [c for c in p.coeff]
            else:
                assert(len(p.coeff) == 0)
                coeff = [1] * input_num

            param = {
                'coeff': coeff
            }
            new_op = self.mlir.add_eltwise_add_op(
                layer.name, operands, output_shape, **param)
            self.addOperand(layer.top[0], new_op, output_shape)
        elif operation == 2:
            assert(len(p.coeff) == 0)
            new_op = self.mlir.add_eltwise_max_op(
                layer.name, operands, output_shape)
            self.addOperand(layer.top[0], new_op, output_shape)
        elif operation == 3:
            assert(len(p.coeff) == 0)
            new_op = self.mlir.add_eltwise_min_op(
                layer.name, operands, output_shape)
            self.addOperand(layer.top[0], new_op, output_shape)

    def convert_flatten_op(self, layer):
        assert(self.layerType(layer) == 'Flatten')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        num_dims = len(input_shape)
        assert(num_dims > 1)
        output_shape = [input_shape[0], 1]
        for i in range(1, num_dims):
            output_shape[1] *= input_shape[i]
        new_op = self.mlir.add_reshape_op(layer.name, operands, output_shape)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_frcn_detection_op(self, layer):
        assert(self.layerType(layer) == 'FrcnDetection')
        _, input_shape, _ = self.getOperand(layer.bottom[2])
        operands = list()
        for bottom in layer.bottom:
            op, _, _ = self.getOperand(bottom)
            operands.append(op)
        p = layer.frcn_detection_param
        class_num = p.class_num
        obj_threshold = p.obj_threshold
        nms_threshold = p.nms_threshold
        keep_topk = p.keep_topk
        param = {
            'class_num': class_num,
            'obj_threshold': obj_threshold,
            'nms_threshold': nms_threshold,
            'keep_topk': keep_topk
        }

        output_shape = [input_shape[0], 1, keep_topk, 6]

        new_op = self.mlir.add_frcn_detection_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op, output_shape)

    def convert_inner_product_op(self, layer):
        assert(self.layerType(layer) == 'InnerProduct')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        p = layer.inner_product_param
        with_bias = p.bias_term
        axis = p.axis
        with_transpose = p.transpose
        assert(with_transpose == False)  # transpose not support now
        N = p.num_output
        output_shape = list()
        for i in range(axis):
            output_shape.append(input_shape[i])
        output_shape.append(N)
        reshape_first = False
        K = 1
        for i in range(axis, len(input_shape)):
            K *= input_shape[i]
        operands.append(op)
        # filter
        filter_op = self.blob_to_weight_op(layer, 0, [N, K])
        operands.append(filter_op)
        if with_bias:
            bias_op = self.blob_to_weight_op(layer, 1, [N])
            operands.append(bias_op)
        else:
            operands.append(self.add_none_op())
        new_op = self.mlir.add_fully_connected_op(
            layer.name, operands, output_shape)
        self.addOperand(layer.top[0], new_op, output_shape)

    def convert_input_op(self, layer):
        assert(self.layerType(layer) == 'Input')

    def convert_interp_op(self, layer):
        assert(self.layerType(layer) == 'Interp')
        raise RuntimeError("Not Support")

    def convert_lrn_op(self, layer):
        assert(self.layerType(layer) == 'LRN')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        p = layer.lrn_param
        param = {
            'alpha': p.alpha,
            'beta': p.beta,
            'bias': p.k,
            'size': p.local_size,
        }
        output_shape = input_shape
        new_op = self.mlir.add_lrn_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_lstm_op(self, layer):
        assert(self.layerType(layer) == 'LSTM')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        seq_length = input_shape[0]
        batch_size = input_shape[1]
        input_size = input_shape[2]
        hidden_size = layer.recurrent_param.num_output

        operands = list()
        operands.append(op)
        # weight
        weight = self.layer_dict[layer.name].blobs[0].data
        weight = weight.reshape([4, hidden_size*input_size])
        weight[[1, 2], :] = weight[[2, 1], :]  # ifoc =>iofc
        weight = weight.reshape([4*hidden_size, input_size])
        weight_name = layer.name + "_0"
        self.addTensor(weight_name, weight, weight.shape)
        weight_op = self.mlir.add_load_file_op(weight_name, weight.shape)
        operands.append(weight_op)
        # bias
        bias = self.layer_dict[layer.name].blobs[1].data
        bias = bias.reshape([4, hidden_size])
        bias[[1, 2], :] = bias[[2, 1], :]  # ifoc =>iofc
        bias = bias.reshape([1, 4*hidden_size])
        bias_name = layer.name + "_2"
        self.addTensor(bias_name, bias, bias.shape)
        bias_op = self.mlir.add_load_file_op(bias_name, bias.shape)
        operands.append(bias_op)
        # FC
        fc_shape = [seq_length, batch_size, 4*hidden_size]
        fc_op = self.mlir.add_fully_connected_op(
            "{}_x_transform".format(layer.name), operands, fc_shape)

        operands.clear()
        operands.append(fc_op)
        # recurrence
        r = self.layer_dict[layer.name].blobs[2].data
        r = r.reshape([4, hidden_size*hidden_size])
        r[[1, 2], :] = r[[2, 1], :]  # ifoc =>iofc
        r = r.reshape([1, 4*hidden_size, hidden_size])
        recurrence_name = layer.name + "_1"
        self.addTensor(recurrence_name, r, r.shape)
        recurrence_op = self.mlir.add_load_file_op(recurrence_name, r.shape)
        operands.append(recurrence_op)

        if len(layer.bottom) > 1:
            cont_op, _, _ = self.getOperand(layer.bottom[1])
            operands.append(self.add_none_op()) # bias
            operands.append(self.add_none_op()) # h
            operands.append(self.add_none_op()) # c
            operands.append(cont_op)

        lstm_param = {
            'bidirectional': bool(False),
        }
        lstm_shape = [seq_length, 1, batch_size, hidden_size]
        lstm_name = layer.name + "_lstm"
        lstm_op = self.mlir.add_lstm_op(
            lstm_name, operands, lstm_shape, **lstm_param)

        # reshape to [seq_length, batch_size, hidden_size]
        operands.clear()
        operands.append(lstm_op)
        output_shape = [seq_length, batch_size, hidden_size]
        new_op = self.mlir.add_reshape_op(layer.name, operands, output_shape)
        self.addOperand(layer.top[0], new_op, output_shape)

    def convert_lstm_jun_op(self, layer):
        assert(self.layerType(layer) == 'Lstm')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        seq_length = input_shape[0]
        batch_size = input_shape[1]
        input_size = input_shape[2]
        hidden_size = layer.lstm_param.num_output

        operands = list()
        operands.append(op)
        # weight
        weight = self.layer_dict[layer.name].blobs[0].data
        weight = weight.reshape([4, hidden_size*input_size])
        weight[[1, 2], :] = weight[[2, 1], :]  # ifoc =>iofc
        weight = weight.reshape([4*hidden_size, input_size])
        weight_name = layer.name + "_0"
        self.addTensor(weight_name, weight, weight.shape)
        weight_op = self.mlir.add_load_file_op(weight_name, weight.shape)
        operands.append(weight_op)
        # bias
        bias = self.layer_dict[layer.name].blobs[2].data
        bias = bias.reshape([4, hidden_size])
        bias[[1, 2], :] = bias[[2, 1], :]  # ifoc =>iofc
        bias = bias.reshape([1, 4*hidden_size])
        bias_name = layer.name + "_2"
        self.addTensor(bias_name, bias, bias.shape)
        bias_op = self.mlir.add_load_file_op(bias_name, bias.shape)
        operands.append(bias_op)
        # FC
        fc_shape = [seq_length, batch_size, 4*hidden_size]
        fc_op = self.mlir.add_fully_connected_op(
            "{}_x_transform".format(layer.name), operands, fc_shape)

        operands.clear()
        operands.append(fc_op)
        # recurrence
        r = self.layer_dict[layer.name].blobs[1].data
        r = r.reshape([4, hidden_size*hidden_size])
        r[[1, 2], :] = r[[2, 1], :]  # ifoc =>iofc
        r = r.reshape([1, 4*hidden_size, hidden_size])
        recurrence_name = layer.name + "_1"
        self.addTensor(recurrence_name, r, r.shape)
        recurrence_op = self.mlir.add_load_file_op(recurrence_name, r.shape)
        operands.append(recurrence_op)
        if len(layer.bottom) > 1:
            cont_op, _, _ = self.getOperand(layer.bottom[1])
            operands.append(self.add_none_op()) # bias
            operands.append(self.add_none_op()) # h
            operands.append(self.add_none_op()) # c
            operands.append(cont_op)

        lstm_param = {
            'bidirectional': bool(False),
        }
        lstm_shape = [seq_length, 1, batch_size, hidden_size]
        lstm_name = layer.name + "_lstm"
        lstm_op = self.mlir.add_lstm_op(
            lstm_name, operands, lstm_shape, **lstm_param)

        # reshape to [seq_length, batch_size, hidden_size]
        operands.clear()
        operands.append(lstm_op)
        output_shape = [seq_length, batch_size, hidden_size]
        new_op = self.mlir.add_reshape_op(layer.name, operands, output_shape)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_normalize_op(self, layer):
        assert(self.layerType(layer) == 'Normalize')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        p = layer.norm_param
        param = {
            'across_spatial': p.across_spatial,
            'channel_shared': p.channel_shared,
        }
        assert(False == p.across_spatial)
        assert(len(input_shape) > 1)
        c = input_shape[1]
        # scale
        scale_shape = [1, c]
        scale_name = layer.name + "_0"
        blob = self.layer_dict[layer.name].blobs[0]
        scale_data = np.array
        if p.channel_shared:
            assert(blob.count == 1)
            value = blob.data.flatten()[0]
            scale_data = np.array([[value for i in range(c)]], dtype=float)
        else:
            assert(blob.count == c)
            scale_data = blob.data.reshape(scale_shape)
        self.addTensor(scale_name, scale_data, scale_shape)
        scale_op = self.mlir.add_load_file_op(scale_name, scale_shape)
        operands.append(scale_op)
        output_shape = input_shape
        new_op = self.mlir.add_normalize_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_mish_op(self, layer):
        assert(self.layerType(layer) == 'Mish')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        output_shape = input_shape
        new_op = self.mlir.add_mish_op(
            layer.name, operands, output_shape)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_padding_op(self, layer):
        assert(self.layerType(layer) == 'Padding')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)

        pad_t = layer.padding_param.pad_t
        pad_l = layer.padding_param.pad_l
        pad_b = layer.padding_param.pad_b
        pad_r = layer.padding_param.pad_r
        pads = [0, 0, pad_t, pad_l, 0, 0, pad_b, pad_r]

        dims = len(input_shape)
        output_shape = np.sum([input_shape, pads[:dims], pads[dims:]], axis=0)
        output_shape = [int(i) for i in output_shape]

        pads_param = {
          "pads": pads,
          "const_val": layer.padding_param.pad_value,
        }
        pads_op = self.mlir.add_pad_op(layer.name, operands, output_shape, **pads_param)
        self.addOperand(layer.top[0], pads_op, output_shape)

    def convert_permute_op(self, layer):
        assert(self.layerType(layer) == 'Permute')
        _, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        for bottom in layer.bottom:
            op, _, _ = self.getOperand(bottom)
            operands.append(op)
        p = layer.permute_param
        output_shape = list(input_shape)
        for i in range(len(p.order)):
            output_shape[i] = input_shape[p.order[i]]
        param = {
            'order': p.order,
        }
        new_op = self.mlir.add_permute_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_pooling_op(self, layer):
        assert(self.layerType(layer) == 'Pooling')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        p = layer.pooling_param
        pool_method = p.pool
        assert(pool_method == 0 or pool_method == 1)
        n = input_shape[0]
        c = input_shape[1]
        ifmap = [input_shape[2], input_shape[3]]
        kernel = [input_shape[2], input_shape[3]]
        is_global_pooling = p.global_pooling
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
        ceil_mode = p.ceil_mode
        round_mode = p.round_mode
        if round_mode == 1:
            ceil_mode = False
        padding_tl = [padding[0], padding[1]]
        padding_br = [padding[0], padding[1]]
        ofmap = [0, 0]
        yolo = p.yolo if p.HasField('yolo') else False
        for i in [0, 1]:
            if ceil_mode:
                ofmap[i] = math.ceil(
                    (ifmap[i] + 2.0 * padding[i] - kernel[i]) / stride[i]) + 1
            else:
                ofmap[i] = math.floor(
                    (ifmap[i] + 2.0 * padding[i] - kernel[i]) / stride[i]) + 1
            remain_pixel = (ifmap[i] + 2 * padding[i] - kernel[i]) % stride[i]
            if remain_pixel > 0:
                if ceil_mode:
                    padding_br[i] += (stride[i] - remain_pixel)
                else:
                    padding_br[i] -= remain_pixel
        if yolo:
            # for stride = 1, add padding right and bottom
            if stride[0] == 1 and stride[1] == 1:
                assert(padding_tl[0] == 0)
                assert(padding_tl[1] == 0)
                assert(padding_br[0] == 0)
                assert(padding_br[1] == 0)
                padding_br[0] = 1
                padding_br[1] = 1

                ofmap[0] += 1
                ofmap[1] += 1

        if is_global_pooling:
            assert((padding[0] == 0) and (padding[1] == 0))
            assert((stride[0] == 1) and (stride[1] == 1))
            assert((ofmap[0] == 1) and (ofmap[1] == 1))
        pool_param = {
            'kernel_h': kernel[0],
            'kernel_w': kernel[1],
            'padding_t': padding_tl[0],
            'padding_b': padding_br[0],
            'padding_l': padding_tl[1],
            'padding_r': padding_br[1],
            'stride_h': stride[0],
            'stride_w': stride[1],
            'do_relu': False,
            'count_include_pad': True,
        }
        output_shape = [n, c, ofmap[0], ofmap[1]]
        if pool_method == 0:  # MAX
            new_op = self.mlir.add_pool_max_2d_op(
                layer.name, operands, output_shape, **pool_param)
            self.addOperand(layer.top[0], new_op, output_shape)
        elif pool_method == 1:  # AVE
            new_op = self.mlir.add_pool_avg_2d_op(
                layer.name, operands, output_shape, **pool_param)
            self.addOperand(layer.top[0], new_op, output_shape)
        if len(layer.top) > 1:
            assert(kernel[0] == kernel[1] and stride[0] == stride[1] and kernel[0] == stride[0])
            param = {
                'scale': kernel[0]
            }
            operands = list()
            operands.append(op)
            pool_mask_name = "{}_mask".format(layer.name)
            mask_shape = list(input_shape)
            mask_shape[2] = self.align_up(mask_shape[2], kernel[0])
            mask_shape[3] = self.align_up(mask_shape[3], kernel[0])
            pool_mask_op = self.mlir.add_pool_mask_op(
                pool_mask_name, operands, mask_shape, **param)
            self.addOperand(layer.top[1], pool_mask_op, mask_shape)

    def convert_power_op(self, layer):
        assert (self.layerType(layer) == 'Power')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        p = layer.power_param
        if p.shift == 0 and p.power == 1 and p.scale == 1:
            # do nothing
            return self.addOperand(layer.top[0], op, input_shape)
        output_shape = list(input_shape)
        operands = []
        if p.power != 1.0:
            param = {'coeff': p.power}
            power_name = layer.name
            if p.shift != 0 or p.scale != 1:
                power_name += "_pow"
            new_op = self.mlir.add_pow_op(power_name, [op], output_shape, **param)
            operands.append(new_op)
        else:
            operands.append(op)
        if p.shift != 0 or p.scale != 1:
            # convert to scale op
            # weight scale
            scale_name = layer.name + "_0"
            c = input_shape[1]
            scale_shape = [c]
            scale_data = np.array([p.scale for i in range(c)], dtype=float)
            self.addTensor(scale_name, scale_data, scale_shape)
            scale_op = self.mlir.add_load_file_op(scale_name, scale_shape)
            operands.append(scale_op)
            # weight bias
            if p.shift != 0.0:
                bias_name = layer.name + "_1"
                bias_shape = [c]
                bias_data = np.array([p.shift for i in range(c)], dtype=float)
                self.addTensor(bias_name, bias_data, bias_shape)
                bias_op = self.mlir.add_load_file_op(bias_name, bias_shape)
                operands.append(bias_op)
            else:
                operands.append(self.add_none_op())
            new_op = self.mlir.add_scale_op(layer.name, operands, output_shape)
        self.addOperand(layer.top[0], new_op, output_shape)

    def convert_prelu_op(self, layer):
        assert(self.layerType(layer) == 'PReLU')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        dim_len = len(input_shape)
        assert(dim_len > 1)
        slope_shape = [1] * dim_len
        slope_shape[1] = input_shape[1]
        # negative_slope
        slope_op = self.blob_to_weight_op(layer, 0, slope_shape, None, 1)
        operands.append(slope_op)
        output_shape = input_shape
        new_op = self.mlir.add_prelu_op(layer.name, operands, output_shape)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_priorbox_op(self, layer):
        assert(self.layerType(layer) == 'PriorBox')
        op0, input_shape0, _ = self.getOperand(layer.bottom[0])
        op1, input_shape1, _ = self.getOperand(layer.bottom[1])
        operands = list()
        operands.append(op0)
        operands.append(op1)
        assert(len(input_shape0) == 4)
        h = input_shape0[2]
        w = input_shape0[3]
        p = layer.prior_box_param
        min_size = [i for i in p.min_size]
        max_size = [i for i in p.max_size]
        aspect_ratio = [i for i in p.aspect_ratio]
        variance = [i for i in p.variance]
        assert(len(variance) == 4)

        param = {
            'min_size': min_size,
            'max_size': max_size,
            'variance': variance,
            'clip': p.clip,
            'offset': p.offset,
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
            assert(len(max_size) == len(min_size))
            num_priors += len(max_size)
        param['num_priors'] = num_priors
        param['aspect_ratios'] = aspect_ratios_
        param['use_default_aspect_ratio'] = use_default_aspect_ratio
        output_shape = [1, 2, int(h * w * num_priors * 4)]
        new_op = self.mlir.add_priorbox_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_proposal_op(self, layer):
        assert(self.layerType(layer) == 'Proposal')
        _, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        for bottom in layer.bottom:
            op, _, _ = self.getOperand(bottom)
            operands.append(op)
        p = layer.proposal_param
        feat_stride = p.feat_stride
        anchor_base_size = p.anchor_base_size
        rpn_obj_threshold = p.rpn_obj_threshold
        rpn_nms_threshold = p.rpn_nms_threshold
        rpn_nms_post_top_n = p.rpn_nms_post_top_n
        net_input_h = p.net_input_h
        net_input_w = p.net_input_w
        param = {
            'net_input_h': net_input_h,
            'net_input_w': net_input_w,
            'feat_stride': feat_stride,
            'anchor_base_size': anchor_base_size,
            'rpn_obj_threshold': rpn_obj_threshold,
            'rpn_nms_threshold': rpn_nms_threshold,
            'rpn_nms_post_top_n': rpn_nms_post_top_n
        }
        output_shape = [input_shape[0], 1, rpn_nms_post_top_n, 5]
        new_op = self.mlir.add_proposal_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_relu_op(self, layer):
        assert(self.layerType(layer) == 'ReLU')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        assert(len(input_shape) == 4 or len(input_shape) == 2)
        negative_slope = layer.relu_param.negative_slope
        output_shape = input_shape
        if negative_slope == 0.0:
            new_op = self.mlir.add_relu_op(
                layer.name, operands, output_shape)
            self.addOperand(layer.top[0], new_op, output_shape)
        else:
            param = {
                'negative_slope': negative_slope
            }
            new_op = self.mlir.add_leaky_relu_op(
                layer.name, operands, output_shape, **param)
            self.addOperand(layer.top[0], new_op, output_shape)

    def convert_relu6_op(self, layer):
        assert(self.layerType(layer) == 'ReLU6')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        assert(len(input_shape) == 4 or len(input_shape) == 2)
        negative_slope = layer.relu_param.negative_slope
        output_shape = input_shape

        clip_param = {
            'min': 0.0,
            'max': 6.0,
        }
        if negative_slope == 0.0:
            clip_op = self.mlir.add_clip_op(layer.name, operands, output_shape, **clip_param)
            self.addOperand(layer.top[0], clip_op, output_shape)
        else:
            param = {
                'negative_slope': negative_slope
            }
            leaky_relu_op = self.mlir.add_leaky_relu_op(
                layer.name, operands, output_shape, **param)
            clip_op = self.mlir.add_clip_op(layer.name, [leaky_relu_op], output_shape, **clip_param)
            self.addOperand(layer.top[0], clip_op, output_shape)

    def convert_reorg_op(self, layer):
        assert(self.layerType(layer) == 'Reorg')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        assert(len(input_shape) == 4)
        reverse = layer.reorg_param.reverse
        assert(reverse == False)
        stride = layer.reorg_param.stride
        output_shape = list(input_shape)
        output_shape[0] = input_shape[0]
        output_shape[1] = int(input_shape[1] * stride * stride)
        output_shape[2] = int(input_shape[2] / stride)
        output_shape[3] = int(input_shape[3] / stride)
        param = {
            'stride': stride
        }
        new_op = self.mlir.add_reorg_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_reshape_op(self, layer):
        assert(self.layerType(layer) == 'Reshape')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        num_dims = len(input_shape)
        assert(num_dims == 4 or num_dims == 2)
        p = layer.reshape_param
        top_dims = list(p.shape.dim)
        output_shape = list()
        input_count = 1
        for dim in input_shape:
            input_count *= dim
        if num_dims == 4:
            num_axes = p.num_axes
            start_axis = p.axis if p.axis >= 0 else (
                num_dims + p.axis + 1)
            assert(start_axis >= 0)
            assert(start_axis <= num_dims)
            assert(num_axes >= -1)
            end_axis = num_dims if num_axes == -1 else (start_axis + num_axes)
            assert(end_axis <= num_dims)

            num_axes_replaced = end_axis - start_axis
            num_axes_retained = num_dims - num_axes_replaced
            num_new_axes = len(top_dims)
            has_inferred = False
            for dim in top_dims:
                if dim == -1:
                    has_inferred = True
                    break
            if start_axis == 0 and top_dims[0] != 0 and \
               top_dims[0] != -1 and has_inferred == False and \
               top_dims[0] != input_shape[0]:
                top_dims[0] = -1

            copy_axes = list()
            inferred_axis = -1
            constant_count = 1
            for i in range(num_new_axes):
                dim = top_dims[i]
                if dim == 0:
                    copy_axes.append(i)
                elif dim == -1:
                    assert(inferred_axis == -1)
                    inferred_axis = i
                else:
                    constant_count *= dim
            output_shape = [0] * (num_axes_retained + num_new_axes)
            output_shape_index = 0
            for i in range(start_axis):
                output_shape[output_shape_index] = input_shape[i]
                output_shape_index += 1
            for i in range(num_new_axes):
                output_shape[output_shape_index] = top_dims[i]
                output_shape_index += 1
            for i in range(end_axis, num_dims):
                output_shape[output_shape_index] = input_shape[i]
                output_shape_index += 1
            assert(output_shape_index == len(output_shape))
            for i in range(len(copy_axes)):
                copy_axis_index = copy_axes[i]
                assert(num_dims > start_axis + copy_axis_index)
                output_shape[start_axis +
                             copy_axis_index] = input_shape[start_axis + copy_axis_index]

            if inferred_axis >= 0:
                # A -1 dim was specified; infer the correct dimension by computing the
                # product of the other dimensions.
                explicit_count = constant_count
                for i in range(start_axis):
                    explicit_count *= input_shape[i]
                for i in range(end_axis, num_dims):
                    explicit_count *= input_shape[i]
                for i in range(len(copy_axes)):
                    copy_axis_index = copy_axes[i]
                    explicit_count *= (output_shape[start_axis +
                                                    copy_axis_index])
                assert(0 == input_count % explicit_count)
                inferred_dim = input_count / explicit_count
                output_shape[start_axis + inferred_axis] = int(inferred_dim)
        elif num_dims == 2:
            output_shape = list()
            inference_dim = -1
            for i in range(len(top_dims)):
                dim = top_dims[i]
                if dim == 0:
                    output_shape.append(int(input_shape[i]))
                    input_count /= output_shape[i]
                elif dim == -1:
                    assert(inference_dim == -1)
                    inference_dim = i
                    output_shape.append(int(0))
                else:
                    output_shape.append(int(top_dims[i]))
                    input_count /= output_shape[i]
            if inference_dim != -1:
                output_shape[inference_dim] = int(input_count)
            else:
                assert(input_count == 1)
        else:
            raise RuntimeError("ReshapeOp only support input shape = 2 or 4 now")
        new_op = self.mlir.add_reshape_op(layer.name, operands, output_shape)
        self.addOperand(layer.top[0], new_op, output_shape)

    def convert_reverse_op(self, layer):
        assert(self.layerType(layer) == 'Reverse')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        axis = layer.reverse_param.axis
        param = {
            'axis': axis,
        }
        output_shape = input_shape
        new_op = self.mlir.add_reverse_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op, output_shape)

    def convert_retinaface_detection_op(self, layer):
        assert(self.layerType(layer) == 'RetinaFaceDetection')
        _, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        for bottom in layer.bottom:
            op, _, _ = self.getOperand(bottom)
            operands.append(op)
        p = layer.retinaface_detection_param
        nms_threshold = p.nms_threshold
        confidence_threshold = p.confidence_threshold
        keep_topk = p.keep_topk
        output_shape = [input_shape[0], 1, keep_topk, 15]
        param = {
            'nms_threshold': nms_threshold,
            'confidence_threshold': confidence_threshold,
            'keep_topk': keep_topk,
        }
        new_op = self.mlir.add_retinaface_detection_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op, output_shape)

    def convert_roipooling_op(self, layer):
        assert(self.layerType(layer) == 'ROIPooling')
        operands = list()
        assert(len(layer.bottom) == 2)
        op0, bottom0_shape, _ = self.getOperand(layer.bottom[0])
        op1, bottom1_shape, _ = self.getOperand(layer.bottom[1])
        operands.append(op0)
        operands.append(op1)
        p = layer.roi_pooling_param
        pooled_h = p.pooled_h
        pooled_w = p.pooled_w
        spatial_scale = p.spatial_scale
        param = {
            'pooled_h': pooled_h,
            'pooled_w': pooled_w,
            'spatial_scale': spatial_scale
        }
        output_shape = [bottom1_shape[0] * bottom1_shape[2], bottom0_shape[1], pooled_h, pooled_w]
        new_op = self.mlir.add_roipooling_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_scale_op(self, layer):
        assert(self.layerType(layer) == 'Scale')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        num_dims = len(input_shape)
        assert(num_dims == 4 or num_dims == 2)
        if len(layer.bottom) == 2:
            op1, _, _ = self.getOperand(layer.bottom[1])
            operands.append(op1)
            output_shape = input_shape
            param = {
                'align_right':False
            }
            new_op = self.mlir.add_broadcast_mul_op(
                layer.name, operands, output_shape, **param)
            self.addOperand(layer.top[0], new_op, output_shape)
        else:
            p = layer.scale_param
            # scale
            scale_op = self.blob_to_weight_op(layer, 0, [input_shape[1]])
            operands.append(scale_op)
            # bias
            if p.bias_term:
                bias_op = self.blob_to_weight_op(layer, 1, [input_shape[1]])
                operands.append(bias_op)
            else:
                operands.append(self.add_none_op())
            output_shape = input_shape
            new_op = self.mlir.add_scale_op(
                layer.name, operands, output_shape)
            self.addOperand(layer.top[0], new_op, output_shape)

    def convert_shufflechannel_op(self, layer):
        assert(self.layerType(layer) == 'ShuffleChannel')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        group = layer.shuffle_channel_param.group
        operands = list()
        operands.append(op)
        param = {
            'group': group
        }
        output_shape = input_shape
        new_op = self.mlir.add_shufflechannel_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_sigmoid_op(self, layer):
        assert(self.layerType(layer) == 'Sigmoid')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        output_shape = input_shape
        new_op = self.mlir.add_sigmoid_op(
            layer.name, operands, output_shape)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_silence_op(self, layer):
        assert(self.layerType(layer) == 'Silence')
        # do nothing now

    def convert_slice_op(self, layer):
        assert(self.layerType(layer) == 'Slice')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        assert(len(input_shape) == 4)
        p = layer.slice_param
        axis = p.axis
        bottom_slice_axis = input_shape[axis]
        top_size = len(layer.top)
        slice_num = len(p.slice_point)
        slices = list()
        if slice_num > 0:
            assert(slice_num == top_size - 1)
            assert(top_size <= bottom_slice_axis)
            prev = 0
            for i in range(slice_num):
                assert(p.slice_point[i] > prev)
                slices.append(p.slice_point[i] - prev)
                prev = p.slice_point[i]
            slices.append(bottom_slice_axis - prev)
        else:
            assert(bottom_slice_axis % top_size == 0)
            for i in range(top_size):
                slices.append(int(bottom_slice_axis / top_size))
        offset = 0
        for i in range(top_size):
            output_shape = list(input_shape)
            output_shape[axis] = slices[i]
            crop_offset = [0] * len(input_shape)
            crop_offset[axis] = offset
            param = {
                'crop_offset': crop_offset,
            }
            new_op = self.mlir.add_crop_op("{}_{}".format(
                layer.name, i), operands, output_shape, **param)
            self.addOperand(layer.top[i], new_op,
                            output_shape)
            offset += slices[i]

    def convert_softmax_op(self, layer):
        assert(self.layerType(layer) == 'Softmax')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        axis = 1
        if layer.HasField('softmax_param') and layer.softmax_param.HasField('axis'):
            axis = layer.softmax_param.axis
        param = {
            'axis': axis
        }
        output_shape = input_shape
        new_op = self.mlir.add_softmax_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op, output_shape)

    def convert_split_op(self, layer):
        assert(self.layerType(layer) == 'Split')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        # simply bypass, register top and bottom blobs to the same tensor
        for top in layer.top:
            self.addOperand(top, op, input_shape)

    def convert_tanh_op(self, layer):
        assert(self.layerType(layer) == 'TanH')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        output_shape = input_shape
        new_op = self.mlir.add_tanh_op(
            layer.name, operands, output_shape)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_tile_op(self, layer):
        assert(self.layerType(layer) == 'Tile')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        assert(len(input_shape) == 4)
        axis = layer.tile_param.axis
        tiles = layer.tile_param.tiles
        output_shape = input_shape
        output_shape[axis] = int(output_shape[axis] * tiles)
        param = {
            'axis': axis,
            'tiles': tiles
        }
        new_op = self.mlir.add_tile_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_upsample_op(self, layer):
        assert(self.layerType(layer) == 'Upsample')
        _, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        op, _, _ = self.getOperand(layer.bottom[0])
        operands.append(op)
        assert(len(input_shape) == 4)
        p = layer.upsample_param
        scale = p.scale
        output_shape = [input_shape[0], input_shape[1],
                        scale * input_shape[2], scale * input_shape[3]]
        if p.HasField('upsample_w'):
            output_shape[3] = p.upsample_w
        if p.HasField('upsample_h'):
            output_shape[2] = p.upsample_h
        param = {
            'scale_h': scale,
            'scale_w': scale
        }

        upsample_name = layer.name
        if len(layer.bottom) > 1:
            op_mask, _, _ = self.getOperand(layer.bottom[1])
            operands.append(op_mask)
        else:
            operands.append(self.add_none_op())

        new_op = self.mlir.add_upsample_op(
            upsample_name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                            output_shape)

    def convert_yolo_detection_op(self, layer):
        assert(self.layerType(layer) == 'YoloDetection')
        _, input_shape, _ = self.getOperand(layer.bottom[0])

        operands = list()
        for bottom in layer.bottom:
            op, _, _ = self.getOperand(bottom)
            operands.append(op)
        p = layer.yolo_detection_param

        if not p.anchors:
            if p.tiny:
                p.anchors = "10,14,23,27,37,58,81,82,135,169,344,319"
            elif p.yolo_v4:
                p.anchors = "142,110,192,243,459,401,36,75,76,55,72,146,12,16,19,36,40,28"
            else:
                p.anchors = "10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326"

        param = {
            'net_input_h': p.net_input_h,
            "net_input_w": p.net_input_w,
            "nms_threshold": p.nms_threshold,
            "obj_threshold": p.obj_threshold,
            "keep_topk": p.keep_topk,
            "spp_net": p.spp_net,
            "tiny": p.tiny,
            "yolo_v4": p.yolo_v4,
            "class_num": p.class_num,
            "anchors": p.anchors
        }
        output_shape = [input_shape[0], 1, p.keep_topk, 6]
        new_op = self.mlir.add_yolo_detection_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape)

    def convert_matmul_op(self, layer):
        assert(self.layerType(layer) == 'MatMul')
        p = layer.matmul_param
        N = p.dim_3
        M = p.dim_1
        K = p.dim_2

        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        assert(len(input_shape) == 4 or len(input_shape) == 2)
        reshape_first = False
        if len(input_shape) > 2:
            reshape_first = True
        fc_op_0 = op
        if reshape_first:
            fc_shape = [M, K]
            fc_operands = [op]
            fc_op_0 = self.mlir.add_reshape_op(
                layer.name + '_reshape_0', fc_operands, fc_shape)
        operands.append(fc_op_0)
        # filter
        op, input_shape, _ = self.getOperand(layer.bottom[1])
        assert(len(input_shape) == 4 or len(input_shape) == 2)
        reshape_first = False
        if len(input_shape) > 2:
            reshape_first = True
        fc_op_1 = op
        if reshape_first:
            fc_shape = [N, K]
            fc_operands = [op]
            fc_op_1 = self.mlir.add_reshape_op(
                layer.name + '_reshape_1', fc_operands, fc_shape)
        operands.append(fc_op_1)

        output_shape = [M, N]
        new_op = self.mlir.add_matmul_op(
            layer.name, operands, output_shape)
        self.addOperand(layer.top[0], new_op, output_shape)

    def convert_graph(self):
        """convert all to mlir"""

        def NoneAndRaise(layer):
            raise RuntimeError(
                "{} Op not support now".format(self.layerType(layer)))

        # add input op
        for idx, name in enumerate(self.inputs):
            input_shape = list(self.blobs[name].shape)
            if self.batch_size != 0:
                input_shape[0] = self.batch_size
            image = (len(input_shape) == 4 and input_shape[1] <=4)
            if not self.preprocess_args or not image:
                input_op = self.mlir.add_input_op(name, idx, **{})
            else:
                preprocess_hint = {
                    'mean': self.preprocess_args['perchannel_mean'],
                    'scale':  self.preprocess_args['perchannel_scale'],
                    'pixel_format': self.preprocess_args["pixel_format"],
                    'channel_order': self.preprocess_args['channel_order'],
                    'keep_aspect_ratio': self.preprocess_args['keep_aspect_ratio'],
                    'resize_dims': self.preprocess_args['resize_dims'],
                    'aligned': self.preprocess_args["aligned"],
                }
                # add input op
                input_op = self.mlir.add_input_op(name, idx, **preprocess_hint)
            self.addOperand(name, input_op, input_shape)

        for layer in self.layers:
            is_test_phase = True
            if len(layer.include) != 0:
                # only test phase convert
                is_test_phase = False
                for include in layer.include:
                    if include.HasField('phase') and include.phase == 1:
                        is_test_phase = True
                        break
            if is_test_phase:
                self.caffeop_factory.get(
                    self.layerType(layer), lambda x: NoneAndRaise(x))(layer)

        # add return op
        return_op = list()
        # Set output
        for output in self.outputs:
            op, _, _ = self.getOperand(output)
            return_op.append(op)

        self.mlir.add_return_op(return_op)
        mlir_txt = self.mlir.print_module()
        with open(self.mlir_file_path, "w") as f:
            f.write(mlir_txt)
        print("Save mlir file: {}".format(self.mlir_file_path))

    def run(self):
        self.convert_graph()
        self.TensortoNpz()
