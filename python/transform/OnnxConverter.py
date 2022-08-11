# Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
#
# Licensed under the Apache License v2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ==============================================================================

# ONNX Node define:
# https://github.com/onnx/onnx/blob/main/docs/Operators.md

from .MLIRImporter import MLIRImporter
from .BaseConverter import BaseConverter
from onnx import numpy_helper, mapping
from numbers import Number
import onnxsim.onnx_simplifier as onnxsim

import onnx
import onnxruntime
import numpy as np
from utils.pad_setting import get_TF_SAME_Padding, set_auto_pad

onnx_attr_translator = {
    "axis": lambda x: int(x),
    "axes": lambda x: [int(a) for a in x],
    "dtype": lambda x: onnx_dtype(x),
    "keepdims": lambda x: bool(x),
    "to": lambda x: onnx_dtype(x),
}


def translate_onnx(key, val):
    return onnx_attr_translator.get(key, lambda x: x)(val)


def onnx_dtype(dtype):
    if isinstance(dtype, Number):
        onnx_dtype = dtype
    elif isinstance(dtype, str):
        onnx_dtype = onnx.TensorProto.DataType.Value(dtype)
    else:
        raise RuntimeError("dtype should be number or str.")
    return mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype]


def convert_onnx_attribute_proto(attr_proto):
    if attr_proto.HasField('f'):
        return attr_proto.f
    elif attr_proto.HasField('i'):
        return attr_proto.i
    elif attr_proto.HasField('s'):
        return attr_proto.s
    elif attr_proto.HasField('t'):
        return attr_proto.t  # this is a proto!
    elif attr_proto.floats:
        return list(attr_proto.floats)
    elif attr_proto.ints:
        return list(attr_proto.ints)
    elif attr_proto.strings:
        str_list = list(attr_proto.strings)
        return str_list
    elif attr_proto.name:
        name_list = list(attr_proto.name)
        return name_list
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(attr_proto))


class BaseNode():

    def __init__(self, info):
        self.name = str(info["name"])
        self.op_type = str(info["op_type"])
        self.attrs = dict(info["attrs"])
        self.inputs = list(info["inputs"])
        self.outputs = list(info["outputs"])


class OnnxNode(BaseNode):

    def __init__(self, node):
        info = dict()
        info["name"] = node.output[0]
        info["op_type"] = node.op_type
        info["attrs"] = [(attr.name, \
                          translate_onnx(attr.name, convert_onnx_attribute_proto(attr))) \
                          for attr in node.attribute]
        info["inputs"] = node.input
        info["outputs"] = node.output
        super().__init__(info)
        self.node_proto = node


class OnnxConverter(BaseConverter):

    def __init__(self,
                 model_name: str,
                 onnx_file,
                 input_shapes: list,
                 output_names: list,
                 preprocess_args=None):
        super().__init__()
        self.model_name = model_name
        self.weight_file = "{}_top_weight.npz".format(model_name)
        self.model = None
        self.mlir = None
        self.load_onnx_model(onnx_file, input_shapes, output_names)
        self.init_MLIRImporter()
        self.preprocess_args = preprocess_args

        self.onnxop_factory = {
            "Add": lambda node: self.convert_add_op(node),
            "AveragePool": lambda node: self.convert_avgpool_op(node),
            "BatchNormalization": lambda node: self.convert_batchnorm_op(node),
            "Concat": lambda node: self.convert_concat_op(node),
            "Conv": lambda node: self.convert_conv_op(node),
            "DepthToSpace": lambda node: self.convert_depth2space_op(node),
            "Flatten": lambda node: self.convert_flatten_op(node),
            "Gemm": lambda node: self.convert_gemm_op(node),
            "GlobalAveragePool": lambda node: self.convert_global_avgpool_op(node),
            "GlobalMaxPool": lambda node: self.convert_global_maxpool_op(node),
            "MaxPool": lambda node: self.convert_maxpool_op(node),
            "Mul": lambda node: self.convert_mul_op(node),
            "Relu": lambda node: self.convert_relu_op(node),
            "Reshape": lambda node: self.convert_reshape_op(node),
            "Resize": lambda node: self.convert_resize_op(node),
            "Sigmoid": lambda node: self.convert_sigmoid_op(node),
            "Slice": lambda node: self.convert_slice_op(node),
            "Transpose": lambda node: self.convert_transpose_op(node),
            "LeakyRelu": lambda node: self.convert_leaky_relu_op(node),
            "Dropout": lambda node: self.convert_skip_op(node),
            "Softmax": lambda node: self.convert_softmax_op(node),
            "Log": lambda node: self.convert_log_op(node),
            "Pad": lambda node: self.convert_pad_op(node),
            "Div": lambda node: self.convert_div_op(node),
            "Squeeze": lambda node: self.convert_squeeze_op(node),
            "Clip": lambda node: self.convert_clip_op(node),
            "ConvTranspose": lambda node: self.convert_conv_transpose_op(node),
            'Split' :  lambda node: self.convert_split_op(node),
        }

    def __del__(self):
        if self.mlir != None:
            del self.mlir
            self.mlir = None

    def select_unuse(self, names):
        for name in names:
            if name in self.all_weights:
                self.all_weights.pop(name)
            if name in self.all_values:
                self.all_values.pop(name)
            if name in self.all_inputs:
                self.all_inputs.pop(name)
            if name in self.all_nodes:
                cur_node = self.all_nodes.pop(name)
                for o in cur_node.output:
                    if o in self.all_nodes:
                        self.all_nodes.pop(o)
                self.select_unuse(cur_node.input)

    def select_output(self, output_names: list):
        # set new output
        self.all_outputs = []
        self.all_inputs = {}
        for x in self.model.graph.input:
            self.all_inputs[x.name] = x
        self.all_values = {}
        for x in self.model.graph.output:
            if x.name in output_names:
                self.all_outputs.append(x.name)
                output_names.remove(x.name)
                if len(output_names) == 0:
                    break
        for x in self.model.graph.value_info:
            self.all_values[x.name] = x
            if x.name not in output_names:
                continue
            self.model.graph.output.append(x)
            self.all_outputs.append(x.name)
            output_names.remove(x.name)
        if len(output_names) != 0:
            raise RuntimeError("Error, can't find {} in model".format(output_names))
        # node map name
        self.all_nodes = {}
        for x in self.model.graph.node:
            for o in x.output:
                self.all_nodes[o] = x
        # weight map name
        self.all_weights = {}
        for w in self.model.graph.initializer:
            self.all_weights[w.name] = w
        # remove unused node
        self.select_unuse(self.all_outputs)
        for n in self.all_nodes.values():
            if n in self.model.graph.node:
                self.model.graph.node.remove(n)
        for w in self.all_weights.values():
            self.model.graph.initializer.remove(w)
        for i in self.all_inputs.values():
            self.model.graph.input.remove(i)
        for v in self.all_values.values():
            self.model.graph.value_info.remove(v)
        unuse_output = []
        for o in self.model.graph.output:
            if o.name not in self.all_outputs:
                unuse_output.append(o)
        for o in unuse_output:
            self.model.graph.output.remove(o)

    def get_inputs(self, model: onnx.ModelProto):
        initializer_names = [x.name for x in model.graph.initializer]
        return [ipt for ipt in model.graph.input if ipt.name not in initializer_names]

    def get_input_names(self, model: onnx.ModelProto):
        input_names = [ipt.name for ipt in self.get_inputs(model)]
        return input_names

    def get_shape_from_value_info_proto(self, v: onnx.ValueInfoProto):
        return [dim.dim_value for dim in v.type.tensor_type.shape.dim]

    def get_input_shapes(self, model: onnx.ModelProto):
        inputs = self.get_inputs(model)
        return [self.get_shape_from_value_info_proto(i) for i in inputs]

    def load_onnx_model(self, onnx_file, input_shapes: list, output_names: list):
        if isinstance(onnx_file, str):
            self.model = onnx.load(onnx_file)
        else:
            self.model = onnx_file
        self.input_names = self.get_input_names(self.model)
        if "image_shape" in self.input_names:
            input_shapes.append([input_shapes[0][0], len(input_shapes[0]) - 2])
        self.num_input = len(self.input_names)
        self.model_shape_infer(input_shapes)
        self.input_shapes = self.get_input_shapes(self.model)
        model_simplified, is_ok = onnxsim.simplify(self.model)
        if is_ok:
            self.model = model_simplified
        if output_names:
            self.select_output(output_names)
        # add all weight
        for tensor in self.model.graph.initializer:
            name = tensor.name
            # all weight convert to f32.
            # TODO: support other type
            # remove astype(np.float32)
            data = numpy_helper.to_array(tensor)
            self.addTensor(name, data)
        # add all shape info
        for info in self.model.graph.value_info:
            shape = [i.dim_value for i in info.type.tensor_type.shape.dim]
            self.addShape(info.name, shape)
        for output in self.model.graph.output:
            if not self.isTensor(output.name):
                self.output_names.append(output.name)
                shape = [i.dim_value for i in output.type.tensor_type.shape.dim]
                self.addShape(output.name, shape)
        self.onnx_file = "{}_opt.onnx".format(self.model_name)
        onnx.save(self.model, self.onnx_file)
        strip_model = onnx.ModelProto()
        strip_model.CopyFrom(self.model)
        strip_model.graph.ClearField("initializer")
        with open(self.onnx_file + ".prototxt", "w") as f:
            f.write(str(strip_model))

    def model_shape_infer(self, input_shapes):
        inputs = self.get_inputs(self.model)
        no_shape = True

        def check_shape(l, r):
            if no_shape == False and l != r:
                raise KeyError("input shapes error:{}, {} vs {}".format(input_shapes, l, r))

        if len(input_shapes) > 0:
            no_shape = False
            check_shape(self.num_input, len(input_shapes))
        for idx, input in enumerate(inputs):
            _dims = input.type.tensor_type.shape.dim
            num_dims = len(_dims)
            if no_shape == False:
                check_shape(num_dims, len(input_shapes[idx]))
            _shape = []
            for _i, _dim in enumerate(_dims):
                if _dim.dim_value <= 0:
                    _dim.dim_value = 1 if no_shape else input_shapes[idx][_i]
                elif not no_shape:
                    check_shape(_dim.dim_value, input_shapes[idx][_i])
                _shape.append(_dim.dim_value)
            self.addShape(input.name, _shape)
        self.model = onnx.shape_inference.infer_shapes(self.model)

    def init_MLIRImporter(self):
        input_shapes = list()
        for _name in self.input_names:
            input_shapes.append(self.getShape(_name))
        output_shapes = list()
        for _name in self.output_names:
            output_shapes.append(self.getShape(_name))
        # init importer
        self.mlir = MLIRImporter(input_shapes, output_shapes, self.model_name)
        self.weight_file = self.mlir.weight_file

    def generate_mlir(self, mlir_file: str):
        """convert all to mlir"""
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

        def NoneAndRaise(node):
            raise RuntimeError("{} Op not support now".format(node.op_type))

        for n in self.model.graph.node:
            node = OnnxNode(n)
            self.onnxop_factory.get(node.op_type, lambda x: NoneAndRaise(x))(node)

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

    def convert_skip_op(self, onnx_node):
        op = self.getOperand(onnx_node.inputs[0])
        self.addOperand(onnx_node.name, op)

    def convert_add_op(self, onnx_node):
        assert (len(onnx_node.inputs) == 2)
        if self.isTensor(onnx_node.inputs[0]) or self.isTensor(onnx_node.inputs[1]):
            # TODO: support tensor
            raise RuntimeError("not support Tensor")
        op0 = self.getOperand(onnx_node.inputs[0])
        op1 = self.getOperand(onnx_node.inputs[1])
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        output_shape = self.getShape(onnx_node.name)
        add_op = self.mlir.create_add_op([op0, op1], output_shape, **p)
        self.addOperand(onnx_node.name, add_op)
        return

    def convert_batchnorm_op(self, onnx_node):
        assert (onnx_node.op_type == "BatchNormalization")
        op = self.getOperand(onnx_node.inputs[0])
        gamma = self.getWeightOp(onnx_node.inputs[1])
        beta = self.getWeightOp(onnx_node.inputs[2])
        mean = self.getWeightOp(onnx_node.inputs[3])
        variance = self.getWeightOp(onnx_node.inputs[4])
        epsilon = onnx_node.attrs.get("epsilon")
        output_shape = self.getShape(onnx_node.name)
        p = {
            "name": "{}_{}".format(onnx_node.name, onnx_node.op_type),
            "epsilon": epsilon,
        }
        new_op = self.mlir.create_batchnorm_op([op, gamma, beta, mean, variance], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_concat_op(self, onnx_node):
        assert (onnx_node.op_type == "Concat")
        output_shape = self.getShape(onnx_node.name)
        num_dims = len(output_shape)
        axis = onnx_node.attrs['axis']
        if axis < 0:
            axis += num_dims
        operands = [self.getOperand(x) for x in onnx_node.inputs]
        p = {"name": "{}_{}".format(onnx_node.name, onnx_node.op_type), "axis": axis}
        new_op = self.mlir.create_concat_op(operands, output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_conv_op(self, onnx_node):
        assert (onnx_node.op_type == "Conv")
        op = self.getOperand(onnx_node.inputs[0])
        kernel_shape = onnx_node.attrs['kernel_shape']
        dim = len(kernel_shape)
        dilations = onnx_node.attrs.get("dilations", dim * [1])
        group = onnx_node.attrs.get("group", 1)
        strides = onnx_node.attrs.get("strides", dim * [1])
        auto_pad = onnx_node.attrs.get("auto_pad", None)
        input_shape = self.getShape(onnx_node.inputs[0])
        if auto_pad:
            pads = set_auto_pad(auto_pad, input_shape, kernel_shape, strides)
        else:
            pads = onnx_node.attrs.get("pads", dim * 2 * [0])

        operands = list()
        operands.append(op)
        filter_op = self.getWeightOp(onnx_node.inputs[1])
        operands.append(filter_op)
        if len(onnx_node.inputs) > 2:
            bias_op = self.getWeightOp(onnx_node.inputs[2])
        else:
            bias_op = self.mlir.none_op
        operands.append(bias_op)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'kernel_shape': kernel_shape,
            'strides': strides,
            'dilations': dilations,
            'pads': pads,
            'group': group,
            'do_relu': False,
            'ins': [],
        }
        output_shape = self.getShape(onnx_node.name)
        new_op = self.mlir.create_conv_op(operands, output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_depth2space_op(self, onnx_node):
        assert (onnx_node.op_type == "DepthToSpace")
        op = self.getOperand(onnx_node.inputs[0])
        blocksize = onnx_node.attrs['blocksize']
        mode = onnx_node.attrs.get("mode", "DCR")
        output_shape = self.getShape(onnx_node.name)
        p = {
            "name": "{}_{}".format(onnx_node.name, onnx_node.op_type),
            "block_h": blocksize,
            "block_w": blocksize,
            "is_CRD": mode != "DCR",
            "is_inversed": False,
        }
        new_op = self.mlir.create_depth2space_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_flatten_op(self, onnx_node):
        assert (onnx_node.op_type == "Flatten")
        op = self.getOperand(onnx_node.inputs[0])
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
        }
        output_shape = self.getShape(onnx_node.name)
        new_op = self.mlir.create_reshape_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_gemm_op(self, onnx_node):
        assert (onnx_node.op_type == "Gemm")
        #(M, K) * (K, N) => (M, N)
        op = self.getOperand(onnx_node.inputs[0])
        alpha = onnx_node.attrs.get('alpha', 1)
        beta = onnx_node.attrs.get('beta', 1)
        trans_a = onnx_node.attrs.get('transA', 0)
        trans_b = onnx_node.attrs.get('transB', 0)
        # TODO:support more situations
        assert (trans_a == 0)
        operands = list()
        operands.append(op)
        B = onnx_node.inputs[1]
        assert (self.isTensor(B))
        if trans_b == 1 or alpha != 1:
            _tensor = self.getTensor(B)
            if trans_b == 1:
                _tensor = np.ascontiguousarray(np.transpose(_tensor, (1, 0)))
            if alpha != 1:
                _tensor *= alpha
            B += '_fix'
            self.addTensor(B, _tensor)
        operands.append(self.getWeightOp(B))
        if len(onnx_node.inputs) > 2 and beta != 0:
            C = onnx_node.inputs[2]
            if beta != 1:
                _tensor = self.getTensor(C)
                _tensor *= beta
                C += '_fix'
                self.addTensor(C, _tensor)
            operands.append(self.getWeightOp(C))
        else:
            operands.append(self.mlir.none_op)
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type), 'do_relu': False}
        output_shape = self.getShape(onnx_node.name)
        new_op = self.mlir.create_matmul_op(operands, output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_global_maxpool_op(self, onnx_node):
        assert (onnx_node.op_type == "GlobalMaxPool")
        op = self.getOperand(onnx_node.inputs[0])
        input_shape = self.getShape(onnx_node.inputs[0])
        num_dim = len(input_shape) - 2
        assert (num_dim > 0)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'kernel_shape': input_shape[2:],
            'strides': num_dim * [1],
            'pads': num_dim * 2 * [0],
            'count_include_pad': True,
            'do_relu': False,
        }
        output_shape = self.getShape(onnx_node.name)
        new_op = self.mlir.create_maxpool_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_global_avgpool_op(self, onnx_node):
        assert (onnx_node.op_type == "GlobalAveragePool")
        op = self.getOperand(onnx_node.inputs[0])
        input_shape = self.getShape(onnx_node.inputs[0])
        num_dim = len(input_shape) - 2
        assert (num_dim > 0)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'kernel_shape': input_shape[2:],
            'strides': num_dim * [1],
            'pads': num_dim * 2 * [0],
            'count_include_pad': True,
            'do_relu': False,
        }
        output_shape = self.getShape(onnx_node.name)
        new_op = self.mlir.create_avgpool_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_avgpool_op(self, onnx_node):
        assert (onnx_node.op_type == "AveragePool")
        op = self.getOperand(onnx_node.inputs[0])
        kernel_shape = onnx_node.attrs['kernel_shape']
        count_include_pad = onnx_node.attrs.get('count_include_pad', False)
        dim = len(kernel_shape)
        input_shape = self.getShape(onnx_node.inputs[0])
        strides = onnx_node.attrs.get("strides", kernel_shape)
        auto_pad = onnx_node.attrs.get("auto_pad", None)
        if auto_pad:
            pads = set_auto_pad(auto_pad, input_shape, kernel_shape, strides)
        else:
            pads = onnx_node.attrs.get("pads", dim * 2 * [0])
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'kernel_shape': kernel_shape,
            'strides': strides,
            'pads': pads,
            'count_include_pad': count_include_pad,
            'do_relu': False,
        }
        output_shape = self.getShape(onnx_node.name)
        new_op = self.mlir.create_avgpool_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_maxpool_op(self, onnx_node):
        assert (onnx_node.op_type == "MaxPool")
        op = self.getOperand(onnx_node.inputs[0])
        kernel_shape = onnx_node.attrs['kernel_shape']
        count_include_pad = onnx_node.attrs.get('count_include_pad', False)
        dim = len(kernel_shape)
        strides = onnx_node.attrs.get("strides", kernel_shape)
        input_shape = self.getShape(onnx_node.inputs[0])
        auto_pad = onnx_node.attrs.get("auto_pad", None)
        if auto_pad:
            pads = set_auto_pad(auto_pad, input_shape, kernel_shape, strides)
        else:
            pads = onnx_node.attrs.get("pads", dim * 2 * [0])
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'kernel_shape': kernel_shape,
            'strides': strides,
            'pads': pads,
            'count_include_pad': count_include_pad,
            'do_relu': False,
        }
        output_shape = self.getShape(onnx_node.name)
        new_op = self.mlir.create_maxpool_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_mul_op(self, onnx_node):
        assert (onnx_node.op_type == "Mul")
        assert (len(onnx_node.inputs) == 2)
        if self.isTensor(onnx_node.inputs[0]) and not self.isTensor(onnx_node.inputs[1]):
            onnx_node.inputs[0], onnx_node.inputs[1] = onnx_node.inputs[1], onnx_node.inputs[0]
            self.convert_mul_op(onnx_node)
            return
        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        if (not self.isTensor(onnx_node.inputs[0])) and self.isTensor(onnx_node.inputs[1]):
            op0 = self.getOperand(onnx_node.inputs[0])
            input1 = self.getTensor(onnx_node.inputs[1])
            p = {'name': name, 'const_val': input1.flatten()[0]}
            output_shape = self.getShape(onnx_node.name)
            mul_const_op = self.mlir.create_mul_const_op([op0], output_shape, **p)
            self.addOperand(onnx_node.name, mul_const_op)
            return
        else:
            op0 = self.getOperand(onnx_node.inputs[0])
            op1 = self.getOperand(onnx_node.inputs[1])
            p = {'name': name}
            output_shape = self.getShape(onnx_node.name)
            mul_op = self.mlir.create_mul_op([op0, op1], output_shape, **p)
            self.addOperand(onnx_node.name, mul_op)
            return

    def convert_dropout_op(self, onnx_node):
        assert (onnx_node.op_type == "Dropout")
        op = self.getOperand(onnx_node.inputs[0])
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'ratio': onnx_node.attrs.get("ratio", 0.5)
        }
        output_shape = self.getShape(onnx_node.name)
        new_op = self.mlir.create_dropout_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_relu_op(self, onnx_node):
        assert (onnx_node.op_type == "Relu")
        op = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'upper_limit': 0.0
        }
        new_op = self.mlir.create_relu_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_leaky_relu_op(self, onnx_node):
        assert (onnx_node.op_type == "LeakyRelu")
        op = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'alpha': onnx_node.attrs.get("alpha", 0.)
        }
        new_op = self.mlir.create_leaky_relu_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_reshape_op(self, onnx_node):
        assert (onnx_node.op_type == "Reshape")
        op = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        new_op = self.mlir.create_reshape_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    # when resize by nearest, with integer scale_h and integer scale_w
    def resize_to_upsample(self, onnx_node, op, input_shape, output_shape, scale_h, scale_w):
        operands = [op]
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'scale_h': int(scale_h),
            'scale_w': int(scale_w),
        }
        new_op = self.mlir.create_upsample_op(operands, output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_resize_op(self, onnx_node):
        assert (onnx_node.op_type == "Resize")
        mode = onnx_node.attrs.get("mode", "nearest")

        op = self.getOperand(onnx_node.inputs[0])
        input_shape = self.getShape(onnx_node.inputs[0])
        scale_factor = []
        sizes = []

        if len(onnx_node.inputs) > 2:
            # onnx opset 11
            scale_factor = self.getTensor(onnx_node.inputs[2])
            if len(scale_factor) == 0:
                sizes = self.getTensor(onnx_node.inputs[3])
                scale_factor = sizes / input_shape
            else:
                sizes = input_shape * scale_factor
        else:
            # opset 10
            scale_factor = self.getTensor(onnx_node.inputs[1])
            sizes = input_shape * scale_factor

        if scale_factor[0] != 1.0 or scale_factor[1] != 1.0:
            raise RuntimeError("Resize only support h/w")

        output_shape = [int(i) for i in sizes]
        scale_h = scale_factor[2]
        scale_w = scale_factor[3]
        if scale_h == 1.0 and scale_w == 1.0:
            self.addOperand(onnx_node.name, op)
            return
        coord_mode = onnx_node.attrs.get("coordinate_transformation_mode", "half_pixel")
        if mode == b'nearest':
            if scale_h == int(scale_h) and scale_w == int(scale_w):
                self.resize_to_upsample(onnx_node, op, input_shape, output_shape, scale_h, scale_w)
                return
        raise RuntimeError("[{}] Unsupported mode: {}, coord_mode: {}".format(
            onnx_node.name, mode, coord_mode))

    def convert_sigmoid_op(self, onnx_node):
        assert (onnx_node.op_type == "Sigmoid")
        op = self.getOperand(onnx_node.inputs[0])
        scale = onnx_node.attrs.get('scale', 1)
        bias = onnx_node.attrs.get('bias', 0)
        output_shape = self.getShape(onnx_node.name)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'scale': scale,
            'bias': bias
        }
        new_op = self.mlir.create_sigmoid_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_slice_op(self, onnx_node):
        assert (onnx_node.op_type == "Slice")
        op = self.getOperand(onnx_node.inputs[0])
        input_shape = self.getShape(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        starts = []
        ends = []
        axes = []
        num_input = len(onnx_node.inputs)
        num_dims = len(input_shape)
        if num_input > 1:
            starts = self.getTensor(onnx_node.inputs[1]).astype(int)
            ends = self.getTensor(onnx_node.inputs[2]).astype(int)
            axes = self.getTensor(onnx_node.inputs[3]).astype(int) if num_input > 3 else list(
                np.arange(num_dims))
            steps = self.getTensor(
                onnx_node.inputs[4]).astype(int) if num_input > 4 else [1] * len(axes)
        else:
            starts = onnx_node.attrs.get('starts')
            ends = onnx_node.attrs.get('ends')
            axes = onnx_node.attrs.get('axes')
            steps = [1] * len(axes)
        assert (len(starts) == len(ends))
        assert (len(axes) == len(ends))
        slice_shape = list(input_shape)
        slice_offset = [0] * num_dims
        slice_step = [1] * num_dims
        for start, end, axis, step in zip(starts, ends, axes, steps):
            start, end, axis, step = int(start), int(end), int(axis), int(step)
            assert (step > 0)
            if axis < 0:
                axis = axis + num_dims
            if end < 0:
                end = end + input_shape[axis]
            if start < 0:
                start = start + input_shape[axis]
            if end > input_shape[axis] or end < 0:
                end = input_shape[axis]
            slice_shape[axis] = (end - start + step - 1) // step
            slice_offset[axis] = start
            slice_step[axis] = step
        assert (slice_shape == output_shape)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'offset': list(slice_offset),
            'steps': list(slice_step)
        }
        new_op = self.mlir.create_slice_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_transpose_op(self, onnx_node):
        assert (onnx_node.op_type == "Transpose")
        op = self.getOperand(onnx_node.inputs[0])
        input_shape = self.getShape(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        # default revert it, eg: shape (2, 3, 4)->(4, 3, 2), per=[2, 1, 0]
        perm_default = list(np.arange(len(input_shape))[::-1])
        transpose_perm = onnx_node.attrs.get('perm', perm_default)
        assert (len(input_shape) == len(transpose_perm))
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'order': transpose_perm,
        }
        new_op = self.mlir.create_permute_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_softmax_op(self, onnx_node):
        assert (onnx_node.op_type == "Softmax")
        op = self.getOperand(onnx_node.inputs[0])
        input_shape = self.getShape(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        axis_default = -1
        for i, shape in enumerate(output_shape):
            if shape > 1:
                axis_default = i
                break
        axis = onnx_node.attrs.get('axis', axis_default)
        if axis < 0:
            axis += len(input_shape)
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type), 'axis': axis}
        new_op = self.mlir.create_softmax_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_log_op(self, onnx_node):
        assert (onnx_node.op_type == "Log")
        op = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        new_op = self.mlir.create_log_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_pad_op(self, onnx_node):
        assert (onnx_node.op_type == "Pad")
        op = self.getOperand(onnx_node.inputs[0])
        input_shape = self.getShape(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        pads = list(self.getTensor(onnx_node.inputs[1]))
        if pads == None:
            raise RuntimeError("No paddings value")
        if len(pads) != 2 * len(input_shape):
            raise RuntimeError(
                "pads number is two times as same as input shape ({} v.s 2 * {})".format(
                    len(pads), len(input_shape)))
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'paddings': pads,
        }

        new_op = self.mlir.create_pad_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_div_op(self, onnx_node):
        assert (len(onnx_node.inputs) == 2)
        # if self.isTensor(onnx_node.inputs[0]) or self.isTensor(onnx_node.inputs[1]):
        #     # TODO: support tensor
        #     raise RuntimeError("not support Tensor")
        op0 = self.getOperand(onnx_node.inputs[0])
        op1 = self.getOperand(onnx_node.inputs[1])
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        output_shape = self.getShape(onnx_node.name)
        div_op = self.mlir.create_div_op([op0, op1], output_shape, **p)
        self.addOperand(onnx_node.name, div_op)

    def convert_squeeze_op(self, onnx_node):
        assert (onnx_node.op_type == "Squeeze")
        input = self.getOperand(onnx_node.inputs[0])
        operand = [input]
        if len(onnx_node.inputs) == 2:
            axes = self.getTensor(onnx_node.inputs[1])
        else :
            axes = onnx_node.attrs.get('axes')
        input_shape = self.getShape(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        dims = len(input_shape)
        for i in range(len(axes)):
            assert axes[i] < dims and axes[i] >= -dims
            axes[i] = axes[i] if axes[i] >= 0 else axes[i] + dims
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'axes': axes,
        }
        new_op = self.mlir.create_squeeze_op(operand, output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_clip_op(self, onnx_node):
        assert (onnx_node.op_type == "Clip")
        input = self.getOperand(onnx_node.inputs[0])
        operand = [input]
        if len(onnx_node.inputs) >= 2:
            # TODO: min is '',
            min = self.getTensor(onnx_node.inputs[1]).tolist()
            if len(onnx_node.inputs) == 3:
                max = self.getTensor(onnx_node.inputs[2]).tolist()
            else :
                max = np.inf
        else :
            min = -np.inf
        input_shape = self.getShape(onnx_node.inputs[0])
        if min == 0.0 and max > min:
            p = {
                'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
                'upper_limit': max if max != np.inf else 0.0,
            }
            new_op = self.mlir.create_relu_op(operand, input_shape, **p)
        else :
            p = {
                'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
                'min': min,
                'max': max,
            }
            new_op = self.mlir.create_clip_op(operand, input_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_conv_transpose_op(self, onnx_node):
        assert(onnx_node.op_type == "ConvTranspose")
        input_shape = self.getShape(onnx_node.inputs[0])
        kernel_shape = onnx_node.attrs['kernel_shape']
        output_shape = self.getShape(onnx_node.name)
        dim = len(kernel_shape)
        dilations = onnx_node.attrs.get('dilations', dim * [1])
        group = onnx_node.attrs.get('group', 1)
        strides = onnx_node.attrs.get('strides', dim * [1])
        pads = onnx_node.attrs.get('pads', dim * 2 * [0])
        output_padding = onnx_node.attrs.get('output_padding', dim * 2 * [0])
        auto_pad = onnx_node.attrs.get('auto_pad', None)

        operands = list()
        input_opd = self.getOperand(onnx_node.inputs[0])
        filter_opd = self.getWeightOp(onnx_node.inputs[1])
        if len(onnx_node.inputs) > 2:
            bias_opd = self.getWeightOp(onnx_node.inputs[2])
        else:
            bias_opd = self.mlir.none_op
        operands.append(input_opd)
        operands.append(filter_opd)
        operands.append(bias_opd)

        # handle ConvTranspose1d case
        is_shape_3 = len(input_shape) == 3
        if is_shape_3:
            assert(dim == 1)
            strides = [1, strides[0]]
            pads = [0, 0, pads[0], pads[1]]
            kernel_shape = [1, kernel_shape[0]]
            output_padding = [0, 0, output_padding[0], output_padding[1]]

            input_shape = [input_shape[0], input_shape[1], 1, input_shape[2]]
            p = {'name': '{}_to4dim'.format(onnx_node.name)}
            reshape0_op = self.mlir.create_reshape_op([input_opd], input_shape, **p)
            operands[0] = reshape0_op

        p = {
            'name': '{}_{}'.format(onnx_node.name + '_reshape', onnx_node.op_type),
            'kernel_shape': kernel_shape,
            'strides': strides,
            'dilations': dilations,
            'pads': pads,
            'group': group,
            'do_relu': False,
            'ins': [],
        }

        new_op = self.mlir.create_conv_transpose_op(operands, output_shape, **p)

        if is_shape_3:
            output_shape = [output_shape[0], output_shape[1], output_shape[3]]
            p = {'name': '{}_backto3dim'.format(onnx_node.name)}
            reshape1_op = self.mlir.create_reshape_op([new_op], output_shape, **p)
            self.addOperand(onnx_node.name, reshape1_op)
        else:
            self.addOperand(onnx_node.name, new_op)

    def convert_split_op(self, onnx_node):
        assert (onnx_node.op_type == "Split")
        input_shape = self.getShape(onnx_node.inputs[0])
        num_output = len(onnx_node.outputs)
        num_dims = len(input_shape)
        axis = onnx_node.attrs['axis']
        if axis < 0:
            axis += num_dims
        slice = input_shape[axis] // num_output
        split = onnx_node.attrs.get('split', [slice] * num_output)
        op = self.getOperand(onnx_node.inputs[0])

        offset = 0
        #replace the split with slice
        for i, name in zip(split, onnx_node.outputs):
            output_shape = list(input_shape)
            output_shape[axis] = i
            slice_offset = [0] * num_dims
            slice_offset[axis] = offset
            slice_step = [1] * num_dims
            p = {
                'name': "{}_{}".format(name, onnx_node.op_type),
                'offset': list(slice_offset),
                'steps': list(slice_step)
            }
            new_op = self.mlir.create_slice_op([op], output_shape, **p)
            self.addOperand(name, new_op)
            offset = offset + i
