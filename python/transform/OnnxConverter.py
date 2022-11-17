# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
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
import copy

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
            #pls add the Op alphabetically
            "Abs": lambda node: self.convert_abs_op(node),
            "Add": lambda node: self.convert_add_op(node),
            "AveragePool": lambda node: self.convert_avgpool_op(node),
            "BatchNormalization": lambda node: self.convert_batchnorm_op(node),
            "Concat": lambda node: self.convert_concat_op(node),
            "Conv": lambda node: self.convert_conv_op(node),
            "Clip": lambda node: self.convert_clip_op(node),
            "ConvTranspose": lambda node: self.convert_conv_transpose_op(node),
            "DepthToSpace": lambda node: self.convert_depth2space_op(node),
            "Div": lambda node: self.convert_div_op(node),
            "Dropout": lambda node: self.convert_skip_op(node),
            "Expand": lambda node: self.convert_expand_op(node),
            "Flatten": lambda node: self.convert_flatten_op(node),
            "Gather": lambda node: self.convert_gather_op(node),
            "Gemm": lambda node: self.convert_gemm_op(node),
            "GlobalAveragePool": lambda node: self.convert_global_avgpool_op(node),
            "GlobalMaxPool": lambda node: self.convert_global_maxpool_op(node),
            "LeakyRelu": lambda node: self.convert_leaky_relu_op(node),
            "Log": lambda node: self.convert_log_op(node),
            "LRN": lambda node: self.convert_lrn_op(node),
            "LSTM": lambda node: self.convert_lstm_op(node),
            "MatMul": lambda node: self.convert_gemm_op(node),
            "Max": lambda node: self.convert_max_op(node),
            "MaxPool": lambda node: self.convert_maxpool_op(node),
            "Min": lambda node: self.convert_min_op(node),
            "Mul": lambda node: self.convert_mul_op(node),
            "Neg": lambda node: self.convert_neg_op(node),
            "Pad": lambda node: self.convert_pad_op(node),
            "PRelu": lambda node: self.convert_prelu_op(node),
            "ReduceMean": lambda node: self.convert_reduce_op(node),
            "Relu": lambda node: self.convert_relu_op(node),
            "Reshape": lambda node: self.convert_reshape_op(node),
            "Resize": lambda node: self.convert_resize_op(node),
            "Sigmoid": lambda node: self.convert_sigmoid_op(node),
            "Slice": lambda node: self.convert_slice_op(node),
            "Softmax": lambda node: self.convert_softmax_op(node),
            "Squeeze": lambda node: self.convert_squeeze_op(node),
            "Split": lambda node: self.convert_split_op(node),
            "Tile": lambda node: self.convert_tile_op(node),
            "Transpose": lambda node: self.convert_transpose_op(node),
            "Unsqueeze": lambda node: self.convert_unsqueeze_op(node),
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

    def get_input_types(self, model: onnx.ModelProto):
        input_types = []
        for input in self.get_inputs(model):
            if input.type.tensor_type.elem_type in [onnx.TensorProto.INT64, onnx.TensorProto.INT32]:
                input_types.append('INT32')
            else:
                input_types.append('F32')
        return input_types

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
        self.input_types = self.get_input_types(self.model)
        model_simplified, is_ok = onnxsim.simplify(self.model)
        if is_ok:
            self.model = onnx.shape_inference.infer_shapes(model_simplified)
        if output_names:
            self.select_output(output_names)
        # add all weight
        for tensor in self.model.graph.initializer:
            name = tensor.name
            # all weight convert to f32.
            # TODO: support other type
            # remove astype(np.float32)
            data = numpy_helper.to_array(tensor)
            self.addWeight(name, data)
        # add all shape info
        for info in self.model.graph.value_info:
            shape = [i.dim_value for i in info.type.tensor_type.shape.dim]
            self.addShape(info.name, shape)
        for output in self.model.graph.output:
            if not self.isWeight(output.name):
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
        self.mlir = MLIRImporter(input_shapes, output_shapes, self.model_name, self.input_types)
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
                input_op = self.mlir.create_input_op(_name, idx, **self.preprocess_args)
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
        assert (onnx_node.op_type == "Add")
        assert (len(onnx_node.inputs) == 2)
        if self.isWeight(onnx_node.inputs[0]) and not self.isWeight(onnx_node.inputs[1]):
            onnx_node.inputs[0], onnx_node.inputs[1] = onnx_node.inputs[1], onnx_node.inputs[0]
            self.convert_add_op(onnx_node)
            return
        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        if not self.isWeight(onnx_node.inputs[0]) and self.isWeight(onnx_node.inputs[1]):
            opd1_num_elem = np.prod(self.getShape(onnx_node.inputs[1]))
            output_shape = self.getShape(onnx_node.name)
            channel = output_shape[1]
            op0 = self.getOperand(onnx_node.inputs[0])
            if opd1_num_elem == channel:
                offset = self.getWeight(onnx_node.inputs[1])
                weight_data = np.ones_like(offset)
                self.addWeight(name + '_scale', weight_data)
                weight_op = self.getWeightOp(name + '_scale')
                offset_op = self.getWeightOp(onnx_node.inputs[1])
                p = {'name': name}
                scale_op = self.mlir.create_scale_op([op0, weight_op, offset_op], output_shape, **p)
                self.addOperand(onnx_node.name, scale_op)
                return
            else:
                const_op = self.getWeightOp(onnx_node.inputs[1])
                p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
                scale_op = self.mlir.create_add_op([op0, const_op], output_shape, **p)
                self.addOperand(onnx_node.name, scale_op)
                return
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
        new_op = self.mlir.create_batchnorm_op([op, mean, variance, gamma, beta], output_shape, **p)
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
        mode = onnx_node.attrs.get("mode", b"DCR")
        output_shape = self.getShape(onnx_node.name)
        p = {
            "name": "{}_{}".format(onnx_node.name, onnx_node.op_type),
            "block_h": blocksize,
            "block_w": blocksize,
            "is_CRD": mode != b"DCR",
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
        assert (onnx_node.op_type == "Gemm" or onnx_node.op_type == 'MatMul')
        #(M, K) * (K, N) => (M, N)
        alpha = onnx_node.attrs.get('alpha', 1)
        beta = onnx_node.attrs.get('beta', 1)
        trans_a = onnx_node.attrs.get('transA', 0)
        trans_b = onnx_node.attrs.get('transB', 0)
        # TODO:support more situations
        assert (trans_a == 0)
        operands = list()
        A = onnx_node.inputs[0]
        B = onnx_node.inputs[1]
        in_op = self.getOperand(A)
        operands.append(in_op)

        if self.isWeight(B):
            if trans_b == 1 or alpha != 1:
                _tensor = self.getWeight(B)
                if trans_b == 1:
                    _tensor = np.ascontiguousarray(np.transpose(_tensor, (1, 0)))
                if alpha != 1:
                    _tensor *= alpha
                B += '_fix'
                self.addWeight(B, _tensor)
            operands.append(self.getWeightOp(B))
        else:
            operands.append(self.getOperand(B))
        if len(onnx_node.inputs) > 2 and beta != 0:
            C = onnx_node.inputs[2]
            if self.isWeight(C):
                if beta != 1:
                    _tensor = self.getWeight(C)
                    _tensor *= beta
                    C += '_fix'
                    self.addWeight(C, _tensor)
                operands.append(self.getWeightOp(C))
            else:
                operands.append(self.getOperand(C))
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
        if auto_pad and auto_pad != b'NOTSET':
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
        if self.isWeight(onnx_node.inputs[0]) and not self.isWeight(onnx_node.inputs[1]):
            onnx_node.inputs[0], onnx_node.inputs[1] = onnx_node.inputs[1], onnx_node.inputs[0]
            self.convert_mul_op(onnx_node)
            return
        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        if (not self.isWeight(onnx_node.inputs[0])) and self.isWeight(onnx_node.inputs[1]):
            op0 = self.getOperand(onnx_node.inputs[0])
            weight = self.getWeight(onnx_node.inputs[1])
            output_shape = self.getShape(onnx_node.name)
            weight_num_elem = np.prod(self.getShape(onnx_node.inputs[1]))
            first_val = weight.flatten()[0]
            channel = output_shape[1]
            if weight_num_elem == channel:
                offset_data = np.zeros_like(weight)
                self.addWeight(name + '_bias', offset_data)
                weight_op = self.getWeightOp(onnx_node.inputs[1])
                offset_op = self.getWeightOp(name + '_bias')
                p = {'name': name}
                scale_op = self.mlir.create_scale_op([op0, weight_op, offset_op], output_shape, **p)
                self.addOperand(onnx_node.name, scale_op)
                return
            elif weight_num_elem == 1 or np.all(weight == first_val):
                p = {'name': name, 'const_val': weight.flatten()[0]}
                mul_const_op = self.mlir.create_mul_const_op([op0], output_shape, **p)
                self.addOperand(onnx_node.name, mul_const_op)
                return
            const_op = self.getWeightOp(onnx_node.inputs[1])
            p = {'name': name}
            scale_op = self.mlir.create_mul_op([op0, const_op], output_shape, **p)
            self.addOperand(onnx_node.name, scale_op)
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
        self.addOperand(onnx_node.name, op)

    def convert_relu_op(self, onnx_node):
        assert (onnx_node.op_type == "Relu")
        op = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
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

    # when resize by linear or nearst, with float scale_h or float scale_w
    def resize_to_interp(self, onnx_node, op, input_shape, output_shape, scale_h, scale_w, mode,
                         coordinate_transformation_mode):
        operands = [op]
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'scale_h': float(scale_h),
            'scale_w': float(scale_w),
            'mode': mode,
            'coordinate_transformation_mode': coordinate_transformation_mode
        }
        new_op = self.mlir.create_interp_op(operands, output_shape, **p)
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
            scale_factor = self.getWeight(onnx_node.inputs[2])
            if len(scale_factor) == 0:
                sizes = self.getWeight(onnx_node.inputs[3])
                scale_factor = sizes / input_shape
            else:
                sizes = input_shape * scale_factor
        else:
            # opset 10
            scale_factor = self.getWeight(onnx_node.inputs[1])
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
        if mode == b'nearest' and scale_h == int(scale_h) and scale_w == int(scale_w):
            self.resize_to_upsample(onnx_node, op, input_shape, output_shape, scale_h, scale_w)
            return
        else:
            self.resize_to_interp(onnx_node, op, input_shape, output_shape, scale_h, scale_w, mode,
                                  coord_mode)
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
            starts = self.getWeight(onnx_node.inputs[1]).astype(int)
            ends = self.getWeight(onnx_node.inputs[2]).astype(int)
            axes = self.getWeight(onnx_node.inputs[3]).astype(int) if num_input > 3 else list(
                np.arange(num_dims))
            steps = self.getWeight(
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
        pad_mode = {"constant": 0, "reflect": 1, "edge": 3}
        op = self.getOperand(onnx_node.inputs[0])
        input_shape = self.getShape(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)

        # get pad mode
        mode = onnx_node.attrs.get("mode", "constant")
        if isinstance(mode, bytes):
            mode = mode.decode("utf-8")
        pads = list(self.getWeight(onnx_node.inputs[1]))
        if pads == None:
            raise RuntimeError("No paddings value")
        if len(pads) != 2 * len(input_shape):
            raise RuntimeError(
                "pads number is two times as same as input shape ({} v.s 2 * {})".format(
                    len(pads), len(input_shape)))
        # opset 11, value from second input
        val = 0.0
        if len(onnx_node.inputs) > 2:
            val = self.getWeight(onnx_node.inputs[2])
        else:
            val = onnx_node.attrs.get("value", 0.0)

        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'paddings': pads,
            'val': val,
            'mode': pad_mode[mode],
        }

        new_op = self.mlir.create_pad_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_div_op(self, onnx_node):
        assert (len(onnx_node.inputs) == 2)
        # if self.isWeight(onnx_node.inputs[0]) or self.isWeight(onnx_node.inputs[1]):
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
        op = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        new_op = self.mlir.create_reshape_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_unsqueeze_op(self, onnx_node):
        assert (onnx_node.op_type == "Unsqueeze")
        op = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        new_op = self.mlir.create_reshape_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_clip_op(self, onnx_node):
        assert (onnx_node.op_type == "Clip")
        input = self.getOperand(onnx_node.inputs[0])
        if len(onnx_node.inputs) == 3:
            min = self.getWeight(onnx_node.inputs[1])
            max = self.getWeight(onnx_node.inputs[2])
        else:
            min = onnx_node.attrs.get('min', -np.inf)
            max = onnx_node.attrs.get('max', np.inf)
        input_shape = self.getShape(onnx_node.inputs[0])
        if min == 0.0 and max > min:
            p = {
                'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
                'relu_limit': max if max != np.inf else 0.0,
            }
            new_op = self.mlir.create_relu_op([input], input_shape, **p)
        else:
            raise RuntimeError("To be supported")
        self.addOperand(onnx_node.name, new_op)

    def convert_conv_transpose_op(self, onnx_node):
        assert (onnx_node.op_type == "ConvTranspose")
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
        weight_name = onnx_node.inputs[1]
        # (ic, oc, kh, kw) --> (oc, ic, kh, kw)
        old_weight = self.tensors[weight_name]
        order = [1, 0] + list(range(len(old_weight.shape))[2:])
        self.tensors[weight_name] = np.ascontiguousarray(np.transpose(old_weight, order))
        self.shapes[weight_name] = self.tensors[weight_name].shape
        filter_opd = self.getWeightOp(onnx_node.inputs[1])
        if len(onnx_node.inputs) > 2:
            bias_opd = self.getWeightOp(onnx_node.inputs[2])
        else:
            bias_opd = self.mlir.none_op
        operands.append(input_opd)
        operands.append(filter_opd)
        operands.append(bias_opd)

        # handle ConvTranspose1d case
        new_name = onnx_node.name
        is_shape_3 = len(input_shape) == 3
        if is_shape_3:
            assert (dim == 1)
            strides = [1, strides[0]]
            pads = [0, 0, pads[0], pads[1]]
            kernel_shape = [1, kernel_shape[0]]
            output_padding = [0, 0, output_padding[0], output_padding[1]]

            input_shape = [input_shape[0], input_shape[1], 1, input_shape[2]]
            p = {'name': '{}_to4dim'.format(onnx_node.name)}
            reshape0_op = self.mlir.create_reshape_op([input_opd], input_shape, **p)
            operands[0] = reshape0_op
            new_name += "_reshape"
        p = {
            'name': '{}_{}'.format(new_name, onnx_node.op_type),
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
            p = {'name': onnx_node.name}
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

    def convert_reduce_op(self, onnx_node):
        assert (onnx_node.op_type == "ReduceMean" or onnx_node.op_type == "ReduceMax"
                or onnx_node.op_type == "ReduceMin" or onnx_node.op_type == "ReduceProd"
                or onnx_node.op_type == "ReduceSum" or onnx_node.op_type == "ReduceSumSquare"
                or onnx_node.op_type == "ReduceL1" or onnx_node.op_type == "ReduceL2"
                or onnx_node.op_type == "ReduceLogSum" or onnx_node.op_type == "ReduceLogSumExp")

        input_shape = self.getShape(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        num_dims = len(input_shape)
        axes = onnx_node.attrs['axes']
        keepdims = onnx_node.attrs['keepdims']
        reduce_n = reduce_c = reduce_h = reduce_w = 0
        stride = [1, 1]
        for i in range(len(axes)):
            axes[i] = axes[i] if axes[i] >= 0 else axes[i] + num_dims
            if axes[i] == 0:
                reduce_n = 1
            elif axes[i] == 1:
                reduce_c = 1
            elif axes[i] == 2:
                reduce_h = 1
            else:
                reduce_w = 1

        op = self.getOperand(onnx_node.inputs[0])
        # yapf: disable
        #if reducemean the h, w, or h & w, replace it with avgpool
        if (onnx_node.op_type == "ReduceMean" and num_dims == 4 and reduce_w and reduce_h):
            onnx_node.op_type = "GlobalAveragePool"
            name = "reduce_w_h"
            num_dims = len(input_shape) - 2
            p = {
                'name': "{}_{}".format(name if (reduce_n or reduce_c) else onnx_node.name, onnx_node.op_type),
                'kernel_shape': input_shape[2:],
                'strides': num_dims * [1],
                'pads': num_dims * 2 * [0],
                'count_include_pad': True,
                'do_relu': False,
            }
        elif onnx_node.op_type == "ReduceMean" and num_dims == 4 and reduce_w:
            onnx_node.op_type = "GlobalAveragePool"
            num_dims = len(input_shape) - 2
            kernel_shape = [1, 1]
            kernel_shape[1] = input_shape[3]
            name = "reduce_w"
            p = {
                'name': "{}_{}".format(name if (reduce_n or reduce_c) else onnx_node.name, onnx_node.op_type),
                'kernel_shape': kernel_shape,
                'strides': stride,
                'pads': num_dims * 2 * [0],
                'count_include_pad': True,
                'do_relu': False,
            }
        elif onnx_node.op_type == "ReduceMean" and num_dims == 4 and reduce_h:
            onnx_node.op_type = "GlobalAveragePool"
            num_dims = len(input_shape) - 2
            kernel_shape = [1, 1]
            kernel_shape[0] = input_shape[2]
            name = "reduce_h"
            p = {
                'name': "{}_{}".format(name if (reduce_n or reduce_c) else onnx_node.name, onnx_node.op_type),
                'kernel_shape': kernel_shape,
                'strides': stride,
                'pads': num_dims * 2 * [0],
                'count_include_pad': True,
                'do_relu': False,
            }
        # yapf: enable
        new_out_shape = copy.deepcopy(input_shape)
        if reduce_w:
            if keepdims:
                new_out_shape[3] = 1
            else:
                new_out_shape.pop()
        if reduce_h:
            if keepdims:
                new_out_shape[2] = 1
            else:
                new_out_shape.pop()

        #reduce h、w or h & w firstly
        if (reduce_w or reduce_h):
            new_op = self.mlir.create_avgpool_op([op], new_out_shape, **p)
            self.addOperand(name if (reduce_n or reduce_c) else onnx_node.name, new_op)
            op = new_op

        # reduce_c
        if reduce_c:
            #tranpose the c/w firstly
            onnx_node.op_type = "Transpose"
            name = "transpose_c"
            transpose_perm = [0, 3, 2, 1]
            assert (len(new_out_shape) == len(transpose_perm))
            tranpose_new_output_shape = copy.deepcopy(new_out_shape)
            tranpose_new_output_shape[1] = new_out_shape[3]
            tranpose_new_output_shape[3] = new_out_shape[1]
            p = {
                'name': "{}_{}".format(name, onnx_node.op_type),
                'order': transpose_perm,
            }
            new_op = self.mlir.create_permute_op([op], tranpose_new_output_shape, **p)
            self.addOperand(name, new_op)

            name = "reduce_c"
            op = new_op
            onnx_node.op_type = "GlobalAveragePool"
            num_dims = len(tranpose_new_output_shape) - 2
            kernel_shape = [1, 1]
            kernel_shape[1] = tranpose_new_output_shape[3]
            new_out_shape = copy.deepcopy(tranpose_new_output_shape)
            new_out_shape[3] = 1
            p = {
                'name': "{}_{}".format(name, onnx_node.op_type),
                'kernel_shape': kernel_shape,
                'strides': stride,
                'pads': num_dims * 2 * [0],
                'count_include_pad': True,
                'do_relu': False,
            }
            new_op = self.mlir.create_avgpool_op([op], new_out_shape, **p)
            self.addOperand(name, new_op)
            op = new_op

            #tranpose w/c again
            transpose_perm = [0, 3, 2, 1]
            assert (len(new_out_shape) == len(transpose_perm))
            tranpose_new_output_shape = copy.deepcopy(new_out_shape)
            tranpose_new_output_shape[1] = new_out_shape[3]
            tranpose_new_output_shape[3] = new_out_shape[1]
            name = "tranpose_c_back"
            p = {
                'name': "{}_{}".format(name if reduce_n else onnx_node.name, onnx_node.op_type),
                'order': transpose_perm,
            }
            new_op = self.mlir.create_permute_op([op], tranpose_new_output_shape, **p)
            self.addOperand(name if reduce_n else onnx_node.name, new_op)
            op = new_op

        #reduce_n
        if reduce_n:
            #tranpose the n/w firstly
            onnx_node.op_type = "Transpose"
            name = "transpose_n"
            transpose_perm = [3, 1, 2, 0]
            new_out_shape = copy.deepcopy(input_shape)
            if reduce_w:
                new_out_shape[3] = 1
            if reduce_h:
                new_out_shape[2] = 1
            if reduce_c:
                new_out_shape[1] = 1
            assert (len(new_out_shape) == len(transpose_perm))
            tranpose_new_output_shape = copy.deepcopy(new_out_shape)
            tranpose_new_output_shape[0] = new_out_shape[3]
            tranpose_new_output_shape[3] = new_out_shape[0]
            p = {
                'name': "{}_{}".format(name, onnx_node.op_type),
                'order': transpose_perm,
            }
            new_op = self.mlir.create_permute_op([op], tranpose_new_output_shape, **p)
            self.addOperand(name, new_op)

            name = "reduce_n"
            op = new_op
            onnx_node.op_type = "GlobalAveragePool"
            num_dims = len(tranpose_new_output_shape) - 2
            kernel_shape = [1, 1]
            kernel_shape[1] = tranpose_new_output_shape[3]
            new_out_shape = copy.deepcopy(tranpose_new_output_shape)
            new_out_shape[3] = 1
            p = {
                'name': "{}_{}".format(name, onnx_node.op_type),
                'kernel_shape': kernel_shape,
                'strides': stride,
                'pads': num_dims * 2 * [0],
                'count_include_pad': True,
                'do_relu': False,
            }
            new_op = self.mlir.create_avgpool_op([op], new_out_shape, **p)
            self.addOperand(name, new_op)
            op = new_op

            #tranpose w/c again
            transpose_perm = [3, 1, 2, 0]
            assert (len(new_out_shape) == len(transpose_perm))
            tranpose_new_output_shape = copy.deepcopy(new_out_shape)
            tranpose_new_output_shape[0] = new_out_shape[3]
            tranpose_new_output_shape[3] = new_out_shape[0]
            p = {
                'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
                'order': transpose_perm,
            }
            new_op = self.mlir.create_permute_op([op], tranpose_new_output_shape, **p)
            self.addOperand(onnx_node.name, new_op)


    def convert_lrn_op(self, onnx_node):
        assert onnx_node.op_type == "LRN"
        op = self.getOperand(onnx_node.inputs[0])

        size = onnx_node.attrs.get("size")
        p = {"name": "{}_{}".format(onnx_node.name, onnx_node.op_type), "size": size}

        def add_if_valid(key):
            value = onnx_node.attrs.get(key)
            if value:
                p[key] = value

        add_if_valid("alpha")
        add_if_valid("beta")
        add_if_valid("bias")
        output_shape = self.getShape(onnx_node.name)
        new_op = self.mlir.create_lrn_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)


    def convert_lstm_op(self, onnx_node):
        assert (onnx_node.op_type == "LSTM")
        direction = onnx_node.attrs.get("direction")
        bidirectional = True if onnx_node.attrs.get("direction",
                                                    'forward') == b'bidirectional' else False
        layout = onnx_node.attrs.get("layout")
        batch_first = True if layout == 1 else False
        hidden_size = onnx_node.attrs.get("hidden_size")
        input_size = len(onnx_node.inputs)
        operands = list()
        operands.append(self.getOperand(onnx_node.inputs[0]))  # in
        operands.append(self.getWeightOp(onnx_node.inputs[1]))  # W
        operands.append(self.getWeightOp(onnx_node.inputs[2]))  # R
        have_bias = False
        if len(onnx_node.inputs) > 3:
            have_bias = True
            operands.append(self.getWeightOp(onnx_node.inputs[3]))
        if len(onnx_node.inputs) > 4:
            if onnx_node.inputs[4] in self.tensors:
                sequence_lens = self.getWeightOp(onnx_node.inputs[4])
        if len(onnx_node.inputs) > 5:
            initial_h = self.getWeightOp(onnx_node.inputs[5])
            operands.append(initial_h)
        if len(onnx_node.inputs) > 6:
            initial_c = self.getWeightOp(onnx_node.inputs[6])
            operands.append(initial_c)
        p = {
            "name": "{}_{}".format(onnx_node.name, onnx_node.op_type),
            "have_bias": have_bias,
            "bidirectional": bidirectional,
            "batch_first": batch_first,
        }
        input_shape = self.getShape(onnx_node.inputs[0])
        weight_shape = self.getShape(onnx_node.inputs[1])
        seq_length = input_shape[1] if layout == 1 else input_shape[0]
        batch_size = input_shape[0] if layout == 1 else input_shape[1]
        input_size = input_shape[2]
        num_direction = weight_shape[0]
        output_shape = [seq_length, num_direction, batch_size, hidden_size]
        new_op = self.mlir.create_lstm_op(operands, output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_gather_op(self, onnx_node):
        assert (onnx_node.op_type == "Gather")

        def getOp(name):
            if name in self.operands:
                return self.getOperand(name)
            if name in self.tensors:
                return self.getWeightOp(name)
            raise Exception(KeyError(f"unregistered varivale: {name}"))

        in0 = getOp(onnx_node.inputs[0])
        indices = getOp(onnx_node.inputs[1])
        output_shape = self.getShape(onnx_node.name)
        axis = onnx_node.attrs.get('axis')
        if axis is None:
            axis = 0
        p = {'name': '{}_{}'.format(onnx_node.name, onnx_node.op_type), 'axis': axis}

        new_op = self.mlir.create_gather_op([in0, indices], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_expand_op(self, onnx_node):
        assert (onnx_node.op_type == 'Expand')
        in0 = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        input_shape = self.getShape(onnx_node.inputs[0])
        assert len(output_shape) >= len(input_shape)
        # tile one axis each time to avoid gmem buffer
        count = sum([input_shape[-i] != output_shape[-i] for i in range(1, len(input_shape) + 1)])
        # remove leading 1
        len_diff = len(output_shape) - len(input_shape)
        for i in range(len_diff):
            if output_shape[i] == 1:
                len_diff -= 1
            else:
                break
        count += len_diff
        assert count > 0
        out_shape = copy.deepcopy(input_shape)
        for i in range(1, len(output_shape) + 1):
            axis = len(out_shape) - i
            if axis < 0:
                out_shape.insert(0, 1)
            if output_shape[-i] != out_shape[-i]:
                p = {'axis': axis, 'tile': output_shape[-i] // out_shape[-i]}
                if count == 1:
                    p['name'] = '{}_{}'.format(onnx_node.name, onnx_node.op_type)
                    out_shape = output_shape
                else:
                    p["name"] = "{}_{}_{}".format(onnx_node.name, onnx_node.op_type, count)
                    out_shape[-i] = output_shape[-i]
                new_op = self.mlir.create_tile_op([in0], out_shape, **p)
                in0 = new_op
                count -= 1
            if count == 0:
                break
        self.addOperand(onnx_node.name, new_op)

    def convert_tile_op(self, onnx_node):
        assert (onnx_node.op_type == "Tile")
        in0_op = self.getOperand(onnx_node.inputs[0])
        input_shape = self.getShape(onnx_node.inputs[0])
        tile_data = self.getWeight(onnx_node.inputs[1])
        if np.prod(tile_data) == 1:
            self.addOperand(onnx_node.name, in0_op)
            return
        last_shape = list(input_shape)
        last_op = in0_op
        last_i = 0
        last_name = ""
        for i in range(tile_data.size):
            last_i = tile_data.size - i - 1
            if tile_data[last_i] != 1:
                break
        for i in range(last_i + 1):
            if tile_data[i] == 1:
                continue
            attr = {'axis': i, 'tile': int(tile_data[i])}
            last_name = onnx_node.name
            if i != last_i:
                last_name += "_{}".format(i)
            attr['name'] = "{}_{}".format(last_name, onnx_node.op_type)
            last_shape[i] = int(last_shape[i] * tile_data[i])
            last_op = self.mlir.create_tile_op([last_op], last_shape, **attr)
        self.addOperand(onnx_node.name, last_op)

    def convert_max_op(self, onnx_node):
        assert (onnx_node.op_type == "Max")
        output_shape = self.getShape(onnx_node.name)
        num_dims = len(output_shape)
        operands = [self.getOperand(x) for x in onnx_node.inputs]
        p = {"name": "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        new_op = self.mlir.create_max_op(operands, output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_min_op(self, onnx_node):
        assert (onnx_node.op_type == "Min")
        output_shape = self.getShape(onnx_node.name)
        num_dims = len(output_shape)
        operands = [self.getOperand(x) for x in onnx_node.inputs]
        p = {"name": "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        new_op = self.mlir.create_min_op(operands, output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_abs_op(self, onnx_node):
        assert (onnx_node.op_type == "Abs")
        operand = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        new_op = self.mlir.create_abs_op([operand], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_neg_op(self, onnx_node):
        assert (onnx_node.op_type == "Neg")
        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        operand = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {'name': name, 'const_val': -1.0}
        mul_const_op = self.mlir.create_mul_const_op([operand], output_shape, **p)
        self.addOperand(onnx_node.name, mul_const_op)

    def convert_prelu_op(self, onnx_node):
        assert (onnx_node.op_type == "PRelu")
        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        p = {'name': name}
        in_op = self.getOperand(onnx_node.inputs[0])
        slope_shape = self.getShape(onnx_node.inputs[1])
        num_slope = np.prod(slope_shape)
        if num_slope == 1:
            weight = self.getWeight(onnx_node.inputs[1])
            p['alpha'] = weight.flatten()[0]
            new_op = self.mlir.create_leaky_relu_op([in_op], output_shape, **p)
            self.addOperand(onnx_node.name, new_op)
            return
        in_shape = self.getShape(onnx_node.inputs[0])
        slope_shape = [1] * len(in_shape)
        if len(in_shape) > 1:
            slope_shape[1] = num_slope
        else:
            slope_shape[0] = num_slope
        slope = self.getWeightOp(onnx_node.inputs[1], slope_shape)
        output_shape = self.getShape(onnx_node.name)
        prelu_op = self.mlir.create_prelu_op([in_op, slope], output_shape, **p)
        self.addOperand(onnx_node.name, prelu_op)
