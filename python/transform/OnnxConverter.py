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
from .OnnxOpt import onnx_opt
from onnx import numpy_helper, mapping
from numbers import Number
import onnxsim.onnx_simplifier as onnxsim

import onnx
import onnxruntime
import numpy as np
import random
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
        info["attrs"] = [(attr.name, translate_onnx(attr.name, convert_onnx_attribute_proto(attr)))
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
        self.node_name_mapping = {}  # used in onnx opt
        self.load_onnx_model(onnx_file, input_shapes, output_names)
        self.init_MLIRImporter()
        self.preprocess_args = preprocess_args
        self.converted_nodes = list()

        self.onnxop_factory = {
            # NOTICE: Please add the Op alphabetically !!!
            "Abs": lambda node: self.convert_abs_op(node),
            "Add": lambda node: self.convert_add_op(node),
            "ArgMax": lambda node: self.convert_arg_op(node),
            "ArgMin": lambda node: self.convert_arg_op(node),
            "Sub": lambda node: self.convert_sub_op(node),
            "AveragePool": lambda node: self.convert_avgpool_op(node),
            "BatchNormalization": lambda node: self.convert_batchnorm_op(node),
            "Cast": lambda node: self.convert_cast_op(node),
            "Concat": lambda node: self.convert_concat_op(node),
            "Constant": lambda node: self.convert_constant_op(node),
            "Conv": lambda node: self.convert_conv_op(node),
            "Clip": lambda node: self.convert_clip_op(node),
            "ConvTranspose": lambda node: self.convert_conv_transpose_op(node),
            "DepthToSpace": lambda node: self.convert_depth2space_op(node),
            "DequantizeLinear": lambda node: self.convert_deqlinear_op(node),
            "Div": lambda node: self.convert_div_op(node),
            "Dropout": lambda node: self.convert_skip_op(node),
            "Erf": lambda node: self.convert_erf_op(node),
            "Exp": lambda node: self.convert_exp_op(node),
            "Expand": lambda node: self.convert_expand_op(node),
            "Equal": lambda node: self.convert_cmp_op(node),
            "Flatten": lambda node: self.convert_flatten_op(node),
            "Floor": lambda node: self.convert_floor_op(node),
            "Gather": lambda node: self.convert_gather_op(node),
            "GELU": lambda node: self.convert_gelu_op(node),
            "Gemm": lambda node: self.convert_gemm_op(node),
            "GlobalAveragePool": lambda node: self.convert_global_avgpool_op(node),
            "GlobalMaxPool": lambda node: self.convert_global_maxpool_op(node),
            "Greater": lambda node: self.convert_cmp_op(node),
            "GreaterOrEqual": lambda node: self.convert_cmp_op(node),
            "GRU": lambda node: self.convert_gru_op(node),
            "HardSigmoid": lambda node: self.convert_hsigmoid_op(node),
            "HardSwish": lambda node: self.convert_hswish_op(node),
            "Identity": lambda node: self.convert_skip_op(node),
            "LayerNormalization": lambda node: self.convert_layer_norm_op(node),
            "LeakyRelu": lambda node: self.convert_leaky_relu_op(node),
            "Log": lambda node: self.convert_log_op(node),
            "LRN": lambda node: self.convert_lrn_op(node),
            "LSTM": lambda node: self.convert_lstm_op(node),
            "LogSoftmax": lambda node: self.convert_softmax_op(node),
            "Less": lambda node: self.convert_cmp_op(node),
            "LessOrEqual": lambda node: self.convert_cmp_op(node),
            "MatMul": lambda node: self.convert_gemm_op(node),
            "Max": lambda node: self.convert_max_op(node),
            "MaxPool": lambda node: self.convert_maxpool_op(node),
            "Min": lambda node: self.convert_min_op(node),
            "Mul": lambda node: self.convert_mul_op(node),
            "Neg": lambda node: self.convert_neg_op(node),
            "Pad": lambda node: self.convert_pad_op(node),
            "PixelNormalization": lambda node: self.convert_pixel_norm_op(node),
            "PRelu": lambda node: self.convert_prelu_op(node),
            "Pow": lambda node: self.convert_pow_op(node),
            "QuantizeLinear": lambda node: self.convert_qlinear_op(node),
            "Reciprocal": lambda node: self.convert_reciprocal_op(node),
            "ReduceMean": lambda node: self.convert_reduce_op(node),
            "ReduceMax": lambda node: self.convert_reduce_op(node),
            "ReduceMin": lambda node: self.convert_reduce_op(node),
            "ReduceL2": lambda node: self.convert_reduce_op(node),
            "ReduceL1": lambda node: self.convert_reduce_op(node),
            "ReduceProd": lambda node: self.convert_reduce_op(node),
            "ReduceSum": lambda node: self.convert_reduce_op(node),
            "Relu": lambda node: self.convert_relu_op(node),
            "Reshape": lambda node: self.convert_reshape_op(node),
            "Resize": lambda node: self.convert_resize_op(node),
            "ScatterND": lambda node: self.convert_scatternd_op(node),
            "Shape": lambda node: self.convert_shape_op(node),
            "Sigmoid": lambda node: self.convert_sigmoid_op(node),
            "Slice": lambda node: self.convert_slice_op(node),
            "Softmax": lambda node: self.convert_softmax_op(node),
            "Softplus": lambda node: self.convert_softplus_op(node),
            "Squeeze": lambda node: self.convert_squeeze_op(node),
            "Split": lambda node: self.convert_split_op(node),
            "Sum": lambda node: self.convert_sum_op(node),
            "Sqrt": lambda node: self.convert_sqrt_op(node),
            "Tanh": lambda node: self.convert_tanh_op(node),
            "Tile": lambda node: self.convert_tile_op(node),
            "TopK": lambda node: self.convert_topk_op(node),
            "Transpose": lambda node: self.convert_transpose_op(node),
            "Unsqueeze": lambda node: self.convert_unsqueeze_op(node),
            "Upsample":lambda node:self.convert_upsample_op(node),
            "Where": lambda node: self.convert_where_op(node),
        }

    def __del__(self):
        if self.mlir != None:
            del self.mlir
            self.mlir = None

    def check_need(self, name):
        for node in self.converted_nodes:
            for i in node.inputs:
                if i == name:
                    return True
        for name in self.output_names:
            return True
        return False

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
        # node map name
        self.all_nodes = {}
        for x in self.model.graph.node:
            for o in x.output:
                self.all_nodes[o] = x
                if o in output_names:
                    intermediate_layer_value_info = onnx.helper.ValueInfoProto()
                    intermediate_layer_value_info.name = o
                    self.model.graph.output.append(intermediate_layer_value_info)
                    output_names.remove(o)
                    self.all_outputs.append(o)
        if len(output_names) != 0:
            raise RuntimeError("Error, can't find {} in model".format(output_names))
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

    def get_outputs(self, model: onnx.ModelProto):
        initializer_names = [x.name for x in model.graph.initializer]
        return [opt for opt in model.graph.output if opt.name not in initializer_names]

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

    def clean_up_shape_info(self):
        # uncomplete shape info may cause onnxsim.simplify failed.
        if self.model.graph.value_info:
            n = len(self.model.graph.value_info)
            for _ in range(n):
                v = self.model.graph.value_info[0]
                self.model.graph.value_info.remove(v)

    def model_simplify(self):
        self.clean_up_shape_info()
        is_ok = False
        times = 0
        for i in range(5):
            try:
                model_simplified, is_ok = onnxsim.simplify(self.model)
            except:
                is_ok = False
            if not is_ok:
                break
            if model_simplified == self.model:
                break
            times += 1
            self.model = model_simplified
        print("Run onnxsim {} times, model simplified: {}".format(times, is_ok))
        if not is_ok:
            try:
                self.model = onnx.shape_inference.infer_shapes(self.model)
            except:
                return is_ok
        return is_ok

    def load_onnx_model(self, onnx_file, input_shapes: list, output_names: list):
        if isinstance(onnx_file, str):
            self.model = onnx.load(onnx_file)
        else:
            self.model = onnx_file
        is_ok_ = self.model_simplify()  # need shape_info for select_output
        # select_output before model_shape_infer to remove useless inputs
        # so that those inputs dont have to be specified in cfg file
        if (not is_ok_):
            print("WARNING: Onnx-sim failed please check onnx model.")
        if output_names:
            self.select_output(output_names)
        self.input_names = self.get_input_names(self.model)
        self.num_input = len(self.input_names)
        self.input_shape_assign(input_shapes)
        self.input_shapes = self.get_input_shapes(self.model)
        self.input_types = self.get_input_types(self.model)
        is_ok = self.model_simplify()
        if (is_ok_ and not is_ok):
            print("WARNING: Onnx-sim failed caused by assign input_shape.")
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
        if is_ok:
            # fuse ops such as layernorm gelu...
            self.model, self.node_name_mapping = onnx_opt(self.model, True)

    def input_shape_assign(self, input_shapes):
        inputs = self.get_inputs(self.model)
        outputs = self.get_outputs(self.model)
        shape_changed = False
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
                # elif not no_shape:
                #     check_shape(_dim_value, input_shapes)
                elif not no_shape and input_shapes[idx][_i] != _dim.dim_value:
                    _dim.dim_value = input_shapes[idx][_i]
                    shape_changed = True
                _shape.append(_dim.dim_value)
            self.addShape(input.name, _shape)

        for o in outputs:
            # for set arbitrary batch_size
            _odims = o.type.tensor_type.shape.dim
            for _odim in _odims:
                if _odim.dim_value <= 0 or shape_changed:
                    _odim.dim_param = '?'

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
            image = (len(input_shape) == 4 and input_shape[channel_axis] <= 4) or \
                    (len(input_shape) == 3)  # gray
            if not self.preprocess_args or not image:
                input_op = self.mlir.create_input_op(_name, idx, **{})
            else:
                input_op = self.mlir.create_input_op(_name, idx, **self.preprocess_args)
            self.addOperand(_name, input_op)

        def NoneAndRaise(node):
            raise RuntimeError("{} Op not support now".format(node.op_type))

        self.converted_nodes.clear()
        for n in self.model.graph.node:
            node = OnnxNode(n)
            self.converted_nodes.append(node)
        # checkout all type is supported
        unsupported = set()
        for n in self.converted_nodes:
            if n.op_type not in self.onnxop_factory:
                unsupported.add(n.op_type)
        if unsupported:
            raise RuntimeError("Op not support:{}".format(unsupported))

        for n in self.converted_nodes:
            self.onnxop_factory.get(n.op_type, lambda x: NoneAndRaise(x))(n)
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
        output_shape = self.getShape(onnx_node.name)
        lhs = onnx_node.inputs[0]
        rhs = onnx_node.inputs[1]
        if self.isWeight(lhs) and not self.isWeight(rhs):
            onnx_node.inputs[0], onnx_node.inputs[1] = onnx_node.inputs[1], onnx_node.inputs[0]
            self.convert_add_op(onnx_node)
            return
        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        p = {'name': name}
        if not self.isWeight(lhs) and self.isWeight(rhs):
            opd1_num_elem = np.prod(self.getShape(rhs))
            channel = output_shape[1]
            lhs_op = self.getOp(lhs)
            if self.isScalar(rhs):
                p['do_relu'] = False
                p['const_val'] = self.getScalar(rhs)
                new_op = self.mlir.create_add_const_op([lhs_op], output_shape, **p)
            elif opd1_num_elem == channel:
                bias = self.getWeight(rhs)
                weight_data = np.ones_like(bias)
                self.addWeight(name + '_scale', weight_data)
                weight_op = self.getWeightOp(name + '_scale')
                bias_op = self.getWeightOp(rhs)
                new_op = self.mlir.create_scale_op([lhs_op, weight_op, bias_op], output_shape, **p)
            else:
                rhs_op = self.getOp(rhs)
                new_op = self.mlir.create_add_op([lhs_op, rhs_op], output_shape, **p)
        else:
            lhs_op = self.getOp(lhs)
            rhs_op = self.getOp(rhs)
            new_op = self.mlir.create_add_op([lhs_op, rhs_op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_sub_op(self, onnx_node):
        assert (onnx_node.op_type == "Sub")
        assert (len(onnx_node.inputs) == 2)
        output_shape = self.getShape(onnx_node.name)
        lhs = onnx_node.inputs[0]
        rhs = onnx_node.inputs[1]
        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        p = {'name': name}
        new_op = None
        if self.isScalar(lhs):
            # lhs_const + (-1 * rhs)
            attr = {'name': name + "_unm", 'const_val': -1}
            rhs_op = self.getOp(rhs)
            unm_op = self.mlir.create_mul_const_op([rhs_op], output_shape, **attr)
            p['const_val'] = self.getScalar(lhs)
            new_op = self.mlir.create_add_const_op([unm_op], output_shape, **p)
        elif self.isScalar(rhs):
            # lhs + (-rhs_const)
            p['const_val'] = -self.getScalar(rhs)
            lhs_op = self.getOp(lhs)
            new_op = self.mlir.create_add_const_op([lhs_op], output_shape, **p)
        else:
            lhs_op = self.getOp(lhs)
            rhs_op = self.getOp(rhs)
            new_op = self.mlir.create_sub_op([lhs_op, rhs_op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

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

    def convert_cast_op(self, onnx_node):
        assert (onnx_node.op_type == "Cast")
        if self.isWeight(onnx_node.inputs[0]):
            dtype = onnx_node.attrs.get('to')
            data = self.getWeight(onnx_node.inputs[0])
            if dtype == "int64":
                data = data.astype(np.int64)
            elif dtype == "int32":
                data = data.astype(np.int32)
            elif dtype == "float32":
                data = data.astype(np.float32)
            elif dtype == "bool":
                data = data.astype(np.bool)
            else:
                raise RuntimeError("{} dtype not support, please add".format(dtype))
            self.addWeight(onnx_node.name, data)
        else:
            op = self.getOperand(onnx_node.inputs[0])
            self.addOperand(onnx_node.name, op)

    def convert_concat_op(self, onnx_node):
        assert (onnx_node.op_type == "Concat")
        output_shape = self.getShape(onnx_node.name)
        num_dims = len(output_shape)
        axis = onnx_node.attrs['axis']
        if axis < 0:
            axis += num_dims
        operands = list()
        weight_data = None
        for x in onnx_node.inputs:
            x_shape = self.getShape(x)
            num_elem = np.prod(x_shape)
            if num_elem == 0:
                print("WARNING:{}'s shape is strange {}".format(x, x_shape))
                continue
            if self.isWeight(x):
                data = self.getWeight(x)
                if weight_data is not None:
                    weight_data = np.concatenate((weight_data, data), axis=axis)
                else:
                    weight_data = data
                continue
            else:
                if weight_data is not None:
                    w_name = x + "_weight"
                    self.addWeight(w_name, weight_data)
                    operands.append(self.getWeightOp(w_name))
                    weight_data = None
                operands.append(self.getOperand(x))
        if len(operands) == 0:
            # all weight
            self.addWeight(onnx_node.name, weight_data)
            return
        if weight_data is not None:
            w_name = onnx_node.name + "_weight"
            self.addWeight(w_name, weight_data)
            operands.append(self.getWeightOp(w_name))
        if len(operands) == 1:
            self.addOperand(onnx_node.name, operands[0])
            return
        p = {"name": "{}_{}".format(onnx_node.name, onnx_node.op_type), "axis": axis}
        new_op = self.mlir.create_concat_op(operands, output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_constant_op(self, onnx_node):
        """
            Constant Op is tensor data at IR,
            we change it to load weight tensor, and store
        """
        assert (onnx_node.op_type == "Constant")
        onnx_tensor = onnx_node.attrs['value']
        np_tensor = numpy_helper.to_array(onnx_tensor)
        data_type = onnx_dtype(onnx_tensor.data_type)
        if data_type in [np.float32, np.float64, np.int32, np.int64, np.uint8, np.bool]:
            self.addWeight(onnx_node.name, np_tensor.astype(data_type))
        else:
            raise ValueError("Not Support {} type".format(data_type))

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
        pads = []
        if auto_pad:
            pads = set_auto_pad(auto_pad, input_shape, kernel_shape, strides)
        if len(pads) == 0:
            pads = onnx_node.attrs.get("pads", dim * 2 * [0])

        operands = list()
        operands.append(op)
        filter_op = self.getOp(onnx_node.inputs[1])
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

    def convert_floor_op(self, onnx_node):
        assert (onnx_node.op_type == "Floor")
        op = self.getOperand(onnx_node.inputs[0])
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
        }
        output_shape = self.getShape(onnx_node.name)
        new_op = self.mlir.create_floor_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_gemm_op(self, onnx_node):
        assert (onnx_node.op_type == "Gemm" or onnx_node.op_type == 'MatMul')
        # (M, K) * (K, N) => (M, N)
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
        auto_pad = onnx_node.attrs.get("auto_pad", b"NOTSET")
        pads = []
        if auto_pad:
            pads = set_auto_pad(auto_pad, input_shape, kernel_shape, strides)
        if len(pads) == 0:
            pads = onnx_node.attrs.get("pads", dim * 2 * [0])
        if np.prod(kernel_shape) == 1 and np.sum(pads) == 0 and np.prod(strides) == 1:
            self.addOperand(onnx_node.name, op)
            return
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
        pads = []
        if auto_pad:
            pads = set_auto_pad(auto_pad, input_shape, kernel_shape, strides)
        if len(pads) == 0:
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
        lhs = onnx_node.inputs[0]
        rhs = onnx_node.inputs[1]
        if self.isWeight(lhs) and not self.isWeight(rhs):
            onnx_node.inputs[0], onnx_node.inputs[1] = rhs, lhs
            self.convert_mul_op(onnx_node)
            return
        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        if (not self.isWeight(lhs)) and self.isWeight(rhs):
            op0 = self.getOperand(lhs)
            rhs = rhs
            output_shape = self.getShape(onnx_node.name)
            if self.isScalar(rhs):
                p = {'name': name, 'const_val': self.getScalar(rhs)}
                mul_const_op = self.mlir.create_mul_const_op([op0], output_shape, **p)
                self.addOperand(onnx_node.name, mul_const_op)
                return
            weight_num_elem = np.prod(self.getShape(rhs))
            channel = output_shape[1]
            if weight_num_elem == channel:
                weight = self.getWeight(rhs)
                offset_data = np.zeros_like(weight)
                self.addWeight(name + '_bias', offset_data)
                weight_op = self.getWeightOp(rhs)
                offset_op = self.getWeightOp(name + '_bias')
                p = {'name': name}
                scale_op = self.mlir.create_scale_op([op0, weight_op, offset_op], output_shape, **p)
                self.addOperand(onnx_node.name, scale_op)
                return
            const_op = self.getWeightOp(rhs)
            p = {'name': name}
            scale_op = self.mlir.create_mul_op([op0, const_op], output_shape, **p)
            self.addOperand(onnx_node.name, scale_op)
            return
        else:
            op0 = self.getOperand(lhs)
            op1 = self.getOperand(rhs)
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

    def convert_upsample_op(self, onnx_node):
        assert (onnx_node.op_type == "Upsample")
        mode = onnx_node.attrs.get("mode", "nearest")
        op = self.getOperand(onnx_node.inputs[0])
        input_shape = self.getShape(onnx_node.inputs[0])
        scale_factor = []
        sizes = []
        scale_factor = self.getWeight(onnx_node.inputs[1])
        sizes = input_shape * scale_factor
        output_shape = [int(i) for i in sizes]
        scale_h = scale_factor[2]   # scale [n, c, h, w]
        scale_w = scale_factor[3]
        coord_mode = onnx_node.attrs.get("coordinate_transformation_mode", "half_pixel")
        if mode == b'nearest' and scale_h == int(scale_h) and scale_w == int(scale_w):
            self.resize_to_upsample(onnx_node, op, input_shape, output_shape, scale_h, scale_w)
            return
        else:
            self.resize_to_interp(onnx_node, op, input_shape, output_shape, scale_h, scale_w, mode,
                                  coord_mode)
            return

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

    def convert_shape_op(self, onnx_node):
        assert (onnx_node.op_type == "Shape")
        input_shape = self.getShape(onnx_node.inputs[0])
        data = np.array(input_shape)
        self.addWeight(onnx_node.name, data)

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
        if self.isWeight(onnx_node.inputs[0]):
            tensor_data = self.getWeight(onnx_node.inputs[0])
            for start, end, axis, step in zip(starts, ends, axes, steps):
                start, end, axis, step = int(start), int(end), int(axis), int(step)
                if axis < 0:
                    axis = axis + num_dims
                s = slice(start, end, step)
                tensor_data = tensor_data[(slice(None), ) * axis + (s, )]
            self.addWeight(onnx_node.name, tensor_data)
            return
        op = self.getOperand(onnx_node.inputs[0])
        slice_shape = list(input_shape)
        slice_offset = [0] * num_dims
        slice_step = [1] * num_dims
        for start, end, axis, step in zip(starts, ends, axes, steps):
            start, end, axis, step = int(start), int(end), int(axis), int(step)
            if axis < 0:
                axis = axis + num_dims
            if end < 0:
                end = end + input_shape[axis]
            if start < 0:
                start = start + input_shape[axis]
            if end > input_shape[axis] or end < -1:
                end = input_shape[axis]
            elif end < 0:
                if step < 0:
                    end = -1
                else:
                    end = input_shape[axis]
            slice_shape[axis] = (abs(end - start) + abs(step) - 1) // abs(step)
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
        assert (onnx_node.op_type in ("Softmax", "LogSoftmax"))
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
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'axis': axis,
            'log': onnx_node.op_type == "LogSoftmax"
        }
        new_op = self.mlir.create_softmax_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_softplus_op(self, onnx_node):
        assert (onnx_node.op_type == "Softplus")
        op = self.getOperand(onnx_node.inputs[0])
        input_shape = self.getShape(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
        }
        new_op = self.mlir.create_softplus_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_log_op(self, onnx_node):
        assert (onnx_node.op_type == "Log")
        op = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        new_op = self.mlir.create_log_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_exp_op(self, onnx_node):
        assert (onnx_node.op_type == "Exp")
        op = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        new_op = self.mlir.create_exp_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_erf_op(self, onnx_node):
        assert (onnx_node.op_type == "Erf")
        op = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        new_op = self.mlir.create_erf_op([op], output_shape, **p)
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
        if len(onnx_node.inputs) > 1:
            pads = list(self.getWeight(onnx_node.inputs[1]))
        else:
            pads = onnx_node.attrs.get("pads")
        if pads == None:
            raise RuntimeError("No paddings value")
        if len(pads) != 2 * len(input_shape):
            raise RuntimeError(
                "pads number is two times as same as input shape ({} v.s 2 * {})".format(
                    len(pads), len(input_shape)))
        # opset 11, value from second input
        val = 0.0
        if len(onnx_node.inputs) > 2 and onnx_node.inputs[2]:
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
        assert (onnx_node.op_type == "Div")
        assert (len(onnx_node.inputs) == 2)
        lhs = onnx_node.inputs[0]
        rhs = onnx_node.inputs[1]
        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        p = {'name': name}
        output_shape = self.getShape(onnx_node.name)
        if self.isScalar(lhs):
            # lhs_const * (1 / rhs)
            attr = {'name': name + "_rcp"}
            rhs_op = self.getOp(rhs)
            rcp_op = self.mlir.create_reciprocal_op([rhs_op], output_shape, **attr)
            p['const_val'] = self.getScalar(lhs)
            new_op = self.mlir.create_mul_const_op([rcp_op], output_shape, **p)
        elif self.isScalar(rhs):
            # lhs * (1 / rhs_const)
            p['const_val'] = 1 / self.getScalar(rhs)
            lhs_op = self.getOp(lhs)
            new_op = self.mlir.create_mul_const_op([lhs_op], output_shape, **p)
        else:
            lhs_op = self.getOp(lhs)
            rhs_op = self.getOp(rhs)
            new_op = self.mlir.create_div_op([lhs_op, rhs_op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_reciprocal_op(self, onnx_node):
        assert (onnx_node.op_type == "Reciprocal")
        assert len(onnx_node.inputs) == 1
        op0 = self.getOperand(onnx_node.inputs[0])
        p = {"name": "{}_{}".format(onnx_node.name, onnx_node.op_type), "const_val": 1}
        output_shape = self.getShape(onnx_node.name)
        div_op = self.mlir.create_reciprocal_op([op0], output_shape, **p)
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
            p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type), 'min': min, 'max': max}
            new_op = self.mlir.create_clip_op([input], input_shape, **p)
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
        split = None
        # to avoid the case that split attr in input
        if len(onnx_node.inputs) > 1:
            split = self.getWeight(onnx_node.inputs[1]).astype(int)
        else:
            split = onnx_node.attrs.get('split', [slice] * num_output)
        op = self.getOperand(onnx_node.inputs[0])

        offset = 0
        # replace the split with slice
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

    # support max ndims to 6
    def convert_reduce_op(self, onnx_node):
        assert (onnx_node.op_type
                in ["ReduceMin", "ReduceMax", "ReduceMean", "ReduceProd", "ReduceL2", "ReduceL1", "ReduceSum"])
        input_shape = self.getShape(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        op = self.getOperand(onnx_node.inputs[0])
        if (np.prod(input_shape) == np.prod(output_shape)):
            p = {"name": "{}_{}".format(onnx_node.name, onnx_node.op_type)}
            new_op = self.mlir.create_reshape_op([op], output_shape, **p)
            self.addOperand(onnx_node.name, new_op)
            return
        num_dims = len(input_shape)
        axes = onnx_node.attrs.get('axes', list(range(num_dims))) \
            if len(onnx_node.inputs) == 1 else self.getWeight(onnx_node.inputs[1])
        keepdims = onnx_node.attrs.get('keepdims', 1)
        for idx, ax in enumerate(axes):
            if ax < 0:
                axes[idx] += num_dims
        axes.sort()
        if onnx_node.op_type in ["ReduceMean", "ReduceMax"] and (num_dims == 4 and axes == [2, 3]):
            p = {
                'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
                'kernel_shape': input_shape[2:],
                'strides': [1, 1],
                'pads': [0, 0, 0, 0],
                'count_include_pad': True,
                'do_relu': False,
            }
            new_op = self.mlir.create_avgpool_op(
                [op], output_shape, **
                p) if onnx_node.op_type == "ReduceMean" else self.mlir.create_maxpool_op(
                    [op], output_shape, **p)
            self.addOperand(onnx_node.name, new_op)
            return
        p = {
            "name": "{}_{}".format(onnx_node.name, onnx_node.op_type),
            "axes": axes,
            "keepdims": keepdims,
            "mode": onnx_node.op_type
        }
        new_op = self.mlir.create_reduce_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_arg_op(self, onnx_node):
        assert (onnx_node.op_type in ["ArgMin", "ArgMax"])
        output_shape = self.getShape(onnx_node.name)
        op = self.getOperand(onnx_node.inputs[0])
        axis = onnx_node.attrs.get('axis', 0)
        keepdims = onnx_node.attrs.get('keepdims', 1)
        p = {
            "name": "{}_{}".format(onnx_node.name, onnx_node.op_type),
            "axis": axis,
            "keepdims": keepdims,
            "mode": onnx_node.op_type
        }
        # out_shapes = [[], []]
        # out_shapes[0] = self.getShape(onnx_node.name)
        # out_shapes[1] = self.getShape(onnx_node.name)
        new_op = self.mlir.create_arg_op([op], output_shape, **p)
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

    def convert_gru_op(self, onnx_node):
        assert (onnx_node.op_type == "GRU")
        direction = onnx_node.attrs.get("direction", 'forward')
        layout = onnx_node.attrs.get("layout", 0)
        hidden_size = onnx_node.attrs.get("hidden_size")
        batch_first = True if layout == 1 else False
        operands = list()
        operands.append(self.getOperand(onnx_node.inputs[0]))  # in
        operands.append(self.getWeightOp(onnx_node.inputs[1]))  # W
        operands.append(self.getWeightOp(onnx_node.inputs[2]))  # R
        num_inputs = len(onnx_node.inputs)
        bias_op, init_h_op = self.mlir.none_op, self.mlir.none_op
        if num_inputs > 3:
            bias_op = self.getWeightOp(onnx_node.inputs[3])
        if num_inputs > 4 and len(onnx_node.inputs[4]) != 0:
            raise RuntimeError("LSTM does not test the case of specify the sequence_lens.")
        if num_inputs > 5 and len(onnx_node.inputs[5]) != 0:
            init_h_op = self.getOp(onnx_node.inputs[5])
        operands.extend([bias_op, init_h_op])
        p = {
            "name": [onnx_node.name + '_GRU', onnx_node.name + '_H'],
            "hidden_size": hidden_size,
            "bidirectional": direction == b'bidirectional',
            "batch_first": batch_first,
        }
        out_shapes = [[], []]
        out_needs = [False, False]
        for idx, out in enumerate(onnx_node.outputs):
            need = len(out) > 0 and self.check_need(out)
            if need:
                p['name'][idx] = "{}_{}".format(out, onnx_node.op_type)
                out_needs[idx] = True
                out_shapes[idx] = self.getShape(out)
        new_op, h_op = self.mlir.create_gru_op(operands, out_shapes, **p)
        out_ops = [new_op, h_op]
        for idx, need in enumerate(out_needs):
            if need:
                self.addOperand(onnx_node.outputs[idx], out_ops[idx])

    def convert_lstm_op(self, onnx_node):
        assert (onnx_node.op_type == "LSTM")
        direction = onnx_node.attrs.get("direction", 'forward')
        layout = onnx_node.attrs.get("layout", 0)
        hidden_size = onnx_node.attrs.get("hidden_size")
        batch_first = True if layout == 1 else False
        operands = list()
        operands.append(self.getOperand(onnx_node.inputs[0]))  # in
        operands.append(self.getWeightOp(onnx_node.inputs[1]))  # W
        operands.append(self.getWeightOp(onnx_node.inputs[2]))  # R
        num_inputs = len(onnx_node.inputs)
        bias_op, init_h_op, init_c_op = self.mlir.none_op, self.mlir.none_op, self.mlir.none_op
        if num_inputs > 3 and len(onnx_node.inputs[3]) != 0:
            bias_op = self.getWeightOp(onnx_node.inputs[3])
        if num_inputs > 4 and len(onnx_node.inputs[4]) != 0:
            raise RuntimeError("LSTM does not test the case of specify the sequence_lens.")
        if num_inputs > 5 and len(onnx_node.inputs[5]) != 0:
            init_h_op = self.getOp(onnx_node.inputs[5])
        if num_inputs > 6 and len(onnx_node.inputs[5]) != 0:
            init_c_op = self.getOp(onnx_node.inputs[6])
        operands.extend([bias_op, init_h_op, init_c_op])
        p = {
            "name": [onnx_node.name + '_LSTM', onnx_node.name + '_H', onnx_node.name + '_C'],
            "hidden_size": hidden_size,
            "bidirectional": direction == b'bidirectional',
            "batch_first": batch_first,
        }
        out_shapes = [[], [], []]
        out_needs = [False, False, False]
        for idx, out in enumerate(onnx_node.outputs):
            need = len(out) > 0 and self.check_need(out)
            if need:
                p['name'][idx] = "{}_{}".format(out, onnx_node.op_type)
                out_needs[idx] = True
                out_shapes[idx] = self.getShape(out)
        new_op, h_op, c_op = self.mlir.create_lstm_op(operands, out_shapes, **p)
        out_ops = [new_op, h_op, c_op]
        for idx, need in enumerate(out_needs):
            if need:
                self.addOperand(onnx_node.outputs[idx], out_ops[idx])

    def convert_gather_op(self, onnx_node):
        assert (onnx_node.op_type == "Gather")
        in0 = self.getOp(onnx_node.inputs[0])
        in0_shape = self.getShape(onnx_node.inputs[0])
        out_shape = self.getShape(onnx_node.name)
        axis = onnx_node.attrs.get('axis', 0)
        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        if  self.isScalar(onnx_node.inputs[1]):
            offset = int(self.getScalar(onnx_node.inputs[1]))
            if offset < 0:
                offset = in0_shape[axis] + offset
            slice_offset = [0] * len(in0_shape)
            slice_steps = [1] * len(in0_shape)
            slice_offset[axis] = offset
            p = {
                'name': name,
                'offset': list(slice_offset),
                'steps': list(slice_steps)
            }
            if axis == 0:
                new_op = self.mlir.create_slice_op([in0], out_shape, **p)
                self.addOperand(onnx_node.name, new_op)
                return
            p['name'] = "{}_Slice".format(onnx_node.name)
            slice_shape = list(np.take(np.ones(in0_shape), np.array([offset]), axis=axis).shape)
            slice_op = self.mlir.create_slice_op([in0], slice_shape, **p)
            new_op = self.mlir.create_reshape_op([slice_op], out_shape, name=name)
            self.addOperand(onnx_node.name, new_op)
            return
        indices = self.getOp(onnx_node.inputs[1])
        p = {'name': name, 'axis': axis}
        new_op = self.mlir.create_gather_op([in0, indices], out_shape, **p)
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

    def convert_topk_op(self, onnx_node):
        assert (onnx_node.op_type == "TopK")
        in_op = self.getOperand(onnx_node.inputs[0])
        in_shape = self.getShape(onnx_node.inputs[0])
        K = self.getScalar(onnx_node.inputs[1])
        axis = onnx_node.attrs.get('axis', -1)
        largest = onnx_node.attrs.get('largest', True)
        sorted = onnx_node.attrs.get('sorted', True)
        num_dim = len(in_shape)
        if axis < 0:
            axis += num_dim
        p = {
            "name": [onnx_node.name + '_TopK_indices', onnx_node.name + "_TopK_values"],
            "axis": axis,
            "K": K,
            "largest": largest,
            "sorted": sorted,
        }
        out_shapes = [[], []]
        out_needs = [False, False]
        for idx, out in enumerate(onnx_node.outputs):
            need = len(out) > 0 and self.check_need(out)
            if need:
                p['name'][idx] = "{}_{}".format(out, onnx_node.op_type)
                out_needs[idx] = True
                out_shapes[idx] = self.getShape(out)
        idx_op, val_op = self.mlir.create_topk_op([in_op], out_shapes, **p)
        out_ops = [idx_op, val_op]
        for idx, need in enumerate(out_needs):
            if need:
                self.addOperand(onnx_node.outputs[idx], out_ops[idx])

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
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        lhs = onnx_node.inputs[0]
        rhs = onnx_node.inputs[1]
        in_op = self.getOperand(lhs)
        output_shape = self.getShape(onnx_node.name)
        if self.isScalar(rhs):
            p['alpha'] = self.getScalar(rhs)
            new_op = self.mlir.create_leaky_relu_op([in_op], output_shape, **p)
            self.addOperand(onnx_node.name, new_op)
            return
        slope_shape = self.getShape(rhs)
        num_slope = np.prod(slope_shape)
        in_shape = self.getShape(lhs)
        slope_shape = [1] * len(in_shape)
        if len(in_shape) > 1:
            slope_shape[1] = num_slope
        else:
            slope_shape[0] = num_slope
        slope = self.getWeightOp(rhs, slope_shape)
        prelu_op = self.mlir.create_prelu_op([in_op, slope], output_shape, **p)
        self.addOperand(onnx_node.name, prelu_op)

    def convert_sum_op(self, onnx_node):
        assert (onnx_node.op_type == "Sum")
        opd0 = self.getOperand(onnx_node.inputs[0])
        num_inputs = len(onnx_node.inputs)
        for i in range(1, num_inputs):
            opd1 = self.getOperand(onnx_node.inputs[i])
            output_shape = self.getShape(onnx_node.name)
            last_name = onnx_node.name
            if i != num_inputs - 1:
                last_name += "_{}".format(str(i))
            p = {'do_relu': False, 'relu_limit': 0}
            p['name'] = "{}_{}".format(last_name, onnx_node.op_type)
            opd0 = self.mlir.create_add_op([opd0, opd1], output_shape, **p)
        self.addOperand(onnx_node.name, opd0)

    def convert_sqrt_op(self, onnx_node):
        assert (onnx_node.op_type == "Sqrt")
        operand = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        new_op = self.mlir.create_sqrt_op([operand], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_tanh_op(self, onnx_node):
        assert (onnx_node.op_type == "Tanh")
        op = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        new_op = self.mlir.create_tanh_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_pow_op(self, onnx_node):
        assert (onnx_node.op_type == "Pow")
        assert (len(onnx_node.inputs) == 2)
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        base = onnx_node.inputs[0]
        expn = onnx_node.inputs[1]
        if self.isScalar(expn):
            base_op = self.getOp(base)
            expn_const = self.getScalar(expn)
            output_shape = self.getShape(onnx_node.name)
            if expn_const == 1.0:
                self.addOperand(onnx_node.name, base_op)
                return
            if expn_const == 2.0:
                mul_op = self.mlir.create_mul_op([base_op, base_op], output_shape, **p)
                self.addOperand(onnx_node.name, mul_op)
                return
            else:
                p['exponent'] = expn_const
                pow_op = self.mlir.create_pow_op([base_op], output_shape, **p)
                self.addOperand(onnx_node.name, pow_op)
        else:
            raise RuntimeError("Not implemented")

    def convert_where_op(self, onnx_node):
        assert (onnx_node.op_type == "Where")
        assert (len(onnx_node.inputs) == 3)
        cond = onnx_node.inputs[0]
        tbrn = onnx_node.inputs[1]
        fbrn = onnx_node.inputs[2]
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        cond_opd = self.getOp(cond)
        tbrn_opd = self.getOp(tbrn)
        fbrn_opd = self.getOp(fbrn)
        num_const = 0
        if self.isScalar(tbrn):
            num_const += 1
        else:
            assert (self.getShape(cond) == self.getShape(tbrn)
                    )  # do not support broadcastable case recently
        if self.isScalar(fbrn):
            num_const += 1
        else:
            assert (self.getShape(cond) == self.getShape(fbrn)
                    )  # do not support broadcastable case recently
        output_shape = self.getShape(onnx_node.name)
        if num_const == 0:
            new_op = self.mlir.create_where_op([cond_opd, tbrn_opd, fbrn_opd], output_shape, **p)
        elif num_const == 1:
            brn_opd = fbrn_opd if self.isScalar(tbrn) else tbrn_opd
            if self.isScalar(tbrn):
                p['inversed'] = True
                p['const_val'] = self.getScalar(tbrn)
            else:
                p['inversed'] = False
                p['const_val'] = self.getScalar(fbrn)
            new_op = self.mlir.create_masked_fill_op([cond_opd, brn_opd], output_shape, **p)
        else:
            assert (0)  # TODO: to be implement
        self.addOperand(onnx_node.name, new_op)

    def convert_cmp_op(self, onnx_node):
        supports = {"Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"}
        assert (onnx_node.op_type in supports)
        assert (len(onnx_node.inputs) == 2)
        p = {"name": "{}_{}".format(onnx_node.name, onnx_node.op_type), "mode": onnx_node.op_type}
        lhs = onnx_node.inputs[0]
        rhs = onnx_node.inputs[1]
        output_shape = self.getShape(onnx_node.name)
        if self.isScalar(lhs):
            rhs_opd = self.getOp(rhs)
            p['const_val'] = self.getScalar(lhs)
            p['inversed'] = True
            cmp_op = self.mlir.create_compare_const_op([rhs_opd], output_shape, **p)
        elif self.isScalar(rhs):
            lhs_opd = self.getOp(lhs)
            p['const_val'] = self.getScalar(rhs)
            p['inversed'] = False
            cmp_op = self.mlir.create_compare_const_op([lhs_opd], output_shape, **p)
        else:
            rhs_opd = self.getOp(rhs)
            lhs_opd = self.getOp(lhs)
            cmp_op = self.mlir.create_compare_op([lhs_opd, rhs_opd], output_shape, **p)
        self.addOperand(onnx_node.name, cmp_op)

    def convert_hsigmoid_op(self, onnx_node):
        # hardsigmoid(x; alpha, beta) := min(max(alpha*x + beta, 0), 1)
        assert (onnx_node.op_type == "HardSigmoid")
        assert (len(onnx_node.inputs) == 1)
        operand = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        alpha = onnx_node.attrs.get("alpha", 1. / 6)
        beta = onnx_node.attrs.get("beta", 0.5)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'alpha': alpha,
            'beta': beta
        }
        new_op = self.mlir.create_hsigmoid_op([operand], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_hswish_op(self, onnx_node):
        # hardswish(x) := x * hardsigmoid(x; 1/6, 0.5)
        assert (onnx_node.op_type == "HardSwish")
        assert (len(onnx_node.inputs) == 1)
        operand = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        new_op = self.mlir.create_hswish_op([operand], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_gelu_op(self, onnx_node):
        # 0.5 * val * (1.0 + std::erf(val / std::sqrt(2.0)));
        assert (onnx_node.op_type == "GELU")
        assert (len(onnx_node.inputs) == 1)
        operand = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        new_op = self.mlir.create_gelu_op([operand], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_qlinear_op(self, onnx_node):
        assert (onnx_node.op_type == "QuantizeLinear")
        assert (len(onnx_node.inputs) == 3)
        operand = self.getOperand(onnx_node.inputs[0])
        y_scale = self.getWeight(onnx_node.inputs[1]).tolist()
        y_zero_point = self.getWeight(onnx_node.inputs[2]).tolist()
        output_shape = self.getShape(onnx_node.name)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'y_scale': y_scale,
            'y_zero_point': y_zero_point
        }
        if hasattr(onnx_node, 'attrs'):
            try:
                p['axis'] = onnx_node.attrs['axis']
            except:
                pass
        new_op = self.mlir.create_qlinear_op([operand], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_deqlinear_op(self, onnx_node):
        assert (onnx_node.op_type == "DequantizeLinear")
        assert (len(onnx_node.inputs) == 3)
        try:
            operand = self.getOperand(onnx_node.inputs[0])
        except:
            operand = self.getWeightOp(onnx_node.inputs[0])
        x_scale = self.getWeight(onnx_node.inputs[1])
        x_zero_point = self.getWeight(onnx_node.inputs[2])
        output_shape = self.getShape(onnx_node.name)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'x_scale': x_scale,
            'x_zero_point': x_zero_point
        }
        if hasattr(onnx_node, 'attrs'):
            try:
                p['axis'] = onnx_node.attrs['axis']
            except:
                pass
        new_op = self.mlir.create_deqlinear_op([operand], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_layer_norm_op(self, onnx_node):
        assert (onnx_node.op_type == "LayerNormalization")
        assert (len(onnx_node.inputs) in (2, 3))
        input_shape = self.getShape(onnx_node.inputs[0])
        num_dims = len(input_shape)
        axis = onnx_node.attrs.get("axis", -1)
        if axis < 0:
            axis += num_dims
        eps = onnx_node.attrs.get("epsilon", 1e-05)
        # stash_type is not important
        p = {
            "name":
            [onnx_node.name + '_LayerNorm', onnx_node.name + '_Mean', onnx_node.name + '_Rstd'],
            "axis": axis,
            "eps": eps
        }
        wb_shape = [1 if i < axis else input_shape[i] for i in range(num_dims)]
        input_opd = self.getOperand(onnx_node.inputs[0])
        scale_opd = self.mlir.none_op
        bias_opd = self.mlir.none_op
        if len(onnx_node.inputs) > 1:
            if not self.isScalar_(onnx_node.inputs[1], 1):
                scale_opd = self.getWeightOp(onnx_node.inputs[1], wb_shape)
        if len(onnx_node.inputs) > 2:
            if not self.isScalar_(onnx_node.inputs[2], 0):
                bias_opd = self.getWeightOp(onnx_node.inputs[2], wb_shape)
        out_shapes = [[], [], []]
        out_needs = [False, False, False]
        for idx, out in enumerate(onnx_node.outputs):
            need = len(out) > 0 and self.check_need(out)
            if need:
                p['name'][idx] = "{}_{}".format(out, onnx_node.op_type)
                out_needs[idx] = True
                out_shapes[idx] = self.getShape(out)
        out_ops = self.mlir.create_layer_norm_op([input_opd, scale_opd, bias_opd], out_shapes, **p)
        for idx, need in enumerate(out_needs):
            if not need: continue
            self.addOperand(onnx_node.outputs[idx], out_ops[idx])

    def convert_pixel_norm_op(self, onnx_node):
        assert (onnx_node.op_type == "PixelNormalization")
        assert (len(onnx_node.inputs) in (1, 2, 3))
        input_shape = self.getShape(onnx_node.inputs[0])
        num_dims = len(input_shape)
        assert (num_dims > 1)
        eps = onnx_node.attrs.get("epsilon", 1e-05)
        p = {
            "name": "{}_{}".format(onnx_node.name, onnx_node.op_type),
            "eps": eps
        }
        wb_shape = [1] * num_dims
        wb_shape[1] = input_shape[1]
        input_opd = self.getOperand(onnx_node.inputs[0])
        scale_opd = self.mlir.none_op
        bias_opd = self.mlir.none_op
        if len(onnx_node.inputs) > 1:
            if not self.isScalar_(onnx_node.inputs[1], 1):
                scale_opd = self.getWeightOp(onnx_node.inputs[1], wb_shape)
        if len(onnx_node.inputs) > 2:
            if not self.isScalar_(onnx_node.inputs[2], 0):
                bias_opd = self.getWeightOp(onnx_node.inputs[2], wb_shape)
        output_shape = self.getShape(onnx_node.name)
        new_op = self.mlir.create_pixel_norm_op([input_opd, scale_opd, bias_opd], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_scatternd_op(self, onnx_node):
        assert (onnx_node.op_type == "ScatterND")
        assert(len(onnx_node.inputs) == 3)
        input_data = self.getOp(onnx_node.inputs[0])
        indices =self.getOp(onnx_node.inputs[1])
        updates = self.getOp(onnx_node.inputs[2])
        output_shape = self.getShape(onnx_node.name)
        reduction = onnx_node.attrs.get("reduction", None)
        p = {
            "name": "{}_{}".format(onnx_node.name, onnx_node.op_type),
            "reduction": reduction
        }
        scatternd_op = self.mlir.create_scatternd_op([input_data, indices, updates], output_shape, **p)
        self.addOperand(onnx_node.name, scatternd_op)
