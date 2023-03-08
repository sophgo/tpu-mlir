# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .MLIRImporter import MLIRImporter
from .BaseConverter import BaseConverter
from numbers import Number
from typing import Union, Iterable, List
from mlir.ir import *
import numpy as np
import random
import copy
import torch
import torch.nn as nn


def _get_constant(node):
    """Retrieve a constant associated with this prim::Constant node"""
    attribute_names = node.attributeNames()
    num_attributes = len(attribute_names)
    name = node.output().debugName()
    is_tensor = False
    type = node.output().type().kind()

    if type == "NoneType":
        return name, None, False
    elif num_attributes == 1:
        attr_name = attribute_names[0]
        if type == "IntType":
            value = node.i(attr_name)
        elif type == "BoolType":
            value = bool(node.i(attr_name))
        elif type in ["FloatType", "LongType"]:
            value = node.f(attr_name)
        elif type in ["DeviceObjType", "StringType"]:
            value = node.s(attr_name)
        elif type in ["TensorType", "CompleteTensorType"]:
            is_tensor = True
            tensor = node.t(attr_name)
            if tensor.is_cuda:
                tensor = tensor.cpu()
            value = tensor.numpy()
        else:
            raise NotImplementedError("Unsupported type: %s" % type)
    else:
        assert num_attributes == 0
        return None
    return name, value, is_tensor


def _data_expand(data, length):
    if isinstance(data, int):
        return [data for i in range(length)]
    else:
        return data


def _compute_pad(stride, dilation, input_size, filter, padding):
    stride = np.array(stride)
    dilation = np.array(dilation)
    input_size = np.array(input_size)
    filter = np.array(filter)
    effective_filter_size = (filter - 1) * dilation + 1
    if padding == "same":
        output_size = (input_size + stride - 1) // stride
    elif padding == "valid":
        output_size = (input_size + stride - effective_filter_size) // stride
    padding_needed = np.int64((output_size - 1) * stride + effective_filter_size - input_size)
    padding_needed = padding_needed.clip(min=0)

    padding_before = padding_needed // 2
    padding_after = padding_needed - padding_before
    pad = [i for i in padding_before] + [i for i in padding_after]
    return pad


class BaseNode():

    def __init__(self, info):
        self.name = str(info["name"])
        self.op_type = str(info["op_type"])
        self.inputs = list(info["inputs"])
        self.outputs = list(info["outputs"])


class TorchNode(BaseNode):

    def __init__(self, node):
        info = dict()
        info["name"] = node.output().debugName()
        info["op_type"] = node.kind()
        info["inputs"] = [inp.debugName() for inp in node.inputs()]
        info["outputs"] = [outp.debugName() for outp in node.outputs()]
        super().__init__(info)
        self.node_proto = node


class TorchConverter(BaseConverter):
    TypeMap = {
        "float64": "F64",
        "float32": "F32",
        "float16": "F16",
        "int8": "INT8",
        "int16": "INT16",
        "int32": "INT32",
        "int64": "INT64",
        "uint8": "UINT8",
        "uint16": "UINT16",
        "uint32": "UINT32",
        "uint64": "UINT64",
        "bool": "BOOL",
    }

    def __init__(self,
                 model_name: str,
                 torch_file,
                 input_shapes: list,
                 output_names: list,
                 preprocess_args=None):
        super().__init__()
        self.model_name = model_name
        self.weight_file = "{}_top_origin_weight.npz".format(model_name)
        self.model = None
        self.mlir = None
        self.node_name_mapping = {}  # used in torch opt
        self.const_val = {}

        self.load_torch_model(torch_file, input_shapes, output_names)
        self.init_MLIRImporter()
        self.preprocess_args = preprocess_args
        self.converted_nodes = list()
        self.op_factory = {
            #############################
            # Torch Convert, Alphabetically
            #############################
            "aten::add": lambda node: self.convert_add_op(node),
            "aten::cat": lambda node: self.convert_concat_op(node),
            "aten::_convolution": lambda node: self.convert_conv_op(node),
            "aten::_convolution_mode": lambda node: self.convert_conv_mode_op(node),
            "aten::div": lambda node: self.convert_div_op(node),
            "aten::elu": lambda node: self.convert_elu_op(node),
            "aten::layer_norm": lambda node: self.convert_layer_norm_op(node),
            "aten::mul": lambda node: self.convert_mul_op(node),
            "aten::pad": lambda node: self.convert_pad_op(node),
            "aten::prelu": lambda node: self.convert_prelu_op(node),
            "aten::permute": lambda node: self.convert_permute_op(node),
            "aten::relu": lambda node: self.convert_relu_op(node),
            "aten::sub": lambda node: self.convert_sub_op(node),
            "aten::t": lambda node: self.convert_transpose_op(node),
            "aten::transpose": lambda node: self.convert_transpose_op(node),
        }
        self.check_op_names()

    def __del__(self):
        if self.mlir != None:
            del self.mlir
            self.mlir = None

    def get_all_op_names(self):
        """Return all operator names in the input graph"""
        self.nodes = list(self.graph.nodes())
        prim_blocks = ["prim::If", "prim::Loop"]
        for prim in prim_blocks:
            prim_nodes = self.graph.findAllNodes(prim, recurse=True)
            for prim_node in prim_nodes:
                for block in prim_node.blocks():
                    self.nodes += block.nodes()
        return set(node.kind() for node in self.nodes)

    def check_op_names(self):
        op_names = self.get_all_op_names()
        known_ops = [
            "prim::CallMethod",
            "prim::Constant",
            "prim::GetAttr",
            "prim::If",
            "prim::ListConstruct",
            "prim::ListUnpack",
            "prim::Loop",
            "prim::RaiseException",
            "prim::TupleConstruct",
            "prim::TupleUnpack",
        ]
        known_ops += list(self.op_factory.keys())

        unknown_ops = []
        for op_name in op_names:
            if op_name not in known_ops:
                if not (op_name.endswith("_") and op_name[:-1] in known_ops):
                    unknown_ops.append(op_name)
        if len(unknown_ops) != 0:
            raise RuntimeError(
                "The following operators are not implemented: {}".format(unknown_ops))

    def load_torch_model(self, torch_file, input_shapes: list, output_names: list):
        if isinstance(torch_file, str):
            self.model = torch.jit.load(torch_file, map_location=torch.device('cpu'))
        else:
            self.model = torch_file
        self.model.eval()
        self.graph = self.model.inlined_graph
        self.state_dict = self.model.state_dict()
        is_module = isinstance(self.model, torch.jit.ScriptModule)
        inputs = list(self.graph.inputs())
        inputs = inputs[1:] if is_module else inputs
        self.input_names = []
        for idx, inp in enumerate(inputs):
            self.input_names.append(inp.debugName())
            self.addShape(inp.debugName(), input_shapes[idx])

        self.output_names = []
        if output_names:
            self.output_names = output_names
        else:
            for outp in self.graph.outputs():
                self.output_names.append(outp.debugName())
        # weight
        # for name, data in self.model.state_dict().items():
        #     self.addWeight(name, data.numpy().astype(np.float32))

        self.weight_names = []
        self.num_input = len(self.input_names)
        self.num_output = len(self.output_names)
        self.input_shapes = input_shapes
        self.input_types = [self.TypeMap["float32"] for i in range(self.num_input)]
        self.output_shapes = [None] * self.num_output

    def init_MLIRImporter(self):
        input_shapes = list()
        for _name in self.input_names:
            input_shapes.append(self.getShape(_name))
        # init importer
        self.mlir = MLIRImporter(input_shapes, self.output_shapes, self.model_name,
                                 self.input_types)
        self.weight_file = self.mlir.weight_file

    def get_list(self, node):
        data = []
        is_const = True
        for input in node.inputs():
            if input.debugName() in self.const_val.keys():
                data.append(self.const_val[input.debugName()])
            else:
                data.append(input.debugName())
                is_const = False
        return node.output().debugName(), data, is_const

    def collect_nodes(self, node):
        if node.kind() == 'prim::Constant':
            name, data, is_tensor = _get_constant(node)
            if not is_tensor:
                self.const_val[name] = data
            else:
                self.addWeight(name, data)
            return
        if node.kind() == 'prim::ListConstruct':
            name, data, is_const = self.get_list(node)
            if is_const:
                self.const_val[name] = data
            else:
                self.tensor_list[name] = data
            return
        if node.kind() == 'prim::GetAttr':
            if node.output().type().kind() != 'TensorType':
                return
            folder = node.input().node().s('name')
            name = node.s('name')
            dict_name = "{}.{}".format(folder, name)
            data = self.state_dict[dict_name].numpy().astype(np.float32)
            weight_name = node.output().debugName()
            self.addWeight(weight_name, data)
            return
        if node.kind().startswith("aten::"):
            nd = TorchNode(node)
            self.converted_nodes.append(nd)
            return
        print(node)
        raise RuntimeError("Not Implemented")

    def generate_mlir(self, mlir_file: str):
        """convert all to mlir"""
        # add input op
        for idx, _name in enumerate(self.input_names):
            input_op = self.mlir.create_input_op(_name, idx, **{})
            self.addOperand(_name, input_op)

        def NoneAndRaise(node):
            raise RuntimeError("{} Op not support now".format(node.op_type))

        self.tensor_list = {}
        self.converted_nodes.clear()
        for node in self.graph.nodes():
            self.collect_nodes(node)
        # checkout all type is supported
        unsupported = set()
        for n in self.converted_nodes:
            if n.op_type not in self.op_factory:
                unsupported.add(n.op_type)
        if unsupported:
            raise RuntimeError("Op not support:{}".format(unsupported))

        for n in self.converted_nodes:
            self.op_factory.get(n.op_type, lambda x: NoneAndRaise(x))(n)
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

    def convert_base_conv_op(self, torch_node: TorchNode, mode=False):
        op = self.getOp(torch_node.inputs[0])
        strides = _data_expand(self.const_val[torch_node.inputs[3]], 2)
        pads = self.const_val[torch_node.inputs[4]]
        dilations = _data_expand(self.const_val[torch_node.inputs[5]], 2)
        group = self.const_val[torch_node.inputs[6 if mode else 8]]
        kernel_shape = self.getShape(torch_node.inputs[1])
        kernel_shape = kernel_shape[2:]
        if mode == True:
            input_size = self.getShape(torch_node.inputs[0])[2:]
            pads = _compute_pad(strides, dilations, input_size, kernel_shape, pads)
        else:
            transposed = self.const_val[torch_node.inputs[6]]
            output_padding = self.const_val[torch_node.inputs[7]]
            if isinstance(pads, int):
                pads = [pads for i in range(4)]
            elif len(pads) == 2:
                pads = [pads[0], pads[0], pads[1], pads[1]]

        operands = list()
        operands.append(op)
        filter_op = self.getOp(torch_node.inputs[1])
        operands.append(filter_op)
        if torch_node.inputs[2] not in self.const_val.keys() or self.const_val[
                torch_node.inputs[2]] is not None:
            bias_op = self.getOp(torch_node.inputs[2])
        else:
            bias_op = self.mlir.none_op
        operands.append(bias_op)
        p = {
            'name': torch_node.name,
            'kernel_shape': kernel_shape,
            'strides': strides,
            'dilations': dilations,
            'pads': pads,
            'group': group,
            'do_relu': False,
            'ins': [],
        }
        new_op = self.mlir.create_conv_op(operands, None, **p)
        self.addOperand(torch_node.name, new_op)

    def convert_conv_op(self, torch_node: TorchNode):
        self.convert_base_conv_op(torch_node)

    def convert_conv_mode_op(self, torch_node: TorchNode):
        self.convert_base_conv_op(torch_node, True)

    def _mul_scale(self, in_name, scale):
        in_op = self.getOp(in_name)
        op_name = in_name + "_ml_mulscale"
        p = {
            'name': op_name,
            'const_val': scale,
        }
        new_op = self.mlir.create_mul_const_op([in_op], None, **p)
        self.addOperand(op_name, new_op)
        return new_op

    def convert_add_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        scale = self.const_val[torch_node.inputs[2]]
        op1 = self.getOp(torch_node.inputs[1]) if scale == 1 else \
              self._mul_scale(torch_node.inputs[1], scale)
        p = {
            'name': torch_node.name,
            'do_relu': False,
        }
        new_op = self.mlir.create_add_op([op0, op1], None, **p)
        self.addOperand(torch_node.name, new_op)

    def convert_sub_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        scale = self.const_val[torch_node.inputs[2]]
        op1 = self.getOp(torch_node.inputs[1]) if scale == 1 else \
              self._mul_scale(torch_node.inputs[1], scale)
        p = {
            'name': torch_node.name,
            'do_relu': False,
        }
        new_op = self.mlir.create_sub_op([op0, op1], None, **p)
        self.addOperand(torch_node.name, new_op)

    def convert_mul_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        op1 = self.getOp(torch_node.inputs[1])
        p = {
            'name': torch_node.name,
            'do_relu': False,
        }
        new_op = self.mlir.create_mul_op([op0, op1], None, **p)
        self.addOperand(torch_node.name, new_op)

    def convert_div_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        op1 = self.getOp(torch_node.inputs[1])
        p = {
            'name': torch_node.name,
            'do_relu': False,
        }
        new_op = self.mlir.create_div_op([op0, op1], None, **p)
        self.addOperand(torch_node.name, new_op)

    def convert_prelu_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        op1 = self.getOp(torch_node.inputs[1])
        new_op = self.mlir.create_prelu_op([op0, op1], None, **{'name': torch_node.name})
        self.addOperand(torch_node.name, new_op)

    def convert_permute_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        order = self.const_val[torch_node.inputs[1]]
        p = {
            'name': torch_node.name,
            'order': order,
        }
        new_op = self.mlir.create_permute_op([op], None, **p)
        self.addOperand(torch_node.name, new_op)

    def convert_transpose_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        no_dims = len(torch_node.inputs) == 1
        dim0 = self.const_val[torch_node.inputs[1]] if not no_dims else 0
        dim1 = self.const_val[torch_node.inputs[2]] if not no_dims else 1
        p = {
            'name': torch_node.name,
            'dim0': dim0,
            'dim1': dim1,
        }
        new_op = self.mlir.create_transpose_op([op], None, **p)
        self.addOperand(torch_node.name, new_op)

    def convert_layer_norm_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        scale_opd = self.mlir.none_op
        bias_opd = self.mlir.none_op
        normalized_shape = self.const_val[torch_node.inputs[1]]
        if not self.isScalar_(torch_node.inputs[2], 1):
            scale_opd = self.getWeightOp(torch_node.inputs[2])
        if not self.isScalar_(torch_node.inputs[3], 0):
            bias_opd = self.getWeightOp(torch_node.inputs[3])
        eps = self.const_val[torch_node.inputs[4]]
        p = {
            'name': [torch_node.name, torch_node.name + "_Mean", torch_node.name + "_Rstd"],
            'normalized_shape': normalized_shape,
            'axis': -len(normalized_shape),
            'eps': eps,
        }
        outs = self.mlir.create_layer_norm_op([op0, scale_opd, bias_opd], [None, [], []], **p)
        self.addOperand(torch_node.name, outs[0])

    def convert_concat_op(self, torch_node: TorchNode):
        operands = list()
        for name in self.tensor_list[torch_node.inputs[0]]:
            op = self.getOp(name)
            operands.append(op)
        axis = self.const_val[torch_node.inputs[1]]
        p = {
            'name': torch_node.name,
            'axis': axis,
        }
        new_op = self.mlir.create_concat_op(operands, None, **p)
        self.addOperand(torch_node.name, new_op)

    def convert_relu_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = self.mlir.create_relu_op([op], None, **{'name': torch_node.name})
        self.addOperand(torch_node.name, new_op)

    def convert_elu_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        alpha = self.const_val[torch_node.inputs[1]]
        p = {
            'name': torch_node.name,
            'alpha': alpha,
        }
        new_op = self.mlir.create_elu_op([op], None, **p)
        self.addOperand(torch_node.name, new_op)

    def convert_pad_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        pads_origin = list(self.const_val[torch_node.inputs[1]])
        input_dim = len(self.getShape(torch_node.inputs[0]))
        pads = [0] * input_dim * 2
        if (len(pads_origin) >= 2):
            # w pad
            pads[input_dim - 1] = pads_origin[0]
            pads[-1] = pads_origin[1]
        if (len(pads_origin) > 2):
            # h pad
            pads[input_dim - 2] = pads_origin[2]
            pads[-2] = pads_origin[3]

        pad_mode = {"constant": 0, "reflect": 1, "replicate": 3}
        mode = pad_mode[self.const_val[torch_node.inputs[2]]]
        val = 0. if mode != 0 else self.const_val[torch_node.inputs[3]]
        p = {
            'name': torch_node.name,
            'paddings': pads,
            'mode': mode,
            'val': val,
        }
        new_op = self.mlir.create_pad_op([op], None, **p)
        self.addOperand(torch_node.name, new_op)
