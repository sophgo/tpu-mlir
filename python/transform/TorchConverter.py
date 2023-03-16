# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .MLIRImporter import MLIRImporter, Platform
from .BaseConverter import BaseConverter
from mlir.ir import *
import numpy as np
import torch


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
        info["op_type"] = node.kind()
        info["inputs"] = [inp.debugName() for inp in node.inputs()]
        info["outputs"] = [outp.debugName() for outp in node.outputs()]
        info["name"] = info["outputs"][0]
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
        # yapf: disable
        self.op_factory = {
            #############################
            # Torch Convert, Alphabetically
            #############################
            "aten::abs": lambda node: self.convert_abs_op(node),
            "aten::add": lambda node: self.convert_add_op(node),
            "aten::avg_pool1d": lambda node: self.convert_avgpool_op(node),
            "aten::avg_pool2d": lambda node: self.convert_avgpool_op(node),
            "aten::avg_pool3d": lambda node: self.convert_avgpool_op(node),
            "aten::cat": lambda node: self.convert_concat_op(node),
            "aten::_convolution": lambda node: self.convert_conv_op(node),
            "aten::_convolution_mode": lambda node: self.convert_conv_mode_op(node),
            "aten::constant_pad_nd": lambda node: self.convert_pad_op(node, mode='constant'),
            "aten::contiguous": lambda node: self.convert_skip_op(node),
            "aten::div": lambda node: self.convert_div_op(node),
            "aten::dropout": lambda node: self.convert_skip_op(node),
            "aten::elu": lambda node: self.convert_elu_op(node),
            "aten::eq": lambda node: self.convert_compare_op(node, "Equal"),
            "aten::ge": lambda node: self.convert_compare_op(node, "GreaterOrEqual"),
            "aten::gelu": lambda node: self.convert_gelu_op(node),
            "aten::gru": lambda node: self.convert_gru_op(node),
            "aten::gt": lambda node: self.convert_compare_op(node, "Greater"),
            "aten::hardsigmoid": lambda node: self.convert_hardsigmoid(node),
            "aten::hardswish": lambda node: self.convert_hardswish(node),
            "aten::hardtanh": lambda node: self.convert_hardtanh(node),  # relu6 is treated as hardtanh
            "aten::index_select": lambda node: self.convert_index_select_op(node),
            "aten::layer_norm": lambda node: self.convert_layer_norm_op(node),
            "aten::le": lambda node: self.convert_compare_op(node, "LessOrEqual"),
            "aten::leaky_relu": lambda node: self.convert_leaky_relu_op(node),
            "aten::log_sigmoid": lambda node: self.convert_sigmoid_op(node, log=True),
            "aten::log_softmax": lambda node: self.convert_softmax_op(node, log=True),
            "aten::lstm": lambda node: self.convert_lstm_op(node),
            "aten::lt": lambda node: self.convert_compare_op(node, "Less"),
            "aten::max_pool1d": lambda node: self.convert_maxpool_op(node),
            "aten::max_pool2d": lambda node: self.convert_maxpool_op(node),
            "aten::max_pool3d": lambda node: self.convert_maxpool_op(node),
            "aten::mish": lambda node: self.convert_mish_op(node),
            "aten::mul": lambda node: self.convert_mul_op(node),
            "aten::neg": lambda node: self.convert_neg_op(node),
            "aten::pad": lambda node: self.convert_pad_op(node, mode='unknown'),
            "aten::prelu": lambda node: self.convert_prelu_op(node),
            "aten::permute": lambda node: self.convert_permute_op(node),
            "aten::reflection_pad1d": lambda node: self.convert_pad_op(node, mode='reflect'),
            "aten::reflection_pad2d": lambda node: self.convert_pad_op(node, mode='reflect'),
            "aten::relu": lambda node: self.convert_relu_op(node),
            "aten::replication_pad1d": lambda node: self.convert_pad_op(node, mode='replicate'),
            "aten::replication_pad2d": lambda node: self.convert_pad_op(node, mode='replicate'),
            "aten::reshape": lambda node: self.convert_reshape_op(node),
            "aten::scatter": lambda node: self.convert_scatter_op(node),
            "aten::sigmoid": lambda node: self.convert_sigmoid_op(node),
            "aten::silu": lambda node: self.convert_silu_op(node),
            "aten::softmax": lambda node: self.convert_softmax_op(node),
            "aten::softplus": lambda node: self.convert_softplus_op(node),
            "aten::sub": lambda node: self.convert_sub_op(node),
            "aten::size": lambda node: self.convert_size_op(node),
            "aten::t": lambda node: self.convert_transpose_op(node),
            "aten::tanh": lambda node: self.convert_tanh_op(node),
            "aten::tile": lambda node: self.convert_tile_op(node),
            "aten::transpose": lambda node: self.convert_transpose_op(node),
            "aten::to": lambda node: self.convert_to_op(node),
            "aten::view": lambda node: self.convert_view_op(node),
            "aten::where": lambda node: self.convert_where_op(node),
            "aten::zeros": lambda node: self.convert_zeros_op(node),
            ###### prim #####
            "prim::Constant": lambda node: self.convert_constant(node),
            "prim::GetAttr": lambda node: self.convert_get_attr(node),
            "prim::ListConstruct": lambda node: self.convert_list(node),
            "prim::TupleConstruct": lambda node: self.convert_tuple(node),
            "prim::TupleUnpack": lambda node: self.convert_tuple_unpack(node),
        }
        # yapf: enable
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
        known_ops = list(self.op_factory.keys())

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
        self.output_shapes = [[]] * self.num_output

    def init_MLIRImporter(self):
        input_shapes = list()
        for _name in self.input_names:
            input_shapes.append(self.getShape(_name))
        # init importer
        self.mlir = MLIRImporter(input_shapes, self.output_shapes, self.model_name, Platform.TORCH,
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
            self.converted_nodes.append(TorchNode(node))
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
        transposed = False
        if mode == True:
            input_size = self.getShape(torch_node.inputs[0])[2:]
            pads = _compute_pad(strides, dilations, input_size, kernel_shape, pads)
        else:
            transposed = self.const_val[torch_node.inputs[6]]
            pads = pads + pads  # the pad of torch is symmetric
        operands = list()
        operands.append(op)
        if transposed:
            # the dim of weight in pytorch is [ic, oc, ... ]
            name = torch_node.inputs[1]
            data = self.getWeight(name)
            shape = data.shape
            data = data.reshape(group, shape[0] // group, *shape[1:])
            data = data.swapaxes(1, 2).reshape(shape[1], shape[0], *shape[2:])
            new_name = name + "_transposed"
            self.addWeight(new_name, data)
            filter_op = self.getOp(new_name)
        else:
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
        }
        if transposed:
            output_padding = self.const_val[torch_node.inputs[7]]
            p["output_padding"] = output_padding
            new_op = self.mlir.create_conv_transpose_op(operands, [], **p)
            return self.addOperand(torch_node.name, new_op)
        new_op = self.mlir.create_conv_op(operands, [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_conv_op(self, torch_node: TorchNode):
        # convolution or transposed_convolution
        self.convert_base_conv_op(torch_node)

    def convert_conv_mode_op(self, torch_node: TorchNode):
        # only for convolution
        self.convert_base_conv_op(torch_node, True)

    def convert_avgpool_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        kernel_shape = self.const_val[torch_node.inputs[1]]
        strides = self.const_val[torch_node.inputs[2]]
        pads = self.const_val[torch_node.inputs[3]]
        ceil_mode = self.const_val[torch_node.inputs[4]]
        count_include_pad = self.const_val[torch_node.inputs[5]]
        assert ceil_mode == False
        if len(torch_node.inputs) == 7:
            # does not supports divisor_override
            assert self.const_val[torch_node.inputs[6]] is None
        pads = pads + pads  # the pad of torch is symmetric
        p = {
            "name": torch_node.name,
            "kernel_shape": kernel_shape,
            "strides": strides,
            "pads": pads,
            "do_relu": False,
            "count_include_pad": count_include_pad,
        }
        new_op = self.mlir.create_avgpool_op([op], [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_maxpool_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        kernel_shape = self.const_val[torch_node.inputs[1]]
        strides = self.const_val[torch_node.inputs[2]]
        pads = self.const_val[torch_node.inputs[3]]
        dilation = self.const_val[torch_node.inputs[4]]
        ceil_mode = self.const_val[torch_node.inputs[5]]
        assert (np.array(dilation) == 1).all()
        pads = pads + pads  # the pad of torch is symmetric
        p = {
            "name": torch_node.name,
            "kernel_shape": kernel_shape,
            "strides": strides,
            "pads": pads,
            "do_relu": False,
            "ceil_mode": ceil_mode,
        }
        new_op = self.mlir.create_maxpool_op([op], [], **p)
        self.addOperand(torch_node.name, new_op)

    def _mul_scale(self, in_name, scale):
        in_op = self.getOp(in_name)
        op_name = in_name + "_ml_mulscale"
        p = {
            'name': op_name,
            'const_val': scale,
        }
        new_op = self.mlir.create_mul_const_op([in_op], [], **p)
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
        new_op = self.mlir.create_add_op([op0, op1], [], **p)
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
        new_op = self.mlir.create_sub_op([op0, op1], [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_size_op(self, torch_node: TorchNode):
        # do nothing
        pass

    def convert_zeros_op(self, torch_node: TorchNode):
        p = {
            'name': torch_node.name,
        }
        new_op = self.mlir.create_zeros_op([], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_constant(self, torch_node: TorchNode):
        name, data, is_tensor = _get_constant(torch_node.node_proto)
        if not is_tensor:
            self.const_val[name] = data
        else:
            self.addWeight(name, data)

    def convert_list(self, torch_node: TorchNode):
        name, data, is_const = self.get_list(torch_node.node_proto)
        if is_const:
            self.const_val[name] = data
        else:
            self.tensor_list[name] = data

    def convert_get_attr(self, torch_node: TorchNode):
        node = torch_node.node_proto
        if node.output().type().kind() != 'TensorType':
            return
        folder = node.input().node().s('name')
        name = node.s('name')
        dict_name = "{}.{}".format(folder, name)
        data = self.state_dict[dict_name].numpy().astype(np.float32)
        weight_name = node.output().debugName()
        self.addWeight(weight_name, data)

    def convert_mul_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        op1 = self.getOp(torch_node.inputs[1])
        p = {
            'name': torch_node.name,
            'do_relu': False,
        }
        new_op = self.mlir.create_mul_op([op0, op1], [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_div_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        op1 = self.getOp(torch_node.inputs[1])
        p = {
            'name': torch_node.name,
            'do_relu': False,
        }
        new_op = self.mlir.create_div_op([op0, op1], [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_skip_op(self, torch_node: TorchNode):
        # warning: in_op.output name shoud change to torch_node name
        in_op = self.getOp(torch_node.inputs[0])
        self.addOperand(torch_node.name, in_op)

    def convert_to_op(self, torch_node: TorchNode):
        in_op = self.getOp(torch_node.inputs[0])
        out_type = self.const_val[torch_node.inputs[1]]
        assert(out_type in [5, 6, 7, 15]) # float
        self.addOperand(torch_node.name, in_op)

    def convert_compare_op(self, torch_node: TorchNode, mode):
        assert mode in ("Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual")
        op0 = self.getOp(torch_node.inputs[0])
        if torch_node.inputs[1] in self.const_val:
            const_val = self.const_val[torch_node.inputs[1]]
            p = {
                "name": torch_node.name,
                "mode": mode,
                "const_val": const_val,
                "inversed": False,
            }
            new_op = self.mlir.create_compare_const_op([op0], [], **p)
        else:
            op1 = self.getOp(torch_node.inputs[1])
            p = {"name": torch_node.name, "mode": mode}
            new_op = self.mlir.create_compare_op([op0, op1], [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_prelu_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        op1 = self.getOp(torch_node.inputs[1])
        new_op = self.mlir.create_prelu_op([op0, op1], [], **{'name': torch_node.name})
        self.addOperand(torch_node.name, new_op)

    def convert_permute_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        order = self.const_val[torch_node.inputs[1]]
        p = {
            'name': torch_node.name,
            'order': order,
        }
        new_op = self.mlir.create_permute_op([op], [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_tile_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        repeats = self.const_val[torch_node.inputs[1]]
        if np.prod(repeats) == 1:
            self.addOperand(torch_node.name, op)
            return

        attr = {
            'name': torch_node.name,
            'repeats': repeats,
        }
        new_op = self.mlir.create_tile_ex_op([op], [], **attr)
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
        new_op = self.mlir.create_transpose_op([op], [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_index_select_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        op1 = self.getOp(torch_node.inputs[2])
        axis = self.const_val[torch_node.inputs[1]]
        p = {'name': torch_node.name, 'axis': axis}
        new_op = self.mlir.create_gather_op([op0, op1], [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_reshape_op(self, torch_node: TorchNode):
        in_op = self.getOp(torch_node.inputs[0])
        shape = self.const_val[torch_node.inputs[1]]
        p = {'name': torch_node.name, 'shape': shape}
        new_op = self.mlir.create_reshape_op([in_op], [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_scatter_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        op1 = self.getOp(torch_node.inputs[2])
        op2 = self.getOp(torch_node.inputs[3])
        axis = self.const_val[torch_node.inputs[1]]
        p = {'name': torch_node.name, 'axis': axis, "reduction": None}
        new_op = self.mlir.create_scatter_elements_op([op0, op1, op2], [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_view_op(self, torch_node: TorchNode):
        in_op = self.getOp(torch_node.inputs[0])
        shape = self.const_val[torch_node.inputs[1]]
        p = {'name': torch_node.name, 'shape': shape}
        new_op = self.mlir.create_reshape_op([in_op], [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_where_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        x_is_const = torch_node.inputs[1] in self.const_val.keys()
        y_is_const = torch_node.inputs[2] in self.const_val.keys()
        x_const_val = self.const_val[torch_node.inputs[1]] if x_is_const else 0
        y_const_val = self.const_val[torch_node.inputs[1]] if x_is_const else 0
        p = {
            'name': torch_node.name,
            'x_is_const': x_is_const,
            'y_is_const': y_is_const,
            'x_const_val': x_const_val,
            'y_const_val': y_const_val
        }

        op1 = self.getOp(torch_node.inputs[1]) if not x_is_const else self.mlir.none_op
        op2 = self.getOp(torch_node.inputs[2]) if not y_is_const else self.mlir.none_op
        new_op = self.mlir.create_where_op([op0, op1, op2], [], **p)
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
        outs = self.mlir.create_layer_norm_op([op0, scale_opd, bias_opd], [[], None, None], **p)
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
        new_op = self.mlir.create_concat_op(operands, [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_relu_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = self.mlir.create_relu_op([op], [], **{'name': torch_node.name})
        self.addOperand(torch_node.name, new_op)

    def convert_elu_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        alpha = self.const_val[torch_node.inputs[1]]
        p = {
            'name': torch_node.name,
            'alpha': alpha,
        }
        new_op = self.mlir.create_elu_op([op], [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_pad_op(self, torch_node: TorchNode, mode: str = 'unknown'):
        op = self.getOp(torch_node.inputs[0])
        pads_origin = self.const_val[torch_node.inputs[1]]
        val = 0.0
        pad_modes = {"constant": 0, "reflect": 1, "replicate": 3}
        if (mode == 'reflect' or mode == 'replicate'):
            pad_mode = pad_modes[mode]
        elif (mode == 'constant'):
            pad_mode = pad_modes[mode]
            val = self.const_val[torch_node.inputs[2]]
        else:
            pad_mode = pad_modes[self.const_val[torch_node.inputs[2]]]
            if pad_mode == 0:
                val = self.const_val[torch_node.inputs[3]]

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

        p = {
            'name': torch_node.name,
            'paddings': pads,
            'mode': pad_mode,
            'val': val,
        }
        new_op = self.mlir.create_pad_op([op], [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_hardsigmoid(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        p = {
            'name': torch_node.name,
            'alpha': 1 / 6,
            'beta': 1 / 2,
        }
        new_op = self.mlir.create_hsigmoid_op([op], [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_hardswish(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = self.mlir.create_hswish_op([op], [], **{'name': torch_node.name})
        self.addOperand(torch_node.name, new_op)

    def convert_hardtanh(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        min_val = self.const_val[torch_node.inputs[1]]
        max_val = self.const_val[torch_node.inputs[2]]
        assert (min_val == 0 and max_val == 6 and "Only support relu6 for now")
        p = {
            'name': torch_node.name,
            'relu_limit': max_val,
        }
        new_op = self.mlir.create_relu_op([op], [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_tuple(self, torch_node: TorchNode):
        ops = [self.getOp(n) for n in torch_node.inputs]
        p = {
            'name': torch_node.name,
        }
        new_op = self.mlir.create_tuple_op(ops, [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_tuple_unpack(self, torch_node: TorchNode):
        ops = [self.getOp(n) for n in torch_node.inputs]
        p = {
            'name': torch_node.outputs,
        }
        num = len(torch_node.outputs)
        assert (num == len(torch_node.outputs))
        shape = [[]] * num
        out_ops = self.mlir.create_untuple_op(ops, shape, **p)
        for out, op in zip(torch_node.outputs, out_ops):
            self.addOperand(out, op)

    def convert_gelu_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = self.mlir.create_gelu_op([op], [], **{'name': torch_node.name})
        self.addOperand(torch_node.name, new_op)

    def convert_sigmoid_op(self, torch_node: TorchNode, log: bool = False):
        op = self.getOp(torch_node.inputs[0])
        p = {
            'name': torch_node.name,
            'scale': 1,
            'bias': 0,
        }
        if log: p['log'] = True
        new_op = self.mlir.create_sigmoid_op([op], [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_tanh_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = self.mlir.create_tanh_op([op], [], **{'name': torch_node.name})
        self.addOperand(torch_node.name, new_op)

    def convert_softplus_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = self.mlir.create_softplus_op([op], [], **{'name': torch_node.name})
        self.addOperand(torch_node.name, new_op)

    def convert_leaky_relu_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        alpha = self.const_val[torch_node.inputs[1]]
        p = {
            'name': torch_node.name,
            'alpha': alpha,
        }
        new_op = self.mlir.create_leaky_relu_op([op], [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_silu_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = self.mlir.create_silu_op([op], [], **{'name': torch_node.name})
        self.addOperand(torch_node.name, new_op)

    def convert_softmax_op(self, torch_node: TorchNode, log: bool = False):
        op = self.getOp(torch_node.inputs[0])
        dim = self.const_val[torch_node.inputs[1]]
        p = {
            'name': torch_node.name,
            'axis': dim,
        }
        if log: p['log'] = True
        new_op = self.mlir.create_softmax_op([op], [], **p)
        self.addOperand(torch_node.name, new_op)

    def convert_mish_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = self.mlir.create_mish_op([op], [], **{'name': torch_node.name})
        self.addOperand(torch_node.name, new_op)

    def ifgo2iofg(self, data):
        shape = data.shape
        d = data.reshape(4, -1)
        d = np.concatenate((d[:1, :], d[3:, :], d[1:3, :]), axis=0)
        return d.reshape(*shape)

    def convert_lstm_op(self, torch_node: TorchNode):
        in_op = self.getOp(torch_node.inputs[0])
        h0, c0 = self.tensor_list[torch_node.inputs[1]]
        weights = self.tensor_list[torch_node.inputs[2]]
        has_bias = self.const_val[torch_node.inputs[3]]
        assert (has_bias)  # no bias Not Implemented
        num_layers = self.const_val[torch_node.inputs[4]]
        assert (num_layers == 1)  # Not Implemented
        bidirectional = self.const_val[torch_node.inputs[7]]
        batch_first = self.const_val[torch_node.inputs[8]]
        assert (batch_first == False)  # Not Implemented
        h0_op = self.getOp(h0)
        c0_op = self.getOp(c0)
        if bidirectional:
            assert (len(weights) == 8)
        else:
            assert (len(weights) == 4)
        datas = []
        for w in weights:
            d = self.getWeight(w)
            d = self.ifgo2iofg(d)
            datas.append(d)
        # filter
        filter = datas[0]
        if bidirectional:
            filter = np.concatenate((datas[0], datas[4]), axis=0)
        filter_name = torch_node.name + "_filter"
        self.addWeight(filter_name, filter)
        filter_op = self.getWeightOp(filter_name)
        # recurrence
        recurrence = datas[1]
        rshape = recurrence.shape
        if bidirectional:
            recurrence = np.concatenate((datas[1], datas[5]), axis=0)
        recurrence_name = torch_node.name + "_recurrence"
        self.addWeight(recurrence_name, recurrence)
        recurrence_op = self.getWeightOp(recurrence_name)
        # bias
        bias = np.concatenate(
            (datas[2], datas[3], datas[6], datas[7]), axis=0) if bidirectional else np.concatenate(
                (datas[2], datas[3]), axis=0)
        bias_name = torch_node.name + "_bias"
        self.addWeight(bias_name, bias)
        bias_op = self.getWeightOp(bias_name)
        p = {
            "name": torch_node.outputs,
            "hidden_size": rshape[-1],
            "bidirectional": bidirectional,
            "batch_first": batch_first,
        }
        operands = [in_op, filter_op, recurrence_op, bias_op, h0_op, c0_op]
        new_op, h_op, c_op = self.mlir.create_lstm_op(operands, [[], [], []], **p)
        self.addOperand(torch_node.outputs[0], new_op)
        self.addOperand(torch_node.outputs[1], h_op)
        self.addOperand(torch_node.outputs[2], c_op)

    def rzh2zrh(self, data):
        shape = data.shape
        d = data.reshape(3, -1)
        d = np.concatenate((d[1:2, :], d[:1, :], d[2:, :]), axis=0)
        return d.reshape(*shape)

    def convert_gru_op(self, torch_node: TorchNode):
        in_op = self.getOp(torch_node.inputs[0])
        h0_op = self.getOp(torch_node.inputs[1])
        weights = self.tensor_list[torch_node.inputs[2]]
        has_bias = self.const_val[torch_node.inputs[3]]
        assert (has_bias)  # no bias Not Implemented
        num_layers = self.const_val[torch_node.inputs[4]]
        assert (num_layers == 1)  # Not Implemented
        bidirectional = self.const_val[torch_node.inputs[7]]
        batch_first = self.const_val[torch_node.inputs[8]]
        assert (batch_first == False)  # Not Implemented
        if bidirectional:
            assert (len(weights) == 8)
        else:
            assert (len(weights) == 4)
        datas = []
        for w in weights:
            d = self.getWeight(w)
            d = self.rzh2zrh(d)
            datas.append(d)
        # filter
        filter = datas[0]
        if bidirectional:
            filter = np.concatenate((datas[0], datas[4]), axis=0)
        filter_name = torch_node.name + "_filter"
        self.addWeight(filter_name, filter)
        filter_op = self.getWeightOp(filter_name)
        # recurrence
        recurrence = datas[1]
        rshape = recurrence.shape
        if bidirectional:
            recurrence = np.concatenate((datas[1], datas[5]), axis=0)
        recurrence_name = torch_node.name + "_recurrence"
        self.addWeight(recurrence_name, recurrence)
        recurrence_op = self.getWeightOp(recurrence_name)
        # bias
        bias = np.concatenate(
            (datas[2], datas[3], datas[6], datas[7]), axis=0) if bidirectional else np.concatenate(
                (datas[2], datas[3]), axis=0)
        bias_name = torch_node.name + "_bias"
        self.addWeight(bias_name, bias)
        bias_op = self.getWeightOp(bias_name)
        p = {
            "name": torch_node.outputs,
            "hidden_size": rshape[-1],
            "bidirectional": bidirectional,
            "batch_first": batch_first,
        }
        operands = [in_op, filter_op, recurrence_op, bias_op, h0_op]
        new_op, h_op = self.mlir.create_gru_op(operands, [[], []], **p)
        self.addOperand(torch_node.outputs[0], new_op)
        self.addOperand(torch_node.outputs[1], h_op)

    def convert_abs_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = self.mlir.create_abs_op([op], [], **{'name': torch_node.name})
        self.addOperand(torch_node.name, new_op)

    def convert_neg_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        p = {
            'name': torch_node.name,
            'const_val': -1,
        }
        new_op = self.mlir.create_mul_const_op([op], [], **p)
        self.addOperand(torch_node.name, new_op)
