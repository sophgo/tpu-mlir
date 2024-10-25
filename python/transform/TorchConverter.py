# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .MLIRImporter import MLIRImporter, Platform
from .BaseConverter import BaseConverter
from .TorchHelper import *
from mlir.ir import *
import mlir.dialects.top as top
import numpy as np
import torchvision
import logging
import copy

logger = logging.getLogger("root")

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
    TorchTypeMap = {
        0: "UINT8",
        1: "INT8",
        2: "INT16",
        3: "INT32",
        4: "INT64",
        6: "F32",
        7: "F64",
        11: "BOOL",
    }

    def __init__(self,
                 model_name: str,
                 torch_file,
                 input_shapes: list,
                 input_types: list,
                 output_names: list,
                 preprocess_args: dict = {},
                 dynamic=False,
                 shape_influencing_input_names: list = []):
        super().__init__()
        self.shape_influencing_input_names = shape_influencing_input_names
        self.dynamic = dynamic
        self.run_mode = "DYNAMIC" if dynamic else "STATIC"
        self.model_name = model_name
        self.weight_file = "{}_top_origin_weight.npz".format(model_name)
        self.model = None
        self.mlir = None
        self.node_name_mapping = {}  # used in torch opt
        self.load_torch_model(torch_file, input_shapes,
                              input_types, output_names)
        self.init_MLIRImporter()
        self.unranked_type = self.mlir.get_tensor_type([])
        self.preprocess_args = {}
        if 'preprocess_list' in preprocess_args:
            if preprocess_args['preprocess_list'] is not None:
                for input_index in preprocess_args['preprocess_list']:
                    assert (0 < input_index <= len(input_shapes)
                            and "Please check --preprocess_list is right input")
            else:
                preprocess_args['preprocess_list'] = [
                    i + 1 for i in range(len(input_shapes))]
        if 'channel_format' in preprocess_args:
            if preprocess_args['channel_format'] != "none":
                self.preprocess_args = preprocess_args
        self.converted_nodes = list()
        self.const_val = dict()
        # yapf: disable
        self.op_factory = {
            #############################
            # Torch Convert, Alphabetically
            #############################
            "aten::abs": lambda node: self.convert_abs_op(node),
            "aten::acos": lambda node: self.convert_math_op(node, "arccos"),
            "aten::adaptive_avg_pool1d": lambda node: self.convert_adaptive_avgpool_op(node, spatial_rank=1),
            "aten::adaptive_avg_pool2d": lambda node: self.convert_adaptive_avgpool_op(node, spatial_rank=2),
            "aten::add": lambda node: self.convert_add_op(node),
            "aten::addmm": lambda node: self.convert_addmm_op(node),
            "aten::arange": lambda node: self.convert_arange_op(node),
            "aten::atan": lambda node: self.convert_arctan_op(node, "arctan"),
            "aten::atanh": lambda node: self.convert_math_op(node, "arctanh"),
            "aten::avg_pool1d": lambda node: self.convert_avgpool_op(node),
            "aten::avg_pool2d": lambda node: self.convert_avgpool_op(node),
            "aten::avg_pool3d": lambda node: self.convert_avgpool_op(node),
            "aten::batch_norm": lambda node: self.convert_batch_norm_op(node),
            "aten::bmm": lambda node: self.convert_matmul_op(node),
            "aten::baddbmm": lambda node: self.convert_baddbmm_op(node),
            "aten::cat": lambda node: self.convert_concat_op(node),
            "aten::ceil": lambda node: self.convert_ceil_op(node),
            "aten::channel_shuffle": lambda node: self.convert_channel_shuffle_op(node),
            "aten::chunk": lambda node: self.convert_chunk_op(node),
            "aten::copy": lambda node: self.convert_skip_op(node),
            "aten::clamp": lambda node: self.convert_clamp_op(node),
            "aten::clone": lambda node: self.convert_clone_op(node),
            "aten::cos": lambda node: self.convert_math_op(node, "cos"),
            "aten::cosh": lambda node: self.convert_math_op(node, "cosh"),
            "aten::_convolution": lambda node: self.convert_conv_op(node),
            "aten::_convolution_mode": lambda node: self.convert_conv_mode_op(node),
            "aten::constant_pad_nd": lambda node: self.convert_pad_op(node, mode='constant'),
            "aten::contiguous": lambda node: self.convert_skip_op(node),
            "aten::cumsum": lambda node: self.convert_cumsum_op(node),
            "aten::detach": lambda node: self.convert_detach_op(node),
            "aten::div": lambda node: self.convert_div_op(node),
            "aten::dot": lambda node: self.convert_dot_op(node),
            "aten::dropout": lambda node: self.convert_skip_op(node),
            "aten::elu": lambda node: self.convert_elu_op(node),
            "aten::embedding": lambda node: self.convert_embedding_op(node),
            "aten::empty": lambda node: self.convert_skip_op(node),
            "aten::exp": lambda node: self.convert_math_op(node, "exp"),
            "aten::expand": lambda node: self.convert_expand_op(node),
            "aten::expand_as": lambda node: self.convert_expand_as_op(node),
            "aten::eq": lambda node: self.convert_compare_op(node, "Equal"),
            "aten::flip": lambda node: self.convert_flip_op(node),
            "aten::floor": lambda node: self.convert_floor_op(node),
            "aten::floor_divide": lambda node: self.convert_floor_divide_op(node),
            "aten::flatten": lambda node: self.convert_flatten_op(node),
            "aten::frobenius_norm": lambda node: self.convert_frobenius_norm_op(node),
            "aten::gather": lambda node: self.convert_gather_op(node),
            "aten::ge": lambda node: self.convert_compare_op(node, "GreaterOrEqual"),
            "aten::gelu": lambda node: self.convert_gelu_op(node),
            "aten::erf": lambda node: self.convert_erf_op(node),
            "aten::grid_sampler": lambda node: self.convert_grid_sampler_op(node),
            "aten::group_norm": lambda node: self.convert_group_norm_op(node),
            "aten::gru": lambda node: self.convert_gru_op(node),
            "aten::gt": lambda node: self.convert_compare_op(node, "Greater"),
            "aten::hardsigmoid": lambda node: self.convert_hardsigmoid(node),
            "aten::hardswish": lambda node: self.convert_hardswish(node),
            # relu6 is treated as hardtanh
            "aten::hardtanh": lambda node: self.convert_hardtanh(node),
            "aten::index_select": lambda node: self.convert_index_select_op(node),
            "aten::instance_norm": lambda node: self.convert_instance_norm_op(node),
            # "aten::is_floating_point": lambda node: self.convert_is_type_op(node, "float"),
            "aten::Int": lambda node: self.convert_skip_op(node),
            "aten::index": lambda node: self.convert_index_op(node),
            "aten::index_put_": lambda node: self.convert_index_put_op(node),
            "aten::index_put": lambda node: self.convert_index_put_op(node),
            "aten::layer_norm": lambda node: self.convert_layer_norm_op(node),
            "aten::le": lambda node: self.convert_compare_op(node, "LessOrEqual"),
            "aten::leaky_relu": lambda node: self.convert_leaky_relu_op(node),
            "aten::linalg_norm": lambda node: self.convert_linalg_norm_op(node),
            "aten::linear": lambda node: self.convert_linear_op(node),
            "aten::log_sigmoid": lambda node: self.convert_sigmoid_op(node, log=True),
            "aten::log_softmax": lambda node: self.convert_softmax_op(node, log=True),
            "aten::log": lambda node: self.convert_log_op(node),
            "aten::log2": lambda node: self.convert_log2_op(node),
            "aten::lstm": lambda node: self.convert_lstm_op(node),
            "aten::lt": lambda node: self.convert_compare_op(node, "Less"),
            "aten::masked_fill": lambda node: self.convert_masked_fill(node),
            "aten::matmul": lambda node: self.convert_matmul_op(node),
            "aten::max": lambda node: self.convert_max_op(node),
            "aten::max_pool1d": lambda node: self.convert_maxpool_op(node),
            "aten::max_pool2d": lambda node: self.convert_maxpool_op(node),
            "aten::max_pool2d_with_indices": lambda node: self.convert_maxpool_indices_op(node),
            "aten::max_pool3d": lambda node: self.convert_maxpool_op(node),
            "aten::mean": lambda node: self.convert_reduce_op(node, method="ReduceMean"),
            "aten::meshgrid": lambda node: self.convert_mesh_grid_op(node),
            "aten::min": lambda node: self.convert_min_op(node),
            "aten::mish": lambda node: self.convert_mish_op(node),
            "aten::mm": lambda node: self.convert_matmul_op(node),
            "aten::mv": lambda node: self.convert_matmul_op(node),
            "aten::mul": lambda node: self.convert_mul_op(node),
            "aten::native_group_norm": lambda node: self.convert_native_group_norm_op(node),
            "aten::native_layer_norm": lambda node: self.convert_native_layer_norm_op(node),
            "aten::ne": lambda node: self.convert_compare_op(node, "NotEqual"),
            "aten::neg": lambda node: self.convert_neg_op(node),
            "aten::new_ones": lambda node: self.convert_new_constant_fill_op(node, 1),
            "aten::new_zeros": lambda node: self.convert_new_constant_fill_op(node, 0),
            "aten::nonzero": lambda node: self.convert_nonzero_op(node),
            "aten::new_full": lambda node: self.convert_new_full(node),
            "aten::ones": lambda node: self.convert_constant_fill_op(node, 1),
            "aten::ones_like": lambda node: self.convert_constant_like_op(node, 1),
            "aten::pad": lambda node: self.convert_pad_op(node, mode='unknown'),
            "aten::pow": lambda node: self.convert_pow_op(node),
            "aten::prelu": lambda node: self.convert_prelu_op(node),
            "aten::permute": lambda node: self.convert_permute_op(node),
            "aten::pixel_shuffle": lambda node: self.convert_pixel_shuffle_op(node),
            "aten::pixel_unshuffle": lambda node: self.convert_pixel_unshuffle_op(node),
            "aten::repeat": lambda node: self.convert_repeat_op(node),
            "aten::reflection_pad1d": lambda node: self.convert_pad_op(node, mode='reflect'),
            "aten::reflection_pad2d": lambda node: self.convert_pad_op(node, mode='reflect'),
            "aten::relu": lambda node: self.convert_relu_op(node),
            "aten::remainder": lambda node: self.convert_remainder_op(node),
            "aten::replication_pad1d": lambda node: self.convert_pad_op(node, mode='replicate'),
            "aten::replication_pad2d": lambda node: self.convert_pad_op(node, mode='replicate'),
            "aten::reshape": lambda node: self.convert_reshape_op(node),
            "aten::roll": lambda node: self.convert_roll_op(node),
            "aten::rsqrt": lambda node: self.convert_rsqrt_op(node),
            "aten::rsub": lambda node: self.convert_sub_op(node, is_reverse=True),
            "aten::scaled_dot_product_attention": lambda node: self.convert_scaled_dot_product_attention_op(node),
            "aten::ScalarImplicit": lambda node: self.convert_skip_op(node),
            "aten::scatter": lambda node: self.convert_scatter_op(node),
            "aten::select": lambda node: self.convert_select_op(node),
            "aten::split": lambda node: self.convert_split_op(node),
            "aten::split_with_sizes": lambda node: self.convert_split_op(node),
            "aten::sqrt": lambda node: self.convert_sqrt_op(node),
            "aten::sigmoid": lambda node: self.convert_sigmoid_op(node),
            "aten::sign": lambda node: self.convert_math_op(node, "sign"),
            "aten::sin": lambda node: self.convert_math_op(node, "sin"),
            "aten::sinh": lambda node: self.convert_math_op(node, "sinh"),
            "aten::silu": lambda node: self.convert_silu_op(node),
            "aten::slice": lambda node: self.convert_slice_op(node),
            "aten::softmax": lambda node: self.convert_softmax_op(node),
            "aten::softplus": lambda node: self.convert_softplus_op(node),
            "aten::sort": lambda node: self.convert_sort_op(node),
            "aten::squeeze": lambda node: self.convert_squeeze_op(node),
            "aten::stack": lambda node: self.convert_stack_op(node),
            "aten::sub": lambda node: self.convert_sub_op(node),
            "aten::sum": lambda node: self.convert_reduce_op(node, method="ReduceSum"),
            "aten::size": lambda node: self.convert_size_op(node),
            "aten::t": lambda node: self.convert_transpose_op(node),
            "aten::tan": lambda node: self.convert_math_op(node, "tan"),
            "aten::tanh": lambda node: self.convert_math_op(node, "tanh"),
            "aten::tile": lambda node: self.convert_repeat_op(node),
            "aten::transpose": lambda node: self.convert_transpose_op(node),
            "aten::to": lambda node: self.convert_to_op(node),
            "aten::topk": lambda node: self.convert_topk_op(node),
            "aten::type_as": lambda node:  self.convert_to_op(node),
            "aten::unsqueeze": lambda node: self.convert_unsqueeze_op(node),
            "aten::upsample_bilinear2d": lambda node: self.convert_upsample_op(node, mode='bilinear'),
            "aten::upsample_linear1d": lambda node: self.convert_upsample_op(node, mode='linear'),
            "aten::upsample_nearest1d": lambda node: self.convert_upsample_op(node, mode='nearest'),
            "aten::upsample_nearest2d": lambda node: self.convert_upsample_op(node, mode='nearest'),
            "aten::upsample_nearest3d": lambda node: self.convert_upsample_op(node, mode='nearest'),
            "aten::view": lambda node: self.convert_reshape_op(node),
            "aten::where": lambda node: self.convert_where_op(node),
            "aten::zeros": lambda node: self.convert_constant_fill_op(node, 0),
            "aten::zeros_like": lambda node: self.convert_constant_like_op(node, 0),
            "aten::unbind": lambda node: self.convert_unbind_unpack(node),
            "aten::argmax": lambda node: self.convert_arg_op(node),
            "aten::argmin": lambda node: self.convert_arg_op(node),
            "aten::_native_batch_norm_legit_functional": lambda node: self.convert_native_batch_norm_legit_functional(node),
            "aten::_unsafe_view": lambda node: self.convert_reshape_op(node),
            "aten::_softmax": lambda node: self.convert_softmax_op(node),
            "aten::_to_copy": lambda node: self.convert_skip_op(node),
            ###### prim #####
            "prim::Constant": lambda node: self.convert_constant(node),
            "prim::DictConstruct": lambda node: self.convert_dict_construct(node),
            "prim::GetAttr": lambda node: self.convert_get_attr(node),
            "prim::ListConstruct": lambda node: self.convert_list_construct(node),
            "prim::ListUnpack": lambda node: self.convert_list_unpack(node),
            "prim::NumToTensor": lambda node: self.convert_skip_op(node),
            "prim::TupleConstruct": lambda node: self.convert_tuple(node),
            "prim::TupleUnpack": lambda node: self.convert_tuple_unpack(node),
            # "prim::If": lambda node: self.convert_if(node),
            ###### torchvision ######
            "torchvision::deform_conv2d": lambda node: self.convert_deform_conv2d_op(node),
            "torchvision::roi_align": lambda node: self.convert_roi_align_op(node),
        }

        # yapf: enable
        self.check_op_types()

    def __del__(self):
        if self.mlir != None:
            del self.mlir
            self.mlir = None

    def get_all_op_types(self):
        """Return all operator names in the input graph"""
        self.nodes = list(self.graph.nodes())
        prim_blocks = ["prim::If", "prim::Loop"]
        for prim in prim_blocks:
            prim_nodes = self.graph.findAllNodes(prim, recurse=True)
            for prim_node in prim_nodes:
                for block in prim_node.blocks():
                    self.nodes += block.nodes()
        return set(node.kind() for node in self.nodes)

    def check_op_types(self):
        op_types = self.get_all_op_types()
        known_ops = list(self.op_factory.keys())
        unknown_ops = []
        for op_type in op_types:
            logger.info(op_type)
            if op_type not in known_ops:
                if not (op_type.endswith("_") and op_type[:-1] in known_ops):
                    unknown_ops.append(op_type)
        if len(unknown_ops) != 0:
            raise RuntimeError(
                "The following operators are not implemented: {}".format(unknown_ops))

    def check_need(self, name):
        for node in self.converted_nodes:
            for i in node.inputs:
                if i == name:
                    return True
        if name in self.output_names:
            return True
        return False

    def load_torch_model(self, torch_file, input_shapes: list, input_types: list,
                         output_names: list):
        if isinstance(torch_file, str):
            self.model = torch.jit.load(
                torch_file, map_location=torch.device('cpu'))
        else:
            self.model = torch_file
        self.model.eval()
        self.graph = self.model.inlined_graph
        self.state_dict = self.model.state_dict()
        is_module = isinstance(self.model, torch.jit.ScriptModule)
        inputs = list(self.graph.inputs())
        inputs = inputs[1:] if is_module else inputs
        self.input_names = []
        if len(input_shapes) != len(inputs):
            raise RuntimeError(f"Input shape not match inputs: {input_shapes}")
        for s, inp in zip(input_shapes, inputs):
            self.input_names.append(inp.debugName())
            self.addShape(inp.debugName(), s)
        self.output_names = []
        self.origin_output_names = []
        if output_names:
            self.output_names = output_names
            self.origin_output_names = output_names
        else:
            for outp in self.graph.outputs():
                if outp.node().kind() == 'prim::TupleConstruct' or \
                   outp.node().kind() == 'prim::ListConstruct':
                    ins = outp.node().inputs()
                    self.output_names.extend([i.debugName() for i in ins])
                elif outp.node().kind() == 'prim::DictConstruct':
                    ins = outp.node().inputs()
                    ls_ins = list(ins)
                    in_num = len(ls_ins)
                    assert in_num % 2 == 0
                    self.output_names.extend(
                        [ls_ins[i*2+1].debugName() for i in range(int(in_num/2))])
                else:
                    self.output_names.append(outp.debugName())
        self.weight_names = []
        self.num_input = len(self.input_names)
        self.num_output = len(self.output_names)
        self.input_shapes = input_shapes
        self.input_types = []
        for t in input_types:
            if t.lower() not in self.TypeMap:
                raise RuntimeError(f"Unknown type {t}")
            self.input_types.append(self.TypeMap[t.lower()])
        self.output_shapes = [[]] * self.num_output
        self.output_types = [None]* self.num_output

        ### create dict for input output
        self.input_dict = {}
        self.output_dict = {}
        for idx,name in enumerate(self.input_names):
            self.input_dict[name] = [self.input_shapes[idx]]
        for idx,name in enumerate(self.output_names):
            self.output_dict[name] = idx
        for name in self.output_names:
            if name in self.input_dict:
                input_info = self.input_dict[name]
                idx = self.output_dict[name]
                self.output_shapes[idx] = input_info[0]
                # self.output_types[idx] = input_info[1]

    def init_MLIRImporter(self):
        # init importer
        self.mlir = MLIRImporter(self.input_shapes, self.output_shapes, self.model_name,
                                 Platform.TORCH, self.input_types,  self.output_types,run_mode=self.run_mode)
        self.weight_file = self.mlir.weight_file

    def generate_list_map(self):
        self.list_map = dict()
        for node in self.converted_nodes:
            if node.op_type == "prim::ListUnpack":
                self.list_map[node.inputs[0]] = node.outputs

    def convert_maxpool_indices_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        kernel_shape = self.const_val[torch_node.inputs[1]]
        strides = self.const_val[torch_node.inputs[2]]
        pads = self.const_val[torch_node.inputs[3]]
        dilation = [1,1]
        ceil_mode = False
        pads = pads + pads  # the pad of torch is symmetric
        new_op = top.MaxPoolWithMaskOp(
            self.unranked_type,
            self.unranked_type,
            op,
            kernel_shape=kernel_shape,
            strides=strides,
            pads=pads,
            ceil_mode=ceil_mode,
            loc = self.get_loc(
                [torch_node.name + ".output", torch_node.name + ".mask"]),
            ip=self.mlir.insert_point
        )
        self.addOperand(torch_node.outputs[0], new_op.output)
        self.addOperand(torch_node.outputs[1], new_op.mask)

    def convert_native_batch_norm_legit_functional(self, torch_node: TorchNode):
        op0   = self.getOp(torch_node.inputs[0])
        weight = self.getOp(torch_node.inputs[1])
        bias = self.getOp(torch_node.inputs[2])
        running_mean = self.getOp(torch_node.inputs[3])
        running_var = self.getOp(torch_node.inputs[4])
        training = self.const_val[torch_node.inputs[5]]
        momentum = self.const_val[torch_node.inputs[6]]
        eps = self.const_val[torch_node.inputs[7]]
        new_op = top.MeanRstdOp(self.unranked_type,
                                self.unranked_type,
                                self.unranked_type,
                                self.unranked_type,
                                self.unranked_type,
                                self.unranked_type,
                                op0,
                                running_mean,
                                running_var,
                                weight,
                                bias,
                                eps = eps,
                                momentum = momentum,
                                loc=self.get_loc(
                                    [torch_node.name + "_mean",
                                    torch_node.name + "_rstd",
                                    torch_node.name + "_running_mean_update",
                                    torch_node.name + "_running_var_update",
                                    torch_node.name + "_scale",
                                    torch_node.name + "_bias_new"]),
                                ip=self.mlir.insert_point)
        bn_out = top.ScaleOp(self.unranked_type,
                                op0,
                                new_op.scale,
                                new_op.bias_new,
                                loc=self.get_loc(torch_node.name + "_output"),
                                ip=self.mlir.insert_point).output
        self.addOperand(torch_node.outputs[0], bn_out)
        self.addOperand(torch_node.outputs[1], new_op.mean)
        self.addOperand(torch_node.outputs[2], new_op.rstd)
        self.addOperand(torch_node.outputs[3], new_op.running_mean_update)
        self.addOperand(torch_node.outputs[4], new_op.running_var_update)



    def generate_mlir(self, mlir_file: str):
        """convert all to mlir"""
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
            input_ = self.mlir.create_input_op(
                self.get_loc(_name), idx, kwargs)
            self.addOperand(_name, input_)

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

        self.generate_list_map()
        for n in self.converted_nodes:
            self.op_factory.get(n.op_type, lambda x: NoneAndRaise(x))(n)
        # add return op
        return_op = list()
        # Set output
        final_output_names = []
        if self.origin_output_names:
            final_output_names = self.origin_output_names
        else:
            final_output_names = self.output_names
        for idx, _name in enumerate(final_output_names):
            op = self.getOperand(_name)
            return_op.append(op)

        self.mlir.create_return_op(return_op)
        mlir_txt = self.mlir.print_module()
        with open(mlir_file, "w") as f:
            f.write(mlir_txt)
        self.WeightToNpz(self.weight_file)
        logger.info("Save mlir file: {}".format(mlir_file))

    def get_input_by_name(self, input):
        return self.const_val[input] if input in self.const_val.keys() else self.getOp(input)

    def get_loc(self, names):
        if isinstance(names, str):
            return Location.fused([Location.name(names)], context=self.mlir.ctx)
        elif isinstance(names, list):
            return Location.fused([Location.name(n) for n in names], context=self.mlir.ctx)
        else:
            raise RuntimeError("Unknown names:{}".format(names))

    def is_custom_triangular(self, weight, threadhold):
        for index in np.ndindex(weight.shape[:-2]):
            matrix = weight[index]

            rows, cols = matrix.shape
            if rows != cols:
                return False

            for i in range(rows):
                for j in range(cols):
                    if i >= j and matrix[i, j] != 0:
                        return False
                    if i < j and matrix[i, j] > threadhold:
                        return False
        return True


    def convert_base_conv_op(self, torch_node: TorchNode, mode=False):

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
                output_size = (input_size + stride -
                               effective_filter_size) // stride
            padding_needed = np.int64((output_size - 1) * stride + effective_filter_size -
                                      input_size)
            padding_needed = padding_needed.clip(min=0)

            padding_before = padding_needed // 2
            padding_after = padding_needed - padding_before
            pad = [i for i in padding_before] + [i for i in padding_after]
            return pad

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
            pads = _compute_pad(strides, dilations,
                                input_size, kernel_shape, pads)
        else:
            transposed = self.const_val[torch_node.inputs[6]]
            pads = pads + pads  # the pad of torch is symmetric
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
        weight_is_coeff = 1 if self.isWeight(torch_node.inputs[1]) else 0
        if torch_node.inputs[2] not in self.const_val.keys() or self.const_val[
                torch_node.inputs[2]] is not None:
            bias_op = self.getOp(torch_node.inputs[2])
        else:
            bias_op = self.mlir.none_op
        if transposed:
            output_padding = self.const_val[torch_node.inputs[7]]
            new_op = top.DeconvOp(self.unranked_type,
                                  op,
                                  filter_op,
                                  bias_op,
                                  kernel_shape=kernel_shape,
                                  strides=strides,
                                  dilations=dilations,
                                  output_padding=output_padding,
                                  pads=pads,
                                  group=group,
                                  do_relu=False,
                                  loc=self.get_loc(torch_node.name),
                                  ip=self.mlir.insert_point).output
            return self.addOperand(torch_node.name, new_op)
        new_op = top.ConvOp(self.unranked_type,
                            op,
                            filter_op,
                            bias_op,
                            kernel_shape=kernel_shape,
                            strides=strides,
                            dilations=dilations,
                            pads=pads,
                            group=group,
                            weight_is_coeff=weight_is_coeff,
                            do_relu=False,
                            loc=self.get_loc(torch_node.name),
                            ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_conv_op(self, torch_node: TorchNode):
        # convolution or transposed_convolution
        self.convert_base_conv_op(torch_node)

    def convert_conv_mode_op(self, torch_node: TorchNode):
        # only for convolution
        self.convert_base_conv_op(torch_node, True)

    def convert_adaptive_avgpool_op(self, torch_node: TorchNode, spatial_rank=1):
        op = self.getOp(torch_node.inputs[0])
        output_size = self.const_val[torch_node.inputs[1]]
        # assert (output_size == [1, 1] or output_size == [1]
        #         and "Currently adaptive_avgpool2d/1d is only taken as global_avgpool")

        new_op = top.AdaptiveAvgPoolOp(self.unranked_type,
                                       op,
                                       output_size=output_size,
                                       loc=self.get_loc(torch_node.name),
                                       ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

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
            op6 = self.getOp(torch_node.inputs[6])
            assert op6 == self.mlir.none_op
        pads = pads + pads  # the pad of torch is symmetric
        new_op = top.AvgPoolOp(self.unranked_type,
                               op,
                               kernel_shape=kernel_shape,
                               strides=strides,
                               pads=pads,
                               do_relu=False,
                               count_include_pad=count_include_pad,
                               loc=self.get_loc(torch_node.name),
                               ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_max_op(self, torch_node: TorchNode):
        op = self.getOperand(torch_node.inputs[0])
        dim = 0  if (len(torch_node.inputs)<=1) else self.const_val[torch_node.inputs[1]]
        keepdims = False if  (len(torch_node.inputs)<=2) else self.const_val[torch_node.inputs[2]]
        select_last_index = True
        out_needs = [False, False]
        for idx, out in enumerate(torch_node.outputs):
            if len(out) > 0 and self.check_need(out):
                out_needs[idx] = True
        new_op = top.ArgOp(self.unranked_type,
                           self.unranked_type,
                           op,
                           axis=dim,
                           keepdims=keepdims,
                           mode=StringAttr.get("ArgMax"),
                           select_last_index=select_last_index,
                           loc=self.get_loc(torch_node.outputs),
                           ip=self.mlir.insert_point)
        out_ops = [new_op.values, new_op.indices]
        for idx, need in enumerate(out_needs):
            if not need:
                continue
            self.addOperand(torch_node.outputs[idx], out_ops[idx])

    def convert_maxpool_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        kernel_shape = self.const_val[torch_node.inputs[1]]
        strides = self.const_val[torch_node.inputs[2]]
        pads = self.const_val[torch_node.inputs[3]]
        dilation = self.const_val[torch_node.inputs[4]]
        ceil_mode = self.const_val[torch_node.inputs[5]]
        assert (np.array(dilation) == 1).all()
        pads = pads + pads  # the pad of torch is symmetric
        new_op = top.MaxPoolOp(self.unranked_type,
                               op,
                               kernel_shape=kernel_shape,
                               strides=strides,
                               pads=pads,
                               do_relu=False,
                               ceil_mode=ceil_mode,
                               loc=self.get_loc(torch_node.name),
                               ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_min_op(self, torch_node: TorchNode):
        op = self.getOperand(torch_node.inputs[0])
        dim = self.const_val[torch_node.inputs[1]]
        keepdims = self.const_val[torch_node.inputs[2]]
        select_last_index = True
        out_needs = [False, False]
        for idx, out in enumerate(torch_node.outputs):
            if len(out) > 0 and self.check_need(out):
                out_needs[idx] = True
        new_op = top.ArgOp(self.unranked_type,
                           self.unranked_type,
                           op,
                           axis=dim,
                           keepdims=keepdims,
                           mode=StringAttr.get("ArgMin"),
                           select_last_index=select_last_index,
                           loc=self.get_loc(torch_node.outputs),
                           ip=self.mlir.insert_point)
        out_ops = [new_op.values, new_op.indices]
        # logger.info(out_needs)
        for idx, need in enumerate(out_needs):
            if not need:
                continue
            self.addOperand(torch_node.outputs[idx], out_ops[idx])

    def _mul_scale(self, in_name, scale):
        in_op = self.getOp(in_name)
        op_name = in_name + "_ml_mulscale"
        new_op = top.MulConstOp(self.unranked_type,
                                in_op,
                                scale,
                                loc=self.get_loc(op_name),
                                ip=self.mlir.insert_point).output
        self.addOperand(op_name, new_op)
        return new_op

    def convert_scaled_dot_product_attention_op(self, torch_node: TorchNode):
        query = self.getOp(torch_node.inputs[0])
        key = self.getOp(torch_node.inputs[1])
        value = self.getOp(torch_node.inputs[2])
        mask = self.getOp(torch_node.inputs[3])
        assert mask == self.mlir.none_op or self.is_custom_triangular(self.getWeight(torch_node.inputs[3]), -3.4028234663852886e+38)
        dropout_p = self.const_val[torch_node.inputs[4]]
        is_causal = self.const_val[torch_node.inputs[5]]
        scale = 0.0 #default value for no scale
        if len(torch_node.inputs) > 6:
            if torch_node.inputs[6] in self.const_val.keys():
                scale = self.const_val[torch_node.inputs[6]]
        new_op = top.ScaleDotProductAttentionOp(self.unranked_type,
                                                 query,
                                                 key,
                                                 value,
                                                 self.mlir.none_op, # only support mask as none now
                                                 dropout_p=dropout_p,
                                                 is_causal=is_causal,
                                                 scale=scale,
                                                 loc=self.get_loc(torch_node.name),
                                                 ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_add_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        scale = self.const_val[torch_node.inputs[2]]
        op1 = self.getOp(torch_node.inputs[1]) if scale == 1 else \
            self._mul_scale(torch_node.inputs[1], scale)
        if self.isWeight(torch_node.inputs[0]):
            op0, op1 = op1, op0
        new_op = top.AddOp(self.unranked_type, [op0, op1],
                           do_relu=False,
                           loc=self.get_loc(torch_node.name),
                           ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_addmm_op(self, torch_node: TorchNode):
        in_op = self.getOp(torch_node.inputs[0])
        mat1_op = self.getOp(torch_node.inputs[1])
        mat2_op = self.getOp(torch_node.inputs[2])
        beta = self.const_val[torch_node.inputs[3]]
        alpha = self.const_val[torch_node.inputs[4]]
        mm_op = top.MatMulOp(self.unranked_type,
                             mat1_op,
                             mat2_op,
                             self.mlir.none_op,
                             do_relu=False,
                             loc=self.get_loc(torch_node.name + "_mm"),
                             ip=self.mlir.insert_point).output
        assert (beta == 1.0 and alpha == 1.0)  # TODO:need to support
        new_op = top.AddOp(self.unranked_type, [in_op, mm_op],
                           coeff=[beta, alpha],
                           loc=self.get_loc(torch_node.name),
                           ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_sub_op(self, torch_node: TorchNode, is_reverse=False):
        op0 = self.getOp(torch_node.inputs[0])
        scale = self.const_val[torch_node.inputs[2]]
        if torch_node.inputs[1] in self.const_val.keys():
            op1 = self.const_val[torch_node.inputs[1]] * scale
            new_op = top.SubConstOp(self.unranked_type,
                                    op0,
                                    op1,
                                    is_reverse=is_reverse,
                                    loc=self.get_loc(torch_node.name),
                                    ip=self.mlir.insert_point).output
        else:
            op1 = self.get_input_by_name(torch_node.inputs[1]) if scale == 1 else \
                self._mul_scale(torch_node.inputs[1], scale)
            new_op = top.SubOp(self.unranked_type, [op0, op1],
                               is_reverse=is_reverse,
                               loc=self.get_loc(torch_node.name),
                               ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_reduce_op(self, torch_node: TorchNode, method):
        assert method in ("ReduceMin", "ReduceMax", "ReduceMean", "ReduceL2", "ReduceL1",
                          "ReduceSum", "ReduceProd")
        op0 = self.getOp(torch_node.inputs[0])
        axes = self.const_val[torch_node.inputs[1]] if torch_node.inputs[1] in self.const_val else [-1]
        if len(torch_node.inputs) < 3:
            keepdims = False
        else:
            keepdims = self.const_val[torch_node.inputs[2]] if torch_node.inputs[2] in self.const_val else False
        # TODO: axes are not consecutive numbers
        # TODO: axes is none
        new_op = top.ReduceOp(self.unranked_type,
                              op0,
                              axes = axes,
                              keepdims = keepdims,
                              mode = StringAttr.get(method),
                              loc=self.get_loc(torch_node.name),
                              ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_pow_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        if torch_node.inputs[1] in self.const_val.keys():
            op1 = self.const_val[torch_node.inputs[1]]
            new_op = top.PowOp(self.unranked_type,
                               op0,
                               op1,
                               loc=self.get_loc(torch_node.name),
                               ip=self.mlir.insert_point).output
        else:
            op1 = self.getOp(torch_node.inputs[1])
            # TODO: exponent is tensor
            assert 0
            new_op = top.PowOp(self.unranked_type,
                               op0,
                               op1,
                               loc=self.get_loc(torch_node.name),
                               ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_size_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        axis = None
        if len(torch_node.inputs) > 1:
            axis = self.const_val[torch_node.inputs[1]]
        new_op = top.SizeOp(self.unranked_type,
                            op0,
                            axis=axis,
                            loc=self.get_loc(torch_node.name),
                            ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_constant_fill_op(self, torch_node: TorchNode, value):
        if torch_node.inputs[0] in self.const_val:
            data = np.array(self.const_val[torch_node.inputs[0]], np.int32)
            self.addWeight(torch_node.inputs[0], data)
        op0 = self.getOp(torch_node.inputs[0])
        new_op = top.ConstantFillOp(self.unranked_type,
                                    op0,
                                    value=value,
                                    loc=self.get_loc(torch_node.name),
                                    ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_constant_like_op(self, torch_node: TorchNode, value):
        op0 = self.getOp(torch_node.inputs[0])
        size_op = top.SizeOp(self.unranked_type,
                             op0,
                             axis=None,
                             loc=self.get_loc(torch_node.inputs[0] + "_size"),
                             ip=self.mlir.insert_point).output
        self.addOperand(torch_node.inputs[0] + "_size", size_op)
        new_op = top.ConstantFillOp(self.unranked_type,
                                    size_op,
                                    value=value,
                                    loc=self.get_loc(torch_node.name),
                                    ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_new_constant_fill_op(self, torch_node: TorchNode, value):
        if torch_node.inputs[1] in self.const_val:
            data = np.array(self.const_val[torch_node.inputs[1]], np.int32)
            self.addWeight(torch_node.inputs[1], data)
        op0 = self.getOp(torch_node.inputs[1])
        new_op = top.ConstantFillOp(self.unranked_type,
                                    op0,
                                    value=value,
                                    loc=self.get_loc(torch_node.name),
                                    ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_nonzero_op(self, onnx_node):
        assert (len(onnx_node.inputs) == 1)
        input_data = self.getOp(onnx_node.inputs[0])
        new_op = top.NonZeroOp(self.unranked_type,
                               input_data,
                               order=StringAttr.get("ColMajor"),
                               loc=self.get_loc("{}_{}".format(
                                   onnx_node.name, onnx_node.op_type)),
                               ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_new_full(self, torch_node: TorchNode):
        assert (len(torch_node.inputs) >= 2)
        op0 = self.getOp(torch_node.inputs[1])
        fill_value = self.const_val[torch_node.inputs[2]]
        new_op = top.ConstantFillOp(self.unranked_type,
                                    op0,
                                    value=fill_value,
                                    loc=self.get_loc(torch_node.name),
                                    ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_expand_op(self, torch_node: TorchNode):
        implict = False
        if torch_node.inputs[2] in self.const_val:
            implict = self.const_val[torch_node.inputs[2]]
        if implict:
            logger.info('not handled')
        opI = self.getOp(torch_node.inputs[0])
        opS = self.getOp(torch_node.inputs[1])
        new_cf = top.ConstantFillOp(self.unranked_type,
                                    opS,
                                    value=1.0,
                                    loc=self.get_loc(torch_node.name+'_size'),
                                    ip=self.mlir.insert_point).output
        new_exp = top.MulOp(self.unranked_type, [opI, new_cf],
                            do_relu=False,
                            loc=self.get_loc(torch_node.name),
                            ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_exp)

    def convert_expand_as_op(self, torch_node: TorchNode):
        opI = self.getOp(torch_node.inputs[0])
        opS = self.getOp(torch_node.inputs[1])
        size_op = top.SizeOp(self.unranked_type,
                             opS,
                             axis=None,
                             loc=self.get_loc(torch_node.inputs[1] + "_size"),
                             ip=self.mlir.insert_point).output
        new_cf = top.ConstantFillOp(self.unranked_type,
                                    size_op,
                                    value=1.0,
                                    loc=self.get_loc(torch_node.name+'_size'),
                                    ip=self.mlir.insert_point).output
        new_exp = top.MulOp(self.unranked_type, [opI, new_cf],
                            do_relu=False,
                            loc=self.get_loc(torch_node.name),
                            ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_exp)

    def convert_arange_op(self, torch_node: TorchNode):
        in0, in1, in2 = torch_node.inputs[:3]
        in_num = len(torch_node.inputs)
        step = self.mlir.none_op
        if in_num == 5:
            start = self.mlir.none_op
            end = self.getOp(in0)
        elif in_num >= 6:
            start = self.getOp(in0)
            end = self.getOp(in1)
        if in_num >= 7:
            step = self.getOp(in2)
        new_op = top.ArangeOp(self.unranked_type,
                              start,
                              end,
                              step,
                              loc=self.get_loc(torch_node.name),
                              ip=self.mlir.insert_point)
        self.addOperand(torch_node.name, new_op.output)

    def convert_constant(self, torch_node: TorchNode):
        name, data, is_tensor = get_constant(torch_node,torch_node.node_proto)
        if data is None:
            self.addOperand(name, self.mlir.none_op)
        elif not is_tensor:
            if isinstance(data, int):
                # avoid to large
                if data > np.iinfo(np.int32).max:
                    data = np.iinfo(np.int32).max
            self.const_val[name] = data
            if not isinstance(data, str):
                self.addWeight(name, np.array([data], dtype=np.float32))
        else:
            self.addWeight(name, data)

    def convert_list_construct(self, torch_node: TorchNode):
        name = torch_node.outputs[0]
        all_const = True
        for input in torch_node.inputs:
            if input not in self.const_val:
                all_const = False
                break
        # all const
        data = []
        if all_const:
            for input in torch_node.inputs:
                data.append(self.const_val[input])
            self.const_val[name] = data
            t = np.array(data, np.float32)
            self.addWeight(name, t)
            return
        ops = []
        # to list op
        for input in torch_node.inputs:
            if input in self.const_val:
                val = self.const_val[input]
                t = np.array([val], np.float32)
                self.addWeight(input, t)
                data.append(input)
                ops.append(self.getWeightOp(input))
            else:
                data.append(input)
                ops.append(self.getOp(input))
        self.tensor_list[name] = data
        if len(ops) == 1:
            self.addOperand(name, ops[0])
        else:
            new_op = top.ListOp(self.unranked_type,
                                ops,
                                loc=self.get_loc(name),
                                ip=self.mlir.insert_point).output
            self.addOperand(name, new_op)

    # def convert_if(self, torch_node: TorchNode):
    # def convert_is_type_op(self, torch_node: TorchNode, type):

    def convert_get_attr(self, torch_node: TorchNode):
        node = torch_node.node_proto

        if node.output().type().kind() != 'TensorType':
            return
        data = get_attr(self.model, node)[0].detach()
        weight_name = get_attr_name(torch_node)
        self.addWeight(weight_name, data.numpy())

    def convert_mul_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        op1 = self.getOp(torch_node.inputs[1])
        new_op = top.MulOp(self.unranked_type, [op0, op1],
                           do_relu=False,
                           loc=self.get_loc(torch_node.name),
                           ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_div_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        op1 = self.getOp(torch_node.inputs[1])
        new_op = top.DivOp(self.unranked_type, [op0, op1],
                           loc=self.get_loc(torch_node.name),
                           ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_dot_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        op1 = self.getOp(torch_node.inputs[1])
        new_op0 = top.UnsqueezeOp(self.unranked_type,
                                  op0,
                                  axes=[0],
                                  loc=self.get_loc("op0_unsqueeze"),
                                  ip=self.mlir.insert_point).output
        new_op1 = top.UnsqueezeOp(self.unranked_type,
                                  op1,
                                  axes=[0],
                                  loc=self.get_loc("op1_unsqueeze"),
                                  ip=self.mlir.insert_point).output
        new_op = top.MatMulOp(self.unranked_type,
                              new_op0,
                              new_op1,
                              self.mlir.none_op,
                              do_relu=False,
                              right_transpose=True,
                              loc=self.get_loc(torch_node.name),
                              ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_detach_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        self.addOperand(torch_node.name, op0)

    def convert_floor_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        floor_op = top.FloorOp(self.unranked_type,
                               op0,
                               loc=self.get_loc(torch_node.name),
                               ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, floor_op)

    def convert_flip_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        axis_data = self.getWeight(torch_node.inputs[1])
        last_op = op0

        for i in range(len(axis_data)):
            last_op = top.SliceOp(self.unranked_type,
                                  last_op,
                                  self.mlir.none_op,
                                  self.mlir.none_op,
                                  self.mlir.none_op,
                                  offset=[-1],
                                  steps=[-1],
                                  ends=[np.iinfo(np.int64).min],
                                  axes=[axis_data[i]],
                                  loc=self.get_loc(
                                      "{}_slice_{}".format(torch_node.name, i)),
                                  ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, last_op)

    def convert_floor_divide_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        if torch_node.inputs[1] in self.const_val:
            var = self.const_val[torch_node.inputs[1]]
            div_op = top.MulConstOp(self.unranked_type,
                                    op0,
                                    const_val=1.0 / var,
                                    loc=self.get_loc(torch_node.name + "_div"),
                                    ip=self.mlir.insert_point).output
        else:
            op1 = self.getOp(torch_node.inputs[1])
            div_op = top.DivOp(self.unranked_type, [op0, op1],
                               loc=self.get_loc(torch_node.name + "_div"),
                               ip=self.mlir.insert_point).output
        floor_op = top.FloorOp(self.unranked_type,
                               div_op,
                               loc=self.get_loc(torch_node.name),
                               ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, floor_op)

    def convert_skip_op(self, torch_node: TorchNode):
        # warning: in_op.output name shoud change to torch_node name
        in_op = self.getOp(torch_node.inputs[0])
        self.addOperand(torch_node.name, in_op)

    def convert_index_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        indices = self.getOp(torch_node.inputs[1])
        new_op = top.GatherOp(self.unranked_type,
                              op0,
                              indices,
                              axis=0,
                              loc=self.get_loc(torch_node.name),
                              ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_index_put_op(self, torch_node: TorchNode):
        input = self.getOp(torch_node.inputs[0])
        indices = self.getOp(torch_node.inputs[1])
        values = self.getOp(torch_node.inputs[2])
        new_op = top.IndexPutOp(self.unranked_type,
                                input,
                                indices,
                                values,
                                loc=self.get_loc(torch_node.name),
                                ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_to_op(self, torch_node: TorchNode):
        in_op = self.getOp(torch_node.inputs[0])
        self.addOperand(torch_node.name, in_op)

    def convert_gather_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        axis = self.const_val[torch_node.inputs[1]]
        op1 = self.getOp(torch_node.inputs[2])
        new_op = top.GatherElementsOp(self.unranked_type,
                                      op0,
                                      op1,
                                      axis=axis,
                                      loc=self.get_loc(torch_node.name),
                                      ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_compare_op(self, torch_node: TorchNode, mode):
        assert mode in ("Equal", "Greater", "GreaterOrEqual",
                        "Less", "LessOrEqual", "NotEqual")
        op0 = self.getOp(torch_node.inputs[0])
        if torch_node.inputs[1] in self.const_val:
            const_val = self.const_val[torch_node.inputs[1]]
            new_op = top.CompareConstOp(self.unranked_type,
                                        op0,
                                        mode=StringAttr.get(mode),
                                        const_val=const_val,
                                        inversed=False,
                                        loc=self.get_loc(torch_node.name),
                                        ip=self.mlir.insert_point).output
        else:
            op1 = self.getOp(torch_node.inputs[1])
            new_op = top.CompareOp(self.unranked_type,
                                   op0,
                                   op1,
                                   mode=StringAttr.get(mode),
                                   loc=self.get_loc(torch_node.name),
                                   ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_math_op(self, torch_node: TorchNode, mode: str):
        assert mode in ["cos", "cosh", "sin", "sinh", "tan",
                        "tanh", "exp", "sign", "arctanh", "arccos"]
        op0 = self.getOp(torch_node.inputs[0])
        cmd = "top.%sOp(self.unranked_type, op0, loc=self.get_loc(torch_node.name), ip=self.mlir.insert_point).output" % mode.capitalize(
        )
        new_op = eval(cmd)
        self.addOperand(torch_node.name, new_op)

    def convert_arctan_op(self, torch_node: TorchNode, mode: str):
        assert mode in ["arctan"]
        # arctan(x) = aign(x) * acos(1 / (sqrt(1 + mul(abs(x), abs(x)))))
        op0 = self.getOp(torch_node.inputs[0])
        op_name = torch_node.name + "_sign"
        sign_op = top.SignOp(self.unranked_type,
                                op0,
                                loc=self.get_loc(op_name),
                                ip=self.mlir.insert_point).output

        op_name = torch_node.name + "_abs"
        abs_op = top.AbsOp(self.unranked_type,
                                op0,
                                loc=self.get_loc(op_name),
                                ip=self.mlir.insert_point).output
        op_name = torch_node.name + "_mul"
        mul_op = top.MulOp(self.unranked_type, [abs_op, abs_op],
                           do_relu=False,
                           loc=self.get_loc(op_name),
                           ip=self.mlir.insert_point).output

        op_name = torch_node.name + "_ml_mulscale"
        add_op = top.AddConstOp(self.unranked_type,
                                mul_op,
                                const_val=1,
                                loc=self.get_loc(op_name),
                                ip=self.mlir.insert_point).output

        op_name = torch_node.name + "_sqrt"
        sqrt_op = top.SqrtOp(self.unranked_type,
                             add_op,
                             loc=self.get_loc(op_name),
                             ip=self.mlir.insert_point).output

        op_name = torch_node.name + "_reciprocal"
        reciprocal_op = top.ReciprocalOp(self.unranked_type,
                                         sqrt_op,
                                         loc=self.get_loc(op_name),
                                         ip=self.mlir.insert_point).output

        op_name = torch_node.name + "_arccos"
        arccos_op = top.ArccosOp(self.unranked_type,
                                 reciprocal_op,
                                 loc=self.get_loc(op_name),
                                 ip=self.mlir.insert_point).output

        arctan_op = top.MulOp(self.unranked_type, [sign_op, arccos_op],
                           do_relu=False,
                           loc=self.get_loc(torch_node.name),
                           ip=self.mlir.insert_point).output

        self.addOperand(torch_node.name, arctan_op)

    def convert_prelu_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        op1 = self.getOp(torch_node.inputs[1])
        new_op = top.PReluOp(self.unranked_type,
                             op0,
                             op1,
                             loc=self.get_loc(torch_node.name),
                             ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_permute_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        order = self.const_val[torch_node.inputs[1]]
        new_op = top.PermuteOp(self.unranked_type,
                               op,
                               order=order,
                               loc=self.get_loc(torch_node.name),
                               ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_repeat_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        repeat = self.getOp(torch_node.inputs[1])
        new_op = top.RepeatOp(self.unranked_type,
                              op,
                              repeat,
                              loc=self.get_loc(torch_node.name),
                              ip=self.mlir.insert_point)
        self.addOperand(torch_node.name, new_op.output)

    def convert_transpose_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        no_dims = len(torch_node.inputs) == 1
        dim0 = self.const_val[torch_node.inputs[1]] if not no_dims else -1
        dim1 = self.const_val[torch_node.inputs[2]] if not no_dims else -2
        new_op = top.TransposeOp(self.unranked_type,
                                 op,
                                 dim0,
                                 dim1,
                                 loc=self.get_loc(torch_node.name),
                                 ip=self.mlir.insert_point)
        self.addOperand(torch_node.name, new_op.output)

    def convert_sqrt_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = top.SqrtOp(self.unranked_type,
                            op,
                            loc=self.get_loc(torch_node.name),
                            ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_index_select_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        op1 = self.getOp(torch_node.inputs[2])
        axis = self.const_val[torch_node.inputs[1]]
        new_op = top.GatherOp(self.unranked_type,
                              op0,
                              op1,
                              axis=axis,
                              loc=self.get_loc(torch_node.name),
                              ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_embedding_op(self, torch_node: TorchNode):
        weight = self.getOp(torch_node.inputs[0])
        indices = self.getOp(torch_node.inputs[1])
        padding_idx = self.const_val[torch_node.inputs[2]]
        scale_grad_by_freq = self.const_val[torch_node.inputs[3]]
        sparse = self.const_val[torch_node.inputs[4]]
        new_op = top.GatherOp(self.unranked_type,
                              weight,
                              indices,
                              axis=0,
                              loc=self.get_loc(torch_node.name),
                              ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_mesh_grid_op(self, torch_node: TorchNode):
        operands = []
        inputs = self.tensor_list[torch_node.inputs[0]]
        num = len(inputs)
        for name in inputs:
            op = self.getOp(name)
            operands.append(op)

        index = self.const_val[torch_node.inputs[1]] == 'xy' \
            if len(torch_node.inputs) == 2 else False

        new_op = top.MeshGridOp([self.unranked_type] * num,
                                operands,
                                index,
                                loc=self.get_loc(torch_node.name),
                                ip=self.mlir.insert_point).outputs
        self.addOperand(torch_node.name, new_op)
        output_names = self.list_map[torch_node.outputs[0]]
        for i, out in enumerate(output_names):
            self.addOperand(out, new_op[i])
        self.tensor_list[torch_node.outputs[0]] = output_names

    def convert_reshape_op(self, torch_node: TorchNode):
        in_op = self.getOp(torch_node.inputs[0])
        if torch_node.inputs[1] in self.const_val:
            shape = self.const_val[torch_node.inputs[1]]
            new_op = top.ReshapeOp(self.unranked_type,
                                   in_op,
                                   shape=shape,
                                   loc=self.get_loc(torch_node.name),
                                   ip=self.mlir.insert_point).output
            self.addOperand(torch_node.name, new_op)
        else:
            shape_op = self.getOp(torch_node.inputs[1])
            new_op = top.ViewOp(self.unranked_type,
                                in_op,
                                shape_op,
                                loc=self.get_loc(torch_node.name),
                                ip=self.mlir.insert_point).output
            self.addOperand(torch_node.name, new_op)

    def convert_roll_op(self, torch_node: TorchNode):
        #
        # ====== case1 (dims is None): =========
        #
        #       roll => flatten -> slice0 -> concat -> reshape
        #                        \        /
        #                          slice1
        #
        #
        #
        # ====== case2 (dims is not None): ========
        #
        #    for i in dims:
        #       roll => slice1_i -> concat_i
        #             \          /
        #               slice2_i
        #
        #    concat(concat_0, ···， concat_dims)
        #
        in_op = self.getOp(torch_node.inputs[0])
        shape = self.getShape(torch_node.inputs[0])
        shifts = self.const_val[torch_node.inputs[1]]
        dims = self.const_val[torch_node.inputs[2]]

        if not dims:
            start_dim = 0
            end_dim = -1
            length = 1
            for i in shape:
                length *= i

            slices_new = []

            new_FlattenOp = top.FlattenOp(self.unranked_type,
                                          in_op,
                                          start_dim=start_dim,
                                          end_dim=end_dim,
                                          loc=self.get_loc(
                                              torch_node.name + "_expand_0"),
                                          ip=self.mlir.insert_point).output

            new_scliceop_0 = top.SliceOp(self.unranked_type,
                                         new_FlattenOp,
                                         self.mlir.none_op,
                                         self.mlir.none_op,
                                         self.mlir.none_op,
                                         offset=[
                                             length - (shifts[0] % length)],
                                         steps=[1],
                                         ends=[length],
                                         axes=[0],
                                         loc=self.get_loc(
                                             torch_node.name + "_expand_1"),
                                         ip=self.mlir.insert_point).output

            new_scliceop_1 = top.SliceOp(self.unranked_type,
                                         new_FlattenOp,
                                         self.mlir.none_op,
                                         self.mlir.none_op,
                                         self.mlir.none_op,
                                         offset=[0],
                                         steps=[1],
                                         ends=[length - (shifts[0] % length)],
                                         axes=[0],
                                         loc=self.get_loc(
                                             torch_node.name + "_expand_2"),
                                         ip=self.mlir.insert_point).output
            slices_new.append(new_scliceop_0)
            slices_new.append(new_scliceop_1)

            new_concat_op = top.ConcatOp(self.unranked_type,
                                         slices_new,
                                         axis=0,
                                         loc=self.get_loc(
                                             torch_node.name + "_expand_3"),
                                         ip=self.mlir.insert_point).output
            new_op = top.ReshapeOp(self.unranked_type,
                                   new_concat_op,
                                   shape=shape,
                                   loc=self.get_loc(torch_node.name),
                                   ip=self.mlir.insert_point).output
            self.addOperand(torch_node.name, new_op)

        else:
            assert len(shifts) == len(dims)
            add_op = None
            cur_in_op = in_op
            idx = 0
            for dim, shift in zip(dims, shifts):
                len_shape = len(shape)
                offset_0 = [0] * len_shape
                steps_0 = [1] * len_shape
                axes_0 = list(range(0, len_shape, 1))
                offset_0[dim] = shape[dim] - (shift % shape[dim])
                slices_new = []
                new_scliceop_0 = top.SliceOp(self.unranked_type,
                                             cur_in_op,
                                             self.mlir.none_op,
                                             self.mlir.none_op,
                                             self.mlir.none_op,
                                             offset=offset_0,
                                             steps=steps_0,
                                             ends=shape,
                                             axes=axes_0,
                                             loc=self.get_loc(torch_node.name
                                                              + "_expand0" + str(dim) + str(shift)),
                                             ip=self.mlir.insert_point).output

                offset_1 = [0] * len_shape
                steps_1 = steps_0
                axes_1 = axes_0
                ends_1 = shape.copy()
                ends_1[dim] = ends_1[dim] - (shift % ends_1[dim])

                new_scliceop_1 = top.SliceOp(self.unranked_type,
                                             cur_in_op,
                                             self.mlir.none_op,
                                             self.mlir.none_op,
                                             self.mlir.none_op,
                                             offset=offset_1,
                                             steps=steps_1,
                                             ends=ends_1,
                                             axes=axes_1,
                                             loc=self.get_loc(torch_node.name
                                                              + "_expand1" + str(dim) + str(shift)),
                                             ip=self.mlir.insert_point).output
                slices_new.append(new_scliceop_0)
                slices_new.append(new_scliceop_1)
                if idx == len(dims) - 1:
                    new_concat_op = top.ConcatOp(self.unranked_type,
                                                 slices_new,
                                                 axis=dim,
                                                 loc=self.get_loc(
                                                     torch_node.name),
                                                 ip=self.mlir.insert_point).output
                    add_op = new_concat_op
                    break
                idx += 1
                new_concat_op = top.ConcatOp(self.unranked_type,
                                             slices_new,
                                             axis=dim,
                                             loc=self.get_loc(torch_node.name
                                                              + "_expand2" + str(dim) + str(shift)),
                                             ip=self.mlir.insert_point).output
                cur_in_op = new_concat_op
                add_op = new_concat_op

            self.addOperand(torch_node.name, add_op)

    def convert_stack_op(self, torch_node: TorchNode):
        inputs = self.tensor_list[torch_node.inputs[0]]
        axis = self.const_val[torch_node.inputs[1]]
        inputs_new = []
        for idx, ins in enumerate(inputs):
            input = self.getOp(ins)
            new_op = top.UnsqueezeOp(self.unranked_type,
                                     input, [axis],
                                     loc=self.get_loc(
                                         torch_node.name + "_expand_" + str(idx)),
                                     ip=self.mlir.insert_point).output
            inputs_new.append(new_op)
        if len(inputs_new) == 1:
            self.addOperand(torch_node.name, inputs_new[0])
        else:
            new_op = top.ConcatOp(self.unranked_type,
                                  inputs_new,
                                  axis=axis,
                                  loc=self.get_loc(torch_node.name),
                                  ip=self.mlir.insert_point).output
            self.addOperand(torch_node.name, new_op)

    def convert_flatten_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        start_dim = 0
        end_dim = -1
        if len(torch_node.inputs) > 1:
            start_dim = self.const_val[torch_node.inputs[1]]
        if len(torch_node.inputs) > 2:
            end_dim = self.const_val[torch_node.inputs[2]]

        new_op = top.FlattenOp(self.unranked_type,
                               op,
                               start_dim=start_dim,
                               end_dim=end_dim,
                               loc=self.get_loc(torch_node.name),
                               ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_scatter_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        op1 = self.getOp(torch_node.inputs[2])
        op2 = self.getOp(torch_node.inputs[3])
        axis = self.const_val[torch_node.inputs[1]]
        p = {'loc': self.get_loc(torch_node.name),
             'axis': axis, "reduction": None}
        new_op = top.ScatterElementsOp(
            self.unranked_type,
            op0,
            op1,
            op2,
            axis=axis,
            #    reduction=None, # unexpected param ignored
            loc=self.get_loc(torch_node.name),
            ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_select_op(self, torch_node: TorchNode):
        step_name = torch_node.inputs[0] + '_tpu_step'
        end_name = torch_node.inputs[0] + torch_node.inputs[2] + '_tpu_end'
        self.addWeight(step_name, np.array([1], dtype=np.float32))
        assert torch_node.inputs[2] in self.const_val.keys()
        end = self.const_val[torch_node.inputs[2]] + 1
        self.addWeight(end_name, np.array([end], dtype=np.float32))
        slice_op = top.SliceAxisOp(self.unranked_type,
                                   self.getOp(torch_node.inputs[0]),
                                   self.getOp(torch_node.inputs[1]),
                                   self.getOp(torch_node.inputs[2]),
                                   self.getOp(step_name),
                                   self.getOp(end_name),
                                   loc=self.get_loc(
                                       "{}_SliceAxis".format(torch_node.name)),
                                   ip=self.mlir.insert_point).output
        axis = self.const_val[torch_node.inputs[1]]
        new_op = top.SqueezeOp(self.unranked_type,
                               slice_op,
                               axes=[axis],
                               loc=self.get_loc(torch_node.name),
                               ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_slice_op(self, torch_node: TorchNode):
        new_op = top.SliceAxisOp(self.unranked_type,
                                 self.getOp(torch_node.inputs[0]),
                                 self.getOp(torch_node.inputs[1]),
                                 self.getOp(torch_node.inputs[2]),
                                 self.getOp(torch_node.inputs[4]),
                                 self.getOp(torch_node.inputs[3]),
                                 loc=self.get_loc(torch_node.name),
                                 ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_split_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        axis = self.const_val[torch_node.inputs[2]]
        split_size = self.const_val[torch_node.inputs[1]]
        output_names = self.list_map[torch_node.outputs[0]]
        if isinstance(split_size, int):
            num = len(output_names)
            split_size = [split_size] * num
        else:
            num = len(split_size)

        new_op = top.SplitOp([self.unranked_type] * num,
                             op0,
                             axis,
                             num,
                             split_size=split_size,
                             loc=self.get_loc(output_names),
                             ip=self.mlir.insert_point).outputs
        for i in range(num):
            self.addOperand(output_names[i], new_op[i])
        self.tensor_list[torch_node.outputs[0]] = output_names

    def convert_upsample_op(self, torch_node: TorchNode, mode: str):
        op0 = self.getOp(torch_node.inputs[0])
        out_size_is_const = torch_node.inputs[1] in self.const_val.keys()
        out_size_is_tensor = torch_node.inputs[1] in self.tensor_list
        has_out_size = out_size_is_const or out_size_is_tensor
        out_size = None
        if has_out_size:
            if out_size_is_tensor:
                out_size = self.getOperand(torch_node.inputs[1])
            else:
                self.addWeight(torch_node.name + "_target_shape",
                               np.array(self.const_val[torch_node.inputs[1]], dtype=np.int64))
                out_size = self.getWeightOp(torch_node.name + "_target_shape")

        align_corners = False
        if mode == "nearest":
            scale = self.const_val[torch_node.inputs[2]
                                   ] if not has_out_size else [-1, -1]
        elif mode == "bilinear":
            mode = "linear"
            align_corners = self.const_val[torch_node.inputs[2]]
            scale = self.const_val[torch_node.inputs[3]
                                   ] if not has_out_size else [-1, -1]
        elif mode == "linear":
            align_corners = self.const_val[torch_node.inputs[2]]
            scale = [1] + self.const_val[torch_node.inputs[3]
                                         ] if not has_out_size else [-1, -1]
        coord_mode = "align_corners" if align_corners else "pytorch_half_pixel"
        new_op = top.InterpOp(self.unranked_type,
                              op0,
                              out_size if has_out_size else self.mlir.none_op,
                              mode=StringAttr.get(mode),
                              coord_mode=StringAttr.get(coord_mode),
                              scale_h=scale[0],
                              scale_w=scale[1],
                              loc=self.get_loc(torch_node.name),
                              ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_squeeze_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        axes = [self.const_val[torch_node.inputs[1]]] if len(
            torch_node.inputs) == 2 else []
        new_op = top.SqueezeOp(self.unranked_type,
                               op0,
                               axes=axes,
                               loc=self.get_loc(torch_node.name),
                               ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_unsqueeze_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        axis = self.const_val[torch_node.inputs[1]]
        new_op = top.UnsqueezeOp(self.unranked_type,
                                 op0,
                                 axes=[axis],
                                 loc=self.get_loc(torch_node.name),
                                 ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_where_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        x_is_const = torch_node.inputs[1] in self.const_val.keys()
        y_is_const = torch_node.inputs[2] in self.const_val.keys()
        x_const_val = self.const_val[torch_node.inputs[1]] if x_is_const else 0
        y_const_val = self.const_val[torch_node.inputs[2]] if y_is_const else 0

        op1 = self.getOp(
            torch_node.inputs[1]) if not x_is_const else self.mlir.none_op
        op2 = self.getOp(
            torch_node.inputs[2]) if not y_is_const else self.mlir.none_op
        new_op = top.WhereOp(self.unranked_type,
                             op0,
                             op1,
                             op2,
                             x_is_const=x_is_const,
                             y_is_const=y_is_const,
                             x_const_val=x_const_val,
                             y_const_val=y_const_val,
                             loc=self.get_loc(torch_node.name),
                             ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_instance_norm_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        weight = self.getOp(torch_node.inputs[1])
        bias = self.getOp(torch_node.inputs[2])
        eps = self.const_val[torch_node.inputs[7]]
        out = top.InstanceNormOp(self.unranked_type,
                                 op0,
                                 weight,
                                 bias,
                                 eps=eps,
                                 loc=self.get_loc(torch_node.name),
                                 ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, out)

    def convert_batch_norm_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        weight = self.getOp(torch_node.inputs[1])
        bias = self.getOp(torch_node.inputs[2])
        mean = self.getOp(torch_node.inputs[3])
        var = self.getOp(torch_node.inputs[4])
        eps = self.const_val[torch_node.inputs[7]]
        out = top.BatchNormOp(self.unranked_type,
                              op0,
                              mean,
                              var,
                              weight,
                              bias,
                              epsilon=eps,
                              loc=self.get_loc(torch_node.name),
                              ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, out)

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
        out = top.LayerNormOp(self.unranked_type,
                              op0,
                              scale_opd,
                              bias_opd,
                              normalized_shape=normalized_shape,
                              axis=-len(normalized_shape),
                              eps=eps,
                              loc=self.get_loc([
                                  torch_node.name, torch_node.name + "_Mean",
                                  torch_node.name + "_Rstd"
                              ]),
                              ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, out)

    def convert_concat_op(self, torch_node: TorchNode):
        operands = list()
        for name in self.tensor_list[torch_node.inputs[0]]:
            op = self.getOp(name)
            operands.append(op)
        axis = self.const_val[torch_node.inputs[1]]
        new_op = top.ConcatOp(self.unranked_type,
                              operands,
                              axis=axis,
                              loc=self.get_loc(torch_node.name),
                              ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_chunk_op(self, torch_node: TorchNode):
        num = self.const_val[torch_node.inputs[1]]
        axis = self.const_val[torch_node.inputs[2]]
        op0 = self.getOp(torch_node.inputs[0])
        tensors = []
        for i in range(num):
            out = "{}_{}".format(torch_node.name, i)
            tensors.append(out)
        new_ops = top.SplitOp([self.unranked_type] * num,
                              op0,
                              axis=axis,
                              num=num,
                              loc=self.get_loc(tensors),
                              ip=self.mlir.insert_point).outputs
        new_ops = [new_ops] if num == 1 else new_ops
        for i in range(num):
            self.addOperand(tensors[i], new_ops[i])
        self.tensor_list[torch_node.outputs[0]] = tensors

    def convert_clamp_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        min_val = self.getOp(torch_node.inputs[1])
        max_val = self.getOp(torch_node.inputs[2])
        if self.mlir.none_op == min_val:
            new_op_max = op
        else:
            new_op_max = top.MaxOp(self.unranked_type,
                                [op, min_val],
                                loc=self.get_loc("{}_{}_{}".format(torch_node.name, torch_node.op_type, "max")),
                                ip=self.mlir.insert_point).output
        if self.mlir.none_op == max_val:
            new_op_min = new_op_max
        else:
            new_op_min = top.MinOp(self.unranked_type,
                                [new_op_max, max_val],
                                loc=self.get_loc("{}_{}_{}".format(torch_node.name, torch_node.op_type, "min")),
                                ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op_min)

    def convert_clone_op(self, torch_node: TorchNode):
        # clone = identity
        op = self.getOp(torch_node.inputs[0])
        # remove the clone node
        self.addOperand(torch_node.name, op)

    def convert_relu_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = top.ReluOp(self.unranked_type,
                            op,
                            loc=self.get_loc(torch_node.name),
                            ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_elu_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        alpha = self.const_val[torch_node.inputs[1]]
        new_op = top.EluOp(self.unranked_type,
                           op,
                           alpha=alpha,
                           loc=self.get_loc(torch_node.name),
                           ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_pad_op(self, torch_node: TorchNode, mode: str = 'unknown'):
        op = self.getOp(torch_node.inputs[0])
        pads = self.get_input_by_name(torch_node.inputs[1])
        val = 0.0
        if (mode == 'reflect' or mode == 'replicate'):
            pass
        elif (mode == 'constant'):
            val = self.const_val[torch_node.inputs[2]]
        else:
            mode = self.const_val[torch_node.inputs[2]]
            if mode == "constant":
                if torch_node.inputs[3] in self.const_val:
                    val = self.const_val[torch_node.inputs[3]]
                elif self.mlir.none_op == self.getOp(torch_node.inputs[3]):
                    val = 0
        if mode == "replicate":
            mode = "edge"
        assert (mode in ("constant", "reflect", "edge"))
        new_op = top.PadOp(self.unranked_type,
                           op,
                           paddings=pads,
                           mode=StringAttr.get(mode),
                           val=val,
                           loc=self.get_loc(torch_node.name),
                           ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_hardsigmoid(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = top.HardSigmoidOp(self.unranked_type,
                                   op,
                                   alpha=1 / 6,
                                   beta=1 / 2,
                                   loc=self.get_loc(torch_node.name),
                                   ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_hardswish(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = top.HardSwishOp(self.unranked_type,
                                 op,
                                 loc=self.get_loc(torch_node.name),
                                 ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_hardtanh(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        min_val = self.const_val[torch_node.inputs[1]]
        max_val = self.const_val[torch_node.inputs[2]]
        assert (min_val == 0 and max_val == 6 and "Only support relu6 for now")
        new_op = top.ReluOp(self.unranked_type,
                            op,
                            relu_limit=max_val,
                            loc=self.get_loc(torch_node.name),
                            ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_tuple(self, torch_node: TorchNode):
        ops = [self.getOp(n) for n in torch_node.inputs]
        new_op = top.TupleOp(self.unranked_type,
                             ops,
                             loc=self.get_loc(torch_node.name),
                             ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_list_unpack(self, torch_node: TorchNode):
        tensors = self.tensor_list[torch_node.inputs[0]]
        for out, tensor in zip(torch_node.outputs, tensors):
            if (self.isWeight(tensor)):
                self.addWeight(out, self.getWeight(tensor))
            else:
                self.addOperand(out, self.getOperand(tensor))

    def convert_tuple_unpack(self, torch_node: TorchNode):
        ops = [self.getOp(n) for n in torch_node.inputs]
        num = len(torch_node.outputs)
        assert (num == len(torch_node.outputs))
        shape = [self.unranked_type] * num
        out_ops = top.UnTupleOp(shape,
                                ops,
                                loc=self.get_loc(torch_node.outputs),
                                ip=self.mlir.insert_point).outputs
        for out, op in zip(torch_node.outputs, out_ops):
            self.addOperand(out, op)

    def convert_gelu_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = top.GELUOp(self.unranked_type,
                            op,
                            loc=self.get_loc(torch_node.name),
                            ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_erf_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = top.ErfOp(self.unranked_type,
                           op,
                           loc=self.get_loc(torch_node.name),
                           ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_masked_fill(self, torch_node: TorchNode):
        x = self.getOp(torch_node.inputs[0])
        mask = self.getOp(torch_node.inputs[1])
        const_val = self.const_val[torch_node.inputs[2]]
        new_op = top.MaskedFillOp(self.unranked_type,
                                  mask,
                                  x,
                                  inversed=True,
                                  const_val=const_val,
                                  loc=self.get_loc(torch_node.name),
                                  ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_matmul_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        op1 = self.getOp(torch_node.inputs[1])
        new_op = top.MatMulOp(self.unranked_type,
                              op0,
                              op1,
                              self.mlir.none_op,
                              do_relu=False,
                              loc=self.get_loc(torch_node.name),
                              ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_linear_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        op2 = self.getOp(torch_node.inputs[2])
        r_trans = False
        if self.isWeight(torch_node.inputs[1]):
            data = self.getWeight(torch_node.inputs[1])
            num_dims = len(data.shape)
            if num_dims == 1:
                op1 = self.getWeightOp(torch_node.inputs[1])
            else:
                order = list(range(num_dims))
                order[-1], order[-2] = order[-2], order[-1]
                data = np.ascontiguousarray(data.transpose(*order))
                new_weight = torch_node.name + "_filter"
                self.addWeight(new_weight, data)
                op1 = self.getWeightOp(new_weight)
        else:
            op1 = self.getOperand(torch_node.inputs[1])
            r_trans = True
        new_op = top.MatMulOp(self.unranked_type,
                              op0,
                              op1,
                              op2,
                              do_relu=False,
                              right_transpose=r_trans,
                              loc=self.get_loc(torch_node.name),
                              ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_sigmoid_op(self, torch_node: TorchNode, log: bool = False):
        op = self.getOp(torch_node.inputs[0])
        new_op = top.SigmoidOp(self.unranked_type,
                               op,
                               scale=1,
                               bias=0,
                               log=log,
                               loc=self.get_loc(torch_node.name),
                               ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_softplus_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = top.SoftplusOp(self.unranked_type,
                                op,
                                loc=self.get_loc(torch_node.name),
                                ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_leaky_relu_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        alpha = self.const_val[torch_node.inputs[1]]
        new_op = top.LeakyReluOp(self.unranked_type,
                                 op,
                                 alpha=alpha,
                                 loc=self.get_loc(torch_node.name),
                                 ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_linalg_norm_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        ord = self.const_val[torch_node.inputs[1]]
        if ord != 'fro':
            raise ValueError("Not Implemented")
        axes = []
        if torch_node.inputs[2] in self.const_val:
            axes = self.const_val[torch_node.inputs[2]]
        keepdims = self.const_val[torch_node.inputs[3]]
        new_op = top.ReduceOp(self.unranked_type,
                              op,
                              axes,
                              keepdims,
                              StringAttr.get("ReduceL2"),
                              loc=self.get_loc(torch_node.name),
                              ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_frobenius_norm_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        axes = []
        if torch_node.inputs[1] in self.const_val:
            axes = self.const_val[torch_node.inputs[1]]
        keepdims = self.const_val[torch_node.inputs[2]]
        new_op = top.ReduceOp(self.unranked_type,
                              op,
                              axes,
                              keepdims,
                              StringAttr.get("ReduceL2"),
                              loc=self.get_loc(torch_node.name),
                              ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)


    def convert_silu_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = top.SiLUOp(self.unranked_type,
                            op,
                            loc=self.get_loc(torch_node.name),
                            ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_channel_shuffle_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = top.ShuffleChannelOp(self.unranked_type,
                                      op,
                                      group=self.const_val[torch_node.inputs[1]],
                                      loc=self.get_loc(torch_node.name),
                                      ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_pixel_shuffle_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        block = self.const_val[torch_node.inputs[1]]
        new_op = top.Depth2SpaceOp(self.unranked_type,
                                   op,
                                   block_h=block,
                                   block_w=block,
                                   is_CRD=True,
                                   is_inversed=False,
                                   loc=self.get_loc(torch_node.name),
                                   ip=self.mlir.insert_point)
        self.addOperand(torch_node.name, new_op.output)

    def convert_pixel_unshuffle_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        block = self.const_val[torch_node.inputs[1]]
        new_op = top.Depth2SpaceOp(self.unranked_type,
                                   op,
                                   block_h=block,
                                   block_w=block,
                                   is_CRD=True,
                                   is_inversed=True,
                                   loc=self.get_loc(torch_node.name),
                                   ip=self.mlir.insert_point)
        self.addOperand(torch_node.name, new_op.output)

    def convert_softmax_op(self, torch_node: TorchNode, log: bool = False):
        op = self.getOp(torch_node.inputs[0])
        dim = self.const_val[torch_node.inputs[1]]
        new_op = top.SoftmaxOp(self.unranked_type,
                               op,
                               axis=dim,
                               log=log,
                               loc=self.get_loc(torch_node.name),
                               ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_mish_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = top.MishOp(self.unranked_type,
                            op,
                            loc=self.get_loc(torch_node.name),
                            ip=self.mlir.insert_point).output
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
        lstm_op = top.LSTMOp(self.unranked_type,
                             self.unranked_type,
                             self.unranked_type,
                             in_op,
                             filter_op,
                             recurrence_op,
                             bias_op,
                             h0_op,
                             c0_op,
                             self.mlir.none_op,
                             hidden_size=rshape[-1],
                             bidirectional=bidirectional,
                             batch_first=batch_first,
                             loc=self.get_loc(torch_node.outputs),
                             ip=self.mlir.insert_point)
        new_op, h_op, c_op = lstm_op.Y, lstm_op.Y_h, lstm_op.Y_c,
        self.addOperand(torch_node.outputs[0], new_op)
        self.addOperand(torch_node.outputs[1], h_op)
        self.addOperand(torch_node.outputs[2], c_op)

    def convert_group_norm_op(self, torch_node: TorchNode):
        in_op = self.getOp(torch_node.inputs[0])
        num_groups = self.const_val[torch_node.inputs[1]]
        weight_op = self.getOp(torch_node.inputs[2])
        bias_op = self.getOp(torch_node.inputs[3])
        eps = self.const_val[torch_node.inputs[4]]
        new_op = top.GroupNormOp(self.unranked_type,
                                 in_op,
                                 weight_op,
                                 bias_op,
                                 num_groups=num_groups,
                                 eps=eps,
                                 loc=self.get_loc(torch_node.name),
                                 ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_native_group_norm_op(self, torch_node: TorchNode):
        innput_op = self.getOp(torch_node.inputs[0])
        weight_op = self.getOp(torch_node.inputs[1])
        bias_op = self.getOp(torch_node.inputs[2])
        num_groups = self.const_val[torch_node.inputs[-2]]
        eps = self.const_val[torch_node.inputs[-1]]
        new_op = top.GroupNormTrainOp(self.unranked_type,
                                      self.unranked_type,
                                      self.unranked_type,
                                    innput_op,
                                    weight_op,
                                    bias_op,
                                    num_groups=num_groups,
                                    eps=eps,
                                    loc=self.get_loc([torch_node.name,
                                                     torch_node.name + "_mean",
                                                     torch_node.name + "_rstd"]),
                                    ip=self.mlir.insert_point)
        self.addOperand(torch_node.outputs[0], new_op.output)
        self.addOperand(torch_node.outputs[1], new_op.mean)
        self.addOperand(torch_node.outputs[2], new_op.rstd)

    def convert_native_layer_norm_op(self, torch_node: TorchNode):
        input_op = self.getOp(torch_node.inputs[0])
        scale_op = self.getOp(torch_node.inputs[2])
        bias_op = self.getOp(torch_node.inputs[3])
        normalized_shape = self.const_val[torch_node.inputs[1]]
        eps = self.const_val[torch_node.inputs[4]]
        axis = -len(normalized_shape)
        new_op = top.LayerNormTrainOp(self.unranked_type,
                                    self.unranked_type,
                                    self.unranked_type,
                                    input_op,
                                    scale_op,
                                    bias_op,
                                    normalized_shape=normalized_shape,
                                    axis=axis,
                                    eps=eps,
                                    loc=self.get_loc([
                                        torch_node.name,
                                        torch_node.name + "_Mean",
                                        torch_node.name + "_Rstd"
                                    ]),
                                    ip=self.mlir.insert_point)
        self.addOperand(torch_node.outputs[0], new_op.output)
        self.addOperand(torch_node.outputs[1], new_op.mean)
        self.addOperand(torch_node.outputs[2], new_op.variance)

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
        gru_op = top.GRUOp(self.unranked_type,
                           self.unranked_type,
                           in_op,
                           filter_op,
                           recurrence_op,
                           bias_op,
                           h0_op,
                           hidden_size=rshape[-1],
                           bidirectional=bidirectional,
                           batch_first=batch_first,
                           loc=self.get_loc(torch_node.outputs),
                           ip=self.mlir.insert_point)
        new_op, h_op = gru_op.Y, gru_op.Y_h
        self.addOperand(torch_node.outputs[0], new_op)
        self.addOperand(torch_node.outputs[1], h_op)

    def convert_abs_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = top.AbsOp(self.unranked_type,
                           op,
                           loc=self.get_loc(torch_node.name),
                           ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_neg_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = top.MulConstOp(self.unranked_type,
                                op,
                                const_val=-1,
                                loc=self.get_loc(torch_node.name),
                                ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_grid_sampler_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        op1 = self.getOp(torch_node.inputs[1])
        mode = self.const_val[torch_node.inputs[2]]
        padding_mode = self.const_val[torch_node.inputs[3]]
        align_corners = self.const_val[torch_node.inputs[4]]
        new_op = top.GridSamplerOp(self.unranked_type,
                                   op0,
                                   op1,
                                   mode=mode,
                                   padding_mode=padding_mode,
                                   align_corners=align_corners,
                                   loc=self.get_loc(torch_node.name),
                                   ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_cumsum_op(self, torch_node: TorchNode):
        if not self.isWeight(torch_node.inputs[1]):
            raise ValueError("Currently, only constant axis is supported")
        axis = self.getWeight(torch_node.inputs[1])
        operands = list()
        operands.append(self.getOperand(torch_node.inputs[0]))
        operands.append(self.getWeightOp(torch_node.inputs[1]))
        new_op = top.CumSumOp(self.unranked_type, *operands,
                              axis=axis,
                              loc=self.get_loc("{}_{}".format(torch_node.name, torch_node.op_type)),
                              ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_deform_conv2d_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        operands = list()
        kernel_shape = self.getShape(torch_node.inputs[1])
        kernel_shape = kernel_shape[2:]
        weight_op = self.getOp(torch_node.inputs[1])
        offset_op = self.getOp(torch_node.inputs[2])
        mask_op = self.getOp(torch_node.inputs[3])
        bias_op = self.getOp(torch_node.inputs[4])
        stride_h = self.const_val[torch_node.inputs[5]]
        stride_w = self.const_val[torch_node.inputs[6]]
        pad_h = self.const_val[torch_node.inputs[7]]
        pad_w = self.const_val[torch_node.inputs[8]]
        dilation_h = self.const_val[torch_node.inputs[9]]
        dilation_w = self.const_val[torch_node.inputs[10]]
        n_weight_grps = self.const_val[torch_node.inputs[11]]
        n_offset_grps = self.const_val[torch_node.inputs[12]]
        use_mask = self.const_val[torch_node.inputs[13]]
        if (use_mask == 0):
            mask_op = self.mlir.none_op
        new_op = top.DeformConv2DOp(self.unranked_type,
                                    op,
                                    weight_op,
                                    offset_op,
                                    mask_op,
                                    bias_op,
                                    kernel_shape=kernel_shape,
                                    strides=[stride_h, stride_w],
                                    pads=[pad_h, pad_w, pad_h, pad_w],
                                    group=n_weight_grps,
                                    deform_group=n_offset_grps,
                                    use_mask=use_mask,
                                    do_relu=False,
                                    loc=self.get_loc(torch_node.name),
                                    ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_ceil_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        new_op = top.CeilOp(self.unranked_type,
                            op,
                            loc=self.get_loc(torch_node.name),
                            ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_rsqrt_op(self, torch_node: TorchNode):
        # rsqrt = sqrt + reciprocal (const_val = 1.0)
        op = self.getOp(torch_node.inputs[0])
        sqrt_name = torch_node.name + "_sqrt"
        sqrt_op = top.SqrtOp(self.unranked_type,
                             op,
                             loc=self.get_loc(sqrt_name),
                             ip=self.mlir.insert_point).output
        self.addOperand(sqrt_name, sqrt_op)
        reciprocal_op = top.ReciprocalOp(self.unranked_type,
                                         sqrt_op,
                                         loc=self.get_loc(torch_node.name),
                                         ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, reciprocal_op)

    def convert_remainder_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        op1 = self.getOp(torch_node.inputs[1])
        new_op = top.RemainderOp(self.unranked_type, [op0, op1],
                                 loc=self.get_loc(torch_node.name),
                                 ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_dict_construct(self, torch_node: TorchNode):
        pass

    def convert_baddbmm_op(self, torch_node: TorchNode):
        """baddbmm: val4*op0 + val3*op1@op2"""
        op0 = self.getOp(torch_node.inputs[0])
        op1 = self.getOp(torch_node.inputs[1])
        op2 = self.getOp(torch_node.inputs[2])
        val4 = self.const_val[torch_node.inputs[3]]
        val3 = self.const_val[torch_node.inputs[4]]
        if val4 == 0 and val3 == 0:  # only zero is need
            new_op3 = top.MulConstOp(self.unranked_type,
                                     op0,
                                     val4,
                                     loc=self.get_loc(torch_node.name+"_mul2"),
                                     ip=self.mlir.insert_point).output
            self.addOperand(torch_node.name, new_op3)
            return
        elif val3 == 0:  # only alpha*op0 is need
            new_op3 = top.MulConstOp(self.unranked_type,
                                     op0,
                                     val4,
                                     loc=self.get_loc(torch_node.name+"_mul2"),
                                     ip=self.mlir.insert_point).output
            self.addOperand(torch_node.name, new_op3)
            return
        elif val4 == 0:  # only beta*op1*op2 is need
            new_op = top.MatMulOp(self.unranked_type,
                                  op1,
                                  op2,
                                  self.mlir.none_op,
                                  loc=self.get_loc(torch_node.name+"_matmul"),
                                  ip=self.mlir.insert_point).output
            new_op2 = top.MulConstOp(self.unranked_type,
                                     new_op,
                                     val3,
                                     loc=self.get_loc(torch_node.name+"_mul"),
                                     ip=self.mlir.insert_point).output
            self.addOperand(torch_node.name, new_op2)
            return
        elif val4 == 1 and val3 == 1:  # only op1*op2 + op0 is need
            new_op = top.MatMulOp(self.unranked_type,
                                  op1,
                                  op2,
                                  self.mlir.none_op,
                                  loc=self.get_loc(torch_node.name+"_matmul"),
                                  ip=self.mlir.insert_point).output
            new_op2 = top.AddOp(self.unranked_type, [new_op, op0],
                                do_relu=False,
                                loc=self.get_loc(torch_node.name+"_add"),
                                ip=self.mlir.insert_point).output
            self.addOperand(torch_node.name, new_op2)
            return
        new_op = top.MatMulOp(self.unranked_type,
                              op1,
                              op2,
                              self.mlir.none_op,
                              loc=self.get_loc(torch_node.name+"_matmul"),
                              ip=self.mlir.insert_point).output
        # new_op2: new_op * op3
        new_op2 = top.MulConstOp(self.unranked_type,
                                 new_op,
                                 val3,
                                 loc=self.get_loc(torch_node.name+"_mul"),
                                 ip=self.mlir.insert_point).output
        # new_op3: op4 * op0
        new_op3 = top.MulConstOp(self.unranked_type,
                                 op0,
                                 val4,
                                 loc=self.get_loc(torch_node.name+"_mul2"),
                                 ip=self.mlir.insert_point).output
        # new_op4: new_op2 + new_op3
        new_op4 = top.AddOp(self.unranked_type, [new_op2, new_op3],
                            do_relu=False,
                            loc=self.get_loc(torch_node.name+"_add"),
                            ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op4)

    def convert_roi_align_op(self, torch_node: TorchNode):
        input = self.getOp(torch_node.inputs[0])
        rois = self.getOp(torch_node.inputs[1])
        spatial_scale = self.const_val[torch_node.inputs[2]]
        output_height = self.const_val[torch_node.inputs[3]]
        output_width = self.const_val[torch_node.inputs[4]]
        sampling_ratio = self.const_val[torch_node.inputs[5]]
        align_corners = self.const_val[torch_node.inputs[6]]
        mode = "Avg"
        new_op = top.RoiAlignOp(self.unranked_type,
                                input,
                                rois,
                                mode=StringAttr.get(mode),
                                output_height=output_height,
                                output_width=output_width,
                                sampling_ratio=sampling_ratio,
                                spatial_scale=spatial_scale,
                                align_corners=align_corners,
                                loc=self.get_loc(torch_node.name),
                                ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_log2_op(self, torch_node: TorchNode):
        input = self.getOp(torch_node.inputs[0])
        new_op = top.LogBOp(self.unranked_type,
                            input,
                            2,
                            loc=self.get_loc(torch_node.name),
                            ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_log_op(self, torch_node: TorchNode):
        input = self.getOp(torch_node.inputs[0])
        new_op = top.LogOp(self.unranked_type,
                           input,
                           loc=self.get_loc(torch_node.name),
                           ip=self.mlir.insert_point).output
        self.addOperand(torch_node.name, new_op)

    def convert_sort_op(self, torch_node: TorchNode):
        input = self.getOp(torch_node.inputs[0])
        input_shape = self.getShape(torch_node.inputs[0])
        axis = self.const_val[torch_node.inputs[-2]]
        descending = self.const_val[torch_node.inputs[-1]]
        if len(torch_node.inputs) == 4:
            stable = self.const_val[torch_node.inputs[1]]
        else:
            stable = False
        new_op = top.TopKOp(self.unranked_type,
                            self.unranked_type,
                            input=input,
                            axis=axis,
                            K=input_shape[axis],
                            kT=None,
                            largest=descending,
                            sorted=True,
                            loc=self.get_loc(torch_node.outputs),
                            ip=self.mlir.insert_point)
        values = new_op.values
        indices = new_op.indices
        self.addOperand(torch_node.outputs[0], values)
        self.addOperand(torch_node.outputs[1], indices)

    def convert_topk_op(self, torch_node: TorchNode):
        input = self.getOp(torch_node.inputs[0])
        K = self.const_val[torch_node.inputs[1]]
        axis = self.const_val[torch_node.inputs[2]]
        largest = self.const_val[torch_node.inputs[3]]
        sorted = self.const_val[torch_node.inputs[4]]
        new_op = top.TopKOp(self.unranked_type,
                            self.unranked_type,
                            input=input,
                            axis=axis,
                            K=K,
                            kT=None,
                            largest=largest,
                            sorted=sorted,
                            loc=self.get_loc(torch_node.outputs),
                            ip=self.mlir.insert_point)
        values = new_op.values
        indices = new_op.indices
        self.addOperand(torch_node.outputs[0], values)
        self.addOperand(torch_node.outputs[1], indices)


    def convert_unbind_unpack(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        axis = self.const_val[torch_node.inputs[1]]
        split_size = 1
        output_names = self.list_map[torch_node.outputs[0]]
        num = len(output_names)
        split_size = [split_size] * num
        new_op = top.SplitOp([self.unranked_type] * num,
                             op0,
                             axis,
                             num,
                             split_size=split_size,
                             loc=self.get_loc(output_names),
                             ip=self.mlir.insert_point).outputs
        for i in range(num):
            self.addOperand(output_names[i], new_op[i])
        self.tensor_list[torch_node.outputs[0]] = output_names

    def convert_arg_op(self, torch_node: TorchNode):
        op = self.getOp(torch_node.inputs[0])
        axis = self.const_val[torch_node.inputs[1]]
        keepdims = self.const_val[torch_node.inputs[2]]
        assert (torch_node.op_type in ["aten::argmax", "aten::argmin"])
        if torch_node.op_type == 'aten::argmax':
            arg_type = 'ArgMax'
        else:
            arg_type = 'ArgMin'
        loc_names = ["{}_{}".format(torch_node.outputs[0], arg_type)]
        out_op = top.ArgOp(*self.mlir.get_tensor_type([[], None]),
                           op,
                           axis=axis,
                           keepdims=keepdims,
                           mode=StringAttr.get(arg_type),
                           select_last_index=False,
                           loc=self.get_loc(loc_names),
                           ip=self.mlir.insert_point)
        self.addOperand(torch_node.outputs[0], out_op.indices)
