import os
import copy
import sys
import torch
import pdb
import gc
import numpy as np
import operator
import importlib
from argparse import Namespace

MIN_BLOCK_SIZE = 5
import tpu_mlir
from mlir.ir import *
import mlir.dialects.top as top
from tools.train.TpuMlirModule import TpuMlirModule
from utils.mlir_shell import mlir_opt_for_top, mlir_lowering, mlir_to_model, f32_blobs_compare
from tools.model_runner import mlir_inference, free_mlir_module, model_inference
from tools.gen_rand_input import run_from_mlir
from numpy_helper.npz_compare import npz_compare
from tools.train.FxMlirImporter import FxMIIRImporter
from . import config


def NoneAndRaise(node):
    raise RuntimeError("{} Op not support now".format(node.target))


def extrace_op_name(node):
    res = ""
    base = torch.typename(node.target).split("::")[-1]
    if ".Tensor" in base:
        res = base.split('.')[0]
    elif ".dim" in base:
        res = base.split(".")[0]
    elif "Scalar" in base:
        res = base.split(".")[0]
    elif "self" in base:
        res = base.split(".")[0]
    elif "value" in base:
        res = base.split(".")[0]
    elif "start" in base:
        res = base.split(".")[0]
    elif "memory_format" in base:
        res = base.split(".")[0]
    elif ".int" in base:
        res = base.split(".")[0]
    else:
        res = base.split(".")[-1]
    # print(base, res)
    return res


def convert_to_nCHW(weight_nIC, oc, ic, kh, kw, N):
    n = (ic + N - 1) // N
    weight = np.zeros((oc, ic, kh, kw), dtype=weight_nIC.dtype)

    for i in range(oc):
        for j in range(n):
            for k in range(kh * kw):
                inner = min(ic - N * j, N)
                for l in range(inner):
                    weight[i, j * N + l, k // kw,
                           k % kw] = weight_nIC[i * n * kh * kw * N + j * kh * kw * N + k * N + l]

    return weight


class fx2mlir(object):

    def __init__(self, submodule_name: str, chip: str, bwd_graph: bool = False, cmp: bool = None):
        self.work_dir = submodule_name.split('_')[0]
        os.makedirs(os.path.join(os.getcwd(), self.work_dir), exist_ok=True)
        self.model_name = f'{submodule_name}'
        self.chip = chip
        self.bwd_graph = bwd_graph
        self.cmp = cmp if cmp is not None else config.cmp
        self.bmodel_path = None
        self.weight_file = f'graph_for_jit_{self.model_name}.npz'
        self.output_nodes = []
        self.all_output_dtypes = []
        self.operands = dict()
        self.name_map = dict()
        self.init_fxmlirimporter()

        self.processed_op = []
        self.pattern_matchers = {
            "sigmoid/empty_like/fill":
            [lambda node: self.sigmoid_matcher(node), lambda node: self.sigmoid_folder(node)],
            "empty/baddbmm": [
                lambda node: self.empty_baddbmm_matcher(node),
                lambda node: self.empty_baddbmm_folder(node)
            ],
        }

        self.op_factory = {
            #############################
            # Torch Convert, Alphabetically
            #############################
            "convolution":
            lambda node: self.convert_base_conv_op(node),
            "convolution.default":
            lambda node: self.convert_base_conv_op(node),
            "convolution_backward":
            lambda node: self.convert_backward_conv_op2(node),
            #"convolution_backward": lambda node: self.convert_backward_conv_op(node),
            "permute":
            lambda node: self.convert_permute_op(node),
            "relu":
            lambda node: self.convert_relu_op(node),
            # "native_dropout": lambda node: self.convert_dropout_op(node),
            # "native_dropout_backward": lambda node: self.convert_native_dropout_backward_op(node),
            "_adaptive_avg_pool2d":
            lambda node: self.convert_adaptive_avg_pool2d_op(node),
            "_adaptive_avg_pool2d_backward":
            lambda node: self.convert_adaptive_avg_pool2d_backward_op(node),
            "max_pool2d_with_indices":
            lambda node: self.convert_maxpool2d_with_indices_op(node),
            "max_pool2d_with_indices_backward":
            lambda node: self.convert_maxpool2d_backward_op(node),
            "add":
            lambda node: self.convert_add_op(node),
            "mul":
            lambda node: self.convert_mul_op(node),
            "view":
            lambda node: self.convert_reshape_op(node),
            "_unsafe_view":
            lambda node: self.convert_reshape_op(node),
            "reshape":
            lambda node: self.convert_reshape_op(node),
            "mm":
            lambda node: self.convert_matmul_op(node),
            "bmm":
            lambda node: self.convert_matmul_op(node),
            "matmul":
            lambda node: self.convert_matmul_op(node),
            "squeeze":
            lambda node: self.convert_squeeze_op(node),
            "unsqueeze":
            lambda node: self.convert_unsqueeze_op(node),
            "getitem":
            lambda node: self.convert_getitem_op(node),
            "to":
            lambda node: self.convert_to_op(node),
            "cat":
            lambda node: self.convert_concat_op(node),
            "sum":
            lambda node: self.convert_sum_op(node),
            "mean":
            lambda node: self.convert_mean_op(node),
            "clone":
            lambda node: self.convert_clone_op(node),
            "lift_fresh_copy":
            lambda node: self.convert_fresh_copy_op(node),
            "detach":
            lambda node: self.convert_clone_op(node),
            "_native_batch_norm_legit_functional":
            lambda node: self.convert_batch_norm_op(node),
            # "_native_batch_norm_legit_functional": lambda node: self.convert_batchnorm_decomp_op(node),
            "native_batch_norm_backward":
            lambda node: self.convert_batch_norm_backward_op(node),
            "full":
            lambda node: self.convert_full_op(node),
            "arange":
            lambda node: self.convert_arange_op(node),
            "scalar_tensor":
            lambda node: self.convert_scalar_tensor_op(node),
            "slice":
            lambda node: self.convert_slice_op(node),
            "embedding":
            lambda node: self.convert_embedding_op(node),
            "ne":
            lambda node: self.convert_compare_op(node, "NotEqual"),
            "where":
            lambda node: self.convert_where_op(node),
            "_to_copy":
            lambda node: self.convert_copy_op(node),
            "var":
            lambda node: self.convert_var_op2(node),  #var_op
            "div":
            lambda node: self.convert_div_op(node),
            "rsqrt":
            lambda node: self.convert_rsqrt_op(node),
            "sub":
            lambda node: self.convert_sub_op(node),
            "addmm":
            lambda node: self.convert_addmm_op(node),
            "split":
            lambda node: self.convert_split_op(node),
            "expand":
            lambda node: self.convert_expand_op(node),
            "amax":
            lambda node: self.convert_amax_op(node, False),
            "exp":
            lambda node: self.convert_math_op(node, 'exp'),
            "erf":
            lambda node: self.convert_math_op(node, 'erf'),
            "select":
            lambda node: self.convert_select_op(node),
            "log":
            lambda node: self.convert_math_op(node, 'log'),
            "gather":
            lambda node: self.convert_gather_op(node),
            "neg":
            lambda node: self.convert_neg_op(node),
            "t":
            lambda node: self.convert_transpose_op(node),
            "native_layer_norm":
            lambda node: self.convert_layer_norm_op(node),
            "native_layer_norm_backward":
            lambda node: self.convert_layer_norm_backward_op(node),
            "transpose":
            lambda node: self.convert_transpose_op(node),
            "_softmax":
            lambda node: self.convert_softmax_op(node, log=False),
            "_log_softmax":
            lambda node: self.convert_softmax_op(node, log=True),
            "nll_loss_forward":
            lambda node: self.convert_nllloss_op(node),
            "le":
            lambda node: self.convert_compare_op(node, "LessOrEqual"),
            "sigmoid":
            lambda node: self.convert_sigmoid_op(node),
            "silu":
            lambda node: self.convert_silu_op(node),
            "pow":
            lambda node: self.convert_pow_op(node),
            "_mseloss":
            lambda node: self.convert_mse_op(node),
            "max_pool2d_with_indices_backward":
            lambda node: self.convert_maxpool2d_backward_op(node),
            "hardswish":
            lambda node: self.convert_hardswish_op(node),
            "leaky_relu":
            lambda node: self.convert_leaky_relu_op(node),
            "gt":
            lambda node: self.convert_compare_op(node, "Greater"),
            "lt":
            lambda node: self.convert_compare_op(node, "Less"),
            "new_zeros":
            lambda node: self.convert_new_constant_op(node, 0),
            # "new_zeros":lambda node:self.convert_zero_op(node),
            "rsub":
            lambda node: self.convert_sub_op(node, is_reverse=True),
            "clamp":
            lambda node: self.convert_clamp_op(node),
            "masked_fill":
            lambda node: self.convert_masked_fill_op(node),
            "index":
            lambda node: self.convert_index_op(node),
            "select_scatter":
            lambda node: self.convert_select_scatter_op(node),
            "slice_scatter":
            lambda node: self.convert_slice_scatter_op(node),
            "index_put":
            lambda node: self.convert_index_put_op(node),
            "tanh":
            lambda node: self.convert_math_op(node, "tanh"),
            "sin":
            lambda node: self.convert_math_op(node, "sin"),
            "cos":
            lambda node: self.convert_math_op(node, "cos"),
            "native_group_norm":
            lambda node: self.convert_group_norm_op(node),
            "gelu":
            lambda node: self.convert_gelu_op(node),
            # "empty_like":lambda node:self.convert_constant_like_op(node,0),
            "empty_like":
            lambda node: self.convert_new_constant_op(node, 0),
            "empty":
            lambda node: self.convert_zero_op(node),
            "fill":
            lambda node: self.convert_full_op(node),
            "constant_pad_nd":
            lambda node: self.convert_pad_op(node, 'constant'),
            "argmax":
            lambda node: self.convert_argmax_op(node),
            "zeros_like":
            lambda node: self.convert_constant_like_op(node, 0),
            "scatter":
            lambda node: self.convert_scatter_op(node),
            "logical_and":
            lambda node: self.convert_logical_and_op(node),
            "zeros":
            lambda node: self.convert_zero_op(node),
            "bernoulli":
            lambda node: self.convert_bernoulli_op(node),
            "rand_like":
            lambda node: self.convert_rand_op(node),
            "randn":
            lambda node: self.convert_randn_op(node),
            "randn_like":
            lambda node: self.convert_randn_op(node),
            "_unsafe_index_put":
            lambda node: self.convert_index_put_op(node),
            "_unsafe_index":
            lambda node: self.convert_index_op(node),
            "baddbmm":
            lambda node: self.convert_baddbmm_op(node),
            ####################################################
            "constant":
            lambda node: self.convert_constant(node),
            'threshold_backward':
            lambda node: self.convert_threshold_backward_op(node),
            '_softmax_backward_data':
            lambda node: self.convert_softmax_backward_data_op(node),
            'embedding_dense_backward':
            lambda node: self.convert_embedding_dense_backward_op(node),
            ######### prims op ##############################
            "ge":
            lambda node: self.convert_compare_op(node, "GreaterOrEqual"),
            "eq":
            lambda node: self.convert_compare_op(node, "Equal"),
            "ne":
            lambda node: self.convert_compare_op(node, "NotEqual"),
            "trunc":
            lambda node: self.convert_trunc_op(node),
            "broadcast_in_dim":
            lambda node: self.convert_broadcast_op(node),
            "ones_like":
            lambda node: self.convert_constant_like_op(node, 1),
            "split_with_sizes":
            lambda node: self.convert_split_op(node),
            "minimum":
            lambda node: self.convert_minimum_op(node),
            "maximum":
            lambda node: self.convert_maximum_op(node),
            "atan":
            lambda node: self.convert_arctan_op(node),
            "full_like":
            lambda node: self.convert_full_like_op(node),
            "clamp_min":
            lambda node: self.convert_clamp_min_op(node),
        }

    def init_fxmlirimporter(self):
        self.mlir = FxMIIRImporter(self.model_name, self.weight_file)

    def get_loc(self, names):
        if isinstance(names, str):
            return Location.fused([Location.name(names)], context=self.mlir.ctx)
        elif isinstance(names, list):
            return Location.fused([Location.name(n) for n in names], context=self.mlir.ctx)
        elif isinstance(names, torch.fx.Node):
            return Location.fused([Location.name(names.name)], context=self.mlir.ctx)
        else:
            raise RuntimeError("Unknown names:{}".format(names))

    def convert(self, module):
        print('>>>>>>>>>> starting parsing...')
        # module.to_folder(f'fx_graph_dumped_{self.model_name}', self.model_name)
        self.output_node = [i for i in module.graph.nodes
                            if i.op == 'output' and len(i.args) > 0][0]
        scalar_tensor_new_fx_graph = {}
        self.excluded_output = []
        self.input_nodes = []
        for i, node in enumerate(module.graph.nodes):
            if node.op == 'placeholder':
                shape = list(node.meta['val'].size())
                print(f'>>> {i}th op, placeholder:', node.name, 'val:', node.meta['val'])
                self.input_nodes.append(node)

        in_args_txt_list = []
        for idx, node in enumerate(self.input_nodes):
            shape = list(node.meta['val'].size())
            shape = [1] if shape == [] else shape
            in_args_txt_list.append("%arg{}: {} loc(unknown)".format(
                idx,
                RankedTensorType.get(shape, F32Type.get()).__str__()))  #F32
        first_call_op = True
        for i, node in enumerate(module.graph.nodes):
            if len(node.users) == 0:
                continue
            if node.op == 'call_module' or node.op == 'call_method' or node.op == 'call_function':
                if first_call_op:
                    output_args_txt = self.parseOutputNode(self.output_node)
                    self.mlir.createMlirModuleAndInput(self.input_nodes,
                                                       ', '.join(in_args_txt_list), output_args_txt,
                                                       self.operands)
                    first_call_op = False
                op_type = extrace_op_name(node)
                print(f'{i}th op, node.name:', node.name, 'target:',
                      node.target, 'op_type:', op_type, 'args:', node.args, 'users:',
                      list(node.users.keys()), 'output_shapes:', self.mlir.get_output_shapes(node),
                      'kwargs:', node.kwargs)
                for matcher_name in self.pattern_matchers:
                    if self.pattern_matchers[matcher_name][0](node):
                        print(f'{matcher_name} matched')
                        self.pattern_matchers[matcher_name][1](node)
                        break
                if node not in self.processed_op:
                    self.op_factory.get(op_type, lambda x: NoneAndRaise(x))(node)
                else:
                    self.processed_op.remove(node)

        # add return op
        return_op = list()
        output_tensor_names = []
        for idx, node in enumerate(self.output_nodes):
            return_op.append(self.operands[node])
            new_name = node.name
            if node in self.name_map:
                new_name = self.name_map[node]
            output_tensor_names.append(new_name)

        self.mlir.create_return_op(return_op)
        mlir_txt = self.mlir.print_module()
        self.top_mlir = f'{self.model_name}_top.mlir'
        mlir_origin = self.top_mlir.replace('.mlir', '_origin.mlir', 1)
        with open(mlir_origin, "w") as f:
            f.write(mlir_txt)
        self.mlir.WeightToNpz(self.weight_file)

        mlir_opt_for_top(mlir_origin, self.top_mlir)

        self.tpu_mlir = f"{self.model_name}_{self.chip}_f16_tpu.mlir"
        self.bmodel_path = os.path.join(self.work_dir,
                                        f"{self.model_name}_{self.chip}_f16_tpu.bmodel")
        mlir_lowering(top_mlir=self.top_mlir,
                      tpu_mlir=self.tpu_mlir,
                      mode="F16",
                      chip=self.chip,
                      num_core=config.get_num_core(self.chip))
        mlir_to_model(tpu_mlir=self.tpu_mlir,
                      bmodel_path=self.bmodel_path,
                      final_mlir=f"{self.model_name}_{self.chip}_f16_final.mlir",
                      quant_input=True,
                      quant_output=not self.bwd_graph,
                      opt=config.compile_opt,
                      debug_info=config.debug_cmd)

        if self.mlir != None:
            del self.mlir
            self.mlir = None
        if self.cmp:
            self.model_validate()
        if not config.unit_test:
            mlir_mod = TpuMlirModule(self.bmodel_path, self.output_count,
                                     scalar_tensor_new_fx_graph, output_tensor_names,
                                     self.all_output_dtypes, self.output_shapes,
                                     self.ori_output_nodes)
            return mlir_mod
        else:
            return None

    def fix(self):
        run_from_mlir(mlir=self.top_mlir, output="tmp_input.npz")
        data = np.load("tmp_input.npz")
        data2 = np.load("input_ref.npz")
        news = {}
        for key in data.files:
            print(data2[key].shape, data[key].shape)
            news[key] = data2[key].reshape(data[key].shape)
            print(key)
        np.savez("input_ref.npz", **news)

    def fix_output(self):
        data = np.load("top.npz")
        data2 = np.load("output_ref.npz")
        news = {}
        idx = 0
        for key in data.files:
            news[key] = data2[str(idx)].reshape(data[key].shape)
            idx += 1
            print(key)
        np.savez("cpu.npz", **news)

    def model_validate(self):
        # fix shapes of input tensors
        self.fix()
        # os.system(f"gen_rand_input.py  --mlir {self.top_mlir} --output input_ref.npz")
        input_ref = np.load("input_ref.npz")
        top_outs = mlir_inference(inputs=input_ref, mlir_file=self.top_mlir, dump_all=False)
        np.savez("top.npz", **top_outs)
        # fix shapes of torch cpu outputs
        self.fix_output()

        tpu_outs = mlir_inference(inputs=input_ref, mlir_file=self.tpu_mlir)
        np.savez("tpu.npz", **tpu_outs)
        model_outs = model_inference(inputs=input_ref, model_file=self.bmodel_path)
        np.savez("bmodel.npz", **model_outs)
        npz_compare(["top.npz", "cpu.npz", "--tolerance", "0.99,0.99", "-v"])
        npz_compare(["tpu.npz", "top.npz", "--tolerance", "0.99,0.99", "-v"])
        npz_compare(["bmodel.npz", "tpu.npz", "--tolerance", "0.99,0.99", "-v"])

    def parseOutputNode(self, node):
        assert node.op == 'output'
        output_nodes = node.args[0]
        if isinstance(node.args[0], torch.fx.node.Node):
            output_nodes = [node.args[0]]
        self.ori_output_nodes = list(output_nodes)

        output_nodes = [i for i in output_nodes if i is not None]
        self.output_count = len(output_nodes)

        self.all_output_dtypes = [i.meta['val'].dtype for i in output_nodes]
        print('output_nodes.size:', len(output_nodes))
        self.output_nodes = [i for i in output_nodes if i not in self.excluded_output]
        is_inputs = [
            i in self.input_nodes
            or (extrace_op_name(i) == '_to_copy' and i.args[0].op == 'placeholder')
            for i in self.output_nodes
        ]
        print('is_inputs:', is_inputs)
        self.output_shapes = [list(i.meta['val'].shape) for i in self.output_nodes]
        output_dtypes = [i.meta['val'].dtype for i in self.output_nodes]
        print('output_dtypes.size:', len(output_dtypes))

        output_txt = ','.join([
            f'{self.mlir.get_tensor_type([] if not is_in else shape, self.mlir.get_dtype(dtype)).__str__()}'
            for is_in, shape, dtype in zip(is_inputs, self.output_shapes, output_dtypes)
        ])
        if len(self.output_shapes) > 1:
            output_txt = "({})".format(output_txt)
        return output_txt

    # unranked_type = UnrankedTensorType.get(F32Type.get())
    def convert_permute_op(self, node):
        op = self.operands[node.args[0]]
        order = node.args[1]
        dtype = self.mlir.get_output_dtypes(node)
        new_op = top.PermuteOp(self.mlir.get_tensor_type([]),
                               op,
                               order=order,
                               loc=self.get_loc(node),
                               ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_relu_op(self, node):
        op = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        new_op = top.ReluOp(self.mlir.get_tensor_type([]),
                            op,
                            loc=self.get_loc(node),
                            ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_full_op(self, node):
        if node.args[0] == []:
            self.operands[node] = self.mlir.create_weight_op(f'fullOp_{node.name}_c', node.args[1])
        elif node.args[0] not in self.operands:
            self.operands[node] = self.mlir.create_constant_weight_op(f'fullOp_{node.name}_c',
                                                                      node.args[0], node.args[1])
        else:
            dtype = self.mlir.get_output_dtypes(node)
            # op0 = self.operands[node.args[0]]
            shape = list(node.args[0].meta['val'].size())
            # new_op = top.ConstantFillOp(*self.mlir.get_tensor_type(self.mlir.get_output_shapes(node), F32Type.get()),
            #                             op0,
            #                             value=node.args[1],
            #                             loc=self.get_loc(node),
            #                             ip=self.mlir.insert_point).output
            # self.operands[node] = new_op
            self.operands[node] = self.mlir.create_constant_weight_op(f'fullOp_{node.name}_c',
                                                                      shape, node.args[1])

    def convert_scalar_tensor_op(self, node):
        self.operands[node] = self.mlir.create_weight_op(f'scalar_tensorOp_{node.name}',
                                                         node.args[0])

    def convert_div_op(self, node):
        if node.args[0] in self.operands:
            in1 = self.operands[node.args[0]]
        else:
            in1 = self.mlir.create_weight_op(f'divOp_{node.name}_input1', node.args[0])

        if node.args[1] in self.operands:
            in2 = self.operands[node.args[1]]
        else:
            in2 = self.mlir.create_weight_op(f'divOp_{node.name}_input2', node.args[1])

        dtype = self.mlir.get_output_dtypes(node)
        new_op = top.DivOp(self.mlir.get_tensor_type([]), [in1, in2],
                           loc=self.get_loc(node.name),
                           ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_softmax_backward_data_op(self, node):
        grad_output = self.operands[node.args[0]]
        output = self.operands[node.args[1]]
        dtype = self.mlir.get_output_dtypes(node)
        new_op = top.SoftmaxBwdOp(self.mlir.get_tensor_type([]),
                                  grad_output,
                                  output,
                                  dim=node.args[2],
                                  loc=self.get_loc(node.name),
                                  ip=self.mlir.insert_point).grad_input
        self.operands[node] = new_op

    def convert_threshold_backward_op(self, node):
        grad_out = self.operands[node.args[0]]
        shape_z = list(node.args[1].meta['val'].size())
        dtype = self.mlir.get_output_dtypes(node)
        self_ = self.operands[node.args[1]]
        threshold = node.args[2]
        shape = list(node.meta['val'].size())
        # condition = top.ReluOp(*self.mlir.get_tensor_type([shape_z], F32Type.get()),
        #                     self_,
        #                     loc=self.get_loc(node.name+'_condition'),
        #                     ip=self.mlir.insert_point).output
        x_is_const = False
        y_is_const = True
        x_const_val = y_const_val = threshold
        new_op = top.WhereOp(self.mlir.get_tensor_type([]),
                             self_,
                             grad_out,
                             self.mlir.none_op,
                             x_is_const=x_is_const,
                             y_is_const=y_is_const,
                             x_const_val=x_const_val,
                             y_const_val=y_const_val,
                             loc=self.get_loc(node.name),
                             ip=self.mlir.insert_point).output

        self.operands[node] = new_op

    def convert_add_op(self, node):
        op0 = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        shape = self.mlir.get_output_shapes(node)
        if isinstance(node.args[1], torch.fx.Node):
            op1 = self.operands[node.args[1]]
            new_op = top.AddOp(self.mlir.get_tensor_type([]), [op0, op1],
                               do_relu=False,
                               coeff=np.atleast_1d(node.kwargs['alpha']).astype(np.float32)
                               if 'alpha' in node.kwargs else None,
                               loc=self.get_loc(node),
                               ip=self.mlir.insert_point).output
        else:
            op1 = np.atleast_1d(node.args[1]).astype(np.float32)
            new_op = top.AddConstOp(self.mlir.get_tensor_type([]),
                                    op0,
                                    const_val=op1,
                                    do_relu=False,
                                    loc=self.get_loc(node),
                                    ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_mul_op(self, node):
        op0 = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        shape = self.mlir.get_output_shapes(node)
        if isinstance(node.args[1], torch.fx.Node):
            op1 = self.operands[node.args[1]]
            new_op = top.MulOp(self.mlir.get_tensor_type([]), [op0, op1],
                               do_relu=False,
                               loc=self.get_loc(node),
                               ip=self.mlir.insert_point).output
        else:
            op1 = node.args[1]
            new_op = top.MulConstOp(self.mlir.get_tensor_type([]),
                                    op0,
                                    op1,
                                    loc=self.get_loc(node),
                                    ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_reshape_op(self, node):
        in_op = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        new_op = top.ReshapeOp(self.mlir.get_tensor_type([]),
                               in_op,
                               shape=node.args[1],
                               loc=self.get_loc(node),
                               ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_concat_op(self, node):
        operands = list()
        for name in node.args[0]:
            op = self.operands[name]
            operands.append(op)
        axis = node.args[1]
        dtype = self.mlir.get_output_dtypes(node)
        new_op = top.ConcatOp(self.mlir.get_tensor_type([]),
                              operands,
                              axis=axis,
                              loc=self.get_loc(node),
                              ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_dropout_op(self, node):
        in_op = self.operands[node.args[0]]
        self.operands[node] = [
            in_op, in_op
        ]  ##todo last output should be mask, op is in decompositions_dict for now

    def convert_native_dropout_backward_op(self, node):
        grad = self.operands[node.args[0]]
        mask = self.operands[node.args[1]]
        scale = node.args[2]
        tmp_out = top.MulOp(self.mlir.get_tensor_type([]), [grad, mask],
                            do_relu=False,
                            loc=self.get_loc(node.name + "_dropout_bwd_mm"),
                            ip=self.mlir.insert_point).output
        scale_out = top.MulConstOp(self.mlir.get_tensor_type([]),
                                   tmp_out,
                                   scale,
                                   loc=self.get_loc(node.name + "_dropout_bwd_scale"),
                                   ip=self.mlir.insert_point).output
        self.operands[node] = scale_out

    def convert_adaptive_avg_pool2d_op(self, node):
        op = self.operands[node.args[0]]
        output_size = node.args[1]
        dtype = self.mlir.get_output_dtypes(node)
        new_op = top.AdaptiveAvgPoolOp(self.mlir.get_tensor_type([]),
                                       op,
                                       output_size=output_size,
                                       loc=self.get_loc(node),
                                       ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_adaptive_avg_pool2d_backward_op(self, node):
        grad_output = self.operands[node.args[0]]
        input = self.operands[node.args[1]]
        dtype = self.mlir.get_output_dtypes(node)
        new_op = top.AddConstOp(
            self.mlir.get_tensor_type([]),  ##need mlir implement
            input,
            const_val=0,
            do_relu=False,
            loc=self.get_loc(node),
            ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_maxpool2d_with_indices_op(self, node):
        op = self.operands[node.args[0]]
        kernel_shape = node.args[1]
        strides = node.args[2]
        pads = [0, 0, 0, 0] if len(node.args) < 4 else node.args[3]
        dilation = [1, 1]
        ceil_mode = False
        assert (np.array(dilation) == 1).all()
        pads = pads + pads
        dtype = self.mlir.get_output_dtypes(node)
        outputs = top.MaxPoolWithMaskOp(self.mlir.get_tensor_type([]),
                                        self.mlir.get_tensor_type([]),
                                        op,
                                        kernel_shape=kernel_shape,
                                        strides=strides,
                                        pads=pads,
                                        ceil_mode=ceil_mode,
                                        loc=self.get_loc(
                                            [node.name + '.output', node.name + '.mask']),
                                        ip=self.mlir.insert_point)
        self.operands[node] = [outputs.output, outputs.mask]

    def convert_matmul_op(self, node):
        op0 = self.operands[node.args[0]]
        op1 = self.operands[node.args[1]]
        dtype = self.mlir.get_output_dtypes(node)
        new_op = top.MatMulOp(self.mlir.get_tensor_type([]),
                              op0,
                              op1,
                              self.mlir.none_op,
                              do_relu=False,
                              loc=self.get_loc(node),
                              ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_embedding_dense_backward_op(self, node):
        grad_output = self.operands[node.args[0]]
        indices = self.operands[node.args[1]]
        dtype = self.mlir.get_output_dtypes(node)
        new_op = top.EmbDenseBwdOp(self.mlir.get_tensor_type([]),
                                   grad_output,
                                   indices,
                                   num_weights=node.args[2],
                                   padding_idx=node.args[3],
                                   scale_grad_by_freq=node.args[4],
                                   loc=self.get_loc(node),
                                   ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_squeeze_op(self, node):
        op0 = self.operands[node.args[0]]
        axes = node.args[1] if len(node.args) == 2 else []
        if isinstance(axes, int):
            axes = [axes]
        new_op = top.SqueezeOp(self.mlir.get_tensor_type([]),
                               op0,
                               axes=axes,
                               loc=self.get_loc(node),
                               ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_base_conv_op(self, node):
        #(primals_7, primals_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)]
        op = self.operands[node.args[0]]
        strides = node.args[3]
        dilations = node.args[5]
        group = node.args[8]
        kernel_shape = self.mlir.get_output_shapes(node.args[1])[0]
        kernel_shape = kernel_shape[2:]
        pads = node.args[4]
        pads = pads + pads
        filter_op = self.operands[node.args[1]]
        if node.args[2] is not None:
            bias_op = self.operands[node.args[2]]
        else:
            bias_op = self.mlir.none_op
        dtype = self.mlir.get_output_dtypes(node)
        weight_is_coeff = 0
        new_op = top.ConvOp(self.mlir.get_tensor_type([]),
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
                            loc=self.get_loc(node),
                            ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_backward_conv_op2(self, node):
        # (Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
        # (getitem_108, relu_14, primals_55, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        #(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, int[] stride, SymInt[] padding, int[] dilation,
        #  bool transposed, SymInt[] output_padding, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
        # pdb.set_trace()
        grad_out = self.operands[node.args[0]]
        input = self.operands[node.args[1]]
        weight = self.operands[node.args[2]]
        kernel_shape = list(node.args[2].meta['val'].size())
        kernel_shape = kernel_shape[2:]
        bias_sizes = node.args[3]
        strides = node.args[4]
        dilations = node.args[6]
        transposed = node.args[7]
        output_padding = node.args[8]
        output_mask = [i for i in node.args[-1]]
        mask_to_users_idx = {}
        idx = 0
        for i, mask in enumerate(output_mask):
            if mask:
                mask_to_users_idx[i] = idx
                idx = idx + 1
        group = node.args[-2]
        pads = node.args[5]
        pads = pads + pads
        grad_input = transposed_grad_weight = grad_bias = self.mlir.none_op
        shape0 = [] if node.meta['val'][0] == None else list(node.meta['val'][0].size())
        shape1 = [] if node.meta['val'][1] == None else list(node.meta['val'][1].size())
        shape2 = [] if node.meta['val'][2] == None else list(node.meta['val'][2].size())
        dtype = self.mlir.get_output_dtypes(node)
        if dtype[0] is None:
            dtype.pop(0)
        bias_op = self.mlir.none_op
        if config.chip == "bm1690" or config.chip == "bm1688":
            grad_out_shape = list(node.args[0].meta['val'].size())
            input_shape = list(node.args[1].meta['val'].size())
            kernel_shape = list(node.args[2].meta['val'].size())
            shape0 = None if node.meta['val'][0] == None else list(node.meta['val'][0].size())
            shape1 = None if node.meta['val'][1] == None else list(node.meta['val'][1].size())
            shape2 = None if node.meta['val'][2] == None else list(node.meta['val'][2].size())
            inserts = [0, 0]
            output = top.ConvbwdOp(
                self.mlir.get_tensor_type([]),
                self.mlir.get_tensor_type([]),  #F32Type.get()
                self.mlir.get_tensor_type([]),
                grad_out,
                input,
                weight,
                group,
                input_shape,
                grad_out_shape,
                kernel_shape,
                strides,
                dilations,
                pads,
                inserts,
                output_mask[0],
                output_mask[1],
                output_mask[2],
                loc=self.get_loc(
                    [node.name + '.gradinput', node.name + '.gradweight', node.name + '.gradbias']),
                ip=self.mlir.insert_point)
            if output_mask[0]:
                grad_input = output.grad_input
            if output_mask[1]:
                transposed_grad_weight = output.grad_weight
            if output_mask[2]:
                grad_bias = output.grad_bias
            self.operands[node] = [grad_input, transposed_grad_weight, grad_bias]

    def convert_backward_conv_op(self, node):
        # (Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
        # (getitem_108, relu_14, primals_55, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        #(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, int[] stride, SymInt[] padding, int[] dilation,
        #  bool transposed, SymInt[] output_padding, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
        # pdb.set_trace()
        grad_out = self.operands[node.args[0]]
        input = self.operands[node.args[1]]
        weight = self.operands[node.args[2]]
        kernel_shape = list(node.args[2].meta['val'].size())
        kernel_shape = kernel_shape[2:]
        bias_sizes = node.args[3]
        strides = node.args[4]
        dilations = node.args[6]
        transposed = node.args[7]
        output_padding = node.args[8]
        output_mask = node.args[-1]
        mask_to_users_idx = {}
        idx = 0
        for i, mask in enumerate(output_mask):
            if mask:
                mask_to_users_idx[i] = idx
                idx = idx + 1
        group = node.args[-2]
        pads = node.args[5]
        pads = pads + pads
        grad_input = transposed_grad_weight = grad_bias = self.mlir.none_op
        shape0 = [] if node.meta['val'][0] == None else list(node.meta['val'][0].size())
        shape1 = [] if node.meta['val'][1] == None else list(node.meta['val'][1].size())
        shape2 = [] if node.meta['val'][2] == None else list(node.meta['val'][2].size())
        dtype = self.mlir.get_output_dtypes(node)
        if dtype[0] is None:
            dtype.pop(0)
        bias_op = self.mlir.none_op
        if output_mask[1]:
            shape = list(node.args[0].meta['val'].size())
            shape[0], shape[1] = shape[1], shape[0]
            transposed_gradout = top.TransposeOp(*self.mlir.get_tensor_type([shape], dtype),
                                                 grad_out,
                                                 0,
                                                 1,
                                                 loc=self.get_loc(node.name +
                                                                  '_transposed_gradout'),
                                                 ip=self.mlir.insert_point).output
            if shape[2] > 16:
                input_shape = list(node.args[1].meta['val'].size())
                grad_out_shape = list(node.args[0].meta['val'].size())
                transposed_grad_weight = top.ConvBwdWeightOp(
                    self.mlir.get_tensor_type([]),
                    input,
                    grad_out,
                    transposed_gradout,  #weight
                    group,
                    input_shape,
                    grad_out_shape,
                    kernel_shape,
                    strides,
                    dilations,
                    pads,
                    output_mask[-1],
                    loc=self.get_loc(node.name + '_grad_weight'),
                    ip=self.mlir.insert_point).output
                tmp = list(node.users.keys())
                self.name_map[tmp[mask_to_users_idx[1]]] = node.name + '_grad_weight'
            else:
                shape = list(node.args[1].meta['val'].size())
                shape[0], shape[1] = shape[1], shape[0]
                transposed_input = top.TransposeOp(*self.mlir.get_tensor_type([shape], dtype),
                                                   input,
                                                   0,
                                                   1,
                                                   loc=self.get_loc(node.name +
                                                                    '_transposed_input'),
                                                   ip=self.mlir.insert_point).output
                # kernel_shape_ = list(node.args[1].meta['val'].size())
                grad_weight_kernel_shape = list(node.args[0].meta['val'].size())
                grad_weight_kernel_shape = grad_weight_kernel_shape[2:]
                grad_weight_shape = shape1
                grad_weight_shape[0], grad_weight_shape[1] = grad_weight_shape[
                    1], grad_weight_shape[0]
                if pads[0] > 0 and strides[0] > 1:
                    new_strides = [1, 1]
                    new_pads = copy.deepcopy(pads)
                    input_shape = list(node.args[1].meta['val'].size())
                    pad_cal = grad_weight_shape[2] - (pads[0] + input_shape[2] - strides[0] *
                                                      (grad_weight_kernel_shape[0] - 1))
                    new_pads[2], new_pads[3] = pad_cal, pad_cal
                    grad_weight = top.ConvOp(*self.mlir.get_tensor_type([grad_weight_shape], dtype),
                                             transposed_input,
                                             transposed_gradout,
                                             bias_op,
                                             kernel_shape=grad_weight_kernel_shape,
                                             strides=new_strides,
                                             dilations=strides,
                                             pads=new_pads,
                                             group=group,
                                             do_relu=False,
                                             loc=self.get_loc(node.name + '_grad_weight'),
                                             ip=self.mlir.insert_point).output
                    tmp = list(node.users.keys())
                    self.name_map[tmp[mask_to_users_idx[1]]] = node.name + '_grad_weight'
                else:
                    input_shape = list(node.args[1].meta['val'].size())
                    dilations_grad_weight = strides
                    if input_shape[-1] % 2 != 0:  #!=
                        strides = [1, 1]
                    grad_weight = top.ConvOp(
                        *self.mlir.get_tensor_type([grad_weight_shape], dtype),
                        transposed_input,
                        transposed_gradout,
                        bias_op,
                        kernel_shape=grad_weight_kernel_shape,
                        strides=strides,
                        #strides = [1,1],
                        #dilations=strides,
                        dilations=dilations_grad_weight,
                        pads=pads,
                        group=group,
                        do_relu=False,
                        loc=self.get_loc(node.name + '_grad_weight'),
                        ip=self.mlir.insert_point).output
                    tmp = list(node.users.keys())
                    self.name_map[tmp[mask_to_users_idx[1]]] = node.name + '_grad_weight'
                temp_shape = shape1
                temp_shape[0], temp_shape[1] = temp_shape[1], temp_shape[0]
                # shape = list(node.args[1].meta['val'].size())
                transposed_grad_weight = top.TransposeOp(
                    *self.mlir.get_tensor_type([temp_shape], dtype),
                    grad_weight,
                    0,
                    1,
                    loc=self.get_loc(node.name + '_transposed_grad_weight'),
                    ip=self.mlir.insert_point).output
                tmp = list(node.users.keys())
                self.name_map[tmp[mask_to_users_idx[1]]] = node.name + '_transposed_grad_weight'
        if output_mask[0]:
            transposed_weight_shape = list(node.args[2].meta['val'].size())
            transposed_weight_shape[0], transposed_weight_shape[1] = transposed_weight_shape[
                1], transposed_weight_shape[0]
            transposed_weight = top.TransposeOp(
                *self.mlir.get_tensor_type([transposed_weight_shape], dtype),
                weight,
                0,
                1,
                loc=self.get_loc(node.name + '_transposed_weight_2'),
                ip=self.mlir.insert_point).output
            grad_input_kernel_shape = list(node.args[0].meta['val'].size())[-1]
            grad_input_output_shape = list(node.args[1].meta['val'].size())[-1]
            output_padding = grad_input_output_shape - strides[0] * (
                grad_input_kernel_shape - 1) + 2 * pads[0] - kernel_shape[0]
            output_padding = [output_padding] * 2
            grad_input = top.DeconvOp(*self.mlir.get_tensor_type([shape0], dtype),
                                      grad_out,
                                      transposed_weight,
                                      bias_op,
                                      kernel_shape=kernel_shape,
                                      strides=strides,
                                      pads=pads,
                                      group=group,
                                      dilations=dilations,
                                      output_padding=output_padding,
                                      do_relu=False,
                                      loc=self.get_loc(node.name + '_grad_input'),
                                      ip=self.mlir.insert_point).output
            tmp = list(node.users.keys())
            self.name_map[tmp[mask_to_users_idx[0]]] = node.name + '_grad_input'
        if output_mask[2]:
            grad_bias = top.ReduceOp(self.mlir.get_tensor_type([]),
                                     grad_out,
                                     axes=[0, 2, 3],
                                     keepdims=False,
                                     mode=StringAttr.get("ReduceSum"),
                                     loc=self.get_loc(node.name + "_grad_bias"),
                                     ip=self.mlir.insert_point).output
            tmp = list(node.users.keys())
            self.name_map[tmp[mask_to_users_idx[2]]] = node.name + '_grad_bias'
        self.operands[node] = [grad_input, transposed_grad_weight, grad_bias]

    def convert_sum_op(self, node):  #aten.sum.default                (getitem_6,)
        op0 = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        # assert method in ("ReduceMin", "ReduceMax", "ReduceMean", "ReduceL2", "ReduceL1",
        #                     "ReduceSum", "ReduceProd")
        in_shape = list(node.args[0].meta['val'].size())
        if len(node.args) >= 3:
            keepdims = node.args[2]
        elif len(list(node.args[0].meta['val'].size())) == 1:
            keepdims = True
        else:
            keepdims = False
        new_op = top.ReduceOp(
            self.mlir.get_tensor_type([]),
            op0,
            axes=sorted(node.args[1]) if len(node.args) > 1 else tuple(range(len(in_shape))),
            # keepdims = node.args[2] if len(node.args) > 2 else False,
            keepdims=keepdims,
            mode=StringAttr.get("ReduceSum"),
            loc=self.get_loc(node),
            ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_mean_op(self, node):
        op0 = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        # assert method in ("ReduceMin", "ReduceMax", "ReduceMean", "ReduceL2", "ReduceL1",
        #                     "ReduceSum", "ReduceProd")
        in_shape = list(node.args[0].meta['val'].size())
        new_op = top.ReduceOp(
            self.mlir.get_tensor_type([]),
            op0,
            axes=sorted(node.args[1]) if len(node.args) > 1 else tuple(range(len(in_shape))),
            keepdims=node.args[2] if len(node.args) > 2 else False,
            mode=StringAttr.get("ReduceMean"),
            loc=self.get_loc(node),
            ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_getitem_op(self, node):
        self.operands[node] = self.operands[node.args[0]][node.args[1]]

    def convert_to_op(self, node):
        op0 = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        shape = self.mlir.get_output_shapes(node)
        new_op = top.CastOp(self.mlir.get_tensor_type([]),
                            op0,
                            loc=self.get_loc(node),
                            ip=self.mlir.insert_point).output

        new_op2 = top.WeightReorderOp(self.mlir.get_tensor_type([]),
                                      new_op,
                                      loc=self.get_loc(f'{node}_weightReorder'),
                                      ip=self.mlir.insert_point).output
        self.operands[node] = new_op2

    def convert_arange_op(self, node):
        dtype = self.mlir.get_output_dtypes(node)
        if len(node.args) < 2:
            if node.args[0] in self.operands:
                end = self.operands[node.args[0]]
            else:
                end = self.mlir.create_weight_op(f'arangeOp_{node.name}_end', node.args[0])
            start = step = self.mlir.none_op
        else:
            if node.args[0] in self.operands:
                start = self.operands[node.args[0]]
            else:
                start = self.mlir.create_weight_op(f'arangeOp_{node.name}_start', node.args[0])
            if node.args[1] in self.operands:
                end = self.operands[node.args[1]]
            else:
                end = self.mlir.create_weight_op(f'arangeOp_{node.name}_end', node.args[1])
            if len(node.args) > 2:
                step = self.mlir.create_weight_op(f'arangeOp_{node.name}_step', node.args[2])
            else:
                step = self.mlir.none_op
        new_op = top.ArangeOp(self.mlir.get_tensor_type([]),
                              start,
                              end,
                              step,
                              loc=self.get_loc(node.name),
                              ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_slice_op(self, node):
        op0 = self.operands[node.args[0]]
        axis = self.mlir.create_weight_op(f'sliceOp_{node.name}_axis', node.args[1])
        start = self.mlir.create_weight_op(f'sliceOp_{node.name}_start', node.args[2])

        value = node.args[3]
        if value > 10000000000:
            value = np.iinfo(np.int32).max - 1024
        end = self.mlir.create_weight_op(f'sliceOp_{node.name}_end', value)

        if len(node.args) > 4:
            step = self.mlir.create_weight_op(f'sliceOp_{node.name}_step', node.args[4])
        else:
            step = self.mlir.create_weight_op(f'sliceOp_{node.name}_step', 1)
        dtype = self.mlir.get_output_dtypes(node)
        new_op = top.SliceAxisOp(self.mlir.get_tensor_type([]),
                                 op0,
                                 axis,
                                 start,
                                 step,
                                 end,
                                 loc=self.get_loc(node.name),
                                 ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_embedding_op(self, node):
        dtype = self.mlir.get_output_dtypes(node)
        if node.args[0] in self.operands:
            weight = self.operands[node.args[0]]
        else:
            weight = self.mlir.create_weight_op(f'embeddingOp_{node.name}_input1', node.args[0])

        if node.args[1] in self.operands:
            indices = self.operands[node.args[1]]
        else:
            indices = self.mlir.create_weight_op(f'embeddingOp_{node.name}_input2', node.args[1])
        # import pdb;pdb.set_trace()
        new_op = top.GatherOp(self.mlir.get_tensor_type([]),
                              weight,
                              indices,
                              axis=0,
                              loc=self.get_loc(node.name),
                              ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_compare_op(self, node, mode):
        assert mode in ("Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "NotEqual")
        op0 = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        if node.args[1] in self.operands:
            op1 = self.operands[node.args[1]]
            new_op = top.CompareOp(self.mlir.get_tensor_type([]),
                                   op0,
                                   op1,
                                   mode=StringAttr.get(mode),
                                   loc=self.get_loc(node.name),
                                   ip=self.mlir.insert_point).output
        else:
            new_op = top.CompareConstOp(self.mlir.get_tensor_type([]),
                                        op0,
                                        mode=StringAttr.get(mode),
                                        const_val=node.args[1],
                                        inversed=False,
                                        loc=self.get_loc(node.name),
                                        ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_where_op(self, node):
        op0 = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        x_is_const = y_is_const = False
        x_const_val = y_const_val = 0
        if node.args[1] in self.operands:
            op1 = self.operands[node.args[1]]
        else:
            x_is_const = True
            op1 = self.mlir.none_op
            x_const_val = node.args[1]

        if node.args[2] in self.operands:
            op2 = self.operands[node.args[2]]
        else:
            y_is_const = True
            op2 = self.mlir.none_op
            y_const_val = node.args[2]

        new_op = top.WhereOp(self.mlir.get_tensor_type([]),
                             op0,
                             op1,
                             op2,
                             x_is_const=x_is_const,
                             y_is_const=y_is_const,
                             x_const_val=x_const_val,
                             y_const_val=y_const_val,
                             loc=self.get_loc(node.name),
                             ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_copy_op(self, node):
        op0 = self.operands[node.args[0]]
        # dtype = self.mlir.get_output_dtypes(node)
        # input_stride = (1,)
        # output_stride = (1,)
        # new_op = top.CopyOp(*self.mlir.get_tensor_type(self.mlir.get_output_shapes(node), F32Type.get()),
        #                     op0,
        #                     node.args[0].meta['val'].size(),
        #                     input_stride,
        #                     output_stride,
        #                     loc=self.get_loc(node.name),
        #                     ip=self.mlir.insert_point).output
        # self.operands[node] = new_op
        self.operands[node] = op0

    def convert_rsqrt_op(self, node):
        dtype = self.mlir.get_output_dtypes(node)
        op0 = self.operands[node.args[0]]
        shape = self.mlir.get_output_shapes(node)
        new_op = top.RsqrtOp(self.mlir.get_tensor_type([]),
                             op0,
                             loc=self.get_loc(node.name),
                             ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_sub_op(self, node, is_reverse=False):
        dtype = self.mlir.get_output_dtypes(node)
        shape = self.mlir.get_output_shapes(node)
        op0 = self.operands[node.args[0]]
        if isinstance(node.args[1], torch.fx.Node):
            op1 = self.operands[node.args[1]]
            new_op = top.SubOp(self.mlir.get_tensor_type([]), [op0, op1],
                               is_reverse=is_reverse,
                               loc=self.get_loc(node.name),
                               ip=self.mlir.insert_point).output
        else:
            op1 = node.args[1]
            new_op = top.SubConstOp(self.mlir.get_tensor_type([]),
                                    op0,
                                    op1,
                                    is_reverse=is_reverse,
                                    loc=self.get_loc(node),
                                    ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_addmm_op(self, node):
        dtype = self.mlir.get_output_dtypes(node)
        shape = self.mlir.get_output_shapes(node)
        op0 = self.operands[node.args[0]]
        mat1_op = self.operands[node.args[1]]
        mat2_op = self.operands[node.args[2]]
        if len(node.args) == 5:
            # beta = self.const_val[node.args[3]]
            # alpha = self.const_val[node.inputs[4]]
            beta = node.args[3]
            alpha = node.args[4]
        else:
            beta = 1.0
            alpha = 1.0
        mm_op = top.MatMulOp(self.mlir.get_tensor_type([]),
                             mat1_op,
                             mat2_op,
                             self.mlir.none_op,
                             do_relu=False,
                             loc=self.get_loc(node.name + "_mm"),
                             ip=self.mlir.insert_point).output
        #assert (beta == 1.0 and alpha == 1.0)  # TODO:need to support
        new_op = top.AddOp(self.mlir.get_tensor_type([]), [op0, mm_op],
                           coeff=[beta, alpha],
                           loc=self.get_loc(node.name),
                           ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_split_op(self, node):
        op0 = self.operands[node.args[0]]
        axis = node.args[2]
        split_size = node.args[1]
        if isinstance(split_size, int):
            num = len(node.meta['val'])
            split_size = [split_size] * num
        else:
            num = len(split_size)
        names = [node.name + '_' + str(i) for i in range(num)]
        output = top.SplitOp([self.mlir.get_tensor_type([])] * num,
                             op0,
                             axis,
                             num,
                             split_size=split_size,
                             loc=self.get_loc(names),
                             ip=self.mlir.insert_point)
        self.operands[node] = output.outputs
        for user, name in zip(node.users, output_names):
            self.name_map[user] = name
        # breakpoint()

    def convert_minimum_op(self, node):
        op0 = self.operands[node.args[0]]
        op1 = self.operands[node.args[1]]
        dtype = self.mlir.get_output_dtypes(node)
        new_op = top.MinOp(self.mlir.get_tensor_type([]), [op0, op1],
                           loc=self.get_loc(node),
                           ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_maximum_op(self, node):
        op0 = self.operands[node.args[0]]
        op1 = self.operands[node.args[1]]
        dtype = self.mlir.get_output_dtypes(node)
        new_op = top.MaxOp(self.mlir.get_tensor_type([]), [op0, op1],
                           loc=self.get_loc(node),
                           ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_arctan_op(self, node):
        op0 = self.operands[node.args[0]]
        op_name = node.name + "_sign"
        sign_op = top.SignOp(self.mlir.get_tensor_type([]),
                             op0,
                             loc=self.get_loc(op_name),
                             ip=self.mlir.insert_point).output

        op_name = node.name + "_abs"
        abs_op = top.AbsOp(self.mlir.get_tensor_type([]),
                           op0,
                           loc=self.get_loc(op_name),
                           ip=self.mlir.insert_point).output
        op_name = node.name + "_mul"
        mul_op = top.MulOp(self.mlir.get_tensor_type([]), [abs_op, abs_op],
                           do_relu=False,
                           loc=self.get_loc(op_name),
                           ip=self.mlir.insert_point).output

        op_name = node.name + "_ml_mulscale"
        add_op = top.AddConstOp(self.mlir.get_tensor_type([]),
                                mul_op,
                                const_val=1,
                                loc=self.get_loc(op_name),
                                ip=self.mlir.insert_point).output

        op_name = node.name + "_sqrt"
        sqrt_op = top.SqrtOp(self.mlir.get_tensor_type([]),
                             add_op,
                             loc=self.get_loc(op_name),
                             ip=self.mlir.insert_point).output

        op_name = node.name + "_reciprocal"
        reciprocal_op = top.ReciprocalOp(self.mlir.get_tensor_type([]),
                                         sqrt_op,
                                         loc=self.get_loc(op_name),
                                         ip=self.mlir.insert_point).output

        op_name = node.name + "_arccos"
        arccos_op = top.ArccosOp(self.mlir.get_tensor_type([]),
                                 reciprocal_op,
                                 loc=self.get_loc(op_name),
                                 ip=self.mlir.insert_point).output

        arctan_op = top.MulOp(self.mlir.get_tensor_type([]), [sign_op, arccos_op],
                              do_relu=False,
                              loc=self.get_loc(node.name),
                              ip=self.mlir.insert_point).output

        self.operands[node] = arctan_op

    def convert_full_like_op(self, node):
        op0 = self.operands[node.args[0]]
        value = node.args[1]
        new_op = top.ConstantFillOp(self.mlir.get_tensor_type([]),
                                    op0,
                                    value=value,
                                    loc=self.get_loc(node),
                                    ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_clamp_min_op(self, node):
        op0 = self.operands[node.args[0]]
        min_val = node.args[1]
        # asssert(min_val == 0)
        new_op = top.ReluOp(self.mlir.get_tensor_type([]),
                            op0,
                            loc=self.get_loc(node),
                            ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_expand_op(self, node):
        dtype = self.mlir.get_output_dtypes(node)
        opI = self.operands[node.args[0]]
        new_exp = top.ExpandOp(self.mlir.get_tensor_type([]),
                               opI,
                               shape=node.args[1],
                               loc=self.get_loc(node.name),
                               ip=self.mlir.insert_point).output
        self.operands[node] = new_exp

    def convert_broadcast_op(self, node):
        dtype = self.mlir.get_output_dtypes(node)
        op = self.operands[node.args[0]]
        # shape = list(node.args[0].meta['val'].size())
        repeat_shape = node.args[1]
        axis = node.args[2]
        unsqueeze_shape = [1] * len(repeat_shape)
        unsqueeze_axis = list(range(len(repeat_shape)))
        for idx in range(len(repeat_shape)):
            if idx in axis:
                unsqueeze_axis.remove(idx)
                unsqueeze_shape[idx] = repeat_shape[idx]
        unsqueeze_op = top.UnsqueezeOp(self.mlir.get_tensor_type([]),
                                       op,
                                       unsqueeze_axis,
                                       loc=self.get_loc(node.name + '_unsqueeze'),
                                       ip=self.mlir.insert_point).output
        new_op = top.ExpandOp(self.mlir.get_tensor_type([]),
                              unsqueeze_op,
                              shape=repeat_shape,
                              loc=self.get_loc(node.name),
                              ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_amax_op(self, node, index):
        dtype = self.mlir.get_output_dtypes(node)
        op0 = self.operands[node.args[0]]
        dim = node.args[1][0]
        keepdims = node.args[2]
        select_last_index = True  # select_last_index = False
        #out_needs = [False, False]
        #for idx, out in enumerate(torch_node.outputs):
        #f len(out) > 0 and self.check_need(out):
        #out_needs[idx] = True
        new_op = top.ArgOp(self.mlir.get_tensor_type([]),
                           self.mlir.get_tensor_type([]),
                           input=op0,
                           axis=dim,
                           keepdims=keepdims,
                           mode=StringAttr.get("ArgMax"),
                           select_last_index=select_last_index,
                           loc=self.get_loc(node.name),
                           ip=self.mlir.insert_point)
        if index:
            out_ops = [new_op.values, new_op.indices]
        else:
            out_ops = new_op.values
        self.operands[node] = out_ops

    def convert_gather_op(self, node):
        # import pdb;pdb.set_trace()
        op0 = self.operands[node.args[0]]
        axis = node.args[1]
        op1 = self.operands[node.args[2]]
        dtype = self.mlir.get_output_dtypes(node)
        new_op = top.GatherElementsOp(self.mlir.get_tensor_type([]),
                                      op0,
                                      op1,
                                      axis=axis,
                                      loc=self.get_loc(node.name),
                                      ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_neg_op(self, node):
        op0 = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        new_op = top.MulConstOp(self.mlir.get_tensor_type([]),
                                op0,
                                const_val=-1,
                                loc=self.get_loc(node.name),
                                ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_transpose_op(self, node):
        op0 = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        no_dims = len(node.args) == 1
        dim0 = node.args[1] if not no_dims else 0
        dim1 = node.args[2] if not no_dims else 1
        new_op = top.TransposeOp(self.mlir.get_tensor_type([]),
                                 op0,
                                 dim0,
                                 dim1,
                                 loc=self.get_loc(node.name),
                                 ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    #_native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean,
    #   Tensor(b!) running_var, bool training, float momentum, float eps)
    def convert_batch_norm_op(self, node):
        dtype = self.mlir.get_output_dtypes(node)
        op0 = self.operands[node.args[0]]
        weight = self.operands[node.args[1]]
        bias = self.operands[node.args[2]]
        mean = self.operands[node.args[3]]
        var = self.operands[node.args[4]]
        momentum = node.args[6]
        eps = node.args[7]
        mean_shape = [1, list(node.meta['val'][3].size())[0], 1, 1]
        out = top.BatchNormTrainOp(self.mlir.get_tensor_type([]),
                                   self.mlir.get_tensor_type([]),
                                   self.mlir.get_tensor_type([]),
                                   self.mlir.get_tensor_type([]),
                                   self.mlir.get_tensor_type([]),
                                   op0,
                                   mean=mean,
                                   var=var,
                                   gamma=weight,
                                   beta=bias,
                                   epsilon=eps,
                                   momentum=momentum,
                                   loc=self.get_loc([
                                       node.name, node.name + '_mean', node.name + '_rstd',
                                       node.name + "_running_mean_update",
                                       node.name + "_running_var_update"
                                   ]),
                                   ip=self.mlir.insert_point)

        self.operands[node] = [
            out.output, out.mean_out, out.saved_invstd, out.running_mean, out.running_var
        ]

    def convert_batchnorm_decomp_op(self, node):
        dtype = self.mlir.get_output_dtypes(node)
        op0 = self.operands[node.args[0]]
        weight = self.operands[node.args[1]]
        bias = self.operands[node.args[2]]
        running_mean = self.operands[node.args[3]]
        running_var = self.operands[node.args[4]]
        training = node.args[5]
        momentum = node.args[6]
        eps = node.args[7]
        # import pdb;pdb.set_trace()
        bn_shape = list(node.meta['val'][0].size())
        mean_shape = [1, list(node.meta['val'][3].size())[0], 1, 1]
        rstd_shape = [1, list(node.meta['val'][4].size())[0], 1, 1]
        running_mean_shape = [1, list(node.meta['val'][1].size())[0], 1, 1]
        running_var_shape = [1, list(node.meta['val'][2].size())[0], 1, 1]

        running_mean_reshape_op = top.ReshapeOp(self.mlir.get_tensor_type([]),
                                                running_mean,
                                                shape=mean_shape,
                                                loc=self.get_loc(node.name + '_mean_reshape'),
                                                ip=self.mlir.insert_point)
        running_var_reshape_op = top.ReshapeOp(self.mlir.get_tensor_type([]),
                                               running_var,
                                               shape=rstd_shape,
                                               loc=self.get_loc(node.name + '_rstd_reshape'),
                                               ip=self.mlir.insert_point)
        weight_reshape_op = top.ReshapeOp(self.mlir.get_tensor_type([]),
                                          weight,
                                          shape=running_mean_shape,
                                          loc=self.get_loc(node.name + '_running_mean_reshape'),
                                          ip=self.mlir.insert_point)
        bias_reshape_op = top.ReshapeOp(self.mlir.get_tensor_type([]),
                                        bias,
                                        shape=running_var_shape,
                                        loc=self.get_loc(node.name + '_running_var_reshape'),
                                        ip=self.mlir.insert_point)

        out = top.MeanRstdOp(self.mlir.get_tensor_type([]),
                             self.mlir.get_tensor_type([]),
                             self.mlir.get_tensor_type([]),
                             self.mlir.get_tensor_type([]),
                             self.mlir.get_tensor_type([]),
                             self.mlir.get_tensor_type([]),
                             op0,
                             running_mean_reshape_op.output,
                             running_var_reshape_op.output,
                             weight_reshape_op.output,
                             bias_reshape_op.output,
                             eps,
                             momentum,
                             loc=self.get_loc([
                                 node.name + '_mean', node.name + '_rstd',
                                 node.name + '_running_mean_update',
                                 node.name + '_running_var_update', node.name + '_scale',
                                 node.name + '_bias_new'
                             ]),
                             ip=self.mlir.insert_point)

        bn_out = top.ScaleOp(self.mlir.get_tensor_type([]),
                             op0,
                             out.scale,
                             out.bias_new,
                             loc=self.get_loc(node.name),
                             ip=self.mlir.insert_point).output

        mean_reshape = top.ReshapeOp(self.mlir.get_tensor_type([]),
                                     out.mean,
                                     shape=node.meta['val'][3].size(),
                                     loc=self.get_loc(node.name + '_mean_reshape2'),
                                     ip=self.mlir.insert_point).output
        rstd_reshape = top.ReshapeOp(self.mlir.get_tensor_type([]),
                                     out.rstd,
                                     shape=node.meta['val'][4].size(),
                                     loc=self.get_loc(node.name + '_rstd_reshape2'),
                                     ip=self.mlir.insert_point).output
        running_mean_update_reshape = top.ReshapeOp(
            self.mlir.get_tensor_type([]),
            out.running_mean_update,
            shape=node.meta['val'][1].size(),
            loc=self.get_loc(node.name + '_running_mean_update_reshape2'),
            ip=self.mlir.insert_point).output
        running_var_update_reshape = top.ReshapeOp(self.mlir.get_tensor_type([]),
                                                   out.running_var_update,
                                                   shape=node.meta['val'][2].size(),
                                                   loc=self.get_loc(node.name +
                                                                    '_running_var_update_reshape2'),
                                                   ip=self.mlir.insert_point).output

        new_output_name = [
            node.name, node.name + '_running_mean_update_reshape',
            node.name + '_running_var_update_reshape', node.name + '_mean_reshape',
            node.name + '_rstd_reshape'
        ]
        for user, name in zip(node.users, new_output_name):
            self.name_map[user] = name
        self.operands[node] = [
            bn_out, mean_reshape, rstd_reshape, running_mean_update_reshape,
            running_var_update_reshape
        ]

        # new_output_name = [node.name,
        #                 node.name+'_running_mean_update',
        #                 node.name+'_running_var_update',
        #                 node.name+'_mean',
        #                 node.name+'_rstd']
        # for user, name in zip(node.users, new_output_name):
        #     self.name_map[user] = name
        # self.operands[node] = [bn_out,out.mean, out.rstd,out.running_mean_update, out.running_var_update]

    #- func: native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var,
    #                                   Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
    def convert_batch_norm_backward_op(self, node):  #running_mean和running_var没有使用
        grad_out = self.operands[node.args[0]]
        input = self.operands[node.args[1]]
        weight = self.operands[node.args[2]]
        mean = self.operands[node.args[5]]
        invstd = self.operands[node.args[6]]
        eps = node.args[8]
        output_mask = node.args[-1]
        dtype = self.mlir.get_output_dtypes(node)
        gradinput = gradweight = gradbias = self.mlir.none_op
        out = top.BatchNormBwdOp(self.mlir.get_tensor_type([]),
                                 self.mlir.get_tensor_type([]),
                                 self.mlir.get_tensor_type([]),
                                 grad_out,
                                 input,
                                 weight,
                                 mean,
                                 invstd,
                                 epsilon=eps,
                                 loc=self.get_loc([
                                     node.name + '_grad_input', node.name + '_grad_weight',
                                     node.name + '_grad_bias'
                                 ]),
                                 ip=self.mlir.insert_point)
        if output_mask[2]:
            gradbias = out.bias_grad
        if output_mask[1]:
            gradweight = out.weight_grad
        if output_mask[0]:
            gradinput = out.grad_in
        self.operands[node] = [gradinput, gradweight, gradbias]
        new_output_name = [
            node.name + '_grad_input', node.name + '_grad_weight', node.name + '_grad_bias'
        ]
        for user, name in zip(node.users, new_output_name):
            self.name_map[user] = name

    def convert_layer_norm_backward_op(self, node):
        #tangents_1, add_3, [10], div_1, rsqrt_1, primals_7, primals_8, [True, True, True]
        #Tensor grad_out, Tensor input, SymInt[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask)
        # -> (Tensor, Tensor, Tensor)
        grad_out = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        assert node.args[1] in self.operands
        input = self.operands[node.args[1]]
        normalized_shape = node.args[2]
        assert node.args[3] in self.operands
        mean = self.operands[node.args[3]]
        assert node.args[4] in self.operands
        rstd = self.operands[node.args[4]]
        weight_opt = self.operands[
            node.args[5]] if node.args[5] in self.operands else self.mlir.none_op
        bias_opt = self.operands[
            node.args[6]] if node.args[6] in self.operands else self.mlir.none_op
        out = top.LayerNormBwdOp(self.mlir.get_tensor_type([]),
                                 self.mlir.get_tensor_type([]),
                                 self.mlir.get_tensor_type([]),
                                 grad_out,
                                 input,
                                 mean,
                                 rstd,
                                 weight_opt,
                                 bias_opt,
                                 normalized_shape=normalized_shape,
                                 loc=self.get_loc([
                                     node.name + "_grad_input", node.name + "_grad_weight",
                                     node.name + "_grad_bias"
                                 ]),
                                 ip=self.mlir.insert_point)
        self.operands[node] = [out.grad_input, out.grad_weight, out.grad_bias]

    def convert_layer_norm_op(self, node):
        op0 = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        shape0 = list(node.meta['val'][0].size())
        shape1 = list(node.meta['val'][1].size())
        shape2 = list(node.meta['val'][2].size())
        normalized_shape = node.args[1]
        assert node.args[2] in self.operands
        scale_opd = self.operands[node.args[2]]
        assert node.args[3] in self.operands
        bias_opd = self.operands[node.args[3]]
        eps = node.args[4]
        axis = np.atleast_1d(-len(normalized_shape)).astype(np.int32)
        out = top.LayerNormTrainOp(self.mlir.get_tensor_type([]),
                                   self.mlir.get_tensor_type([]),
                                   self.mlir.get_tensor_type([]),
                                   op0,
                                   scale_opd,
                                   bias_opd,
                                   normalized_shape=normalized_shape,
                                   axis=axis,
                                   eps=eps,
                                   loc=self.get_loc(
                                       [node.name, node.name + "_Mean", node.name + "_Rstd"]),
                                   ip=self.mlir.insert_point)
        new_op = out.output
        mean = out.mean
        rstd = out.variance
        self.operands[node] = [new_op, mean, rstd]

    def convert_softmax_op(self, node, log):
        op0 = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        in_dim_len = len(list(node.args[0].meta['val'].size()))
        tmp = node.args[1] + in_dim_len if node.args[1] < 0 else node.args[1]
        dim = np.atleast_1d(tmp).astype(np.int32)
        new_op = top.SoftmaxOp(self.mlir.get_tensor_type([]),
                               op0,
                               axis=dim,
                               log=log,
                               loc=self.get_loc(node.name),
                               ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_nllloss_op(self, node):
        op0 = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        shape = list(node.meta['val'][0].size())
        shape = [1] if shape == [] else shape
        op1 = self.operands[node.args[1]]
        if node.args[2] != None:
            weight = self.mlir.create_weight_op(f'nlllossOp_{node.name}_input1', node.args[2])
        else:
            weight = self.mlir.none_op
        redecution = node.args[3]
        ignore_index = np.atleast_1d(node.args[4]).astype(np.int32)
        new_op = top.NLLlossOp(self.mlir.get_tensor_type([]),
                               self.mlir.get_tensor_type([]),
                               op0,
                               op1,
                               weight,
                               redecution,
                               ignore_index,
                               loc=self.get_loc(node.name),
                               ip=self.mlir.insert_point)
        self.operands[node] = [new_op.output, new_op.total_weight]

    def convert_var_op(self, node):
        op0 = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        reduce_list = node.args[1]
        correction = node.kwargs['correction']
        new_op = top.VarianceOp(self.mlir.get_tensor_type([]),
                                op0,
                                reduce_list,
                                correction,
                                loc=self.get_loc(node.name),
                                ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_var_op2(self, node):
        op = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        op_shape = list(node.args[0].meta['val'].size())
        dim = node.args[1]
        ######### cal mean #########
        mean = top.ReduceOp(self.mlir.get_tensor_type([]),
                            op,
                            axes=dim,
                            keepdims=False,
                            mode=StringAttr.get("ReduceMean"),
                            loc=self.get_loc(node.name + "_mean"),
                            ip=self.mlir.insert_point).output
        ######## cal squared_mean ########
        squared = top.MulOp(self.mlir.get_tensor_type([]), [op, op],
                            do_relu=False,
                            loc=self.get_loc(node.name + "_squared"),
                            ip=self.mlir.insert_point).output
        squared_mean = top.ReduceOp(self.mlir.get_tensor_type([]),
                                    squared,
                                    axes=dim,
                                    keepdims=False,
                                    mode=StringAttr.get("ReduceMean"),
                                    loc=self.get_loc(node.name + "_squared_mean"),
                                    ip=self.mlir.insert_point).output
        ######## cal var #########
        mean_squared = top.MulOp(self.mlir.get_tensor_type([]), [mean, mean],
                                 do_relu=False,
                                 loc=self.get_loc(node.name + "_mean_squared"),
                                 ip=self.mlir.insert_point).output
        var = top.SubOp(self.mlir.get_tensor_type([]), [squared_mean, mean_squared],
                        is_reverse=False,
                        loc=self.get_loc(node.name + "_var"),
                        ip=self.mlir.insert_point).output
        self.operands[node] = var

    def convert_select_op(self, node):  #aten.select.int           (view, 0, 0)
        # step_name = torch_node.inputs[0] + '_tpu_step'
        # end_name = torch_node.inputs[0] + torch_node.inputs[2] + '_tpu_end'
        # self.addWeight(step_name, np.array([1], dtype=np.float32))
        # assert torch_node.inputs[2] in self.const_val.keys()
        # end = self.const_val[torch_node.inputs[2]] + 1
        # self.addWeight(end_name, np.array([end], dtype=np.float32))
        # slice_op = top.SliceAxisOp(self.unranked_type,
        #                            self.getOp(torch_node.inputs[0]),
        #                            self.getOp(torch_node.inputs[1]),
        #                            self.getOp(torch_node.inputs[2]),
        #                            self.getOp(step_name),
        #                            self.getOp(end_name),
        #                            loc=self.get_loc(
        #                                "{}_SliceAxis".format(torch_node.name)),
        #                            ip=self.mlir.insert_point).output
        # axis = self.const_val[torch_node.inputs[1]]
        # new_op = top.SqueezeOp(self.unranked_type,
        #                        slice_op,
        #                        axes=[axis],
        #                        loc=self.get_loc(torch_node.name),
        #                        ip=self.mlir.insert_point).output
        # self.addOperand(torch_node.name, new_op)
        op0 = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        shape.insert(node.args[1], 1)
        axis = self.mlir.create_weight_op(f'sliceOp_{node.name}_axis', node.args[1])
        start = self.mlir.create_weight_op(f'sliceOp_{node.name}_start', node.args[2])
        step = self.mlir.create_weight_op(f'sliceOp_{node.name}_step', 1)
        end = self.mlir.create_weight_op(f'sliceOp_{node.name}_end', node.args[2] + 1)
        slice_op = top.SliceAxisOp(self.mlir.get_tensor_type([]),
                                   op0,
                                   axis,
                                   start,
                                   step,
                                   end,
                                   loc=self.get_loc(node.name + "_slice"),
                                   ip=self.mlir.insert_point).output
        new_op = top.SqueezeOp(self.mlir.get_tensor_type([]),
                               slice_op,
                               axes=[node.args[1]],
                               loc=self.get_loc(node),
                               ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_math_op(self, node, mode):
        assert mode in ["cos", "cosh", "sin", "sinh", "tan", "tanh", "exp", "erf", "log"]
        op0 = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        cmd = "top.%sOp(*self.mlir.get_tensor_type([shape], F32Type.get()), op0, loc=self.get_loc(node.name), ip=self.mlir.insert_point).output" % mode.capitalize(
        )
        new_op = eval(cmd)
        self.operands[node] = new_op

    def convert_clone_op(self, node):
        assert len(node.args) == 1
        self.operands[node] = self.operands[node.args[0]]

    def convert_fresh_copy_op(self, node):
        pass

    def convert_unsqueeze_op(self, node):
        op0 = self.operands[node.args[0]]
        axis = node.args[1]
        dtype = self.mlir.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        new_op = top.UnsqueezeOp(self.mlir.get_tensor_type([]),
                                 op0,
                                 axes=[axis],
                                 loc=self.get_loc(node),
                                 ip=self.mlir.insert_point).output
        # import pdb
        # pdb.set_trace()
        # origin_shape = list(node.args[0].meta['val'].size())
        # if origin_shape == []:
        #     new_op = op0
        #     print("changed!")
        self.operands[node] = new_op

    def convert_sigmoid_op(self, node):
        op = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        new_op = top.SigmoidOp(self.mlir.get_tensor_type([]),
                               op,
                               loc=self.get_loc(node),
                               ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_silu_op(self, node):
        op = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        new_op = top.SiLUOp(self.mlir.get_tensor_type([]),
                            op,
                            loc=self.get_loc(node),
                            ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_pow_op(self, node):
        op = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        if not isinstance(node.args[1], torch.fx.Node):
            power = node.args[1]
            new_op = top.PowOp(self.mlir.get_tensor_type([]),
                               op,
                               power,
                               loc=self.get_loc(node),
                               ip=self.mlir.insert_point).output
        else:
            power = self.operands[node.args[1]]
            new_op = top.PowTensorOp(
                self.mlir.get_tensor_type([]),
                # op,
                # power,
                [op, power],
                loc=self.get_loc(node),
                ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_mse_op(self, node):
        op0 = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        op1 = self.operands[node.args[1]]
        new_op = top.MSELossOp(self.mlir.get_tensor_type([]),
                               op0,
                               op1,
                               loc=self.get_loc(node),
                               ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    # def convert_maxpool2d_backward_op(self,node):
    #     # Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices
    #     grad_out = self.operands[node.args[0]]
    #     input = self.operands[node.args[1]]
    #     kernel_size = node.args[2]
    #     stride = node.args[3]
    #     padding = node.args[4]
    #     padding = padding+padding
    #     dilation = node.args[5]
    #     ceil_mode = node.args[6]
    #     indices = self.operands[node.args[-1]]
    #     dtype = self.mlir.get_output_dtypes(node)
    #     shape = list(node.meta['val'].size())
    #     shape_input = list(node.args[1].meta['val'].size())
    #     shape_grad = list(node.args[0].meta['val'].size())
    #     import pdb;pdb.set_trace()
    #     x_is_const = False
    #     y_is_const = True
    #     x_const_val = y_const_val = 0
    #     new_op = top.WhereOp(self.mlir.get_tensor_type([]),
    #                             indices,
    #                             grad_out,
    #                             self.mlir.none_op,
    #                             x_is_const = x_is_const,
    #                             y_is_const = y_is_const,
    #                             x_const_val = x_const_val,
    #                             y_const_val = y_const_val,
    #                             loc=self.get_loc(node.name+'_before_padding'),
    #                             ip=self.mlir.insert_point).output
    #     pad_num = (shape_input[-1]-shape_grad[-1])//2
    #     padding = [pad_num]*4
    #     mode = 'constant'
    #     new_op_pad = top.PadOp(
    #         self.mlir.get_tensor_type([]),
    #         new_op,
    #         self.mlir.none_op,
    #         paddings = padding,
    #         mode = StringAttr.get(mode),
    #         # val = 0.0,
    #         loc=self.get_loc(node.name),
    #         ip=self.mlir.insert_point).output

    #     self.operands[node] = new_op_pad

    def convert_maxpool2d_backward_op(self, node):
        grad_out = self.operands[node.args[0]]  # 上游梯度
        input = self.operands[node.args[1]]  # 原始输入
        kernel = node.args[2]
        stride = node.args[3]
        padding = node.args[4] + node.args[4]
        dilation = node.args[5] + node.args[5]
        ceil_mode = node.args[6]
        indices = self.operands[node.args[-1]]  #Index of the maximum value of forward propagation
        input_shape = list(node.args[1].meta['val'].size())
        new_op = top.MaxPoolingIndicesBwdOp(self.mlir.get_tensor_type([]),
                                            grad_out,
                                            indices,
                                            kernel_shape=kernel,
                                            strides=stride,
                                            pads=padding,
                                            dilations=dilation,
                                            input_shape=input_shape,
                                            loc=self.get_loc(node.name),
                                            ip=self.mlir.insert_point).grad_input
        self.operands[node] = new_op

    # def convert_maxpool2d_backward_op(self, node):
    #     # Parse the input parameters of the operation node
    #     grad_out = self.operands[node.args[0]]  # 上游梯度
    #     input    = self.operands[node.args[1]]  # 原始输入
    #     kernel_size = node.args[2]
    #     stride    = node.args[3]
    #     padding   = node.args[4]
    #     dilation  = node.args[5]
    #     ceil_mode = node.args[6]
    #     indices   = self.operands[node.args[-1]]
    #     dtype       = self.mlir.get_output_dtypes(node)
    #     shape_input = list(node.args[1].meta['val'].size())  # 原始输入形状
    #     shape_grad  = list(node.args[0].meta['val'].size())  # 上游梯度形状
    #     zero_data   = np.zeros(shape_input, dtype=np.float32)  # 用于填充的零数据
    #     zero_weight = self.mlir.create_weight_op(f'{node.name}_zero', zero_data)
    #     # scatter
    #     new_shape   = [shape_input[0], shape_input[1], shape_input[2] * shape_input[3]]
    #     zero_weight_tensor = top.ReshapeOp(self.mlir.get_tensor_type([]),
    #                                          zero_weight,
    #                                          shape = new_shape,
    #                                          loc=self.get_loc(node.name+'_reshape'),
    #                                          ip=self.mlir.insert_point).output

    #     new_indices_shape = [shape_grad[0], shape_grad[1], shape_grad[2] * shape_grad[3]]

    #     indices_reshape = top.ReshapeOp(self.mlir.get_tensor_type([]),
    #                                     indices,
    #                                     shape = new_indices_shape,
    #                                     loc=self.get_loc(node.name+'_reshape1'),
    #                                     ip=self.mlir.insert_point).output
    #     grad_out_reshape = top.ReshapeOp(self.mlir.get_tensor_type([]),
    #                                     grad_out,
    #                                     shape = new_indices_shape,
    #                                     loc=self.get_loc(node.name+'_reshape2'),
    #                                     ip=self.mlir.insert_point).output

    #     scatter = top.ScatterElementsOp(self.mlir.get_tensor_type([]),
    #                             zero_weight_tensor,
    #                             indices_reshape,
    #                             grad_out_reshape,
    #                             -1,
    #                             reduction=1,
    #                             nc_can_split=True,
    #                             loc=self.get_loc(node.name +"_scatter"),
    #                             ip=self.mlir.insert_point).output

    #     res = top.ReshapeOp(self.mlir.get_tensor_type([]),
    #                         scatter,
    #                         shape = shape_input,
    #                         loc=self.get_loc(node.name),
    #                         ip=self.mlir.insert_point).output

    #     self.operands[node] = res

    def convert_trunc_op(self, node):
        dtype = self.mlir.get_output_dtypes(node)
        op0 = self.operands[node.args[0]]
        shape = self.mlir.get_output_shapes(node)
        new_op = top.TruncOp(self.mlir.get_tensor_type([]),
                             op0,
                             loc=self.get_loc(node.name),
                             ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_hardswish_op(self, node):
        dtype = self.mlir.get_output_dtypes(node)
        op0 = self.operands[node.args[0]]
        new_op = top.HardSwishOp(self.mlir.get_tensor_type([]),
                                 op0,
                                 loc=self.get_loc(node.name),
                                 ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_leaky_relu_op(self, node):
        dtype = self.mlir.get_output_dtypes(node)
        op0 = self.operands[node.args[0]]
        alpha = 0.5
        new_op = top.LeakyReluOp(self.mlir.get_tensor_type([]),
                                 op0,
                                 alpha,
                                 loc=self.get_loc(node.name),
                                 ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_zero_op(self, node):
        if not isinstance(node.args[0], list) and node.args[0] in self.operands:
            dtype = self.mlir.get_output_dtypes(node)
            op0 = self.operands[node.args[0]]
            new_op = top.ConstantFillOp(self.mlir.get_tensor_type([]),
                                        op0,
                                        value=0,
                                        loc=self.get_loc(node),
                                        ip=self.mlir.insert_point).output
            self.operands[node] = new_op
        else:
            shape = node.args[0]
            new_op = self.mlir.create_constant_weight_op(f'{node.name}_c', shape, 0)
            # data_type = "F32"
            # tensor_type = RankedTensorType.get(list(node.args[0]), self.mlir_type[data_type])
            # op = Operation.create("top.Weight",
            #                   results=[tensor_type],
            #                   loc=Location.fused([Location.name(name)]))
            # self.mlir.insert_point.insert(op)
            # result = op.results[0]
            # self.load_weight[name] = (result, node.args[0], data_type)
            # shape = tuple(node.args[0])
            # self.weights_data[name] = np.full(shape,0,dtype = np.float32)
            # if name in self.load_weight:
            #     _op, _shape, _type = self.load_weight[name]
            #     self.operands[node] = _op
            # else:
            #     self.operands[node] = result
            self.operands[node] = new_op

    def convert_clamp_op(self, node):
        input = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        if node.args[1] != None:
            min_val = node.args[1]
            if node.args[1] in self.operands:
                op1 = self.operands[node.args[1]]
            else:
                op1 = self.mlir.create_weight_op(f'{node.name}_minval', node.args[1])
            condition = top.CompareOp(self.mlir.get_tensor_type([]),
                                      input,
                                      op1,
                                      mode=StringAttr.get('GreaterOrEqual'),
                                      loc=self.get_loc(node.name + "min_cond"),
                                      ip=self.mlir.insert_point).output
            x_is_const = False
            y_is_const = True
            x_const_val = y_const_val = min_val
            new_op = top.WhereOp(self.mlir.get_tensor_type([]),
                                 condition,
                                 input,
                                 self.mlir.none_op,
                                 x_is_const=x_is_const,
                                 y_is_const=y_is_const,
                                 x_const_val=x_const_val,
                                 y_const_val=y_const_val,
                                 loc=self.get_loc(node.name),
                                 ip=self.mlir.insert_point).output
        if len(node.args) > 2:
            max_val = node.args[2]
            if node.args[1] in self.operands:
                op1 = self.operands[node.args[1]]
            else:
                op1 = self.mlir.create_weight_op(f'{node.name}_maxval', node.args[2])
            condition = top.CompareOp(self.mlir.get_tensor_type([]),
                                      input,
                                      op1,
                                      mode=StringAttr.get('LessOrEqual'),
                                      loc=self.get_loc(node.name + "max_cond"),
                                      ip=self.mlir.insert_point).output
            x_is_const = False
            y_is_const = True
            x_const_val = y_const_val = max_val
            new_op = top.WhereOp(self.mlir.get_tensor_type([]),
                                 condition,
                                 input,
                                 self.mlir.none_op,
                                 x_is_const=x_is_const,
                                 y_is_const=y_is_const,
                                 x_const_val=x_const_val,
                                 y_const_val=y_const_val,
                                 loc=self.get_loc(node.name),
                                 ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_masked_fill_op(self, node):
        # input = self.operands[node.args[0]]
        # dtype = self.mlir.get_output_dtypes(node)
        # shape = list(node.meta['val'].size())
        # val = node.args[2]
        # mask = self.operands[node.args[1]]
        # x_is_const = False
        # y_is_const = True
        # x_const_val = y_const_val = val
        # new_op = top.WhereOp(*self.mlir.get_tensor_type([shape], F32Type.get()),
        #                         mask,
        #                         input,
        #                         self.mlir.none_op,
        #                         x_is_const = x_is_const,
        #                         y_is_const = y_is_const,
        #                         x_const_val = x_const_val,
        #                         y_const_val = y_const_val,
        #                         loc=self.get_loc(node.name),
        #                         ip=self.mlir.insert_point).output
        # self.operands[node] = new_op
        input = self.operands[node.args[0]]
        mask = self.operands[node.args[1]]
        dtype = self.mlir.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        const_val = node.args[2]
        new_op = top.MaskedFillOp(self.mlir.get_tensor_type([]),
                                  mask,
                                  input,
                                  inversed=True,
                                  const_val=const_val,
                                  loc=self.get_loc(node.name),
                                  ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_index_op(
        self, node
    ):  #(slice_12, [None, None, unsqueeze, _to_copy_1]) 每个tensor对应一个dim gather的输出再作为下一个gather输入
        op0 = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        idx_store = []
        for k, idx in enumerate(node.args[1]):
            if idx is not None:
                indices = [k, idx]
                idx_store.append(indices)
        # shape process
        for i in range(len(idx_store)):
            nd = idx_store[i]
            if len(list(nd[1].meta['val'].size())) != 1:
                target_shape = list(nd[1].meta['val'].size())
                squeeze_axis = target_shape.index(1)
                target_shape.pop(squeeze_axis)
                tmp_op = self.operands[nd[1]]
                out = top.SqueezeOp(self.mlir.get_tensor_type([]),
                                    tmp_op,
                                    axes=[squeeze_axis],
                                    loc=self.get_loc(node.name + "_shape_update_" + str(i)),
                                    ip=self.mlir.insert_point).output
                idx_store[i][1] = out
            else:
                target_shape = list(nd[1].meta['val'].size())
                idx_store[i][1] = self.operands[nd[1]]
            idx_store[i].append(target_shape[0])
        origin_shape = list(node.args[0].meta['val'].size())
        while idx_store:
            info = idx_store[0]
            axis = info[0]
            indices = info[1]
            gather_shape = origin_shape
            gather_shape[axis] = info[2]
            new_op = top.GatherOp(self.mlir.get_tensor_type([]),
                                  op0,
                                  indices,
                                  axis=axis,
                                  loc=self.get_loc(node.name + "_" + str(axis)),
                                  ip=self.mlir.insert_point).output
            op0 = new_op
            idx_store.pop(0)
        # indices = [self.operands[i] for i in node.args[1] if i is not None]
        # indices = indices[0]
        # shape = list(node.meta['val'].size())
        # new_op = top.GatherOp(*self.mlir.get_tensor_type([shape], F32Type.get()),
        #                       op0,
        #                       indices,
        #                       axis=3,
        #                       loc=self.get_loc(node.name),
        #                       ip=self.mlir.insert_point).output
        if len(node.args[1]) < len(origin_shape):
            output_shape = self.mlir.get_output_shapes(node)[0]
            if len(origin_shape) > len(output_shape):
                new_op = top.SqueezeOp(self.mlir.get_tensor_type([]),
                                       new_op,
                                       axes=[0],
                                       loc=self.get_loc(node.name + "_squeeze"),
                                       ip=self.mlir.insert_point).output
            else:
                new_op = top.UnsqueezeOp(self.mlir.get_tensor_type([]),
                                         new_op,
                                         axes=[0],
                                         loc=self.get_loc(node.name + "_unsqueeze"),
                                         ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_select_scatter_op(self, node):  #(new_zeros, mm_73, 1, 0)
        op0 = self.operands[node.args[0]]
        op1 = self.operands[node.args[1]]
        dtype = self.mlir.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        axis = node.args[2]
        update_shape = list(node.args[1].meta['val'].size())
        update_shape.insert(axis, 1)
        update = top.UnsqueezeOp(self.mlir.get_tensor_type([]),
                                 op1,
                                 axes=[axis],
                                 loc=self.get_loc(node.name + "update"),
                                 ip=self.mlir.insert_point).output
        # index = top.ConstantFillOp(*self.mlir.get_tensor_type([update_shape], F32Type.get()),
        #                             update,
        #                             value = node.args[3],
        #                             loc=self.get_loc(node.name+"index"),
        #                             ip=self.mlir.insert_point).output
        index = self.mlir.create_constant_weight_op(f'{node.name}_index', update_shape,
                                                    node.args[3])
        # index = np.array([[node.args[3]]])
        # indices = self.mlir.create_weight_op(f'{node.name}_indices', index)
        # new_op = top.ScatterNDOp(*self.mlir.get_tensor_type([shape], F32Type.get()),
        #                                 op0,
        #                                 indices,
        #                                 update,
        #                                 loc=self.get_loc(node.name),
        #                                 ip=self.mlir.insert_point).output
        new_op = top.ScatterElementsOp(self.mlir.get_tensor_type([]),
                                       op0,
                                       index,
                                       update,
                                       axis,
                                       loc=self.get_loc(node.name),
                                       ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_slice_scatter_op(self, node):
        # import pdb;pdb.set_trace()
        input = self.operands[node.args[0]]
        update = self.operands[node.args[1]]
        dtype = self.mlir.get_output_dtypes(node)
        start = node.args[3]
        end = node.args[4]
        shape = list(node.meta['val'].size())
        update_shape = list(node.args[1].meta['val'].size())
        axis = node.args[2]
        if end > shape[axis]:
            end = shape[axis]
        elif end < 0:
            end += shape[axis]
        step = node.args[5] if len(node.args) >= 6 else 1
        expand_shape = tuple(-1 if i == axis else 1 for i in range(len(shape)))
        indices = np.arange(start, end, step)
        indices = indices.reshape(expand_shape)
        broadcast_indices = np.ones(tuple(update_shape)) * indices
        index = self.mlir.create_constant_weight_op(f'{node.name}_index', update_shape,
                                                    broadcast_indices)
        new_op = top.ScatterElementsOp(self.mlir.get_tensor_type([]),
                                       input,
                                       index,
                                       update,
                                       axis,
                                       loc=self.get_loc(node.name),
                                       ip=self.mlir.insert_point).output
        self.operands[node] = new_op

        # input = self.operands[node.args[0]]
        # update = self.operands[node.args[1]]
        # dtype = self.mlir.get_output_dtypes(node)
        # shape = list(node.meta['val'].size())
        # update_shape = list(node.args[1].meta['val'].size())
        # axis = self.mlir.create_weight_op(f'sliceOp_{node.name}_axis', node.args[2])

        # start = self.mlir.create_weight_op(f'sliceOp_{node.name}_start', node.args[3])
        # step = self.mlir.create_weight_op(f'sliceOp_{node.name}_step', 1)
        # end = self.mlir.create_weight_op(f'sliceOp_{node.name}_end', node.args[4] if node.args[4]>0 else 999999)
        # slice_op = top.SliceAxisOp(*self.mlir.get_tensor_type([update_shape], F32Type.get()),
        #                          update,
        #                          axis,
        #                          start,
        #                          step,
        #                          end,
        #                          loc=self.get_loc(node.name+"_slice"),
        #                         ip=self.mlir.insert_point).output
        # # update_shape = list(node.args[1].meta['val'].size())
        # # if shape == update_shape:
        # #     index = np.array([[0]])
        # # else:
        # #     index = np.arange(node.args[3],node.args[4])
        # index = np.array([[0]])
        # indices = self.mlir.create_weight_op(f'{node.name}_indices', index)
        # new_op = top.ScatterNDOp(*self.mlir.get_tensor_type([shape], F32Type.get()),
        #                                 input,
        #                                 indices,
        #                                 slice_op,
        #                                 loc=self.get_loc(node.name),
        #                                 ip=self.mlir.insert_point).output
        # self.operands[node] = new_op

    def convert_index_put_op(self, node):
        if len(node.args[1]) <= 1:
            dtype = self.mlir.get_output_dtypes(node)
            input = self.operands[node.args[0]]
            indices = self.operands[node.args[1][0]]
            values = self.operands[node.args[2]]
            if len(node.args) > 3:
                accumulate = node.args[3]
            else:
                accumulate = False
            new_op = top.IndexPutOp(self.mlir.get_tensor_type([]),
                                    input,
                                    indices,
                                    values,
                                    accumulate=accumulate,
                                    loc=self.get_loc(node.name),
                                    ip=self.mlir.insert_point).output
            self.operands[node] = new_op
        else:

            # op0 = self.operands[node.args[0]]
            # value = self.operands[node.args[2]]
            # accumulate = node.args[3]
            # dtype = self.mlir.get_output_dtypes(node)
            # indices = node.args[1]
            # indices_store = []
            # input_shape = list(node.args[0].meta['val'].size())
            # start = self.mlir.create_weight_op(f'{node.name}_start',0)
            # step = self.mlir.none_op
            # end_num = input_shape[0]
            # end = self.mlir.create_weight_op(f'{node.name}_end', end_num)
            # # idx_op = self.mlir.create_constant_weight_op(f'{node.name}_indice',[end_num],np.arange(end_num))
            # idx_op = top.ArangeOp(*self.mlir.get_tensor_type([[end_num]], F32Type.get()),
            #             start,
            #             end,
            #             step,
            #             loc=self.get_loc(node.name+"arange"),
            #             ip=self.mlir.insert_point).output
            # new_op = top.IndexPutOp(*self.mlir.get_tensor_type(self.mlir.get_output_shapes(node), F32Type.get()),
            #         op0,
            #         idx_op,
            #         value,
            #         accumulate = accumulate,
            #         loc=self.get_loc(node.name),
            #         ip=self.mlir.insert_point).output
            # self.operands[node] = new_op

            op0 = self.operands[node.args[0]]
            self.operands[node] = op0

            # op0 = self.operands[node.args[0]]
            # value = self.operands[node.args[2]]
            # accumulate = node.args[3]
            # dtype = self.mlir.get_output_dtypes(node)
            # idx_store = []
            # for k,idx in enumerate(node.args[1]):
            #     if idx is not None:
            #         indices = [k,idx]
            #         idx_store.append(indices)
            # # shape process
            # for i in range(len(idx_store)):
            #     nd = idx_store[i]
            #     if len(list(nd[1].meta['val'].size()))!= 1:
            #         target_shape = list(nd[1].meta['val'].size())
            #         pop_idx = target_shape.index(1)
            #         target_shape.pop(pop_idx)
            #         tmp_op = self.operands[nd[1]]
            #         out = top.SqueezeOp(*self.mlir.get_tensor_type([target_shape], F32Type.get()),
            #                             tmp_op,
            #                             axes=[pop_idx],
            #                             loc=self.get_loc(node.name+"_shape_update_"+str(i)),
            #                             ip=self.mlir.insert_point).output
            #         idx_store[i][1] = out
            #     else:
            #         target_shape = list(nd[1].meta['val'].size())
            #         idx_store[i][1] = self.operands[nd[1]]
            #     idx_store[i].append(target_shape[0])
            # while idx_store:
            #     info = idx_store[0]
            #     axis = info[0]
            #     indices = info[1]
            #     new_op = top.IndexPutOp(*self.mlir.get_tensor_type(self.mlir.get_output_shapes(node), F32Type.get()),
            #                     op0,
            #                     indices,
            #                     value,
            #                     accumulate = accumulate,
            #                     loc=self.get_loc(node.name+"_"+str(axis)),
            #                     ip=self.mlir.insert_point).output
            #     op0 = new_op
            #     idx_store.pop(0)
            # self.operands[node] = new_op

    def convert_group_norm_op(
        self, node
    ):  #(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps)
        dtype = self.mlir.get_output_dtypes(node)
        input = self.operands[node.args[0]]
        weight = self.operands[node.args[1]]
        bias = self.operands[node.args[2]]
        group = node.args[-2]
        eps = node.args[-1]
        new_op = top.GroupNormTrainOp(
            self.mlir.get_tensor_type([]),
            self.mlir.get_tensor_type([]),
            self.mlir.get_tensor_type([]),
            input,
            weight,
            bias,
            group,
            eps,
            loc=self.get_loc([node.name + ".output", node.name + ".mean", node.name + ".vars"]),
            ip=self.mlir.insert_point)
        self.operands[node] = [new_op.output, new_op.mean, new_op.rstd]

    def convert_gelu_op(self, node):
        op = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        new_op = top.GELUOp(self.mlir.get_tensor_type([]),
                            op,
                            loc=self.get_loc(node),
                            ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_pad_op(self, node, mode):
        op = self.operands[node.args[0]]
        dtype = self.mlir.get_output_dtypes(node)
        padding = node.args[1]
        if len(node.args) >= 3:
            val = node.args[2]
        else:
            val = 0
        new_op = top.PadOp(self.mlir.get_tensor_type([]),
                           op,
                           paddings=padding,
                           val=val,
                           mode=StringAttr.get(mode),
                           loc=self.get_loc(node.name),
                           ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_argmax_op(self, node):
        dtype = self.mlir.get_output_dtypes(node)
        op0 = self.operands[node.args[0]]
        dim = node.args[1]
        if len(node.args) > 2:
            keepdims = node.args[2]
        else:
            keepdims = False
        select_last_index = True  # select_last_index = False
        new_op = top.ArgOp(self.mlir.get_tensor_type([]),
                           self.mlir.get_tensor_type([]),
                           input=op0,
                           axis=dim,
                           keepdims=keepdims,
                           mode=StringAttr.get("ArgMax"),
                           select_last_index=select_last_index,
                           loc=self.get_loc(node.name),
                           ip=self.mlir.insert_point)
        out_ops = new_op.indices
        self.operands[node] = out_ops

    def convert_scatter_op(self, node):  #(zeros_like, 1, where, -1.0)
        dtype = self.mlir.get_output_dtypes(node)
        input = self.operands[node.args[0]]
        axis = node.args[1]
        index = self.operands[node.args[2]]
        index_shape = list(node.args[2].meta['val'].size())
        if isinstance(node.args[3], torch.fx.Node):
            update = self.operands[node.args[3]]
        else:
            if isinstance(node.args[3], float):
                values = np.full(index_shape, node.args[3])
                update = self.mlir.create_weight_op(f'{node.name}_update', values)
        update_op = top.ReshapeOp(self.mlir.get_tensor_type([]),
                                  update,
                                  shape=index_shape,
                                  loc=self.get_loc(node.name + "_reshape"),
                                  ip=self.mlir.insert_point).output

        new_op = top.ScatterElementsOp(self.mlir.get_tensor_type([]),
                                       input,
                                       index,
                                       update_op,
                                       axis,
                                       loc=self.get_loc(node.name),
                                       ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_logical_and_op(self, node):
        dtype = self.mlir.get_output_dtypes(node)
        op0 = self.operands[node.args[0]]
        op1 = self.operands[node.args[1]]
        new_op = top.LogicalAndOp(self.mlir.get_tensor_type([]), [op0, op1],
                                  loc=self.get_loc(node.name),
                                  ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_bernoulli_op(self, node):
        dtype = self.mlir.get_output_dtypes(node)
        if len(node.args) > 1:
            p = node.args[1]
        op0 = self.operands[node.args[0]]
        shape = list(node.meta['val'].size())
        ## create random number
        result = np.zeros(shape)
        # np.random.seed(0)
        for i in range(result.size):
            random_num = np.random.random()
            result.flat[i] = 1 if random_num <= p else 0
        op = self.mlir.create_constant_weight_op(f'{node.name}_random', shape, result)
        self.operands[node] = op

    def convert_rand_op(self, node):
        dtype = self.mlir.get_output_dtypes(node)
        if isinstance(node.args[0], torch.fx.Node):
            shape = list(node.args[0].meta['val'].size())
        else:
            shape = list(node.args[0])
        result = np.random.rand(*shape)
        op = self.mlir.create_constant_weight_op(f'{node.name}_random', shape, result)
        self.operands[node] = op

    def convert_randn_op(self, node):
        dtype = self.mlir.get_output_dtypes(node)
        if isinstance(node.args[0], torch.fx.Node):
            shape = list(node.args[0].meta['val'].size())
        else:
            shape = list(node.args[0])
        result = np.random.randn(*shape)
        op = self.mlir.create_constant_weight_op(f'{node.name}_random', shape, result)
        self.operands[node] = op

    def convert_new_constant_op(self, node, value):
        if len(node.args) > 1:
            shape = node.args[1]
        else:
            shape = list(node.args[0].meta['val'].size())
        op = self.mlir.create_constant_weight_op(f'{node.name}_c', shape, value)
        # dtype = self.mlir.get_output_dtypes(node)
        # new_op = top.ConstantFillOp(*self.mlir.get_tensor_type(self.mlir.get_output_shapes(node), F32Type.get()),
        #                             op,
        #                             value=value,
        #                             loc=self.get_loc(node),
        #                             ip=self.mlir.insert_point).output
        self.operands[node] = op

    def convert_constant_like_op(self, node, value):
        dtype = self.mlir.get_output_dtypes(node)
        op0 = self.operands[node.args[0]]
        size_op = top.SizeOp(self.mlir.get_tensor_type([]),
                             op0,
                             axis=None,
                             loc=self.get_loc(node.name + "_size"),
                             ip=self.mlir.insert_point).output
        new_op = top.ConstantFillOp(self.mlir.get_tensor_type([]),
                                    size_op,
                                    value=value,
                                    loc=self.get_loc(node),
                                    ip=self.mlir.insert_point).output
        self.operands[node] = new_op

    def convert_spit_with_size_op(self, node):
        dtype = self.mlir.get_output_dtypes(node)
        op0 = self.operands[node.args[0]]
        size = node.args[1]
        new_op = top.SplitOp(self.mlir.get_tensor_type([]),
                             op0,
                             size,
                             loc=self.get_loc(node.name),
                             ip=self.mlir.insert_point).outputs
        self.operands[node] = new_op

    def convert_baddbmm_op(self, node):
        """baddbmm: beta*op0 + alpha*op1@op2"""
        op0 = self.operands[node.args[0]]
        op1 = self.operands[node.args[1]]
        op2 = self.operands[node.args[2]]
        val3 = node.kwargs['alpha']
        val4 = node.kwargs['beta']
        dtype = self.mlir.get_output_dtypes(node)
        if val4 == 0 and val3 == 0:  # only zero is need
            new_op3 = top.MulConstOp(self.mlir.get_tensor_type([]),
                                     op0,
                                     val4,
                                     loc=self.get_loc(node.name + "_mul2"),
                                     ip=self.mlir.insert_point).output
            self.operands[node] = new_op3
            return
        elif val3 == 0:  # only beta*op0 is need
            new_op3 = top.MulConstOp(self.mlir.get_tensor_type([]),
                                     op0,
                                     val4,
                                     loc=self.get_loc(node.name + "_mul2"),
                                     ip=self.mlir.insert_point).output
            self.operands[node] = new_op3
            return
        elif val4 == 0:  # only alpha*op1*op2 is need
            new_op = top.MatMulOp(self.mlir.get_tensor_type([]),
                                  op1,
                                  op2,
                                  self.mlir.none_op,
                                  loc=self.get_loc(node.name + "_matmul"),
                                  ip=self.mlir.insert_point).output
            new_op2 = top.MulConstOp(self.mlir.get_tensor_type([]),
                                     new_op,
                                     val3,
                                     loc=self.get_loc(node.name + "_mul"),
                                     ip=self.mlir.insert_point).output
            self.operands[node] = new_op2
            return
        elif val4 == 1 and val3 == 1:  # only op1*op2 + op0 is need
            new_op = top.MatMulOp(self.mlir.get_tensor_type([]),
                                  op1,
                                  op2,
                                  self.mlir.none_op,
                                  loc=self.get_loc(node.name + "_matmul"),
                                  ip=self.mlir.insert_point).output
            new_op2 = top.AddOp(self.mlir.get_tensor_type([]), [new_op, op0],
                                do_relu=False,
                                loc=self.get_loc(node.name + "_add"),
                                ip=self.mlir.insert_point).output
            self.operands[node] = new_op2
            return
        new_op = top.MatMulOp(self.mlir.get_tensor_type([]),
                              op1,
                              op2,
                              self.mlir.none_op,
                              loc=self.get_loc(node.name + "_matmul"),
                              ip=self.mlir.insert_point).output
        # new_op2: new_op * op3
        new_op2 = top.MulConstOp(self.mlir.get_tensor_type([]),
                                 new_op,
                                 val3,
                                 loc=self.get_loc(node.name + "_mul"),
                                 ip=self.mlir.insert_point).output
        # new_op3: op4 * op0
        new_op3 = top.MulConstOp(self.mlir.get_tensor_type([]),
                                 op0,
                                 val4,
                                 loc=self.get_loc(node.name + "_mul2"),
                                 ip=self.mlir.insert_point).output
        # new_op4: new_op2 + new_op3
        new_op4 = top.AddOp(self.mlir.get_tensor_type([]), [new_op2, new_op3],
                            do_relu=False,
                            loc=self.get_loc(node.name + "_add"),
                            ip=self.mlir.insert_point).output
        self.operands[node] = new_op4

    def get_op_type_by_node(self, node):
        return torch.typename(node.target).split('.')[-1]

    def empty_baddbmm_matcher(self, node):
        if self.get_op_type_by_node(node) == 'empty':
            for user in node.users:
                if self.get_op_type_by_node(user) != 'baddbmm':
                    return False
                if 'beta' not in user.kwargs or ('beta' in user.kwargs
                                                 and user.kwargs['beta'] != 0):
                    return False
            self.processed_op.append(node)
            return True
        return False

    def empty_baddbmm_folder(self, node):
        pass

    def sigmoid_matcher(self, node):
        if self.get_op_type_by_node(node) == 'empty_like':
            subnet_input = node.args[0]
            assert (len(node.users) == 1)
            next_node = next(iter(node.users))
            matched_ops = [node, next_node]
            if self.get_op_type_by_node(next_node) == 'fill':
                assert (len(next_node.users) == 1)
                next_node = next(iter(next_node.users))
                matched_ops.append(next_node)
                if self.get_op_type_by_node(next_node) == 'sub':
                    for i in next_node.args:
                        if i == subnet_input:
                            self.processed_op.extend(matched_ops)
                            return True
        return False

    def sigmoid_folder(self, node):
        last_op = next(iter(node.users))
        last_op = next(iter(last_op.users))
        out_shape = self.mlir.get_output_shapes(last_op)
        shape = self.mlir.get_output_shapes(node.args[0])[0]
        new_shape = [1] * len(shape)
        new_shape[1] = shape[1]
        in_value = self.operands[node.args[0]]
        # self.mlir.create_weight_op(f'scalar_tensorOp_{node.name}', node.args[0])
        scale = self.mlir.create_constant_weight_op(f'sigmoid_folder_{node.name}_scale', new_shape,
                                                    -1)
        bias = self.mlir.create_constant_weight_op(f'sigmoid_folder_{node.name}_bias', new_shape, 1)
        dtype = self.mlir.get_output_dtypes(last_op)
        out = top.ScaleOp(*self.mlir.get_tensor_type(out_shape, dtype),
                          in_value,
                          scale,
                          bias,
                          loc=self.get_loc(node.name),
                          ip=self.mlir.insert_point).output
        self.operands[last_op] = out
