import os
import copy
import sys
import torch
import pdb
import gc
import numpy as np
import importlib
from argparse import Namespace
MIN_BLOCK_SIZE = 5
from mlir.ir import *
import mlir.dialects.top as top
from tools.train.TpuMlirModule import TpuMlirModule
from utils.mlir_shell import mlir_opt_for_top, mlir_lowering, mlir_to_model, f32_blobs_compare
from tools.model_runner import mlir_inference, free_mlir_module

sys.path.append('/workspace/tpu-mlir/python/')

class State:
    TOP_F32 = 'TOP_F32'
    TOP_F16 = 'TOP_F16'
    TOP_BF16 = 'TOP_F32'

def bmodel_inference(model_file, input_dict):
    pyruntime = importlib.import_module("pyruntime_bm")
    model = pyruntime.Model(model_file)
    net = model.Net(model.networks[0])

    input_shapes = []
    for i in net.inputs:
        input_shapes.append(i.data.shape)
        i.data[:] = input_dict[i.name]

    dyn_output_shapes = net.forward_dynamic(input_shapes)

    tpu_outputs = {}
    dyn_idx = 0
    for i in net.outputs:
        output = np.array(i.data)
        if output.shape != dyn_output_shapes[dyn_idx]:
            dyn_len = np.prod(dyn_output_shapes[dyn_idx])
            output = output.flatten()[:dyn_len].reshape(*dyn_output_shapes[dyn_idx])
            dyn_idx += 1
        tpu_outputs[i.name] = output
    return tpu_outputs

def NoneAndRaise(node):
    raise RuntimeError("{} Op not support now".format(node.target))

class fx2mlir(object):
    def __init__(self,
                 submodule_name: str,
                 args:Namespace,
                 bwd_graph:bool
                ):
        self.work_dir = submodule_name.split('_')[0]
        tmp = 'bwd' if bwd_graph else 'fwd'
        self.model_name = f'{submodule_name}_{tmp}'
        self.args = args
        self.bwd = bwd_graph
        self.bmodel_path = None
        self.ctx = Context()
        self.ctx.allow_unregistered_dialects = True
        loc = Location.unknown(self.ctx)
        self.ctx.__enter__()
        loc.__enter__()

        self.weight_file = f'graph_for_jit_{self.model_name}.npz'
        self.input_nodes = []
        self.output_nodes = []
        self.output_dtypes = []
        self.return_none_count = 0
        self.operands = dict()
        self.weights_data = dict()
        self.load_weight = dict()
        self.const_val = dict()
        self.num_core = 1
        if self.args.chip =="sg2260":
            self.num_core = 8

        self.op_factory = {
            #############################
            # Torch Convert, Alphabetically
            #############################
            "convolution": lambda node: self.convert_base_conv_op(node),
            "convolution_backward": lambda node: self.convert_backward_conv_op(node),
            "permute": lambda node: self.convert_permute_op(node),
            "relu": lambda node: self.convert_relu_op(node),
            "max_pool2d_with_indices": lambda node: self.convert_maxpool2d_with_indices_op(node),
            "add": lambda node: self.convert_add_op(node),
            "mul": lambda node: self.convert_mul_op(node),
            "view": lambda node: self.convert_reshape_op(node),
            "_unsafe_view": lambda node: self.convert_reshape_op(node),
            "reshape": lambda node: self.convert_reshape_op(node),
            "mm": lambda node: self.convert_matmul_op(node),
            "bmm": lambda node: self.convert_matmul_op(node),
            "matmul": lambda node: self.convert_matmul_op(node),
            "squeeze": lambda node: self.convert_squeeze_op(node),
            "unsqueeze": lambda node: self.convert_unsqueeze_op(node),
            "getitem": lambda node: self.convert_getitem_op(node),
            "to": lambda node: self.convert_to_op(node),
            "cat": lambda node: self.convert_concat_op(node),
            "sum": lambda node: self.convert_sum_op(node),
            "mean": lambda node: self.convert_mean_op(node),
            "clone": lambda node: self.convert_clone_op(node),
            "_native_batch_norm_legit_functional": lambda node: self.convert_batch_norm_op(node),
            "native_batch_norm_backward": lambda node: self.convert_batch_norm_backward_op(node),
            "full":lambda node: self.convert_full_op(node),
            "arange":lambda node: self.convert_arange_op(node),
            "scalar_tensor":lambda node:self.convert_scalar_tensor_op(node),
            "slice":lambda node:self.convert_slice_op(node),
            "embedding":lambda node:self.convert_embedding_op(node),
            "ne":lambda node: self.convert_compare_op(node, "NotEqual"),
            "where":lambda node: self.convert_where_op(node),
            "_to_copy":lambda node: self.convert_copy_op(node),
            "var":lambda node:self.convert_var_op(node),
            "div":lambda node:self.convert_div_op(node),
            "rsqrt":lambda node:self.convert_rsqrt_op(node),
            "sub":lambda node:self.convert_sub_op(node),
            "addmm":lambda node:self.convert_addmm_op(node),
            "split":lambda node:self.convert_split_op(node),
            "expand":lambda node:self.convert_expand_op(node),
            "amax":lambda node:self.convert_amax_op(node,False),
            "exp":lambda node:self.convert_math_op(node,'exp'),
            "erf":lambda node:self.convert_math_op(node,'erf'),
            "select":lambda node:self.convert_select_op(node),
            "log":lambda node:self.convert_math_op(node,'log'),
            "gather":lambda node:self.convert_gather_op(node),
            "neg":lambda node:self.convert_neg_op(node),
            "t":lambda node:self.convert_transpose_op(node),
            "native_layer_norm":lambda node:self.convert_layer_norm_op(node),
            "native_layer_norm_backward":lambda node:self.convert_layer_norm_backward_op(node),
            "transpose":lambda node:self.convert_transpose_op(node),
            "_softmax":lambda node:self.convert_softmax_op(node,log = False),
            "_log_softmax":lambda node: self.convert_softmax_op(node, log=True),
            "nll_loss_forward":lambda node:self.convert_nllloss_op(node),
            "le":lambda node:self.convert_compare_op(node,"LessOrEqual"),
            "sigmoid":lambda node:self.convert_sigmoid_op(node),
            "silu":lambda node:self.convert_silu_op(node),
            "pow":lambda node:self.convert_pow_op(node),
            "_mseloss":lambda node:self.convert_mse_op(node),
            "max_pool2d_with_indices_backward":lambda node:self.convert_maxpool2d_backward_op(node),
            "hardswish":lambda node:self.convert_hardswish_op(node),
            "leaky_relu":lambda node:self.convert_leaky_relu_op(node),
            "gt":lambda node:self.convert_compare_op(node,"Greater"),
            "lt":lambda node:self.convert_compare_op(node,"Less"),
            "new_zeros":lambda node:self.convert_zero_op(node),
            "rsub":lambda node:self.convert_sub_op(node,is_reverse = True),
            "clamp":lambda node:self.convert_clamp_op(node),
            "masked_fill":lambda node:self.convert_masked_fill_op(node),
            "index":lambda node:self.convert_index_op(node),
            "select_scatter":lambda node:self.convert_select_scatter_op(node),
            "slice_scatter":lambda node:self.convert_slice_scatter_op(node),
            "index_put":lambda node:self.convert_index_put_op(node),
            "tanh":lambda node:self.convert_math_op(node,"tanh"),
            "sin":lambda node:self.convert_math_op(node,"sin"),
            "cos":lambda node:self.convert_math_op(node,"cos"),
            "native_group_norm":lambda node:self.convert_group_norm_op(node),
            "gelu":lambda node:self.convert_gelu_op(node),
            "empty_like":lambda node:self.convert_zero_op(node),
            "fill":lambda node:self.convert_full_op(node),
            "constant_pad_nd":lambda node:self.convert_pad_op(node,'constant'),
            "argmax":lambda node:self.convert_argmax_op(node),
            "zeros_like":lambda node:self.convert_zero_op(node),
            "scatter":lambda node:self.convert_scatter_op(node),
            "logical_and":lambda node:self.convert_logical_and_op(node),
            "zeros":lambda node:self.convert_zero_op(node),
            "bernoulli":lambda node:self.convert_bernoulli_op(node),
            ####################################################
            "constant":lambda node:self.convert_constant(node),
            'threshold_backward':lambda node:self.convert_threshold_backward_op(node),
            '_softmax_backward_data':lambda node:self.convert_softmax_backward_data_op(node),
            'embedding_dense_backward':lambda node:self.convert_embedding_dense_backward_op(node),
            ######### prims op ##############################
            "ge":lambda node:self.convert_compare_op(node,"GreaterOrEqual"),
            "eq":lambda node:self.convert_compare_op(node,"Equal"),
            "trunc":lambda node:self.convert_trunc_op(node),
            "broadcast_in_dim":lambda node:self.convert_broadcast_op(node),

        }

        self.mlir_type = {
            "F32": F32Type.get(),
            "F16": F16Type.get(),
            "BF16": BF16Type.get()
        }

    def convert_a_op(self, node):
        print(f'convert_a_op, node.name:', node.name, 'target:',node.target, 'args:', node.args, 'users:', list(node.users.keys()), 'kwargs:', node.kwargs
              , 'val:', node.meta['val'] if 'val' in node.meta else 'None', 'tensor_meta:', node.meta['tensor_meta'] if 'tensor_meta' in node.meta else 'None')
        print('in shapes:', [list(i.meta['val'].shape) for i in node.args if isinstance(i, torch.fx.Node)])
        op_type = torch.typename(node.target).split('.')[-1]
        if op_type not in self.op_factory:
            print(f'{op_type} not in op_factory')
            return None,None
        in_args_txt_list = []
        in_ref_data = {}
        for i, arg in enumerate(node.args):
            if isinstance(arg, torch.fx.Node):
                self.input_nodes.append([i, arg])
                shape = list(arg.meta['val'].size())
                shape = [1] if shape == [] else shape
                if 'val' in arg.meta and arg.meta['val'].dtype == torch.int64:
                    in_ref_data[arg.name] = torch.randint(0, 10, shape)
                else:
                    in_ref_data[arg.name] = torch.randn(shape)
                in_args_txt_list.append("%args{}: {} loc(unknown)".format(i, RankedTensorType.get(shape, F32Type.get()).__str__()))
        np.savez(f'in_ref_data_a_op_{node.name}.npz', **in_ref_data)

        if isinstance(node.meta['val'], (tuple,list)):
            self.output_dtypes = [i.dtype for i in node.meta['val'] if i is not None]
        else:
            self.output_dtypes.append(node.meta['val'].dtype)

        output_txt = []
        for shape, dtype in zip(self.get_output_shapes(node), self.get_output_dtypes(node)):
            output_txt.append(self.get_tensor_type(shape, dtype).__str__())
        output_str = ', '.join(output_txt)
        if len(output_txt) > 1:
            output_str = "({})".format(output_str)
        # if isinstance(node.meta['val'], (list,tuple)) and len(node.meta['val']) > 1:
        #     output_txt = ', '.join([f'{self.get_tensor_type(list(o.size()), self.get_output_dtypes(node)).__str__()}' for o in node.meta['val'] if o is not None])
        #     output_txt = "({})".format(output_txt)
        # else:
        #     shape = list(node.meta['val'].size())
        #     output_txt = f'{self.get_tensor_type(shape, self.get_output_dtypes(node)).__str__()}'

        self.createMlirModuleAndInput(', '.join(in_args_txt_list), output_str)
        self.op_factory.get(op_type, lambda x: NoneAndRaise(x))(node)
        operands = []
        if isinstance(self.operands[node], list):
            operands.extend(self.operands[node])
        else:
            operands.append(self.operands[node])
        return_op = Operation.create("func.return", operands=operands, results=[])
        self.insert_point.insert(return_op)

        mlir_txt = self.mlir_module.operation.get_asm(enable_debug_info=True)
        mlir_file = 'out.mlir'
        mlir_origin = mlir_file.replace('.mlir', '_origin.mlir', 1)
        with open(mlir_origin, "w") as f:
            f.write(mlir_txt)
        self.WeightToNpz(self.weight_file)

        mlir_opt_for_top(mlir_origin, mlir_file)
        print("Save mlir file: {}".format(mlir_file))
        if self.args.cmp:
            tensors = mlir_inference(in_ref_data, mlir_file, True)
            print('out num:', len(tensors))
            np.savez('ref_data.npz', **tensors)
            del tensors
            free_mlir_module()
            gc.collect()

        tpu_ir = 'tpu_'+mlir_file
        self.bmodel_path = tpu_ir+'.bmodel'
        mlir_lowering(mlir_file, tpu_ir, 'F32', self.args.chip)
        if self.args.cmp:
            tensors = mlir_inference(in_ref_data, tpu_ir, True)
            np.savez('tpu_ir_out_data.npz', **tensors)
            del tensors
            free_mlir_module()
            gc.collect()
            f32_blobs_compare('tpu_ir_out_data.npz', 'ref_data.npz', '0.99,0.99')

        mlir_to_model(tpu_ir, self.bmodel_path, 'final_'+mlir_file)
        if self.args.cmp:
            tensors = bmodel_inference(self.bmodel_path, in_ref_data)
            np.savez('bmodel_out_data.npz', **tensors)
            del tensors
            gc.collect()
            f32_blobs_compare('bmodel_out_data.npz', 'ref_data.npz', '0.99,0.99')

        print('jit compile ok')
        return TpuMlirModule(self.bmodel_path, self.output_dtypes, self.return_none_count), list(in_ref_data.values())

    def convert(self, module):
        print('>>>>>>>>>> starting parsing...')
        module.to_folder(f'fx_graph_dumped_{self.model_name}', self.model_name)
        with open(f'fx_graph_dumped_{self.model_name}/input_shape', "w+") as fd:
            for i, node in enumerate(module.graph.nodes):
                if node.op == 'placeholder':
                    fd.write(f"{node.name}*{list(node.meta['val'].size())}*{node.meta['val'].dtype}\n")

        in_ref_data = {}
        for i, node in enumerate(module.graph.nodes):
            if node.op == 'placeholder':
                shape = list(node.meta['val'].size())
                print(f'>>> {i}th op, placeholder:', node.name, 'shape:', shape, 'val:', node.meta['val'])
                self.input_nodes.append([i, node])
                tmp =  np.random.rand(*shape) if node.meta['val'].dtype == torch.float32 else  np.random.randint(0,1,shape)
                if node.meta['val'].dtype == torch.bool:
                    tmp = tmp.astype(np.bool_)
                in_ref_data[node.name+'_weight_or_param' if node.meta['tensor_meta'].requires_grad else node.name] = tmp
        np.savez(f'in_ref_data_{self.model_name}.npz', **in_ref_data)

        # outs = module.forward(*[torch.from_numpy(i) for i in list(in_ref_data.values())])
        # print(f'out num:{len(outs)}', outs[0][0])

        self.input_nodes = sorted(self.input_nodes, key=lambda x:x[0], reverse=False)
        in_args_txt_list = []
        for node in self.input_nodes:
            shape = list(node[1].meta['val'].size())
            shape = [1] if shape == [] else shape
            in_args_txt_list.append("%args{}: {} loc(unknown)".format(node[0], RankedTensorType.get(shape, F32Type.get()).__str__()))

        first_call_op = True
        for i, node in enumerate(module.graph.nodes):
            if node.op == 'call_module' or node.op == 'call_method' or node.op == 'call_function':
                print(f'>>>> {i}th op, new op start:', node.name, 'val:', node.meta['val'] if 'val' in node.meta else 'None', 'tensor_meta:', node.meta['tensor_meta'] if 'tensor_meta' in node.meta else 'None')
                if first_call_op:
                    output_args_txt = self.parseOutputNode([i for i in module.graph.nodes if i.op == 'output' and len(i.args) > 0][0])
                    self.createMlirModuleAndInput(', '.join(in_args_txt_list), output_args_txt)
                    first_call_op = False
                op_type = torch.typename(node.target).split('.')[-1]
                print(f'{i}th op, node.name:', node.name, 'target:',node.target, 'op_type:', op_type, 'args:', node.args, 'users:', list(node.users.keys()), 'kwargs:', node.kwargs)
                self.op_factory.get(op_type, lambda x: NoneAndRaise(x))(node)

        # add return op
        return_op = list()
        for idx, _name in enumerate(self.output_nodes):
            if _name is not None:
                return_op.append(self.operands[_name])
            else:
                self.return_none_count += 1

        self.create_return_op(return_op)
        mlir_txt = self.mlir_module.operation.get_asm(enable_debug_info=True)
        mlir_file = f'out_{self.model_name}.mlir'
        mlir_origin = mlir_file.replace('.mlir', '_origin.mlir', 1)
        with open(mlir_origin, "w") as f:
            f.write(mlir_txt)
        self.WeightToNpz(self.weight_file)

        mlir_opt_for_top(mlir_origin, mlir_file)
        print("Save mlir file: {}".format(mlir_file))
        if self.args.cmp:
            tensors = mlir_inference(in_ref_data, mlir_file, True)
            print('out num:', len(tensors))
            np.savez('ref_data.npz', **tensors)
            del tensors
            free_mlir_module()
            gc.collect()

        tpu_ir = 'tpu_'+mlir_file
        self.bmodel_path = os.path.join(self.work_dir, tpu_ir+'.bmodel')
        mlir_lowering(mlir_file, tpu_ir, 'F32', self.args.chip)
        if self.args.cmp:
            tensors = mlir_inference(in_ref_data, tpu_ir, True)
            np.savez('tpu_ir_out_data.npz', **tensors)
            del tensors
            free_mlir_module()
            gc.collect()
            f32_blobs_compare('tpu_ir_out_data.npz', 'ref_data.npz', '0.99,0.99')

        mlir_to_model(tpu_ir, self.bmodel_path, 'final_'+mlir_file,num_core=self.num_core)
        if self.args.cmp:
            tensors = bmodel_inference(self.bmodel_path, in_ref_data)
            np.savez('bmodel_out_data.npz', **tensors)
            del tensors
            gc.collect()
            f32_blobs_compare('bmodel_out_data.npz', 'ref_data.npz', '0.99,0.99')

        print('jit compile ok, start cmp')
        mlir_mod = TpuMlirModule(self.bmodel_path, self.output_dtypes, self.return_none_count)
        # in_ref_data = list(in_ref_data.values())
        # outs = mlir_mod(*in_ref_data) #运行生成的bmodel
        # 在另一个进程中执行估计有可能能成功
        # in_ref_data = [torch.from_numpy(i) for i in in_ref_data]
        # exec(f'from fx_graph_dumped.module import {self.model_name}')
        # from torch.fx._symbolic_trace import symbolic_trace
        # mod = eval(f'symbolic_trace({self.model_name}())')
        # ref_outs = mod(*in_ref_data)
        # from tools.train.tpu_mlir_jit import cosine_similarity
        # cosine_similarity(ref_outs, outs)
        # print('compile end')
        return mlir_mod

    def parseOutputNode(self, node):
        assert node.op == 'output'
        self.output_nodes = node.args[0]
        output_shapes = [list(i.meta['tensor_meta'].shape) for i in node.args[0] if i is not None]
        output_shapes = [[1] if i == [] else i for i in output_shapes]
        self.output_dtypes = [i.meta['val'].dtype for i in node.args[0] if i is not None]
        assert len(output_shapes) == len(self.output_dtypes)
        output_txt = ','.join([f'{self.get_tensor_type(shape, self.get_dtype(dtype)).__str__()}' for shape, dtype in zip(output_shapes, self.output_dtypes)])
        if len(output_shapes) > 1:
            output_txt = "({})".format(output_txt)
        return output_txt

    def createMlirModuleAndInput(self, in_args_txt, output_args_txt):
        num_output = len(output_args_txt.split(','))
        result_var_name = "%1"
        result_types = output_args_txt
        if num_output > 1:
            result_var_name = ",".join([f"%1#{var_id}" for var_id in range(num_output)])
            result_types = output_args_txt[1:-1]
        main_func = """
            module attributes {{sym_name = \"{name}\", module.weight_file= \"{weight_file}\", module.platform=\"TORCH\", module.state=\"{state}\", module.chip=\"{chip}\", module.train=\"{train}\"}} {{
                func.func @main({args}) -> {output} {{
                    %0 = \"top.None\"() : () -> none loc(unknown)
                    %1:{last_output_num} = \"Placeholder.Op\"() : () -> {output}
                    return {result_var} : {result_types}
                }} loc(unknown)
            }} loc(unknown)
        """.format(name=self.model_name,
                    weight_file=self.weight_file,
                    state=State.TOP_F32,
                    chip="ALL",
                    train='true',
                    args=in_args_txt,
                    last_output_num=num_output,
                    result_var=result_var_name,
                    result_types=result_types,
                    output=output_args_txt)
        print(f'main_func:\n{main_func}\nmain_func end')
        self.mlir_module = Module.parse(main_func, self.ctx)
        func = self.mlir_module.body.operations[0]
        entry_block = func.regions[0].blocks[0]
        self.insert_point = InsertionPoint(entry_block)
        self.none_op = entry_block.operations[0].operation.results[0]
        entry_block.operations[2].operation.erase()
        entry_block.operations[1].operation.erase()
        for node, arg in zip(self.input_nodes, entry_block.arguments):
            self.create_input_op(node[1], arg)

    def get_loc(self, names):
        if isinstance(names, str):
            return Location.fused([Location.name(names)], context=self.ctx)
        elif isinstance(names, list):
            return Location.fused([Location.name(n) for n in names], context=self.ctx)
        elif isinstance(names, torch.fx.Node):
            return Location.fused([Location.name(names.name)], context=self.ctx)
        else:
            raise RuntimeError("Unknown names:{}".format(names))

    def convert_scalar_param(self, scalar):
        if not isinstance(scalar, (int, float)):
            return np.atleast_2d(scalar).astype(np.float32)
        else:
            return np.atleast_1d(scalar).astype(np.float32)

    # shape: [] => [* x f32]; None => NoneType; [None, None] => [NoneType, NoneType]
    # type: None => f32; or type
    def get_tensor_type(self, output_shapes, type=None):
        if type is None:
            type = F32Type.get()
        if output_shapes == []:
            return UnrankedTensorType.get(type)
        if output_shapes is None:
            return NoneType.get()
        if isinstance(output_shapes, tuple):
            output_shapes = list(output_shapes)
        assert (isinstance(output_shapes, list))
        assert (len(output_shapes) > 0)
        if not isinstance(output_shapes[0], list) and output_shapes[0] is not None:
            return RankedTensorType.get(tuple(output_shapes), type) #why?
        # multi output
        out_types = []
        if isinstance(type, list):
            for s,t in zip(output_shapes, type):
                if s == []:
                    out_types.append(UnrankedTensorType.get(t))
                elif s is None:
                    out_types.append(NoneType.get())
                else:
                    out_types.append(RankedTensorType.get(tuple(s), t))
        else:
            for s in output_shapes:
                if s == []:
                    out_types.append(UnrankedTensorType.get(type))
                elif s is None:
                    out_types.append(NoneType.get())
                else:
                    out_types.append(RankedTensorType.get(tuple(s), type))
        return out_types


    def get_dtype(self, type1):
        # return F32Type.get()  #??? todo
        dtype = None
        if type1 == torch.float16:
            dtype = F16Type.get()
        elif type1 == torch.float32:
            dtype = F32Type.get()
        elif type1 == torch.int64:
            dtype = IntegerType.get_signless(64)
        elif type1 == torch.int32:
            dtype = IntegerType.get_signless(32)
        return dtype

    def get_output_dtypes(self, node):
        dtypes = []
        if 'val' in node.meta:
            if isinstance(node.meta['val'], (tuple,list)):
                dtypes = [i.dtype if i is not None else None for i in node.meta['val'] ]
            else:
                dtypes.append(node.meta['val'].dtype)
        else:
            dtypes.append(torch.float16)
        dtypes = [self.get_dtype(i) if i is not None else None for i in dtypes]
        # if len(dtypes) == 1:
        #     dtypes = dtypes[0]
        return dtypes

    def get_output_shapes(self, node, exclude_num = 0):
        shapes = []
        if isinstance(node.meta['val'], (tuple,list)):
            shapes = [list(i.size()) if i is not None else None for i in node.meta['val']]
        else:
            shapes.append(list(node.meta['val'].size()))

        shapes = [[1] if i is not None and i == [] else i for i in shapes]
        for _ in range(exclude_num):
            shapes.pop()
        return shapes

    def create_input_op(self, node, func_arg):
        init_args = {}
        output_shapes = self.get_output_shapes(node)
        if not self.bwd and 'tensor_meta' in node.meta and node.meta['tensor_meta'].requires_grad:
            init_args["loc"] = self.get_loc(f'{node.name}_weight_or_param')
            init_args["ip"] = self.insert_point
            init_args["input"] = func_arg
            init_args["output"] = RankedTensorType.get(output_shapes[0], F32Type.get())
            input_op = top.InputOp(**init_args).output
            if node.meta['tensor_meta'].dtype != torch.float32:
                new_op2 = top.WeightReorderOp(*self.get_tensor_type(output_shapes, F32Type.get()),
                                            input_op,
                                            loc=self.get_loc(node),
                                            ip=self.insert_point).output
                self.operands[node] = new_op2
                return
            self.operands[node] = input_op
        else:
            init_args["loc"] = self.get_loc(node)
            init_args["ip"] = self.insert_point
            init_args["input"] = func_arg
            init_args["output"] = RankedTensorType.get(output_shapes[0], F32Type.get())
            input_op = top.InputOp(**init_args)
            self.operands[node] = input_op.output

    # unranked_type = UnrankedTensorType.get(F32Type.get())
    def convert_permute_op(self, node):
        op = self.operands[node.args[0]]
        order = node.args[1]
        dtype = self.get_output_dtypes(node)
        new_op = top.PermuteOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                                op,
                                order=order,
                                loc=self.get_loc(node),
                                ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_relu_op(self, node):
        op = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        new_op = top.ReluOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                            op,
                            loc=self.get_loc(node),
                            ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_full_op(self, node):
        if node.args[0] == []:
            self.operands[node] = self.create_weight_op(f'fullOp_{node.name}_c', node.args[1])
            # value = self.convert_scalar_param(node.args[1])
            # dtype = self.get_output_dtypes(node)
            # new_op = top.WeightOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
            #                 scale = value,
            #                 loc=self.get_loc(node),
            #                 ip=self.insert_point).output
        elif node.args[0] not in self.operands:
            name = f'fullOp_{node.name}_c'
            data_type = "F32"
            tensor_type = RankedTensorType.get(list(node.args[0]), self.mlir_type[data_type])
            op = Operation.create("top.Weight",
                              results=[tensor_type],
                              loc=Location.fused([Location.name(name)]))
            self.insert_point.insert(op)
            result = op.results[0]
            self.load_weight[name] = (result, node.args[0], data_type)
            shape = tuple(node.args[0])
            self.weights_data[name] = np.full(shape,node.args[1],dtype = np.float32)
            # dtype = self.get_output_dtypes(node)
            # new_op = top.ConstantFillOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
            #                             op,
            #                             value=node.args[1],
            #                             loc=self.get_loc(node),
            #                             ip=self.insert_point).output
            if name in self.load_weight:
                _op, _shape, _type = self.load_weight[name]
                self.operands[node] = _op
            else:
                self.operands[node] = result
            # self.operands[node] = new_op
        else:
            dtype = self.get_output_dtypes(node)
            op0 = self.operands[node.args[0]]
            new_op = top.ConstantFillOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                                        op0,
                                        value=node.args[1],
                                        loc=self.get_loc(node),
                                        ip=self.insert_point).output
            self.operands[node] = new_op



    def convert_scalar_tensor_op(self, node):
        self.operands[node] = self.create_weight_op(f'scalar_tensorOp_{node.name}', node.args[0])

    def convert_div_op(self, node):
        if node.args[0] in self.operands:

            in1 = self.operands[node.args[0]]
        else:
            in1 = self.create_weight_op(f'divOp_{node.name}_input1', node.args[0])

        if node.args[1] in self.operands:
            in2 = self.operands[node.args[1]]
        else:
            in2 = self.create_weight_op(f'divOp_{node.name}_input2', node.args[1])

        dtype = self.get_output_dtypes(node)
        new_op = top.DivOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                           [in1, in2],
                           loc=self.get_loc(node.name),
                            ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_softmax_backward_data_op(self,node):
        grad_output = self.operands[node.args[0]]
        output = self.operands[node.args[1]]
        dtype = self.get_output_dtypes(node)
        new_op = top.SoftmaxBwdOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                            grad_output,
                            output,
                            dim = node.args[2],
                            loc=self.get_loc(node.name),
                            ip=self.insert_point).grad_input
        self.operands[node] = new_op

    def convert_threshold_backward_op(self, node):
        grad_out = self.operands[node.args[0]]
        shape_z = list(node.args[1].meta['val'].size())
        dtype = self.get_output_dtypes(node)
        self_ = self.operands[node.args[1]]
        threshold = node.args[2]
        shape = list(node.meta['val'].size())
        # condition = top.ReluOp(*self.get_tensor_type([shape_z], dtype),
        #                     self_,
        #                     loc=self.get_loc(node.name+'_condition'),
        #                     ip=self.insert_point).output
        x_is_const = False
        y_is_const = True
        x_const_val = y_const_val = threshold
        new_op = top.WhereOp(*self.get_tensor_type([shape], dtype),
                                self_,
                                grad_out,
                                self.none_op,
                                x_is_const = x_is_const,
                                y_is_const = y_is_const,
                                x_const_val = x_const_val,
                                y_const_val = y_const_val,
                                loc=self.get_loc(node.name),
                                ip=self.insert_point).output

        self.operands[node] = new_op

    def convert_add_op(self, node):
        op0 = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        shape = self.get_output_shapes(node)
        if isinstance(node.args[1], torch.fx.Node):
            op1 = self.operands[node.args[1]]
            new_op = top.AddOp(*self.get_tensor_type(shape, dtype), [op0, op1],
                                do_relu=False,
                                coeff = np.atleast_1d(node.kwargs['alpha']).astype(np.float32) if 'alpha' in node.kwargs else None,
                                loc=self.get_loc(node),
                                ip=self.insert_point).output
        else:
            op1 = np.atleast_1d(node.args[1]).astype(np.float32)
            new_op = top.AddConstOp(*self.get_tensor_type(shape, dtype),
                                    op0,
                                    const_val = op1,
                                    do_relu=False,
                                    loc=self.get_loc(node),
                                    ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_mul_op(self, node):
        op0 = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        shape = self.get_output_shapes(node)
        if isinstance(node.args[1], torch.fx.Node):
            op1 = self.operands[node.args[1]]
            new_op = top.MulOp(*self.get_tensor_type(shape, dtype), [op0, op1],
                                do_relu=False,
                                loc=self.get_loc(node),
                                ip=self.insert_point).output
        else:
            op1 = node.args[1]
            new_op = top.MulConstOp(*self.get_tensor_type(shape, dtype),
                                    op0,
                                    op1,
                                    loc=self.get_loc(node),
                                    ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_reshape_op(self, node):
        in_op = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        new_op = top.ReshapeOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                                in_op,
                                shape=node.args[1],
                                loc=self.get_loc(node),
                                ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_concat_op(self, node):
        operands = list()
        for name in node.args[0]:
            op = self.operands[name]
            operands.append(op)
        axis = node.args[1]
        dtype = self.get_output_dtypes(node)
        new_op = top.ConcatOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                              operands,
                              axis=axis,
                              loc=self.get_loc(node),
                              ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_maxpool2d_with_indices_op(self, node):
        op = self.operands[node.args[0]]
        kernel_shape = node.args[1]
        strides = node.args[2]
        pads = node.args[3]
        dilation = [1,1] #pytorch默认值
        ceil_mode = False
        assert (np.array(dilation) == 1).all()
        pads = pads + pads  # the pad of torch is symmetric
        dtype = self.get_output_dtypes(node)
        outputs = top.MaxPoolWithMaskOp(*self.get_tensor_type(self.get_output_shapes(node), dtype), #[dtype, IntegerType.get_signless(32)]
                                op,
                                kernel_shape=kernel_shape,
                                strides=strides,
                                pads=pads,
                                ceil_mode=ceil_mode,
                                loc=self.get_loc(node),
                                ip=self.insert_point)
        self.operands[node] = [outputs.output, outputs.mask]

    def convert_matmul_op(self, node):
        op0 = self.operands[node.args[0]]
        op1 = self.operands[node.args[1]]
        dtype = self.get_output_dtypes(node)
        new_op = top.MatMulOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                                op0,
                                op1,
                                self.none_op,
                                do_relu=False,
                                loc=self.get_loc(node),
                                ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_embedding_dense_backward_op(self, node):
        grad_output = self.operands[node.args[0]]
        indices = self.operands[node.args[1]]
        dtype = self.get_output_dtypes(node)
        new_op = top.EmbDenseBwdOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                                grad_output,
                                indices,
                                num_weights = node.args[2],
                                padding_idx = node.args[3],
                                scale_grad_by_freq = node.args[4],
                                loc=self.get_loc(node),
                                ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_squeeze_op(self, node):
        op0 = self.operands[node.args[0]]
        axes = node.args[1]
        if isinstance(axes, int):
            axes = [axes]
        dtype = self.get_output_dtypes(node)
        new_op = top.SqueezeOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                                op0,
                                axes=axes,
                                loc=self.get_loc(node),
                                ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_base_conv_op(self, node):
        #(primals_7, primals_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)]
        op = self.operands[node.args[0]]
        strides = node.args[3]
        dilations = node.args[5]
        group = node.args[8]
        kernel_shape = self.get_output_shapes(node.args[1])[0]
        kernel_shape = kernel_shape[2:]
        pads = node.args[4]
        pads = pads + pads
        filter_op = self.operands[node.args[1]]
        if node.args[2] is not None:
            bias_op = self.operands[node.args[2]]
        else:
            bias_op = self.none_op
        dtype = self.get_output_dtypes(node)
        new_op = top.ConvOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                            op,
                            filter_op,
                            bias_op,
                            kernel_shape=kernel_shape,
                            strides=strides,
                            dilations=dilations,
                            pads=pads,
                            group=group,
                            do_relu=False,
                            loc=self.get_loc(node),
                            ip=self.insert_point).output
        self.operands[node] = new_op

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
        group = node.args[-2]
        pads = node.args[5]
        pads = pads + pads
        grad_input = grad_weight = grad_bias =  self.none_op
        shape0 = [] if node.meta['val'][0]==None else list(node.meta['val'][0].size())
        shape1 = [] if node.meta['val'][1]==None else list(node.meta['val'][1].size())
        shape2 = [] if node.meta['val'][2]==None else list(node.meta['val'][2].size())
        dtype = self.get_output_dtypes(node)
        if dtype[0] is None:
            dtype.pop(0)
        bias_op = self.none_op
        if output_mask[1]:
            shape = list(node.args[0].meta['val'].size())
            shape[0],shape[1] = shape[1],shape[0]
            transposed_gradout = top.TransposeOp(*self.get_tensor_type([shape], dtype),
	                                 grad_out,
	                                 0,
	                                 1,
	                                 loc=self.get_loc(node.name+'_transposed_gradout'),
	                                 ip=self.insert_point).output
            if shape[2]>56:
                input_shape = list(node.args[1].meta['val'].size())
                grad_out_shape = list(node.args[0].meta['val'].size())
                transposed_grad_weight = top.ConvBwdWeightOp(*self.get_tensor_type([shape1], dtype),
                                                           input,
                                                           grad_out,
                                                           group,
                                                           input_shape,
                                                           grad_out_shape,
                                                           kernel_shape,
                                                           strides,
                                                           dilations,
                                                           pads,
                                                           output_mask[-1],
                                                           loc=self.get_loc(node.name+'_grad_weight'),
                                                           ip=self.insert_point).output
            else:
                shape = list(node.args[1].meta['val'].size())
                shape[0],shape[1] = shape[1],shape[0]
                transposed_input = top.TransposeOp(*self.get_tensor_type([shape], dtype),
	                                 input,
	                                 0,
	                                 1,
	                                 loc=self.get_loc(node.name+'_transposed_input'),
	                                 ip=self.insert_point).output
	            # kernel_shape_ = list(node.args[1].meta['val'].size())
	            grad_weight_kernel_shape = list(node.args[0].meta['val'].size())
	            grad_weight_kernel_shape = grad_weight_kernel_shape[2:]
	            grad_weight_shape = shape1
	            grad_weight_shape[0],grad_weight_shape[1] = grad_weight_shape[1],grad_weight_shape[0]
	            if pads[0]>0 and strides[0]>1:
	                new_strides = [1,1]
	                new_pads = copy.deepcopy(pads)
	                input_shape = list(node.args[1].meta['val'].size())
	                pad_cal = grad_weight_shape[2]-(pads[0]+input_shape[2]-strides[0]*(grad_weight_kernel_shape[0]-1))
	                new_pads[2],new_pads[3] = pad_cal,pad_cal
	                grad_weight = top.ConvOp(*self.get_tensor_type([grad_weight_shape], dtype),
	                                    transposed_input,
	                                    transposed_gradout,
	                                    bias_op,
	                                    kernel_shape=grad_weight_kernel_shape,
	                                    strides=new_strides,
	                                    dilations=strides,
	                                    pads = new_pads,
	                                    group=group,
	                                    do_relu=False,
	                                    loc=self.get_loc(node.name+'_grad_weight'),
	                                    ip=self.insert_point).output
                else:
                    input_shape = list(node.args[1].meta['val'].size())
                    dilations_grad_weight = strides
                    if input_shape[-1] % 2!=0: #!=
                        strides = [1,1]
                    grad_weight = top.ConvOp(*self.get_tensor_type([grad_weight_shape], dtype),
	                                    transposed_input,
	                                    transposed_gradout,
	                                    bias_op,
	                                    kernel_shape=grad_weight_kernel_shape,
	                                    strides=strides,
                                        #strides = [1,1],
	                                    #dilations=strides,
                                        dilations = dilations_grad_weight,
	                                    pads = pads,
	                                    group=group,
	                                    do_relu=False,
	                                    loc=self.get_loc(node.name+'_grad_weight'),
	                                    ip=self.insert_point).output
	            temp_shape = shape1
	            temp_shape[0],temp_shape[1] = temp_shape[1],temp_shape[0]
	            # shape = list(node.args[1].meta['val'].size())
	            transposed_grad_weight = top.TransposeOp(*self.get_tensor_type([temp_shape], dtype),
	                                grad_weight,
	                                0,
	                                1,
	                                loc=self.get_loc(node.name+'_transposed_grad_weight'),
	                                ip=self.insert_point).output
        if output_mask[0]:
            transposed_weight_shape = shape1
            transposed_weight_shape[0],transposed_weight_shape[1] = transposed_weight_shape[1],transposed_weight_shape[0]
            transposed_weight = top.TransposeOp(*self.get_tensor_type([transposed_weight_shape], dtype),
                                 weight,
                                 0,
                                 1,
                                 loc=self.get_loc(node.name+'_transposed_weight_2'),
                                 ip=self.insert_point).output
            grad_input_kernel_shape = list(node.args[0].meta['val'].size())[-1]
            grad_input_output_shape = list(node.args[1].meta['val'].size())[-1]
            output_padding = grad_input_output_shape-strides[0]*(grad_input_kernel_shape-1)+2*pads[0]-kernel_shape[0]
            output_padding = [output_padding]*2
            grad_input = top.DeconvOp(*self.get_tensor_type([shape0],dtype),
                                  grad_out,
                                  transposed_weight,
                                  bias_op,
                                  kernel_shape = kernel_shape,
                                  strides = strides,
                                  pads = pads,
                                  group = group,
                                  dilations = dilations,
                                  output_padding = output_padding,
                                  do_relu = False,
                                  loc=self.get_loc(node.name+'_grad_input'),
                                  ip=self.insert_point).output
        if output_mask[2]:
            grad_bias = top.ReduceOp(*self.get_tensor_type([bias_sizes], dtype),
                                grad_out,
                                axes = [0,2,3],
                                keepdims = False,
                                mode = StringAttr.get("ReduceSum"),
                                loc=self.get_loc(node.name+"_grad_bias"),
                                ip=self.insert_point).output
        self.operands[node] = [grad_input,transposed_grad_weight,grad_bias]

    def convert_sum_op(self, node): #aten.sum.default                (getitem_6,)
        op0 = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        # assert method in ("ReduceMin", "ReduceMax", "ReduceMean", "ReduceL2", "ReduceL1",
        #                     "ReduceSum", "ReduceProd")
        in_shape = list(node.args[0].meta['val'].size())
        if len(node.args)>=3:
            keepdims = node.args[2]
        elif len(list(node.args[0].meta['val'].size())) ==1:
            keepdims = True
        else:
            keepdims = False
        new_op = top.ReduceOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                                op0,
                                axes = sorted(node.args[1]) if len(node.args) > 1 else tuple(range(len(in_shape))),
                                # keepdims = node.args[2] if len(node.args) > 2 else False,
                                keepdims = keepdims,
                                mode = StringAttr.get("ReduceSum"),
                                loc=self.get_loc(node),
                                ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_mean_op(self, node):
        op0 = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        # assert method in ("ReduceMin", "ReduceMax", "ReduceMean", "ReduceL2", "ReduceL1",
        #                     "ReduceSum", "ReduceProd")
        in_shape = list(node.args[0].meta['val'].size())
        new_op = top.ReduceOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                                op0,
                                axes = sorted(node.args[1]) if len(node.args) > 1 else tuple(range(len(in_shape))),
                                keepdims = node.args[2] if len(node.args) > 2 else False,
                                mode = StringAttr.get("ReduceMean"),
                                loc=self.get_loc(node),
                                ip=self.insert_point).output
        self.operands[node] = new_op


    def convert_getitem_op(self, node):
        self.operands[node] = self.operands[node.args[0]][node.args[1]]

    def convert_to_op(self, node):
        op0 = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        shape = self.get_output_shapes(node)
        new_op = top.CastOp(*self.get_tensor_type(shape, dtype),
                                    op0,
                                    loc=self.get_loc(node),
                                    ip=self.insert_point).output

        new_op2 = top.WeightReorderOp(*self.get_tensor_type(shape, dtype),
                                    new_op,
                                    loc=self.get_loc(f'{node}_weightReorder'),
                                    ip=self.insert_point).output
        self.operands[node] = new_op2

    def create_weight_op(self, name, arg, data_type = 'F32'):
        arg_t = self.convert_scalar_param(arg)
        arg_shape = list(arg_t.shape)
        if name in self.load_weight:
            _op, _shape, _type = self.load_weight[name]
            if _shape != arg_shape or _type != data_type:
                raise RuntimeError("{} weight conflict".format(name))
            return _op
        tensor_type = RankedTensorType.get(arg_shape, self.mlir_type[data_type])
        op = Operation.create("top.Weight",
                              results=[tensor_type],
                              loc=Location.fused([Location.name(name)]))
        self.insert_point.insert(op)
        result = op.results[0]
        self.load_weight[name] = (result, arg_shape, data_type)
        self.weights_data[name] = arg_t
        return result

    def create_constant_weight_op(self,name,shape,val,data_type = 'F32'):
        name = f'{name}_c'
        tensor_type = RankedTensorType.get(list(shape), self.mlir_type[data_type])
        op = Operation.create("top.Weight",
                            results=[tensor_type],
                            loc=Location.fused([Location.name(name)]))
        shape = tuple(shape)
        if name in self.load_weight:
            _op, _shape, _type = self.load_weight[name]
            if _shape != shape or _type != data_type:
                raise RuntimeError("{} weight conflict".format(name))
            return _op
        self.insert_point.insert(op)
        result = op.results[0]
        self.load_weight[name] = (result, shape, data_type)
        self.weights_data[name] = np.full(shape,val,dtype = np.float32)
        return result

    def convert_arange_op(self, node):
        dtype = self.get_output_dtypes(node)
        if node.args[0] in self.operands:
            start = self.operands[node.args[0]]
        else:
            start = self.create_weight_op(f'arangeOp_{node.name}_start', node.args[0])

        if node.args[1] in self.operands:
            end = self.operands[node.args[1]]
        else:
            end = self.create_weight_op(f'arangeOp_{node.name}_end', node.args[1])

        if len(node.args) > 2:
            step = self.create_weight_op(f'arangeOp_{node.name}_step', node.args[2])
        else:
            step = self.none_op
        new_op = top.ArangeOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                              start,
                              end,
                              step,
                              loc=self.get_loc(node.name),
                              ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_slice_op(self,node):
        op0 = self.operands[node.args[0]]
        axis = self.create_weight_op(f'sliceOp_{node.name}_axis', node.args[1])
        start = self.create_weight_op(f'sliceOp_{node.name}_start', node.args[2])
        end = self.create_weight_op(f'sliceOp_{node.name}_end', node.args[3])
        if len(node.args)> 4:
            step = self.create_weight_op(f'sliceOp_{node.name}_step', node.args[4])
        else:
            step = self.create_weight_op(f'sliceOp_{node.name}_step', 1)
        dtype = self.get_output_dtypes(node)
        new_op = top.SliceAxisOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                                 op0,
                                 axis,
                                 start,
                                 step,
                                 end,
                                 loc=self.get_loc(node.name),
                                ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_embedding_op(self,node):
        dtype = self.get_output_dtypes(node)
        if node.args[0] in self.operands:
            weight = self.operands[node.args[0]]
        else:
            weight = self.create_weight_op(f'embeddingOp_{node.name}_input1', node.args[0])

        if node.args[1] in self.operands:
            indices = self.operands[node.args[1]]
        else:
            indices = self.create_weight_op(f'embeddingOp_{node.name}_input2', node.args[1])

        new_op = top.GatherOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                              weight,
                              indices,
                              axis=0,
                              loc=self.get_loc(node.name),
                                ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_compare_op(self,node,mode):
        assert mode in ("Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "NotEqual")
        op0 = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        if node.args[1] in self.operands:
            op1 = self.operands[node.args[1]]
        else:
            op1 = self.create_weight_op(f'compareOp_{node.name}_input1', node.args[1])
        new_op = top.CompareOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                                   op0,
                                   op1,
                                   mode=StringAttr.get(mode),
                                   loc=self.get_loc(node.name),
                                    ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_where_op(self,node):
        op0 = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        x_is_const = y_is_const =  False
        x_const_val = y_const_val = 0
        if node.args[1] in self.operands:
            op1 = self.operands[node.args[1]]
        else:
            x_is_const = True
            op1 = self.none_op
            x_const_val = node.args[1]

        if node.args[2] in self.operands:
            op2 = self.operands[node.args[2]]
        else:
            y_is_const = True
            op2 = self.none_op
            y_const_val = node.args[2]

        new_op = top.WhereOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                             op0,
                             op1,
                             op2,
                             x_is_const=x_is_const,
                             y_is_const=y_is_const,
                             x_const_val=x_const_val,
                             y_const_val=y_const_val,
                             loc=self.get_loc(node.name),
                            ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_copy_op(self,node):
        op0 = self.operands[node.args[0]]
        # dtype = self.get_output_dtypes(node)
        # input_stride = (1,)
        # output_stride = (1,)
        # new_op = top.CopyOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
        #                     op0,
        #                     node.args[0].meta['val'].size(),
        #                     input_stride,
        #                     output_stride,
        #                     loc=self.get_loc(node.name),
        #                     ip=self.insert_point).output
        # self.operands[node] = new_op
        self.operands[node] = op0

    def convert_rsqrt_op(self,node):
        dtype = self.get_output_dtypes(node)
        op0 = self.operands[node.args[0]]
        shape = self.get_output_shapes(node)
        new_op = top.RsqrtOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                            op0,
                            loc=self.get_loc(node.name),
                            ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_sub_op(self,node,is_reverse=False):
        dtype = self.get_output_dtypes(node)
        shape = self.get_output_shapes(node)
        op0 = self.operands[node.args[0]]
        if isinstance(node.args[1], torch.fx.Node):
            op1 = self.operands[node.args[1]]
            new_op = top.SubOp(*self.get_tensor_type(shape, dtype),
                                [op0,op1],
                                is_reverse=is_reverse,
                                loc=self.get_loc(node.name),
                                ip=self.insert_point).output
        else:
            op1 = node.args[1]
            new_op = top.SubConstOp(*self.get_tensor_type(shape, dtype),
                                    op0,
                                    op1,
                                    is_reverse=is_reverse,
                                    loc=self.get_loc(node),
                                    ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_addmm_op(self,node):
        dtype = self.get_output_dtypes(node)
        shape = self.get_output_shapes(node)
        op0 = self.operands[node.args[0]]
        mat1_op = self.operands[node.args[1]]
        mat2_op = self.operands[node.args[2]]
        if len(node.args)==5:
            beta = self.const_val[node.args[3]]
            alpha = self.const_val[node.inputs[4]]
        else:
            beta = 1.0
            alpha = 1.0
        mm_op = top.MatMulOp(*self.get_tensor_type(shape, dtype),
                             mat1_op,
                             mat2_op,
                             self.none_op,
                             do_relu=False,
                             loc=self.get_loc(node.name + "_mm"),
                             ip=self.insert_point).output
        #assert (beta == 1.0 and alpha == 1.0)  # TODO:need to support
        new_op = top.AddOp(*self.get_tensor_type(shape, dtype),
                           [op0,mm_op],
                           coeff=[beta, alpha],
                            loc=self.get_loc(node.name),
                            ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_split_op(self, node):
        dtype = self.get_output_dtypes(node)
        op0 = self.operands[node.args[0]]
        axis = node.args[2]
        split_size = node.args[1]
        if isinstance(split_size, int):
            num = len(node.meta['val'])
            split_size = [split_size] * num
        else:
            num = len(split_size)
        names = [node.name+'_'+str(i) for i in range(num)]
        output = top.SplitOp(self.get_tensor_type(self.get_output_shapes(node), dtype),
                             op0,
                             axis,
                             num,
                             split_size=split_size,
                            loc=self.get_loc(names),
                            ip=self.insert_point)
        self.operands[node] = output.outputs

    def convert_expand_op(self,node):
        dtype = self.get_output_dtypes(node)
        opI = self.operands[node.args[0]]
        new_exp = top.ExpandOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                        opI,
                        shape = node.args[1],
                        loc=self.get_loc(node.name),
                        ip=self.insert_point).output
        self.operands[node] = new_exp

    def convert_broadcast_op(self,node):
        dtype = self.get_output_dtypes(node)
        op = self.operands[node.args[0]]
        # shape = list(node.args[0].meta['val'].size())
        repeat_shape = node.args[1]
        axis = node.args[2]
        unsqueeze_shape = [1]*len(repeat_shape)
        unsqueeze_axis = list(range(len(repeat_shape)))
        for idx in range(len(repeat_shape)):
            if idx in axis:
                unsqueeze_axis.remove(idx)
                unsqueeze_shape[idx] = repeat_shape[idx]
        unsqueeze_op = top.UnsqueezeOp(*self.get_tensor_type([unsqueeze_shape], dtype),
                                     op,
                                     unsqueeze_axis,
                                     loc=self.get_loc(node.name+'_unsqueeze'),
                                    ip=self.insert_point).output
        new_op = top.ExpandOp(*self.get_tensor_type(self.get_output_shapes(node),dtype),
                              unsqueeze_op,
                              shape = repeat_shape,
                              loc = self.get_loc(node.name),
                              ip = self.insert_point).output
        self.operands[node] = new_op

    def convert_amax_op(self, node,index):
        dtype = self.get_output_dtypes(node)
        op0 = self.operands[node.args[0]]
        dim = node.args[1][0]
        keepdims = node.args[2]
        select_last_index = True   # select_last_index = False
        #out_needs = [False, False]
        #for idx, out in enumerate(torch_node.outputs):
            #f len(out) > 0 and self.check_need(out):
                #out_needs[idx] = True
        new_op = top.ArgOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                           *self.get_tensor_type(self.get_output_shapes(node), dtype),
                           input = op0,
                           axis=dim,
                           keepdims=keepdims,
                           mode=StringAttr.get("ArgMax"),
                           select_last_index=select_last_index,
                           loc=self.get_loc(node.name),
                           ip=self.insert_point)

        if index:
            out_ops = [new_op.values, new_op.indices]
        else:
            out_ops = new_op.values
        self.operands[node] = out_ops

    def convert_gather_op(self, node):
        op0 = self.operands[node.args[0]]
        axis = node.args[1]
        op1 = self.operands[node.args[2]]
        dtype = self.get_output_dtypes(node)
        new_op = top.GatherElementsOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                                      op0,
                                      op1,
                                      axis=axis,
                                      loc=self.get_loc(node.name),
                                      ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_neg_op(self, node):
        op0 = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        new_op = top.MulConstOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                                op0,
                                const_val=-1,
                                loc=self.get_loc(node.name),
                                ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_transpose_op(self, node):
        op0 = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        no_dims = len(node.args) == 1
        dim0 = node.args[1] if not no_dims else 0
        dim1 = node.args[2] if not no_dims else 1
        new_op = top.TransposeOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                                 op0,
                                 dim0,
                                 dim1,
                                 loc=self.get_loc(node.name),
                                 ip=self.insert_point).output
        self.operands[node] = new_op

    #_native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean,
    #   Tensor(b!) running_var, bool training, float momentum, float eps)
    def convert_batch_norm_op(self, node):
        dtype = self.get_output_dtypes(node)
        op0 = self.operands[node.args[0]]
        weight = self.operands[node.args[1]]
        bias = self.operands[node.args[2]]
        mean = self.operands[node.args[3]]
        var = self.operands[node.args[4]]
        momentum = node.args[6]
        eps = node.args[7]
        out = top.BatchNormTrainOp(*self.get_tensor_type(self.get_output_shapes(node, 2), dtype),
                                op0,
                                mean = mean,
                                variance = var,
                                gamma = weight,
                                beta = bias,
                                epsilon=eps,
                                momentum=momentum,
                                loc=self.get_loc(node),
                                ip=self.insert_point)
        self.operands[node] = [out.output, out.mean_out, out.variance_out, mean, var]

     #- func: native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var,
     #                                   Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
    def convert_batch_norm_backward_op(self, node): #running_mean和running_var没有使用
        grad_out = self.operands[node.args[0]]
        input = self.operands[node.args[1]]
        weight = self.operands[node.args[2]]
        mean = self.operands[node.args[5]]
        invstd = self.operands[node.args[6]]
        eps = node.args[8]
        output_mask = node.args[-1]
        dtype = self.get_output_dtypes(node)
        gradinput = gradweight = gradbias = self.none_op
        out = top.BatchNormBwdOp(*self.get_tensor_type(self.get_output_shapes(node),dtype),
                                grad_out,
                                input,
                                weight,
                                mean,
                                invstd,
                                epsilon = eps,
                                loc=self.get_loc([node.name+'grad_input',
                                                node.name+'grad_weight',
                                                node.name+'grad_bias']),
                                ip=self.insert_point)
        if output_mask[2]:
            gradbias = out.bias_grad
        if output_mask[1]:
            gradweight = out.weight_grad
        if output_mask[0]:
            gradinput = out.grad_in
        self.operands[node] = [gradinput,gradweight,gradbias]

    def convert_layer_norm_backward_op(self, node):
        grad_out = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        assert node.args[1] in self.operands
        input = self.operands[node.args[1]]
        normalized_shape = node.args[2]
        assert node.args[3] in self.operands
        mean = self.operands[node.args[3]]
        assert node.args[4] in self.operands
        rstd = self.operands[node.args[4]]
        weight_opt = self.operands[node.args[5]] if node.args[5] in self.operands else self.none_op
        bias_opt = self.operands[node.args[6]] if node.args[6] in self.operands else self.none_op
        out = top.LayerNormBwdOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                              grad_out,
                              input,
                              mean,
                              rstd,
                              weight_opt,
                              bias_opt,
                              normalized_shape=normalized_shape,
                              loc=self.get_loc([
                                  node.name+"_grad_input", node.name + "_grad_weight",
                                  node.name + "_grad_bias"
                              ]),
                              ip=self.insert_point)
        self.operands[node] = [out.grad_input, out.grad_weight, out.grad_bias]

    def convert_layer_norm_op(self, node):
        op0 = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
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
        out = top.LayerNormTrainOp(*self.get_tensor_type([shape0, shape1, shape2], dtype),
                              op0,
                              scale_opd,
                              bias_opd,
                              normalized_shape=normalized_shape,
                              axis=axis,
                              eps=eps,
                              loc=self.get_loc([
                                  node.name, node.name + "_Mean",
                                  node.name + "_Rstd"
                              ]),
                              ip=self.insert_point)
        new_op = out.output
        mean = out.mean
        rstd = out.variance
        self.operands[node] = [new_op,mean,rstd]

    def convert_softmax_op(self, node, log):
        op0 = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        in_dim_len = len(list(node.args[0].meta['val'].size()))
        tmp = node.args[1] + in_dim_len if node.args[1] < 0 else node.args[1]
        dim = np.atleast_1d(tmp).astype(np.int32)
        new_op = top.SoftmaxOp(*self.get_tensor_type([shape], dtype),
                               op0,
                               axis=dim,
                               log=log,
                               loc=self.get_loc(node.name),
                               ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_nllloss_op(self,node):
        op0 = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        shape = list(node.meta['val'][0].size())
        shape = [1] if shape == [] else shape
        op1 = self.operands[node.args[1]]
        if node.args[2]!= None:
            weight = self.create_weight_op(f'nlllossOp_{node.name}_input1', node.args[2])
        else:
            weight = self.none_op
        redecution = node.args[3]
        ignore_index = np.atleast_1d(node.args[4]).astype(np.int32)
        new_op = top.NLLlossOp(*self.get_tensor_type([shape], dtype),
                               *self.get_tensor_type([shape], dtype),
                               op0,
                               op1,
                               weight,
                               redecution,
                               ignore_index,
                               loc=self.get_loc(node.name),
                               ip=self.insert_point)
        self.operands[node] = [new_op.output,new_op.total_weight]

    def convert_var_op(self,node):
        op0 = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        reduce_list = node.args[1]
        correction = node.kwargs['correction']
        new_op = top.VarianceOp(
                            *self.get_tensor_type(self.get_output_shapes(node), dtype),
                            op0,
                            reduce_list,
                            correction,
                            loc=self.get_loc(node.name),
                            ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_select_op(self, node): #aten.select.int           (view, 0, 0)

        op0 = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        shape.insert(node.args[1],1)
        axis = self.create_weight_op(f'sliceOp_{node.name}_axis', node.args[1])
        start = self.create_weight_op(f'sliceOp_{node.name}_start', node.args[2])
        step = self.create_weight_op(f'sliceOp_{node.name}_step', 1)
        end = self.create_weight_op(f'sliceOp_{node.name}_end', node.args[2]+1)
        slice_op = top.SliceAxisOp(*self.get_tensor_type([shape], dtype),
                                 op0,
                                 axis,
                                 start,
                                 step,
                                 end,
                                 loc=self.get_loc(node.name+"_slice"),
                                ip=self.insert_point).output
        new_op = top.SqueezeOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                                slice_op,
                                axes=[node.args[1]],
                                loc=self.get_loc(node),
                                ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_math_op(self, node, mode):
        assert mode in ["cos", "cosh", "sin", "sinh", "tan", "tanh", "exp","erf","log"]
        op0 = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        cmd = "top.%sOp(*self.get_tensor_type([shape], dtype), op0, loc=self.get_loc(node.name), ip=self.insert_point).output" % mode.capitalize(
        )
        new_op = eval(cmd)
        self.operands[node] = new_op
    def convert_clone_op(self, node):
        assert len(node.args) == 1
        self.operands[node] = self.operands[node.args[0]]

    def convert_unsqueeze_op(self, node):
        op0 = self.operands[node.args[0]]
        axis = node.args[1]
        dtype = self.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        new_op = top.UnsqueezeOp(*self.get_tensor_type([shape], dtype),
                                    op0,
                                    axes=[axis],
                                    loc=self.get_loc(node),
                                    ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_sigmoid_op(self,node):
        op = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        new_op = top.SigmoidOp(*self.get_tensor_type([shape], dtype),
                                    op,
                                    loc=self.get_loc(node),
                                    ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_silu_op(self,node):
        op = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        new_op = top.SiLUOp(*self.get_tensor_type([shape], dtype),
                                    op,
                                    loc=self.get_loc(node),
                                    ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_pow_op(self,node):
        op = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        if not isinstance(node.args[1], torch.fx.Node):
            power = node.args[1]
            new_op = top.PowOp(*self.get_tensor_type([shape], dtype),
                                        op,
                                        power,
                                        loc=self.get_loc(node),
                                        ip=self.insert_point).output
        else:
            power = self.operands[node.args[1]]
            new_op = top.PowTensorOp(*self.get_tensor_type([shape], dtype),
                                        # op,
                                        # power,
                                        [op,power],
                                        loc=self.get_loc(node),
                                        ip=self.insert_point).output
        self.operands[node] = new_op
    def convert_mse_op(self,node):
        op0 = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        op1 = self.operands[node.args[1]]
        new_op = top.MSELossOp(*self.get_tensor_type([shape], dtype),
                                    op0,
                                    op1,
                                    loc=self.get_loc(node),
                                    ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_maxpool2d_backward_op(self,node):
        # Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices
        grad_out = self.operands[node.args[0]]
        input = self.operands[node.args[1]]
        kernel_size = node.args[2]
        stride = node.args[3]
        padding = node.args[4]
        padding = padding+padding
        dilation = node.args[5]
        ceil_mode = node.args[6]
        indices = self.operands[node.args[-1]]
        dtype = self.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        shape_input = list(node.args[1].meta['val'].size())
        shape_grad = list(node.args[0].meta['val'].size())

        # condition = top.ReluOp(*get_tensor_type([shape_grad], dtype),
        #                     self_,
        #                     loc=self.get_loc(node.name+'_condition'),
        #                     ip=self.insert_point).output
        x_is_const = False
        y_is_const = True
        x_const_val = y_const_val = 0
        new_op = top.WhereOp(*self.get_tensor_type([shape_grad], dtype),
                                indices,
                                grad_out,
                                self.none_op,
                                x_is_const = x_is_const,
                                y_is_const = y_is_const,
                                x_const_val = x_const_val,
                                y_const_val = y_const_val,
                                loc=self.get_loc(node.name+'_before_padding'),
                                ip=self.insert_point).output
        pad_num = (shape_input[-1]-shape_grad[-1])//2
        padding = [pad_num]*4
        mode = 'constant'
        new_op_pad = top.PadOp(
            *self.get_tensor_type([shape], dtype),
            new_op,
            paddings = padding,
            mode = StringAttr.get(mode),
            # val = 0.0,
            loc=self.get_loc(node.name),
            ip=self.insert_point).output

        self.operands[node] = new_op_pad

    def convert_trunc_op(self,node):
        dtype = self.get_output_dtypes(node)
        op0 = self.operands[node.args[0]]
        shape = self.get_output_shapes(node)
        new_op = top.TruncOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                            op0,
                            loc=self.get_loc(node.name),
                            ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_hardswish_op(self,node):
        dtype = self.get_output_dtypes(node)
        op0 = self.operands[node.args[0]]
        new_op = top.HardSwishOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                            op0,
                            loc=self.get_loc(node.name),
                            ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_leaky_relu_op(self,node):
        dtype = self.get_output_dtypes(node)
        op0 = self.operands[node.args[0]]
        alpha = 0.5
        new_op = top.LeakyReluOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                            op0,
                            alpha,
                            loc=self.get_loc(node.name),
                            ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_zero_op(self,node):
        if node.args[0] in self.operands:
            dtype = self.get_output_dtypes(node)
            op0 = self.operands[node.args[0]]
            new_op = top.ConstantFillOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                                        op0,
                                        value=0,
                                        loc=self.get_loc(node),
                                        ip=self.insert_point).output
            self.operands[node] = new_op
        else:
            shape = node.args[0]
            new_op = self.create_constant_weight_op(f'{node.name}_c',shape,0)

            self.operands[node] = new_op

    def convert_clamp_op(self,node):
        input = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        if node.args[1]!=None:
            min_val = node.args[1]
            if node.args[1] in self.operands:
                op1 = self.operands[node.args[1]]
            else:
                op1 = self.create_weight_op(f'{node.name}_minval', node.args[1])
            condition = top.CompareOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                                    input,
                                    op1,
                                    mode=StringAttr.get('GreaterOrEqual'),
                                    loc=self.get_loc(node.name+"min_cond"),
                                        ip=self.insert_point).output
            x_is_const = False
            y_is_const = True
            x_const_val = y_const_val = min_val
            new_op = top.WhereOp(*self.get_tensor_type([shape], dtype),
                                    condition,
                                    input,
                                    self.none_op,
                                    x_is_const = x_is_const,
                                    y_is_const = y_is_const,
                                    x_const_val = x_const_val,
                                    y_const_val = y_const_val,
                                    loc=self.get_loc(node.name),
                                    ip=self.insert_point).output
        if len(node.args)>2:
            max_val = node.args[2]
            if node.args[1] in self.operands:
                op1 = self.operands[node.args[1]]
            else:
                op1 = self.create_weight_op(f'{node.name}_maxval', node.args[2])
            condition = top.CompareOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                                    input,
                                    op1,
                                    mode=StringAttr.get('LessOrEqual'),
                                    loc=self.get_loc(node.name+"max_cond"),
                                        ip=self.insert_point).output
            x_is_const = False
            y_is_const = True
            x_const_val = y_const_val = max_val
            new_op = top.WhereOp(*self.get_tensor_type([shape], dtype),
                                    condition,
                                    input,
                                    self.none_op,
                                    x_is_const = x_is_const,
                                    y_is_const = y_is_const,
                                    x_const_val = x_const_val,
                                    y_const_val = y_const_val,
                                    loc=self.get_loc(node.name),
                                    ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_masked_fill_op(self,node):

        input = self.operands[node.args[0]]
        mask = self.operands[node.args[1]]
        dtype = self.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        const_val = node.args[2]
        new_op = top.MaskedFillOp(*self.get_tensor_type([shape], dtype),
                                  mask,
                                  input,
                                  inversed=True,
                                  const_val=const_val,
                                  loc=self.get_loc(node.name),
                                  ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_index_op(self,node): #(slice_12, [None, None, unsqueeze, _to_copy_1]) 每个tensor对应一个dim gather的输出再作为下一个gather输入
        op0 = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        idx_store = []
        for k,idx in enumerate(node.args[1]):
            if idx is not None:
                indices = [k,idx]
                idx_store.append(indices)
        # shape process
        for i in range(len(idx_store)):
            nd = idx_store[i]
            if len(list(nd[1].meta['val'].size()))!= 1:
                target_shape = list(nd[1].meta['val'].size())
                squeeze_axis = target_shape.index(1)
                target_shape.pop(squeeze_axis)
                tmp_op = self.operands[nd[1]]
                out = top.SqueezeOp(*self.get_tensor_type([target_shape], dtype),
                                    tmp_op,
                                    axes=[squeeze_axis],
                                    loc=self.get_loc(node.name+"_shape_update_"+str(i)),
                                    ip=self.insert_point).output
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
            new_op = top.GatherOp(*self.get_tensor_type([gather_shape], dtype),
                              op0,
                              indices,
                              axis=axis,
                              loc=self.get_loc(node.name+"_"+str(axis)),
                              ip=self.insert_point).output
            op0 = new_op
            idx_store.pop(0)

        if len(node.args[1])< len(origin_shape):
            output_shape = self.get_output_shapes(node)[0]
            if len(origin_shape)>len(output_shape):
                new_op = top.SqueezeOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                                        new_op,
                                        axes=[0],
                                        loc=self.get_loc(node.name+"_squeeze"),
                                        ip=self.insert_point).output
            else:
                new_op = top.UnsqueezeOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                                        new_op,
                                        axes=[0],
                                        loc=self.get_loc(node.name+"_unsqueeze"),
                                        ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_select_scatter_op(self,node): #(new_zeros, mm_73, 1, 0)
        op0 = self.operands[node.args[0]]
        op1 = self.operands[node.args[1]]
        dtype = self.get_output_dtypes(node)
        shape = list(node.meta['val'].size())
        axis = node.args[2]
        update_shape = list(node.args[1].meta['val'].size())
        update_shape.insert(axis,1)
        update = top.UnsqueezeOp(*self.get_tensor_type([update_shape], dtype),
                                    op1,
                                    axes=[axis],
                                    loc=self.get_loc(node.name+"update"),
                                    ip=self.insert_point).output
        index = np.array([[node.args[3]]])
        indices = self.create_weight_op(f'{node.name}_indices', index)
        new_op = top.ScatterNDOp(*self.get_tensor_type([shape], dtype),
                                        op0,
                                        indices,
                                        update,
                                        loc=self.get_loc(node.name),
                                        ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_slice_scatter_op(self,node):
        input = self.operands[node.args[0]]
        update = self.operands[node.args[1]]
        dtype = self.get_output_dtypes(node)
        start = node.args[3]
        end = node.args[4]
        shape = list(node.meta['val'].size())
        update_shape = list(node.args[1].meta['val'].size())
        axis = node.args[2]
        if end > shape[axis]:
            end = shape[axis]
        step = node.args[5] if len(node.args)>=6 else 1
        expand_shape = tuple(-1 if i == axis else 1 for i in range(len(shape)))
        indices = np.arange(start,end,step)
        indices = indices.reshape(expand_shape)
        broadcast_indices = np.ones(tuple(update_shape))*indices
        index = self.create_constant_weight_op(f'{node.name}_index',update_shape,broadcast_indices)
        new_op = top.ScatterElementsOp(
            *self.get_tensor_type(self.get_output_shapes(node), dtype),
            input,
            index,
            update,
            axis,
            loc=self.get_loc(node.name),
            ip=self.insert_point).output
        self.operands[node] = new_op



    def convert_index_put_op(self, node):


        op0 = self.operands[node.args[0]]
        value = self.operands[node.args[2]]
        accumulate = node.args[3]
        dtype = self.get_output_dtypes(node)
        idx_store = []
        for k,idx in enumerate(node.args[1]):
            if idx is not None:
                indices = [k,idx]
                idx_store.append(indices)
        # shape process
        for i in range(len(idx_store)):
            nd = idx_store[i]
            if len(list(nd[1].meta['val'].size()))!= 1:
                target_shape = list(nd[1].meta['val'].size())
                pop_idx = target_shape.index(1)
                target_shape.pop(pop_idx)
                tmp_op = self.operands[nd[1]]
                out = top.SqueezeOp(*self.get_tensor_type([target_shape], dtype),
                                    tmp_op,
                                    axes=[pop_idx],
                                    loc=self.get_loc(node.name+"_shape_update_"+str(i)),
                                    ip=self.insert_point).output
                idx_store[i][1] = out
            else:
                target_shape = list(nd[1].meta['val'].size())
                idx_store[i][1] = self.operands[nd[1]]
            idx_store[i].append(target_shape[0])
        while idx_store:
            info = idx_store[0]
            axis = info[0]
            indices = info[1]
            new_op = top.IndexPutOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                              op0,
                              indices,
                              value,
                              accumulate = accumulate,
                              loc=self.get_loc(node.name+"_"+str(axis)),
                              ip=self.insert_point).output
            op0 = new_op
            idx_store.pop(0)
        self.operands[node] = new_op

    def convert_group_norm_op(self,node):#(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps)
        dtype = self.get_output_dtypes(node)
        input = self.operands[node.args[0]]
        weight = self.operands[node.args[1]]
        bias = self.operands[node.args[2]]
        group = node.args[-2]
        eps = node.args[-1]
        new_op = top.GroupNormTrainOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                              input,
                              weight,
                              bias,
                              group,
                              eps,
                              loc=self.get_loc(node.name),
                              ip=self.insert_point)
        self.operands[node] = [new_op.output,new_op.mean,new_op.rstd]

    def convert_gelu_op(self,node):
        op = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        new_op = top.GELUOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                            op,
                            loc=self.get_loc(node),
                            ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_pad_op(self,node,mode):
        op = self.operands[node.args[0]]
        dtype = self.get_output_dtypes(node)
        padding = node.args[1]
        if len(node.args) >= 3:
            val = node.args[2]
        else:
            val = 0
        new_op = top.PadOp(
            *self.get_tensor_type(self.get_output_shapes(node), dtype),
            op,
            paddings = padding,
            val = val,
            mode = StringAttr.get(mode),
            loc=self.get_loc(node.name),
            ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_argmax_op(self,node):
        dtype = self.get_output_dtypes(node)
        op0 = self.operands[node.args[0]]
        dim = node.args[1]
        if len(node.args)>2:
            keepdims = node.args[2]
        else:
            keepdims = False
        select_last_index = True   # select_last_index = False
        new_op = top.ArgOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                           *self.get_tensor_type(self.get_output_shapes(node), dtype),
                           input = op0,
                           axis=dim,
                           keepdims=keepdims,
                           mode=StringAttr.get("ArgMax"),
                           select_last_index=select_last_index,
                           loc=self.get_loc(node.name),
                           ip=self.insert_point)
        out_ops = new_op.indices
        self.operands[node] = out_ops

    def convert_scatter_op(self,node): #(zeros_like, 1, where, -1.0)
        dtype = self.get_output_dtypes(node)
        input = self.operands[node.args[0]]
        axis = node.args[1]
        index = self.operands[node.args[2]]
        if isinstance(node.args[3], torch.fx.Node):
            update = self.operands[node.args[3]]
        else:
            update = self.create_weight_op(f'{node.name}_update', node.args[3])
        new_op = top.ScatterElementsOp(
            *self.get_tensor_type(self.get_output_shapes(node), dtype),
            input,
            index,
            update,
            axis,
            loc=self.get_loc(node.name),
            ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_logical_and_op(self,node):
        dtype = self.get_output_dtypes(node)
        op0 = self.operands[node.args[0]]
        op1 = self.operands[node.args[1]]
        new_op  = top.LogicalAndOp(*self.get_tensor_type(self.get_output_shapes(node), dtype),
                            [op0,op1],
                            loc=self.get_loc(node.name),
                            ip=self.insert_point).output
        self.operands[node] = new_op

    def convert_bernoulli_op(self,node):
        dtype = self.get_output_dtypes(node)
        if len(node.args)>1:
            p = node.args[1]
        op0 = self.operands[node.args[0]]
        shape = list(node.meta['val'].size())
        ## create random number
        result = np.zeros(shape)
        # np.random.seed(0)
        for i in range(result.size):
            random_num = np.random.random()
            result.flat[i] = 1 if random_num<=p else 0
        op = self.create_constant_weight_op(f'{node.name}_random',shape,result)
        self.operands[node] = op

    def create_return_op(self, Operands):
        return_op = Operation.create("func.return", operands=Operands, results=[])
        self.insert_point.insert(return_op)
        return return_op

    def WeightToNpz(self, weight_file):
        tensor_npz = {}
        for name in self.weights_data:
            tensor_npz[name] = self.weights_data[name]
        np.savez(weight_file, **tensor_npz)
