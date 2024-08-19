# -*- coding: utf-8 -*-
import os
import torch
import time
import copy
from argparse import Namespace
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func
from torch._functorch import compilers
from mlir.ir import *
import mlir.dialects.top as top
# import mlir.dialects.train as train
# from torch.cuda.amp import autocast, GradScaler
#from apex import amp
from tools.train.partition import partition
from tools.train.TpuMlirModule import TpuMlirModule
# from tools.train.fx2mlir import fx2mlir
from tools.train.FxGraphConvertor import fx2mlir
from tools.train.fx_pass import fx_pass_for_bmm_expand

# TPUC_ROOT = os.environ.get('TPUC_ROOT')
# torch.ops.load_library(f'{TPUC_ROOT}/lib/liblibtorch_plugin.so')

args = None
graph_idx = 0

COSINE_THRESHOLD = 0.99
def cosine_similarity(gt_tensors, pred_tensors):
    if not isinstance(gt_tensors, (tuple, list)):
        gt_tensors = (gt_tensors,)
        pred_tensors = (pred_tensors,)
    for gt_tensor, pred_tensor in zip(gt_tensors, pred_tensors):
        gt_tensor = gt_tensor.flatten().to(torch.float32)
        pred_tensor = pred_tensor.flatten().to(torch.float32)
        print(f'ref_outs:{gt_tensor[:8]}')
        print(f'outs:{pred_tensor[:8]}')
        if torch.sum(gt_tensor) == 0.0 or torch.sum(pred_tensor) == 0.0:
            if torch.allclose(gt_tensor, pred_tensor, atol=1e-4, rtol=1e-4, equal_nan=True):
                return 1.0
        res = torch.nn.functional.cosine_similarity(gt_tensor, pred_tensor, dim=0, eps=1e-6)
        res = res.cpu().detach().item()
        print(f'>>>cos:{res}')
        if res < 0.8:
            print('cmp fail')
    return res

def _get_disc_decomp():
    from torch._decomp import get_decompositions
    aten = torch.ops.aten
    decompositions_dict = get_decompositions(
        [
            # aten.var_mean,
            # aten._adaptive_avg_pool2d_backward,
            # aten.addcmul,
            # aten.avg_pool2d_backward,
            # aten.binary_cross_entropy_with_logits,
            aten.gelu,
            aten.gelu_backward,
            # aten.glu_backward,
            # aten.grid_sampler_2d,
            # aten.hardsigmoid,
            # aten.hardsigmoid_backward,
            # aten.hardswish,
            # aten.hardswish_backward,
            # aten.hardtanh,
            # aten.hardtanh_backward,
            # aten.logsumexp.default,
            # aten.max_pool2d_with_indices_backward,
            # aten.mse_loss,
            # aten.mse_loss_backward,
            # aten.mv,
            # aten.narrow,
            # aten.native_batch_norm,
            # aten.native_batch_norm_backward,
            # aten.native_dropout_backward,
            # aten.native_group_norm,
            aten.native_group_norm_backward,
            # aten.native_layer_norm,
            aten.native_layer_norm_backward,
            # aten.std_mean.correction,
            # aten._softmax,
            aten._softmax_backward_data,
            # aten.stack,
            # aten.t,
            aten.tanh_backward,
            # aten.threshold_backward,
            # aten.transpose.int,
            # aten.tril.default,
            # aten.upsample_bilinear2d.vec,
            # aten.upsample_nearest2d_backward,
            # aten._unsafe_view,
            # aten._native_batch_norm_legit_functional,
            # aten._log_softmax,
            # aten.nll_loss_forward,
            # aten.addmm,
            # aten.leaky_relu,
            # aten.leaky_relu_backward,
            aten.slice_backward,
            # aten.convolution_backward,
            aten.select_backward,
            aten.embedding_dense_backward,
            # aten.select_scatter,
            # aten.slice_scatter,
            aten.sigmoid_backward,
            aten.nll_loss_backward,
            aten._log_softmax_backward_data,
            aten.nll_loss_forward,
            aten.mse_loss,
            aten.mse_loss_backward,
        ]
    )
    return decompositions_dict

def convert_module_fx(
    submodule_name: str,
    module: torch.fx.GraphModule,
    args:Namespace,
    bwd_graph:bool
) -> TpuMlirModule:
    c = fx2mlir(submodule_name, args, bwd_graph)
    return c.convert(module)

def save_fxgraph_dot(name, module):
    if 'enable_dot_graph' in args.debug:
        from torch.fx.passes.graph_drawer import FxGraphDrawer
        g = FxGraphDrawer(module, name)
        with open(f'{name}.svg', "wb") as f:
            f.write(g.get_dot_graph().create_svg())

def tpu_mlir_compiler(fx_g, example_inputs):
    if 'const_name' not in args.debug:
        time_str = time.strftime("time%Y%m%d%H%M%S", time.localtime())
    else:
        global graph_idx
        time_str = f'{graph_idx}'
        graph_idx += 1
    os.system(f'rm -rf fx_graph_dumped*;mkdir -p {time_str}')
    print('run tpu_mlir_compiler, original graph:')
    #fx_g.graph.print_tabular()
    save_fxgraph_dot(f"fx_g_{time_str}", fx_g)

    # for i, node in enumerate(fx_g.graph.nodes):
    #     print(f'>>> {i}th op, name:', node.name, 'target:',node.target, 'args:', node.args, 'users:', list(node.users.keys()), 'kwargs:', node.kwargs,
    #           'val:', node.meta['val'] if 'val' in node.meta else 'None')
    if args.skip_module_num > 0:
        args.skip_module_num -= 1
        print('skip_module_num:', args.skip_module_num)
        return make_boxed_func(fx_g.forward)

    fx_g_bk = copy.deepcopy(fx_g)

    from torch.fx.passes.fake_tensor_prop import FakeTensorProp
    # FakeTensorProp(fx_g).propagate(*example_inputs)

    if fx_pass_for_bmm_expand(fx_g):
        print('run tpu_mlir_compiler, updated graph:')
        #fx_g.graph.print_tabular()

    with compilers._disable_jit_autocast():
        compilers.strip_overloads(fx_g) #删除掉node.target的重载,比如将aten.sum.dim_IntList变为aten.sum
        # for node in fx_g.graph.nodes:
        #     if (
        #         node.target == torch.ops.aten._to_copy
        #         and len(node.args) == 1
        #         and len(node.kwargs) == 1
        #         and "dtype" in node.kwargs
        #     ):
        #         node.target = torch.ops.aten.to
            # if node.target == torch.ops.prims.div:
            #     node.target = torch.ops.aten.div
            # if node.target == torch.ops.aten.alias:
            #     node.target = torch.ops.aten.clone
            # if node.target == torch.ops.prims.var:
            #     node.target = torch.ops.aten.var
            # if node.target == torch.ops.prims.sum:
            #     print('change prims.sum')
            #     node.target = torch.ops.aten.sum
            # if node.target == torch.ops.prims.convert_element_type:
            #     node.target = torch.ops.aten.to
            # if node.target == torch.ops.aten.view:
            #     node.target = torch.ops.aten.reshape

        # for node in fx_g.graph.nodes:
        #     new_kwargs = {}
        #     for k, v in node.kwargs.items():
        #         if isinstance(v, torch.device):
        #             v = v.type # device(type='cuda', index=0)
        #         new_kwargs[k] = v #将device改为字符串形式, why?
        #     node.kwargs = new_kwargs
        # fx_g.graph.lint()
        # fx_g.recompile()

        bwd_graph = len([node for node in fx_g.graph.nodes if node.op == 'placeholder' and node.name == 'tangents_1']) > 0
        # partitioned_module = partition(fx_g, min_block_size = 3)
        # save_fxgraph_dot(f"partitioned_module_{time_str}", partitioned_module)

        # if len(list(partitioned_module.named_children())) > 0:
        #     for name, _ in partitioned_module.named_children():
        #         submodule = getattr(partitioned_module, name)
        #         print(name, 'submodule:', submodule)

        #         tpu_mlir_mod = convert_module_fx(f'{time_str}_{name}', submodule, args, bwd_graph)
        #         if tpu_mlir_mod is not None:
        #             setattr(partitioned_module, name, tpu_mlir_mod)
        # else:
        #     partitioned_module = convert_module_fx(f'{time_str}_main_mod', partitioned_module, args, bwd_graph)
        partitioned_module = convert_module_fx(f'{time_str}_main_mod', fx_g, args, bwd_graph)

    return make_boxed_func(partitioned_module.forward)


def skip_compiler(gm, example_inputs):
    print('run compiler, graph:')

    #gm.graph.print_tabular()

    # FakeTensorProp(gm).propagate(*example_inputs)
    # fwd_compiler(gm, example_inputs)
    return make_boxed_func(gm.forward)

def test_compiler(gm, example_inputs):
    print('run compiler, graph:')
    for i, node in enumerate(gm.graph.nodes):
        print(f'>>> {i}th op, name:', node.name, 'target:',node.target, 'args:', node.args, 'users:', list(node.users.keys()), 'kwargs:', node.kwargs,
              'val:', node.meta['val'] if 'val' in node.meta else 'None')
    #gm.graph.print_tabular()

    # FakeTensorProp(gm).propagate(*example_inputs)
    # fwd_compiler(gm, example_inputs)
    return make_boxed_func(gm.forward)

### change start
tpu_dev = "cpu"
# tpu_dev = "cuda:0"
# tpu_dev = "privateuseone:0"
device = torch.device(tpu_dev)
# import torch_tpu
# device = (
#     "tpu"
#     if torch.tpu.is_available()
#     else "cpu"
# )

#from functorch.compile import min_cut_rematerialization_partition
aot_backend = aot_autograd(bw_compiler = tpu_mlir_compiler,fw_compiler=tpu_mlir_compiler,decompositions=_get_disc_decomp())#fw_compiler=skip_compiler,
