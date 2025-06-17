# -*- coding: utf-8 -*-
import os
import torch
import time
import copy
import numpy as np
from argparse import Namespace
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func
from torch._functorch import compilers
from mlir.ir import *
import mlir.dialects.top as top
from tools.train.partition import partition
from tools.train.TpuMlirModule import TpuMlirModule
from python.tools.train.FxGraphConverter import fx2mlir
from tools.train.fx_pass import fx_pass_for_bmm_expand
from . import config
# TPUC_ROOT = os.environ.get('TPUC_ROOT')
# torch.ops.load_library(f'{TPUC_ROOT}/lib/liblibtorch_plugin.so')

graph_idx = 1000


def _get_disc_decomp():
    from torch._decomp import get_decompositions
    aten = torch.ops.aten
    decompositions_dict = get_decompositions([
        # aten.var_mean,
        # aten._adaptive_avg_pool2d_backward,
        # aten.addcmul,
        # aten.avg_pool2d_backward,
        # aten.binary_cross_entropy_with_logits,
        # aten.gelu,
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
        aten.mse_loss,
        aten.mse_loss_backward,
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
    ])
    return decompositions_dict


def convert_module_fx(submodule_name: str, module: torch.fx.GraphModule,
                      bwd_graph: bool) -> TpuMlirModule:
    c = fx2mlir(submodule_name=submodule_name,
                chip=config.chip,
                bwd_graph=bwd_graph,
                cmp=config.cmp)
    return c.convert(module)


def save_fxgraph_dot(name, module):
    from torch.fx.passes.graph_drawer import FxGraphDrawer
    g = FxGraphDrawer(module, name)
    with open(f'{name}.svg', "wb") as f:
        f.write(g.get_dot_graph().create_svg())


def tpu_mlir_compiler(fx_g, example_inputs):
    if config.only_compile_graph_id == float('inf'):
        print('skip compile for inf')
        return make_boxed_func(fx_g.forward)
    else:
        if config.only_compile_graph_id > 0:
            config.only_compile_graph_id -= 1
            return make_boxed_func(fx_g.forward)
        else:
            if config.only_compile_graph_id == 0:
                config.only_compile_graph_id = float('inf')

    global graph_idx
    time_str = f'{graph_idx}'
    if 'const_name' not in config.debug_cmd:
        time_str = time.strftime("time%Y%m%d%H%M%S", time.localtime())
    graph_idx += 1
    if graph_idx == 1001:
        os.system(f'rm -rf fx_graph_dumped* bmodel_inputs_*')
    os.system(f'mkdir -p {time_str}')

    bwd_graph = len([
        node for node in fx_g.graph.nodes if node.op == 'placeholder' and node.name == 'tangents_1'
    ]) > 0
    fwdStr = 'bwd' if bwd_graph else 'fwd'
    time_str = f'{time_str}_{fwdStr}'
    print('run tpu_mlir_compiler, original graph:', len(fx_g.graph.nodes))
    if 'print_ori_fx_graph' in config.debug_cmd:
        fx_g.graph.print_tabular()
    if 'save_fx_graph_dot' in config.debug_cmd:
        save_fxgraph_dot(f"fx_g_{time_str}", fx_g)
    if 'dump_fx_graph' in config.debug_cmd:
        fx_g.to_folder(f'fx_graph_dumped_{time_str}', time_str)
    if 'skip_tpu_compile' in config.debug_cmd:
        return make_boxed_func(fx_g.forward)

    if 'dump_bmodel_input' in config.debug_cmd:
        print('dump_bmodel_input')
        input_names = []
        for i, node in enumerate(fx_g.graph.nodes):
            if node.op == 'placeholder':
                input_names.append(node.name)

        class tmpModule(torch.nn.Module):

            def __init__(self, fx_g, input_names, graph_name):
                super(tmpModule, self).__init__()
                self.fx_g = fx_g
                self.input_names = input_names
                self.graph_name = graph_name

            def forward(self, *inputs):
                npz_file = f'./bmodel_inputs_{self.graph_name}.npz'
                if not os.path.exists(npz_file):
                    inputs_dict = {}
                    for input_name, input in zip(input_names, inputs):
                        inputs_dict[input_name] = input.numpy()
                    np.savez(npz_file, **inputs_dict)
                    print(f'save to {npz_file}')
                return self.fx_g.forward(*inputs)

        return make_boxed_func(tmpModule(fx_g, input_names, time_str).forward)

    from torch.fx.passes.fake_tensor_prop import FakeTensorProp
    FakeTensorProp(fx_g).propagate(*example_inputs)
    # from torch.fx.passes.fake_tensor_prop import FakeTensorProp
    # from torch._subclasses.fake_tensor import FakeTensorMode
    # # 创建统一的 FakeTensorMode 实例
    # fake_mode = FakeTensorMode()
    # # 将输入转换为同一 FakeTensorMode 下的 Fake Tensor
    # fake_example_inputs = [
    #     fake_mode.from_tensor(inp) if isinstance(inp, torch.Tensor) else inp
    #     for inp in example_inputs
    # ]
    # # 使用转换后的输入进行传播
    # FakeTensorProp(fx_g, mode=fake_mode).propagate(*fake_example_inputs)

    if fx_pass_for_bmm_expand(fx_g):
        print('run tpu_mlir_compiler, updated graph:')
        #fx_g.graph.print_tabular()

    with compilers._disable_jit_autocast():
        compilers.strip_overloads(
            fx_g)  #Remove overloading of node.target, such as aten.sum.dim_intlist to aten.sum
        partitioned_module = convert_module_fx(f'{time_str}_mod', fx_g, bwd_graph)

    return make_boxed_func(partitioned_module.forward)


tpu_dev = "cpu"
# tpu_dev = "cuda:0"
# tpu_dev = "privateuseone"
device = torch.device(tpu_dev)
# import torch_tpu
# device = (
#     "tpu"
#     if torch.tpu.is_available()
#     else "cpu"
# )

from functorch.compile import min_cut_rematerialization_partition

aot_backend = aot_autograd(bw_compiler=tpu_mlir_compiler,
                           fw_compiler=tpu_mlir_compiler,
                           partition_fn=min_cut_rematerialization_partition,
                           decompositions=_get_disc_decomp())
