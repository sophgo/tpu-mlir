#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import torch
from tools.model_transform import *
from utils.mlir_shell import *
from tools.train.tpu_mlir_jit import cosine_similarity
from tools.train.fx2mlir import fx2mlir
import tools.train.tpu_mlir_jit as tpu_mlir_jit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", default="bm1684x", type=str,
                        choices=['bm1684x', 'sg2260'], help="chip platform name")
    parser.add_argument("--case", default="all", type=str, help="test one case, if all, then test all cases")
    parser.add_argument("--mode", default="all", type=str, choices=['all', 'f32', 'f16', 'bf16', 'int8'],
                        help="chip platform name")
    parser.add_argument("--simple", action="store_true", help='do simple test for commit test')
    parser.add_argument("--disable_thread", action="store_true", help='do test without multi thread')
    parser.add_argument("--show_all", action="store_true", help='show all cases')
    parser.add_argument("--debug", default="my_model", help="debug")
    parser.add_argument("--skip_count", default=-1, type=int, help="debug")
    parser.add_argument("--run_op", default="", help="debug")
    parser.add_argument("--submodel", default="submodel",
                        help="submodel")
    parser.add_argument("--cmp", action='store_true',
                        help="enable cmp")
    parser.add_argument("--check_module", action='store_true',
                        help="check the whole module")

    args = parser.parse_args()
    tpu_mlir_jit.args = args

    exec(f'from fx_graph_dumped_{args.submodel}.module import {args.submodel}')
    from torch.fx._symbolic_trace import symbolic_trace
    mod = eval(f'symbolic_trace({args.submodel}())')
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp
    example_inputs = []
    lines = open(f'fx_graph_dumped_{args.submodel}/input_shape', "r").readlines()
    for line in lines:
        items = line.strip().split('*')
        if items[2] == 'torch.int64':
            example_inputs.append(torch.randint(0, 10, eval(items[1])))
            print(f'input {items[0]} dtype is int64:',example_inputs[-1])
        else:
            example_inputs.append(torch.randn(eval(items[1])))

    if args.check_module:
        # ref_outs = mod(*example_inputs)
        # c = fx2mlir(f'mod_{args.submodel}', args)
        # tpu_mlir_mod = c.convert(mod)
        # outs = tpu_mlir_mod(*example_inputs)
        # cosine_similarity(ref_outs, outs)
        bmodel = f'tpu_out_{args.submodel}.mlir.bmodel'
        from tools.train.TpuMlirModule import TpuMlirModule
        #mod.graph.print_tabular()
        mlir_mod = TpuMlirModule(bmodel)
        tmp = mlir_mod(*example_inputs)
        exit(0)

    # example_inputs = [i.contiguous() for i in example_inputs]
    FakeTensorProp(mod).propagate(*example_inputs)
    for i, node in enumerate(mod.graph.nodes):
        if node.op == 'call_module' or node.op == 'call_method' or node.op == 'call_function':
            if args.run_op == '':
                if args.skip_count != -1 and i < args.skip_count:
                    continue
            else:
                if args.run_op != node.name:
                    continue

            print(f'>>> {i}th op, name:', node.name, 'target:',node.target, 'args:', node.args, 'users:', list(node.users.keys()), 'kwargs:', node.kwargs,
                'val:', node.meta['val'] if 'val' in node.meta else 'None')
            if node.name.startswith('getitem'):
                print('skip getitem')
                continue
            print('args shape:', [[list(i.meta['val'].size()), i.meta['val'].dtype] for i in node.args if isinstance(i, torch.fx.Node) and 'val' in i.meta])

            c = fx2mlir(f'node_{node.name}', args)
            mlir_mod, in_ref_data = c.convert_a_op(node)
            for i,data in enumerate(in_ref_data):
                print(f'{i}th in, data:{data.flatten()[:8]}')
            if mlir_mod is not None:
                outs = mlir_mod(*in_ref_data)
                if not isinstance(outs, (list, tuple)):
                    outs = [outs]

            other_args = [i for i in node.args if not isinstance(i, torch.fx.Node)]
            ref_outs = node.target(*in_ref_data, *other_args)
            if not isinstance(ref_outs, (list, tuple)):
                ref_outs = [ref_outs]
            print(f'len(ref_outs):{len(ref_outs)}')
            # m3 = torch.nn.functional.batch_norm(in_ref_data[0],in_ref_data[3],in_ref_data[4], in_ref_data[1],in_ref_data[2],training=True, eps=1e-5, momentum=0.1)
            # ref_outs2 = [m3, in_ref_data[1],in_ref_data[2],in_ref_data[3],in_ref_data[4]]
            cosine_similarity(ref_outs, outs)
            if args.run_op == node.name:
                break
            # exit(0)


