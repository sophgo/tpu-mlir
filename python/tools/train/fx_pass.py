import os
import torch
import pdb
import gc
import time
import copy
import numpy as np

def fx_pass_for_bmm_expand(gm: torch.fx.GraphModule):
    update = False
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.expand.default:
            if node.meta['val'].size() == node.args[0].meta['val'].size():
                print(f'find expand same shape')
                node.replace_all_uses_with(node.args[0])
                gm.graph.erase_node(node)
                update = True
    gm.graph.lint()
    gm.recompile()


    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.clone.default:
            node.replace_all_uses_with(node.args[0])
            print(f'del clone node:{node.name}')
            gm.graph.erase_node(node)
            update = True
    gm.graph.lint()
    gm.recompile()

    return update

