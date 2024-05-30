import operator

import prettytable

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from sophgo_mq.utils.utils import deepcopy_graphmodule, deepcopy_mixedmodule
from sophgo_mq.utils.logger import logger
from sophgo_mq.utils.hook import DataSaverHook, StopForwardException
from sophgo_mq.utils.state import enable_quantization, disable_all
from sophgo_mq.fake_quantize.quantize_base import QuantizeBase

__all__ = ['profiling']

QUANT_MODULE_TYPE = (nn.Conv2d, nn.Linear, nn.ReLU, nn.ReLU6, nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.ConvTranspose2d)
QUANT_FUNCTION_TYPE = [F.conv2d, F.linear, F.relu, F.relu6, F.adaptive_avg_pool2d, F.avg_pool2d, F.conv_transpose2d, F.interpolate, torch.cat, operator.add, operator.sub]

def _type_of_nn_module(class_type):
    class_type = str(class_type).split('.')[-1][:-2]
    return class_type

def to_device(data, device='cpu'):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
        return data
    elif isinstance(data, list):
        for idx, _ in enumerate(data):
            data[idx] = to_device(data[idx], device)
        return data
    else:
        return data

def node2modules(name2modules, nodes):
    modules = dict()
    for node in nodes:
        if node.target in name2modules:
            modules[node] = name2modules[node.target]
    return modules

def _fix_succ_recursivly(args, target_node, inserted_node):
    # List / Tuple
    if isinstance(args, (list, tuple)):
        _tmp = list(args)
        for _i, _arg in enumerate(args):
            if _arg == target_node:
                _tmp[_i] = inserted_node
            elif isinstance(_arg, tuple):
                _tmp[_i] = _fix_succ_recursivly(_arg, target_node, inserted_node)
            elif isinstance(_arg, list):
                _tmp[_i] = list(self._fix_succ_recursivly(_arg, target_node, inserted_node))
            elif isinstance(_arg, dict):
                _tmp[_i] = _fix_succ_recursivly(_arg, target_node, inserted_node)
        return tuple(_tmp)
    # Dict
    elif isinstance(args, dict):
        _tmp = {}
        for k, v in args.items():
            if v == target_node:
                _tmp[k] = inserted_node
            elif not isinstance(v, torch.fx.node.Node):
                _tmp[k] = _fix_succ_recursivly(v, target_node, inserted_node)
            else:
                _tmp[k] = v
        return _tmp
    else:
        raise NotImplementedError('{} can not be handled now.'.format(type(args)))

def update_model_with_dummy_module(fp_model: torch.fx.GraphModule, quant_model: torch.fx.GraphModule, quant_node: torch.fx.GraphModule, fp_node: torch.fx.GraphModule, quant_nodes: list, fp_nodes: list, quant_node2module: dict, fp_node2module: dict):
    if isinstance(fp_model, torch.fx.GraphModule) is False:
        raise ValueError('Not supported yet!')
    else:
        quant_dummy = torch.nn.Identity()
        fp_dummy = torch.nn.Identity()
        with quant_model.graph.inserting_after(quant_node):
            if isinstance(quant_node.target, str) is False:
                name = quant_node.name
            setattr(quant_model, name + '_dummy', quant_dummy)
            inserted_node = quant_model.graph.create_node(name=quant_node.name + '_dummy', 
                                                          target=name + '_dummy',
                                                          args=(quant_node, ),
                                                          op='call_module',
                                                          kwargs={})
            q_dummy_node = inserted_node
            for _node in quant_nodes:
                _node.args = _fix_succ_recursivly(_node.args, quant_node, inserted_node)
        quant_node2module[inserted_node] = quant_dummy
        quant_nodes = list(quant_model.graph.nodes)
        with fp_model.graph.inserting_after(fp_node):
            setattr(fp_model, name + '_dummy', fp_dummy)
            inserted_node = fp_model.graph.create_node(name=fp_node.name + '_dummy', 
                                                       target=name + '_dummy',
                                                       args=(fp_node, ),
                                                       op='call_module',
                                                       kwargs={})
            f_dummy_node = inserted_node
            for _node in fp_nodes:
                _node.args = _fix_succ_recursivly(_node.args, fp_node, inserted_node)
        fp_node2module[inserted_node] = fp_dummy
        fp_nodes = list(fp_model.graph.nodes)
        fp_model.recompile()
        quant_model.recompile()
        fp_model.graph.lint()
        quant_model.graph.lint()
    return q_dummy_node, f_dummy_node, fp_model, quant_model, quant_node2module, fp_node2module  # , fp_nodes, quant_nodes

def cosine(x, y):
    x, y = x.flatten(), y.flatten()
    return (x * y).sum().abs() / (x.norm(2)) / y.norm(2)

def profile_summary(result):
    cos = [res['cos'] for res in result]
    sorted_result = sorted(result, key=lambda x: x['cos'])
    avg_cos = sum(cos) / len(cos)
    min_cos, min_name, min_nodes, min_op = sorted_result[0]['cos'], sorted_result[0]['name'], sorted_result[0]['nodes'], sorted_result[0]['op']
    return f'avg cos: {avg_cos}\nworst layer: {min_name}({min_op}) with nodes {min_nodes} cos {min_cos}'


def profiling(model: torch.fx.GraphModule, cali_data, profiling_type='standalone', module_list=None):
    r'''
    args:
        model: the model to profile
        cali_data: batches used to profile
        profiling_type: 
            'standalone' means to quantize each module by topology order, make de-quantize it after evaluation
            'interaction' means to quantize each module and then keep
    '''
    fp_model = model
    disable_all(fp_model)
    if module_list is None:
        quant_model = deepcopy_graphmodule(model)
        nodes = list(quant_model.graph.nodes)
        f_nodes = list(fp_model.graph.nodes)
        name2node = {node[0].target if isinstance(node[0].target, str) else node[0].name : node for node in zip(quant_model.graph.nodes, fp_model.graph.nodes)}
        fp_node2module = node2modules(dict(fp_model.named_modules()), fp_model.graph.nodes)
        quant_node2module = node2modules(dict(quant_model.named_modules()), quant_model.graph.nodes)
    else:
        quant_model = deepcopy_mixedmodule(model, module_list)
        fp_node2module = {}
        quant_node2module = {}
        name2node = {}
        nodes = []
        f_nodes = []
        for mname in module_list:
            fp_child = getattr(fp_model, mname)
            quant_child = getattr(quant_model, mname)
            nodes += list(quant_child.graph.nodes)
            f_nodes += list(fp_child.graph.nodes)
            name2node.update(
                {f'{mname}.{node.target}': node for node in zip(quant_child.graph.nodes, fp_child.graph.nodes)}
            )
            quant_node2module.update(
                node2modules(dict(quant_child.named_modules()), quant_child.graph.nodes)
            )
            fp_node2module.update(
                node2modules(dict(fp_child.named_modules()), fp_child.graph.nodes)
            )

    name2cosine = {}
    name2profiling_node = {}
    quant_node2fp_node = {}
    profile_module_names = {}
    profile_function_names = {}
    for q_node, f_node in zip(nodes, f_nodes):
        quant_node2fp_node[q_node] = f_node
    for name in name2node:
        quant_node, fp_node = name2node[name]
        if quant_node.op == 'call_module' and isinstance(quant_node2module[quant_node], QUANT_MODULE_TYPE):
            profile_module_names[name] = type(quant_node2module[quant_node])
            if len(quant_node.users) == 1:
                q_user = list(quant_node.users)[0]
                f_user = list(fp_node.users)[0]
                if q_user.op == 'call_module' and isinstance(quant_node2module[q_user], QuantizeBase):
                    user = [quant_node, q_user]
                else:
                    user = [quant_node]
            else:
                user = [quant_node]
            name2profiling_node[name] = user
        elif (quant_node.op == 'call_function' or quant_node.op == 'call_method') and quant_node.target in QUANT_FUNCTION_TYPE:
            profile_function_names[name] = quant_node.target.__name__
            if len(quant_node.users) == 1:
                q_user = list(quant_node.users)[0]
                f_user = list(fp_node.users)[0]
                if q_user.op == 'call_module' and isinstance(quant_node2module[q_user], QuantizeBase):
                    user = [quant_node, q_user]
                else:
                    user = None
            else:
                user = None
            if user is None:
                q_dummy_node, f_dummy_node, fp_model, quant_model, quant_node2module, fp_node2module = \
                    update_model_with_dummy_module(fp_model, quant_model, quant_node, fp_node, 
                                                   nodes, [quant_node2fp_node[_node] for _node in nodes],
                                                   quant_node2module, fp_node2module)
                user = [q_dummy_node]
                quant_node2fp_node[q_dummy_node] = f_dummy_node
            name2profiling_node[name] = user
    for name in name2profiling_node:
        quant_module = quant_node2module[name2profiling_node[name][-1]]
        fp_module = fp_node2module[quant_node2fp_node[name2profiling_node[name][-1]]]
        quant_saver = DataSaverHook(store_input=False, store_output=True, stop_forward=False)
        fp_saver = DataSaverHook(store_input=False, store_output=True, stop_forward=False)
        quant_hook = quant_module.register_forward_hook(quant_saver)
        fp_hook = fp_module.register_forward_hook(fp_saver)
        name2cosine[name] = []
        logger.setLevel('CRITICAL')
        for node in name2profiling_node[name]:
            module = quant_node2module[node]
            enable_quantization(module)
        device = next(quant_model.parameters()).device
        with torch.no_grad():
            for batch in cali_data:
                try:
                    _ = fp_model(to_device(batch, device))
                except StopForwardException:
                    pass
                try:
                    _ = quant_model(to_device(batch, device))
                except StopForwardException:
                    pass
                fp_out = fp_saver.output_store
                quant_out = quant_saver.output_store
                name2cosine[name].append(cosine(fp_out, quant_out))
        name2cosine[name] = sum(name2cosine[name]) / len(name2cosine[name])
        quant_hook.remove()
        fp_hook.remove()
        if profiling_type == 'standalone':
            for node in name2profiling_node[name]:
                module = quant_node2module[node]
                disable_all(module)
        logger.setLevel('INFO')

    table = prettytable.PrettyTable(['op', 'name', 'nodes', 'cosine'])
    profile_result = []
    for name in name2cosine:
        op = _type_of_nn_module(profile_module_names[name]) if name in profile_module_names else profile_function_names[name]
        cos = float(name2cosine[name])
        nodes = name2profiling_node[name]
        table.add_row([op, name, nodes, cos])
        profile_result.append({'op': op, 'name': name, 'nodes': nodes, 'cos': cos})
    logger.critical(f'Profile Type {profiling_type}\n{table}')
    logger.critical(profile_summary(profile_result))