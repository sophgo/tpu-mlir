import copy
import operator
from collections import OrderedDict
from typing import (
    List, Dict, Any, Callable
)

import torch
from torch.fx import (
    GraphModule
)
from torch.quantization import (
    propagate_qconfig_,
    swap_module
)
from torch.nn.intrinsic import (
    _FusedModule
)
from torch.quantization.quantization_mappings import (
    get_default_qat_module_mappings,
    get_default_static_quant_module_mappings
)
from torch.quantization.utils import (
    get_combined_dict
)

from torch.quantization.quantize_fx import (
    _fuse_fx
)

from sophgo_mq.utils import get_flattened_qconfig_dict
from sophgo_mq.utils import getitem2node
from sophgo_mq.utils.logger import logger
from sophgo_mq.utils.registry import register_model_quantizer
import sophgo_mq.nn.intrinsic.qat as qnniqat
import sophgo_mq.nn.qat as qnnqat

class ModelQuantizer(object):
    """General model quantizer class.
    First, replace common float module to nn.qat.modules to make weight fake
    quantized.
    Second, insert activation fake quantize node before specific layers. Layer
    type is defined in function_type_to_quant_input / module_type_to_quant_input.
    We only quantize the inputs of layers and leave the output not quantized
    since it is next layer's input.
    """

    def __init__(self, extra_quantizer_dict, extra_fuse_dict,quant_dict):
        self.additional_function_type = extra_quantizer_dict.get('additional_function_type', [])
        self.additional_module_type = extra_quantizer_dict.get('additional_module_type', ())
        self.additional_fuser_method_mapping = extra_fuse_dict.get('additional_fuser_method_mapping', {})
        self.additional_fusion_pattern = extra_fuse_dict.get('additional_fusion_pattern', {})
        self.additional_qat_module_mapping = extra_fuse_dict.get('additional_qat_module_mapping', {})
        self.additional_node_name = extra_quantizer_dict.get('additional_node_name', [])
        self.exclude_module_name = extra_quantizer_dict.get('exclude_module_name', [])
        self.exclude_function_type = extra_quantizer_dict.get('exclude_function_type', [])
        self.exclude_node_name = extra_quantizer_dict.get('exclude_node_name', [])
        self.module_only_enable_observer = extra_quantizer_dict.get('module_only_enable_observer', [])
        self.quantmode=quant_dict.get('quantmode', 'weight_activation')
        self.strategy=quant_dict.get('strategy', 'CNN')
        self.extra_fuse_dict = extra_fuse_dict

    def prepare(self, model: GraphModule, qconfig):
        model = _fuse_fx(model, self.extra_fuse_dict)
        model = self._weight_quant(model, qconfig)
        if self.quantmode=="weight_activation":
            model = self._insert_fake_quantize_for_act_quant(model, qconfig)
        return model
    
    def prepare_swint(self, model: GraphModule, qconfig):
        model = _fuse_fx(model, self.extra_fuse_dict)
        model = self._weight_quant(model, qconfig)
        if self.quantmode=="weight_activation":
            model = self._insert_fake_quantize_for_act_quant_swint(model, qconfig)
        return model    
    
    def _insert_fake_quantize_for_act_quant_swint(
            self,
            model: GraphModule,
            qconfig: Any):
        graph = model.graph
        nodes = list(model.graph.nodes)
        quantizer_prefix = "_post_act_fake_quantizer"
        node_to_quantize_output = self._find_act_quants_swint(model, qconfig)
        node_to_quantize_output = OrderedDict.fromkeys(node_to_quantize_output).keys()
        qconfig1 = qconfig['']
        qconfig2 = {}

        if 'module_name' in qconfig:
            qconfig2 = qconfig['module_name']    
        for node in node_to_quantize_output:
            quantizer_name = node.name + quantizer_prefix
            if quantizer_name in qconfig2:
                fake_quantizer = qconfig2[str(quantizer_name)].activation()
            else:
                fake_quantizer = qconfig1.activation()
            setattr(model, quantizer_name, fake_quantizer)
            logger.info("Insert act quant {}".format(quantizer_name))
            with graph.inserting_after(node):
                inserted_node = graph.create_node("call_module", quantizer_name, (node,), {})
                for _node in nodes:
                    _node.args = self._fix_succ_recursivly(_node.args, node, inserted_node)
        model.recompile()
        model.graph.lint()
        return model
    
    def _find_act_quants_swint(self, model: GraphModule, qconfig) -> List:
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())
        node_need_to_quantize_output = []
        g2node = getitem2node(model)
        for node in nodes:
            if "matmul" in node.name:
                node_need_to_quantize_output.append(node)
            if ((node.op == "call_module" and node.target in self.exclude_module_name) or
                ((node.op == 'call_function' or node.op == 'call_method') and
                    node.target in self.exclude_function_type) or
                    node.name in self.exclude_node_name) and node.name not in self.additional_node_name:
                logger.info("Exclude skip: {}".format(node.name))
                continue
            if (node.op == "call_module" and isinstance(modules[node.target], self.module_type_to_quant_input)) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                    node.target in self.function_type_to_quant_input) or node.name in self.additional_node_name:
                input_node_list = self._flatten_args(node.all_input_nodes)
                # Means this is not Tensor + Tensor.
                if not all([isinstance(_node, torch.fx.node.Node) for _node in input_node_list]):
                    continue
                for _node in input_node_list:
                    if self._is_implicit_merge(modules, (node, _node)):
                        logger.info("Implicit merge: {} + {}".format(_node.name, node.name))
                        continue
                    if _node.op == "placeholder" and 'tensor_meta' in node.meta:
                        if len(_node.meta['tensor_meta'].shape) == 1:
                            continue
                    if _node in node_need_to_quantize_output:
                        continue
                    if _node in g2node:
                        _node = g2node[_node]
                    node_need_to_quantize_output.append(_node)
        if "module_name" in qconfig:
            additional_nodes = qconfig["module_name"]
            additional_nodes_list = [str(node) for node in additional_nodes.keys()]
            for _node in nodes:
                _quantizer_name = _node.name + '_post_act_fake_quantizer'
                if _quantizer_name in additional_nodes_list:
                    node_need_to_quantize_output.append(_node)
                    continue
        return node_need_to_quantize_output

    def _insert_fake_quantize_for_act_quant(
            self,
            model: GraphModule,
            qconfig: Any):
        graph = model.graph
        nodes = list(model.graph.nodes)

        quantizer_prefix = "_post_act_fake_quantizer"
        node_to_quantize_output = self._find_act_quants(model)
        node_to_quantize_output = OrderedDict.fromkeys(node_to_quantize_output).keys()
        qconfig = qconfig['']
        for node in node_to_quantize_output:
            fake_quantizer = qconfig.activation()
            quantizer_name = node.name + quantizer_prefix
            setattr(model, quantizer_name, fake_quantizer)
            logger.info("Insert act quant {}".format(quantizer_name))
            with graph.inserting_after(node):
                inserted_node = graph.create_node("call_module", quantizer_name, (node,), {})
                for _node in nodes:
                    _node.args = self._fix_succ_recursivly(_node.args, node, inserted_node)

        model.recompile()
        model.graph.lint()
        return model

    def _fix_succ_recursivly(self, args, target_node, inserted_node):
        # List / Tuple
        if isinstance(args, (list, tuple)):
            _tmp = list(args)
            for _i, _arg in enumerate(args):
                if _arg == target_node:
                    _tmp[_i] = inserted_node
                elif isinstance(_arg, tuple):
                    _tmp[_i] = self._fix_succ_recursivly(_arg, target_node, inserted_node)
                elif isinstance(_arg, list):
                    _tmp[_i] = list(self._fix_succ_recursivly(_arg, target_node, inserted_node))
                elif isinstance(_arg, dict):
                    _tmp[_i] = self._fix_succ_recursivly(_arg, target_node, inserted_node)
            return tuple(_tmp)
        # Dict
        elif isinstance(args, dict):
            _tmp = {}
            for k, v in args.items():
                if v == target_node:
                    _tmp[k] = inserted_node
                elif not isinstance(v, torch.fx.node.Node):
                    _tmp[k] = self._fix_succ_recursivly(v, target_node, inserted_node)
                else:
                    _tmp[k] = v
            return _tmp
        else:
            raise NotImplementedError('{} can not be handled now.'.format(type(args)))

    def _weight_quant(self, model: GraphModule, qconfig):
        logger.info("Replace module to qat module.")
        flattened_qconfig_dict = get_flattened_qconfig_dict(qconfig)#torch�ӿ�
        print('flattened_qconfig_dict:', flattened_qconfig_dict)
        propagate_qconfig_(model, flattened_qconfig_dict)
        self._qat_swap_modules(model, self.additional_qat_module_mapping)
        return model

    @property
    def implicit_merge_patterns(self) -> list:
        # Layers which do not need quantize among them.
        # In reversed order!
        return [
            (operator.add, operator.mul)
        ]

    def _on_merge_chain(self, modules, pattern, pair, p_pos=0, v_pos=0):
        if v_pos == len(pair):
            return True
        if p_pos == len(pattern):
            return v_pos == len(pair)
        node = pair[v_pos]
        cur_pattern = pattern[p_pos]
        # Means current node is matched.
        if (node.op == "call_module" and type(modules[node.target]) == cur_pattern) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                    node.target == cur_pattern):
            # Means compairing pair.
            if len(pattern) > p_pos and len(pair) > v_pos:
                return self._on_merge_chain(modules, pattern, pair, p_pos + 1, v_pos + 1)
            # Means compairing extra node.
            matched = False
            flatten_args = self._flatten_args(node.args)
            for _arg in flatten_args:
                extra_pair = (*pair, _arg)
                if isinstance(_arg, torch.fx.node.Node) and \
                        self._on_merge_chain(modules, pattern, extra_pair, p_pos + 1, v_pos + 1):
                    matched = True
            return matched
        # Current node is not matched, skip to next.
        else:
            return self._on_merge_chain(modules, pattern, pair, p_pos + 1, v_pos)

    def _is_implicit_merge(self, modules, pair):
        for pattern in self.implicit_merge_patterns:
            if self._on_merge_chain(modules, pattern, pair):
                return True
        return False

    @property
    def function_type_to_update_data_struct(self) -> list:
        return [
            'update'
        ]

    @property
    def function_type_to_quant_input(self) -> list:
        return [
            operator.add,
            operator.mul,
            torch.nn.functional.adaptive_avg_pool2d,
            torch.nn.functional.interpolate
        ] + self.additional_function_type

    @property
    def module_type_to_quant_input(self) -> tuple:
        return (
            # Conv
            torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d,
            torch.nn.intrinsic.qat.modules.conv_fused.ConvBn2d,
            torch.nn.qat.modules.conv.Conv2d,
            # ConvTranspose
            torch.nn.ConvTranspose2d,
            # Linear
            torch.nn.qat.modules.linear.Linear,
            # Pooling
            torch.nn.modules.pooling.MaxPool2d,
            torch.nn.modules.pooling.AvgPool2d,
            torch.nn.modules.pooling.AdaptiveAvgPool2d,
            # BN
            torch.nn.BatchNorm2d,
            # Prelu mostly do not merge.
            torch.nn.PReLU,
            # Upsample
            torch.nn.Upsample
        ) + self.additional_module_type

    def _flatten_args(self, node):
        flattned_args = []
        if isinstance(node, dict):
            for v in node.values():
                flattned_args.extend(self._flatten_args(v))
        elif isinstance(node, tuple) or isinstance(node, list):
            for n in node:
                flattned_args.extend(self._flatten_args(n))
        else:
            flattned_args.extend([node])
        return flattned_args

    def _find_act_quants(self, model: GraphModule) -> List:
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())
        node_need_to_quantize_output = []
        g2node = getitem2node(model)
        for node in nodes:
            if node.name=="bert_encoder_layer_11_output_layer_norm":
                node_need_to_quantize_output.append(node)
            if ((node.op == "call_module" and node.target in self.exclude_module_name) or
                ((node.op == 'call_function' or node.op == 'call_method') and
                 node.target in self.exclude_function_type) or
                    node.name in self.exclude_node_name) and node.name not in self.additional_node_name:
                logger.info("Exclude skip: {}".format(node.name))
                continue
            if (node.op == 'call_function' and node.target in self._passed_func_type) or \
                (node.op == 'call_module' and isinstance(modules[node.target], self._passed_module_type)):
                continue
            if (node.op == "call_module" and isinstance(modules[node.target], self.module_type_to_quant_input)) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                    node.target in self.function_type_to_quant_input) or node.name in self.additional_node_name:
                input_node_list = self._flatten_args(node.all_input_nodes)
                # Means this is not Tensor + Tensor.
                if not all([isinstance(_node, torch.fx.node.Node) for _node in input_node_list]):
                    continue
                for _node in input_node_list:
                    if _node.target in ["type_as","long"] or _node.name in ['arange','pixel_values'] or _node.target in [operator.mod,operator.floordiv]:#,"expand","to","long"]:
                        continue
                    if _node.op == 'call_function' and 'getitem' in _node.name:
                        if _node.args[0].target=="size":
                            continue   
                    if self._is_implicit_merge(modules, (node, _node)):
                        logger.info("Implicit merge: {} + {}".format(_node.name, node.name))
                        continue
                    if _node.op == "placeholder" and 'tensor_meta' in node.meta:
                        if len(_node.meta['tensor_meta'].shape) == 1:
                            continue
                    if _node in node_need_to_quantize_output:
                        continue
                    if _node in g2node:
                        _node = g2node[_node]
                    node_need_to_quantize_output.append(_node)
        return node_need_to_quantize_output

    def _qat_swap_modules(self, root: GraphModule, additional_qat_module_mapping: Dict[Callable, Callable]):
        all_mappings = get_combined_dict(
            get_default_qat_module_mappings(), additional_qat_module_mapping)
        root = self._convert(root, all_mappings, inplace=True)
        return root

    def _convert(self, module, mapping=None, inplace=False, scope=''):
        if mapping is None:
            mapping = get_default_static_quant_module_mappings()

        if not inplace:
            module = copy.deepcopy(module)
        reassign = {}
        for name, mod in module.named_children():
            # fused modules are swapped as one unit
            new_scope = "{}.{}".format(scope, name) if scope != '' else name
            if new_scope in self.exclude_module_name:
                logger.info("Skip quant layer: " + new_scope)
                continue
            if not isinstance(mod, _FusedModule):
                self._convert(mod, mapping, True, new_scope)
            reassign[name] = swap_module(mod, mapping, {})
        for key, value in reassign.items():
            module._modules[key] = value

        return module