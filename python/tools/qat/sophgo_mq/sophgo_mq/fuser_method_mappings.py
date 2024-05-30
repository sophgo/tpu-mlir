from typing import Optional, Type

import torch
import torch.nn as nn
from torch.fx.graph import Node

import sophgo_mq.nn as qnn
import sophgo_mq.nn.intrinsic as qnni
import sophgo_mq.nn.intrinsic.qat as qnniqat
from sophgo_mq.utils.fusion import fuse_deconv_bn_eval
from sophgo_mq.nn.modules import FrozenBatchNorm2d

from collections import OrderedDict
from torch.ao.quantization.fuser_method_mappings import get_fuser_method
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict
QuantizerCls = Any

# pattern for conv bn fusion
DEFAULT_FUSION_PATTERNS = OrderedDict()
def register_fusion_pattern(pattern):
    def insert(fn):
        DEFAULT_FUSION_PATTERNS[pattern] = fn
        return fn
    return insert

# turn foo.bar -> ['foo', 'bar']
def _parent_name(target):
    r = target.rsplit('.', 1)
    if len(r) == 1:
        return '', r[0]
    else:
        return r[0], r[1]

# ---------------------
# Fusion Pattern Registrations
# ---------------------

# Base Pattern Handler
class FuseHandler(ABC):
    """ Base handler class for the fusion patterns
    """
    def __init__(self, quantizer: QuantizerCls, node: Node):
        pass

    @abstractmethod
    def fuse(self, quantizer: QuantizerCls, load_arg: Callable,
             fuse_custom_config_dict: Dict[str, Any] = None) -> Node:
        pass

@register_fusion_pattern((torch.nn.ReLU, torch.nn.Conv1d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.Conv3d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Conv1d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Conv3d))
@register_fusion_pattern((torch.nn.BatchNorm1d, torch.nn.Conv1d))
@register_fusion_pattern((torch.nn.BatchNorm2d, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.BatchNorm3d, torch.nn.Conv3d))
@register_fusion_pattern((torch.nn.ReLU, (torch.nn.BatchNorm1d, torch.nn.Conv1d)))
@register_fusion_pattern((torch.nn.ReLU, (torch.nn.BatchNorm2d, torch.nn.Conv2d)))
@register_fusion_pattern((torch.nn.ReLU, (torch.nn.BatchNorm3d, torch.nn.Conv3d)))
@register_fusion_pattern((torch.nn.functional.relu, (torch.nn.BatchNorm1d, torch.nn.Conv1d)))
@register_fusion_pattern((torch.nn.functional.relu, (torch.nn.BatchNorm2d, torch.nn.Conv2d)))
@register_fusion_pattern((torch.nn.functional.relu, (torch.nn.BatchNorm3d, torch.nn.Conv3d)))
class ConvBNReLUFusion(FuseHandler):
    def __init__(self, quantizer: QuantizerCls, node: Node):
        super().__init__(quantizer, node)
        self.relu_node = None
        self.bn_node = None
        if (node.op == 'call_function' and node.target is torch.nn.functional.relu) or \
           (node.op == 'call_module' and type(quantizer.modules[node.target]) == torch.nn.ReLU):
            self.relu_node = node
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == 'call_module'
        if type(quantizer.modules[node.target]) in [torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d]:
            self.bn_node = node
            self.bn = quantizer.modules[self.bn_node.target]
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == 'call_module'
        self.conv_node = node
        self.conv = quantizer.modules[self.conv_node.target]

    def fuse(self, quantizer: QuantizerCls, load_arg: Callable,
             fuse_custom_config_dict: Dict[str, Any] = None) -> Node:
        if fuse_custom_config_dict is None:
            fuse_custom_config_dict = {}
        additional_fuser_method_mapping = fuse_custom_config_dict.get("additional_fuser_method_mapping", {})
        op_list = []
        if self.relu_node is not None:
            # since relu can be used multiple times, we'll need to create a relu module for each match
            if self.relu_node.op == 'call_module':
                relu = torch.nn.ReLU(quantizer.modules[self.relu_node.target].inplace)
            else:
                # TODO: get inplace argument from functional
                relu = torch.nn.ReLU()
            op_list.append(relu)
            relu.training = self.conv.training
            if self.bn_node is not None:
                op_list.append(self.bn)
            op_list.append(self.conv)
        else:
            assert self.bn_node is not None
            op_list.append(self.bn)
            op_list.append(self.conv)

        # the modules are added in order of relu - bn - conv
        # so we need to correct it
        op_list.reverse()
        op_type_list = tuple(type(m) for m in op_list)
        conv_parent_name, conv_name = _parent_name(self.conv_node.target)
        fuser_method = get_fuser_method(op_type_list, additional_fuser_method_mapping)
        if fuser_method is None:
            raise NotImplementedError("Cannot fuse modules: {}".format(op_type_list))
        fused = fuser_method(*op_list)
        setattr(quantizer.modules[conv_parent_name], conv_name, fused)

        # TODO: do we need to make sure bn is only used once?
        if self.bn_node is not None:
            parent_name, name = _parent_name(self.bn_node.target)
            setattr(quantizer.modules[parent_name], name, torch.nn.Identity())
        # relu may be used multiple times, so we don't set relu to identity
        return quantizer.fused_graph.node_copy(self.conv_node, load_arg)

@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Linear))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.Linear))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.BatchNorm2d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.BatchNorm2d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.BatchNorm3d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.BatchNorm3d))
class ModuleReLUFusion(FuseHandler):
    def __init__(self, quantizer: QuantizerCls, node: Node):
        super().__init__(quantizer, node)
        self.relu_node = node
        assert isinstance(node.args[0], Node)
        node = node.args[0]
        assert node.op == 'call_module'
        self.module_node = node
        self.module = quantizer.modules[self.module_node.target]

    def fuse(self, quantizer: QuantizerCls, load_arg: Callable,
             fuse_custom_config_dict: Dict[str, Any] = None) -> Node:
        if fuse_custom_config_dict is None:
            fuse_custom_config_dict = {}
        additional_fuser_method_mapping = fuse_custom_config_dict.get("additional_fuser_method_mapping", {})
        op_list = []
        # since relu can be used multiple times, we'll need to create a relu module for each match
        if self.relu_node.op == 'call_module':
            relu = torch.nn.ReLU(quantizer.modules[self.relu_node.target].inplace)
        else:
            # TODO: get inplace argument from functional
            relu = torch.nn.ReLU()
        relu.training = self.module.training
        op_list.append(relu)
        op_list.append(self.module)

        op_list.reverse()
        op_type_list = tuple(type(m) for m in op_list)
        module_parent_name, module_name = _parent_name(self.module_node.target)
        fuser_method = get_fuser_method(op_type_list, additional_fuser_method_mapping)
        setattr(quantizer.modules[module_parent_name], module_name, fuser_method(*op_list))
        return quantizer.fused_graph.node_copy(self.module_node, load_arg)

import torch.nn.intrinsic as nni
from typing import Union, Callable, Tuple, Dict, Optional, Type
from torch.ao.quantization.utils import get_combined_dict

def fuse_conv_bn(conv, bn):
    r"""Given the conv and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    assert(conv.training == bn.training),\
        "Conv and BN both must be in the same mode (train or eval)."

    fused_module_class_map = {
        nn.Conv1d: nni.ConvBn1d,
        nn.Conv2d: nni.ConvBn2d,
        nn.Conv3d: nni.ConvBn3d,
    }

    if conv.training:
        assert bn.num_features == conv.out_channels, 'Output channel of Conv2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        fused_module_class = fused_module_class_map.get((type(conv)), None)
        if fused_module_class is not None:
            return fused_module_class(conv, bn)
        else:
            raise NotImplementedError("Cannot fuse train modules: {}".format((conv, bn)))
    else:
        return nn.utils.fuse_conv_bn_eval(conv, bn)

def fuse_conv_bn_relu(conv, bn, relu):
    r"""Given the conv and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> r1 = nn.ReLU(inplace=False)
        >>> m2 = fuse_conv_bn_relu(m1, b1, r1)
    """
    assert(conv.training == bn.training == relu.training),\
        "Conv and BN both must be in the same mode (train or eval)."
    fused_module : Optional[Type[nn.Sequential]] = None
    if conv.training:
        map_to_fused_module_train = {
            nn.Conv1d: nni.ConvBnReLU1d,
            nn.Conv2d: nni.ConvBnReLU2d,
            nn.Conv3d: nni.ConvBnReLU3d,
        }
        assert bn.num_features == conv.out_channels, 'Output channel of Conv must match num_features of BatchNorm'
        assert bn.affine, 'Only support fusing BatchNorm with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm with tracking_running_stats set to True'
        fused_module = map_to_fused_module_train.get(type(conv), None)
        if fused_module is not None:
            return fused_module(conv, bn, relu)
        else:
            raise NotImplementedError("Cannot fuse train modules: {}".format((conv, bn, relu)))
    else:
        map_to_fused_module_eval = {
            nn.Conv1d: nni.ConvReLU1d,
            nn.Conv2d: nni.ConvReLU2d,
            nn.Conv3d: nni.ConvReLU3d,
        }
        fused_module = map_to_fused_module_eval.get(type(conv), None)
        if fused_module is not None:
            fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
            return fused_module(fused_conv, relu)
        else:
            raise NotImplementedError("Cannot fuse eval modules: {}".format((conv, bn, relu)))

def fuse_linear_bn(linear, bn):
    r"""Given the linear and bn modules, fuses them and returns the fused module

    Args:
        linear: Module instance of type Linear
        bn: BatchNorm1d instance that needs to be fused with the linear layer

    Examples::

        >>> m1 = nn.Linear(20, 10)
        >>> b1 = nn.BatchNorm1d(10)
        >>> m2 = fuse_linear_bn(m1, b1)
    """
    assert(linear.training == bn.training),\
        "Linear and BN both must be in the same mode (train or eval)."

    if linear.training:
        raise Exception("Fusing Linear+BatchNorm not yet supported in training.")
    else:
        return nn.utils.fusion.fuse_linear_bn_eval(linear, bn)

DEFAULT_OP_LIST_TO_FUSER_METHOD : Dict[Tuple, Union[nn.Sequential, Callable]] = {
    (nn.Conv1d, nn.BatchNorm1d): fuse_conv_bn,
    (nn.Conv1d, nn.BatchNorm1d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv2d, nn.BatchNorm2d): fuse_conv_bn,
    (nn.Conv2d, nn.BatchNorm2d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv3d, nn.BatchNorm3d): fuse_conv_bn,
    (nn.Conv3d, nn.BatchNorm3d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv1d, nn.ReLU): nni.ConvReLU1d,
    (nn.Conv2d, nn.ReLU): nni.ConvReLU2d,
    (nn.Conv3d, nn.ReLU): nni.ConvReLU3d,
    (nn.Linear, nn.BatchNorm1d): fuse_linear_bn,
    (nn.Linear, nn.ReLU): nni.LinearReLU,
    (nn.BatchNorm2d, nn.ReLU): nni.BNReLU2d,
    (nn.BatchNorm3d, nn.ReLU): nni.BNReLU3d,
}

class ConvFreezebnReLUFusion(ConvBNReLUFusion):
    def __init__(self, quantizer: QuantizerCls, node: Node):
        super(ConvBNReLUFusion, self).__init__(quantizer, node)
        self.relu_node = None
        self.bn_node = None
        if (node.op == 'call_function' and node.target is torch.nn.functional.relu) or \
           (node.op == 'call_module' and type(quantizer.modules[node.target]) == torch.nn.ReLU):
            self.relu_node = node
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == 'call_module'
        if type(quantizer.modules[node.target]) in [torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, FrozenBatchNorm2d]:
            self.bn_node = node
            self.bn = quantizer.modules[self.bn_node.target]
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == 'call_module'
        self.conv_node = node
        self.conv = quantizer.modules[self.conv_node.target]


def fuse_linear_bn(linear, bn):
    r"""Given the linear and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type Linear
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Linear(10, 20)
        >>> b1 = nn.BatchNorm1d(20)
        >>> m2 = fuse_linear_bn(m1, b1)
    """
    assert(linear.training == bn.training),\
        "Linear and BN both must be in the same mode (train or eval)."

    if linear.training:
        assert bn.affine, 'Only support fusing BatchNorm1d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm1d with tracking_running_stats set to True'
        return qnn.intrinsic.LinearBn1d(linear, bn)
    else:
        return nn.utils.fusion.fuse_linear_bn_eval(linear, bn)


def fuse_deconv_bn(deconv, bn):
    assert(deconv.training == bn.training),\
        'DeConv and BN must be in the same mode (train or eval)'

    if deconv.training:
        assert bn.num_features == deconv.out_channels, 'Output channel of ConvTranspose2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        return qnni.ConvTransposeBn2d(deconv, bn)
    else:
        return fuse_deconv_bn_eval(deconv, bn)


def fuse_deconv_bn_relu(deconv, bn, relu):
    assert(deconv.training == bn.training == relu.training),\
        "DeConv and BN both must be in the same mode (train or eval)."

    if deconv.training:
        assert bn.num_features == deconv.out_channels, 'Output channel of ConvTranspose2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        return qnni.ConvTransposeBnReLU2d(deconv, bn, relu)
    else:
        return qnni.ConvTransposeReLU2d(fuse_deconv_bn_eval(deconv, bn), relu)


def fuse_conv_freezebn(conv, bn):
    assert(bn.training is False), "Freezebn must be eval."

    fused_module_class_map = {
        nn.Conv2d: qnni.ConvFreezebn2d,
    }

    if conv.training:
        assert bn.num_features == conv.out_channels, 'Output channel of Conv2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        fused_module_class = fused_module_class_map.get((type(conv)), None)
        return fused_module_class(conv, bn)
    else:
        return nn.utils.fuse_conv_bn_eval(conv, bn)


def fuse_conv_freezebn_relu(conv, bn, relu):
    assert(conv.training == relu.training and bn.training is False), "Conv and relu both must be in the same mode (train or eval) and bn must be eval."
    fused_module : Optional[Type[nn.Sequential]] = None
    if conv.training:
        map_to_fused_module_train = {
            nn.Conv2d: qnni.ConvFreezebnReLU2d,
        }
        assert bn.num_features == conv.out_channels, 'Output channel of Conv must match num_features of BatchNorm'
        assert bn.affine, 'Only support fusing BatchNorm with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm with tracking_running_stats set to True'
        fused_module = map_to_fused_module_train.get(type(conv), None)
        return fused_module(conv, bn, relu)
    else:
        map_to_fused_module_eval = {
            nn.Conv2d: nn.intrinsic.ConvReLU2d,
        }
        fused_module = map_to_fused_module_eval.get(type(conv), None)
        fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        return fused_module(fused_conv, relu)


def fuse_deconv_freezebn(deconv, bn):
    assert(bn.training is False), "Freezebn must be eval."

    if deconv.training:
        assert bn.num_features == deconv.out_channels, 'Output channel of ConvTranspose2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        return qnni.ConvTransposeFreezebn2d(deconv, bn)
    else:
        return fuse_deconv_bn_eval(deconv, bn)


def fuse_deconv_freezebn_relu(deconv, bn, relu):
    assert(deconv.training == relu.training and bn.training is False), "Conv and relu both must be in the same mode (train or eval) and bn must be eval."

    if deconv.training:
        assert bn.num_features == deconv.out_channels, 'Output channel of ConvTranspose2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        return qnni.ConvTransposeFreezebnReLU2d(deconv, bn, relu)
    else:
        return qnni.ConvTransposeReLU2d(fuse_deconv_bn_eval(deconv, bn), relu)


fuse_custom_config_dict = {
    "additional_fuser_method_mapping": {
        (torch.nn.Linear, torch.nn.BatchNorm1d): fuse_linear_bn,
        (torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d): fuse_deconv_bn,
        (torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d, torch.nn.ReLU): fuse_deconv_bn_relu,
        (torch.nn.ConvTranspose2d, torch.nn.ReLU): qnni.ConvTransposeReLU2d,
        (nn.Conv2d, FrozenBatchNorm2d, nn.ReLU): fuse_conv_freezebn_relu,
        (nn.Conv2d, FrozenBatchNorm2d): fuse_conv_freezebn,
        (nn.ConvTranspose2d, FrozenBatchNorm2d, nn.ReLU): fuse_deconv_freezebn_relu,
        (nn.ConvTranspose2d, FrozenBatchNorm2d): fuse_deconv_freezebn,
    },
    "additional_fusion_pattern": {
        (torch.nn.BatchNorm1d, torch.nn.Linear):
        ConvBNReLUFusion,
        (torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d):
        ConvBNReLUFusion,
        (torch.nn.ReLU, torch.nn.ConvTranspose2d):
        ConvBNReLUFusion,
        (torch.nn.ReLU, (torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d)):
        ConvBNReLUFusion,
        (torch.nn.functional.relu, torch.nn.ConvTranspose2d):
        ConvBNReLUFusion,
        (torch.nn.functional.relu, (torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d)):
        ConvBNReLUFusion,
        (torch.nn.ReLU, (FrozenBatchNorm2d, torch.nn.Conv2d)):
        ConvFreezebnReLUFusion,
        (FrozenBatchNorm2d, torch.nn.Conv2d):
        ConvFreezebnReLUFusion,
        (torch.nn.ReLU, (FrozenBatchNorm2d, torch.nn.ConvTranspose2d)):
        ConvFreezebnReLUFusion,
        (FrozenBatchNorm2d, torch.nn.ConvTranspose2d):
        ConvFreezebnReLUFusion,
    },
    "additional_qat_module_mappings": {
        nn.ConvTranspose2d: qnn.qat.ConvTranspose2d,
        qnni.LinearBn1d: qnniqat.LinearBn1d,
        qnni.ConvTransposeBn2d: qnniqat.ConvTransposeBn2d,
        qnni.ConvTransposeReLU2d: qnniqat.ConvTransposeReLU2d,
        qnni.ConvTransposeBnReLU2d: qnniqat.ConvTransposeBnReLU2d,
        qnni.ConvFreezebn2d: qnniqat.ConvFreezebn2d,
        qnni.ConvFreezebnReLU2d: qnniqat.ConvFreezebnReLU2d,
        qnni.ConvTransposeFreezebn2d: qnniqat.ConvTransposeFreezebn2d,
        qnni.ConvTransposeFreezebnReLU2d: qnniqat.ConvTransposeFreezebnReLU2d,
        nn.Embedding: qnn.qat.Embedding,
    },
}


def _sort_fusion_patterns(pats):
    keys = []
    for key in pats.keys():
        if pats[key] is ModuleReLUFusion:
            keys.append(key)
    for key in keys:
        pats.move_to_end(key)

# Sinse additional_fuser_method_mapping will not be set because fuser.py:54
# do not pass this dict.
from torch.quantization.quantization_mappings import DEFAULT_QAT_MODULE_MAPPINGS

DEFAULT_OP_LIST_TO_FUSER_METHOD.update(
    fuse_custom_config_dict['additional_fuser_method_mapping'])
DEFAULT_FUSION_PATTERNS.update(
    fuse_custom_config_dict['additional_fusion_pattern'])
# Make longer matched pattern prior.
# i.e. Conv + BN + Relu should match ConvBnRelu before BNRelu.
# Any thing registered in class ConvBNReLUFusion should be
# proir than class ModuleReLUFusion.
_sort_fusion_patterns(DEFAULT_FUSION_PATTERNS)
DEFAULT_QAT_MODULE_MAPPINGS.update(
    fuse_custom_config_dict['additional_qat_module_mappings'])
