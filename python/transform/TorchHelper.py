# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import torch


class BaseNode():

    def __init__(self, info):
        self.name = str(info["name"])
        self.op_type = str(info["op_type"])
        self.inputs = list(info["inputs"])
        self.outputs = list(info["outputs"])


class TorchNode(BaseNode):

    def __init__(self, node):
        info = dict()
        op_type = node.kind()
        info["op_type"] = op_type if not op_type.endswith("_") else op_type[:-1]
        info["inputs"] = [inp.debugName() for inp in node.inputs()]
        info["outputs"] = [outp.debugName() for outp in node.outputs()]
        info["name"] = info["outputs"][0]
        super().__init__(info)
        self.node_proto = node


def get_attr(model: torch.jit.RecursiveScriptModule, node: torch.Node):
    if node.kind() == 'prim::Param':
        return (model, '')
    if node.kind() == 'prim::GetAttr':
        name = node.s('name')
        obj, parent = get_attr(model, node.input().node())
        return (getattr(obj, name), parent + '.' + name if len(parent) > 0 else name)


def get_constant(TorchNode: TorchNode, node: torch.Node):
    """Retrieve a constant associated with this prim::Constant node"""
    attribute_names = node.attributeNames()
    num_attributes = len(attribute_names)
    name = TorchNode.outputs[0]
    is_tensor = False
    type = node.output().type().kind()
    value = None
    if type == "NoneType":
        return name, None, True
    elif num_attributes == 1:
        attr_name = attribute_names[0]
        if type == "IntType":
            value = node.i(attr_name)
        elif type == "BoolType":
            value = bool(node.i(attr_name))
        elif type in ["FloatType", "LongType"]:
            value = node.f(attr_name)
        elif type in ["DeviceObjType", "StringType"]:
            value = node.s(attr_name)
        elif type in ["TensorType", "CompleteTensorType"]:
            is_tensor = True
            tensor = node.t(attr_name)
            if tensor.is_cuda:
                tensor = tensor.cpu()
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            value = tensor.numpy()
        else:
            raise NotImplementedError("Unsupported type: %s" % type)
    else:
        assert num_attributes == 0
        return None
    return name, value, is_tensor

def get_attr_name(TorchNode: TorchNode):
    name_native      = TorchNode.node_proto.output().debugName()
    name_with_MaskRCNN_prefix = TorchNode.outputs[0]
    return name_with_MaskRCNN_prefix
