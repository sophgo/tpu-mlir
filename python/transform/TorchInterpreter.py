# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .TorchHelper import *
import numpy as np

class TorchInterpreter():

    def __init__(self, torch_model):
        self.const_val = {}
        self.ref_tensor = {}
        self.coeff = {}

        if isinstance(torch_model, str):
            self.model = torch.jit.load(torch_model, map_location=torch.device('cpu'))
        else:
            self.model = torch_model
        self.model.eval()
        self.graph = self.model.inlined_graph

    def get_input(self, name):
        if name in self.const_val:
            return self.const_val[name]
        elif name in self.coeff:
            return self.coeff[name]
        else:
            return self.ref_tensor[name]

    def convert_attr(self, torch_node: TorchNode):
        node = torch_node.node_proto
        if node.output().type().kind() != 'TensorType':
            return
        data = get_attr(self.model, node)[0].detach()
        weight_name = node.output().debugName()
        self.coeff[weight_name] = data

    def convert_constant(self, torch_node: TorchNode):
        name, data, is_tensor = get_constant(torch_node.node_proto)
        if not is_tensor:
            if isinstance(data, int):
                if data > np.iinfo(np.int32).max:
                    data = np.iinfo(np.int32).max
            self.const_val[name] = data
        elif data is None:
            self.ref_tensor[name] = None
        else:
            self.coeff[name] = torch.from_numpy(data)

    def run_prim(self, node: TorchNode):
        assert node.op_type.split('::')[0] == 'prim'
        if node.op_type == "prim::ListConstruct" or node.op_type == "prim::TupleConstruct":
            self.ref_tensor[node.outputs[0]] = [self.get_input(name) for name in node.inputs]
        elif node.op_type in ["prim::TupleUnpack", "prim::ListUnpack"]:
            for i in range(len(node.outputs)):
                self.ref_tensor[node.outputs[i]] = self.get_input(node.inputs[0])[i]
        elif node.op_type == "prim::NumToTensor":
            self.ref_tensor[node.outputs[0]] = torch.tensor(self.get_input(node.inputs[0]))
        elif node.op_type == "prim::Constant":
            self.convert_constant(node)
        elif node.op_type == "prim::GetAttr":
            self.convert_attr(node)
        else:
            raise RuntimeError("{} not suppose".format(node.op_type))

    def run_aten(self, node: TorchNode):
        ParamMap = {
            "aten::ones": ['dtype', 'layout', 'device', 'pin_memory'],
            "aten::zeros": ['dtype', 'layout', 'device', 'pin_memory'],
            "aten::mean": ['dtype'],
            "aten::sum": ['dtype'],
            "aten::gelu": ['approximate'],
            "aten::add": ['alpha'],
            "aten::sub": ['alpha'],
            "aten::arange": ['dtype', 'layout', 'device', 'pin_memory'],
            "aten::addmm": ['beta', 'alpha'],
            "aten::to":
            ['dtype', 'layout', 'device', 'pin_memory', 'non_blocking', 'copy', 'memory_format'],
        }
        assert node.op_type.split('::')[0] == 'aten'
        # get input list
        input_list = [self.get_input(name) for name in node.inputs]

        # special aten op
        if node.op_type == "aten::contiguous":
            self.ref_tensor[node.outputs[0]] = self.get_input(node.inputs[0])
            return
        if node.op_type == "aten::Int":
            self.const_val[node.outputs[0]] = int(self.get_input(node.inputs[0]))
            return

        # get function
        func = getattr(torch.ops.aten, node.op_type.split('::')[1])
        if node.op_type == "aten::view":
            func = torch.ops.aten.reshape

        # run function
        if node.op_type == "aten::div":
            if len(input_list) == 2:
                output = func(*input_list)
            else:
                mode = input_list[2]
                input_list = input_list[:-1]
                output = func(*input_list, rounding_mode=mode)
        elif node.op_type in ParamMap.keys() and (node.op_type != "aten::to"
                                                  or len(node.inputs) > 6):
            end_param = ParamMap[node.op_type]
            param_len = len(end_param)
            param_dict = {}
            for i in range(param_len):
                param_dict[end_param[i]] = input_list[i - param_len]
            input_list = input_list[:-param_len]
            output = func(*input_list, **param_dict)
        else:
            output = func(*input_list)

        #save output
        for i in range(len(node.outputs)):
            out = output if len(node.outputs) == 1 else output[i]
            if isinstance(output, torch.Tensor):
                self.ref_tensor[node.outputs[i]] = out
            else:
                self.const_val[node.outputs[i]] = out

    def run_model(self, inputs: dict):
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                self.ref_tensor[key] = value
            else:
                self.ref_tensor[key] = torch.tensor(value)

        for node in self.graph.nodes():
            if node.kind().startswith('aten'):
                self.run_aten(TorchNode(node))
            else:
                self.run_prim(TorchNode(node))
