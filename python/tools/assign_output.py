#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import onnx
import argparse
import copy
from transform.OnnxOpt import onnx_opt

class AssignOutput(object):
    def __init__(self, model):
        self.model = copy.deepcopy(model)
        onnx.checker.check_model(self.model)
        model = onnx.shape_inference.infer_shapes(model)
        self.inp_name = [i.name for i in model.graph.input]
        self.nodes = model.graph.node
        self.nodes_name = []
        self.needed_node = []
        self.out2name = {}
        for n in self.nodes:
            self.nodes_name.append(n.name)
            for o in n.output:
                self.out2name.update({o: n.name})
        # get shape info
        self.shape_info = [info for info in model.graph.value_info]
        self.shape_info.extend(model.graph.input)
        self.shape_info.extend(model.graph.output)
        self.shape_info = {info.name: {"shape": [i.dim_value for i in info.type.tensor_type.shape.dim if i.dim_value > 0],
                                       "dtype": info.type.tensor_type.elem_type}
                            for info in self.shape_info}

    def is_node(self, out):
        return out in self.out2name.keys()

    def assign_output(self, keys):
        oinput = []
        self.cleanup(self.model.graph.output)
        for k in keys:
            idx = self.nodes_name.index(k)
            for o in self.model.graph.node[idx].output:
                oinput.append(o)
                info = self.shape_info[o]
                output = onnx.helper.make_tensor_value_info(
                           o, info["dtype"], info["shape"])
                self.model.graph.output.extend([output])
        return oinput

    def find_needed_node(self, input):
        for k in input:
            if self.is_node(k):
                name = self.out2name[k]
                if name not in self.needed_node:
                    self.needed_node.append(name)
                    cur_node = self.model.graph.node[self.nodes_name.index(name)]
                    self.find_needed_node(cur_node.input)

    def cleanup(self, input):
        for i in range(len(input)):
            input.remove(input[0])

    def remove_weight(self):
        all_input = []
        unused_weight = []
        weights = self.model.graph.initializer
        all_node = [n for n in self.model.graph.node]
        for n in all_node:
            all_input.extend(n.input)
        for w in self.model.graph.initializer:
            if w.name in all_input:
                continue
            unused_weight.append(w)
        for w in unused_weight:
            weights.remove(w)

    def remove_node(self):
        self.cleanup(self.model.graph.node)
        for n in self.model.graph.node:
            self.model.graph.node.remove(n)
        for n in self.nodes:
            if n.name in self.needed_node:
                self.model.graph.node.append(n)

    def dump(self, name):
        data = self.model.SerializeToString()
        with open(name, "wb") as file:
            file.write(data)

    def run(self, tesnsor_name, new_onnx_name):
        oinp = self.assign_output(tesnsor_name)
        self.find_needed_node(oinp)
        self.remove_node()
        self.remove_weight()
        onnx.checker.check_model(self.model)
        self.dump(new_onnx_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="onnx model file")
    parser.add_argument("--output", required=True, help="assign outputs by using node name eg: conv1,conv2")
    args = parser.parse_args()
    model = onnx.load(args.model)
    opt_model, _ = onnx_opt(model)
    assigner = AssignOutput(opt_model)
    onames = [s.strip() for s in args.output.split(",")]
    new_model_name = args.model.replace(".onnx", "_new.onnx")
    assigner.run(onames, new_model_name)

