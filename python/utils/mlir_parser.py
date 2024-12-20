#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from typing import List
from itertools import chain
import sys
import mlir
import re
from mlir.ir import *
from mlir.dialects import quant
import mlir.ir

class Operation:
    cache_map = {}

    def __init__(self, op, body, idx):
        self.name = Operation.name(op)
        self.type = Operation.type(op)
        self.loc = Operation.loc(op)
        self.shape = Operation.shape(op)
        self.opds = Operation.operands_v2(op, body, idx)

        self.attrs = Operation.attrs(op)
        self.attrs = Operation.append_attr(op, self.attrs)
        self.outputs = Operation.outputs(op)
        self.op = op

    def __str__(self):
        return (
            self.name
            + ","
            + self.type
            + ","
            + self.loc
            + ","
            + str(self.shape)
            + ","
            + str(self.opds)
        )

    @staticmethod
    def name(op):
        loc = op.location
        if str(loc) == "loc(unknown)":
            return None
        num_results = len(op.results.types)
        if num_results > 1:
            names = re.findall(r'(fused\[.+?\])', str(loc))
            if (len(names) > 0):
                return names[0]
        return re.findall(r'\"(.+?)\"', str(loc))[0]

    @staticmethod
    def outputs(op):
        loc = op.location
        if loc == "loc(unknown)":
            return None
        return re.findall(r"\"(.+?)\"", str(loc))

    @staticmethod
    def type(op):
        return op.operation.name

    @staticmethod
    def str(value):
        return mlir.ir.StringAttr(value).value

    @staticmethod
    def bool(value):
        return mlir.ir.BoolAttr(value).value

    @staticmethod
    def int(value):
        return mlir.ir.IntegerAttr(value).value

    @staticmethod
    def int_array(value):
        return [mlir.ir.IntegerAttr(x).value for x in mlir.ir.ArrayAttr(value)]

    @staticmethod
    def fp_array(value):
        return [mlir.ir.FloatAttr(x).value for x in mlir.ir.ArrayAttr(value)]

    @staticmethod
    def attrs(op):
        arr_map = {}
        for i in range(len(op.attributes)):
            attr = op.attributes[i]
            k, v = str(attr.name), str(attr.attr)
            arr_map[k] = v
        return arr_map

    @staticmethod
    def append_attr(op, attrs):
        if len(op.results) != 1:
            return attrs
        shape_type = mlir.ir.ShapedType(op.results[0].type)
        element_type = shape_type.element_type
        if quant.UniformQuantizedType.isinstance(element_type):
            quant_type = quant.UniformQuantizedType(element_type)
            attrs["quant_scale"] = str(quant_type.scale)
            attrs["quant_zero_point"] = str(quant_type.zero_point)
        if quant.CalibratedQuantizedType.isinstance(element_type):
            quant_type = quant.CalibratedQuantizedType(element_type)
            attrs["calibrate_min"] = str(quant_type.min)
            attrs["calibrate_max"] = str(quant_type.max)
        return attrs

    @staticmethod
    def dictattr(op, field_name):
        return mlir.ir.DictAttr(op.attributes[field_name])

    @staticmethod
    def loc(op):
        return op.get_asm().split("=")[0].strip("% ")

    @staticmethod
    def shape(op):
        shape = []
        for result in op.results:
            if str(result.type) != "none":
                shape_type = mlir.ir.ShapedType(result.type)
                shape = [shape_type.get_dim_size(i) for i in range(shape_type.rank)] if shape_type.has_rank else []
                break
        return shape

    @staticmethod
    def operands(op, body, idx):
        opds = []
        for opd in op.operands:
            for j in reversed(range(idx)):
                prev_op = body.operations[j]
                if prev_op.results[0] == opd:
                    if Operation.type(prev_op) not in [
                        "tpu.None",
                        "top.None",
                        "tpu.load_weight",
                        "tpu.weight_file",
                    ]:
                        opds.append(Operation.name(prev_op))
        return opds

    @staticmethod
    def operands_v2(op, body, idx):
        opds = []

        for opd in op.operands:
            if opd in Operation.cache_map:
                for i, prev_op_name in Operation.cache_map[opd]:
                    if i < idx:
                        opds.append(prev_op_name)

        return opds


class MlirParser:
    def __init__(self, mlir_file):
        with open(mlir_file, "r") as f:
            context = f.read()
        self.ctx = mlir.ir.Context()
        self.ctx.allow_unregistered_dialects = True
        self.module = mlir.ir.Module.parse(context, self.ctx)
        self.body = self.module.body.operations[0].regions[0].blocks[0]
        self.attrs = Operation.attrs(self.module.operation)
        self.module_name = eval(self.attrs["sym_name"])
        self.module_state = eval(self.attrs["module.state"])
        self.module_weight_file = eval(self.attrs["module.weight_file"])
        self.module_chip = eval(self.attrs["module.chip"])
        self.ops: List[Operation] = []
        self.return_op = None
        self._none_type = mlir.ir.Type.parse("none", self.ctx)

        cache_map = {}
        for i in range(len(self.body.operations)):
            prev_op = self.body.operations[i]
            if (
                Operation.type(prev_op)
                not in [
                    "tpu.None",
                    "top.None",
                    "tpu.load_weight",
                    "tpu.weight_file",
                ]
                and len(prev_op.results) > 0
            ):
                skip_value = 0
                for idx, r in enumerate(prev_op.results):
                    if str(r.type) == "none":
                        skip_value += 1
                        continue
                    t = idx - skip_value
                    if t >= len(Operation.outputs(prev_op)):
                        t = len(Operation.outputs(prev_op)) - 1
                    cache_map.setdefault(r, []).append(
                        [i, Operation.outputs(prev_op)[t]]
                    )
        Operation.cache_map = cache_map

        for i in range(len(self.body.operations)):
            op = self.body.operations[i]
            type = Operation.type(op)
            if type in ["top.None", "top.Weight", "func.return"]:
                if type == "func.return":
                    self.return_op = op
                continue
            self.ops.append(Operation(op, self.body, i))
        self.inputs = []
        for op in self.ops:
            if op.type == "top.Input":
                self.inputs.append(op)

    def get_op_name_list(self):
        return [op.name for op in self.ops]

    def get_op_output_name_list(self):
        return list(chain(*[op.outputs for op in self.ops]))

    def get_input_num(self):
        return len(self.inputs)

    def get_input_shapes(self):
        mlir_shape = []
        for op in self.inputs:
            shape_type = mlir.ir.ShapedType(op.op.input.type)
            mlir_shape.append(shape_type.shape)
        return mlir_shape

    def get_input_types(self):
        mlir_type = []
        for op in self.inputs:
            shape_type = mlir.ir.ShapedType(op.op.input.type)
            mlir_type.append(str(shape_type.element_type))
        return mlir_type

    def get_input_op_by_idx(self, idx):
        return self.inputs[idx].op

    def get_batch_size(self):
        return Operation.shape(self.inputs[0].op)[0]

    def get_pre_op_by_op_name(self, op_name):
        op_input_tensor = []
        for op in self.ops:
            if op.name == op_name:
                for opd in op.opds:
                    if opd in self.get_op_output_name_list():
                        op_input_tensor.append(opd)
        return op_input_tensor

    def get_next_op_by_op_name(self, op_name):
        op_output_tensor = []
        for op in self.ops:
            if op_name in op.opds:
                if op.name in self.get_op_output_name_list():
                    op_output_tensor.append(op.name)
        return op_output_tensor

    def get_all_pre_ops_by_op_name(self, op_name):
        all_pre_ops = [op_name] + self.get_pre_op_by_op_name(op_name)
        cur_pre_ops = self.get_pre_op_by_op_name(op_name)
        while cur_pre_ops:
            tmp = cur_pre_ops.pop(0)
            new_pre_ops = self.get_pre_op_by_op_name(tmp)
            for new_pre_op in new_pre_ops:
                if new_pre_op not in all_pre_ops:
                    cur_pre_ops.append(new_pre_op)
                    if self.get_pre_op_by_op_name(new_pre_op):
                        all_pre_ops.append(new_pre_op)
        return all_pre_ops

    def get_all_next_ops_by_op_name(self, op_name):
        all_next_ops = [op_name] + self.get_next_op_by_op_name(op_name)
        cur_next_ops = self.get_next_op_by_op_name(op_name)
        while cur_next_ops:
            tmp = cur_next_ops.pop(0)
            new_next_ops = self.get_next_op_by_op_name(tmp)
            for new_next_op in new_next_ops:
                if new_next_op not in all_next_ops:
                    cur_next_ops.append(new_next_op)
                    all_next_ops.append(new_next_op)
        return all_next_ops

    def get_block_ops_by_op_name(self, name_list1, name_list2):
        all_pre_ops = set(self.get_all_pre_ops_by_op_name(name_list2))
        for name_list in name_list2:
            all_pre_ops.update(set(self.get_all_pre_ops_by_op_name(name_list)))

        all_next_ops = set(self.get_all_next_ops_by_op_name(name_list1))
        for name_list in name_list1:
            all_next_ops.update(set(self.get_all_next_ops_by_op_name(name_list)))
        block_ops = all_pre_ops.union(all_next_ops)
        return list(block_ops)

    def get_user_count_by_op_name(self, op_name):
        count = 0
        for op in self.ops:
            if op_name in op.opds:
                count += 1
        return count

    def get_use_count_by_op_name(self, op_name):
        count = 0
        for op in self.ops:
            count += op.opds.count(op_name)
        return count

    def get_outputs_by_op_name(self, op_name):
        for op in self.ops:
            if op.name == op_name:
                return op.outputs
        return None

    def get_op_by_op_name(self, op_name):
        for op in self.ops:
            if op.name == op_name:
                return op
        return None

    def get_opds_by_op_name(self, op_name):
        for op in self.ops:
            if op.name == op_name:
                return op.opds
        return None

    def get_op_type_by_op_name(self, op_name):
        for op in self.ops:
            if op.name == op_name:
                return op.type
        return None

    # the func is to get a dict with output names and corresponding shapes
    def get_output_op_names_n_shapes(self):
        if not self.return_op:
            return []
        outputs = {}
        for op in self.body.operations:
            if op == self.return_op:
                continue
            for opd in self.return_op.operands:
                if opd in op.results:
                    idx = list(op.results).index(opd)
                    shape_type = mlir.ir.ShapedType(opd.type)
                    shape = [shape_type.get_dim_size(i) for i in range(shape_type.rank)]
                    name = Operation.outputs(op)[idx]
                    outputs[name] = shape
        return outputs

    def get_middle_op_names_n_shape_type(self):
        middles = {}
        for i in range(len(self.body.operations)):
            op = self.body.operations[i]
            op_type = Operation.type(op)
            if op_type in ["top.None", "top.Input", "func.return"]:
                continue

            if op.results[0].type == self._none_type:
                shape_type = mlir.ir.ShapedType.parse("none", self.ctx)
            else:
                shape_type = mlir.ir.ShapedType(op.results[0].type)
            name = Operation.name(op)
            middles[name] = shape_type
        return middles

    def get_initializer_op_names_n_shape_type(self):
        initializer = {}
        for i in range(len(self.body.operations)):
            op = self.body.operations[i]
            type = Operation.type(op)
            if type != "top.Weight":
                continue
            shape_type = mlir.ir.ShapedType(op.results[0].type)
            name = Operation.name(op)
            initializer[name] = shape_type
        return initializer

    def collect_op_name_dict(self):
        op_dict = {}
        for op in self.ops:
            op_dict[op.name] = op
        return op_dict


if __name__ == "__main__":
    parser = MlirParser(sys.argv[1])
    for op in parser.ops:
        print(op.name, op.loc, op.shape)
    print(parser.module)
