#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import os
from typing import Tuple
from itertools import chain
from typing import Dict
from .nodes import *
import mlir.ir

AST_CONTEXT: "MlirAST" = None


class MlirASTParser:
    def __init__(self, mlir_file) -> None:
        with open(mlir_file) as r:
            self.mlir_fn = mlir_file
            self.lines = r.readlines()
            self.lno = 0
        self._module = None
        self.ast = MlirAST()
        self.ast.mlir_file = os.path.abspath(mlir_file)

    def reload(self):
        self.ast = MlirAST()
        self.lno = 0
        self.parse()

    def next_line(self):
        # ./hrnet_bm1684x_f32_final.mlir
        if len(self.lines) > self.lno:
            res = self.lines[self.lno].rstrip()
            self.lno += 1
            # next(self.tqdm)
            return res
        return ""

    def parse(self):
        with self.ast:
            line = self.next_line()
            while line != "":
                self.parse_line(line)
                line = self.next_line()

            # module should be added at last (after all locations loaded)
            self.ast.add_module(self._module)
        return self.ast

    def parse_line(self, line: str, context=None):
        # if line
        if len(line) == 0 or line.startswith("\n"):
            return
        line = line.strip()

        if line[0].startswith("#"):
            self.parse_location(location_token=line)

        if line.startswith("module"):
            self._module = self.parse_module(line)

    def parse_location(self, location_token):
        loc = Location.parse(location_token)
        self.ast.add_location(loc)

    def parse_module(self, module_start_line: str):
        name_start_idx = module_start_line.find("@") + 1
        name_end_idx = module_start_line.find("attributes") - 1

        attr_start_idx = module_start_line.find("{")

        attr_end_idx = module_start_line.find("}") + 1

        name = module_start_line[name_start_idx:name_end_idx]
        attr_content = module_start_line[attr_start_idx:attr_end_idx]
        attrs = Attributes.parse(attr_content)
        funcs = []
        line = self.next_line()
        sub_modules = []
        while line.strip() != "} loc(#loc)":
            if line.lstrip().startswith("module"):
                sub_module = self.parse_module(line)
                assert sub_module.dump_head().strip() == line.strip()
                sub_modules.append(sub_module)
            elif line.lstrip().startswith("func.fun"):
                func = self.parse_func(line)
                funcs.append(func)
            else:
                raise NotImplementedError(line)
            line = self.next_line()

        module = Module(name=name, attrs=attrs, funcs=funcs, sub_modules=sub_modules)
        return module

    def parse_func(self, func_line: str) -> Func:
        name_start_idx = func_line.find("@") + 1
        name_end_idx = func_line.find("(", name_start_idx)

        finputs_start_idx = name_end_idx + 1
        finputs_end_idx = func_line.find(") ->", finputs_start_idx)

        dtype_start_idx = finputs_end_idx + 5
        attr_start_idx = dtype_end_idx = func_line.find("attributes", dtype_start_idx)
        if dtype_end_idx == -1:
            dtype_end_idx = func_line.find(" {", dtype_start_idx)

        attr = None
        if attr_start_idx != -1:
            attr_start_idx += 11
            attr_end_idx = func_line.find("{", attr_start_idx + 1)
            func_attr_str = func_line[attr_start_idx:attr_end_idx]
            attr = Attributes.parse(func_attr_str)

        func_name = func_line[name_start_idx:name_end_idx]
        func_inputs_str = func_line[finputs_start_idx:finputs_end_idx]
        func_output_types_str = func_line[dtype_start_idx:dtype_end_idx]
        func_output_types = Type.parse_output(func_output_types_str)

        arg_names, arg_types, arg_locs = Func.parse_func_inputs(func_inputs_str)

        ops = []
        line = self.next_line()

        while line.strip() != "} loc(#loc)":
            if '"tpu.Group"' in line:
                op = self.parse_groupop(line)
            elif line.lstrip().startswith("%"):
                line = line.lstrip()
                try:
                    op = self.parse_simple_operation(line)
                except:
                    raise NotImplementedError(line)
            elif line.lstrip().startswith("return"):
                op = self.parse_return(line)
            elif "tpu.Yield" in line:
                op = self.parse_yield(line)
            elif "return" in line:
                op = self.parse_return(line)
            else:
                raise NotImplementedError(line)
            ops.append(op)
            line = self.next_line()

        func = Func(
            func_name, arg_names, arg_types, arg_locs, func_output_types, attr, ops
        )
        return func

    def parse_groupop(self, operation_line) -> GroupOp:
        ops: List[Operation] = []

        attr = input_types = output_types = loc_label = None
        op_id_str, op_define = operation_line.strip().split("=", maxsplit=1)
        op_id_str = op_id_str.strip()

        op_type_end_idx = op_define.rfind("(")
        op_type_str = op_define[:op_type_end_idx].strip()
        op_type = OperationType.parse(op_type_str)

        line = self.next_line()

        while line.strip() != "} loc(#loc)":
            if line.lstrip().startswith("%"):
                op = self.parse_simple_operation(line)
            elif "tpu.Yield" in line:
                op = self.parse_yield(line)
            elif line.lstrip().startswith("})"):
                attr_start_idx = line.find("{")
                attr_end_idx = line.find("}", attr_start_idx) + 1
                attr_str = line[attr_start_idx:attr_end_idx]
                attr = Attributes.parse(attr_str)

                input_start_idx = line.find(":") + 1
                input_end_idx = line.find("->", input_start_idx)
                input_type_str = line[input_start_idx:input_end_idx]

                output_start_idx = input_end_idx + 3
                output_str = line[output_start_idx:]
                input_types = Type.parse_inputs_tuple(input_type_str)
                output_types, loc_label = parse_outputs_str(output_str)
                break
            else:
                raise NotImplementedError(line)
            ops.append(op)
            line = self.next_line()

        group_op = GroupOp(
            op_id_str.split(":")[0],
            op_type,
            ops,
            attr,
            input_types,
            output_types,
            loc_label,
        )
        for op in ops:
            op._parent = group_op
        return group_op

    def parse_simple_operation(self, operation_line: str) -> Operation:
        """
        <op_id_str> = <type_name> (<attr>) : <inputs> -> <outputs>
        """
        op = Operation.parse(operation_line)
        return op

    def parse_return(self, operation_line) -> Return:
        return_op = Return.parse(operation_line)
        return return_op

    def parse_yield(self, yield_line) -> Yield:
        yield_op = Yield.parse(yield_line)
        # self.ast.add_yield(yield_op)
        return yield_op


class MlirAST:
    def __init__(self) -> None:
        self.mlir_file = None
        self.ctx = mlir.ir.Context()
        self.ctx.allow_unregistered_dialects = True

        self.locid2op = {}
        self.locid2opname = {}
        self.opname2locid = {}
        self.opid2op: Dict[str, Operation] = {}

        self.module: Module = None
        self.locid2loc: Dict[str, Location] = {}

        self.optype2op = {}

        self.opname2preop: Dict[str, List[Operation]] = {}
        self.opname2op: Dict[str, Operation] = {}
        self.opname2nextop: Dict[str, List[Operation]] = {}

        self.return_op = None

        self.max_opd_id: int = 0
        self.max_loc_id: int = 0
        self.name2func: Dict[str, Func] = {}

    def __enter__(self):
        global AST_CONTEXT
        assert AST_CONTEXT is None
        AST_CONTEXT = self

    def __exit__(self, *args, **kwargs):
        global AST_CONTEXT
        assert AST_CONTEXT == self
        AST_CONTEXT = None

    @property
    def is_final(self) -> bool:
        module_state = self.module.attrs["module.state"]
        return module_state == '"TPU_ADDRESSED"'

    @property
    def ops(self) -> List[Operation]:
        op_lis = []
        for func in self.module.funcs:
            for op in func.ops:
                op_lis.append(op)
        return op_lis

    def update_max_opd_id(self, opd_id_str: str):
        try:
            self.max_opd_id = max(int(opd_id_str[1:]), self.max_opd_id)
        except:
            pass

    def malloc_opd_id(self):
        self.max_opd_id += 1
        return f"%{self.max_opd_id}"

    def malloc_loc_id(self):
        "#loc660"
        self.max_loc_id += 1
        return f"#loc{self.max_loc_id}"

    def add_operation(self, op: Operation, func: Func, group_op: GroupOp = None):
        self.locid2op[op.loc_label.loc_id_str] = op
        self.optype2op.setdefault(op.op_type.op_type_name, []).append(op)
        op_name = self.locid2opname[op.loc_label.loc_id_str]

        if group_op is None:
            local_id_start = 1e7
        else:
            local_id_start = int(group_op.ops[0].opd_ids[0][1:])

        def is_local_opid(opd_str):
            try:
                return int(opd_str[1:]) >= local_id_start
            except:
                return False

        use_ns = self.is_final
        ns = group_op.name if group_op is not None and use_ns else ""
        func_ns = func.name + "::" if func is not None and use_ns else ""

        for opd_id in op.opd_ids:
            self.update_max_opd_id(opd_id)
            self.opid2op[func_ns + ns + opd_id] = op
        self.opname2op[op_name] = op
        op.name = op_name

        preops = self.opname2preop.setdefault(op_name, [])
        if not op.op_type.isa("top.Input", "top.Weight", "top.None", "tosa.const"):
            visited = set()
            for iop in op.op_type.opds:
                if "arg" in iop:
                    # skip final.mlir function input
                    continue

                if is_local_opid(iop):
                    ns_iop = func_ns + ns + iop
                else:
                    # load out of this group op
                    ns_iop = func_ns + iop

                pre_op = self.opid2op[ns_iop]

                if pre_op.op_type.isa("top.None", "top.Weight"):
                    continue
                preops.append(pre_op)

                if iop in visited:
                    continue
                visited.add(iop)

                nextops = self.opname2nextop.setdefault(
                    self.get_op_name_by_op_id(ns_iop), []
                )
                nextops.append(op)
            try:
                pass
            except:
                print(f"skip add preops for {op.dump()}")
                return

    def set_return(self, op: Return):
        self.return_op = op

    def add_function(self, func: Func):
        self.name2func[func.name] = func
        for op in func.ops:
            if isinstance(op, Return):
                self.set_return(op)
            elif isinstance(op, Yield):
                pass
            elif isinstance(op, Operation):
                self.add_operation(op, func=func)
                if isinstance(op, GroupOp):
                    for oop in op.ops:
                        self.add_operation(oop, func=func, group_op=op)

    def add_module(self, module: Module):
        self.module = module
        [self.add_function(i) for i in module.funcs]

    def add_location(self, loc: Location):
        loc_name = loc._loc_name if not loc.isfused else self.locid2opname[loc.fused[0]]
        self.locid2opname[loc.loc_id_str] = loc_name
        self.opname2locid[loc_name] = loc.loc_id_str
        self.locid2loc[loc.loc_id_str] = loc

        try:
            self.max_loc_id = max(int(loc.loc_id_str[4:]), self.max_loc_id)
        except:
            pass

    def get_op_name_by_op_id(self, opd_str: str) -> str:
        """
        for Operation input, use op.op_type.opds
        for Operation output, use op.opd_ids
        """
        op = self.opid2op[opd_str]
        return self.locid2opname[op.loc_label.loc_id_str]

    def get_op_by_op_name(self, op_name: str) -> Operation:
        return self.opname2op[op_name]

    def dump(self):
        module_str = self.module.dump()
        location_str = "\n".join([i.dump() for i in self.locid2loc.values()])
        return f"{module_str}\n{location_str}\n"


class MlirParserV2:
    def __init__(self, mlir_file):
        self.ctx = mlir.ir.Context()
        self.ctx.allow_unregistered_dialects = True
        parser = MlirASTParser(mlir_file)
        parser.parse()
        self.ast = parser.ast

        self.attrs = self.ast.module.attrs.to_dict()
        self.module = self.ast.module
        self.module_state = self.attrs["module.state"]
        self.module_weight_file = self.attrs["module.weight_file"]
        self.module_chip = self.attrs["module.chip"]
        self.ops = self.ast.ops
        self.return_op = self.ast.return_op

        self.inputs = []
        for op in self.ops:
            if op.type == "top.Input":
                self.inputs.append(op)

    @staticmethod
    def from_astparser(parser: MlirASTParser):
        self = MlirParserV2.__new__()  # type: MlirParserV2
        self.ctx = mlir.ir.Context()
        self.ctx.allow_unregistered_dialects = True
        self.ast = parser.ast

        self.attrs = self.ast.module.attrs.to_dict()
        self.module = self.ast.module
        self.module_name = self.attrs["module.name"]
        self.module_state = self.attrs["module.state"]
        self.module_weight_file = self.attrs["module.weight_file"]
        self.module_chip = self.attrs["module.chip"]
        self.ops = self.ast.ops
        self.return_op = self.ast.return_op

        self.inputs = []
        for op in self.ops:
            if op.type == "top.Input":
                self.inputs.append(op)
        return self

    def get_op_by_op_id(self, op_id: str) -> Operation:
        return self.ast.opid2op[op_id]

    def get_op_name_by_op_id(self, op_id_str: str) -> str:
        return self.ast.get_op_name_by_op_id(op_id_str)

    def get_op_name_by_op(self, op: Operation) -> str:
        return self.ast.locid2opname[op.loc_label.loc_id_str]

    def get_op_name_list(self) -> List[str]:
        op_names = []
        for func in self.module.funcs:
            for op in func.ops:
                if not op.op_type.isa("top.None", "top.Weight", "func.return"):
                    op_name = self.ast.locid2opname.get(op.loc_label.loc_id_str, None)
                    op_names.append(op_name)
        return op_names

    def get_input_num(self) -> int:
        return len(self.ast.optype2op["top.Input"])

    def get_input_op_by_idx(self, idx) -> Operation:
        return self.ast.optype2op["top.Input"][idx]

    def get_batch_size(self) -> int:
        return self.get_input_op_by_idx(0).input_types[0].shape[0]

    def get_pre_op_by_op_name(self, op_name: str) -> List[str]:
        return [
            self.get_op_name_by_op(op) for op in self.ast.opname2preop.get(op_name, [])
        ]

    def get_next_op_by_op_name(self, op_name: str) -> List[str]:
        return [
            self.get_op_name_by_op(op) for op in self.ast.opname2nextop.get(op_name, [])
        ]

    def get_all_next_ops_by_op_name(self, op_name: str) -> List[str]:
        queue = [op_name]
        counter = Counter()
        while len(queue) > 0:
            top = queue.pop()
            counter.update([top])
            next_ops = self.get_next_op_by_op_name(top)
            next_ops = [next_op for next_op in next_ops if next_op not in counter]
            queue.extend(next_ops)
        return list(counter.keys())

    def get_all_pre_ops_by_op_name(self, op_name: str) -> List[str]:
        queue = [op_name]
        counter = Counter()
        while len(queue) > 0:
            top = queue.pop()
            counter.update([top])
            pre_ops = self.get_pre_op_by_op_name(top)
            pre_ops = [pre_op for pre_op in pre_ops if pre_op not in counter]
            queue.extend(pre_ops)
        return list(counter.keys())

    def get_block_ops_by_op_name(self, name_list1: str, name_list2: str) -> List[str]:
        all_pre_ops = set(self.get_all_pre_ops_by_op_name(name_list2))
        for name_list in name_list2:
            all_pre_ops.update(set(self.get_all_pre_ops_by_op_name(name_list)))

        all_next_ops = set(self.get_all_next_ops_by_op_name(name_list1))
        for name_list in name_list1:
            all_next_ops.update(set(self.get_all_next_ops_by_op_name(name_list)))
        block_ops = all_pre_ops.union(all_next_ops)
        return list(block_ops)

    def get_user_count_by_op_name(self, op_name: str) -> int:
        """count operations that use this op
        if operation has multiple outputs, any usage of its output should be taken into considered
        """
        # count = 0
        # op = self.get_op_by_op_name(op_name)
        return len(self.ast.opname2nextop.get(op_name, []))

    def get_use_count_by_op_name(self, op_name: str) -> int:
        """count args that use this op as arg ( sub(%1,%1) takes two )"""
        count = 0
        op = self.ast.opname2op[op_name]
        for opd_id in op.opd_ids:
            for next_op in self.ast.opname2nextop.get(op_name, []):
                count += next_op.op_type.opds.count(opd_id)
        return count

    def get_outputs_by_op_name(self, op_name: str) -> str:
        loc_id = self.ast.opname2locid[op_name]
        loc = self.ast.locid2loc[loc_id]
        if loc.isfused:
            return [self.ast.locid2opname[i] for i in loc.fused]
        else:
            return [loc._loc_name]

    def get_op_by_op_name(self, op_name: str) -> Operation:
        return self.ast.get_op_by_op_name(op_name)

    def get_opds_by_op_name(self, op_name: str):
        op = self.ast.opname2op[op_name]
        if op.op_type.isa("top.Input"):
            return []
        return [
            self.get_op_name_by_op_id(iop)
            for iop in op.op_type.opds
            if not self.ast.opid2op[iop].op_type.isa("top.None")
        ]

    def get_op_type_by_op_name(self, op_name: str) -> str:
        """top.Conv"""
        return self.ast.opname2op[op_name].op_type.op_type_name

    def get_output_op_names_n_shapes(self):
        outputs = {}
        for i, idx in enumerate(self.return_op.output_ids):
            op = self.get_op_by_op_id(idx)
            op_name = self.ast.locid2opname[op.loc_label.loc_id_str]
            shape = self.return_op.output_types[i].shape
            outputs[op_name] = shape
        return outputs

    def get_middle_op_names_n_shape_type(self):
        middles = {}
        for func in self.module.funcs:
            for op in func.ops:
                if op.op_type.isa("top.None", "top.Input", "func.return"):
                    continue
                op_name = self.get_op_name_by_op(op)
                middles[op_name] = mlir.ir.ShapedType.parse(
                    op.output_types[0].dump(), self.ctx
                )

        return middles

    def get_initializer_op_names_n_shape_type(self):
        middles = {}
        for func in self.module.funcs:
            for op in func.ops:
                if not op.op_type.isa("top.Weight"):
                    continue
                op_name = self.get_op_name_by_op(op)
                middles[op_name] = mlir.ir.ShapedType.parse(
                    op.output_types[0].dump(), self.ctx
                )

        return middles

    def get_all_parents_with_distance(self, op_name: str) -> Dict[str, int]:
        queue = [(op_name, 0)]
        visited = set()
        dic = {}
        while len(queue) > 0:
            top, distance = queue.pop()
            visited.update([top])
            pre_ops = self.get_pre_op_by_op_name(top)
            pre_ops = [(pre_op, distance + 1) for pre_op in pre_ops]
            dic[top] = min(dic.get(top, 1e10), distance)
            queue.extend(pre_ops)
        return dic

    def get_all_children_with_distance(self, op_name: str) -> Dict[str, int]:
        queue = [(op_name, 0)]
        visited = set()
        dic = {}
        while len(queue) > 0:
            top, distance = queue.pop()
            visited.update([top])
            pre_ops = self.get_next_op_by_op_name(top)
            pre_ops = [(pre_op, distance + 1) for pre_op in pre_ops]
            dic[top] = min(dic.get(top, 1e10), distance)
            queue.extend(pre_ops)
        return dic

    def find_earliest_common_parent(self, a: str, b: str) -> List[Tuple[str, int]]:
        """
        for DAG, multiple parents may be found

        return:
            List of tuple, in each tuple:
            first element: operation name of common parent
            second element (int): operation distance from a and b
        """

        a_parent = self.get_all_parents_with_distance(a)

        cand = []
        queue = [(b, 0, 0)]
        counter = Counter()

        branch_map = {}
        unique_id = 0

        def unique_branch(pre_op, branch_id, idx):
            nonlocal unique_id
            if pre_op in branch_map:
                return branch_map[pre_op]

            if idx == 0:
                return branch_id
            unique_id += 1
            return unique_id

        while len(queue) > 0:
            top, branch_id, distance = queue.pop()
            branch_map[top] = branch_id
            counter.update([top])
            pre_ops = self.get_pre_op_by_op_name(top)
            pre_ops = [
                (pre_op, unique_branch(pre_op, branch_id, i), distance + 1)
                for i, pre_op in enumerate(pre_ops)
                if pre_op not in counter
            ]

            for pre_op_tuple in pre_ops:
                if top not in a_parent and pre_op_tuple[0] in a_parent:
                    cand.append([pre_op_tuple[0], distance])
                else:
                    queue.append(pre_op_tuple)

        return [(i[0], i[1] + a_parent[i[0]]) for i in cand]

    def find_common_operations(self, names: List[str]) -> List[str]:
        dic = {}

        if len(names) == 1:
            # dic[]
            pass

        counter = Counter()
        for name in names:
            queue = [name]
            sub_counter = Counter()
            while len(queue) > 0:
                top = queue.pop()
                counter.update([top])
                pre_ops = self.get_pre_op_by_op_name(top)
                pre_ops = [pre_op for pre_op in pre_ops if pre_op not in sub_counter]
                queue.extend(pre_ops)
                for pre in pre_ops:
                    dic.setdefault(pre, set()).add(name)
            counter.update(sub_counter)

        return dic, counter
