#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import sys
import ast
import pkgutil
import itertools


class Node(object):
    def __init__(self):
        self.depth = 1
        self.children = []
        self.text = None

    def add_child(self, node):
        self.children.append(node)
        return True


class Table(Node):
    """
    .. doctest::

        >>> tbl = Table('My friends', ['Name', 'Major Project'])
        >>> tbl.add_item(('Ramki', 'Python'))
        >>> tbl.add_item(('Pradeepto', 'Kde'))
        >>> print(tbl)

        .. list-table:: My friends
            :header-rows: 1

            * -  Name
              -  Major Project
            * -  Ramki
              -  Python
    """

    def __init__(self, title="", header=None, width=None):
        Node.__init__(self)
        self.text = title
        self.header = header
        self.width = width

    def add_item(self, row):
        """
        Adds a new row to the table.

        :arg row: list of items in the table.
        """
        self.children.append([("`" + txt + "`" if txt else txt) for txt in row])

    def __repr__(self):
        def print_table(header):
            items = []
            for i, hdr in enumerate(header):
                if i == 0:
                    items.append(f"    * -  {hdr}")
                else:
                    items.append(f"      -  {hdr}")
            return items

        out = [f".. list-table:: {self.text}"]
        if self.width:
            out.append(f"    :widths: {str(self.width)[1:-1]}")
        if self.header:
            out.append("    :header-rows: 1\n")
            out.extend(print_table(self.header))
        for ch in self.children:
            out.extend(print_table(ch))

        return "\n".join(out)


class CustomVisitor(ast.NodeVisitor):
    def __init__(self, op_set_name):
        super().__init__()
        self.op_set_name = op_set_name
        self.ops_name = []

    def visit_Assign(self, node):
        variable_name = ""
        keys = []

        if isinstance(node.value, ast.Str) and isinstance(node.targets[0], ast.Name):
            variable_name = node.targets[0].id
            keys = [node.value.s]

        elif isinstance(node.value, ast.Dict) and isinstance(
            node.targets[0], ast.Attribute
        ):
            variable_name = node.targets[0].attr
            keys = [ast.literal_eval(k) for k in node.value.keys]
        if variable_name == self.op_set_name:
            self.ops_name.extend(keys)


class OpSetBuilder:
    @classmethod
    def get_op_set(cls):
        module = pkgutil.get_loader(cls.module)
        if module:
            path = module.get_filename()
            visitor = CustomVisitor(cls.opset_name)
            module_ast = ast.parse(open(path, "r").read())
            visitor.visit(module_ast)
            ops = visitor.ops_name
            ops = sorted(cls.op_filter(ops), key=cls.op_sort_key)
            return {k: list(v) for k, v in itertools.groupby(ops, key=cls.op_group_fun)}

        raise FileExistsError(f"can not find: {cls.module}")


class Caffe(OpSetBuilder):
    module = "transform.CaffeConverter"
    opset_name = "caffeop_factory"
    op_filter = lambda x: x
    op_sort_key = lambda x: x.lower()
    op_group_fun = lambda x: x[0].upper()


class Onnx(OpSetBuilder):
    module = "transform.OnnxConverter"
    opset_name = "onnxop_factory"
    op_filter = lambda x: x
    op_sort_key = lambda x: x.lower()
    op_group_fun = lambda x: x[0].upper()


class Pytorch(OpSetBuilder):
    module = "transform.TorchConverter"
    opset_name = "op_factory"
    op_filter = lambda x: filter(lambda y: "aten::" in y, x)
    op_sort_key = lambda x: "".join(filter(lambda y: y != "_", x))
    op_group_fun = lambda x: x[6].upper() if x[6] != "_" else x[7].upper()  # aten::?


class TOP(OpSetBuilder):
    module = "mlir.dialects._top_ops_gen"
    opset_name = "OPERATION_NAME"
    op_filter = lambda x: x
    op_sort_key = lambda x: x
    op_group_fun = lambda x: x[4].upper()  # top.?


if __name__ == "__main__":
    tables = []
    frameworks = [Onnx, Pytorch, Caffe, TOP]
    header = [x.__name__ for x in frameworks]
    op_set = [x.get_op_set() for x in frameworks]
    for i in range(97, 123):
        alph_key = chr(i).upper()
        ops_set = list(x.get(alph_key, []) for x in op_set)
        if any(ops_set):
            tbl = Table(alph_key, header, width=[40, 50, 40, 40])
            for ops in itertools.zip_longest(*ops_set, fillvalue=""):
                tbl.add_item(ops)
            tables.append(tbl)

    with open(sys.argv[1], "w") as f:
        f.write("\n\n".join(str(x) for x in tables))
