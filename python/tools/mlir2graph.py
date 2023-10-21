#!/usr/bin/python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# graphvis api doc: https://graphviz.org/doc/info/lang.html
# ==============================================================================
from mlir_ast import MlirASTParser
import os
import numpy as np
from mlir_ast.nodes import GroupOp, Operation, CallFunc, Func
import argparse
import pydot
from pprint import pformat
import json


def escape(name: str):
    return name.replace(":", "_")


def is_local_opid(opd_str: str, local_id_start: int):
    """
    %326
    %10#1
    """

    try:
        return int(opd_str[1:]) >= local_id_start
    except:
        return False


label_template = """
<<table border="0" cellborder="1" cellspacing="0">
    <tr><td align="center">{opd_ids} = {op_type} -&gt; {shape}</td></tr>
    <tr><td align="center">{name}</td></tr>
</table>>
""".strip()


def first_valid_op(func: Func):
    for op in func.ops:
        if not op.op_type.isa("top.None"):
            return op
    raise ValueError("No Valid Operation")


def make_label(op: Operation, **kwargs):
    opd_ids = ", ".join(op.opd_ids)
    op_type = op.op_type.dump()
    shape = ", ".join(["x".join(map(str, i.shape)) for i in op.output_types])

    html = label_template.format(
        opd_ids=opd_ids, op_type=op_type, shape=shape, name=op.name
    )
    # print(html)
    return html


def make_tooltips(op: Operation):
    print(json.dumps(op.attrs, ensure_ascii=False, indent=2))
    res = json.dumps(op.attrs, ensure_ascii=False, indent=2)
    res = f"<{res}>"
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlir", required=True, help="model name")
    parser.add_argument(
        "--output", help="output dot file, if not assigned, use {mlir file}.{suffix}"
    )
    parser.add_argument(
        "--bmodel_checker_data",
        type=str,
        default=None,
        help="for final.mlir, a bmodel_checker_data will render red for failed operation",
    )
    parser.add_argument(
        "--isbig",
        type=bool,
        default=False,
        help="for mlir file with large number of operations, use this option to spped up graph generation.",
    )

    args = parser.parse_args()

    # dot = graphviz.Digraph(comment="The Round Table", node_attr={"shape": "box"})
    dot = pydot.Dot("my_graph", graph_type="digraph", compound=True)

    failed_keys = set()
    if args.bmodel_checker_data is not None:
        report = np.load(args.bmodel_checker_data, allow_pickle=True)
        failed_keys = [k.split("_asm_")[0] for k in list(report.files) if "actual" in k]

    ast_parser = MlirASTParser(args.mlir)
    ast = ast_parser.parse()

    is_final = len(ast.module.funcs) > 1

    func_inputs = set()

    for i, func in enumerate(ast.module.funcs):
        func_graph = pydot.Subgraph(
            f"cluster_{func.name}", labeljust="l", label=func.name
        )

        func_ns = func.name + "::"
        for op in func.ops:
            if op.op_type.isa("top.None", "func.return"):
                continue

            if op.op_type.isa("top.Input"):
                node = pydot.Node(
                    escape(op.name),
                    id=escape(op.name),
                    label=make_label(op),
                    shape="plain",
                )
                node.set_tooltip(make_tooltips(op))
                continue

            if isinstance(op, CallFunc):
                for iop in op.op_type.opds:
                    if "arg" in iop:
                        continue

                    pre_op = ast.opid2op[func_ns + iop]
                    pre_op_name = pre_op.name
                    op_func = ast.name2func[op.op_type.op_type_name]
                    lhead = f"cluster_{op.op_type.op_type_name}"

                    if isinstance(pre_op, CallFunc):
                        ltail = f"cluster_{pre_op.op_type.op_type_name}"
                        pre_func = ast.name2func[pre_op.op_type.op_type_name]

                        edge = pydot.Edge(
                            escape(first_valid_op(pre_func).name),
                            escape(first_valid_op(op_func).name),
                            label=iop,
                            ltail=ltail,
                            lhead=lhead,
                            href=f"#{escape(pre_op_name)}",
                        )
                    else:
                        edge = pydot.Edge(
                            escape(pre_op_name),
                            escape(first_valid_op(op_func).name),
                            label=iop,
                            lhead=lhead,
                            href=f"#{escape(pre_op_name)}",
                        )

                    dot.add_edge(edge)
                continue

            if isinstance(op, GroupOp):
                local_id_start = int(op.ops[0].opd_ids[0][1:])

                group_graph = pydot.Subgraph(
                    "cluster_" + escape(op.name),
                    id="cluster_" + escape(op.name),
                    label=make_label(op),
                    shape="plain",
                    labeljust="l",
                )
                ns = op.name
                for oop in op.ops:
                    if oop.op_type.isa("tpu.Yield", "tpu.Store"):
                        continue

                    op_name = oop.name
                    node_attrs = {}
                    # node_attrs["shape"] = "box"

                    for iop in oop.op_type.opds:
                        if "arg" in iop:
                            continue

                        if is_local_opid(iop, local_id_start):
                            pre_op = ast.opid2op[func_ns + ns + iop]
                        else:
                            pre_op = ast.opid2op[func_ns + iop]

                        if pre_op.op_type.isa("top.None", "tpu.Yield"):
                            continue
                        pre_op_name = pre_op.name
                        if pre_op_name == oop.name:
                            continue

                        if isinstance(pre_op, GroupOp):
                            edge = pydot.Edge(
                                escape(pre_op.ops[0].name),
                                escape(oop.name),
                                label=iop,
                                ltail="cluster_" + escape(pre_op.name),
                                href=f"#cluster_{escape(pre_op_name)}",
                            )
                        else:
                            edge = pydot.Edge(
                                escape(pre_op_name),
                                escape(oop.name),
                                label=iop,
                                href=f"#{escape(pre_op_name)}",
                            )

                        dot.add_edge(edge)

                    if op_name in failed_keys:
                        node_attrs["color"] = "red"
                    node = pydot.Node(
                        escape(oop.name),
                        id=escape(oop.name),
                        **node_attrs,
                        label=make_label(oop),
                        shape="plain",
                    )
                    node.set_tooltip(make_tooltips(oop))
                    group_graph.add_node(node)
                func_graph.add_subgraph(group_graph)
            else:
                op_name = op.name
                if escape(op_name) == "onnx__Transpose_266_Reshape":
                    import pdb

                    pdb.set_trace()
                node_attrs = {}
                # node_attrs["shape"] = "box"

                for iop in op.op_type.opds:
                    if "arg" in iop:
                        continue

                    pre_op = ast.opid2op[func_ns + iop]

                    if pre_op.op_type.isa("top.None", "tpu.Yield"):
                        "skip generate edge for None and Yield"
                        continue

                    pre_op_name = pre_op.name
                    if pre_op_name == op.name:
                        continue

                    if isinstance(pre_op, GroupOp):
                        edge = pydot.Edge(
                            escape(pre_op.ops[-1].name),
                            escape(op.name),
                            label=iop,
                            ltail="cluster_" + escape(pre_op.name),
                            href=f"#cluster_{escape(pre_op_name)}",
                        )
                    else:
                        edge = pydot.Edge(
                            escape(pre_op_name),
                            escape(op.name),
                            label=iop,
                            href=f"#{escape(pre_op_name)}",
                        )

                    dot.add_edge(edge)

                if op_name in failed_keys:
                    node_attrs["color"] = "red"
                node = pydot.Node(
                    escape(op.name),
                    id=escape(op.name),
                    **node_attrs,
                    label=make_label(op),
                    shape="plain",
                )
                node.set_tooltip(make_tooltips(op))
                func_graph.add_node(node)

        dot.add_subgraph(func_graph)

    with open(f"{args.mlir}.dot", "w") as w:
        w.write(str(dot.to_string()))

    if args.isbig:
        cmd = f"""dot -v5 -Gnslimit=2 -Gnslimit1=2 -Gmaxiter=5000 -Tsvg {args.mlir}.dot -o {args.mlir}.svg"""
    else:
        cmd = f"""dot -Tsvg {args.mlir}.dot -o {args.mlir}.svg"""
    os.system(cmd)
    print(cmd)
    print(os.path.abspath(f"{args.mlir}.dot"))
    print(os.path.abspath(f"{args.mlir}.svg"))
