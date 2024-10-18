#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# graphvis api doc: https://graphviz.org/doc/info/lang.html
# ==============================================================================
from collections import defaultdict
import os
import numpy as np
from pathlib import Path
import argparse
import pydot
import json
from mlir.ir import *
from tqdm import tqdm
import mlir.ir
from mlir.dialects.func import FuncOp
from typing import List
import re
from utils.log_setting import setup_logger
import random

random.seed(10)

logger = setup_logger("mlir2graph")

escape_pattern = re.compile('[:"]')

# picked from https://graphviz.org/doc/info/colors.html
INPUT_COLOR = "cadetblue1"
SUBNET_COLOR = "antiquewhite"
GROUP_COLOR = "gray95"
FAILED_COLOR = "red"

MAP_COLOR = [
    "aliceblue",
    "antiquewhite",
    "antiquewhite1",
    "antiquewhite2",
    "antiquewhite3",
    "antiquewhite4",
    "aqua",
    "aquamarine",
    "aquamarine1",
    "aquamarine2",
    "aquamarine3",
    "aquamarine4",
    "azure",
    "azure1",
    "azure2",
    "azure3",
    "azure4",
    "beige",
    "bisque",
    "bisque1",
    "bisque2",
    "bisque3",
    "bisque4",
    "black",
    "blanchedalmond",
    "blue",
    "blue1",
    "blue2",
    "blue3",
    "blue4",
    "blueviolet",
    "brown",
    "brown1",
    "brown2",
    "brown3",
    "brown4",
    "burlywood",
    "burlywood1",
    "burlywood2",
    "burlywood3",
    "burlywood4",
    "chartreuse",
    "chartreuse1",
    "chartreuse2",
    "chartreuse3",
    "chartreuse4",
    "chocolate",
    "chocolate1",
    "chocolate2",
    "chocolate3",
    "chocolate4",
    "coral",
    "coral1",
    "coral2",
    "coral3",
    "coral4",
    "cornflowerblue",
    "cornsilk",
    "cornsilk1",
    "cornsilk2",
    "cornsilk3",
    "cornsilk4",
    "crimson",
    "cyan",
    "cyan1",
    "cyan2",
    "cyan3",
    "cyan4",
    "darkblue",
    "darkcyan",
    "darkgoldenrod",
    "darkgoldenrod1",
    "darkgoldenrod2",
    "darkgoldenrod3",
    "darkgoldenrod4",
    "darkgray",
    "darkgreen",
    "darkgrey",
    "darkkhaki",
    "darkmagenta",
    "darkolivegreen",
    "darkolivegreen1",
    "darkolivegreen2",
    "darkolivegreen3",
    "darkolivegreen4",
    "darkorange",
    "darkorange1",
    "darkorange2",
    "darkorange3",
    "darkorange4",
    "darkorchid",
    "darkorchid1",
    "darkorchid2",
    "darkorchid3",
    "darkorchid4",
    "darkred",
    "darksalmon",
    "darkseagreen",
    "darkseagreen1",
    "darkseagreen2",
    "darkseagreen3",
    "darkseagreen4",
    "darkslateblue",
    "darkslategray",
    "darkslategray1",
    "darkslategray2",
    "darkslategray3",
    "darkslategray4",
    "darkslategrey",
    "darkturquoise",
]

random.shuffle(MAP_COLOR)


def name_escape(arg):
    """
    pydot 1.4.2 will do escape automatically but >3.0.0 will not.
    Force to escape to ensure compatibility.
    """
    return f'"{arg}"'


def escape(name: str):
    return escape_pattern.sub("_", name).strip('"()_')


label_template = """
<<table border="0" cellborder="1" cellspacing="0">
    <tr><td align="center">{res_ids} = {op_type}({opd_ids})</td></tr>
    <tr><td align="center"> -&gt; {shape}</td></tr>
    <tr><td align="center">{name}</td></tr>
</table>>
""".strip()


def parse_attribute(attr: Attribute) -> dict:
    if isinstance(attr, OpAttributeMap):
        dic = {}
        [dic.update(parse_attribute(i)) for i in list(attr)]
        return dic
    if isinstance(attr, NamedAttribute):
        return {attr.name: parse_attribute(attr.attr)}
    if isinstance(attr, (ArrayAttr)):
        return [parse_attribute(i) for i in attr]
    elif isinstance(attr, (StringAttr, BoolAttr, IntegerAttr, FloatAttr)):
        return attr.value
    else:
        return str(attr)


def make_group_label(op: OpView, **kwargs):
    res_ids = ", ".join([i.get_name() for i in op.results])
    opd_ids = ", ".join([i.get_name() for i in op.operands])
    op_type = get_opname(op)

    html = label_template.format(
        res_ids=res_ids,
        opd_ids=opd_ids,
        op_type=op_type,
        shape="",
        name=get_op_loc(op),
    )
    logger.debug(html)
    return html


def make_label(op: OpView, **kwargs):
    res_ids = ", ".join([i.get_name() for i in op.results])
    opd_ids = ", ".join([i.get_name() for i in op.operands])
    op_type = get_opname(op)
    shape = ", ".join([str(opr.type) for opr in op.results])

    shape = shape.replace("tensor", "").replace("<", "").replace(">", "")
    name = get_op_loc(op)
    if kwargs.setdefault("failed", False):
        name = f"{name} (failed)"
    if kwargs.get("suffix"):
        name = f"{name} ({kwargs['suffix']})"

    html = label_template.format(
        res_ids=res_ids,
        opd_ids=opd_ids,
        op_type=op_type,
        shape=shape,
        name=name,
    )
    logger.debug(html)
    return html


def make_tooltips(op: Operation):
    attr_str = json.dumps(parse_attribute(op.attributes), ensure_ascii=False, indent=2)
    logger.debug(attr_str)
    res = f"<{attr_str}>"
    return res


def iter_operations(op):
    if isinstance(op, Module):
        for oop in op.body.operations:
            yield from iter_operations(oop)
    elif isinstance(op, OpView):
        if op.operation.name == "builtin.module":
            for region in op.regions:
                yield from iter_operations(region)
        elif isinstance(op, FuncOp):
            for region in op.regions:
                yield from iter_operations(region)
        else:
            raise NotImplementedError(op)
    elif isinstance(op, Region):
        for block in op.blocks:
            yield from iter_operations(block)
    elif isinstance(op, Block):
        for operation in op.operations:
            if isinstance(operation, FuncOp):
                yield from iter_operations(operation)
            else:
                yield operation.operation
    else:
        raise NotImplementedError(op)


def get_opname(op):
    if isinstance(op, OpView):
        return op.operation.name
    elif isinstance(op, Operation):
        return op.name
    elif isinstance(op, Value):
        return get_opname(op.owner)
    raise NotImplementedError(op)


match_fused_loc = re.compile(r"""fused\["([^"]*)"(, "([^"]*)")+\]""")


# 'fused["252_LayerNormalization", "263_LayerNormalization"]'
def get_op_loc(op):
    if isinstance(op, (OpView, Operation)):
        res = str(op.location).replace("loc(", "").strip(")")

        if "fused" in res:
            res = match_fused_loc.search(res).group(1)
        return escape(res)
    elif isinstance(op, Value):
        return get_op_loc(op.owner)
    raise NotImplementedError()


def is_opname(op, *names: str):
    return any(get_opname(op) == name for name in names)


class EscapeNode(pydot.Node):

    def __init__(self, name="", obj_dict=None, **attrs):
        super().__init__(name_escape(name), obj_dict, **attrs)


class EscapeEdge(pydot.Edge):

    def __init__(self, src="", dst="", obj_dict=None, **attrs):
        super().__init__(name_escape(src), name_escape(dst), obj_dict, **attrs)


def create_node(op, op_loc, node_attrs: dict):
    logger.debug("node: ", op_loc)
    node = EscapeNode(
        op_loc,
        id=op_loc,
        **node_attrs,
        label=make_label(op, **node_attrs),
        shape="plain",
    )
    node.set_tooltip(make_tooltips(op))
    return node


def create_edge(pre_op_loc, op_loc, label, ltail=None, href=None):
    edge_attr = {
        "ltail": ltail,
        "href": href,
    }

    logger.debug("edge: ", pre_op_loc, op_loc)
    edge_attr = {k: v for k, v in edge_attr.items() if v is not None}
    edge = EscapeEdge(pre_op_loc, op_loc, xlabel=label, **edge_attr)
    return edge


weight_op = set()

skip_op = set()

group_names = defaultdict(list)

color_map = {}
graph = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlir", required=True, help="model name")
    parser.add_argument("--output",
                        help="output dot file, if not assigned, use {mlir file}.{suffix}")
    parser.add_argument(
        "--skip",
        default="",
        help="output dot file, if not assigned, use {mlir file}.{suffix}",
    )

    parser.add_argument(
        "--bmodel_checker_data",
        type=str,
        default=None,
        help="for final.mlir, a bmodel_checker_data will render red for failed operation",
    )
    parser.add_argument(
        "--failed_keys",
        type=str,
        default=None,
        help="split by quote",
    )
    parser.add_argument(
        "--isbig",
        type=bool,
        default=False,
        help=
        "for mlir file with large number of operations, use this option to spped up graph generation.",
    )
    parser.add_argument(
        "--export_group",
        action="store_true",
        help="export group keys in final.mlir for tpu.mlir colorful render",
    )
    parser.add_argument(
        "--colorful",
        default=None,
        help="render op color by groups",
    )

    args = parser.parse_args()
    skiped_op = set(args.skip.split(","))
    if args.colorful:
        group_keys = json.loads(Path(args.colorful).read_text())
        index = 0

        for k, v in group_keys.items():
            ret = [k, *v]
            for kk in ret:
                color_map[kk] = index
            index += 1

    # dot = graphviz.Digraph(comment="The Round Table", node_attr={"shape": "box"})
    dot = pydot.Dot("my_graph", graph_type="digraph", compound=True, splines="polyline")

    failed_keys = set()
    if args.bmodel_checker_data is not None:
        report = np.load(args.bmodel_checker_data, allow_pickle=True)
        failed_keys.update({k.split("_asm_")[0] for k in list(report.files) if "actual" in k})

    if args.failed_keys is not None:
        failed_keys.update({i.strip() for i in args.failed_keys.split(",")})

    with open(args.mlir, "r") as r:
        context = r.read()

    ctx = mlir.ir.Context()
    ctx.allow_unregistered_dialects = True
    module = mlir.ir.Module.parse(context, ctx)

    func_inputs = set()
    func_inputs_names = defaultdict(list)
    func_output_names = defaultdict(list)

    multiple_subnet = False

    def iter_block(block: Block):
        for operation in block.operations:
            yield operation

    def iter_function_op(func: FuncOp) -> List[OpView]:
        for region in func.regions:  # type: Region
            for block in region.blocks:
                yield from iter_block(block)

    def draw_group_op(group):
        op_loc = get_op_loc(group)
        group_graph = pydot.Subgraph(
            name_escape("cluster_" + op_loc),
            id="cluster_" + op_loc,
            label=make_group_label(group),
            shape="plain",
            labeljust="l",
            bgcolor=GROUP_COLOR,
        )

        for op in group.regions[0].blocks[0].operations:
            if is_opname(op, "tpu.Yield", "tpu.Store"):
                continue

            oop_loc = get_op_loc(op)

            group_names[op_loc].append(oop_loc)

            node_attrs = {}
            # node_attrs["shape"] = "box"
            if op_loc in color_map:
                node_attrs["fillcolor"] = MAP_COLOR[color_map[op_loc]  % len(MAP_COLOR)]
                node_attrs["style"] = "filled"
                node_attrs["suffix"] = f"group({color_map[op_loc]})"

            if oop_loc in failed_keys:
                node_attrs["failed"] = True
                node_attrs["color"] = FAILED_COLOR

            if is_opname(op, *skiped_op):
                skip_op.add(op_loc)
                continue

            node = create_node(op, oop_loc, node_attrs)
            node.set_tooltip(make_tooltips(op))
            group_graph.add_node(node)

            for iop in op.operands:
                if "arg" in iop.get_name():
                    continue
                pre_op = iop.owner

                if is_opname(pre_op, "top.None", "tpu.Yield"):
                    continue

                pre_op_loc = get_op_loc(pre_op)
                if pre_op_loc == oop_loc:
                    continue

                if pre_op_loc in skip_op:
                    continue

                if is_opname(pre_op, "tpu.Group"):
                    edge = create_edge(
                        pre_op_loc,
                        oop_loc,
                        label=iop.get_name(),
                        ltail="cluster_" + pre_op_loc,
                        href=f"#cluster_{pre_op_loc}",
                    )
                else:
                    edge = create_edge(
                        pre_op_loc,
                        oop_loc,
                        label=iop.get_name(),
                        href=f"#{pre_op_loc}",
                    )

                dot.add_edge(edge)

        return group_graph

    def draw_func_op(func: FuncOp):
        func_name = func.name.value
        func_graph = pydot.Subgraph(
            f"cluster_{func_name}",
            labeljust="l",
            label=func_name,
            bgcolor=SUBNET_COLOR,
        )
        in_main_func = func_name == "main"
        if in_main_func:
            for arg in func.arguments:
                arg_name = f"main_{escape(arg.get_name())}"
                node = EscapeNode(
                    arg_name,
                    id=arg_name,
                    shape="plain",
                )
        logger.info(f"parse func {func_name}")
        for op in tqdm(list(iter_function_op(func))):
            if is_opname(op, "top.None"):
                continue
            op_loc = get_op_loc(op)
            if is_opname(op, "func.call"):
                subfunc_name = op.callee.value
                # get input name from pre subfunc outputs
                for opd in op.operands:
                    if is_opname(opd, "top.Input"):
                        func_inputs_names[subfunc_name].append(get_op_loc(opd))
                    elif is_opname(opd, "func.call"):
                        opd_name = opd.owner.opview.callee.value
                        opd_ref = opd.get_name().split("#")
                        if len(opd_ref) == 1:
                            index = 0
                        else:
                            index = int(opd_ref[1])
                        func_inputs_names[subfunc_name].append([opd_name, index])

            elif is_opname(op, "tpu.Group", "tpu.CoreParallel", "tpu.Parallel"):
                group_graph = draw_group_op(op)
                func_graph.add_subgraph(group_graph)
            else:
                node_attrs = {}

                # node_attrs["shape"] = "box"
                node_attrs["failed"] = op_loc in failed_keys

                if op_loc in color_map:
                    node_attrs["fillcolor"] = MAP_COLOR[color_map[op_loc] % len(MAP_COLOR)]
                    node_attrs["style"] = "filled"
                    node_attrs["suffix"] = f"group({color_map[op_loc]})"

                if op_loc in failed_keys:
                    node_attrs["color"] = FAILED_COLOR
                if is_opname(op, "top.Input", "top.Weight"):
                    node_attrs["fillcolor"] = INPUT_COLOR
                    node_attrs["style"] = "filled"

                if is_opname(op, *skiped_op):
                    skip_op.add(op_loc)
                    continue

                if not is_opname(op, "func.return"):
                    node = create_node(op, op_loc, node_attrs)
                    func_graph.add_node(node)

                    for opd_index, iopd in enumerate(op.operands):
                        if "arg" in iopd.get_name():
                            if in_main_func:
                                pre_op_loc = f"main_{iopd.get_name()}"
                            else:
                                pre_op_loc = func_inputs_names[func_name][opd_index]
                                if isinstance(pre_op_loc, list):
                                    pre_subnet_name, pre_subnet_opd_index = pre_op_loc
                                    pre_op_loc = func_output_names[pre_subnet_name][
                                        pre_subnet_opd_index]
                        elif is_opname(iopd, "top.None", "tpu.Yield"):
                            continue
                        else:
                            pre_op_loc = get_op_loc(iopd)
                            if pre_op_loc == op_loc:
                                continue

                        if pre_op_loc in skip_op:
                            continue
                        graph.append([pre_op_loc, op_loc])
                        edge = create_edge(
                            pre_op_loc,
                            op_loc,
                            label=iopd.get_name(),
                            href=f"#{pre_op_loc}",
                        )

                        dot.add_edge(edge)

                if multiple_subnet and is_opname(op, "func.return"):
                    if in_main_func:
                        pass
                    else:
                        for opd in op.operands:
                            func_output_names[func_name].append(get_op_loc(opd))
        dot.add_subgraph(func_graph)

    for func in module.body.operations:

        if isinstance(func, FuncOp):
            draw_func_op(func)
        elif getattr(func.operation, "name") == "builtin.module":
            multiple_subnet = True
            for subfunc in list(func.regions[0].blocks[0].operations):
                draw_func_op(subfunc)
        else:
            raise NotImplementedError(func)

    with open(f"{args.mlir}.dot", "w") as w:
        w.write(str(dot.to_string()))

    if args.isbig:
        cmd = f"""dot -v5 -Gnslimit=2 -Gnslimit1=2 -Gmaxiter=5000 -Tsvg {args.mlir}.dot -o {args.mlir}.svg"""
    else:
        cmd = f"""dot -Tsvg {args.mlir}.dot -o {args.mlir}.svg"""
    os.system(cmd)
    print(cmd)

    with open(f"{args.mlir}.graph", "w") as w:
        w.write(json.dumps(graph, indent=2))

    print(os.path.abspath(f"{args.mlir}.dot"))
    print(os.path.abspath(f"{args.mlir}.svg"))
    print(os.path.abspath(f"{args.mlir}.graph"))

    if len(group_names) > 0:
        with open("final_group_keys.json", "w") as w:
            w.write(json.dumps(group_names))
