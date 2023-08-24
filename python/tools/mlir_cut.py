#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import argparse
import pymlir
from mlir_ast.mlir_ast import *
from utils.misc import *
from mlir_ast.mlir_ast import MlirASTParser


def cut_mlir_output(ast: MlirAST, output_names: List[str]):
    ops = [ast.get_op_by_op_name(name) for name in output_names]
    opd_ids = list(chain(*[i.opd_ids for i in ops]))
    output_types = list(chain(*[i.output_types for i in ops]))
    loc_label = ast.return_op.loc_label

    module_state = ast.module.attrs["module.state"]
    can_new_cast = module_state != "TPU_ADDRESSED"
    replace_dict = {}

    # create cast if needed
    # %658 = "tpu.Cast"(%657) {with_scale = true} : (input_types) -> output_types loc(#loc658)
    for i, (opd_id, output_type) in enumerate(zip(opd_ids, output_types)):
        if output_type.dtype != "f32":
            assert can_new_cast

            new_loc = Location(ast.malloc_loc_id(), f"{ops[i].name}_f32")
            ast.add_location(new_loc)

            new_loc_label = new_loc.to_label()
            new_cast_op = Operation(
                [ast.malloc_opd_id()],
                OperationType("tpu.Cast", [opd_id]),
                [output_type],
                [output_type.create_a_f32()],
                new_loc_label,
            )
            ast.module.funcs[-1].ops.insert(-1, new_cast_op)
            replace_dict[i] = new_cast_op

    for i, cast in replace_dict.items():
        opd_ids[i] = cast.opd_ids[0]
        output_types[i] = cast.output_types[0]

    new_return = Return(opd_ids, output_types, loc_label)
    ast.set_return(new_return)
    ast.module.funcs[-1].ops[-1] = new_return
    ast.module.funcs[-1].output_types = output_types

    if len(ast.module.funcs) > 1:
        # final.mlir, hack replace
        ast.module.funcs[0].output_types = output_types
        # return op
        return_op = ast.module.funcs[0].ops[-1]
        return_op.output_types = output_types
        prefix = return_op.op_type.opds[0].split("#")[0]
        new_opds = [f"{prefix}#{i}" for i in range(len(output_types))]
        return_op.op_type.opds = new_opds

        # call op
        ast.module.funcs[0].ops[-2].output_types = output_types
        ast.module.funcs[0].ops[-2].update_opd()


def cut_mlir_input(ast: MlirAST, input_names: List[str]):
    ops = [ast.get_op_by_op_name(name) for name in input_names]

    opd_ids = list(chain(*[i.opd_ids for i in ops]))
    for op in ops:
        assert (
            len(op.output_types) == 1
        ), "currently only support one operands operation"
    input_types = list(chain(*[i.output_types for i in ops]))

    module_state = ast.module.attrs["module.state"]
    assert module_state != "TPU_ADDRESSED", "can not cut final.mlir input_names"

    func = ast.module.funcs[-1]
    # func.input_types = input_types
    # func.input_names = [f"%arg{i}" for i in range(len(input_types))]
    # func.input_locs = [func.input_locs[0]] * len(input_types)
    indexs = [func.ops.index(i) for i in ops]

    for i, index in enumerate(indexs):
        with ast:
            func.ops[index] = Operation(
                [opd_ids[i]],
                OperationType("top.Input", [f"%arg{i}"]),
                [input_types[i]],
                [input_types[i]],
                ops[i].loc_label,
            )


def make_fake_feed_input(ast: MlirAST):
    import numpy as np

    dic = {}
    for op in ast.module.funcs[0].ops:
        if op.op_type.isa("top.Input"):
            dic[op.name] = np.random.random(op.input_types[0].shape)
    return dic


def cut_mlir(ast: MlirAST, input_names: List[str] = [], output_names: List[str] = []):
    i = 0
    fn = f"{ast.mlir_file}_v{i}.mlir"
    while os.path.exists(fn):
        i += 1
        fn = f"{ast.mlir_file}_v{i}.mlir"
    print(ast.mlir_file)

    module_state = ast.module.attrs["module.state"]
    if len(input_names) > 0:
        cut_mlir_input(ast, input_names)

    if len(output_names) > 0:
        cut_mlir_output(ast, output_names)

    ast.module.funcs[-1].erase_unused_op()
    ast.module.funcs[-1].align_input()

    with open(fn, "w") as w:
        w.write(ast.dump())

    print(f"new version is rewrite, also copy to {fn}")

    # if len(input_names) > 0:
    dic = make_fake_feed_input(ast)
    np.savez("fake_data.npz", **dic)

    wd = os.path.dirname(fn)
    basename = os.path.basename(fn)

    if module_state == "TOP_F32":
        tgt_name = basename.replace("_origin", "")
        print(
            f"tpuc-opt {basename} --shape-infer --canonicalize --extra-optimize -o {tgt_name}.mlir"
        )
    elif module_state == "TPU_LOWERED":
        print(
            " ".join(
                [
                    f"tpuc-opt {basename}",
                    "--mlir-disable-threading",
                    '''--strip-io-quant="quant_input=False quant_output=False"''',
                    "--chip-tpu-optimize",
                    "--distribute='num_device=1'",
                    "--weight-reorder",
                    '''--subnet-divide="dynamic=False"''',
                    "--op-reorder",
                    '''--layer-group="opt=2"''',
                    "--parallel='num_core=1'",
                    "--address-assign",
                    f"-o {ast.mlir_file.replace('tpu.mlir','final.mlir')}",
                ]
            )
        )

    elif module_state == "TPU_ADDRESSED":
        print(
            rf""" tpuc-opt {basename} \
            --codegen="model_file={fn}.bmodel embed_debug_info=false" \
            -o /dev/null"""
        )


if __name__ == "__main__":
    print("SOPHGO Toolchain {}".format(pymlir.module().version))
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--mlir", required=True, help="model name")
    parser.add_argument("--input_names", type=str2list, default=list(),
                        help="if set, will find names in model and set as real outputs")
    parser.add_argument("--output_names", type=str2list, default=list(),
                        help="if set, will find names in model and set as real outputs")

    args, unknown_args = parser.parse_known_args()


    parser = MlirASTParser(args.mlir)
    parser.parse()
    cut_mlir(parser.ast, args.input_names, args.output_names)
    exit(0)
