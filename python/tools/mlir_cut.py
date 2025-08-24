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
import json
import os
import pymlir
from mlir_ast.mlir_ast import *
from utils.misc import *
from mlir_ast.mlir_ast import MlirASTParser
from utils.mlir_shell import _os_system
from model_runner import *


# print message with color
def colored(text, color):
    """Apply color to text with ANSI escape codes"""
    colors = {
        'red': '\033[91m',
        'green': '\033[1;32m',
        'yellow': '\033[93m',
        'blue': '\033[1;36m',
        'cyan': '\033[1;36m',
        'orange': '\033[1;33m',
        'reset': '\033[0m',
        'bold': '\033[1m'
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"


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
        assert (len(op.output_types) == 1), "currently only support one operands operation"
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


def make_fake_feed_input(ast: MlirAST, ref: Dict[str, np.ndarray]):
    import numpy as np

    dic = {}
    for op in ast.module.funcs[0].ops:
        if op.op_type.isa("top.Input"):
            if op.name in ref:
                dic[op.name] = ref[op.name]
            else:
                dic[op.name] = np.random.random(op.input_types[0].shape)

    return dic


def cut_mlir(
    ast: MlirAST,
    input_names: List[str] = [],
    output_names: List[str] = [],
    ref_data: Dict[str, np.ndarray] = None,
):
    if ref_data is None:
        ref_data = {}

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

    latest_fn = f"{ast.mlir_file}_latest.mlir"
    with open(latest_fn, "w") as w:
        w.write(ast.dump())

    print(f"new version is rewrite to {fn}, also copy to {latest_fn}")

    dic = make_fake_feed_input(ast, ref_data)
    np.savez("fake_data.npz", **dic)

    mlir_name = os.path.basename(fn)
    if module_state == '"TOP_F32"':
        tgt_name = mlir_name.replace("_origin", "")
        cmd = [f"tpuc-opt {mlir_name} --shape-infer --canonicalize --extra-optimize -o {tgt_name}.mlir"]
        _os_system(cmd)
        mlir_name = tgt_name

    elif module_state == '"TPU_LOWERED"':
        print(" ".join([
            f"tpuc-opt {mlir_name}",
            "--mlir-disable-threading",
            '''--strip-io-quant="quant_input=False quant_output=False"''',
            "--processor-tpu-optimize",
            "--dev-parallel",
            "--weight-reorder",
            '''--subnet-divide="dynamic=False"''',
            "--op-reorder",
            '''--layer-group="opt=2"''',
            "--core-parallel",
            "--address-assign",
            f"-o {ast.mlir_file.replace('tpu.mlir','final.mlir')}",
        ]))

    elif module_state == '"TPU_ADDRESSED"':
        print(rf""" tpuc-opt {mlir_name} \
            --codegen="model_file={fn}.bmodel embed_debug_info=false" \
            -o /dev/null""")

    return mlir_name


def backtrace(ast_parser: MlirASTParser, output_names: List[str], number: int, dir="bt"):
    parser = MlirParserV2.from_astparser(ast_parser)
    side_names = []
    for name in output_names:
        count = 0
        queue = [name]
        while count < number:
            top = queue.pop()
            neighbor = (parser.get_pre_op_by_op_name(top)
                        if dir == "bt" else parser.get_next_op_by_op_name(top))
            queue.extend(neighbor)
            count += len(neighbor)
        side_names.extend(queue)
    return side_names


def get_input_output_names(mlir_file: str):
    if mlir_file.endswith(".mlir"):
        assert not mlir_file.endswith("_cut.mlir") or mlir_file.endswith("final.mlir")
        cmd_0 = f"tpuc-opt {new_mlir} --init -o canonicalized.mlir"
        _os_system([cmd_0])
        parser = MlirASTParser("canonicalized.mlir")
        parser.parse()
        output_names = parser.ast.module.context.output_names
        input_names = []
        for op in parser.ast.module.funcs[-1].ops:
            if op.op_type.isa("top.Input"):
                input_names.append(op.name)
        return input_names, output_names
    elif mlir_file.endswith(".bmodel"):
        chip = get_chip_from_model(mlir_file)
        pyruntime = "pyruntime_" + ("tpuv7" if chip in ["BM1690", "SG2262"] else "bm")
        with FileLock("/tmp/cmodel_so.lock"):
            link_cmodel_so(chip)
            link_custom_so(chip)
            pyrtlib = importlib.import_module(pyruntime)
            model = pyrtlib.Model(mlir_file, 0, "")
        net = model.Net(model.networks[0])
        input_names = [tensor.name for tensor in net.inputs]
        output_names = [tensor.name for tensor in net.outputs]
        return input_names, output_names
    else:
        raise ValueError(f"Unsupported file type: {mlir_file}")


def pack_bmodel_ref_tensors(tensors, ref_tensors):
    cmodel_data, onboard_data = {}, {}
    for tensor in tensors:
        if tensor.name not in ref_tensors:
            print(f"{colored('[SKIP]', 'yellow')} {tensor.name} not found in ref_tensors")
            continue
        ref_f32 = ref_tensors[tensor.name]
        if list(ref_f32.shape) != []:
            assert np.prod(tensor.data.shape) == np.prod(ref_f32.shape)
        ref_quant = lowering(ref_f32,
                             pdtype=tensor.dtype,
                             pshape=tensor.data.shape,
                             pzero_point=tensor.qzero_point,
                             pscale=tensor.qscale)
        cmodel_data[tensor.name] = ref_f32
        onboard_data[tensor.name] = ref_quant
    return cmodel_data, onboard_data


def generate_config_file(input_names=None, output_names=None, config_file="config.json"):
    if input_names is None:
        input_names = []
    if output_names is None:
        output_names = []
    config = {
        "new_input_names": input_names,
        "new_output_names": output_names,
        "assign_new_io_addrs": True,
        "remove_unused_local_ops": True,
        "put_storeop_near_producer": True
    }
    config_path = os.path.join(os.getcwd(), config_file)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"new config file is generated at: {colored(config_path, 'green')}")


def cut_final_mlir(final_mlir: str, cfg_file: str, ref_data: str):

    # 0. dump config file.
    with open(cfg_file, 'r') as f:
        cfg = json.load(f)
    print(f"cut final mlir with configs:\n {colored(json.dumps(cfg, indent=2), 'green')}")
    assert cfg["new_output_names"] != [], "at least one output is required"

    # 1. cut final mlir
    new_final_mlir = os.path.splitext(final_mlir)[0] + "_cut.mlir"
    cmd1 = [
        "tpuc-opt", final_mlir,
        f"--cut-final-mlir=\"config_file={cfg_file}\"",
        "-o", new_final_mlir
    ]
    try:
        _os_system(cmd1)
        print(f"[success] {' '.join(cmd1)}")
    except Exception as e:
        print(f"[failed] {' '.join(cmd1)} ;with error: {e}")
        exit(1)

    # 2. run codegen
    bmodel_file = new_final_mlir.replace(".mlir", ".bmodel")
    cmd2 = [
        "tpuc-opt", new_final_mlir,
        f"--codegen=\"model_file={bmodel_file} embed_debug_info=true\"",
        "-o", "/dev/null"
    ]
    try:
        _os_system(cmd2)
    except Exception as e:
        print(f"{colored('[failed]', 'red')} {' '.join(cmd2)} ;with error: {e}")
        exit(1)

    save_dir = "dummy_bmodel"
    os.makedirs(save_dir, exist_ok=True)
    print(f"{colored(f'results saved to ./{save_dir}', 'green')}")
    shutil.copy2(new_final_mlir, os.path.join(save_dir, "final.mlir"))
    shutil.copy2(bmodel_file, os.path.join(save_dir, f"compilation.bmodel"))
    if os.path.exists(bmodel_file + ".json"):
        shutil.copy2(bmodel_file + ".json", os.path.join(save_dir, "tensor_location.json"))

    # 3. save ref data and collect files.
    if ref_data is not None:
        ref_tensors = np.load(ref_data, allow_pickle=True)
        chip = get_chip_from_model(bmodel_file)
        pyruntime = "pyruntime_" + (chip == "BM1690" or chip == "SG2262" and "tpuv7" or "bm")

        with FileLock("/tmp/cmodel_so.lock"):
            link_cmodel_so(chip)
            link_custom_so(chip)
            pyrtlib = importlib.import_module(pyruntime)
            model = pyrtlib.Model(bmodel_file, 0, "")
        net = model.Net(model.networks[0])

        input_data_cmodel, input_data_card = pack_bmodel_ref_tensors(net.inputs, ref_tensors)
        output_data_cmodel, output_data_card = pack_bmodel_ref_tensors(net.outputs, ref_tensors)

        np.savez(os.path.join(save_dir, "input_ref_data_cmodel.npz"), **input_data_cmodel)
        np.savez(os.path.join(save_dir, "input_ref_data_card.npz"), **input_data_card)
        np.savez(os.path.join(save_dir, "output_ref_data_cmodel.npz"), **output_data_cmodel)
        np.savez(os.path.join(save_dir, "output_ref_data_card.npz"), **output_data_card)
        with open(os.path.join(save_dir, "input_ref_data.dat"), "wb") as f:
            for i in input_data_card.values():
                i.tofile(f)
        with open(os.path.join(save_dir, "output_ref_data.dat"), "wb") as f:
            for i in output_data_card.values():
                i.tofile(f)

    return bmodel_file


if __name__ == "__main__":
    print("TPU-MLIR {}".format(pymlir.__version__))
    parser = argparse.ArgumentParser(
        description="cut .MLIR file with certain input/output names.",
        epilog=f"""
{colored('Examples:', 'cyan')}
{colored('# case.1: cut final.mlir in order to debug intermediate results of bmodel.', 'orange')}
  {colored('# 1.a:', 'cyan')} cut final.mlir with both input_names and output_names. \\
  mlir_cut.py will first generate a config file, then run cut_final_mlir pass with that config file. \\
  results (bmodel, final.mlir, ref_data, etc.) will defaultly be saved to {colored('`dummy_bmodel`', 'green')} directory.
     PS: output_names must be given.
  {colored('($) mlir_cut.py --mlir xxx_final.mlir --input_names input1,input2 --output_names output1,output2 (--ref_data xxx_tpu_outputs.npz)', 'cyan')}

  {colored('# 1.b:', 'cyan')} cut final.mlir with a config file. user can decide whether assign new addresses for new input/output, etc.
     PS: one can run {colored('case.1.a', 'cyan')} first, then modify the auto-generated config file, and run {colored('case.1.b', 'cyan')} agian with the same config file.
  {colored('($) mlir_cut.py --mlir xxx_final.mlir  --config_file mlir_cut_cfg.json (--ref_data xxx_tpu_outputs.npz)', 'cyan')}

{colored('# case.2: cut top.mlir/tpu.mlir in order to debug intermediate results of top.mlir/tpu.mlir.', 'orange')}
  {colored('# 2.a:', 'cyan')} cut top.mlir/tpu.mlir with both input_names and output_names.
  {colored('($) mlir_cut.py --mlir xxx_top/tpu.mlir (--mode io) --input_names input1,input2 --output_names output1,output2 (--ref_data xxx_top_outputs.npz)', 'cyan')}

  {colored('# 2.b:', 'cyan')} cut top.mlir/tpu.mlir with input_names and forward trace layer-number.
  {colored('($) mlir_cut.py --mlir xxx_top/tpu.mlir --mode ft --input_names input1,input2 --num 3 (--ref_data xxx_top_outputs.npz)', 'cyan')}

  {colored('# 2.c:', 'cyan')} cut top.mlir/tpu.mlir with output_names and backtrace layer-number.
  {colored('($) mlir_cut.py --mlir xxx_top/tpu.mlir --mode bt --output_names output1,output2 --num 3 (--ref_data xxx_top_outputs.npz)', 'cyan')}
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # yapf: disable
    parser.add_argument("--mlir", required=True, help="model name")
    parser.add_argument("--mode", choices=['io','bt','ft'], help="""
                         - io mean assign input_names and output_names;
                         - bt (backtrace) means cut `num` operations back for each output_name;
                         - ft (forward trace) means cut `num` operation back for each input_name
                        """, default='io')
    parser.add_argument("--num", type=int, help="number of operations to be cut, only used for bt and ft mode")
    parser.add_argument("--input_names", type=str2list, default=list(),
                        help="If set, will find names in model and set as real outputs")
    parser.add_argument("--output_names", type=str2list, default=list(),
                        help="If set, will output names in model and set as real outputs")
    parser.add_argument("--ref_data", type=str, default= None,
                        help="Reference data of the model after cutting. Dictionary stored as npz file.")
    parser.add_argument("--config_file", type=str, default=None,
                        help="Json file with configs of cut_final_mlir pass")
    parser.add_argument("--do_verify", action="store_true", default=False,
                        help="Run model_runner.py and check the new outputs.")
    args, unknown_args = parser.parse_known_args()


    # NOTE: walk-around a bug with parsing multi-core mlir.
    if not args.mlir.endswith("_final.mlir"):
        parser = MlirASTParser(args.mlir)
        parser.parse()

    # ===---------- cut the model ----------===

    if args.mlir.endswith("_final.mlir") or parser.ast.module.attrs["module.state"].strip('"') == 'TPU_ADDRESSED':
        # ===---------- cut final.mlir ----------===
        # 0. create config file if not exist
        cfg_file = args.config_file
        assert(args.mode == 'io'), "final.mlir only support use `io` mode"
        if cfg_file is None:

            cfg_file = "mlir_cut_cfg.json"  # default file name
            generate_config_file(args.input_names, args.output_names, cfg_file)
        # 1. run final_cut pass.
        new_mlir = cut_final_mlir(args.mlir, cfg_file, args.ref_data)
    else:
        # ===---------- cut top/tpu.mlir ----------===
        if args.mode == 'bt':
            assert len(args.input_names) == 0 and len(args.output_names) != 0
            args.input_names = backtrace(parser, args.output_names, args.num, dir=args.mode)
        elif args.mode == 'ft':
            assert len(args.output_names) == 0 and len(args.input_names) != 0
            args.output_names = backtrace(parser, args.input_names, args.num, dir=args.mode)

        ref_data =None
        if args.ref_data is not None:
            ref_data = np.load(args.ref_data, allow_pickle=True)

        new_mlir = cut_mlir(parser.ast, args.input_names, args.output_names, ref_data)


    # ===---------- verify new model ----------===

    assert new_mlir is not None
    # check input/output names.
    expect_input_names = args.input_names
    expect_output_names = args.output_names
    if args.config_file is not None:
        with open(args.config_file, 'r') as f:
            cfg = json.load(f)
        expect_input_names = cfg["new_input_names"]
        expect_output_names = cfg["new_output_names"]
    input_names, output_names = get_input_output_names(new_mlir)
    if args.mode == 'ft':
        print(colored(f"new input_names: {input_names}", 'green'))
        print(colored(f"expect input_names: {expect_input_names}", 'green'))
        assert not (set(expect_input_names) -set(input_names)), \
            "all names in --input_names should be added to new mlir"
    else:
        print(colored(f"new output_names: {output_names}", 'green'))
        print(colored(f"expect output_names: {expect_output_names}", 'green'))
        # remove "_f32" suffix before compare.
        output_names = [name.replace('_f32', '') for name in output_names]
        expect_names = [name.replace('_f32', '') for name in expect_output_names]
        assert set(output_names) == set(expect_names), \
            "--output_names should be exactly the same with new mlir"

    # call model_runner and compare the outputs.
    if args.do_verify:
        assert args.ref_data is not None, "ref_data is required for verification"
        infer_cmd = f"model_runner.py --input {args.ref_data} --model {new_mlir} --output _outputs.npz"
        compare_cmd = f"npz_tool.py compare {args.ref_data} _outputs.npz -v"
        _os_system([infer_cmd])
        _os_system([compare_cmd])
