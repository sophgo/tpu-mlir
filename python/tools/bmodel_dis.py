#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from collections import namedtuple
import numpy as np
from utils.bmodel_dis import opdef_1684x, opparam_1684x
from utils.bmodel_dis import bmodel_fbs
import itertools

# Example:
"""
# bmodel file
code = Bmodel2MLIR(file_name)


# bdc binary file
bdc_code = decode_bdc(file_name)

# gdma binary file
gama_code = decode_gdma(file_name)
"""


def read_file(cmd_file):
    cmmand_buf = np.fromfile(cmd_file, dtype=np.uint8)
    return np.unpackbits(cmmand_buf, bitorder="little")


def read_buf(cmd_buf):
    cmmand_buf = np.frombuffer(cmd_buf, dtype=np.uint8)
    return np.unpackbits(cmmand_buf, bitorder="little")


class BmodelReader:
    header_t = np.dtype(
        [
            ("magic", np.uint32),
            ("header_size", np.uint32),
            ("flatbuffers_size", np.uint32),
            ("binary_size", np.uint32),
            ("reserved", np.uint32, 12),
        ]
    )

    class cmd_group_cls:
        def __init__(self, fbs: bmodel_fbs.CmdGroup, cmd_buf):
            self.bdc_num = fbs.BdcNum()
            self.gdma_num = fbs.GdmaNum()
            binary_bdc = (fbs.BinaryBdc().Start(), fbs.BinaryBdc().Size())
            binary_gdma = (fbs.BinaryGdma().Start(), fbs.BinaryGdma().Size())
            self.bdc_cmd = cmd_buf[binary_bdc[0] : sum(binary_bdc)]
            self.gdma_cmd = cmd_buf[binary_gdma[0] : sum(binary_gdma)]

        def __repr__(self):
            return f"bdc_num: {self.bdc_num}\ngdma_num: {self.gdma_num}"

    class data_cls:
        def __init__(self, fbs: bmodel_fbs.CoeffMem, buffer):
            self.address = fbs.Address()
            self.check_code = fbs.CheckCodeAsNumpy()
            binary_data = (fbs.BinaryCoeff().Start(), fbs.BinaryCoeff().Size())
            self.data = buffer[binary_data[0] : sum(binary_data)]

        def __repr__(self):
            check_code = "".join(f"{x:0>2x}" for x in self.check_code)
            return f"data: {self.address}\nshr256: {check_code}"

    class tensor_cls:
        to_DType = {
            0: opparam_1684x.DType.f32,
            1: opparam_1684x.DType.f16,
            2: opparam_1684x.DType.si8,
            3: opparam_1684x.DType.ui8,
            4: opparam_1684x.DType.si16,
            5: opparam_1684x.DType.ui16,
            6: opparam_1684x.DType.si32,
            7: opparam_1684x.DType.ui32,
        }

        def __init__(self, fbs: bmodel_fbs.Tensor, buffer):
            self.name = fbs.Name().decode()
            self.dtype = self.to_DType[fbs.DataType()]
            self.device_addr = fbs.DeviceAddr()
            self.shape = [
                list(fbs.Shape(i).DimAsNumpy()) for i in range(fbs.ShapeLength())
            ]
            self.scale = fbs.Scale()

        def __repr__(self):
            return f"{self.name}: {self.shape}({self.device_addr})"

    def __init__(self, bmodel_file):
        self.head = None
        self.binary_desc = None
        self.binary = None
        with open(bmodel_file, "rb") as file_obj:
            file_obj.seek(0, 0)
            self.head = np.frombuffer(
                file_obj.read(self.header_t.itemsize), dtype=self.header_t
            )
            self.binary_desc = file_obj.read(self.head["flatbuffers_size"][0])
            self.binary = file_obj.read(self.head["binary_size"][0])
        bmodel = bmodel_fbs.Model.GetRootAsModel(self.binary_desc, 0)

        def fbs_adaptor(param, _fields):
            records = {}
            for field, _cls in _fields.items():
                if not hasattr(param, field + "Length"):
                    cmd = getattr(param, field)()
                    if isinstance(_cls, dict):
                        mult = [fbs_adaptor(cmd, _cls)]
                    elif cmd is None:
                        mult = []
                    else:
                        mult = [_cls(cmd, self.binary)]
                else:
                    mult = []
                    for s in range(getattr(param, field + "Length")()):
                        cmd = getattr(param, field)(s)
                        if isinstance(_cls, dict):
                            mult.append(fbs_adaptor(cmd, _cls))
                        else:
                            mult.append(_cls(cmd, self.binary))
                records[field] = mult
            return records

        fields = {  # module
            "Chip": lambda x, _: x.decode(),
            "Version": lambda x, _: x.decode(),
            "Type": lambda x, _: x.decode(),
            "Net": {  # function
                "Name": lambda x, _: x.decode(),
                "Parameter": {  # region
                    "InputTensor": self.tensor_cls,  # signature
                    "OutputTensor": self.tensor_cls,
                    "SubNet": {  # block
                        "CmdGroup": self.cmd_group_cls,
                        "Id": lambda x, _: x,  # label
                        "InputTensor": self.tensor_cls,  # block-arg
                        "OutputTensor": self.tensor_cls,  # terminator
                        "NextSubnetIds": lambda x, _: x,  # successor
                    },
                    "CoeffMem": self.data_cls,
                },
            },
        }

        self.nets = fbs_adaptor(bmodel, fields)


# ==============================================================================
# decode the command buffer to operations.


def _decode_base(cmd_buf, cmd_bits, cmd_set, sys_end):
    code = None
    cur = 0
    l, h = cmd_bits
    while cmd_buf.size > 0:
        cmd_key = opdef_1684x.packbits(cmd_buf[l:h])
        if cmd_key in cmd_set:
            recognize = False
            for op in cmd_set[cmd_key]:
                if op.is_comp(cmd_buf):
                    # check whether this command is recognized by the operation
                    code = op.decode(cmd_buf)
                    yield code
                    # consume this command_code
                    cmd_buf = cmd_buf[op.length :]
                    cur += op.length
                    recognize = True
                    break
            is_sys = isinstance(code, sys_end)
            is_less_1024 = cmd_buf.size < 1025
            is_all_zeros = np.all(cmd_buf == 0)
            if is_sys and is_less_1024 and is_all_zeros:
                break  # all the BDC have been processed
            if not recognize:
                raise ValueError(
                    "Can not decode cmd, with opcode: {}, at {}.".format(cmd_key, cur)
                )
        else:
            raise ValueError(
                "Can not decode cmd, with opcode: {}, at {}.".format(cmd_key, cur)
            )


def _decode_bdc(cmd_buf):
    return _decode_base(
        cmd_buf,
        opdef_1684x.bdc_base.opcode_bits,
        opdef_1684x.bdc_cmd,
        opdef_1684x.sysid_op,
    )


def _decode_gdma(cmd_buf):
    return _decode_base(
        cmd_buf,
        opdef_1684x.dma_base.opcode_bits,
        opdef_1684x.dma_cmd,
        opdef_1684x.sdma_sys,
    )


# BModel to MLIR
# ==============================================================================


def merge_cmd(bdc, dma, _id=None):
    main_cmd, inserted_cmd = dma, bdc
    # remove the system command
    def get_end(cmd):
        if isinstance(main_cmd[-1], (opdef_1684x.sysid_op, opdef_1684x.sdma_sys)):
            return -1
        else:
            return len(main_cmd)

    # remove system instruction
    main_id = [(m.cmd_id, m) for m in main_cmd[: get_end(main_cmd)]]
    inserted_id = [(i.cmd_id_dep, i) for i in inserted_cmd[: get_end(inserted_cmd)]]
    # "sorted" is stable, which keeps the inserted commands
    # after the main instructions.
    cmd = main_id + inserted_id
    cmd_sorted = sorted(cmd, key=lambda x: x[0])
    return [x[1] for x in cmd_sorted]


MERGE_CMD_FUN = merge_cmd


def decode_cmd(cmd, _id=0):
    CMD = namedtuple("cmd", ["bdc", "gdma", "all"])
    bdc_cmd = read_buf(cmd.bdc_cmd)
    gdma_cmd = read_buf(cmd.gdma_cmd)
    bdc = itertools.islice(_decode_bdc(bdc_cmd), cmd.bdc_num)
    gdma = itertools.islice(_decode_gdma(gdma_cmd), cmd.gdma_num)
    bdc = list(bdc)
    gdma = list(gdma)
    return CMD(bdc, gdma, MERGE_CMD_FUN(bdc, gdma, _id))


def tensor2memref(tensor: BmodelReader.tensor_cls):
    return opparam_1684x.MemRef(
        tensor.device_addr, tensor.shape[0], opparam_1684x.DType(tensor.dtype)
    )


IndentSize = 2


class Block:
    def __init__(self, subnet, indent=0):
        assert subnet["Id"] != []
        self.label = subnet["Id"][0]
        self.indent = indent
        self.cmds = [decode_cmd(x, self.label) for x in subnet["CmdGroup"]]
        self.operations = []
        for x in self.cmds:
            self.operations.extend(x.all)
        self.args = subnet["InputTensor"]
        self.terminator = subnet["OutputTensor"]
        self.successor = subnet["NextSubnetIds"]

    def __repr__(self):
        indent = " " * IndentSize * (self.indent + 1)
        ops = "\n".join((indent + f"{x}" for x in self.operations))
        args = [f"%{a.name}: {tensor2memref(a).type_str}" for a in self.args]
        args = ", ".join(args)
        if all((x == -1 for x in self.successor)):
            tem = [tensor2memref(x) for x in self.terminator]
            rets = (
                "return "
                + ", ".join((x.name for x in tem))
                + ": "
                + ", ".join((x.type_str for x in tem))
            )
        else:
            rets = f"Successor {self.successor}"  # TODO
        rets = indent + rets
        return f"^bb{self.label}({args})\n{ops}\n{rets}"


class Region:
    def __init__(self, net_stage, indent=0):
        self.indent = indent
        self.blocks = [Block(x, indent) for x in net_stage["SubNet"]]
        self.signature = (net_stage["InputTensor"], net_stage["OutputTensor"])
        self.data = net_stage["CoeffMem"][0] if net_stage["CoeffMem"] else None

    def __repr__(self):
        blocks = "\n".join((f"{b}" for b in self.blocks))
        return f"{blocks}"


class Function:
    def __init__(self, net, indent=0):
        self.indent = indent
        self.name = net["Name"][0]
        self.regions = [Region(x, indent) for x in net["Parameter"]]
        self.signature = self.regions[0].signature

    def __repr__(self):
        indent = " " * IndentSize * self.indent
        regions = (indent + "\n}, {\n").join((indent + f"{r}" for r in self.regions))

        def fmt_names(x):
            names = (f'"{n.name}"' for n in x)
            return f"[{', '.join(names)}]"

        arg = f"arg_attrs = {fmt_names(self.signature[0])}"
        ret = f"res_attrs = {fmt_names(self.signature[1])}"
        attr = f"{{function_type = {{{arg}, {ret}}}}}"
        operands = ", ".join((str(tensor2memref(x)) for x in self.signature[0]))
        returns = ", ".join((tensor2memref(x).type_str for x in self.signature[1]))
        return (
            indent
            + f"func.func @{self.name}({operands}) -> ({returns}) ({{\n{regions}\n"
            + indent
            + f"}}) {attr}"
        )


class Module:
    def __init__(self, nets):
        self.functions = [Function(x, 1) for x in nets["Net"]]
        self.chip = nets["Chip"][0]
        self.version = nets["Version"][0]
        self.type = nets["Type"][0]

    def __repr__(self):
        funs = "\n".join((f"{x}" for x in self.functions))
        attrs = f"attributes {{chip = {self.chip}, version= {self.version}}}"
        return f"module {attrs} {{\n{funs}\n}}"


def Bmodel2MLIR(bmodel_file):
    bmodel = BmodelReader(bmodel_file)
    return Module(bmodel.nets)


def Bmodel2Reg(bmodel_file):
    bmodel = BmodelReader(bmodel_file)
    for net in bmodel.nets["Net"]:
        for param in net["Parameter"]:
            for subnet in param["SubNet"]:
                _id = subnet["Id"][0]
                for _net in subnet["CmdGroup"]:
                    yield _id, decode_cmd(_net, _id)


def Bmodel2Bin(bmodel_file):
    bmodel = BmodelReader(bmodel_file)
    for net in bmodel.nets["Net"]:
        for param in net["Parameter"]:
            for subnet in param["SubNet"]:
                _id = subnet["Id"][0]
                for _net in subnet["CmdGroup"]:
                    with open(bmodel_file + f".{_id}.tiu.bin", "wb") as f:
                        f.write(_net.bdc_cmd)
                    with open(bmodel_file + f".{_id}.dma.bin", "wb") as f:
                        f.write(_net.gdma_cmd)


def decode_bdc(file_name):
    a = read_file(file_name)
    return [x for x in _decode_bdc(a)]


def decode_gdma(file_name):
    a = read_file(file_name)
    return [x for x in _decode_gdma(a)]


def unified_diff(a, b, fromfile="", tofile="", n=3, format="mlir"):
    r"""
    Compare the operations of two BModel; generate the delta as a unified diff.

    Unified diffs are a compact way of showing line changes and a few
    lines of context.  The number of context lines is set by 'n' which
    defaults to three.
    """
    import difflib

    fmt_op = {
        "raw": lambda op: str(op.attr),
        "mlir": lambda op: str(op),
        "bits": lambda op: "".join((str(x) for x in op.cmd)),
    }
    fmt = fmt_op[format]

    lineterm = "\n"
    started = False
    for group in difflib.SequenceMatcher(None, a, b).get_grouped_opcodes(n):
        if not started:
            started = True
            yield f"--- {fromfile}"
            yield f"+++ {tofile}"

        first, last = group[0], group[-1]
        file1_range = difflib._format_range_unified(first[1], last[2])
        file2_range = difflib._format_range_unified(first[3], last[4])
        yield "@@ -{} +{} @@{}".format(file1_range, file2_range, lineterm)

        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                for line in a[i1:i2]:
                    yield "    " + fmt(line)
                continue
            if tag in {"replace", "delete"}:
                for line in a[i1:i2]:
                    yield "-   " + fmt(line)
            if tag in {"replace", "insert"}:
                for line in b[j1:j2]:
                    yield "+   " + fmt(line)
        yield ""


def __main():
    import argparse

    parser = argparse.ArgumentParser(description="BModel disassembler.")
    parser.add_argument(
        "bmodels",
        type=str,
        nargs="+",
        help="The path of BModels. If one BModel is provided, the assemble code will be printed. Compare the Bmodels if two models provided.",
    )
    parser.add_argument(
        "--fmt",
        dest="format",
        choices=["mlir", "reg", "bits", "bin"],
        default="mlir",
        help="The format of format operations.",
    )
    parser.add_argument(
        "--N",
        dest="N",
        type=int,
        default=3,
        help="The number of context lines.",
    )
    args = parser.parse_args()
    if len(args.bmodels) == 1:
        if args.format == "mlir":
            module = Bmodel2MLIR(args.bmodels[0])
            print(module, flush=True)
        elif args.format == "reg":
            import json

            module = Bmodel2Reg(args.bmodels[0])
            outs = {
                _id: {
                    "tiu": [x.reg for x in ops.bdc],
                    "dma": [x.reg for x in ops.gdma],
                }
                for _id, ops in module
            }
            print(json.dumps(outs, indent=2), flush=True)

        elif args.format == "bin":
            Bmodel2Bin(args.bmodels[0])
        else:
            raise NotImplementedError("Not supports bits mode.")
        exit(0)

    # FIX ME: the code below doe not compatible with the new Module.
    if len(args.bmodels) == 2:
        tpu_cmd_a = Bmodel2MLIR(args.bmodels[0])
        tpu_cmd_b = Bmodel2MLIR(args.bmodels[1])
        is_same = True
        for (idx, cmd_a), (_, cmd_b) in zip(tpu_cmd_a.cmd, tpu_cmd_b.cmd):
            fmt_cmd = [
                "\n" + x
                for x in unified_diff(
                    cmd_a.all,
                    cmd_b.all,
                    args.bmodels[0],
                    args.bmodels[1],
                    n=args.N,
                    format=args.format,
                )
            ]
            fun_name = "graph" + "".join((str(x) for x in idx))
            if fmt_cmd != []:
                is_same = False
                fmt_cmd = "".join(fmt_cmd[:-1]) + "\n"
                print(f"func.func @{fun_name}() {{{fmt_cmd}}}")
        if is_same:
            print(f""""{args.bmodels[0]}" and "{args.bmodels[1]}" are the same!""")
            exit(0)
        else:
            exit(1)
    parser.error("Too many BModels.")


if __name__ == "__main__":
    __main()
