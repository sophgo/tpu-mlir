# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from collections import namedtuple
import itertools, functools
import numpy as np


try:
    from . import op_support
    from . import bmodel_fbs
except:
    import op_support
    import bmodel_fbs


class Decoder:
    CMD = namedtuple("cmd", ["tiu", "dma", "all"])

    def __init__(self, context):
        self.context = context

    def _decode_base(self, cmd_buf, engine):
        op_factory, is_end = self.context.opdef.op_factory(engine)
        cmd_buf = np.frombuffer(cmd_buf, dtype=np.uint8)
        while len(cmd_buf) > 0:
            operation = op_factory(cmd_buf)
            yield operation
            btyes_slice = operation.length // 8
            cmd_buf = cmd_buf[btyes_slice:]
            if is_end(cmd_buf, operation):
                break

    def decode_tiu_buf(self, cmd_buf):
        if cmd_buf:
            return self._decode_base(cmd_buf, self.context.opdef.Engine.TIU)
        return cmd_buf

    def decode_dma_buf(self, cmd_buf):
        if cmd_buf:
            return self._decode_base(cmd_buf, self.context.opdef.Engine.DMA)
        return cmd_buf

    def merge_instruction(self, tiu, dma, subnet_id=0):
        return self.context.opdef.merge_instruction(tiu, dma)

    def decode_bmodel_cmd(self, bmodel_cmd, subnet_id):
        tiu = itertools.islice(
            self.decode_tiu_buf(bmodel_cmd.tiu_cmd), bmodel_cmd.tiu_num
        )
        dma = itertools.islice(
            self.decode_dma_buf(bmodel_cmd.dma_cmd), bmodel_cmd.dma_num
        )
        tiu = list(tiu)
        dma = list(dma)
        return self.CMD(tiu, dma, self.merge_instruction(tiu, dma, subnet_id))


class BModelReader:
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
        def __init__(self, fbs: bmodel_fbs.CmdGroup, cmd_buf_bits):
            self.tiu_num = fbs.BdcNum()
            self.dma_num = fbs.GdmaNum()
            if fbs.BinaryBdc():
                binary_tiu = (fbs.BinaryBdc().Start(), fbs.BinaryBdc().Size())
                self.tiu_cmd = cmd_buf_bits[binary_tiu[0] : sum(binary_tiu)]
            else:
                self.tiu_cmd = []
            if fbs.BinaryGdma():
                binary_dma = (fbs.BinaryGdma().Start(), fbs.BinaryGdma().Size())
                self.dma_cmd = cmd_buf_bits[binary_dma[0] : sum(binary_dma)]
            else:
                self.dma_cmd = []

        def __repr__(self):
            return f"tiu_num: {self.tiu_num}\ndma_num: {self.dma_num}"

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
            0: op_support.DType.f32,
            1: op_support.DType.f16,
            2: op_support.DType.si8,
            3: op_support.DType.ui8,
            4: op_support.DType.si16,
            5: op_support.DType.ui16,
            6: op_support.DType.si32,
            7: op_support.DType.ui32,
        }

        def __init__(self, fbs: bmodel_fbs.Tensor, buffer):
            self.name = fbs.Name().decode()
            self.dtype = self.to_DType[fbs.DataType()]
            self.device_addr = fbs.DeviceAddr()
            self.st_mode = fbs.GmemStmode()  # 0->1N, 1->2N, 2->4N
            self.pad_h = fbs.PadH()  # 1684
            self.shape = [
                list(fbs.Shape(i).DimAsNumpy()) for i in range(fbs.ShapeLength())
            ]
            self.scale = fbs.Scale()
            self.zero_point = fbs.ZeroPoint()

        @property
        def dtype_name(self):
            return self.dtype.name

        def __repr__(self):
            return f"{self.name}: {self.shape}({self.device_addr})"

    def __init__(self, bmodel_file):
        self.head = None
        self.binary_desc = None
        self.binary = None
        self.file_name = bmodel_file
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


def BModel2MLIR(bmodel_net, decoder: Decoder, indenr_size=2):
    chip = bmodel_net.nets["Chip"][0]
    assert chip.upper() == decoder.context.device.name
    context = decoder.context

    class Block:
        def __init__(self, subnet, indent=0):
            assert subnet["Id"] != []
            self.label = subnet["Id"][0]
            self.indent = indent
            self.cmds = [
                decoder.decode_bmodel_cmd(x, self.label) for x in subnet["CmdGroup"]
            ]
            self.operations = []
            for x in self.cmds:
                self.operations.extend(x.all)
            self.args = subnet["InputTensor"]
            self.terminator = subnet["OutputTensor"]
            self.successor = subnet["NextSubnetIds"]

        def __repr__(self):
            indent = " " * indenr_size * (self.indent + 1)
            ops = "\n".join((indent + f"{x}" for x in self.operations))
            args = [
                f"%{a.name}: {context.tensor2memref(a).type_str}" for a in self.args
            ]
            args = ", ".join(args)
            if all((x == -1 for x in self.successor)):
                tem = [context.tensor2memref(x) for x in self.terminator]
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
            indent = " " * indenr_size * self.indent
            regions = (indent + "\n}, {\n").join(
                (indent + f"{r}" for r in self.regions)
            )

            def fmt_names(x):
                names = (f'"{n.name}"' for n in x)
                return f"[{', '.join(names)}]"

            arg = f"arg_attrs = {fmt_names(self.signature[0])}"
            ret = f"res_attrs = {fmt_names(self.signature[1])}"
            attr = f"{{function_type = {{{arg}, {ret}}}}}"
            operands = ", ".join(
                (str(context.tensor2memref(x)) for x in self.signature[0])
            )
            returns = ", ".join(
                (context.tensor2memref(x).type_str for x in self.signature[1])
            )
            return (
                indent
                + f"func.func @{self.name}({operands}) -> ({returns}) ({{\n{regions}\n"
                + indent
                + f"}}) {attr}"
            )

    class Module:
        def __init__(self, nets):
            self.__nets = nets
            self.chip = nets["Chip"][0]
            self.version = nets["Version"][0]
            self.type = nets["Type"][0]

        @property
        @functools.lru_cache()
        def functions(self):  # lazy eval
            return [Function(x, 1) for x in self.__nets["Net"]]

        def __repr__(self):
            funs = "\n".join((f"{x}" for x in self.functions))
            attrs = f'attributes {{chip = "{self.chip}", version = {self.version}}}'
            return f"module {attrs} {{\n{funs}\n}}"

    if context.device.name == "BM1686":
        coeff = Module(bmodel_net.nets).functions[0].regions[0].data
        if coeff:
            context.base_addr[1] += len(coeff.data)

    return Module(bmodel_net.nets)
