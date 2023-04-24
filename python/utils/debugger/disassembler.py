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


def buffer_to_bits(buffer):
    cmmand_buf = np.frombuffer(buffer, dtype=np.uint8)
    return np.unpackbits(cmmand_buf, bitorder="little")


class Decoder:
    CMD = namedtuple("cmd", ["bdc", "dma", "all"])

    def __init__(self, context):
        self.context = context

    def _decode_base(self, cmd_buf_bits, opcode_bits, cmd_set, sys_end=None):
        operation = None
        cur = 0
        l, h = opcode_bits
        while cmd_buf_bits.size > 0:
            cmd_key = op_support.packbits(cmd_buf_bits[l:h])
            if cmd_key in cmd_set:
                recognize = False
                for op in cmd_set[cmd_key]:
                    if op.is_comp(cmd_buf_bits):
                        # check whether this command is recognized by the decoder
                        operation = op.decode(cmd_buf_bits)
                        yield operation
                        # consume this command_code
                        cmd_buf_bits = cmd_buf_bits[op.length :]
                        cur += op.length
                        recognize = True
                        break
                is_sys = sys_end is None or isinstance(operation, sys_end)
                is_less_1024 = cmd_buf_bits.size < 1025
                if is_sys and is_less_1024 and not np.any(cmd_buf_bits):
                    break  # all the code have been processed
                if not recognize:
                    raise ValueError(
                        "Can not decode cmd, with opcode: {}, at {}.".format(
                            cmd_key, cur
                        )
                    )
            else:
                raise ValueError(
                    "Can not decode cmd, with opcode: {}, at {}.".format(cmd_key, cur)
                )

    def decode_bdc_buf(self, cmd_buf):
        if cmd_buf:
            # input is a buffer
            return self.decode_bdc_bits(buffer_to_bits(cmd_buf))
        return cmd_buf

    def decode_dma_buf(self, cmd_buf):
        if cmd_buf:
            # input is a buffer
            return self.decode_dma_bits(buffer_to_bits(cmd_buf))
        return cmd_buf

    def decode_bdc_bits(self, cmd_buf_bits):
        # input is a bits vector
        return self._decode_base(
            cmd_buf_bits,
            self.context.opdef.bdc_base.opcode_bits,
            self.context.opdef.bdc_cmd,
            self.context.opdef.bdc_sys,
        )

    def decode_dma_bits(self, cmd_buf_bits):
        # input is a bits vector
        return self._decode_base(
            cmd_buf_bits,
            self.context.opdef.dma_base.opcode_bits,
            self.context.opdef.dma_cmd,
            self.context.opdef.dma_sys,
        )

    def merge_instruction(self, bdc, dma, subnet_id=None):
        main_cmd, inserted_cmd = dma, bdc
        # remove the system command
        def get_end(cmd):
            sys = (self.context.opdef.bdc_sys, self.context.opdef.dma_sys)
            if all(sys):
                if isinstance(cmd[-1], sys):
                    return -1
            else:
                return len(cmd)

        # remove system instruction
        main_id = [(m.cmd_id, m) for m in main_cmd[: get_end(main_cmd)]]
        inserted_id = [(i.cmd_id_dep, i) for i in inserted_cmd[: get_end(inserted_cmd)]]
        # "sorted" is stable, which keeps the inserted commands
        # after the main instructions.
        cmd = main_id + inserted_id
        cmd_sorted = sorted(cmd, key=lambda x: x[0])
        return [x[1] for x in cmd_sorted]

    def decode_bmodel_cmd(self, bmodel_cmd, subnet_id):
        bdc = itertools.islice(
            self.decode_bdc_buf(bmodel_cmd.bdc_cmd), bmodel_cmd.bdc_num
        )
        dma = itertools.islice(
            self.decode_dma_buf(bmodel_cmd.dma_cmd), bmodel_cmd.dma_num
        )
        bdc = list(bdc)
        dma = list(dma)
        return self.CMD(bdc, dma, self.merge_instruction(bdc, dma, subnet_id))


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
            self.bdc_num = fbs.BdcNum()
            self.dma_num = fbs.GdmaNum()
            if fbs.BinaryBdc():
                binary_bdc = (fbs.BinaryBdc().Start(), fbs.BinaryBdc().Size())
                self.bdc_cmd = cmd_buf_bits[binary_bdc[0] : sum(binary_bdc)]
            else:
                self.bdc_cmd = []
            if fbs.BinaryGdma():
                binary_dma = (fbs.BinaryGdma().Start(), fbs.BinaryGdma().Size())
                self.dma_cmd = cmd_buf_bits[binary_dma[0] : sum(binary_dma)]
            else:
                self.dma_cmd = []

        def __repr__(self):
            return f"bdc_num: {self.bdc_num}\ndma_num: {self.dma_num}"

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

    return Module(bmodel_net.nets)
