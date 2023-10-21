# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from collections import namedtuple
import itertools
import functools
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

    def _decode_base(self, cmd_buf, engine, core_id):
        op_factory, is_end = self.context.opdef.op_factory(engine)
        cmd_buf = np.frombuffer(cmd_buf, dtype=np.uint8)
        while len(cmd_buf) > 0:
            operation = op_factory(cmd_buf)
            operation.core_id = core_id
            yield operation
            btyes_slice = operation.length // 8
            cmd_buf = cmd_buf[btyes_slice:]
            if is_end(cmd_buf, operation):
                break

    def decode_tiu_buf(self, cmd_buf, core_id=0):
        if cmd_buf:
            return self._decode_base(cmd_buf, self.context.opdef.Engine.TIU, core_id)
        return cmd_buf

    def decode_dma_buf(self, cmd_buf, core_id=0):
        if cmd_buf:
            return self._decode_base(cmd_buf, self.context.opdef.Engine.DMA, core_id)
        return cmd_buf

    def merge_instruction(self, tiu, dma, subnet_id=0):
        return self.context.opdef.merge_instruction(tiu, dma)

    def decode_bmodel_cmd(self, bmodel_cmd, subnet_id, core_id=0):
        tiu = itertools.islice(
            self.decode_tiu_buf(bmodel_cmd.tiu_cmd, core_id), bmodel_cmd.tiu_num
        )
        dma = itertools.islice(
            self.decode_dma_buf(bmodel_cmd.dma_cmd, core_id), bmodel_cmd.dma_num
        )
        tiu = list(tiu)
        dma = list(dma)
        return self.CMD(tiu, dma, self.merge_instruction(tiu, dma, subnet_id))


class FBSArray:
    # flatbuffer array adapter, act like a list.
    def __init__(self, fbs, field, binary):
        self.field_name, self.field_cls = field
        self.fbs = fbs
        name = self.field_name
        assert hasattr(fbs, name + "Length")
        items = []
        for s in range(getattr(fbs, name + "Length")()):
            cmd = getattr(fbs, name)(s)
            items.append(self.field_cls(cmd, binary))
        self.items = items

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return self.items.__iter__()

    def serialize(self, builder, save_binary_fun):
        if self:
            array = [t.serialize(builder, save_binary_fun) for t in self.items]
            getattr(self.fbs, f"Start{self.field_name}Vector")(builder, len(self.items))
            for t in reversed(array):
                builder.PrependUOffsetTRelative(t)
            return builder.EndVector(len(self.items))
        return None

    def __bool__(self):
        return len(self) != 0

    def __repr__(self):
        return str(self.items)


class FBSOptional:
    def __init__(self, fbs, *args):
        self.has = bool(fbs)
        if self:
            self.init(fbs, *args)

    def serialize(self, *args):
        if self:
            return self._serialize(*args)
        return None

    def __bool__(self):
        return self.has

    def init(self, *_):
        raise NotImplementedError("need a init method as _init__.")

    def _serialize(self, *_):
        raise NotImplementedError("need concrete serialize method for this message.")


class BModel:
    header_t = np.dtype(
        [
            ("magic", np.uint32),
            ("header_size", np.uint32),
            ("flatbuffers_size", np.uint32),
            ("binary_size", np.uint32),
            ("reserved", np.uint32, 12),
        ]
    )

    class CmdGroup(FBSOptional):
        def init(self, fbs: bmodel_fbs.CmdGroup, buffer):
            self.tiu_num = fbs.BdcNum()
            self.dma_num = fbs.GdmaNum()
            if fbs.BinaryBdc():
                binary_tiu = (fbs.BinaryBdc().Start(), fbs.BinaryBdc().Size())
                self.tiu_cmd = buffer[binary_tiu[0] : sum(binary_tiu)]
            else:
                self.tiu_cmd = []
            if fbs.BinaryGdma():
                binary_dma = (fbs.BinaryGdma().Start(), fbs.BinaryGdma().Size())
                self.dma_cmd = buffer[binary_dma[0] : sum(binary_dma)]
            else:
                self.dma_cmd = []

        def _serialize(self, builder, save_binary_fun):
            module = bmodel_fbs.CmdGroup
            module.Start(builder)
            module.AddBdcNum(builder, self.tiu_num)
            module.AddGdmaNum(builder, self.dma_num)
            tiu_range = save_binary_fun(self.tiu_cmd)
            bmodel_fbs.CmdGroup.AddBinaryBdc(
                builder, bmodel_fbs.Binary.CreateBinary(builder, *tiu_range)
            )
            dma_range = save_binary_fun(self.dma_cmd)
            bmodel_fbs.CmdGroup.AddBinaryGdma(
                builder, bmodel_fbs.Binary.CreateBinary(builder, *dma_range)
            )
            module.AddBdcCmdByte(builder, tiu_range[1])
            module.AddGdmaCmdByte(builder, dma_range[1])
            return module.End(builder)

        def __repr__(self):
            if self:
                return f"tiu_num: {self.tiu_num}\ndma_num: {self.dma_num}"

    class CoreCmdGroup(FBSOptional):
        def init(self, fbs: bmodel_fbs.CoreCommands, buffer):
            self.gdma_tiu_commands = FBSArray(
                fbs, ("GdmaTiuCommands", BModel.CmdGroup), buffer
            )

        def _serialize(self, builder, save_binary_fun):
            module = bmodel_fbs.CoreCommands
            gdma_tiu_commands = self.gdma_tiu_commands.serialize(
                builder, save_binary_fun
            )
            module.Start(builder)
            if gdma_tiu_commands:
                module.AddGdmaTiuCommands(builder, gdma_tiu_commands)
            return module.End(builder)

        def __repr__(self):
            if self:
                return f"gdma_tiu_commands: {self.gdma_tiu_commands}"

    class ROData(FBSOptional):
        def init(self, fbs: bmodel_fbs.CoeffMem, buffer):
            self.address = fbs.Address()
            self.check_code = fbs.CheckCodeAsNumpy()
            binary_data = (fbs.BinaryCoeff().Start(), fbs.BinaryCoeff().Size())
            self.data = buffer[binary_data[0] : sum(binary_data)]

        def _serialize(self, builder, save_binary_fun):
            module = bmodel_fbs.CoeffMem

            check_code = builder.CreateNumpyVector(
                np.array(self.check_code, dtype=np.uint8)
            )
            module.Start(builder)
            module.AddAddress(builder, self.address)
            module.AddCheckCode(builder, check_code)
            coeff_range = save_binary_fun(self.data)
            module.AddBinaryCoeff(
                builder, bmodel_fbs.Binary.CreateBinary(builder, *coeff_range)
            )
            return module.End(builder)

        def __repr__(self):
            if self:
                check_code = "".join(f"{x:0>2x}" for x in self.check_code)
                return f"data: {self.address}\nshr256: {check_code}"

    class Tensor(FBSOptional):
        # fmt: off
        to_DType = {
            0: op_support.DType.f32, op_support.DType.f32: 0,
            1: op_support.DType.f16, op_support.DType.f16: 1,
            2: op_support.DType.si8, op_support.DType.si8: 2,
            3: op_support.DType.ui8, op_support.DType.ui8: 3,
            4: op_support.DType.si16, op_support.DType.si16: 4,
            5: op_support.DType.ui16, op_support.DType.ui16: 5,
            6: op_support.DType.si32, op_support.DType.si32: 6,
            7: op_support.DType.ui32, op_support.DType.ui32: 7,
        }
        # fmt: on
        def init(self, fbs: bmodel_fbs.Tensor, _):
            self.name = fbs.Name().decode()
            self.dtype = self.to_DType[fbs.DataType()]
            self.device_addr = fbs.DeviceAddr()
            self.st_mode = fbs.GmemStmode()  # 0->1N, 1->2N, 2->4N
            self.mem_type = fbs.MemType()
            self.pad_h = fbs.PadH()  # 1684
            self.shape = [
                list(fbs.Shape(i).DimAsNumpy()) for i in range(fbs.ShapeLength())
            ]
            self.scale = fbs.Scale()
            self.zero_point = fbs.ZeroPoint()
            self.size = fbs.Size()

        def _serialize(self, builder, _):
            module = bmodel_fbs.Tensor

            def build_shape(shape):
                dims = builder.CreateNumpyVector(np.array(shape, dtype=np.uint64))
                bmodel_fbs.Shape.Start(builder)
                bmodel_fbs.Shape.AddDim(builder, dims)
                return bmodel_fbs.Shape.End(builder)

            shapes = [build_shape(shape) for shape in self.shape]
            name = builder.CreateString(self.name)
            bmodel_fbs.Tensor.StartShapeVector(builder, len(self.shape))
            for shape in reversed(shapes):
                builder.PrependUOffsetTRelative(shape)
            shapes = builder.EndVector(len(self.shape))
            module.Start(builder)
            module.AddName(builder, name)
            module.AddDataType(builder, self.to_DType[self.dtype])
            module.AddGmemStmode(builder, self.st_mode)
            module.AddDeviceAddr(builder, self.device_addr)
            module.AddPadH(builder, self.pad_h)
            module.AddShape(builder, shapes)
            module.AddScale(builder, self.scale)
            module.AddZeroPoint(builder, self.zero_point)
            module.AddSize(builder, self.size)
            module.AddMemType(builder, self.mem_type)
            return module.End(builder)

        @property
        def dtype_name(self):
            return self.dtype.name

        def __repr__(self):
            return f"{self.name}: {self.shape} {self.dtype.name} ({self.device_addr})"

    class KernelModule(FBSOptional):
        def init(self, fbs: bmodel_fbs.KernelModule, buffer):
            self.file_name = fbs.FileName().decode()
            binary_data = (fbs.Binary().Start(), fbs.Binary().Size())
            self.data = buffer[binary_data[0] : sum(binary_data)]

        def _serialize(self, builder, save_binary_fun):
            module = bmodel_fbs.KernelModule
            file_name = builder.CreateString(self.file_name)
            module.Start(builder)
            module.AddFileName(builder, file_name)
            binary_range = save_binary_fun(self.data)
            module.AddBinary(
                builder, bmodel_fbs.Binary.CreateBinary(builder, *binary_range)
            )
            return module.End(builder)

        def __repr__(self) -> str:
            return f"kernel: {self.file_name}"

    class SubNet(FBSOptional):
        def init(self, fbs: bmodel_fbs.SubNet, buffer):
            self.input_tensor = FBSArray(fbs, ("InputTensor", BModel.Tensor), buffer)
            self.output_tensor = FBSArray(fbs, ("OutputTensor", BModel.Tensor), buffer)
            self.cmd_group = FBSArray(fbs, ("CmdGroup", BModel.CmdGroup), buffer)
            self.next_subnet_ids = fbs.NextSubnetIdsAsNumpy()
            self.id = fbs.Id()
            self.core_commands = FBSArray(
                fbs, ("CoreCommands", BModel.CoreCmdGroup), buffer
            )

        def _serialize(self, builder, save_binary_fun):
            module = bmodel_fbs.SubNet
            cmd_group = self.cmd_group.serialize(builder, save_binary_fun)
            core_commands = self.core_commands.serialize(builder, save_binary_fun)
            input_tensor = self.input_tensor.serialize(builder, save_binary_fun)
            output_tensor = self.output_tensor.serialize(builder, save_binary_fun)
            next_id = builder.CreateNumpyVector(
                np.array(self.next_subnet_ids, dtype=np.int32)
            )
            module.Start(builder)
            module.AddSubnetMode(builder, 0)
            if cmd_group:
                module.AddCmdGroup(builder, cmd_group)
            if core_commands:
                module.AddCoreCommands(builder, core_commands)
            module.AddInputTensor(builder, input_tensor)
            module.AddOutputTensor(builder, output_tensor)
            module.AddNextSubnetIds(builder, next_id)
            return module.End(builder)

        def __repr__(self):
            if self:
                return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

    class Parameter(FBSOptional):
        def init(self, fbs: bmodel_fbs.NetParameter, buffer):
            self.input_tensor = FBSArray(fbs, ("InputTensor", BModel.Tensor), buffer)
            self.output_tensor = FBSArray(fbs, ("OutputTensor", BModel.Tensor), buffer)
            self.sub_net = FBSArray(fbs, ("SubNet", BModel.SubNet), buffer)
            self.cmd_group = FBSArray(fbs, ("CmdGroup", BModel.CmdGroup), buffer)
            self.ctx_addr = fbs.CtxAddr()
            self.ctx_size = fbs.CtxSize()
            self.coeff_mem = BModel.ROData(fbs.CoeffMem(), buffer)
            self.core_num = fbs.CoreNum()

        def _serialize(self, builder, save_binary_fun):
            module = bmodel_fbs.NetParameter
            input_tensor = self.input_tensor.serialize(builder, save_binary_fun)
            output_tensor = self.output_tensor.serialize(builder, save_binary_fun)
            sub_net = self.sub_net.serialize(builder, save_binary_fun)
            cmd_group = self.cmd_group.serialize(builder, save_binary_fun)
            coeff_mem = self.coeff_mem.serialize(builder, save_binary_fun)

            module.Start(builder)
            module.AddInputTensor(builder, input_tensor)
            module.AddOutputTensor(builder, output_tensor)
            module.AddCtxAddr(builder, self.ctx_addr)
            module.AddCtxSize(builder, self.ctx_size)
            if coeff_mem:
                module.AddCoeffMem(builder, coeff_mem)
            module.AddIsDynamic(builder, 0)
            module.AddNDynamic(builder, 0)
            module.AddHWDynamic(builder, 0)
            module.AddSubNet(builder, sub_net)
            if cmd_group:
                module.AddCmdGroup(builder, cmd_group)
            module.AddCoreNum(builder, self.core_num)
            return module.End(builder)

        def __repr__(self) -> str:
            return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

    class Net:
        def __init__(self, fbs: bmodel_fbs.Net, buffer):
            self.name = fbs.Name().decode()
            self.parameter = FBSArray(fbs, ("Parameter", BModel.Parameter), buffer)

        def serialize(self, builder, save_binary_fun):
            module = bmodel_fbs.Net
            name = builder.CreateString(self.name)
            parameter = self.parameter.serialize(builder, save_binary_fun)
            module.Start(builder)
            module.AddName(builder, name)
            module.AddParameter(builder, parameter)
            return module.End(builder)

        def __repr__(self) -> str:
            return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

    def __init__(self, bmodel_file):
        self.head = None
        binary_desc = None
        binary = None
        self.file_name = bmodel_file
        with open(bmodel_file, "rb") as file_obj:
            file_obj.seek(0, 0)
            self.head = np.frombuffer(
                file_obj.read(self.header_t.itemsize), dtype=self.header_t
            )
            binary_desc = file_obj.read(self.head["flatbuffers_size"][0])
            binary = file_obj.read(self.head["binary_size"][0])
        bmodel = bmodel_fbs.Model.GetRootAsModel(binary_desc, 0)

        self.chip = bmodel.Chip().decode()
        self.version = bmodel.Version().decode()
        self.type = bmodel.Type().decode()
        self.kernel_module = self.KernelModule(bmodel.KernelModule(), binary)
        self.neuron_size = bmodel.NeuronSize()
        self.time = bmodel.Time().decode()
        self.net = FBSArray(bmodel, ("Net", self.Net), binary)
        self.core_num = self.net[0].parameter[0].core_num

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

    def serialize(self, file_name):
        import flatbuffers

        builder = flatbuffers.Builder(1024)
        payload = []

        def save_binary(data):
            start = len(payload)
            size = len(data)
            payload.extend(data)
            return start, size

        module = bmodel_fbs.Model
        chip = builder.CreateString(self.chip)
        version = builder.CreateString(self.version)
        type_ = builder.CreateString(self.type)
        time = builder.CreateString(self.time)

        kernel_module = self.kernel_module.serialize(builder, save_binary)

        net = self.net.serialize(builder, save_binary)
        module.Start(builder)
        module.AddType(builder, type_)
        module.AddVersion(builder, version)
        module.AddChip(builder, chip)
        module.AddTime(builder, time)
        module.AddNet(builder, net)
        module.AddNeuronSize(builder, self.neuron_size)
        if kernel_module:
            module.AddKernelModule(builder, kernel_module)
        model = bmodel_fbs.Model.End(builder)

        builder.Finish(model)
        buffer = builder.Output()
        magic = self.head["magic"]
        header_size = self.head["header_size"]
        reserved = self.head["reserved"]

        header = np.array(
            (magic, header_size, len(buffer), len(payload), reserved),
            dtype=self.header_t,
        )

        with open(file_name, "w") as f:
            header.tofile(f)
            np.array(buffer).tofile(f)
            np.array(payload, np.uint8).tofile(f)


def BModel2MLIR(bmodel_net, decoder: Decoder, indenr_size=2):
    chip = bmodel_net.chip
    assert chip.upper() == decoder.context.device.name
    context = decoder.context

    class Block:
        def __init__(self, subnet, indent=0):
            self.label = subnet.id
            self.indent = indent
            self.operations = []
            if bmodel_net.core_num > 1:
                self.cmds = [
                    decoder.decode_bmodel_cmd(cmd, self.label, core_id)
                    for core_id, x in enumerate(subnet.core_commands)
                    for cmd in x.gdma_tiu_commands
                ]
                sorter = context.opdef.MultiCoreCmd([i.all for i in self.cmds])
                self.operations = sorter.consume_cmd()
            else:
                self.cmds = [
                    decoder.decode_bmodel_cmd(x, self.label) for x in subnet.cmd_group
                ]
                for x in self.cmds:
                    self.operations.extend(x.all)
            self.args = subnet.input_tensor
            self.terminator = subnet.output_tensor
            self.successor = subnet.next_subnet_ids

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
            self.blocks = [Block(x, indent) for x in net_stage.sub_net]
            self.signature = (net_stage.input_tensor, net_stage.output_tensor)
            self.data = net_stage.coeff_mem if net_stage.coeff_mem else None

        def __repr__(self):
            blocks = "\n".join((f"{b}" for b in self.blocks))
            return f"{blocks}"

    class Function:
        def __init__(self, net, indent=0):
            self.indent = indent
            self.name = net.name
            self.regions = [Region(x, indent) for x in net.parameter]
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
        def __init__(self, bmodel):
            self.bmodel = bmodel
            self.chip = bmodel.chip
            self.version = bmodel.version
            self.type = bmodel.type
            self.core_num = bmodel.core_num

        @property
        @functools.lru_cache()
        def functions(self):  # lazy eval
            return [Function(x, 1) for x in self.bmodel.net]

        def __repr__(self):
            funs = "\n".join((f"{x}" for x in self.functions))
            attrs = f'attributes {{chip = "{self.chip}", version = {self.version}}}'
            return f"module {attrs} {{\n{funs}\n}}"

    if context.device.name == "BM1686":
        coeff = Module(bmodel_net).functions[0].regions[0].data
        if coeff:
            context.base_addr[1] += len(coeff.data)

    return Module(bmodel_net)
