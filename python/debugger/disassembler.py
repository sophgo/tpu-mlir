# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import numpy as np
from enum import Enum
from pprint import pformat
from .target_common import (
    DType,
    Layout,
    MemRefBase,
    cmd_base_reg,
    get_target_context,
    CpuLayerType,
)
from . import bmodel_fbs
from typing import List, NamedTuple
from functools import lru_cache


class StaticCmdGroup(NamedTuple):
    tiu: List[cmd_base_reg]
    dma: List[cmd_base_reg]
    all: List[cmd_base_reg]


class _BModelContext:
    def __call__(self, bmodel_net: "BModel"):
        self.bmodel_net = bmodel_net
        return self

    def __enter__(self):
        pass

    def __exit__(self, *exc_info):
        self.bmodel_net = None


bmodel_context = _BModelContext()


class FBSArray:
    # flatbuffer array adapter, act like a list.
    def __init__(self, fbs, field, binary):
        self.field_name, self.field_cls = field
        name = self.field_name

        self.fbs = fbs
        assert hasattr(fbs, name + "Length"), name + "Length"
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
        return pformat(self.items)


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


class CpuConst(FBSOptional):
    def init(self, fbs: bmodel_fbs.CpuConst, buffer):
        self.name = fbs.Name()
        binary = fbs.ConstData()
        start, size = binary.Start(), binary.Size()
        self.const_data = buffer[start : start + size]
        # skip check_code


class CpuParam(FBSOptional):
    def init(self, fbs: bmodel_fbs.CpuParam, buffer: bytes):
        self.op_type = CpuLayerType(fbs.OpType())
        binary = fbs.BinaryParam()
        start, size = binary.Start(), binary.Size()
        self.cpu_cmd = buffer[start : start + size]
        self.cpu_const: List[CpuConst] = FBSArray(fbs, ("CpuConst", CpuConst), buffer)

    def _serialize(self, builder, save_binary_fun):
        module = bmodel_fbs.CpuParam
        module.Start(builder)
        module.AddOpType(builder, self.op_type)
        cpu_range = save_binary_fun(self.cpu_cmd)
        module.AddBinaryParam(
            builder, bmodel_fbs.Binary.CreateBinary(builder, *cpu_range)
        )

    def __repr__(self) -> str:
        return f"CPU_OP: {self.op_type.name}"


class CmdGroup(FBSOptional):
    def init(self, fbs: bmodel_fbs.CmdGroup, buffer: bytes):
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
            return f"tiu_cmd[{self.tiu_num}], dma_cmd[{self.dma_num}]"


class CoreCmdGroup(FBSOptional):
    def init(self, fbs: bmodel_fbs.CoreCommands, buffer: bytes):
        self.gdma_tiu_commands: List[CmdGroup] = FBSArray(
            fbs, ("GdmaTiuCommands", CmdGroup), buffer
        )

    def _serialize(self, builder, save_binary_fun):
        module = bmodel_fbs.CoreCommands
        gdma_tiu_commands = self.gdma_tiu_commands.serialize(builder, save_binary_fun)
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
            return f"[data: {self.address}, shr256: {check_code}]"


class Tensor(FBSOptional):
    # fmt: off
    to_DType = {
        0: DType.f32, DType.f32: 0,
        1: DType.f16, DType.f16: 1,
        2: DType.si8, DType.si8: 2,
        3: DType.ui8, DType.ui8: 3,
        4: DType.si16, DType.si16: 4,
        5: DType.ui16, DType.ui16: 5,
        6: DType.si32, DType.si32: 6,
        7: DType.ui32, DType.ui32: 7,
    }
    # fmt: on

    def init(self, fbs: bmodel_fbs.Tensor, _):
        self.name = fbs.Name().decode()
        self.dtype: DType = self.to_DType[fbs.DataType()]
        self.device_addr = fbs.DeviceAddr()
        self.st_mode = fbs.GmemStmode()  # 0->1N, 1->2N, 2->4N
        self.mem_type = fbs.MemType()
        self.pad_h = fbs.PadH()  # 1684
        self.shape = [list(fbs.Shape(i).DimAsNumpy()) for i in range(fbs.ShapeLength())]
        self.scale = fbs.Scale()
        self.zero_point = fbs.ZeroPoint()
        self.size = fbs.Size()

        self._bmodel = bmodel_context.bmodel_net

    @property
    @lru_cache()
    def memref(self) -> MemRefBase:
        assert self.pad_h == 0, "Not supports pad_h."
        context = self._bmodel.context

        dtype = self.dtype
        if self.st_mode in (1, 2):  # 2N/4N
            assert context.device == context.device.BM1684
            # The type should be declared explicitly.
            layout = Layout.continuous_XN
            xn = 4 // dtype.itemsize
            stride = context.get_continuous_stride(self.shape[0]) * xn
        else:
            stride = context.get_continuous_stride(self.shape[0])
            layout = None  # global is continuous
        return context.MemRef(
            self.device_addr,
            self.shape[0],
            dtype,
            stride=stride,
            layout=layout,
        )

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


class RunMode(Enum):
    TPU_STATIC = 0
    TPU_DYNAMIC = 1
    CPU = 2
    LOOP = 3
    SWITCH = 4
    MERGE = 5
    UNKNOWN = 9


class SubNet(FBSOptional):
    def init(self, fbs: bmodel_fbs.SubNet, buffer):
        # tpu-static, tpu-dynamic, cpu
        self.subnet_mode = fbs.SubnetMode()  # tpu:0 cpu:1
        self.is_dynamic = fbs.IsDynamic() == 1
        self.run_mode = RunMode.UNKNOWN
        if self.subnet_mode == 1:
            self.run_mode = RunMode.CPU
        elif self.is_dynamic:
            self.run_mode = RunMode.TPU_DYNAMIC
        else:
            self.run_mode = RunMode.TPU_STATIC

        self.input_tensor: List[Tensor] = FBSArray(fbs, ("InputTensor", Tensor), buffer)
        self.output_tensor: List[Tensor] = FBSArray(
            fbs, ("OutputTensor", Tensor), buffer
        )
        self.next_subnet_ids: int = fbs.NextSubnetIdsAsNumpy()
        self.id: int = fbs.Id()

        # tpu-static info
        self.cmd_group: List[CmdGroup] = FBSArray(fbs, ("CmdGroup", CmdGroup), buffer)
        self.core_commands: List[CoreCmdGroup] = FBSArray(
            fbs, ("CoreCommands", CoreCmdGroup), buffer
        )

        # tpu-dynamic ir info
        self.ir_len = fbs.IrLen()
        self.ir_offset = fbs.IrOffset()

        if self.run_mode == RunMode.TPU_DYNAMIC:
            self.ir_buffer = buffer[self.ir_offset : self.ir_offset + self.ir_len]
        elif self.run_mode == RunMode.CPU:
            self.cpu_params: List[CpuParam] = FBSArray(
                fbs, ("CpuParam", CpuParam), buffer
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
            return pformat(self.__dict__, indent=2)


class Parameter(FBSOptional):
    def init(self, fbs: bmodel_fbs.NetParameter, buffer):
        self.input_tensor: List[Tensor] = FBSArray(fbs, ("InputTensor", Tensor), buffer)
        self.output_tensor: List[Tensor] = FBSArray(
            fbs, ("OutputTensor", Tensor), buffer
        )
        self.sub_net: List[SubNet] = FBSArray(fbs, ("SubNet", SubNet), buffer)
        self.cmd_group: List[CmdGroup] = FBSArray(fbs, ("CmdGroup", CmdGroup), buffer)
        self.ctx_addr = fbs.CtxAddr()
        self.ctx_size = fbs.CtxSize()
        self.coeff_mem = ROData(fbs.CoeffMem(), buffer)
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
        return pformat(self.__dict__)


class Net:
    def __init__(self, fbs: bmodel_fbs.Net, buffer: bytes):
        self.name = fbs.Name().decode()
        self.parameter: List[Parameter] = FBSArray(
            fbs, ("Parameter", Parameter), buffer
        )

    def serialize(self, builder, save_binary_fun):
        module = bmodel_fbs.Net
        name = builder.CreateString(self.name)
        parameter = self.parameter.serialize(builder, save_binary_fun)
        module.Start(builder)
        module.AddName(builder, name)
        module.AddParameter(builder, parameter)
        return module.End(builder)

    def __repr__(self) -> str:
        return pformat(self.__dict__)


bmodel_header_type = np.dtype(
    [
        ("magic", np.uint32),
        ("header_size", np.uint32),
        ("flatbuffers_size", np.uint32),
        ("binary_size", np.uint32),
        ("reserved", np.uint32, 12),
    ]
)


class BModel:
    def __init__(self, bmodel_file):
        with bmodel_context(self):
            self.head = None
            binary_desc = None
            binary = None
            self.file_name = bmodel_file
            with open(bmodel_file, "rb") as file_obj:
                file_obj.seek(0, 0)
                self.head = np.frombuffer(
                    file_obj.read(bmodel_header_type.itemsize), dtype=bmodel_header_type
                )
                binary_desc = file_obj.read(self.head["flatbuffers_size"][0])
                binary = file_obj.read(self.head["binary_size"][0])
            bmodel = bmodel_fbs.Model.GetRootAsModel(binary_desc, 0)

            self.chip = bmodel.Chip().decode()
            self.version = bmodel.Version().decode()
            self.type = bmodel.Type().decode()
            self.kernel_module = KernelModule(bmodel.KernelModule(), binary)
            self.neuron_size = bmodel.NeuronSize()
            self.time = bmodel.Time().decode()
            self.net: List[Net] = FBSArray(bmodel, ("Net", Net), binary)
            self.core_num = self.net[0].parameter[0].core_num
            self.context = get_target_context(self.chip)

    def __repr__(self):
        return pformat(self.__dict__)

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
            dtype=bmodel_header_type,
        )

        with open(file_name, "w") as f:
            header.tofile(f)
            np.array(buffer).tofile(f)
            np.array(payload, np.uint8).tofile(f)

    def decode_cpu_op(self, cpu_param: CpuParam):
        return cpu_param

    def decode_dynamic_ir(self, ir_buffer):
        return ir_buffer
