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
    atomic_reg,
    get_target_context,
    CpuLayerType,
    BaseTpuCmd,
)
from . import bmodel_fbs
from typing import List, NamedTuple
from functools import lru_cache



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
    def __init__(self, fbs, field, *args):
        self.field_name, self.field_cls = field
        name = self.field_name

        self.fbs = fbs
        assert hasattr(fbs, name + "Length"), name + "Length"
        items = []
        for s in range(getattr(fbs, name + "Length")()):
            cmd = getattr(fbs, name)(s)
            items.append(self.field_cls(cmd, *args))
        self.items = items

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return self.items.__iter__()

    def serialize(self, builder, save_binary_fun):
        if self:
            if issubclass(self.field_cls, Binary):
                # Structs should be stored inline in their parent.
                getattr(self.fbs, f"Start{self.field_name}Vector")(
                    builder, len(self.items)
                )
                [t.serialize(builder, save_binary_fun) for t in self.items]
                return builder.EndVector()
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
    def init(self, fbs: bmodel_fbs.CpuParam, buffer: memoryview):
        self.op_type = CpuLayerType(fbs.OpType())
        binary = fbs.BinaryParam()
        start, size = binary.Start(), binary.Size()
        self.cpu_cmd = buffer[start : start + size]
        self.cpu_const: List[CpuConst] = FBSArray(fbs, ("CpuConst", CpuConst), buffer)

    def _serialize(self, builder, save_binary_fun):
        module = bmodel_fbs.CpuParam
        cpu_const = self.cpu_const.serialize(builder, save_binary_fun)
        module.Start(builder)
        module.AddOpType(builder, self.op_type)
        cpu_range = save_binary_fun(self.cpu_cmd)
        module.AddBinaryParam(
            builder, bmodel_fbs.Binary.CreateBinary(builder, *cpu_range)
        )
        module.AddCpuConst(builder, cpu_const)
        return module.End(builder)

    def __repr__(self) -> str:
        return f"CPU_OP: {self.op_type.name}"


class StageIr(FBSOptional):
    def init(
        self, fbs: bmodel_fbs.StageIR, binary_ir: bmodel_fbs.Binary, buffer: memoryview
    ):
        self.ir_len = fbs.IrInfoLen()
        self.ir_bytelen = self.ir_len * 4  # sizeof(u32)

    def _serialize(self, builder, save_binary_fun):
        module = bmodel_fbs.StageIR
        module.Start(builder)
        module.AddIrInfoLen(builder, self.ir_len)  # 0
        module.AddHeightHigh(builder, 0)  # 1
        module.AddHeightLow(builder, 0)  # 2
        module.AddWidthHigh(builder, 0)  # 3
        module.AddWidthLow(builder, 0)  # 4
        return module.End(builder)


class Binary(FBSOptional):
    def init(self, fbs: bmodel_fbs.Binary, buffer: memoryview):
        binary = (fbs.Start(), fbs.Size())
        self._bytes = buffer[binary[0] : sum(binary)]

    @property
    def bytes(self):
        return getattr(self, "_bytes", memoryview(bytes()))

    def __bytes__(self):
        return bytes(self.bytes)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.bytes[key]
        return self.bytes[key]

    def __len__(self):
        return len(self.bytes)

    def _serialize(self, builder, save_binary_fun):
        data_range = save_binary_fun(self.bytes)
        return bmodel_fbs.Binary.CreateBinary(builder, *data_range)

    def __array__(self, dtype=np.uint8):
        # Convert to numpy array
        return np.frombuffer(self.bytes, dtype=dtype)

    def __repr__(self):
        if self:
            return f"bin:{len(self.bytes)}"
        return "empty"


class CmdGroup(FBSOptional):
    def init(self, fbs: bmodel_fbs.CmdGroup, buffer: memoryview):
        self.tiu_num = fbs.BdcNum()
        self.dma_num = fbs.GdmaNum()
        self.tiu_cmd = Binary(fbs.BinaryBdc(), buffer)
        self.tiu_cmd.bytes
        self.dma_cmd = Binary(fbs.BinaryGdma(), buffer)

    def _serialize(self, builder, save_binary_fun):
        module = bmodel_fbs.CmdGroup
        module.Start(builder)
        module.AddBdcNum(builder, self.tiu_num)  # 0
        module.AddGdmaNum(builder, self.dma_num)  # 1
        module.AddBinaryBdc(builder, self.tiu_cmd.serialize(builder, save_binary_fun))
        module.AddBinaryGdma(builder, self.dma_cmd.serialize(builder, save_binary_fun))
        module.AddBdcCmdByte(builder, len(self.tiu_cmd))  # 4
        module.AddGdmaCmdByte(builder, len(self.dma_cmd))  # 5

        return module.End(builder)

    def __repr__(self):
        if self:
            return f"tiu_cmd[{self.tiu_num}], dma_cmd[{self.dma_num}]"
        return ""


class CoreCmdGroup(FBSOptional):
    def init(self, fbs: bmodel_fbs.CoreCommands, buffer: memoryview):
        self.gdma_tiu_commands = FBSArray(fbs, ("GdmaTiuCommands", CmdGroup), buffer)
        self.sdma_commands = FBSArray(fbs, ("SdmaCommands", Binary), buffer)
        self.hau_commands = FBSArray(fbs, ("HauCommands", Binary), buffer)
        self.cdma_commands = FBSArray(fbs, ("CdmaCommands", Binary), buffer)

    def _serialize(self, builder, save_binary_fun):
        module = bmodel_fbs.CoreCommands
        gdma_tiu_commands = self.gdma_tiu_commands.serialize(builder, save_binary_fun)
        sdma_commands = self.sdma_commands.serialize(builder, save_binary_fun)
        hau_commands = self.hau_commands.serialize(builder, save_binary_fun)
        cdma_commands = self.cdma_commands.serialize(builder, save_binary_fun)

        module.Start(builder)
        if gdma_tiu_commands:
            module.AddGdmaTiuCommands(builder, gdma_tiu_commands)
        if sdma_commands:
            module.AddSdmaCommands(builder, sdma_commands)
        if hau_commands:
            module.AddHauCommands(builder, hau_commands)
        if cdma_commands:
            module.AddCdmaCommands(builder, cdma_commands)

        return module.End(builder)

    def __repr__(self):
        if self:
            return (
                f"gdma_tiu_commands: {self.gdma_tiu_commands}"
                + f", sdma_commands: {self.sdma_commands}"
                + f", hau_commands: {self.hau_commands}"
                + f", cdma_commands: {self.cdma_commands}"
            )
        return ""


class ROData(FBSOptional):
    def init(self, fbs: bmodel_fbs.CoeffMem, buffer):
        self.address = fbs.Address()
        self.check_code = fbs.CheckCodeAsNumpy()
        binary_data = (fbs.BinaryCoeff().Start(), fbs.BinaryCoeff().Size())
        self.data = buffer[binary_data[0] : sum(binary_data)]
        self.coeff_size = binary_data[1]

    def _serialize(self, builder, save_binary_fun):
        module = bmodel_fbs.CoeffMem

        check_code = builder.CreateNumpyVector(
            np.array(self.check_code, dtype=np.uint8)
        )
        module.Start(builder)
        module.AddAddress(builder, self.address)  # 0
        module.AddCheckCode(builder, check_code)  # 1
        coeff_range = save_binary_fun(self.data)
        module.AddBinaryCoeff(
            builder, bmodel_fbs.Binary.CreateBinary(builder, *coeff_range)
        )  # 4
        return module.End(builder)

    def __repr__(self):
        if self:
            check_code = "".join(f"{x:0>2x}" for x in self.check_code)
            return f"[data: {self.address}, shr256: {check_code}]"
        return ""


class Tensor(FBSOptional):
    # fmt: off
    to_DType = {
        0:  DType.f32,   DType.f32:  0,
        1:  DType.f16,   DType.f16:  1,
        2:  DType.si8,   DType.si8:  2,
        3:  DType.ui8,   DType.ui8:  3,
        4:  DType.si16,  DType.si16: 4,
        5:  DType.ui16,  DType.ui16: 5,
        6:  DType.si32,  DType.si32: 6,
        7:  DType.ui32,  DType.ui32: 7,
        8:  DType.bf16,  DType.bf16: 8,
        9:  DType.i4,    DType.i4:   9,
        10: DType.ui4,   DType.ui4:  10,
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
        self.cpu_addr = fbs.CpuAddr()
        self.hidden = fbs.Hidden()
        self.index = fbs.Index()

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
        module.AddName(builder, name)  # 0
        module.AddDataType(builder, self.to_DType[self.dtype])  # 1
        module.AddGmemStmode(builder, self.st_mode)  # 2
        module.AddDeviceAddr(builder, self.device_addr)  # 3
        module.AddSize(builder, self.size)  # 4
        module.AddShape(builder, shapes)  # 5
        module.AddMemType(builder, self.mem_type)  # 6
        module.AddScale(builder, self.scale)  # 7
        module.AddCpuAddr(builder, self.cpu_addr)  # 8
        module.AddPadH(builder, self.pad_h)  # 9
        module.AddZeroPoint(builder, self.zero_point)  # 10
        module.AddHidden(builder, self.hidden)  # 11
        module.AddIndex(builder, self.index)  # 12
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
        module.AddFileName(builder, file_name)  # 0
        binary_range = save_binary_fun(self.data)
        module.AddBinary(
            builder, bmodel_fbs.Binary.CreateBinary(builder, *binary_range)
        )  # 1
        return module.End(builder)

    def __repr__(self) -> str:
        if self:
            return f"kernel: {self.file_name}"
        return ""


# /workspace/nntoolchain/tpu-runtime/include/bmruntime.h:112
class RunMode(Enum):
    TPU_STATIC = 0
    TPU_DYNAMIC = 1
    CPU = 2
    LOOP = 3
    SWITCH = 4
    MERGE = 5
    UNKNOWN = 9


# class MergeParam()
#


class SubNet(FBSOptional):
    def init(self, fbs: bmodel_fbs.SubNet, buffer, binary_ir: memoryview):
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

        # self.merge_param = fbs.MergeParam()
        # self.switch_param = fbs.SwitchParam()
        self.ir_buffer = self.cpu_param = None
        if self.run_mode == RunMode.TPU_DYNAMIC:
            self.ir_buffer = binary_ir[self.ir_offset : self.ir_offset + self.ir_len]
        elif self.run_mode == RunMode.CPU:
            self.cpu_param: List[CpuParam] = FBSArray(
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
        module.AddSubnetMode(builder, 0)  # 0
        if cmd_group:
            module.AddCmdGroup(builder, cmd_group)  # 1

        if self.cpu_param:
            cpu_param = self.cpu_param.serialize(builder, save_binary_fun)
            module.AddCpuParam(builder, cpu_param)  # 2

        module.AddInputTensor(builder, input_tensor)  # 3
        module.AddOutputTensor(builder, output_tensor)  # 4

        module.AddIsDynamic(builder, self.is_dynamic)  # 5
        module.AddIrOffset(builder, self.ir_offset)  # 6
        module.AddIrLen(builder, self.ir_len)  # 7
        module.AddNDynamic(builder, 0)  # 8
        module.AddHWDynamic(builder, 0)  # 9
        module.AddId(builder, self.id)  # 10
        module.AddNextSubnetIds(builder, next_id)  # 11

        # module.AddMergeParam(builder, self.merge_param)  # 12
        # module.AddSwitchParam(builder, self.switch_param)  # 13

        if core_commands:
            module.AddCoreCommands(builder, core_commands)  # 14
        return module.End(builder)

    def __repr__(self):
        if self:
            return pformat(self.__dict__, indent=2)
        return ""


class NetStatic(FBSOptional):
    pass


class NetDynamic(FBSOptional):
    pass


class Parameter(FBSOptional):
    def init(self, fbs: bmodel_fbs.NetParameter, buffer: memoryview):
        self.input_tensor: List[Tensor] = FBSArray(fbs, ("InputTensor", Tensor), buffer)
        self.output_tensor: List[Tensor] = FBSArray(
            fbs, ("OutputTensor", Tensor), buffer
        )
        self.stage_ir: List[StageIr] = FBSArray(
            fbs, ("StageIr", StageIr), fbs.BinaryIr(), buffer
        )
        self.binary_ir = Binary(fbs.BinaryIr(), buffer)
        self.sub_net: List[SubNet] = FBSArray(
            fbs, ("SubNet", SubNet), buffer, self.binary_ir
        )
        self.cmd_group: List[CmdGroup] = FBSArray(fbs, ("CmdGroup", CmdGroup), buffer)
        self.ctx_addr = fbs.CtxAddr()
        self.ctx_size = fbs.CtxSize()
        self.ctx_sizes: List[int] = FBSArray(fbs, ("CtxSizes", int))
        self.coeff_mem = ROData(fbs.CoeffMem(), buffer)
        self.core_num = fbs.CoreNum()
        self.cpu_mem_size = fbs.CpuMemSize()
        self.net_profile = Binary(fbs.NetProfile(), buffer)

    def _serialize(self, builder, save_binary_fun):
        module = bmodel_fbs.NetParameter

        input_tensor = self.input_tensor.serialize(builder, save_binary_fun)
        output_tensor = self.output_tensor.serialize(builder, save_binary_fun)
        sub_net = self.sub_net.serialize(builder, save_binary_fun)
        cmd_group = self.cmd_group.serialize(builder, save_binary_fun)
        coeff_mem = self.coeff_mem.serialize(builder, save_binary_fun)
        stage_ir = self.stage_ir.serialize(builder, save_binary_fun)
        ctx_sizes = builder.CreateNumpyVector(np.array(self.ctx_sizes, dtype=np.uint64))

        module.Start(builder)
        module.AddInputTensor(builder, input_tensor)  # 0
        module.AddOutputTensor(builder, output_tensor)  # 1
        module.AddCtxAddr(builder, self.ctx_addr)  # 2
        module.AddCtxSize(builder, self.ctx_size)  # 3
        if coeff_mem:
            module.AddCoeffMem(builder, coeff_mem)  # 4
        module.AddIsDynamic(builder, 0)  # 5
        module.AddNDynamic(builder, 0)  # 6
        module.AddHWDynamic(builder, 0)  # 7
        if cmd_group:
            module.AddCmdGroup(builder, cmd_group)  # 8
        # Structs should be stored inline in their parent.
        net_profile = self.net_profile.serialize(builder, save_binary_fun)
        if net_profile:
            module.AddNetProfile(builder, net_profile)
        if stage_ir:
            module.AddStageIr(builder, stage_ir)  # 10
        binary_ir = self.binary_ir.serialize(builder, save_binary_fun)
        if binary_ir:
            module.AddBinaryIr(builder, binary_ir)  # 11
        module.AddSubNet(builder, sub_net)  # 12
        module.AddCpuMemSize(builder, self.cpu_mem_size)  # 13

        module.AddCtxSizes(builder, ctx_sizes)  # 14
        # module.AddNetStat(builder, ) # 15
        module.AddCoreNum(builder, self.core_num)  # 16
        return module.End(builder)

    def __repr__(self) -> str:
        return pformat(self.__dict__)


class Net:
    def __init__(self, fbs: bmodel_fbs.Net, buffer: memoryview):
        self.name = fbs.Name().decode()
        self.parameter: List[Parameter] = FBSArray(
            fbs, ("Parameter", Parameter), buffer
        )

        self.cascade = fbs.Cascade()

        try:
            self.addr_mode = fbs.AddrMode()
        except AttributeError:
            # catch for compatibility
            self.addr_mode = 0

        # for old bmodel, not use any more
        # self.net_static = FBSArray(fbs, ("NetStatic", NetStatic))
        # self.net_dynamic = FBSArray(fbs, ("NetStatic", NetDynamic))

    def serialize(self, builder, save_binary_fun):
        module = bmodel_fbs.Net
        name = builder.CreateString(self.name)
        parameter = self.parameter.serialize(builder, save_binary_fun)

        # net_static = self.net_static.serialize(builder, save_binary_fun)
        # net_dynamic = self.net_dynamic.serialize(builder, save_binary_fun)

        module.Start(builder)
        module.AddName(builder, name)  # 0
        # if net_static:
        #     module.AddNetStatic(builder, net_static)  # 1
        # if net_dynamic:
        #     module.AddNetDynamic(builder, net_dynamic)  # 2

        module.AddParameter(builder, parameter)  # 3

        if self.cascade:
            module.AddCascade(builder, self.cascade)
        module.AddAddrMode(builder, self.addr_mode)
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
                binary = memoryview(
                    bytearray(file_obj.read(self.head["binary_size"][0]))
                )
            bmodel: bmodel_fbs.Model = bmodel_fbs.Model.GetRootAsModel(binary_desc, 0)

            self.binary = binary
            self.chip = bmodel.Chip().decode()
            self.version = bmodel.Version().decode()
            self.type = bmodel.Type().decode()
            self.kernel_module = KernelModule(bmodel.KernelModule(), binary)
            self.neuron_size = bmodel.NeuronSize()

            self.time = bmodel.Time().decode()
            self.net: List[Net] = FBSArray(bmodel, ("Net", Net), binary)
            self.addr_mode = self.net[0].addr_mode

            self.device_num = bmodel.DeviceNum()

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
        net = self.net.serialize(builder, save_binary)
        kernel_module = self.kernel_module.serialize(builder, save_binary)

        module.Start(builder)
        module.AddType(builder, type_)  # id 0
        module.AddVersion(builder, version)  # id 1
        module.AddTime(builder, time)  # 2
        module.AddChip(builder, chip)  # 3
        module.AddNet(builder, net)  # 4
        module.AddNeuronSize(builder, self.neuron_size)  # 5

        if kernel_module:
            module.AddKernelModule(builder, kernel_module)  # 6

        module.AddDeviceNum(builder, self.device_num)  # 7
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
        # TODO split dynamic ir into multiple
        return [ir_buffer]
