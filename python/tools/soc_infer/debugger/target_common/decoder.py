# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import ctypes
from typing import List, Type
from .op_support import (
    atomic_reg,
    BaseTpuCmd,
    CpuCmd,
    CpuLayerType,
    MemRefBase,
    CMDType,
    DynIrCmd,
)


class HeadDef(ctypes.Structure):
    def __repr__(self):
        return str(dict(self))

    def __iter__(self):
        for field in self._fields_:
            yield (field[0], getattr(self, field[0]))

    def __eq__(self, other):
        return hash(self) == hash(other)



class DecoderBase:
    tiu_head_length = None
    dma_head_length = None

    @staticmethod
    def decode_reg(clazz: Type[atomic_reg], buf: memoryview, *, offset=0) -> atomic_reg:
        res = clazz.from_buffer(buf, offset)  # type: atomic_reg
        return res

    def decode_tiu_cmd(
        self,
        reg_buf: memoryview,
        *,
        offset,
        subnet_id,
        **kwargs,
    ) -> BaseTpuCmd:
        """
        The optional offset parameter specifies an offset into the source buffer in bytes. (for ctypes.Structure.from_buffer)
        """
        raise NotImplementedError()

    def decode_dma_cmd(
        self,
        reg_buf: memoryview,
        *,
        offset,
        subnet_id,
        **kwargs,
    ) -> BaseTpuCmd:
        """
        The optional offset parameter specifies an offset into the source buffer in bytes. (for ctypes.Structure.from_buffer)
        """
        raise NotImplementedError()

    def decode_dma_cmds(self, reg_buf: bytes, *, subnet_id, **kw) -> List[BaseTpuCmd]:
        raise NotImplementedError()

    def decode_tiu_cmds(self, reg_buf: bytes, *, subnet_id, **kw) -> List[BaseTpuCmd]:
        raise NotImplementedError()
    def decode_cmds(self, cmd_arry: bytes, core_id: int, cmd_id: int, t: int) -> list:
        raise NotImplementedError()
    def decode_cpu_cmd(
        self,
        op_type: CpuLayerType,
        buf: bytes,
        input_memref: List[MemRefBase] = None,
        output_memref: List[MemRefBase] = None,
        subnet_id: int = 0,
        cmd_id: int = 0,
    ) -> CpuCmd:
        return CpuCmd(
            op_type,
            buf,
            buf_size=len(buf),
            input_memref=input_memref,
            output_memref=output_memref,
            subnet_id=subnet_id,
            cmd_id=cmd_id,
        )

    def decode_ir_cmd(
        self,
        ir_buf: bytes,
        ir_size: int,
        input_memref: List[MemRefBase] = None,
        output_memref: List[MemRefBase] = None,
        subnet_id: int = 0,
        cmd_id: int = 0,
        ctx_addr: int = 0,
        ctx_size: int = 0,
    ) -> DynIrCmd:
        return DynIrCmd(
            ir_buf,
            ir_size,
            input_memref,
            output_memref,
            cmd_id,
            subnet_id,
            ctx_addr=ctx_addr,
            ctx_size=ctx_size,
        )

    @staticmethod
    def buf_is_end(*args, **kwargs):
        raise NotImplementedError()
