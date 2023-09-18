# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from typing import List, Type
from .op_support import (
    cmd_base_reg,
    BaseTpuOp,
    CpuOp,
    CpuLayerType,
    MemRefBase,
    CMDType,
    DynIrOp,
)


class DecoderBase:
    tiu_head_length = None
    dma_head_length = None

    @staticmethod
    def decode_cmd(
        clazz: Type[cmd_base_reg], buf: memoryview, *, offset=0, core_id=0, cmd_id=None
    ) -> cmd_base_reg:
        res = clazz.from_buffer(buf, offset)  # type: cmd_base_reg

        # hack for 1688
        if cmd_id is not None:
            res.cmd_id = cmd_id
        # bind buf for bug checking
        res.core_id = core_id
        res.buf = buf[offset : offset + clazz.length]
        return res

    def decode_tiu_cmd(
        self, cmd_buf: memoryview, *, offset=0, core_id=0, cmd_id=None
    ) -> cmd_base_reg:
        """
        The optional offset parameter specifies an offset into the source buffer in bytes. (for ctypes.Structure.from_buffer)
        """
        raise NotImplementedError()

    def decode_dma_cmd(
        self, cmd_buf: memoryview, *, offset=0, core_id=0, cmd_id=None
    ) -> cmd_base_reg:
        """
        The optional offset parameter specifies an offset into the source buffer in bytes. (for ctypes.Structure.from_buffer)
        """
        raise NotImplementedError()

    def decode_dma_cmds(self, cmd_buf: bytes, *, core_id=0) -> List[cmd_base_reg]:
        raise NotImplementedError()

    def decode_tiu_cmds(self, cmd_buf: bytes, *, core_id=0) -> List[cmd_base_reg]:
        raise NotImplementedError()

    def decode_cmd_params(self, cmd: cmd_base_reg) -> BaseTpuOp:
        raise NotImplementedError()

    def decode_cpu_params(
        self,
        op_type: CpuLayerType,
        param: bytes,
        input_memref: List[MemRefBase] = None,
        output_memref: List[MemRefBase] = None,
        subnet_id: int = 0,
        cmd_id: int = 0,
    ) -> CpuOp:
        return CpuOp(
            op_type,
            param,
            param_size=len(param),
            input_memref=input_memref,
            output_memref=output_memref,
            subnet_id=subnet_id,
            cmd_id=cmd_id,
        )

    def decode_ir_param(
        self,
        ir_buf: bytes,
        ir_size: int,
        input_memref: List[MemRefBase] = None,
        output_memref: List[MemRefBase] = None,
        subnet_id: int = 0,
        cmd_id: int = 0,
    ) -> DynIrOp:
        return DynIrOp(ir_buf, ir_size, input_memref, output_memref, cmd_id, subnet_id)

    def get_cmd_type(self, cmd: cmd_base_reg) -> CMDType:
        return CMDType.unknown

    def is_end(self, cmd: cmd_base_reg):
        raise NotImplementedError()
