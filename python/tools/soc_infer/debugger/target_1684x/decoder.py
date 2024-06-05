# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from typing import List
import numpy as np
import ctypes
from .regdef import op_class_dic
from .regdef import sDMA_sys_reg as dma_sys, SYSID_reg as tiu_sys
from ..target_common import atomic_reg, DecoderBase, BaseTpuCmd, CMDType, HeadDef
from .opdef import (
    tiu_cls,
    dma_cls,
    tiu_index,
    dma_index,
    TiuCmd,
    DmaCmd,
)


class TiuHead(HeadDef):
    _fields_ = [
        ("cmd_short", ctypes.c_uint64, 1),
        ("cmd_id", ctypes.c_uint64, 20),
        ("cmd_id_dep", ctypes.c_uint64, 20),
        ("tsk_typ", ctypes.c_uint64, 4),
        ("tsk_eu_typ", ctypes.c_uint64, 5),
    ]
    cmd_short: int
    cmd_id: int
    cmd_id_dep: int
    tsk_typ: int
    tsk_eu_typ: int

    @property
    def op_type(self):
        return self.tsk_typ

    @property
    def eu_type(self):
        return self.tsk_eu_typ


class DmaHead(HeadDef):
    _fields_ = [
        ("intr_en", ctypes.c_uint64, 1),
        ("stride_enable", ctypes.c_uint64, 1),
        ("nchw_copy", ctypes.c_uint64, 1),
        ("cmd_short", ctypes.c_uint64, 1),
        ("decompress_enable", ctypes.c_uint64, 1),
        ("cmd_id_en", ctypes.c_uint64, 4),
        ("cmd_id", ctypes.c_uint64, 20),
        ("reserved", ctypes.c_uint64, 3),
        ("cmd_type", ctypes.c_uint64, 4),
        ("cmd_sp_func", ctypes.c_uint64, 3),
    ]
    intr_en: int
    stride_enable: int
    nchw_copy: int
    cmd_short: int
    decompress_enable: int
    cmd_id_en: int
    cmd_id: int
    reserved: int
    cmd_type: int
    cmd_sp_func: int


class Decoder(DecoderBase):
    tiu_head_length = 50
    dma_head_length = 39

    def decode_tiu_cmd(self, reg_buf: memoryview, *, offset, subnet_id) -> BaseTpuCmd:
        head = TiuHead.from_buffer(reg_buf, offset)  # type: TiuHead
        op_info = tiu_index.get(
            (bool(head.cmd_short), head.tsk_typ, head.tsk_eu_typ), None
        )
        assert op_info is not None, (
            f"Unable to decode TIU code at offset {offset} out of {len(reg_buf)} total."
            f" Potential head identified as {head}"
        )

        # get op struct
        op_clazz = op_class_dic[op_info.name]
        reg = self.decode_reg(op_clazz, reg_buf, offset=offset)
        buf = reg_buf[offset : offset + op_clazz.length // 8]
        cmd = op_info(reg, buf=buf, subnet_id=subnet_id)
        return cmd

    def decode_dma_cmd(self, reg_buf: memoryview, *, offset, subnet_id) -> BaseTpuCmd:
        head = DmaHead.from_buffer(reg_buf, offset)  # type: DmaHead
        op_info = dma_index.get(
            (bool(head.cmd_short), head.cmd_type, head.cmd_sp_func), None
    )
        # get op struct
        assert op_info is not None, (
            f"Unable to decode DMA code at offset {offset} out of {len(reg_buf)} total."
            f" Potential head identified as {head}"
        )
        op_clazz = op_class_dic[op_info.name]
        reg = self.decode_reg(op_clazz, reg_buf, offset=offset)
        buf = reg_buf[offset : offset + op_clazz.length // 8]
        cmd = op_info(reg, buf=buf, subnet_id=subnet_id)
        return cmd

    def decode_dma_cmds(
        self, reg_buf: memoryview, subnet_id=0, **_
    ) -> List[BaseTpuCmd]:
        """
        reg_buf: editable memoryview directly passed from bmodel binary buffer
        """
        offset = 0
        res = []
        while offset < len(reg_buf):
            cmd = self.decode_dma_cmd(reg_buf, offset=offset, subnet_id=subnet_id)
            offset += cmd.reg.length // 8
            res.append(cmd)
            if self.buf_is_end(reg_buf[offset:], cmd, dma_sys):
                break
        return res

    def decode_tiu_cmds(
        self, reg_buf: memoryview, subnet_id=0, **_
    ) -> List[BaseTpuCmd]:
        """
        reg_buf: editable memoryview directly passed from bmodel binary buffer
        """
        offset = 0
        res = []
        while offset < len(reg_buf):
            cmd = self.decode_tiu_cmd(reg_buf, offset=offset, subnet_id=subnet_id)
            offset += cmd.reg.length // 8
            res.append(cmd)
            if self.buf_is_end(reg_buf[offset:], cmd, tiu_sys):
                break

        return res

    def buf_is_end(self, reg_buf, operation: BaseTpuCmd, end_op):
        is_sys = isinstance(operation.reg, end_op)
        is_less_1024 = len(reg_buf) * 8 < 1025
        if is_sys and is_less_1024 and not np.any(np.frombuffer(reg_buf, np.uint8)):
            return True
        return False


decoder_instance = Decoder()
