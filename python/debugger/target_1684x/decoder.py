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
from ..target_common import cmd_base_reg, DecoderBase, BaseTpuOp, CMDType
from .opdef import (
    tiu_cls,
    dma_cls,
    tiu_index,
    dma_index,
    TiuCmdOp,
    DmaCmdOp,
)


class TiuHead(ctypes.Structure):
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


class DmaHead(ctypes.Structure):
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

    def decode_tiu_cmd(
        self, cmd_buf: memoryview, *, offset=0, core_id=0
    ) -> cmd_base_reg:
        head = TiuHead.from_buffer(cmd_buf, offset)  # type: TiuHead
        op_info = tiu_index.get((head.cmd_short, head.tsk_typ, head.tsk_eu_typ), None)

        # get op struct
        op_clazz = op_class_dic[op_info.op_name]
        return self.decode_cmd(op_clazz, cmd_buf, offset=offset, core_id=core_id)

    def decode_dma_cmd(
        self, cmd_buf: memoryview, *, offset=0, core_id=0
    ) -> cmd_base_reg:
        head = DmaHead.from_buffer(cmd_buf, offset)  # type: DmaHead
        op_info = dma_index.get((head.cmd_short, head.cmd_type, head.cmd_sp_func), None)
        # get op struct
        op_clazz = op_class_dic[op_info.op_name]
        return self.decode_cmd(op_clazz, cmd_buf, offset=offset, core_id=core_id)

    def decode_dma_cmds(self, cmd_buf: bytes, core_id=0) -> List[cmd_base_reg]:
        cmd_buf = memoryview(bytearray(cmd_buf))
        offset = 0
        res = []
        while offset < len(cmd_buf):
            cmd = self.decode_dma_cmd(cmd_buf, offset=offset, core_id=core_id)
            offset += cmd.length // 8
            res.append(cmd)
            if self.buf_is_end(cmd_buf[offset:], cmd, dma_sys):
                break
        return res

    def decode_tiu_cmds(self, cmd_buf: bytes, core_id=0) -> List[cmd_base_reg]:
        cmd_buf = memoryview(bytearray(cmd_buf))
        offset = 0
        res = []
        while offset < len(cmd_buf):
            cmd = self.decode_tiu_cmd(cmd_buf, offset=offset, core_id=core_id)
            offset += cmd.length // 8
            res.append(cmd)
            if self.buf_is_end(cmd_buf[offset:], cmd, tiu_sys):
                break

        return res

    def decode_cmd_params(self, cmd: cmd_base_reg) -> BaseTpuOp:
        cmd_type = self.get_cmd_type(cmd)
        if cmd_type == CMDType.tiu:
            return TiuCmdOp(cmd)
        elif cmd_type == CMDType.dma:
            return DmaCmdOp(cmd)
        raise NotImplementedError()

    def get_cmd_type(self, cmd: cmd_base_reg) -> CMDType:
        if cmd.OP_NAME in tiu_cls:
            return CMDType.tiu
        elif cmd.OP_NAME in dma_cls:
            return CMDType.dma
        else:
            return CMDType.unknown

    def is_end(self, cmd: cmd_base_reg):
        return isinstance(cmd, (dma_sys, tiu_sys))

    def buf_is_end(self, cmd_buf, operation: cmd_base_reg, end_op):
        is_sys = operation.__class__ == end_op
        is_less_1024 = len(cmd_buf) * 8 < 1025
        if is_sys and is_less_1024 and not np.any(np.frombuffer(cmd_buf, np.uint8)):
            return True
        return False


decoder_instance = Decoder()
