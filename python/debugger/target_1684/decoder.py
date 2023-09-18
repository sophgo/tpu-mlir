# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
# 1684
import ctypes
from .opdef import tiu_cls, tiu_index, DmaCmdOp, TiuCmdOp, BaseTpuOp
import numpy as np
from typing import List
from .regdef import (
    op_class_dic,
    tiu_high_bits,
    dma_high_bits,
)
from ..target_common import cmd_base_reg, DecoderBase, CMDType


class TiuHead(ctypes.Structure):
    _fields_ = [
        ("reserve", ctypes.c_uint64, 37),
        ("tsk_typ", ctypes.c_uint64, 4),
        ("tsk_eu_typ", ctypes.c_uint64, 5),
    ]

    tsk_typ: int
    tsk_eu_typ: int

    @property
    def op_type(self):
        return self.tsk_typ

    @property
    def eu_type(self):
        return self.tsk_eu_typ


# cache for convert binary to unsigned integer.
_TABLE = 2 ** np.arange(64, dtype=np.uint64)


def packbits(arr: np.ndarray):
    if arr.size > 64:
        return 0
    return int(arr.dot(_TABLE[: arr.size]))


def buffer_to_bits(buffer):
    cmmand_buf = np.frombuffer(buffer, dtype=np.uint8)
    return np.unpackbits(cmmand_buf, bitorder="little")


class Decoder(DecoderBase):
    def decode_tiu_cmd(self, cmd_buf: memoryview, core_id=0, offset=0) -> cmd_base_reg:
        head = TiuHead.from_buffer(cmd_buf, offset)  # type: TiuHead
        op_info = tiu_index.get((head.tsk_typ, head.tsk_eu_typ), None)

        # get op struct
        op_clazz = op_class_dic[op_info.op_name]
        buffer = buffer_to_bits(cmd_buf[offset : offset + op_clazz.length // 8])
        bits_sec = np.split(buffer, tiu_high_bits)  # slow
        # import pdb; pdb.set_trace()
        values = [packbits(x) for x in bits_sec]  # slow
        res = op_clazz.from_values(values)  # type: cmd_base_reg
        res.buf = cmd_buf[offset : offset + op_clazz.length // 8]
        return res

    def decode_dma_cmd(
        self, cmd_buf: memoryview, *, core_id=0, offset=0
    ) -> cmd_base_reg:
        # get op struct
        op_clazz = op_class_dic["dma_tensor"]

        buffer = buffer_to_bits(cmd_buf[offset : offset + op_clazz.length // 8])
        bits_sec = np.split(buffer, dma_high_bits)  # slow
        values = [packbits(x) for x in bits_sec]  # slow
        res = op_clazz.from_values(values)  # type: cmd_base_reg
        res.buf = cmd_buf[offset : offset + op_clazz.length // 8]
        return res

    def decode_dma_cmds(self, cmd_buf: bytes, *, core_id=0) -> List[cmd_base_reg]:
        raw_size = len(cmd_buf)
        cmd_buf = bytearray(cmd_buf)
        cmd_buf = memoryview(cmd_buf)
        offset = 0
        res = []
        while len(cmd_buf) - offset > 0:
            cmd = self.decode_dma_cmd(cmd_buf, offset=offset, core_id=core_id)
            offset += cmd.length // 8
            res.append(cmd)
            if self.buf_is_end(cmd_buf[offset:raw_size]):
                break
        return res

    def decode_tiu_cmds(self, cmd_buf: bytes, core_id=0) -> List[cmd_base_reg]:
        raw_size = len(cmd_buf)
        cmd_buf = bytearray(cmd_buf)
        cmd_buf = memoryview(cmd_buf)
        offset = 0
        res = []
        while offset < len(cmd_buf):
            cmd = self.decode_tiu_cmd(cmd_buf, offset=offset, core_id=core_id)
            offset += cmd.length // 8
            res.append(cmd)

            if self.buf_is_end(cmd_buf[offset:raw_size]):
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
        else:
            return CMDType.dma

    def is_end(self, cmd: cmd_base_reg):
        return False

    def buf_is_end(self, cmd_buf):
        is_less_1024 = len(cmd_buf) * 8 < 1025
        if is_less_1024:
            if not np.any(np.unpackbits(cmd_buf, bitorder="little")):
                return True
        return False


decoder_instance = Decoder()
