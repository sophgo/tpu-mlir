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
from .opdef import tiu_cls, tiu_index, DmaCmd, TiuCmd, BaseTpuCmd
import numpy as np
from typing import List
from .regdef import (
    op_class_dic,
    tiu_high_bits,
    dma_high_bits,
)
from ..target_common import (
    atomic_reg,
    DecoderBase,
)


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
    def decode_tiu_cmd(self, reg_buf: memoryview, offset, subnet_id) -> TiuCmd:
        head = TiuHead.from_buffer(reg_buf, offset)  # type: TiuHead
        op_info = tiu_index.get((head.tsk_typ, head.tsk_eu_typ), None)

        # get op struct
        op_clazz = op_class_dic[op_info.op_name]
        buffer = buffer_to_bits(reg_buf[offset : offset + op_clazz.length // 8])
        bits_sec = np.split(buffer, tiu_high_bits)  # slow
        values = [packbits(x) for x in bits_sec]  # slow
        reg = op_clazz.from_values(values)  # type: atomic_reg
        cmd = TiuCmd(
            reg,
            buf=reg_buf[offset : offset + op_clazz.length // 8],
            subnet_id=subnet_id,
        )
        return cmd

    def decode_dma_cmd(self, reg_buf: memoryview, *, offset, subnet_id) -> atomic_reg:
        # get op struct
        op_clazz = op_class_dic["dma_tensor"]

        buffer = buffer_to_bits(reg_buf[offset : offset + op_clazz.length // 8])
        bits_sec = np.split(buffer, dma_high_bits)  # slow
        values = [packbits(x) for x in bits_sec]  # slow
        reg = op_clazz.from_values(values)  # type: atomic_reg

        cmd = DmaCmd(
            reg,
            buf=reg_buf[offset : offset + op_clazz.length // 8],
            subnet_id=subnet_id,
        )
        return cmd

    def decode_dma_cmds(
        self,
        reg_buf: memoryview,
        *,
        subnet_id=0,
        **_,
    ) -> List[BaseTpuCmd]:
        raw_size = len(reg_buf)
        offset = 0
        res = []
        while len(reg_buf) - offset > 0:
            cmd = self.decode_dma_cmd(reg_buf, offset=offset, subnet_id=subnet_id)
            offset += cmd.reg.length // 8
            res.append(cmd)
            if self.buf_is_end(reg_buf[offset:raw_size]):
                break
        return res

    def decode_tiu_cmds(
        self,
        reg_buf: memoryview,
        *,
        subnet_id=0,
        **_,
    ) -> List[BaseTpuCmd]:
        raw_size = len(reg_buf)
        offset = 0
        res = []
        while offset < len(reg_buf):
            cmd = self.decode_tiu_cmd(reg_buf, offset=offset, subnet_id=subnet_id)
            offset += cmd.reg.length // 8
            res.append(cmd)

            if self.buf_is_end(reg_buf[offset:raw_size]):
                break

        return res

    @staticmethod
    def is_sys(reg: atomic_reg):
        return False

    @staticmethod
    def buf_is_end(buf: memoryview):
        is_less_1024 = len(buf) * 8 < 1025
        if is_less_1024:
            if not np.any(np.unpackbits(buf, bitorder="little")):
                return True
        return False


decoder_instance = Decoder()
