# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

"""
regdef.py provide a wrap class for regdef.py,
for better code struct and IDE auto-completation.
"""
from typing import List, Tuple, Dict, TYPE_CHECKING
import numpy as np
import ctypes

from .regdef import op_class_dic
from .regdef import sDMA_sys_reg as dma_sys, SYS_reg as tiu_sys
from ..target_common import (
    cmd_base_reg,
    CMDType,
    DecoderBase,
    BaseTpuOp,
    OpInfo,
)
from .opdef import tiu_cls, dma_cls, TiuCmdOp, DmaCmdOp

if TYPE_CHECKING:
    from .context import BM1688Context


# build tiu and dma search tree
# search by cmd_short, tsk_typ, tsk_eu_type
tiu_index: Dict[Tuple[int, int, int], OpInfo] = {}

for k, v in tiu_cls.items():
    if len(v["tsk_eu_typ"]) == 0:
        tsk_eu_typ = {0: "none"}
    else:
        tsk_eu_typ = v["tsk_eu_typ"]

    if isinstance(tsk_eu_typ, range):
        tsk_eu_typ = {i: f"ana_{i}" for i in tsk_eu_typ}

    for eu_type, eu_name in tsk_eu_typ.items():
        if v["short_cmd"] is None:
            v["short_cmd"] = 0
        tiu_index[(int(v["short_cmd"]), v["tsk_typ"], eu_type)] = OpInfo(k, eu_name)


# search by cmd_short, tsk_typ, sp_fun(special function)
dma_index: Dict[Tuple[int, int, int], OpInfo] = {}
for k, v in dma_cls.items():
    if len(v["sp_fun"]) == 0:
        sp_fun = {0: "none"}
    else:
        sp_fun = v["sp_fun"]

    for sp_typ, sp_name in sp_fun.items():
        dma_index[(int(v["short_cmd"]), v["tsk_typ"], sp_typ)] = OpInfo(k, sp_name)


class TiuHead(ctypes.Structure):
    _fields_ = [
        ("cmd_short", ctypes.c_uint64, 1),
        ("reserved", ctypes.c_uint64, 40),
        ("tsk_typ", ctypes.c_uint64, 4),
        ("tsk_eu_typ", ctypes.c_uint64, 5),
    ]
    cmd_short: int
    tsk_typ: int
    tsk_eu_typ: int

    @property
    def op_type(self):
        return self.tsk_typ

    @property
    def eu_type(self):
        return self.tsk_eu_typ


class SYS_TR_ACC_HEAD(ctypes.Structure):
    """
    SYS_TR_ACC has different tsk_eu_typ bits with other tiu cmds
    """

    _fields_ = [
        ("cmd_short", ctypes.c_uint64, 1),
        ("reserved", ctypes.c_uint64, 40),
        ("tsk_typ", ctypes.c_uint64, 4),
        ("tsk_eu_typ", ctypes.c_uint64, 3),
    ]
    cmd_short: int
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
        ("reserved", ctypes.c_uint64, 28),
        ("cmd_type", ctypes.c_uint64, 4),
        ("cmd_sp_func", ctypes.c_uint64, 3),
    ]
    intr_en: int
    stride_enable: int
    nchw_copy: int
    cmd_short: int
    reserved: int
    cmd_type: int
    cmd_sp_func: int


TiuHeads: List[ctypes.Structure] = [TiuHead, SYS_TR_ACC_HEAD]


class Decoder(DecoderBase):
    tiu_head_length = 50
    dma_head_length = 39

    def __init__(self, context: "BM1688Context") -> None:
        super().__init__()
        self.context = context

    def decode_tiu_cmd(
        self, cmd_buf: memoryview, *, offset=0, core_id=0, cmd_id=None
    ) -> cmd_base_reg:
        assert cmd_id is not None, "1688 must assign cmd_id manully"
        for head_cls in TiuHeads:  # type: cmd_base_t
            head = head_cls.from_buffer(cmd_buf, offset)  # type: TiuHead
            op_info = tiu_index.get(
                (head.cmd_short, head.tsk_typ, head.tsk_eu_typ), None
            )
            if op_info is not None:
                break

        # get op struct
        op_clazz = op_class_dic[op_info.op_name]
        return self.decode_cmd(
            op_clazz, cmd_buf, offset=offset, core_id=core_id, cmd_id=cmd_id
        )

    def decode_dma_cmd(
        self, cmd_buf: memoryview, *, offset=0, core_id=0, cmd_id=None
    ) -> cmd_base_reg:
        assert cmd_id is not None, "1688 must assign cmd_id manully"
        head = DmaHead.from_buffer(cmd_buf, offset)  # type: DmaHead
        op_info = dma_index.get((head.cmd_short, head.cmd_type, head.cmd_sp_func), None)
        # get op struct
        op_clazz = op_class_dic[op_info.op_name]
        return self.decode_cmd(
            op_clazz, cmd_buf, offset=offset, core_id=core_id, cmd_id=cmd_id
        )

    def decode_dma_cmds(self, cmd_buf: bytes, core_id=0) -> List[cmd_base_reg]:
        cmd_buf = memoryview(bytearray(cmd_buf))
        offset = 0
        res = []
        cmd_id = 1
        while offset < len(cmd_buf):
            cmd = self.decode_dma_cmd(
                cmd_buf,
                offset=offset,
                core_id=core_id,
                cmd_id=cmd_id,
            )
            cmd_id += 1
            offset += cmd.length // 8
            res.append(cmd)
            if self.buf_is_end(cmd_buf[offset:], cmd, dma_sys):
                break
        return res

    def decode_tiu_cmds(self, cmd_buf: bytes, core_id=0) -> List[cmd_base_reg]:
        cmd_buf = memoryview(bytearray(cmd_buf))
        offset = 0
        res = []
        cmd_id = 1
        while offset < len(cmd_buf):
            cmd = self.decode_tiu_cmd(
                cmd_buf,
                offset=offset,
                core_id=core_id,
                cmd_id=cmd_id,
            )
            cmd_id += 1
            offset += cmd.length // 8
            res.append(cmd)
            if self.buf_is_end(cmd_buf[offset:], cmd, tiu_sys):
                break

        return res

    def decode_cmd_params(self, cmd: cmd_base_reg) -> BaseTpuOp:
        cmd_type = self.get_cmd_type(cmd)
        if cmd_type == CMDType.tiu:
            TiuCmdOp.opparam_converter = self.context.opparam_converter
            return TiuCmdOp(cmd)
        elif cmd_type == CMDType.dma:
            DmaCmdOp.opparam_converter = self.context.opparam_converter
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


# decoder_instance = Decoder()
