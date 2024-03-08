# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
#import disassembler as dis

#from .target_common import use_backend


import numpy as np
import enum
from typing import List
from pprint import pformat
from .target_common import (
    DType,
    Layout,
    MemRefBase,
    atomic_reg,
    get_target_context,
    CpuLayerType,
    BaseTpuCmd,
    BaseCmd,
    use_backend,
)
from .target_1690.opdef import DmaCmd

class EngineType(enum.Enum) :
    TPU  = 1
    GDMA = 2
    SDMA = 3
    HAU  = 4
    Engine_TYPE_END = 5

class BaseCmd:
    buf: bytes
    def __init__(self, buf, type):
        self.buf = buf
        self.cmd_type = type
    def get_cmd_type(self) -> int:
        return self.cmd_type
    def get_cmd_buff(self) -> bytes:
        return self.buf

def decode_cmds(chip : str, cmds_dict : dict) -> dict:
    res = []
    decoded_cmds = {}
    context = get_target_context(chip)
    decoder = context.decoder
    cmd_gdma_id = 0
    cmd_sdma_id = 0
    cmd_tpu_id = 0
    core_id = -1

    for core_id in cmds_dict:
        for cmd in cmds_dict[core_id]:
            t = cmd.get_cmd_type()
            cmd_arry = cmd.get_cmd_buff()
            if(t == EngineType.TPU):
                tiu = decoder.decode_cmds(cmd_arry, core_id,  cmd_tpu_id, t)
                cmd_tpu_id += 1
                res.append([ t, tiu])
            elif(t == EngineType.GDMA):
                dma = decoder.decode_cmds(cmd_arry, core_id,  cmd_gdma_id, t)
                cmd_gdma_id += 1
                res.append([ t, dma])
            elif(t == EngineType.SDMA):
                dma = decoder.decode_cmds(cmd_arry, core_id,  cmd_sdma_id, t)
                cmd_sdma_id += 1
                res.append([ t, dma])
            elif(t == EngineType.HAU):
                continue
            else:
                assert(0)
        decoded_cmds[core_id] = res.copy()
        res.clear()
    return decoded_cmds

