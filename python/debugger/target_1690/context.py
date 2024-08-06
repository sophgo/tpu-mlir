# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from functools import partial, lru_cache

import numpy as np
from .regdef import sDMA_sys_reg as dma_sys, SYS_reg as tiu_sys, SYS_TR_ACC_reg
from .memmap import *
from .decoder import Decoder
from typing import List, Type
from .cmodel import BM1690Runner
from ..target_common import *
from .opparam import get_opparam_converter_with_context, opparam_converter


def GET_LMEM_START_ADDR(core_id):
    return 0x6900000000 + core_id * info.LMEM_SIZE


class BM1690Context(BModelContext):
    device = Target.BM1690

    memmap = memmap

    dma_sys = dma_sys
    tiu_sys = tiu_sys

    local_layout_to_stride = local_layout_to_stride
    valid_tag = {1: 0, 2: 1}  # {tag : corresponding index in self.base_addr}
    base_addr = [0, 0x5F11000, GET_LMEM_START_ADDR]

    def __init__(self) -> None:
        super().__init__()
        self.decoder = Decoder(self)
        self._runner = None

    @property
    @lru_cache()
    def opparam_converter(self):
        return get_opparam_converter_with_context(self, opparam_converter)

    @property
    def MemRef(self) -> Type[MemRef]:
        return partial(MemRef, context=self)

    def get_memory_type(self, reg_address):
        # tag,
        # for dma ops, tags always in {1, 2, 31}, {1, 2} are GMEM, {31} is LMEM
        # for tiu ops, tags always equal to 0, which represents LMEM
        tag = (reg_address >> 40) & 0x1F
        if tag in (1, 2):
            return MType.G
        elif tag == 0:
            return MType.R
        elif tag == 30:
            return MType.L
        elif tag == 31:
            if np.binary_repr(reg_address)[-27] == "1":
                return MType.S
            else:
                return MType.R
        elif tag == 30:
            return MType.L
        return MType.UNKNOWN

    def fix_addr(self, reg_address: int) -> int:
        # asser reg_address has 45 bits
        assert 0 <= reg_address < 2**45
        tag = (reg_address >> 40) & 0x1F
        if tag == 31:  # when tag==31, return the 26bits of reg_address as local offset
            return reg_address & 0x3FFFFFF
        elif tag == 30:
            return reg_address & 0xFFFFFFFFFF
        fixed_addr = self.base_addr[self.valid_tag[tag]] + (reg_address & 0xFFFFFFFFFF)
        return fixed_addr

    @classmethod
    def merge_instruction(
        cls, tiu: List[BaseTpuCmd], dma: List[BaseTpuCmd]
    ) -> List[BaseTpuCmd]:
        main_cmd, inserted_cmd = dma, tiu

        # remove the system command
        def get_end(cmds: List[BaseTpuCmd]):
            if len(cmds) == 0:
                return 0

            if cls.is_sys(cmds[-1]):
                return -1
            else:
                return len(cmds)

        def fix_tgcr_cmd_id_dp(tiu_cmd: List[BaseTpuCmd]):
            for i, v in enumerate(tiu_cmd):
                if isinstance(v.reg, SYS_TR_ACC_reg):
                    # same as v.op_code == 12, changed because short cmd do not have op_code
                    v.cmd_id_dep = (
                        tiu_cmd[i + 1].cmd_id_dep
                        if tiu_cmd[i + 1].cmd_id_dep != None
                        else tiu_cmd[i + 2].cmd_id_dep
                    )

        fix_tgcr_cmd_id_dp(inserted_cmd[: get_end(inserted_cmd)])
        # remove system instruction
        main_id = [(m.cmd_id, m) for m in main_cmd[: get_end(main_cmd)]]
        inserted_id = [(i.cmd_id_dep, i) for i in inserted_cmd[: get_end(inserted_cmd)]]
        # "sorted" is stable, which keeps the inserted commands
        # after the main instructions.
        cmd = main_id + inserted_id
        cmd_sorted = sorted(cmd, key=lambda x: x[0])

        return [x[1] for x in cmd_sorted]

    @classmethod
    def is_sys(cls, cmd: BaseTpuCmd):
        return isinstance(cmd.reg, (dma_sys, tiu_sys))

    def get_runner(self, memory_size: int) -> CModelRunner:
        assert self.using_cmodel, "2260 currently only support cmodel mode"
        if self._runner is None:
            self._runner = BM1690Runner(memory_size, self.base_addr)
        return self._runner
