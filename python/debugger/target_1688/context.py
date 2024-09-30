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
from .cmodel import MemRef
from .decoder import Decoder
from typing import List, Type
from .cmodel import BM1688Runner
from ..target_common import *
from .opparam import get_opparam_converter_with_context, opparam_converter


def GET_LMEM_START_ADDR(core_id):
    return 0x25000000 + core_id * info.LMEM_SIZE


class BM1688Context(BModelContext):
    device = Target.BM1688

    memmap = memmap

    dma_sys = dma_sys
    tiu_sys = tiu_sys

    local_layout_to_stride = local_layout_to_stride
    valid_tag = {1: 0, 2: 1, 3:2}  # {tag : corresponding index in self.base_addr}
    base_addr = [2**32, 0x5F11000 + 2**32, GET_LMEM_START_ADDR]

    def __init__(self) -> None:
        super().__init__()
        self.decoder = Decoder(self)

    @property
    @lru_cache()
    def opparam_converter(self):
        return get_opparam_converter_with_context(self, opparam_converter)

    @property
    def MemRef(self) -> Type[MemRef]:
        return partial(MemRef, context=self)

    def get_memory_type(self, reg_address: int) -> MType:
        assert 0 <= reg_address < 2**40
        if reg_address >> 39:
            return MType.G
        else:
            if (reg_address >> 23) & 0x1:
                return MType.S
            else:
                return MType.R

    def fix_addr(self, reg_address: int) -> int:
        # asser reg_address has 40 bits
        assert 0 <= reg_address < 2**40
        if reg_address & (1 << 39):  # GMEM
            tag = (reg_address >> 36) & 0x7
            fixed_addr = self.base_addr[self.valid_tag[tag]] + (
                reg_address & 0x7FFFFFFFF
            )
        else:
            fixed_addr = reg_address & (0xFFFFFF)
        return fixed_addr

    @classmethod
    def merge_instruction(cls, tiu: List[BaseTpuCmd], dma: List[BaseTpuCmd]):
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

    def get_runner(self, memory_size: int) -> Runner:
        from .cmodel import BM1688Runner as BM1688CModel
        from .device_rt import BM1688Runner as BM1688SOC

        if self.using_cmodel:
            if self._runner is None:
                self._runner = BM1688CModel(memory_size, self.base_addr)
            runner = self._runner
        else:
            if self._runner is None:
                self._runner = BM1688SOC(memory_size)
            runner = self._runner
        return runner
