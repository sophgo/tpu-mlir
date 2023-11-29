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
from .cmodel import SG2260Runner
from ..target_common import *
from .opparam import get_opparam_converter_with_context, opparam_converter


class SG2260Context(BModelContext):
    device = Target.SG2260

    memmap = memmap

    dma_sys = dma_sys
    tiu_sys = tiu_sys

    local_layout_to_stride = local_layout_to_stride

    def __init__(self) -> None:
        super().__init__()
        self.base_addr = [memmap[MType.G][0] for _ in range(2)]
        self.decoder = Decoder(self)

    @property
    @lru_cache()
    def opparam_converter(self):
        return get_opparam_converter_with_context(self, opparam_converter)

    # @property
    def MemRef(self, address, shape, dtype, stride, layout, cmd_type="tiu"):
        return MemRef(address, shape, dtype, stride, layout, SG2260Context, cmd_type)

    def get_tiu_memory_type(self, address) -> MType:
        # tiu addr is 32 bits
        if address >> 26:
            if np.binary_repr(address)[-27]:
                return MType.S
            else:
                return MType.R
        return MType.R

    def get_dma_memory_type(self, address) -> MType:
        tag = (address >> 40) & 0x1F  # tag
        if 0 <= tag <= 30:
            return MType.G
        elif tag == 31:
            if np.binary_repr(address)[-27]:
                return MType.S
            else:
                return MType.R
        return MType.UNKNOWN

    @staticmethod
    def cmd_base_reg2dma_base_reg(cmdBaseRegList):
        for cmdBaseReg in cmdBaseRegList:
            origin_fields = cmdBaseReg._fields_
            for field in origin_fields:
                value = getattr(cmdBaseReg, field[0])
                if field[0] == "cmd_id_dep":
                    if value > 2**16:
                        value = value & 0x0FFFF
                    else:
                        value = 0
                    setattr(cmdBaseReg, "cmd_id_dep", value)

    @staticmethod
    def merge_instruction(
        tiu: List[cmd_base_reg], dma: List[cmd_base_reg], transfer_dep_id: bool
    ):
        main_cmd, inserted_cmd = dma, tiu
        if transfer_dep_id:
            SG2260Context.cmd_base_reg2dma_base_reg(main_cmd)
            SG2260Context.cmd_base_reg2dma_base_reg(inserted_cmd)

        # remove the system command

        def get_end(cmd: List[cmd_base_reg]):
            if len(cmd) == 0:
                return 0
            if isinstance(cmd[-1], (tiu_sys, dma_sys)):
                return -1
            else:
                return len(cmd)

        def fix_tgcr_cmd_id_dp(tiu_cmd: List[cmd_base_reg]):
            for i, v in enumerate(tiu_cmd):
                if isinstance(v, SYS_TR_ACC_reg):
                    # same as v.op_code == 12, changed because short cmd do not have op_code
                    v.cmd_id_dep = (
                        tiu_cmd[i + 1].cmd_id_dep
                        if tiu_cmd[i + 1].cmd_id_dep != 0
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

    def get_runner(self, memory_size: int) -> CModelRunner:
        if self._runner is None:
            self._runner = SG2260Runner(memory_size)
        return self._runner
