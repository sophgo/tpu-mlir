# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from functools import partial, lru_cache
from .regdef import sDMA_sys_reg as dma_sys, SYS_reg as tiu_sys, SYS_TR_ACC_reg
from .memmap import *
from .cmodel import MemRef
from .decoder import Decoder
from typing import List, Type
from .cmodel import BM1688Runner
from ..target_common import *
from .opparam import get_opparam_converter_with_context, opparam_converter


class BM1688Context(CModelContext):
    device = Device.BM1688

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

    @property
    def MemRef(self) -> Type[MemRef]:
        return partial(MemRef, context=self)

    def get_memory_type(self, address: int) -> MType:
        address = self.fix_tag(address)
        for k, v in memmap.items():
            if address >= v[0] and address < v[1]:
                return k
        return MType.UNKNOWN

    def fix_tag(self, address: int) -> int:
        fixed_addr = address
        if address & (1 << 39):
            base_addr_idx = (address >> 36) & 0x7
            fixed_addr = (address + self.base_addr[base_addr_idx - 1]) & ((1 << 35) - 1)
        return fixed_addr

    @staticmethod
    def merge_instruction(tiu: List[cmd_base_reg], dma: List[cmd_base_reg]):
        main_cmd, inserted_cmd = dma, tiu
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
            self._runner = BM1688Runner(memory_size)
        return self._runner
