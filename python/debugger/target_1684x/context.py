# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import os
from typing import List
from ..target_common import *
from .cmodel import BM1684XRunner as BM1684XCModel
from .pcie import BM1684XRunner as BM1684XPcie
from .decoder import decoder_instance
from .memmap import memmap, MemRef
from .regdef import sDMA_sys_reg as dma_sys, SYSID_reg as tiu_sys
from .memmap import *


class BM1684XContext(BModelContext):
    MemRef = MemRef
    device = Target.BM1684X
    decoder = decoder_instance

    memmap = memmap

    dma_sys = dma_sys
    tiu_sys = tiu_sys

    get_memory_type = get_memory_type
    local_layout_to_stride = local_layout_to_stride

    @staticmethod
    def merge_instruction(
        tiu: List[cmd_base_reg], dma: List[cmd_base_reg]
    ) -> List[cmd_base_reg]:
        main_cmd, inserted_cmd = dma, tiu

        # remove the system command
        def get_end(cmd):
            if len(cmd) == 0:
                return 0

            sys = (tiu_sys, dma_sys)
            if isinstance(cmd[-1], sys):
                return -1
            else:
                return len(cmd)

        # remove system instruction
        main_id = [(m.cmd_id, m) for m in main_cmd[: get_end(main_cmd)]]
        inserted_id = [(i.cmd_id_dep, i) for i in inserted_cmd[: get_end(inserted_cmd)]]
        # "sorted" is stable, which keeps the inserted commands
        # after the main instructions.
        cmd = main_id + inserted_id
        cmd_sorted = sorted(cmd, key=lambda x: x[0])
        return [x[1] for x in cmd_sorted]

    def get_runner(self, memory_size: int) -> CModelRunner:
        if self.using_cmodel:
            if self._cmodel_runner is None:
                self._cmodel_runner = BM1684XCModel(memory_size)
            runner = self._cmodel_runner
        else:
            if self._chip_runner is None:
                self._chip_runner = BM1684XPcie(memory_size)
            runner = self._chip_runner

        return runner
