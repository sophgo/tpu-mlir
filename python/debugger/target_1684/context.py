# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from typing import List
from ..target_common import *
from .cmodel import BM1684Runner
from .memmap import memmap, get_memory_type, local_layout_to_stride, MemRef
from .decoder import decoder_instance


class BM1684Context(BModelContext):
    MemRef = MemRef
    device = Target.BM1684
    decoder = decoder_instance

    memmap = memmap

    dma_sys = None
    tiu_sys = None

    get_memory_type = get_memory_type
    local_layout_to_stride = local_layout_to_stride

    @classmethod
    def merge_instruction(
        cls, tiu: List[BaseTpuCmd], dma: List[BaseTpuCmd]
    ) -> List[BaseTpuCmd]:
        main_cmd, inserted_cmd = dma, tiu

        # remove the system command
        def get_end(cmd):
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
        assert self.using_cmodel, "1684 currently only support cmodel mode"
        if self._runner is None:
            self._runner = BM1684Runner(memory_size)
        return self._runner

    @classmethod
    def is_sys(cls, _: BaseTpuCmd):
        return False
