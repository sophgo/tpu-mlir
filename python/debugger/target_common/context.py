# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import os
from .runner import MemRefBase, MemoryBase, Runner
from contextlib import contextmanager
import numpy as np
from typing import Dict, Tuple, List, Type
from .decoder import DecoderBase
from .op_support import atomic_reg, MType, Target, BaseTpuCmd
import platform


class BModelContext:
    base_addr = [0, 0]  # special defined for 1688
    MemRef = MemRefBase
    device: Target
    decoder: DecoderBase

    memmap: Dict[MType, Tuple[int, int]]
    dma_sys: atomic_reg = None
    tiu_sys: atomic_reg = None

    def __init__(self) -> None:
        self.using_cmodel = eval(
            os.environ.get("USING_CMODEL", "True"))

        if platform.machine() == 'aarch64':
            self.using_cmodel = False

        self._runner: Runner = None

    @staticmethod
    def get_continuous_stride(shape):
        return np.cumprod([1] + list(shape[-1:0:-1]), dtype=int)[::-1]

    @property
    def memory(self) -> MemoryBase:
        return self._runner.memory

    @classmethod
    def get_instance(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = cls()

        return cls._instance

    # abstract methods
    @staticmethod
    def get_memory_type(address: int) -> MType:
        raise NotImplementedError()

    @staticmethod
    def local_layout_to_stride(memref: MemRefBase) -> Tuple[int, int, int, int]:
        raise NotImplementedError()

    @classmethod
    def merge_instruction(cls, tiu: List[BaseTpuCmd], dma: List[BaseTpuCmd]):
        raise NotImplementedError()

    @classmethod
    def is_sys(cls, _: BaseTpuCmd):
        raise NotImplementedError()

    def get_runner(self, memory_size: int) -> Runner:
        raise NotImplementedError()


def get_target_context_cls(chip: str) -> Type[BModelContext]:
    assert chip in {"BM1684X", "BM1684", "BM1688", "BM1690", "CV186X", "SG2380", "MARS3"}

    context = None
    if chip == "BM1684X":
        from ..target_1684x.context import BM1684XContext

        return BM1684XContext
    elif chip in ("BM1688", "CV186X"):
        from ..target_1688.context import BM1688Context

        context = BM1688Context
    elif chip == "BM1684":
        from ..target_1684.context import BM1684Context

        context = BM1684Context
    elif chip == "BM1690":
        from ..target_1690.context import BM1690Context

        context = BM1690Context
    elif chip == "SG2380":
        from ..target_2380.context import SG2380Context

        context = SG2380Context
    elif chip == "MARS3":
        from ..target_mars3.context import MARS3Context
        context = MARS3Context

    assert context is not None, f"target {chip} not found"
    return context


def get_target_context(chip: str) -> BModelContext:
    assert chip in {"BM1684X", "BM1684", "BM1688", "BM1690", "CV186X", "SG2380", "MARS3"}
    context_cls = get_target_context_cls(chip)
    return context_cls()


@contextmanager
def use_backend(chip: str):
    backend = get_target_context(chip=chip)
    try:
        yield backend
    finally:
        pass
