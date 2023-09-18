# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .cmodel import MemRefBase, MemoryBase, CModelRunner
from contextlib import contextmanager
import numpy as np
from typing import Dict, Tuple, List, Type
from .decoder import DecoderBase
from .op_support import cmd_base_reg, MType, Device


class CModelContext:
    base_addr = [0, 0]  # special defined for 1688
    MemRef = MemRefBase
    device: Device
    decoder: DecoderBase

    memmap: Dict[MType, Tuple[int, int]]
    dma_sys: cmd_base_reg = None
    tiu_sys: cmd_base_reg = None

    def __init__(self) -> None:
        self._runner: CModelRunner = None

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

    @staticmethod
    def merge_instruction(tiu: List[cmd_base_reg], dma: List[cmd_base_reg]):
        raise NotImplementedError()

    def get_runner(self, memory_size: int) -> CModelRunner:
        raise NotImplementedError()


def get_target_context_cls(chip: str) -> Type[CModelContext]:
    assert chip in {"BM1684X", "BM1684", "BM1688"}

    context = None
    if chip == "BM1684X":
        from ..target_1684x.context import BM1684XContext

        return BM1684XContext
    elif chip == "BM1688":
        from ..target_1688.context import BM1688Context

        context = BM1688Context
    elif chip == "BM1684":
        from ..target_1684.context import BM1684Context

        context = BM1684Context

    assert context is not None, f"target {chip} not found"
    return context


def get_target_context(chip: str) -> CModelContext:
    assert chip in {"BM1684X", "BM1684", "BM1688"}
    context_cls = get_target_context_cls(chip)
    return context_cls()


@contextmanager
def use_backend(chip: str):
    backend = get_target_context(chip=chip)
    try:
        yield backend
    finally:
        pass
