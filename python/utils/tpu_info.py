#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
"""TPU hardware specifications extracted from include/tpu_mlir/Backend/BM168x/*.h

Provides per-chip constants (npu_num, eu_bytes, lmem_bytes, etc.) to other
Python code without requiring the C++ backend libraries to be loaded.

Usage::

    from utils.tpu_info import TPUInfo, get_tpu_info

    info = get_tpu_info("bm1684x")
    print(info.npu_num)   # 64
    print(info.lmem_kb)   # 256
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Per-chip hardware constants — extracted from BM168x subclass constructors in:
#   include/tpu_mlir/Backend/BM168x/{BM1684,BM1684X,BM1684X2,BM1688,BM1690,
#                                    BM1690E,SG2380,SGTPUV8}.h
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TPUInfo:
    """Immutable per-chip hardware specification."""

    npu_num: int  # number of NPUs per core
    eu_bytes: int  # EU width in bytes (vector lane byte count)
    lmem_bytes: int  # local memory per NPU (one lane)
    lmem_banks: int  # local memory banks
    ic_parallel: int  # IC parallel count (for convolution parallelism)
    dma_algn_bytes: int  # DMA alignment in bytes
    alignment: int  # general memory page alignment
    max_core_num: int  # maximum number of cores supported
    l2_sram_size: int  # L2 SRAM size (0 if not available)
    lmem_bank_bytes: int  # local memory per bank (= lmem_bytes // lmem_banks)

    @property
    def lmem_kb(self) -> int:
        return self.lmem_bytes // 1024

    @property
    def lmem_bank_kb(self) -> int:
        return self.lmem_bank_bytes // 1024

    @property
    def l2_sram_kb(self) -> int:
        return self.l2_sram_size // 1024

    @property
    def total_lmem_bytes(self) -> int:
        """Total local memory across all NPUs on one core."""
        return self.npu_num * self.lmem_bytes

    @property
    def total_lmem_kb(self) -> int:
        return self.total_lmem_bytes // 1024

    def eu_num(self, dtype_bytes: float) -> int:
        """EU parallelism count for a given dtype in bytes (e.g. 1 for int8)."""
        return int(self.eu_bytes / dtype_bytes)

    def ic_num(self, dtype_bytes: float) -> int:
        """IC parallelism count for a given dtype in bytes."""
        return int(self.ic_parallel / dtype_bytes)


# The canonical chip-name list (order matches regression/chip.py CHIP_COLUMNS for
# consistency).  CV18xx / SG2262 chips are *not* covered here because
# their specs live in include/tpu_mlir/Backend/CV18xx/, not BM168x.
_CHIPS: dict[str, TPUInfo] = {
    #
    # BM1684 — BM1684.h lines 462–485
    #
    "bm1684":
    TPUInfo(
        npu_num=64,
        eu_bytes=128,  # 128 for fp32; only int8: 128 per NPU
        lmem_bytes=1 << 19,  # 512 KB
        lmem_banks=8,
        ic_parallel=64,  # not explicitly set; NPU_NUM-based
        dma_algn_bytes=512,
        alignment=0x1000,
        max_core_num=1,
        l2_sram_size=0,
        lmem_bank_bytes=(1 << 19) // 8,
    ),
    #
    # BM1684X — BM1684X.h lines 102–139
    #
    "bm1684x":
    TPUInfo(
        npu_num=64,
        eu_bytes=64,
        lmem_bytes=1 << 18,  # 256 KB
        lmem_banks=16,
        ic_parallel=64,
        dma_algn_bytes=512,
        alignment=0x1000,
        max_core_num=1,
        l2_sram_size=0x1FB000,
        lmem_bank_bytes=(1 << 18) // 16,
    ),
    #
    # BM1684X2 — BM1684X2.h lines 76–109
    #
    "bm1684x2":
    TPUInfo(
        npu_num=16,
        eu_bytes=32,
        lmem_bytes=1 << 18,  # 256 KB
        lmem_banks=16,
        ic_parallel=32,
        dma_algn_bytes=512,  # inherited from BM1684X
        alignment=0x1000,
        max_core_num=4,  # not explicitly set; defaults to 1
        l2_sram_size=0x100000,
        lmem_bank_bytes=(1 << 18) // 16,
    ),
    #
    # BM1688 — BM1688.h lines 111–145
    #
    "bm1688":
    TPUInfo(
        npu_num=32,
        eu_bytes=16,
        lmem_bytes=1 << 17,  # 128 KB
        lmem_banks=16,
        ic_parallel=32,
        dma_algn_bytes=512,
        alignment=0x1000,
        max_core_num=2,
        l2_sram_size=0,
        lmem_bank_bytes=(1 << 17) // 16,
    ),
    #
    # cv186x — CV186x.h; BM1688-based with 32-bit device bus
    #
    "cv186x":
    TPUInfo(
        npu_num=32,
        eu_bytes=16,
        lmem_bytes=1 << 17,  # 128 KB
        lmem_banks=16,
        ic_parallel=32,
        dma_algn_bytes=512,
        alignment=0x1000,
        max_core_num=1,
        l2_sram_size=0,
        lmem_bank_bytes=(1 << 17) // 16,
    ),
    #
    # BM1690 — BM1690.h lines 98–136
    #
    "bm1690":
    TPUInfo(
        npu_num=64,
        eu_bytes=64,
        lmem_bytes=1 << 18,  # 256 KB
        lmem_banks=16,
        ic_parallel=64,
        dma_algn_bytes=256,
        alignment=0x1000,
        max_core_num=8,
        l2_sram_size=0x8000000,
        lmem_bank_bytes=(1 << 18) // 16,
    ),
    #
    # BM1690E — BM1690E.h lines 101–137
    #
    "bm1690e":
    TPUInfo(
        npu_num=64,
        eu_bytes=64,
        lmem_bytes=1 << 18,  # 256 KB
        lmem_banks=16,
        ic_parallel=64,
        dma_algn_bytes=512,  # inherited from BM1684X
        alignment=0x1000,
        max_core_num=4,  # not explicitly set; defaults to 1
        l2_sram_size=0x1000000,
        lmem_bank_bytes=(1 << 18) // 16,
    ),
    #
    # SG2380 — SG2380.h lines 133–165
    #
    "sg2380":
    TPUInfo(
        npu_num=32,
        eu_bytes=16,
        lmem_bytes=1 << 17,  # 128 KB
        lmem_banks=16,
        ic_parallel=32,
        dma_algn_bytes=256,
        alignment=0x1000,
        max_core_num=4,
        l2_sram_size=0,
        lmem_bank_bytes=(1 << 17) // 16,
    ),
    #
    # SGTPUV8 — SGTPUV8.h lines 84–113
    #
    "sgtpuv8":
    TPUInfo(
        npu_num=8,
        eu_bytes=16,
        lmem_bytes=65536,
        lmem_banks=16,
        ic_parallel=16,
        dma_algn_bytes=256,
        alignment=0x1000,
        max_core_num=1,
        l2_sram_size=0,
        lmem_bank_bytes=65536 // 16,
    ),
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_tpu_info(chip: str) -> TPUInfo:
    """Return :class:`TPUInfo` for *chip* (lowercase, e.g. ``"bm1684x"``).

    Raises :class:`KeyError` if *chip* is unknown.
    """
    return _CHIPS[chip]


def supported_chips() -> list[str]:
    """Return sorted list of chip names known to this module."""
    return sorted(_CHIPS.keys())


def has_l2_sram(chip: str) -> bool:
    """Return ``True`` if *chip* has L2 SRAM."""
    return _CHIPS[chip].l2_sram_size > 0


def get_core_num(chip: str) -> int:
    """Shortcut: maximum core number for *chip*."""
    return _CHIPS[chip].max_core_num
