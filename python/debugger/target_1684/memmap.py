# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from functools import lru_cache
from ..target_common import DType, MType, Layout, MemRefBase, Target

NPU_NUM = 64
BANK_SIZE = 4 * 2**14
LANE_SIZE = BANK_SIZE * 8
LMEM_SIZE = LANE_SIZE * NPU_NUM

memmap = {
    MType.R: (int("0x8000000", 16), int("0x10000000", 16)),  # lmen_base 16M
    MType.L: (int("0x10000000", 16), int("0x10300000", 16)),  # L2 SRAM  2M
    MType.G: (int("0x100000000", 16), int("0x300000000", 16)),  # global memory
}


def get_memory_type(address: int) -> MType:
    # R : "npu_offset", "bank_index", "bank_offset", "r_addr"
    # G/S/L : "r_addr"
    for k, v in memmap.items():
        if address >= v[0] and address < v[1]:
            return k
    return MType.UNKNOWN


def Ceil(x, y):
    return (x + y - 1) // y


def AlignY(x, y):
    return Ceil(x, y) * y


def local_layout_to_stride(memref: MemRefBase):
    """
    Layout Canonicalize. Convert special layout to stride layout.
    """

    def compute_stride(is_aligned=False):
        _, c, h, w = memref.shape
        if is_aligned:
            # The original formula is:
            # align_num = EU_NUM * (32 / bitsWidth)
            # c_stride = ceil(w * h / align_num)
            # EU_NUM = 32, and with different bitsWidth, we can build a table like this:
            # 8bits  -> algin_num = 128
            # 16bits -> algin_num = 64
            # 32bits -> algin_num = 32
            # after apply the byte size, we can conclude that using 128 bytes is fine.
            align_type = 128 // memref.itemsize
            c_stride = AlignY(w * h, align_type)
        else:
            c_stride = w * h
        n_stride = Ceil(c + memref.npu_offset, NPU_NUM) * c_stride
        return (n_stride, c_stride, w, 1)

    if memref.layout == Layout.alignEU:
        return compute_stride(True)
    if memref.layout == Layout.compact:
        return compute_stride(False)

    return memref.stride


class MemRef(MemRefBase):
    """
    A description of tensor in memory.
    """

    device = Target.BM1684

    def __init__(self, address, shape, dtype: DType, stride=None, layout=None):
        super().__init__(address, shape, dtype, stride, layout)
        if self.mtype == MType.R and layout != Layout.stride:
            self.stride = local_layout_to_stride(self)

    @property
    @lru_cache()
    def r_addr(self):
        if self.mtype == MType.UNKNOWN:
            return self.address

        r_addr = self.address - memmap[self.mtype][0]
        return r_addr

    def get_mtype(self, address) -> MType:
        return get_memory_type(address)

    @property
    def npu_offset(self):
        assert self.mtype == MType.R
        return self.r_addr // LANE_SIZE

    @property
    def bank_index(self):
        assert self.mtype == MType.R
        addr_len = self.r_addr - self.npu_offset * LANE_SIZE
        return addr_len // BANK_SIZE

    @property
    def bank_offset(self):
        assert self.mtype == MType.R
        addr_len = self.r_addr - self.npu_offset * LANE_SIZE
        return addr_len % BANK_SIZE
