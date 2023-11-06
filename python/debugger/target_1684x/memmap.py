# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from ..target_common import DType, MType, Layout, MemRefBase, Device
import functools
from typing import Tuple

NPU_NUM = 64
BANK_SIZE = 2**14
LANE_SIZE = BANK_SIZE * 16
LMEM_SIZE = LANE_SIZE * NPU_NUM
BASE_EU_NUM = 16
TYPED_EU_NUM = {
    DType.f32: BASE_EU_NUM,
    DType.i32: BASE_EU_NUM,
    DType.si32: BASE_EU_NUM,
    DType.ui32: BASE_EU_NUM,
    DType.f16: BASE_EU_NUM * 2,
    DType.bf16: BASE_EU_NUM * 2,
    DType.i16: BASE_EU_NUM * 2,
    DType.ui16: BASE_EU_NUM * 2,
    DType.si16: BASE_EU_NUM * 2,
    DType.i8: BASE_EU_NUM * 4,
    DType.ui8: BASE_EU_NUM * 4,
    DType.si8: BASE_EU_NUM * 4,
}


def EU_NUM(dtype: DType) -> int:
    return TYPED_EU_NUM[dtype]


# TPU1688/bm1688/spec/include/memmap.h
memmap = {
    MType.R: (0x8000000, 0x9000000),  # lmen_base 16M
    MType.S: (0x9000000, 0x9004000),  # static memory 16KB
    MType.L: (0x10000000, 0x10200000),  # L2 SRAM  2M
    MType.G: (0x100000000, 0x300000000),  # global memory
}


@staticmethod
def local_layout_to_stride(memref: MemRefBase) -> Tuple[int, int, int, int]:
    """
    Layout Canonicalize. Convert special layout to stride layout.
    """

    def alignEU_stride():
        _, c, h, w = memref.shape
        align_type = 64 // memref.itemsize
        c_stride = (w * h + align_type - 1) // align_type * align_type
        n_stride = (c + memref.npu_offset + 63) // 64 * c_stride
        return (n_stride, c_stride, w, 1)

    def compact_stride():
        _, c, h, w = memref.shape
        c_stride = w * h
        n_stride = (c + memref.npu_offset + 63) // 64 * c_stride
        return (n_stride, c_stride, w, 1)

    def offset_stride():
        return (0, 1, 0, 0)

    def t3_stride():
        _, c, h, w = memref.shape
        eu_num = 64 // memref.itemsize
        h_stride = (w + eu_num - 1) / eu_num * eu_num
        c_stride = h * h_stride
        n_stride = c_stride * (c + memref.npu_offset + 63) // 64
        return (n_stride, c_stride, h_stride, 1)

    def t4_stride():
        _, _, _, w = memref.shape
        eu_num = 64 // memref.itemsize
        h_stride = (w + eu_num - 1) / eu_num * eu_num
        return (0, 0, h_stride, 1)

    def t5_stride():
        _, _, _, w = memref.shape
        eu_num = 64 // memref.itemsize
        c_stride = (w + eu_num - 1) / eu_num * eu_num
        n_stride = LANE_SIZE // 8 // memref.itemsize
        return (n_stride, c_stride, w, 1)

    def dma_nolane_mask_to_stride():
        lane_mask = memref.layout.args[0]
        if lane_mask == (2**64 - 1):
            memref.layout = Layout.stride
        return memref.stride

    if memref.layout == Layout.alignEU:
        return alignEU_stride()
    if memref.layout == Layout.compact:
        return compact_stride()
    if memref.layout == Layout.offset:
        return offset_stride()
    if memref.layout == Layout.T3:
        return t3_stride()
    if memref.layout == Layout.T4:
        return t4_stride()
    if memref.layout == Layout.T5:
        return t5_stride()
    if memref.layout == Layout.DMAstride:
        return dma_nolane_mask_to_stride()

    return memref.stride


@staticmethod
def get_memory_type(address: int) -> MType:
    # R : "npu_offset", "bank_index", "bank_offset", "r_addr"
    # G/S/L : "r_addr"
    for k, v in memmap.items():
        if address >= v[0] and address < v[1]:
            return k
    return MType.UNKNOWN


class MemRef(MemRefBase):
    """
    A description of tensor in memory.
    """

    device = Device.BM1684X

    def __init__(self, address, shape, dtype: DType, stride=None, layout=None):
        super().__init__(address, shape, dtype, stride, layout)
        if self.mtype == MType.R and layout != Layout.stride:
            self.stride = local_layout_to_stride(self)

    @property
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

    @property
    @functools.lru_cache()
    def local_shape(self):
        NPU_OFFSET = self.npu_offset
        n, c, h, w, *_ = *self.shape, 1, 1

        def get_cnum(c):
            return (c + NPU_OFFSET + NPU_NUM - 1) // NPU_NUM

        if self.layout == Layout._64IC:
            return (c + 63) // 64, get_cnum(n), h, w * 64

        if self.layout == Layout._32IC:
            return (c + 32) // 32, get_cnum(n), h, w * 32

        if self.layout == Layout._1IC:
            return c, get_cnum(n), h, w

        if self.layout == Layout.matrix:
            w = self.layout.args[0]
            return n, get_cnum((c + w - 1) // w), 1, w

        if self.layout == Layout.matrix2:
            return 1, get_cnum(n), 1, c

        if self.layout == Layout.DMA4Bank:
            # FIX ME
            return n, get_cnum(c), h, w

        if self.layout == Layout.DMAstride:
            return n, get_cnum(c), h, w

        if self.layout == Layout.DMAmatrix:
            w = self.layout.args[1]
            return n, get_cnum((c + w - 1) // w), 1, w

        if self.layout == Layout.DMAlinear:
            return self.shape

        return n, get_cnum(c), h, w

    @property
    @functools.lru_cache()
    def local_stride(self):
        n, c, h, w, *_ = *self.shape, 1, 1
        NPU_OFFSET = self.npu_offset

        def get_eu_align_stride(shape):
            _, _c, _h, _w = shape
            align_type = 64 // self.itemsize
            c_stride = (_w * _h + align_type - 1) // align_type * align_type
            n_stride = (_c + NPU_OFFSET + NPU_NUM - 1) // NPU_NUM * c_stride
            return n_stride, c_stride, _w, 1

        if self.layout == Layout._64IC:
            return 64 * h * w, (c + 63) // 64 * 64 * h * w, 64 * w, 1

        if self.layout == Layout._32IC:
            return 32 * h * w, (c + 32) // 32 * 32 * h * w, 32 * w, 1

        if self.layout == Layout._1IC:
            return h * w, c * h * w, w, 1

        if self.layout == Layout.matrix:
            w = self.layout.args[0]
            shape = (n, (c + w - 1) // w, 1, w)
            return get_eu_align_stride(shape)

        if self.layout == Layout.matrix2:
            shape = (1, n, 1, c)
            return get_eu_align_stride(shape)

        return self.stride
