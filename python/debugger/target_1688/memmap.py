# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from ..target_common import DType, MType, Layout, MemRefBase, Device
from typing import Tuple, TYPE_CHECKING
import functools

if TYPE_CHECKING:
    from .context import BM1688Context


NPU_SHIFT = 5
NPU_NUM = 32
BANK_NUM = 16
LANE_SIZE = 1 << 17
BANK_SIZE = LANE_SIZE // BANK_NUM
LMEM_SIZE = LANE_SIZE * NPU_NUM
ALIGN_EU_BASE = 16


BASE_EU_NUM = 4
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


def EU_NUM(dtype):
    return TYPED_EU_NUM[dtype]


# TPU1688/bm1688/spec/include/memmap.h
memmap = {
    MType.R: (int("0x0", 16), int("0x400000", 16)),  # lmen_base 8M
    # static memory 16KB
    MType.S: (int("0x25800000", 16), int("0x25804000", 16)),
    MType.G: (int("0x100000000", 16), int("0x200000000", 16)),  # global memory
}
# base_addr currently only used for BM16846
BASE_ADDR = [memmap[MType.G][0] for _ in range(2)]


@staticmethod
def local_layout_to_stride(memref: MemRefBase) -> Tuple[int, int, int, int]:
    """
    Layout Canonicalize. Convert special layout to stride layout.
    """

    def alignEU_stride():
        _, c, h, w = memref.shape
        align_type = ALIGN_EU_BASE // memref.itemsize
        c_stride = int((w * h + align_type - 1) // align_type * align_type)
        n_stride = int((c + memref.npu_offset + NPU_NUM - 1) // NPU_NUM * c_stride)
        return (n_stride, c_stride, w, 1)

    def compact_stride():
        _, c, h, w = memref.shape
        c_stride = int(w * h)
        n_stride = int((c + memref.npu_offset + NPU_NUM - 1) // NPU_NUM * c_stride)
        return (n_stride, c_stride, w, 1)

    def offset_stride():
        return (0, 1, 0, 0)

    def t3_stride():
        _, c, h, w = memref.shape
        eu_num = ALIGN_EU_BASE // memref.itemsize
        h_stride = (w + eu_num - 1) / eu_num * eu_num
        c_stride = h * h_stride
        n_stride = c_stride * (c + memref.npu_offset + NPU_NUM - 1) // NPU_NUM
        return (n_stride, c_stride, h_stride, 1)

    def t4_stride():
        _, _, _, w = memref.shape
        eu_num = ALIGN_EU_BASE // memref.itemsize
        h_stride = (w + eu_num - 1) / eu_num * eu_num
        return (0, 0, h_stride, 1)

    def t5_stride():
        _, _, _, w = memref.shape
        eu_num = ALIGN_EU_BASE // memref.itemsize
        c_stride = (w + eu_num - 1) / eu_num * eu_num
        n_stride = LANE_SIZE // 8 // memref.itemsize
        return (n_stride, c_stride, w, 1)

    def dma_nolane_mask_to_stride():
        lane_mask = memref.layout.args[0]
        if lane_mask == (2**64 - 1):
            memref.layout = Layout.stride
        return memref.stride

    if memref.dtype in (DType.i4, DType.si4, DType.ui4):
        memref.itemsize = 0.5

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


class MemRef(MemRefBase):
    """
    A description of tensor in memory.
    """

    device = Device.BM1688

    def __init__(
        self,
        address,
        shape,
        dtype: DType,
        stride=None,
        layout=None,
        context: "BM1688Context" = None,
    ):
        assert context is not None
        self.context = context
        address = self.context.fix_tag(address)
        super().__init__(address, shape, dtype, stride, layout)
        if self.mtype == MType.R and layout != Layout.stride:
            self.stride = local_layout_to_stride(self)

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

    def get_mtype(self, address) -> MType:
        return self.context.get_memory_type(address)

    @property
    @functools.lru_cache()
    def r_addr(self):
        if self.mtype == MType.UNKNOWN:
            return self.address

        r_addr = self.address - memmap[self.mtype][0]
        return r_addr

    @property
    @functools.lru_cache()
    def local_shape(self):
        # NPU_OFFSET = self.npu_offset
        NPU_OFFSET = 0
        n, c, h, w, *_ = *self.shape, 1, 1

        def get_cnum(c):
            return (c + NPU_OFFSET + NPU_NUM - 1) // NPU_NUM

        if self.layout == Layout._64IC:
            return (c + 63) // 64, get_cnum(n), h, w * 64

        if self.layout == Layout._32IC:
            return (c + 32) // 32, get_cnum(n), h, w * 32

        if self.layout == Layout._16IC:
            return (c + 15) // 16, get_cnum(n), h, w * 16

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
            align_type = ALIGN_EU_BASE // self.itemsize
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
