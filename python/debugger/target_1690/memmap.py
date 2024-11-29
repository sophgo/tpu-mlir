# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from ..target_common import DType, MType, Layout, MemRefBase, Target, div_up, align_up, lib_wrapper, open_lib
from typing import Tuple, TYPE_CHECKING
import functools
from functools import lru_cache

if TYPE_CHECKING:
    from .context import BM1690Context


class BM1690Info:
    def __init__(self) -> None:
        self.lib_name = "libtpuv7_emulator.so"
        self._lib = None

    @property
    @lru_cache()
    def lib(self):
        if not self._lib:
            self._lib = lib_wrapper(open_lib(self.lib_name))
        return self._lib

    @property
    def NPU_NUM(self) -> int:
        return self.lib.tpu_npu_num()

    @property
    def BANK_NUM(self) -> int:
        return self.lib.tpu_bank_num()

    @property
    def LANE_SIZE(self) -> int:
        return self.lib.tpu_local_mem_size_per_npu()

    @property
    def BANK_SIZE(self) -> int:
        return self.LANE_SIZE // self.BANK_NUM

    @property
    def LMEM_SIZE(self) -> int:
        return self.LANE_SIZE * self.NPU_NUM

    @property
    def L2MEM_SIZE(self) -> int:
        return self.lib.tpu_l2_sram_size()

    # eu_num when byte size is 1
    @property
    def ALIGN_EU_BASE(self) -> int:
        return self.lib.tpu_eu_num(1)

    def EU_NUM(self, dtype: DType) -> int:
        BASE_EU_NUM = self.ALIGN_EU_BASE
        TYPED_EU_NUM = {
            DType.f32: BASE_EU_NUM // 4,
            DType.i32: BASE_EU_NUM // 4,
            DType.si32: BASE_EU_NUM // 4,
            DType.ui32: BASE_EU_NUM // 4,
            DType.f16: BASE_EU_NUM // 2,
            DType.bf16: BASE_EU_NUM // 2,
            DType.i16: BASE_EU_NUM // 2,
            DType.ui16: BASE_EU_NUM // 2,
            DType.si16: BASE_EU_NUM // 2,
            DType.i8: BASE_EU_NUM,
            DType.ui8: BASE_EU_NUM,
            DType.si8: BASE_EU_NUM,
            DType.i4: BASE_EU_NUM * 2,
            DType.ui4: BASE_EU_NUM * 2,
            DType.si4: BASE_EU_NUM * 2,
        }
        return TYPED_EU_NUM[dtype]

    # eu_num when byte size is 1
    @property
    def CUBE_NUM_BASE(self) -> int:
        return self.lib.tpu_get_ic_parallel(1)

    def CUBE_NUM(self, dtype: DType) -> int:
        BASE_CUBE_NUM = self.CUBE_NUM_BASE
        TYPED_CUBE_NUM = {
            DType.f32: 1,
            DType.i32: 1,
            DType.si32: 1,
            DType.ui32: 1,
            DType.f16: BASE_CUBE_NUM // 2,
            DType.bf16: BASE_CUBE_NUM // 2,
            DType.i16: BASE_CUBE_NUM // 2,
            DType.ui16: BASE_CUBE_NUM // 2,
            DType.si16: BASE_CUBE_NUM // 2,
            DType.i8: BASE_CUBE_NUM,
            DType.ui8: BASE_CUBE_NUM,
            DType.si8: BASE_CUBE_NUM,
            DType.i4: BASE_CUBE_NUM * 2,
            DType.ui4: BASE_CUBE_NUM * 2,
            DType.si4: BASE_CUBE_NUM * 2,
        }
        return TYPED_CUBE_NUM[dtype]

info = BM1690Info()


# /TPU1686/sg2260/spec/include/memmap.h
memmap = {
    MType.R: (0, (1 << 18)*64),  # each lmem has 256KB, use addr[26] to determine if it's lmem or smem
    MType.S: (0, 64 * 1024),  # 64KB
    MType.L: (0x6980000000, 0x6980000000 + info.L2MEM_SIZE),  # L2 SRAM 128M
    MType.G: (0x0, 0x100000000),  # global memory 4G
}

def local_layout_to_stride(memref: MemRefBase) -> Tuple[int, int, int, int]:
    """
    Layout Canonicalize. Convert special layout to stride layout.
    """

    def alignEU_stride():
        _, c, h, w = memref.shape
        align_num = int(info.ALIGN_EU_BASE / memref.itemsize)
        c_stride = align_up(w * h, align_num)
        n_stride = div_up(c + memref.npu_offset, info.NPU_NUM) * c_stride
        return (n_stride, c_stride, w, 1)

    def compact_stride():
        _, c, h, w = memref.shape
        c_stride = w * h
        n_stride = div_up(c + memref.npu_offset, info.NPU_NUM) * c_stride
        return (n_stride, c_stride, w, 1)

    def offset_stride():
        return (0, 1, 0, 0)

    def alignLine_stride():
        _, c, h, w = memref.shape
        eu_num = int(info.ALIGN_EU_BASE / memref.itemsize)
        h_stride = align_up(w, eu_num)
        c_stride = h * h_stride
        n_stride = c_stride * div_up(c + memref.npu_offset, info.NPU_NUM)
        return (n_stride, c_stride, h_stride, 1)

    def t4_stride():
        _, _, _, w = memref.shape
        align_num = int(info.ALIGN_EU_BASE / memref.itemsize)
        h_stride = align_up(w, align_num)
        return (0, 0, h_stride, 1)

    def t5_stride():
        _, _, _, w = memref.shape
        align_num = int(info.ALIGN_EU_BASE / memref.itemsize)
        c_stride = align_up(w, align_num)
        n_stride = info.LANE_SIZE // 8 // memref.itemsize
        return (n_stride, c_stride, w, 1)

    def dma_nolane_mask_to_stride():
        lane_mask = memref.layout.args[0]
        if lane_mask == (2**info.NPU_NUM - 1):
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
    if memref.layout == Layout.alignLine:
        return alignLine_stride()
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

    device = Target.BM1690

    # return context.MemRef(147456, (1, 3, 46, 640), DType.f16, (0, 0, 0, 0), Layout.alignEU)
    def __init__(
        self,
        address,
        shape,
        dtype: DType,
        stride=None,
        layout=None,
        context: "BM1690Context" = None,
    ):
        assert context is not None
        self.context = context

        super().__init__(address, shape, dtype, stride, layout)
        if self.mtype == MType.R and layout != Layout.stride:
            self.stride = local_layout_to_stride(self)

    @property
    def npu_offset(self):
        assert self.mtype == MType.R
        return self.r_addr // info.LANE_SIZE

    @property
    def bank_index(self):
        assert self.mtype == MType.R
        addr_len = self.r_addr - self.npu_offset * info.LANE_SIZE
        return addr_len // info.BANK_SIZE

    @property
    def bank_offset(self):
        assert self.mtype == MType.R
        addr_len = self.r_addr - self.npu_offset * info.LANE_SIZE
        return addr_len % info.BANK_SIZE

    def get_mtype(self, address) -> MType:
        return self.context.get_memory_type(address)

    @property
    @functools.lru_cache()
    def r_addr(self):
        if self.mtype in [MType.UNKNOWN, MType.G, MType.L]:
            return self.context.fix_addr(self.address) - self.context.memmap[self.mtype][0]

        r_addr = self.address & 0x3FFFFFF  # remain 26 bit as local offset
        return r_addr

    @property
    def _backend_name(self):
        k = self.mtype
        if k == MType.UNKNOWN:
            return f"%?.{self.address}"
        if k == MType.R:
            # R{bank_index}.{bank_offset}.L{NPU_OFFSET}
            # R0.123.L0
            mem_str = f"%{k.name}{self.bank_index}"
            if self.bank_offset:
                mem_str += f".0x{self.bank_offset:x}"
            if self.npu_offset:
                mem_str += f".L0x{self.npu_offset:x}"
            return mem_str
        if k == MType.G:
            tag = (self.address >> 40) & 0x1F
            offset = self.address & 0xFFFFFFFFFF
            return f"%{k.name}{tag}.0x{offset:x}"
        elif k == MType.L:
            tag = (self.address >> 40) & 0x1E
            offset = self.address & 0xFFFFFFFFFF
            return f"%{k.name}{tag}.0x{offset:x}"
        return f"%{k.name}0x{self.r_addr:x}"

    @property
    @functools.lru_cache()
    def local_shape(self):
        n, c, h, w, *_ = *self.shape, 1, 1

        if self.layout == Layout.alignIC:
            cube_num = info.CUBE_NUM(self.dtype)
            return 1, min(n, info.NPU_NUM), 1, div_up(n, info.NPU_NUM) * h * w * align_up(c, cube_num)

        if self.layout == Layout.matrix:
            w = self.layout.args[0]
            return n, div_up(c, w), 1, w

        if self.layout == Layout.matrix2:
            return 1, n, 1, c

        if self.layout == Layout.DMA4Bank:
            # FIX ME
            return n, c, h, w

        if self.layout == Layout.DMAstride:
            return n, c, h, w

        if self.layout == Layout.DMAmatrix:
            w = self.layout.args[1]
            return n, div_up(c, w), 1, w

        if self.layout == Layout.DMAlinear:
            return self.shape

        return n, c, h, w

    @property
    @functools.lru_cache()
    def local_stride(self):
        NPU_OFFSET = self.npu_offset

        def compact_stride():
            _, _c, _h, _w = self.local_shape
            c_stride = _w * _h
            n_stride = div_up(_c + NPU_OFFSET, info.NPU_NUM) * c_stride
            return n_stride, c_stride, _w, 1

        def line_align_stride():
            _, _c, _h, _w = self.local_shape
            align_num = int(info.ALIGN_EU_BASE / self.itemsize)
            h_stride = align_up(_w, align_num)
            c_stride = _h * h_stride
            n_stride = div_up(_c + NPU_OFFSET, info.NPU_NUM) * c_stride
            return n_stride, c_stride, _w, 1

        def eu_align_stride():
            _, _c, _h, _w = self.local_shape
            align_num = int(info.ALIGN_EU_BASE / self.itemsize)
            c_stride = align_up(_w * _h, align_num)
            n_stride = div_up(_c + NPU_OFFSET, info.NPU_NUM) * c_stride
            return n_stride, c_stride, _w, 1

        def t4_stride():
            _, _, _, w = self.local_shape
            align_num = int(info.ALIGN_EU_BASE / self.itemsize)
            h_stride = align_up(w, align_num)
            return (0, 0, h_stride, 1)

        def t5_stride():
            _, _, _, w = self.local_shape
            align_num = int(info.ALIGN_EU_BASE / self.itemsize)
            c_stride = align_up(w, align_num)
            n_stride = info.LANE_SIZE // 8 // self.itemsize
            return (n_stride, c_stride, w, 1)

        if self.layout in (Layout.alignEU, Layout.alignIC):
            return eu_align_stride()

        if self.layout == Layout.compact:
            return compact_stride()

        if self.layout == Layout.alignLine:
            return line_align_stride()

        if self.layout == Layout.T4:
            return t4_stride()

        if self.layout == Layout.T5:
            return t5_stride()

        return self.stride
