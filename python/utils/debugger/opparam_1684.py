# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

# This file contains some functions to convert the reg information to MLIR IR
# It is dirty and heavy.

import numpy as np

try:
    from . import op_support
    from .op_support import MType, DType, Scalar, ExtEnum, Layout
    from .op_support import get_dtype
except:
    import op_support
    from op_support import MType, DType, Scalar, ExtEnum, Layout
    from op_support import get_dtype

__all__ = ["opparam_converter"]

# Value: MemRef | Scalar
# Scalar: Number
# Number: int | float
# MemRef: address, shape, Dtype, stride, offset?

NPU_NUM = 64
BANK_SIZE = 4 * 2**14
LANE_SIZE = BANK_SIZE * 8
LMEM_SIZE = LANE_SIZE * NPU_NUM

opparam_converter = {}


def opparam_converter_regitstry(class_name):
    def add_converter(fun):
        if class_name in opparam_converter:
            raise KeyError(f"{class_name} have already registered.")
        opparam_converter[class_name] = fun
        return

    return add_converter


memmap = {
    MType.R: (int("0x8000000", 16), int("0x10000000", 16)),  # lmen_base 16M
    MType.L: (int("0x10000000", 16), int("0x10300000", 16)),  # L2 SRAM  2M
    MType.G: (int("0x100000000", 16), int("0x300000000", 16)),  # global memory
}


class MemRef(op_support.MemRef):
    """
    A description of tensor in memory.
    """

    def __init__(self, address, shape, dtype: DType, stride=None, layout=None):
        super().__init__(address, shape, dtype, stride, layout)

    def get_mtype(self, address):
        # R : "npu_offset", "bank_index", "bank_offset", "r_addr"
        # G/S/L : "r_addr"
        for k, v in memmap.items():
            if address >= v[0] and address < v[1]:
                if k == MType.R:
                    r_addr = address - v[0]
                    npu_offset = r_addr // LANE_SIZE
                    addr_len = r_addr - npu_offset * LANE_SIZE
                    bank_index = addr_len // BANK_SIZE
                    bank_offset = addr_len % BANK_SIZE
                    # Local memory Type
                    return MType.R(
                        npu_offset=npu_offset,
                        bank_index=bank_index,
                        bank_offset=bank_offset,
                        r_addr=r_addr,
                    )
                return k(r_addr=address - v[0])
        return MType.UNKNOWN


class Memory:
    """
    Memory agent. Extract/Set data from a give MemRef object.
    This class should handle all the tenors type in all kinds of storage.
    """

    def __init__(self, LMEM, DDR) -> None:
        self.LMEM = LMEM.ravel()
        self.DDR = DDR.ravel()

    def _local_mem_to_numpy(self, memref):
        NPU_OFFSET = memref.mtype.npu_offset
        itemsize = memref.itemsize

        def data_view(shape, stride):
            offset = memref.mtype.r_addr - NPU_OFFSET * LANE_SIZE
            return np.lib.stride_tricks.as_strided(
                self.LMEM[offset : offset + 4].view(memref.np_dtype),
                shape,
                np.array(stride) * itemsize,
            )

        def get_stride_data_base(shape, stride):
            n, c, h, w = shape
            n_s, c_s, h_s, w_s = stride
            _shape = [n, (NPU_OFFSET + c + 63) // 64, 64, h, w]
            _stride = (n_s, c_s, LANE_SIZE // itemsize, h_s, w_s)
            return data_view(_shape, _stride).reshape(n, -1, h, w)[
                :n, NPU_OFFSET : NPU_OFFSET + c, :, :
            ]

    def get_data(self, value):
        if isinstance(value, Scalar):
            return value.data
        assert isinstance(value, MemRef)
        if value.mtype == MType.G:
            return self._ddr_to_numpy(value)
        if value.mtype == MType.R:
            return self._local_mem_to_numpy(value)
        raise ValueError(f"unsupported memory view: {value}")

    def set_data(self, value, data: np.ndarray):
        m_type = value.mtype
        if m_type == MType.G:
            offset = m_type.r_addr
            assert data.dtype == value.np_dtype
            src_u8 = np.ascontiguousarray(data.flatten()).view(np.uint8)
            self.DDR[offset : offset + src_u8.size] = src_u8.flatten()
        if m_type == "L":
            raise Exception("Not implemented.")


@opparam_converter_regitstry("conv_op")
def _converter(reg):
    return ([],) * 3


@opparam_converter_regitstry("pord_op")
def _converter(reg):
    return ([],) * 3


@opparam_converter_regitstry("mm_op")
def _converter(reg):
    return ([],) * 3


@opparam_converter_regitstry("ar_op")
def _converter(reg):
    return ([],) * 3


@opparam_converter_regitstry("mm2_op")
def _converter(reg):
    return ([],) * 3


@opparam_converter_regitstry("cc_op")
def _converter(reg):
    return ([],) * 3


@opparam_converter_regitstry("lut_op")
def _converter(reg):
    return ([],) * 3


@opparam_converter_regitstry("md_sum_op")
def _converter(reg):
    return ([],) * 3


@opparam_converter_regitstry("md_scalar_op")
def _converter(reg):
    return ([],) * 3


@opparam_converter_regitstry("md_sfu_op")
def _converter(reg):
    return ([],) * 3


@opparam_converter_regitstry("md_linear_op")
def _converter(reg):
    return ([],) * 3


@opparam_converter_regitstry("lma_op")
def _converter(reg):
    return ([],) * 3


@opparam_converter_regitstry("decompress_op")
def _converter(reg):
    return ([],) * 3


@opparam_converter_regitstry("md_cmp_op")
def _converter(reg):
    return ([],) * 3


@opparam_converter_regitstry("vc_op")
def _converter(reg):
    return ([],) * 3


@opparam_converter_regitstry("dma_tensor")
def _converter(reg):
    return ([],) * 3
