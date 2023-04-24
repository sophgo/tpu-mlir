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
except:
    import op_support
    from op_support import MType, DType, Scalar, ExtEnum, Layout


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
        if self.mtype == MType.R and layout != Layout.stride:
            self.stride = local_layout_to_stride(self)

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


def Ceil(x, y):
    return (x + y - 1) // y


def AlignY(x, y):
    return Ceil(x, y) * y


def local_layout_to_stride(memref):
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
        n_stride = Ceil(c + memref.mtype.npu_offset, NPU_NUM) * c_stride
        return (n_stride, c_stride, w, 1)

    if memref.layout == Layout.alignEU:
        return compute_stride(True)
    if memref.layout == Layout.compact:
        return compute_stride(False)

    return memref.stride


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
                writeable=False,
            )

        def get_stride_data_base(shape, stride):
            n, c, h, w = shape
            n_s, c_s, h_s, w_s = stride
            _shape = [n, Ceil(NPU_OFFSET + c, NPU_NUM), NPU_NUM, h, w]
            _stride = (n_s, c_s, LANE_SIZE // itemsize, h_s, w_s)
            return data_view(_shape, _stride).reshape(n, -1, h, w)[
                :n, NPU_OFFSET : NPU_OFFSET + c, :, :
            ]

        def get_stride_data():
            return get_stride_data_base(memref.shape, memref.stride)

        def get_4n_2n_data(is_aligned=False):
            def func():
                assert itemsize in (1, 2)
                xn = 4 // itemsize
                n, c, h, w = memref.shape
                if is_aligned:
                    align_type = 128 // memref.itemsize
                    c_stride = AlignY(xn * h * w, align_type)
                else:
                    c_stride = xn * h * w
                lc = Ceil(c + NPU_OFFSET, NPU_NUM)
                shape = (Ceil(n, xn), xn, lc, NPU_NUM, h, w)
                stride = (lc * c_stride, 1, c_stride, LANE_SIZE // itemsize, xn * w, xn)
                return data_view(shape, stride).reshape(shape[0] * shape[1], -1, h, w)[
                    :n, NPU_OFFSET : NPU_OFFSET + c, :, :
                ]

            return func

        get_data = {
            Layout.alignEU: get_stride_data,
            Layout.compact: get_stride_data,
            Layout.stride: get_stride_data,
            Layout.alignEU_XN: get_4n_2n_data(True),
            Layout.compact_XN: get_4n_2n_data(False),
        }
        return get_data[memref.layout]()

    def _get_xn_shape_stride(self, memref):
        assert memref.itemsize in (1, 2)
        xn = 4 // memref.itemsize
        n, *dims = memref.shape
        shape = (Ceil(n, xn), xn, *dims)
        stride = memref.stride
        stride = (stride[0], 1, *stride[1:])
        return np.uint64(shape), np.uint64(stride)

    def _ddr_to_numpy(self, memref):
        assert memref.shape is not None
        assert memref.stride is not None
        assert all(memref.shape)
        assert any(memref.stride)
        offset = memref.mtype.r_addr

        if memref.layout == Layout.continuous_XN:
            n, *dims = memref.shape
            shape, stride = self._get_xn_shape_stride(memref)
            return np.lib.stride_tricks.as_strided(
                self.DDR[offset : offset + 4].view(memref.np_dtype),
                np.ctypeslib.as_array(shape),
                np.ctypeslib.as_array(stride) * memref.itemsize,
                writeable=False,
            ).reshape((-1, *dims))[:n]

        return np.lib.stride_tricks.as_strided(
            self.DDR[offset : offset + 4].view(memref.np_dtype),
            np.ctypeslib.as_array(memref.shape),
            np.ctypeslib.as_array(memref.stride) * memref.itemsize,
            writeable=False,
        )

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
            assert data.dtype == value.np_dtype
            offset = m_type.r_addr
            if value.layout == Layout.continuous_XN:
                assert value.stride is not None
                shape, stride = self._get_xn_shape_stride(value)
                ddr_view = np.lib.stride_tricks.as_strided(
                    self.DDR[offset : offset + 4].view(value.np_dtype),
                    np.ctypeslib.as_array(shape),
                    np.ctypeslib.as_array(stride) * value.itemsize,
                    writeable=True,
                )
                data = data.copy()
                data.resize(shape)
                ddr_view[...] = data
                return

            # continuous memory
            src_u8 = np.ascontiguousarray(data.flatten()).view(np.uint8)
            self.DDR[offset : offset + src_u8.size] = src_u8.ravel()
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
