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
import copy
import functools

try:
    from . import op_support
    from .op_support import MType, DType, Scalar, ExtEnum, Layout
    from .op_support import get_dtype, bf16_to_fp32
except:
    import op_support
    from op_support import MType, DType, Scalar, ExtEnum, Layout
    from op_support import get_dtype, bf16_to_fp32


# Value: MemRef | Scalar
# Scalar: Number
# Number: int | float
# MemRef: address, shape, Dtype, stride, offset?

NPU_SHIFT = 5
NPU_NUM = 32
BANK_NUM = 16
LANE_SIZE = 1 << 17
BANK_SIZE = LANE_SIZE / BANK_NUM
LMEM_SIZE = LANE_SIZE * NPU_NUM
ALIGN_EU_BASE = 16

opparam_converter = {}

__base_eu_num = 4
__eu_num_map = {
    DType.f32: __base_eu_num,
    DType.i32: __base_eu_num,
    DType.si32: __base_eu_num,
    DType.ui32: __base_eu_num,
    DType.f16: __base_eu_num * 2,
    DType.bf16: __base_eu_num * 2,
    DType.i16: __base_eu_num * 2,
    DType.ui16: __base_eu_num * 2,
    DType.si16: __base_eu_num * 2,
    DType.i8: __base_eu_num * 4,
    DType.ui8: __base_eu_num * 4,
    DType.si8: __base_eu_num * 4,
}


def EU_NUM(dtype):
    return __eu_num_map[dtype]


def opparam_converter_regitstry(sheet_name):
    def add_converter(fun):
        if sheet_name in opparam_converter:
            raise KeyError(f"{sheet_name} have already registered.")
        opparam_converter[sheet_name] = fun
        return

    return add_converter


# TPU1686/bm1686/spec/include/memmap.h
memmap = {
    MType.R: (int("0x0", 16), int("0x400000", 16)),  # lmen_base 8M
    # static memory 16KB
    MType.S: (int("0x25800000", 16), int("0x25804000", 16)),
    MType.G: (int("0x100000000", 16), int("0x200000000", 16)),  # global memory
}


def local_layout_to_stride(memref):
    """
    Layout Canonicalize. Convert special layout to stride layout.
    """
    def alignEU_stride():
        _, c, h, w = memref.shape
        align_type = ALIGN_EU_BASE // memref.itemsize
        c_stride = int((w * h + align_type - 1) // align_type * align_type)
        n_stride = int((c + memref.mtype.npu_offset +
                       NPU_NUM - 1) // NPU_NUM * c_stride)
        return (n_stride, c_stride, w, 1)

    def compact_stride():
        _, c, h, w = memref.shape
        c_stride = int(w * h)
        n_stride = int((c + memref.mtype.npu_offset +
                       NPU_NUM - 1) // NPU_NUM * c_stride)
        return (n_stride, c_stride, w, 1)

    def offset_stride():
        return (0, 1, 0, 0)

    def t3_stride():
        _, c, h, w = memref.shape
        eu_num = ALIGN_EU_BASE // memref.itemsize
        h_stride = (w + eu_num - 1) / eu_num * eu_num
        c_stride = h * h_stride
        n_stride = c_stride * \
            (c + memref.mtype.npu_offset + NPU_NUM - 1) // NPU_NUM
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


class MemRef(op_support.MemRef):
    """
    A description of tensor in memory.
    """

    def __init__(self, address, shape, dtype: DType, stride=None, layout=None):
        super().__init__(address, shape, dtype, stride, layout)
        if self.mtype == MType.R and layout != Layout.stride:
            self.stride = local_layout_to_stride(self)

    @staticmethod
    def fix_tag(address, base_addr):
        assert (len(base_addr) == 2)
        fixed_addr = address
        if address & (1 << 39):
            base_addr_idx = (address >> 36) & 0x7
            fixed_addr = (
                address + base_addr[base_addr_idx - 1]) & ((1 << 35) - 1)
        return fixed_addr

    def get_mtype(self, address):
        address = self.fix_tag(address, self.base_addr)
        for k, v in memmap.items():
            if address >= v[0] and address < v[1]:
                if k == MType.R:
                    r_addr = address - v[0]
                    npu_offset = r_addr // LANE_SIZE
                    addr_len = r_addr - npu_offset * LANE_SIZE
                    bank_index = int(addr_len // BANK_SIZE)
                    bank_offset = int(addr_len % BANK_SIZE)
                    # Local memory Type
                    return MType.R(
                        npu_offset=npu_offset,
                        bank_index=bank_index,
                        bank_offset=bank_offset,
                        r_addr=r_addr,
                    )
                return k(r_addr=address - v[0])
        return MType.UNKNOWN

    @property
    @functools.lru_cache()
    def local_shape(self):
        # NPU_OFFSET = self.mtype.npu_offset
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
        NPU_OFFSET = self.mtype.npu_offset

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


class Memory:
    """
    Memory agent. Extract/Set data from a give MemRef object.
    This class should handle all the tenors type in all kinds of storage.
    """

    def __init__(self, LMEM, DDR) -> None:
        self.lmem_list = LMEM
        for lmem in self.lmem_list:
            lmem.ravel()
        self.LMEM = self.lmem_list[0]
        self.DDR = DDR.ravel()

    def _local_mem_to_numpy(self, memref):
        NPU_OFFSET = memref.mtype.npu_offset
        itemsize = memref.itemsize

        # referece: TPU1686/bm1686/cmodel/src/cmodel_common.cpp
        # improve me: use cmodel interface to get data
        def data_view_int4(shape, stride):
            result = np.zeros(shape, dtype=np.uint8).reshape(
                [shape[0], shape[1], -1])
            laddr = memref.mtype.r_addr
            start_npu_idx = NPU_OFFSET
            start_offset = laddr % LANE_SIZE
            for nidx in range(0, shape[0]):
                n_offset = nidx * stride[0]
                for cidx in range(0, shape[1]):
                    npu_idx = (start_npu_idx + cidx) % NPU_NUM
                    LMEM = self.LMEM[npu_idx *
                                     LANE_SIZE: (npu_idx + 1) * LANE_SIZE]
                    c_offset = ((start_npu_idx + cidx)
                                >> NPU_SHIFT) * stride[1]
                    h_offset = np.arange(0, shape[2]) * stride[2]
                    w_offset = np.arange(0, shape[3]) * stride[3]
                    dst_offset = np.add.outer(n_offset, np.add.outer(
                        c_offset, np.add.outer(h_offset, w_offset))).ravel()
                    index = start_offset + (dst_offset >> 1)
                    values = LMEM[index].view(np.uint8)
                    result[nidx][cidx] = np.where(
                        dst_offset & 1 == 0, values & 0xf, values >> 4)
            result.reshape(shape)
            if memref.dtype == DType.si4:
                return np.where(result > 7, result - 16, result).astype(np.int8)
            return data

        def data_view(shape, stride):
            offset = memref.mtype.r_addr - NPU_OFFSET * LANE_SIZE
            return np.lib.stride_tricks.as_strided(
                self.LMEM[offset: offset + 4].view(memref.np_dtype),
                shape,
                np.array(stride) * itemsize,
                writeable=False,
            )

        def get_stride_data_base(shape, stride):
            n, c, h, w = shape
            n_s, c_s, h_s, w_s = stride
            _shape = [n, (NPU_OFFSET + c + NPU_NUM - 1) //
                      NPU_NUM, NPU_NUM, h, w]
            _stride = (n_s, c_s, LANE_SIZE // itemsize, h_s, w_s)
            return data_view(_shape, _stride).reshape(n, -1, h, w)[
                :n, NPU_OFFSET: NPU_OFFSET + c, :, :
            ]

        def get_stride_data():
            if memref.dtype in (DType.i4, DType.si4, DType.ui4):
                return data_view_int4(memref.shape, memref.stride)
            return get_stride_data_base(memref.shape, memref.stride)

        def get_64ic_data():
            n, c, h, w = memref.shape
            shape = ((n + 63) // 64, 64, (c + 63) // 64, 64, h, w)
            stride = (
                (c + 63) // 64 * 64 * h * w,
                LANE_SIZE // itemsize,
                64 * h * w,
                1,
                64 * w,
                64,
            )
            return data_view(shape, stride).reshape(shape[0] * shape[1], -1, h, w)[
                :n, NPU_OFFSET: NPU_OFFSET + c, :, :
            ]

        def get_32ic_data():
            n, c, h, w = memref.shape
            shape = ((n + 63) // 64, 64, (c + 32) // 32, 32, h, w)
            stride = (
                (c + 32) // 32 * 32 * h * w,
                LANE_SIZE // itemsize,
                32 * h * w,
                1,
                32 * w,
                32,
            )
            return data_view(shape, stride).reshape(shape[0] * shape[1], -1, h, w)[
                :n, NPU_OFFSET: NPU_OFFSET + c, :, :
            ]

        def get_1ic_data():
            n, c, h, w = memref.shape
            shape = ((n + 63) // 64, 64, c, h, w)
            stride = (
                c * h * w,
                LANE_SIZE // itemsize,
                h * w,
                w,
                1,
            )
            return data_view(shape, stride).reshape(-1, c, h, w)[
                :n, NPU_OFFSET: NPU_OFFSET + c, :, :
            ]

        def get_matrix_data():
            r, c = memref.shape
            w = memref.layout.args[0]
            shape = (r, (c + w - 1) // w, 1, w)
            _memref = copy.copy(memref)
            _memref.shape = shape
            _memref.layout = Layout.alignEU
            stride = local_layout_to_stride(_memref)
            return get_stride_data_base(shape, stride).reshape(r, -1)[:r, :c]

        def get_matrix2_data():
            r, c = memref.shape
            shape = (1, r, 1, c)
            _memref = copy.copy(memref)
            _memref.shape = shape
            _memref.layout = Layout.alignEU
            stride = local_layout_to_stride(_memref)
            return get_stride_data_base(shape, stride).reshape(r, c)

        def _lane_mask_filter(c, lane_mask):
            lane_mask = np.unpackbits(
                np.uint64([lane_mask]).view(np.uint8), bitorder="little"
            )
            _c = (NPU_OFFSET + c + 63) // 64
            index = np.zeros(_c * 64, bool)
            index[NPU_OFFSET: NPU_OFFSET + c] = True
            index = index.reshape(_c, 64)
            index[:, lane_mask == 0] = False
            return index.flatten()

        def get_dma4bank_data():
            n, c, h, w = memref.shape
            shape = (4, n, (NPU_OFFSET + c + 63) // 64, 64, h, w)
            n_s, c_s, h_s, w_s = memref.stride
            stride = (BANK_SIZE, n_s, c_s, LANE_SIZE // itemsize, h_s, w_s)
            index = _lane_mask_filter(c, memref.layout.args[0])
            return data_view(shape, stride).reshape(4, n, -1, h, w)[:, :, index, :, :]

        def get_dma_stride_data(_memref=memref):
            n, c, h, w = _memref.shape
            shape = (n, (NPU_OFFSET + c + NPU_NUM - 1) //
                     NPU_NUM, NPU_NUM, h, w)
            n_s, c_s, h_s, w_s = _memref.stride
            stride = (n_s, c_s, LANE_SIZE // itemsize, h_s, w_s)
            index = _lane_mask_filter(c, _memref.layout.args[0])
            return data_view(shape, stride).reshape(n, -1, h, w)[:, index, :, :]

        def get_dma_matrix_data():
            r, c = memref.shape
            w = memref.layout.args[1]
            shape = (r, (c + w - 1) // w, 1, w)
            _memref = copy.copy(memref)
            _memref.shape = shape
            return get_dma_stride_data(_memref).reshape(r, -1)[:r, :c]

        def get_dma_linear_data():
            return data_view(memref.shape, memref.stride)

        get_data = {
            Layout.alignEU: get_stride_data,
            Layout.compact: get_stride_data,
            Layout.offset: get_stride_data,
            Layout.stride: get_stride_data,
            Layout._64IC: get_64ic_data,
            Layout._32IC: get_32ic_data,
            Layout._1IC: get_1ic_data,
            Layout.matrix: get_matrix_data,
            Layout.matrix2: get_matrix2_data,
            Layout.T3: get_stride_data,
            Layout.T4: get_stride_data,
            Layout.T5: get_stride_data,
            Layout.DMA4Bank: get_dma4bank_data,
            Layout.DMAstride: get_dma_stride_data,
            Layout.DMAmatrix: get_dma_matrix_data,
            Layout.DMAlinear: get_dma_linear_data,
        }
        data = get_data[memref.layout]()
        if memref.dtype == DType.bf16:
            return bf16_to_fp32(data)
        return data

    def _ddr_to_numpy(self, memref):
        def _ddr_to_numpy_int4(shape, stride):
            result = np.zeros(shape, dtype=np.uint8)
            n_offset = np.arange(shape[0]) * stride[0]
            c_offset = np.arange(shape[1]) * stride[1]
            h_offset = np.arange(shape[2]) * stride[2]
            w_offset = np.arange(shape[3]) * stride[3]
            dst_offset = np.add.outer(n_offset, np.add.outer(
                c_offset, np.add.outer(h_offset, w_offset))).ravel()
            index = memref.mtype.r_addr + (dst_offset >> 1)
            values = self.DDR[index].view(np.uint8)
            result = np.where(dst_offset & 1 == 0, values &
                              0xf, values >> 4).reshape(shape)
            if memref.dtype == DType.si4:
                return np.where(result > 7, result - 16, result).astype(np.int8)
            return result
        assert memref.shape is not None
        assert memref.stride is not None
        assert all(memref.shape)
        assert any(memref.stride)
        offset = memref.mtype.r_addr
        if memref.dtype in (DType.i4, DType.si4, DType.ui4):
            return _ddr_to_numpy_int4(memref.shape, memref.stride)
        data = np.lib.stride_tricks.as_strided(
            self.DDR[offset: offset + 4].view(memref.np_dtype),
            np.ctypeslib.as_array(memref.shape),
            np.ctypeslib.as_array(memref.stride) * memref.itemsize,
            writeable=False,
        )
        if memref.dtype == DType.bf16:
            return bf16_to_fp32(data)
        return data

    def get_data(self, value):
        if isinstance(value, Scalar):
            return value.data
        assert isinstance(value, MemRef)
        if value.mtype == MType.G:
            return self._ddr_to_numpy(value)
        if value.mtype == MType.R:
            self.LMEM = self.lmem_list[value.core_id]
            return self._local_mem_to_numpy(value)
        raise ValueError(f"unsupported memory view: {value}")

    def set_data(self, value, data: np.ndarray):
        m_type = value.mtype
        if m_type == MType.G:
            offset = m_type.r_addr
            assert data.dtype == value.np_dtype
            src_u8 = np.ascontiguousarray(data.flatten()).view(np.uint8)
            self.DDR[offset: offset + src_u8.size] = src_u8.flatten()
            return
        raise NotImplementedError(f"Not support setting {m_type} memory data.")


def get_value(
    address=-1,
    shape=None,
    stride=None,
    dtype=None,
    layout=None,
    is_const=False,
):
    if dtype is None:
        raise ValueError("The dtype of this tensor is invalid.")
    _dtype = dtype
    if not isinstance(dtype, DType):
        _dtype = get_dtype(*dtype)
    if is_const:
        return Scalar(address, _dtype)
    else:
        _layout = layout
        if not isinstance(layout, ExtEnum):
            _layout = Layout(layout)
        return MemRef(address, shape, _dtype, stride, _layout)


class TGCR:

    def __init__(self):
        self.regs = dict(
            T5=0,
            T6=0,
            T32=0,
            T33=0,
            T127=0,
        )

    def setter(self, index, value):
        self.regs["T" + str(index)] = value

    def getter(self, index):
        return int(self.regs["T" + str(index)])


tgcr = TGCR()


@opparam_converter_regitstry("sCONV")
def _converter(reg):
    FP16 = 1
    FP32 = 2
    BF16 = 5
    INT4 = 6
    opd0 = dict(
        address=reg.opd0_addr,
        shape=(reg.res0_n, reg.opd0_c, reg.opd0_h, reg.opd0_w),
        stride=[reg[f"opd0_{i}_str"] for i in "nchw"],
        dtype=(reg.opt_opd0_prec, reg.opt_opd0_sign),
        layout=reg.short_opd0_str,
    )
    opd1 = dict(
        address=reg.opd1_addr,
        shape=[reg.res0_c, reg.opd0_c, reg.opd1_h, reg.opd1_w],
        dtype=(reg.opt_opd0_prec, reg.opt_opd1_sign),
        is_const=reg.opt_opd1_const,
        layout=Layout._1IC,
    )
    opd2 = dict(
        address=reg.opd2_addr,
        is_const=reg.opt_opd2_const,
        shape=[1, reg.res0_c, 1, 1],
        layout=Layout.compact,
        dtype=(DType.i32, reg.opt_opd2_sign),
    )
    opd3 = dict(
        address=reg.opd3_addr,
        is_const=reg.opt_opd3_const,
        shape=[1, reg.res0_c, 1, 2],
        layout=Layout.compact,
        dtype=(reg.opt_opd0_prec, reg.opt_opd0_sign)
    )
    opd4 = dict(
        address=tgcr.getter(5),
        is_const=reg.opt_opd4_const,
        shape=[1, reg.res0_c, 1, 1],
        layout=Layout.compact,
        dtype=(DType.i16, 1),
    )
    opd5 = dict(
        address=tgcr.getter(6),
        shape=[1, reg.res0_c, 1, 2],
        is_const=reg.opt_opd5_const,
        layout=Layout.compact,
        dtype=(DType.i32, 1),
    )
    res0 = dict(
        address=reg.res0_addr,
        shape=[reg[f"res0_{i}"] for i in "nchw"],
        dtype=(reg.opt_res0_prec,
               reg.opt_opd0_sign or reg.opt_opd1_sign or reg.opt_opd2_sign),
        layout=Layout.alignEU,
    )
    opds = [opd0, opd1, opd2, opd3, opd4, opd5]
    if reg.opt_opd0_prec == INT4 and reg.opt_opd1_prec == INT4:
        opd1["layout"] = Layout._64IC
        opd3["shape"] = [1, reg.res0_c, 1, 1]
        opd3["dtype"] = (DType.i8, 1)
    elif reg.opt_opd0_prec == FP16 or reg.opt_opd0_prec == BF16:
        opd1["layout"] = Layout._16IC
    elif reg.opt_opd0_prec == FP32:
        opd1["layout"] = Layout._1IC
    else:
        opd1["layout"] = Layout._32IC
    results = [get_value(**res0)]

    attr = dict(
        kernel=[reg.opd1_h, reg.opd1_w],
        stride=[reg.res_op_y_str, reg.res_op_x_str],
        in_zero=[reg.opd0_y_ins0, reg.opd0_x_ins0],
        ke_zero=[reg.opd1_y_ins0, reg.opd1_x_ins0],
        opt_kernel_rotate=bool(reg.opt_kernel_rotate),
        pad_mode=reg.pad_mode,
        pad=[reg[f"opd0_{x}_pad"] for x in ("up", "dn", "lf", "rt")],
        opt_res_add=bool(reg.opt_res_add),
        do_relu=bool(reg.opt_relu),
        sym_range=bool(reg.sym_range),
        do_rq=bool(reg.opt_rq),
        round_mode=reg.opd2_n_str,
    )
    if bool(reg.opt_rq) == False:
        opds.remove(opd4)
        opds.remove(opd5)
    else:
        if bool(reg.opt_opd4_const):
            opds.remove(opd4)
            attr["kzp"] = int(tgcr.getter(5))
        if bool(reg.opt_opd5_const):
            opds.remove(opd5)
            attr["multiplier"] = tgcr.getter(6)
            attr["shift"] = int(np.binary_repr(
                tgcr.getter(32), width=32)[-8:-1], 2)
            attr["yzp"] = int(np.binary_repr(
                tgcr.getter(33), width=32)[-15:-1], 2)
    operands = [get_value(**x) for x in opds]
    return (results, attr, operands)


@opparam_converter_regitstry("sMM")
def _converter(reg):
    L_row = reg.opd0_n
    L_col = reg.opd0_w * (reg.opd0_c - 1) + reg.opd1_w
    R_col = reg.res0_c * reg.res0_w
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=(reg.opt_opd0_prec, reg.opt_opd0_sign),
        shape=(L_row, L_col),
        layout=Layout.matrix(reg.opd0_w),
        is_const=reg.opt_opd0_const,
    )
    if reg.opt_left_tran:
        L_row, L_col = L_col, L_row
    res0 = dict(
        address=reg.res0_addr,
        dtype=(reg.opt_res0_prec, 1),
        shape=(L_row, R_col),
        layout=Layout.matrix(reg.res0_w),
    )
    opd1 = dict(
        address=reg.opd1_addr,
        dtype=(reg.opt_opd0_prec, reg.opt_opd1_sign),
        shape=(L_col, R_col),
        layout=Layout.matrix(reg.res0_w),
    )
    opd2 = dict(
        address=reg.opd2_addr,
        is_const=reg.opt_opd2_const,
        shape=(1, R_col),
        layout=Layout.matrix(reg.res0_w),
    )
    attr = dict(
        l_trans=bool(reg.opt_left_tran),
        res_add=bool(reg.opt_res_add),
        do_relu=bool(reg.opt_relu),
        sym_range=bool(reg.sym_range),
        do_rq=bool(reg.opt_rq),
        round_mode=reg.opd2_n_str,
        multiplier=tgcr.getter(6),
        shift=int(np.binary_repr(tgcr.getter(32), width=32)[-7:-1], 2),
        yzp=int(np.binary_repr(tgcr.getter(33), width=32)[-16:-1], 2),
    )
    assert (reg.tsk_eu_typ == 1)  # mm_normal
    if reg.opt_opd0_prec == DType.f32:
        opd2["dtype"] = opd0["dtype"]  # bias
    else:
        opd2["dtype"] = (DType.i8, 1)  # shift

    operands = [get_value(**x) for x in (opd0, opd1, opd2)]
    results = [get_value(**res0)]

    return (results, attr, operands)


@opparam_converter_regitstry("sMM2")
def _converter(reg):
    L_row = reg.res0_c
    L_col = reg.opd1_c
    l_trans, r_trans = False, False
    if reg.tsk_eu_typ == 5:
        L_col = reg.opd1_w
        l_trans, r_trans = False, True
    elif reg.tsk_eu_typ == 6:
        L_row = reg.opd1_w
        L_col = reg.res0_c
        l_trans, r_trans = True, True
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=(reg.opt_opd0_prec, reg.opt_opd0_sign),
        shape=(L_row, L_col),
        layout=Layout.matrix2,
        is_const=reg.opt_opd0_const,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=(reg.opt_res0_prec, 1),
        shape=(reg.res0_c, reg.res0_w),
        layout=Layout.matrix2,
    )
    opd1 = dict(
        address=reg.opd1_addr,
        dtype=(reg.opt_opd0_prec, reg.opt_opd1_sign),
        shape=(reg.opd1_c, reg.opd1_w),
        layout=Layout.matrix2,
    )
    opd2 = dict(
        address=reg.opd2_addr,
        is_const=reg.opt_opd2_const,
        shape=(1, 32, 1, reg.res0_w),
        layout=Layout.alignEU,
    )
    opd4 = dict(
        address=tgcr.getter(5),
        is_const=reg.opt_opd4_const,
        dtype=DType.si16,
        shape=(1, 32, 1, reg.res0_w),
        layout=Layout.alignEU,
    )
    opd5 = dict(
        address=tgcr.getter(6),
        is_const=reg.opt_opd5_const,
        dtype=DType.si32,
        shape=(1, 32, reg.res0_w, 2),
    )
    attr = dict(
        l_trans=bool(l_trans),
        r_trans=bool(r_trans),
        res_add=bool(reg.opt_res_add),
        do_relu=bool(reg.opt_relu),
        sym_range=bool(reg.sym_range),
        do_rq=bool(reg.opt_rq),
        round_mode=reg.opd2_n_str,
    )
    opds = []
    if reg.tsk_eu_typ in [4, 5]:
        opds = [opd0, opd1, opd2, opd4, opd5]
    else:
        opd2["shape"] = (1, reg.res0_w, 1, 1)
        opd2["layout"] = Layout.compact
        opd4["shape"] = (1, reg.res0_w, 1, 1)
        opd4["layout"] = Layout.compact
        opd5["shape"] = (1, reg.res0_w, 1, 2)
        opd5["layout"] = Layout.compact
    if reg.opt_opd0_prec in (0, 6):
        opd2["dtype"] = (DType.i32, reg.opt_opd2_sign)
        opd5["dtype"] = DType.i32
        opds = [opd0, opd1, opd2, opd4, opd5]
        if reg.opt_opd4_const:
            opds.remove(opd4)
            attr["zp"] = int(np.binary_repr(
                tgcr.getter(5), width=32)[-16:-1], 2)
        if reg.opt_opd5_const:
            opds.remove(opd5)
            attr["multiplier"] = tgcr.getter(6)
            attr["shift"] = int(np.binary_repr(
                tgcr.getter(32), width=32)[-8:-1], 2)
            attr["yzp"] = int(np.binary_repr(
                tgcr.getter(33), width=32)[-16:-1], 2)
    else:
        opd2["dtype"] = DType.f32
        opds = [opd0, opd1, opd2]
    # if reg.opt_opd0_prec == 0:
    operands = [get_value(**x) for x in opds]
    results = [get_value(**res0)]

    return (results, attr, operands)


@opparam_converter_regitstry("sCMP")
def _converter(reg):
    shape = tuple(reg[f"res0_{d}"] for d in "nchw")
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=(reg.opt_opd0_prec, reg.opt_opd0_sign),
        shape=shape,
        layout=reg.short_opd0_str,
        is_const=reg.opt_opd0_const,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=opd0["dtype"],
        shape=shape,
        layout=Layout.alignEU,
    )
    res1 = dict(
        address=reg.res1_addr,
        dtype=DType(reg.opt_opd2_prec),
        shape=shape,
        layout=Layout.alignEU,
    )
    opd1 = dict(
        address=reg.opd1_addr,
        dtype=opd0["dtype"],
        shape=shape,
        layout=reg.short_opd1_str,
        is_const=reg.opt_opd1_const,
    )
    opd2 = dict(
        address=reg.opd2_addr,
        dtype=DType(reg.opt_opd2_prec),
        shape=shape,
        layout=Layout.alignEU,
        is_const=reg.opt_opd2_const,
    )
    opd3 = dict(
        address=reg.opd3_addr,
        dtype=DType(reg.opt_opd2_prec),
        shape=shape,
        layout=Layout.alignEU,
        is_const=reg.opt_opd3_const,
    )

    rets = [res0, res1]
    if reg.short_opd0_str == Layout.stride:
        opd0["stride"] = (0, 0, 0, 0)
    if reg.short_opd1_str == Layout.stride:
        opd1["stride"] = (0, 0, 0, 0)
    if reg.tsk_eu_typ in (23, 24, 26):
        res0["dtype"] = res1["dtype"]
        rets = [res0]

    operands = [get_value(**x) for x in (opd0, opd1, opd2, opd3)]
    results = [get_value(**x) for x in rets]

    return (results, {}, operands)


@opparam_converter_regitstry("sSFU")
def _converter(reg):
    shape = tuple(reg[f"res0_{d}"] for d in "nchw")
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=(reg.opt_opd0_prec, 1),
        shape=shape,
        layout=Layout.alignEU,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=(reg.opt_res0_prec, 1),
        shape=shape,
        layout=Layout.alignEU,
    )
    opd1 = dict(
        address=reg.opd1_addr,
        dtype=(reg.opt_opd0_prec, 1),
        shape=(1, min(shape[1], 64), 1, reg.opd1_n),
        stride=(0, 0, reg.opd1_n, 1),
        layout=Layout.stride,
    )
    attr = {}
    opd_num = 1
    if reg.tsk_eu_typ == 17:
        attr = dict(rsqrt_n=reg.opd2_n_str + 1)
    elif reg.tsk_eu_typ in [12, 13]:
        opd_num = 2

    operands = [get_value(**x) for x in (opd0, opd1)[:opd_num]]
    results = [get_value(**res0)]

    return (results, attr, operands)


@opparam_converter_regitstry("sLIN")
def _converter(reg):
    shape = tuple(reg[f"res0_{d}"] for d in "nchw")
    c = shape[1]
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=(reg.opt_res0_prec, 1),
        shape=shape,
        layout=Layout.alignEU,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=(reg.opt_res0_prec, 1),
        shape=shape,
        layout=Layout.alignEU,
    )
    opd1 = dict(
        address=reg.opd1_addr,
        dtype=opd0["dtype"],
        shape=(1, c, 1, 1),
        layout=Layout.compact,
        is_const=reg.opt_opd1_const,
    )
    opd2 = dict(
        address=reg.opd2_addr,
        dtype=opd0["dtype"],
        shape=(1, c, 1, 1),
        layout=Layout.compact,
        is_const=reg.opt_opd2_const,
    )

    opd_num = 2
    if reg.tsk_eu_typ == 1:
        opd_num = 3

    operands = [get_value(**x) for x in (opd0, opd1, opd2)[:opd_num]]
    results = [get_value(**res0)]

    return (results, {}, operands)


@opparam_converter_regitstry("sVC")
def _converter(reg):
    n = (reg.opd0_c - 1) * reg.opd0_w + reg.opd1_w
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=(reg.opt_opd0_prec, reg.opt_opd0_sign),
        shape=(1, reg.opd0_c, 1, reg.opd0_w),
        layout=Layout.alignEU,
    )

    res0 = dict(
        address=reg.res0_addr,
        dtype=(reg.opt_res0_prec, 1),
        shape=(n, reg.res0_c, 1, reg.res0_w),
        layout=Layout.alignEU,
    )
    opd1 = dict(
        address=reg.opd1_addr,
        dtype=(reg.opt_opd1_prec, reg.opt_opd1_sign),
        shape=(1, reg.res0_c, 1, reg.res0_w),
        layout=Layout.alignEU,
    )
    attr = {}
    if reg.tsk_eu_typ == 23:
        attr = dict(round_mode=reg.opd2_n_str)

    operands = [get_value(**x) for x in (opd0, opd1)]
    results = [get_value(**res0)]

    return (results, attr, operands)


def restore_org_shape(operand_def):
    shape = list(operand_def["shape"])
    if operand_def["layout"] == Layout.stride:
        stride = operand_def["stride"]
        assert len(shape) == len(stride)
        for i, t in enumerate(stride):
            if t == 0:
                shape[i] = 1
    return shape


@opparam_converter_regitstry("sAR")
def _converter(reg):
    n, c, h, w = (reg[f"res0_{d}"] for d in "nchw")
    # round mm
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=(reg.opt_opd0_prec, reg.opt_opd0_sign),
        shape=(n, c, h, w),
        stride=tuple(reg[f"opd0_{d}_str"] for d in "nchw"),
        layout=reg.short_opd0_str,
        is_const=reg.opt_opd0_const,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=(reg.opt_res0_prec, reg.opt_opd0_sign or reg.opt_opd1_sign),
        shape=(n, c, h, w),
        stride=tuple(reg[f"res0_{d}_str"] for d in "nchw"),
        layout=reg.short_res0_str,
    )
    opd1 = dict(
        address=reg.opd1_addr,
        dtype=(reg.opt_opd1_prec, reg.opt_opd1_sign),
        shape=(n, c, h, w),
        stride=tuple(reg[f"opd1_{d}_str"] for d in "nchw"),
        layout=reg.short_opd1_str,
        is_const=reg.opt_opd1_const,
    )
    opd2 = dict(
        address=reg.opd2_addr,
        dtype=(reg.opt_opd2_prec, reg.opt_opd2_sign),
        shape=(1, c, 1, 1),
        layout=Layout.compact,
        is_const=reg.opt_opd2_const,
    )

    if reg.tsk_eu_typ == 14:  # DATA_CONVERT
        res0["dtype"] = (reg.opt_res0_prec, reg.opt_opd2_sign)

    if reg.tsk_eu_typ == 12:
        attr = dict(iter=reg.opd2_n_str + 1)
    else:
        attr = dict(round_mode=reg.opd2_n_str)

    opd0["shape"] = restore_org_shape(opd0)
    opd1["shape"] = restore_org_shape(opd1)

    opd_num = reg.tsk_opd_num
    operands = [get_value(**x) for x in (opd0, opd1, opd2)[:opd_num]]
    results = [get_value(**res0)]

    return (results, attr, operands)


@opparam_converter_regitstry("sPorD")
def _converter(reg):
    n, c, h, w = (reg[f"res0_{d}"] for d in "nchw")
    # round mm
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=(reg.opt_opd0_prec, reg.opt_opd0_sign),
        shape=(n, c, reg.opd0_h, reg.opd0_w),
        layout=Layout.alignEU,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=(reg.opt_res0_prec, reg.opt_opd0_sign),
        shape=(n, c, h, w),
        layout=Layout.alignEU,
    )
    opd1 = dict(
        address=reg.opd1_addr,
        dtype=(reg.opt_opd0_prec, reg.opt_opd1_sign),
        shape=(1, c, reg.opd1_h, reg.opd1_w),
        layout=Layout.compact,
        is_const=reg.opt_opd1_const,
    )
    opd2 = dict(
        address=reg.opd2_addr,
        dtype=(reg.opt_opd0_prec, reg.opt_opd2_sign),
        shape=(1, c, 1, 1),
        layout=Layout.compact,
        is_const=reg.opt_opd2_const,
    )
    # padding
    opd3 = dict(
        address=reg.opd3_addr,
        dtype=(reg.opt_opd0_prec, reg.opt_opd0_sign),
        shape=(1, c, 1, 2),
        layout=Layout.compact,
        is_const=reg.opt_opd3_const,
    )

    opd5 = dict(
        address=0,
        dtype=DType.si32,
        shape=(1, c, 1, 2),
        layout=Layout.compact,
        is_const=reg.opt_opd5_const,
    )

    opds = []
    if reg.tsk_eu_typ in [0, 4, 3, 1]:
        # if reg.tsk_eu_typ in [0, 1]:
        if reg.tsk_eu_typ == 0:
            opds = [opd0, opd1, opd2, opd3, opd5]
        elif reg.tsk_eu_typ == 1:
            opds = [opd0, opd1, opd3, opd5]
        else:
            opds = [opd0, opd3, opd5]
    else:
        opd3["shape"] = (1, c, 1, 4)
        opd3["dtype"] = DType.ui16
        if reg.tsk_eu_typ in [5, 6]:
            opds = [opd0, opd1, opd2, opd3, opd5]
        else:
            opds = [opd0, opd2, opd3, opd5]
    attr = dict(
        kernel=[reg.opd1_h, reg.opd1_w],
        stride=[reg.res_op_y_str, reg.res_op_x_str],
        in_zero=[reg.opd0_y_ins0, reg.opd0_x_ins0],
        ke_zero=[reg.opd1_y_ins0, reg.opd1_x_ins0],
        opt_kernel_rotate=bool(reg.opt_kernel_rotate),
        pad_mode=reg.pad_mode,
        pad=[reg[f"opd0_{x}_pad"] for x in ("up", "dn", "lf", "rt")],
        round_mode=reg.opd2_n_str,
        shift=np.uint32([reg.res1_addr]).view(np.int8)[0],
    )
    if bool(reg.opt_rq) == False:
        opds.remove(opd5)
    else:
        if bool(reg.opt_opd5_const):
            opds.remove(opd5)
            attr["multiplier"] = tgcr.getter(6)
            attr["shift"] = int(np.binary_repr(
                tgcr.getter(32), width=32)[-8:-1], 2)
            attr["yzp"] = int(np.binary_repr(
                tgcr.getter(33), width=32)[-16:-1], 2)

    operands = [get_value(**x) for x in opds]
    results = [get_value(**res0)]

    return (results, attr, operands)


def cw_trans_reg_format(reg):
    n, c, h, w = (reg[f"res0_{d}"] for d in "nchw")
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=DType(reg.opt_res0_prec),
        shape=(n, w, h, c),
        layout=Layout.alignEU,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=DType(reg.opt_res0_prec),
        shape=(n, c, h, w),
        layout=Layout.alignEU,
    )

    if reg.tsk_eu_typ == 0:  # cw_ts
        opd0["layout"] = Layout.T3

    operands = [get_value(**opd0)]
    results = [get_value(**res0)]

    return (results, {}, operands)


def bc_reg_format(reg):
    n, c, h, w = (reg[f"res0_{d}"] for d in "nchw")
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=DType(reg.opt_res0_prec),
        shape=(n, c, h, w),
        layout=Layout.alignEU,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=DType(reg.opt_res0_prec),
        shape=(n, c, h, w),
        layout=Layout.alignEU,
    )

    if reg.tsk_eu_typ == 3:
        opd0["shape"] = (n, 1, h, w)
    if reg.tsk_eu_typ == 4:
        res0["shape"] = (1, c, 1, w)
    if reg.tsk_eu_typ == 5:
        res0["shape"] = (1, c, 1, 1)
        res0["layout"] = Layout.compact

    operands = [get_value(**opd0)]
    results = [get_value(**res0)]

    return (results, {}, operands)


@opparam_converter_regitstry("sCW&sBC")
def _converter(reg):
    if reg.tsk_eu_typ in (0, 1):
        return cw_trans_reg_format(reg)
    else:
        return bc_reg_format(reg)


@opparam_converter_regitstry("sRQ&sDQ")
def _converter(reg):
    n, c, h, w = (reg[f"res0_{d}"] for d in "nchw")
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=(reg.opt_opd0_prec, reg.opt_opd0_sign),
        shape=(n, c, h, w),
        layout=Layout.alignEU,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=(reg.opt_res0_prec, reg.opt_opd2_sign),
        shape=(n, c, h, w),
        layout=Layout.alignEU,
    )
    opd1 = dict(
        address=reg.opd1_addr,
        is_const=reg.opt_opd1_const,
        layout=Layout.compact,
    )
    opds = []
    if reg.tsk_eu_typ == 0:  # rq_0
        opd1["dtype"] = DType.f32
        opd1["shape"] = (1, c, 1, 1)
        opd2 = dict(opd1)
        opd2["address"] += 4
        if opd1["is_const"]:
            opd2["address"] = reg.opd2_addr
        opds = [opd0, opd1, opd2]
    elif reg.tsk_eu_typ == 1:  # rq_1
        opd1["dtype"] = DType.si32
        opd1["shape"] = (1, c, 1, 1)
        opd2 = dict(opd1)
        opd2["address"] += 4
        opd3 = dict(opd2)
        opd3["address"] += 4
        if opd1["is_const"]:
            opd2["address"] = reg.opd2_addr
            opd2["dtype"] = DType.si8
            opd3["address"] = reg.opd2_addr // 2**8
            opd3["dtype"] = DType.si16
        opds = [opd0, opd1, opd2, opd3]
    elif reg.tsk_eu_typ == 3:  # dq_0
        opd1["dtype"] = DType.si16
        opd1["shape"] = (1, c, 1, 1)
        opd2 = dict(opd1)
        opd2["address"] += 4
        opd2["shape"] = (1, c, 1, 1)
        opd2["dtype"] = DType.f32
        if opd1["is_const"]:
            opd2["address"] = reg.opd2_addr
        opds = [opd0, opd1, opd2]
    elif reg.tsk_eu_typ == 4:  # dq_1
        opd1["dtype"] = DType.si32  # multiplier
        opd1["shape"] = (1, c, 1, 1)
        opd2 = dict(opd1)  # shift
        opd2["dtype"] = DType.si8
        opd2["address"] += 4
        opd3 = dict(opd2)  # yzp
        opd3["address"] += 4
        opds = [opd0, opd1, opd2, opd3]
        if opd1["is_const"]:
            opd1["dtype"] = (DType.i16, reg.opt_opd0_sign)  # multiplier
            opd2["address"] = reg.opd2_addr // 2**16  # shift
            opd2["dtype"] = DType.si16
            opd3["address"] = reg.opd2_addr  # scale
            opd3["dtype"] = DType.si8
            opds = [opd0, opd1, opd3, opd2]
    else:
        raise KeyError("Should not be here.")
    attr = dict(round_mode=reg.opd2_n_str)
    operands = [get_value(**x) for x in opds]
    results = [get_value(**res0)]

    return (results, attr, operands)


@opparam_converter_regitstry("sSG")
def _converter(reg):
    n, c, h, w = (reg[f"res0_{d}"] for d in "nchw")
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=DType(reg.opt_opd0_prec),
        layout=reg.short_opd0_str,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=DType(reg.opt_res0_prec),
        shape=(n, c, h, w),
        layout=Layout.alignEU,
    )
    opd1 = dict(
        address=reg.opd0_addr,
        dtype=(reg.opt_opd1_prec, 0),
        layout=Layout.alignEU,
    )
    opds = [opd0, opd1]
    rets = [res0]
    if reg.tsk_eu_typ in [0, 3]:
        opd0["shape"] = (n, c, 1, reg.opd0_w)
        opd1["shape"] = (1, reg.opd1_c, 1, 1)
        opd1["layout"] = Layout.compact
    elif reg.tsk_eu_typ in [1, 4]:
        opd0["shape"] = (n, c, reg.opd0_h, reg.opd0_w)
        opd1["shape"] = (1, reg.opd1_c, 1, 1)
        opd1["layout"] = Layout.compact
    elif reg.tsk_eu_typ in [5, 6, 13, 14]:
        opd0["shape"] = (1, c, reg.opd0_h, reg.opd0_w)
        if reg.short_opd0_str != 0:
            opd0["layout"] = Layout.T4
        opd1["shape"] = (n, c, 1, reg.opd1_w)
        opd1["layout"] = Layout.alignEU
    elif reg.tsk_eu_typ in [8, 15]:
        opd0["shape"] = (1, c, 1, reg.opd0_w)
        if reg.short_opd0_str != 0:
            opd0["layout"] = Layout.T4
        opd1["shape"] = (n, c, 1, reg.opd1_w)
        res1 = dict(
            address=reg.res1_addr,
            dtype=DType.si16,
            shape=(n, c, 1, 1),
            layout=Layout.compact,
        )
        rests = [res0, res1]
    elif reg.tsk_eu_typ in [9, 16]:
        opd0["shape"] = (n, c, 1, reg.opd0_w)
        opds = [opd0]
        res1 = dict(
            address=reg.res1_addr,
            dtype=DType.si16,
            shape=(n, c, 1, 1),
            layout=Layout.compact,
        )
        rests = [res0, res1]
    else:
        raise KeyError("Should not be here.")

    attr = dict(
        limit_enable=bool(reg.opt_opd3_const),
        fill_const=bool(reg.opt_opd2_const),
        const_value=reg.opd2_addr,
    )
    operands = [get_value(**x) for x in opds]
    results = [get_value(**x) for x in rets]

    return (results, attr, operands)


@opparam_converter_regitstry("sSGL")
def _converter(reg):
    n, c, h, w = (reg[f"res0_{d}"] for d in "nchw")
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=DType(reg.opt_res0_prec),
        layout=Layout.stride,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=DType(reg.opt_res0_prec),
        shape=(n, c, h, w),
        layout=Layout.stride,
    )
    opd1 = dict(
        address=reg.opd0_addr,
        dtype=(reg.opt_opd1_prec, 0),
        layout=Layout.compact,
    )
    opd0["shape"] = (1, c, reg.opd0_h, w)
    if reg.short_short_opd0_str == 3:
        opd0["layout"] = Layout.T3
    else:
        opd0["layout"] = Layout.T4

    rets = [res0]
    assert (reg.tsk_eu_typ in [17, 18])
    opd1_h = reg.res0_h if reg.tsk_eu_typ == 17 else reg.opd0_h
    opd1["shape"] = (n, c, opd1_h, 1)
    res0["layout"] = Layout.T3

    attr = dict(
        limit_enable=bool(reg.opt_opd3_const),
        fill_const=bool(reg.opt_opd2_const),
        const_value=reg.opd2_addr,
    )

    operands = [get_value(**x) for x in (opd0, opd1)]
    results = [get_value(**x) for x in rets]

    return (results, attr, operands)


@opparam_converter_regitstry("SYS_TR_ACC")
def _converter(reg):
    def int2bin(x, width):
        return np.binary_repr(x, width=width)
    des_imm0 = int2bin(reg.imm0, 32)
    des_imm0 = int(des_imm0, 2)
    des_imm1 = int2bin(reg.imm1, 64)
    des_imm1_h32 = int(des_imm1[0:32], 2)
    des_imm1_l32 = int(des_imm1[32:64], 2)
    tgcr.setter(reg.reg_idx0, des_imm0)
    tgcr.setter(reg.reg_idx1, des_imm1_h32)
    tgcr.setter(reg.reg_idx2, des_imm1_l32)
    attrs = dict(
        reg_idx0=reg.reg_idx0,
        reg_idx1=reg.reg_idx1,
        reg_idx2=reg.reg_idx2,
        des_imm0=des_imm0,
        des_imm1_h32=des_imm1_h32,
        des_imm1_l32=des_imm1_l32,
    )
    return ([], attrs, [])


@opparam_converter_regitstry("SYS")
def _converter(reg):
    des_imm = reg.imm
    msg_id = des_imm & 0x7f
    cnt = des_imm >> 16 & 0x7
    if reg.tsk_eu_typ in (8, 9):
        attrs = dict(
            msg_id=msg_id,
            cnt=cnt,
        )
    else:
        raise KeyError("Should not be here.")
    return ([], attrs, [])


def dma_addr(H, L):
    return H * 2 ** 32 + L


def dma_reg_fmt_base(reg):
    lane_mask = reg.localmem_mask_h32 * 2**32 + reg.localmem_mask_l32
    opd0 = dict(
        address=dma_addr(reg.src_start_addr_h8, reg.src_start_addr_l32),
        dtype=DType(reg.src_data_format),
        shape=tuple(reg[f"src_{d}size"] for d in "nchw"),
        stride=tuple(reg[f"src_{d}stride"] for d in "nchw"),
        layout=Layout.DMAstride(lane_mask),
    )
    res0 = dict(
        address=dma_addr(reg.dst_start_addr_h8, reg.dst_start_addr_l32),
        dtype=DType(reg.src_data_format),
        shape=tuple(reg[f"dst_{d}size"] for d in "nchw"),
        stride=tuple(reg[f"dst_{d}stride"] for d in "nchw"),
        layout=Layout.DMAstride(lane_mask),
    )
    if reg.nchw_copy:
        res0["shape"] = opd0["shape"]

    attr = dict()
    if lane_mask != (2**64 - 1):
        attr["lane_mask"] = hex(lane_mask)

    if reg.fill_constant_en:
        attr = {}
        opd0 = dict(
            address=reg.constant_value, dtype=DType(reg.src_data_format), is_const=True
        )

    return res0, attr, opd0


@opparam_converter_regitstry("DMA_tensor（0x000）")
def _converter(reg):
    NONE = 0
    TRANS = 1  # NC Transpose or Matrix Transpose
    COLLECT = 2  # CW Transpose from lmem to gmem
    BROADCAST = 3
    DISTRIBUTE = 4  # CW Transpose from gmem to lmem
    res0, attr, opd0 = dma_reg_fmt_base(reg)
    if reg.nchw_copy:
        res0["shape"] = opd0["shape"]

    if reg.cmd_special_function == BROADCAST:  # broadcast
        n, _, h, w = res0["shape"]
        res0["shape"] = (n, reg.src_csize, h, w)
    elif reg.cmd_special_function == TRANS:  # transpose
        n, c, h, w = opd0["shape"]
        res0["shape"] = (c, n, h, w)
    elif reg.cmd_special_function in (COLLECT, DISTRIBUTE):  # cw transpose
        n, c, h, w = opd0["shape"]
        res0["shape"] = (n, w, h, c)

    if reg.cmd_special_function != NONE:
        opd0["stride"] = (*opd0["stride"][:-1], 1)
        res0["stride"] = (*res0["stride"][:-1], 1)

    if reg.cmd_special_function in (BROADCAST, DISTRIBUTE):
        # disable lane mask
        opd0["layout"] = Layout.stride
        res0["layout"] = Layout.stride

    operands = [get_value(**opd0)]
    results = [get_value(**res0)]

    return (results, attr, operands)


@opparam_converter_regitstry("DMA_matrix")
def _converter(reg):
    res0, attr, opd0 = dma_reg_fmt_base(reg)
    lane_mask = opd0["layout"].args[0]
    l, r = memmap[MType.R]
    s_addr = opd0["address"]
    is_trans = reg.cmd_special_function == 1
    if s_addr >= l and s_addr < r and (reg.fill_constant_en == 0):
        # glocal
        _, _, H, W = res0["shape"]
        res0["shape"] = (1, 1, H, W)
        _, _, h, _ = res0["stride"]
        res0["stride"] = (0, 0, h, 1)
        # local
        if is_trans:
            H, W = W, H
        opd0["shape"] = (H, W)
        opd0["layout"] = Layout.DMAmatrix(lane_mask, reg.src_wsize)
        n, c, _, _ = opd0["stride"]
        opd0["stride"] = (n, c, 0, 1)
    else:
        # glocal
        _, _, H, W = opd0["shape"]
        opd0["shape"] = (1, 1, H, W)
        _, _, h, _ = opd0["stride"]
        opd0["stride"] = (0, 0, h, 1)
        # local
        if is_trans:
            H, W = W, H
        res0["shape"] = (H, W)
        n, c, _, _ = res0["stride"]
        res0["stride"] = (n, c, 0, 1)
        res0["layout"] = Layout.DMAmatrix(lane_mask, reg.dst_wsize)

    operands = [get_value(**opd0)]
    results = [get_value(**res0)]
    return (results, attr, operands)


@opparam_converter_regitstry("DMA_masked_select")
def _converter(reg):
    shape = tuple(reg[f"src_{d}size"] for d in "nchw")
    opd0 = dict(
        address=dma_addr(reg.src_start_addr_h8, reg.src_start_addr_l32),
        dtype=DType(reg.src_data_format),
        shape=shape,
        layout=Layout.alignEU,
    )
    _, c, h, w = shape
    res0 = dict(
        address=dma_addr(reg.dst_start_addr_h8, reg.dst_start_addr_l32),
        dtype=DType(reg.src_data_format),
        shape=shape,
        stride=(c * h * w, h * w, w),
    )
    opd1 = dict(
        address=dma_addr(reg.mask_start_addr_h8, reg.mask_start_addr_l32),
        dtype=DType(reg.mask_data_format),
        shape=shape,
        layout=Layout.alignEU,
    )

    operands = [get_value(**x) for x in (opd0, opd1)]
    results = [get_value(**res0)]

    return (results, {}, operands)


@opparam_converter_regitstry("DMA_general")
def _converter(reg):
    copy_len = reg.src_cstride
    opd0 = dict(
        address=dma_addr(reg.src_start_addr_h8, reg.src_start_addr_l32),
        dtype=DType(reg.src_data_format),
        shape=(copy_len,),
        stride=(1,),
        layout=Layout.DMAlinear,
    )
    res0 = dict(
        address=dma_addr(reg.dst_start_addr_h8, reg.dst_start_addr_l32),
        dtype=DType(reg.src_data_format),
        shape=(copy_len,),
        stride=(1,),
        layout=Layout.DMAlinear,
    )
    lane_mask = reg.localmem_mask_h32 * 2**32 + reg.localmem_mask_l32
    attr = dict()
    if lane_mask != (2**64 - 1):
        attr["lane_mask"] = hex(lane_mask)

    if reg.fill_constant_en:
        opd0 = dict(
            address=reg.constant_value, dtype=DType(reg.src_data_format), is_const=True
        )
    bc_size = reg.dst_csize
    if reg.cmd_special_function == 1:
        res0["shape"] = (bc_size, copy_len)
        res0["stride"] = (LANE_SIZE, 1)

    operands = [get_value(**opd0)]
    results = [get_value(**res0)]

    return (results, attr, operands)


@opparam_converter_regitstry("DMA_cw_transpose")
def _converter(reg):
    lane_mask = reg.localmem_mask_h32 * 2**32 + reg.localmem_mask_l32
    n, c, h, w = (reg[f"src_{d}size"] for d in "nchw")
    opd0 = dict(
        address=dma_addr(reg.src_start_addr_h8, reg.src_start_addr_l32),
        dtype=DType(reg.src_data_format),
        shape=(n, c, h, w),
        stride=(*(reg[f"src_{d}stride"] for d in "nch"), 1),
        layout=Layout.DMAstride(lane_mask),
    )
    res0 = dict(
        address=dma_addr(reg.dst_start_addr_h8, reg.dst_start_addr_l32),
        dtype=DType(reg.src_data_format),
        shape=(n, w, h, c),
        stride=(*(reg[f"dst_{d}stride"] for d in "nch"), 1),
        layout=Layout.DMAstride(lane_mask),
    )
    attr = dict()
    if lane_mask != (2**64 - 1):
        attr["lane_mask"] = hex(lane_mask)

    if reg.fill_constant_en:
        attr = {}
        opd0 = dict(
            address=reg.constant_value, dtype=DType(reg.src_data_format), is_const=True
        )

    operands = [get_value(**opd0)]
    results = [get_value(**res0)]

    return (results, attr, operands)


@opparam_converter_regitstry("DMA_nonzero")
def _converter(reg):
    n, c, h, w = (reg[f"src_{d}size"] for d in "nchw")
    stride = (c * h * w, h * w, w, 1)
    opd0 = dict(
        address=dma_addr(reg.src_start_addr_h8, reg.src_start_addr_l32),
        dtype=DType(reg.src_data_format),
        shape=(n, c, h, w),
        stride=stride,
        layout=Layout.alignEU,
    )
    res0 = dict(
        address=dma_addr(reg.dst_start_addr_h8, reg.dst_start_addr_l32),
        dtype=DType(reg.src_data_format),
        shape=(n, h, w, c),
        stride=stride,
        layout=Layout.alignEU,
    )

    attr = dict(base=reg.dst_nstride)

    operands = [get_value(**opd0)]
    results = [get_value(**res0)]

    return (results, attr, operands)


def dma_gather_base(reg):
    lane_mask = reg.localmem_mask_h32 * 2**32 + reg.localmem_mask_l32
    c, h, w = (reg[f"src_{d}size"] for d in "chw")
    d_h = reg.dst_hsize
    if reg.nchw_copy:
        d_h = h
    stride = (c * h * w, h * w, w, 1)
    opd0 = dict(
        address=dma_addr(reg.src_start_addr_h8, reg.src_start_addr_l32),
        dtype=DType(reg.src_data_format),
        shape=(1, c, h, w),
        stride=(0, reg.src_cstride, reg.src_hstride, 1),
        layout=Layout.stride,
    )
    res0 = dict(
        address=dma_addr(reg.dst_start_addr_h8, reg.dst_start_addr_l32),
        dtype=DType(reg.src_data_format),
        shape=(1, max(c, reg.index_csize), d_h, w),
        stride=(0, reg.dst_cstride, reg.dst_hstride, 1),
        layout=Layout.DMAstride(lane_mask),
    )
    opd1 = dict(
        address=dma_addr(reg.index_start_addr_h8, reg.index_start_addr_l32),
        dtype=DType.ui32,
        shape=(1, reg.index_csize, d_h, 1),
        stride=(0, reg.index_cstride, reg.index_hstride, 1),
        layout=Layout.stride,
    )
    const = get_value(
        address=reg.constant_value, dtype=DType(reg.src_data_format), is_const=True
    ).data
    attr = dict(const=const)

    operands = [get_value(**x) for x in (opd0, opd1)]
    results = [get_value(**res0)]

    return (results, attr, operands)


@opparam_converter_regitstry("DMA_gather")
def _converter(reg):
    return dma_gather_base(reg)


@opparam_converter_regitstry("DMA_scatter ")
def _converter(reg):
    results, _, operands = dma_gather_base(reg)

    return (results, {}, operands)


@opparam_converter_regitstry("DMA_reverse")
def _converter(reg):
    n, c, h, w = (reg[f"src_{d}size"] for d in "nchw")
    stride = (c * h * w, h * w, w, 1)
    opd0 = dict(
        address=dma_addr(reg.src_start_addr_h8, reg.src_start_addr_l32),
        dtype=DType(reg.src_data_format),
        shape=(n, c, h, w),
        stride=stride,
        layout=Layout.alignEU,
    )
    res0 = dict(
        address=dma_addr(reg.dst_start_addr_h8, reg.dst_start_addr_l32),
        dtype=DType(reg.src_data_format),
        shape=(n, h, w, c),
        stride=stride,
        layout=Layout.alignEU,
    )
    attr = dict(base=reg.dst_nstride)

    operands = [get_value(**opd0)]
    results = [get_value(**res0)]

    return (results, attr, operands)


@opparam_converter_regitstry("DMA_compress")
def _converter(reg):
    return ([] * 3)


@opparam_converter_regitstry("DMA_decompress ")
def _converter(reg):
    return ([] * 3)


@opparam_converter_regitstry("sDMA_sys")
def _converter(reg):
    des_imm = reg.constant_value
    msg_id = des_imm & 0x7f
    cnt = (des_imm >> 8) & 0x7f
    if reg.cmd_special_function in (3, 4):
        attr = dict(msg_id=msg_id, cnt=cnt)
    else:
        raise KeyError(
            f"cmd_special_function {reg.cmd_special_function} not supported")
    return ([], attr, [])
