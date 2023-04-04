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

from enum import Enum, IntEnum
import numpy as np
import copy

# Value: MemRef | Const
# Const: Number
# Number: int | float
# MemRef: address, shape, Dtype, stride, offset?

BANK_SIZE = 2**14
LANE_SIZE = BANK_SIZE * 16

opparam_converter = {}


def regitstry_opparam_converter(sheet_name):
    def add_converter(fun):
        if sheet_name in opparam_converter:
            raise KeyError(f"{sheet_name} have already registered.")
        opparam_converter[sheet_name] = fun
        return

    return add_converter


class ExtEnum:
    """
    Add additional information to Enumerate.
    Enumerate is a constant type and a singleton.
    """

    def __init__(self, enum, *args, **kargs) -> None:
        assert isinstance(enum, Enum)
        self.eunm = enum
        self.args = args
        for k, v in kargs.items():
            if k not in ("name", "value"):
                self.__dict__[k] = v
            else:
                raise KeyError(f"{k} is a reserved key.")
        self._member_ = kargs.keys()

    @property
    def name(self):
        return self.eunm.name

    @property
    def value(self):
        return self.eunm.value

    def __hash__(self):
        return hash(self.eunm)

    def __eq__(self, other):
        return self.eunm == other

    def __repr__(self) -> str:
        kargs = {k: self.__dict__[k] for k in self._member_}
        return repr(self.eunm) + f"{self.args}{kargs}"


class MType(Enum):
    """
    The type of memory.
    """

    R = 0  # local memory
    S = 1  # static memory
    L = 2  # L2 SRAM
    G = 3  # DDR
    UNKNOWN = 7

    def __call__(self, *args, **kargs):
        return ExtEnum(self, *args, **kargs)


class Layout(Enum):
    """
    Data layout type in different storage.
    """

    # Tensor alignment
    alignEU = 0
    compact = 1
    offset = 2
    stride = 3
    # Matrix alignment
    matrix = 10
    matrix2 = 11
    # Weight alignment
    _64IC = 20
    _32IC = 21
    _1IC = 22
    # special alignment. TODO: give it a better name
    T3 = 30
    T4 = 31
    T5 = 32
    # GDMA special layout
    continuous = 40
    DMAstride = 41  # contains lane mask
    DMA4Bank = 42
    DMAmatrix = 43
    DMAlinear = 44

    def __call__(self, *args, **kargs):
        return ExtEnum(self, *args, **kargs)


# TPU1686/common/include/memmap.h
memmap = {
    MType.R: (int("0x8000000", 16), int("0x9000000", 16)),  # lmen_base 16M
    MType.S: (int("0x9000000", 16), int("0x9004000", 16)),  # static memory 16KB
    MType.L: (int("0x10000000", 16), int("0x10200000", 16)),  # L2 SRAM  2M
    MType.G: (int("0x100000000", 16), int("0x300000000", 16)),  # global memory
}


class DType(IntEnum):
    """
    The numeric type of the data.
    """

    # Signless
    # Only the bits width is correct.
    i8 = 0
    f16 = 1
    f32 = 2
    i16 = 3
    i32 = 4
    bf16 = 5
    i64 = 6
    # unsign integer
    ui8 = i8 + 8  # type: ignore
    ui16 = i16 + 8  # type: ignore
    ui32 = i32 + 8  # type: ignore
    # sign integer
    si8 = i8 + 16  # type: ignore
    si16 = i16 + 16  # type: ignore
    si32 = i32 + 16  # type: ignore

    def is_float(self):
        return self in (DType.f32, DType.f16, DType.bf16)

    def is_int(self):
        return not self.is_float()


def get_dtype(prec, sign=1):  # unsigned -> 0; sign -> 1
    if prec in (DType.f32, DType.bf16, DType.f16):
        return DType(prec)
    return DType(prec + 8 + (sign == 1) * 8)


to_np_dtype = {
    DType.si8: np.int8,
    DType.ui8: np.uint8,
    DType.f16: np.float16,
    DType.f32: np.float32,
    DType.si16: np.int16,
    DType.ui16: np.uint16,
    DType.si32: np.int32,
    DType.ui32: np.uint32,
    DType.i8: np.uint8,
    DType.i16: np.uint16,
    DType.i32: np.uint32,
    DType.bf16: np.uint16,
}


def local_layout_to_stride(memref):
    """
    Layout Canonicalize. Convert special layout to stride layout.
    """

    def aligenEU_stride():
        _, c, h, w = memref.shape
        align_type = 64 // memref.itemsize
        c_stride = (w * h + align_type - 1) // align_type * align_type
        n_stride = (c + memref.mtype.npu_offset + 63) // 64 * c_stride
        return (n_stride, c_stride, w, 1)

    def compact_stride():
        _, c, h, w = memref.shape
        c_stride = w * h
        n_stride = (c + memref.mtype.npu_offset + 63) // 64 * c_stride
        return (n_stride, c_stride, w, 1)

    def offset_stride():
        return (0, 1, 0, 0)

    def t3_stride():
        _, c, h, w = memref.shape
        eu_num = 64 // memref.itemsize
        h_stride = (w + eu_num - 1) / eu_num * eu_num
        c_stride = h * h_stride
        n_stride = c_stride * (c + memref.mtype.npu_offset + 63) // 64
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
        return aligenEU_stride()
    if memref.layout == Layout.compact:
        return compact_stride()
    if memref.layout == Layout.offset:
        return offset_stride()
    if memref.layout == Layout.T3:
        return t3_stride()
    if memref.layout == Layout.T4:
        return t4_stride()
    if memref.layout == Layout.T3:
        return t5_stride()
    if memref.layout == Layout.DMAstride:
        return dma_nolane_mask_to_stride()

    return None


class MemRef:
    """
    A description of tensor in memory.
    """

    def __init__(self, address, shape, dtype: DType, stride=None, layout=None):
        self.address = address
        self.mtype = MemRef.get_mtype(address)  # extended enumerate type
        self.shape = shape
        self.dtype = dtype
        self.layout = layout

        self.np_dtype = to_np_dtype[dtype]
        self.itemsize = self.np_dtype().itemsize
        self.stride = stride

        if self.mtype == MType.R and layout != Layout.stride:
            self.stride = local_layout_to_stride(self)

        if self.mtype == MType.G and layout == Layout.continuous:
            self.stride = tuple(np.cumprod([1] + shape[-1:0:-1])[::-1])

        # print information
        self.name = self.__name()
        self.type_str = self.__type_str()

    @staticmethod
    def get_mtype(address):
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
                    return MType.R(
                        npu_offset=npu_offset,
                        bank_index=bank_index,
                        bank_offset=bank_offset,
                        r_addr=r_addr,
                    )
                return k(r_addr=address - v[0])
        return MType.UNKNOWN

    def __name(self):
        k = self.mtype
        if k == MType.UNKNOWN:
            return f"%?.{self.address}"
        if k == MType.R:
            # R{bank_index}.{bank_offset}.L{NPU_OFFSET}
            # R0.123.L0
            mem_str = f"%{k.name}{k.bank_index}"
            if k.bank_offset:
                mem_str += f".{k.bank_offset}"
            if k.npu_offset:
                mem_str += f".L{k.npu_offset}"
            return mem_str
        return f"%{k.name}{k.r_addr}"

    def __type_str(self):
        s = [str(x) for x in self.shape]
        if self.stride is not None and any((x != 0 for x in self.stride)):
            return f"memref<{'x'.join(s)}x{self.dtype.name}, strides: [{str(self.stride)[1:-1]}]>"
        return f"memref<{'x'.join(s)}x{self.dtype.name}>"

    def __repr__(self):
        return f"{self.name}: {self.type_str}"


class Const:
    def __init__(self, value, dtype: DType):
        data = np.uint32([value])
        self.dtype = dtype
        if dtype != DType.bf16:
            np_dtype = to_np_dtype[dtype]
            self.data = data.view(np_dtype)[0]
        else:
            self.data = data.view(np.float32)[0]
        self.name = f"%C{self.data}"
        self.type_str = f"{self.dtype.name}"

    def __repr__(self):
        return f"{self.name}: {self.type_str}"


class NamedDict(dict):
    def __init__(self, dic, del_prefix=[]):
        self.__dict__.update(dic)
        if isinstance(del_prefix, str):
            del_prefix = [del_prefix]
        for prefix in del_prefix:
            for k in list(self.__dict__.keys()):
                if k[: len(prefix)] == prefix:
                    self.__dict__[k[len(prefix) :]] = self.__dict__.pop(k)

    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise KeyError(f"invalid key: {key}")

    def __setitem__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            raise KeyError(f"invalid key: {key}")

    def __repr__(self):
        return str(self.__dict__)


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

        def get_stride_data():
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
                :n, NPU_OFFSET : NPU_OFFSET + c, :, :
            ]

        def get_32ic_data():
            n, c, h, w = memref.shape
            shape = ((n + 63) // 64, 64, (c + 32) // 32, 32, h, w)
            stride = (
                (c + 63) // 64 * 64 * h * w,
                LANE_SIZE // itemsize,
                32 * h * w,
                1,
                32 * w,
                32,
            )
            return data_view(shape, stride).reshape(shape[0] * shape[1], -1, h, w)[
                :n, NPU_OFFSET : NPU_OFFSET + c, :, :
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
                :n, NPU_OFFSET : NPU_OFFSET + c, :, :
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
            index[NPU_OFFSET : NPU_OFFSET + c] = True
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
            shape = (n, (NPU_OFFSET + c + 63) // 64, 64, h, w)
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

        return get_data[memref.layout]()

    def _ddr_to_numpy(self, memref):
        assert memref.shape != None
        assert memref.stride != None
        assert all(memref.shape)
        assert any(memref.stride)
        offset = memref.mtype.r_addr
        return np.lib.stride_tricks.as_strided(
            self.DDR[offset : offset + 4].view(memref.np_dtype),
            np.ctypeslib.as_array(memref.shape),
            np.ctypeslib.as_array(memref.stride) * memref.itemsize,
        )

    def get_data(self, value):
        if isinstance(value, Const):
            return value.data
        assert isinstance(value, MemRef)
        if value.mtype == MType.G:
            return self._ddr_to_numpy(value)
        if value.mtype == MType.R:
            return self._local_mem_to_numpy(value)
        raise ValueError(f"unsupported memory view: {value}")


def get_value(
    address=None,
    offset=0,
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
        return Const(address, _dtype)
    else:
        _layout = layout
        if not isinstance(layout, ExtEnum):
            _layout = Layout(layout)
        return MemRef(address + offset, shape, _dtype, stride, _layout)


@regitstry_opparam_converter("sCONV")
def _converter(reg):
    opd0 = dict(
        address=reg.opd0_addr,
        shape=(reg.res0_n, reg.opd0_c, reg.opd0_h, reg.opd0_w),
        stride=[reg[f"opd0_{i}_str"] for i in "nchw"],
        dtype=(reg.opd0_prec, reg.opd0_sign),
        layout=reg.opd0_str,
    )
    res0 = dict(
        address=reg.res0_addr,
        shape=[reg[f"res0_{i}"] for i in "nchw"],
        dtype=(reg.res0_prec, reg.opd0_sign or reg.opd1_sign or reg.opd2_sign),
        layout=Layout.alignEU,
    )
    opd1 = dict(
        address=reg.opd1_addr,
        shape=[reg.res0_c, reg.opd0_c, reg.opd1_h, reg.opd1_w],
        dtype=(reg.opd0_prec, reg.opd1_sign),
        is_const=reg.opd1_const,
        layout=Layout._1IC,
    )
    opd2 = dict(
        address=reg.opd2_addr,
        is_const=reg.opd2_const,
        layout=Layout.compact,
    )
    opd3 = dict(
        address=reg.opd3_addr,
        is_const=reg.opd3_const,
        layout=Layout.compact,
    )

    if reg.opd0_prec == DType.i8:  # 8bits
        opd1["layout"] = Layout._64IC
        if reg.tsk_eu_typ == 0:  # conv_normal
            # kzp
            opd2["shape"] = [1, reg.opd0_c, 1, 1]
            opd2["dtype"] = (DType.i8, reg.opd2_sign)
            res0["dtype"] = (res0["dtype"][0], res0["dtype"][1] or reg.opd2_addr)
            opd3["shape"] = [1, reg.opd0_c, 1, 2]
            opd3["dtype"] = opd0["dtype"]
        else:
            opd2["shape"] = [1, reg.res0_c, 1, 1]
            opd2["dtype"] = (DType.i16, reg.opd2_sign)
            opd3["dtype"] = (DType.i8, 0)  # ui8
    else:
        # bias
        opd2["shape"] = [1, reg.res0_c, 1, 1]
        opd2["dtype"] = (DType.f32, reg.opd0_sign)
        opd3["shape"] = [1, reg.opd0_c, 1, 2]
        opd3["dtype"] = opd0["dtype"]
        if reg.opd0_prec in (DType.f16, DType.bf16):
            opd1["layout"] = Layout._32IC

    if reg.tsk_eu_typ in [1, 2]:  # conv_wrq
        assert self.opd2["is_const"] > 0
        assert self.opd3["is_const"] > 0

    operands = [
        get_value(**x, offset=memmap[MType.R][0]) for x in (opd0, opd1, opd2, opd3)
    ]
    results = [get_value(**res0, offset=memmap[MType.R][0])]

    attr = dict(
        kernel=[reg.opd1_h, reg.opd1_w],
        stride=[reg.res_op_y_str, reg.res_op_x_str],
        in_zero=[reg.opd0_y_ins0, reg.opd0_x_ins0],
        ke_zero=[reg.opd1_y_ins0, reg.opd1_x_ins0],
        kernel_rotate=bool(reg.kernel_rotate),
        pad_mode=reg.pad_mode,
        pad=[reg[f"opd0_{x}_pad"] for x in ("up", "dn", "lf", "rt")],
        res_add=bool(reg.res_add),
    )
    return (results, attr, operands)


@regitstry_opparam_converter("sMM")
def _converter(reg):
    L_row = reg.opd0_n
    L_col = reg.opd0_w * (reg.opd0_c - 1) + reg.opd1_w
    R_col = reg.res0_c * reg.res0_w
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=(reg.opd0_prec, reg.opd0_sign),
        shape=(L_row, L_col),
        layout=Layout.matrix(reg.opd0_w),
        is_const=reg.opd0_const,
    )
    if reg.left_tran:
        L_row, L_col = L_col, L_row
    res0 = dict(
        address=reg.res0_addr,
        dtype=(reg.res0_prec, 1),
        shape=(L_row, R_col),
        layout=Layout.matrix(reg.res0_w),
    )
    opd1 = dict(
        address=reg.opd1_addr,
        dtype=(reg.opd0_prec, reg.opd1_sign),
        shape=(L_col, R_col),
        layout=Layout.matrix(reg.res0_w),
    )
    opd2 = dict(
        address=reg.opd2_addr,
        is_const=reg.opd2_const,
        shape=(1, R_col),
        layout=Layout.matrix(reg.res0_w),
    )

    attr = dict(
        l_trans=bool(reg.left_tran),
        res_add=bool(reg.res_add),
    )
    opd_num = 3
    if reg.tsk_eu_typ == 1:  # mm_normal
        if reg.opd0_prec == DType.f32:
            opd2["dtype"] = opd0["dtype"]  # bias
        elif reg.opd0_prec == DType.i32:
            opd2["dtype"] = (DType.i8, 1)  # shift
        else:
            opd_num = 2
    elif reg.tsk_eu_typ in [2, 3]:
        assert reg0.opd0_prec == DType.i8
        opd2["dtype"] = (DType.i16, 1)  # bias
        attr["shift"] = reg.opd3_addr  # shift

    operands = [
        get_value(**x, offset=memmap[MType.R][0]) for x in (opd0, opd1, opd2)[:opd_num]
    ]
    results = [get_value(**res0, offset=memmap[MType.R][0])]

    return (results, attr, operands)


@regitstry_opparam_converter("sMM2")
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
        dtype=(reg.opd0_prec, reg.opd0_sign),
        shape=(L_row, L_col),
        layout=Layout.matrix2,
        is_const=reg.opd0_const,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=(reg.res0_prec, 1),
        shape=(reg.res0_c, reg.res0_w),
        layout=Layout.matrix2,
    )
    opd1 = dict(
        address=reg.opd1_addr,
        dtype=(reg.opd0_prec, reg.opd1_sign),
        shape=(reg.opd1_c, reg.opd1_w),
        layout=Layout.matrix2,
    )
    opd2 = dict(
        address=reg.opd2_addr,
        is_const=reg.opd2_const,
    )
    attr = dict(
        l_trans=bool(l_trans),
        r_trans=bool(r_trans),
        res_add=bool(reg.res_add),
    )
    opd_num = 2
    if reg.opd0_prec == DType.i8:
        opd_num = 3
        opd2["dtype"] = opd1["dtype"]  # zp
        if reg.tsk_eu_typ in [4, 5]:  # NN, NT
            opd2["shape"] = (1, 64, 1, reg.res0_w)
            opd2["layout"] = Layout.alignEU
        else:
            opd2["shape"] = (1, reg.res0_c, 1, 1)
            opd2["layout"] = Layout.compact

    operands = [
        get_value(**x, offset=memmap[MType.R][0]) for x in (opd0, opd1, opd2)[:opd_num]
    ]
    results = [get_value(**res0, offset=memmap[MType.R][0])]

    return (results, attr, operands)


@regitstry_opparam_converter("sCMP")
def _converter(reg):
    shape = tuple(reg[f"res0_{d}"] for d in "nchw")
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=(reg.opd0_prec, reg.opd0_sign),
        shape=shape,
        layout=reg.opd0_str,
        is_const=reg.opd0_const,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=opd0["dtype"],
        shape=shape,
        layout=Layout.alignEU,
    )
    res1 = dict(
        address=reg.res1_addr,
        dtype=DType(reg.opd2_prec),
        shape=shape,
        layout=Layout.alignEU,
    )
    opd1 = dict(
        address=reg.opd1_addr,
        dtype=opd0["dtype"],
        shape=shape,
        layout=reg.opd1_str,
        is_const=reg.opd1_const,
    )
    opd2 = dict(
        address=reg.opd2_addr,
        dtype=DType(reg.opd2_prec),
        shape=shape,
        layout=Layout.alignEU,
        is_const=reg.opd2_const,
    )
    opd3 = dict(
        address=reg.opd3_addr,
        dtype=DType(reg.opd2_prec),
        shape=shape,
        layout=Layout.alignEU,
        is_const=reg.opd3_const,
    )

    rets = [res0, res1]
    if reg.opd0_str == Layout.stride:
        opd0["stride"] = (0, 0, 0, 0)
    if reg.opd1_str == Layout.stride:
        opd1["stride"] = (0, 0, 0, 0)

    if reg.tsk_eu_typ in (23, 24, 26):
        res0["dtype"] = res1["dtype"]
        rets = [res0]

    operands = [
        get_value(**x, offset=memmap[MType.R][0]) for x in (opd0, opd1, opd2, opd3)
    ]
    results = [get_value(**x, offset=memmap[MType.R][0]) for x in rets]

    return (results, {}, operands)


@regitstry_opparam_converter("sSFU")
def _converter(reg):
    shape = tuple(reg[f"res0_{d}"] for d in "nchw")
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=(reg.opd0_prec, 1),
        shape=shape,
        layout=Layout.alignEU,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=(reg.res0_prec, 1),
        shape=shape,
        layout=Layout.alignEU,
    )
    opd1 = dict(
        address=reg.opd1_addr,
        dtype=(reg.opd0_prec, 1),
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

    operands = [
        get_value(**x, offset=memmap[MType.R][0]) for x in (opd0, opd1)[:opd_num]
    ]
    results = [get_value(**res0, offset=memmap[MType.R][0])]

    return (results, attr, operands)


@regitstry_opparam_converter("sLIN")
def _converter(reg):
    shape = tuple(reg[f"res0_{d}"] for d in "nchw")
    c = shape[1]
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=(reg.res0_prec, 1),
        shape=shape,
        layout=Layout.alignEU,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=(reg.res0_prec, 1),
        shape=shape,
        layout=Layout.alignEU,
    )
    opd1 = dict(
        address=reg.opd1_addr,
        dtype=opd0["dtype"],
        shape=(1, c, 1, 1),
        layout=Layout.compact,
        is_const=reg.opd1_const,
    )
    opd2 = dict(
        address=reg.opd2_addr,
        dtype=opd0["dtype"],
        shape=(1, c, 1, 1),
        layout=Layout.compact,
        is_const=reg.opd2_const,
    )

    opd_num = 2
    if reg.tsk_eu_typ == 1:
        opd_num = 3

    operands = [
        get_value(**x, offset=memmap[MType.R][0]) for x in (opd0, opd1, opd2)[:opd_num]
    ]
    results = [get_value(**res0, offset=memmap[MType.R][0])]

    return (results, {}, operands)


@regitstry_opparam_converter("sVC")
def _converter(reg):
    n = (reg.opd0_c - 1) * reg.opd0_w + reg.opd1_w
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=(reg.opd0_prec, reg.opd0_sign),
        shape=(1, reg.opd0_c, 1, reg.opd0_w),
        layout=Layout.alignEU,
    )

    res0 = dict(
        address=reg.res0_addr,
        dtype=(reg.res0_prec, 1),
        shape=(n, reg.res0_c, 1, reg.res0_w),
        layout=Layout.alignEU,
    )
    opd1 = dict(
        address=reg.opd1_addr,
        dtype=(reg.opd1_prec, reg.opd1_sign),
        shape=(1, reg.res0_c, 1, reg.res0_w),
        layout=Layout.alignEU,
    )
    attr = {}
    if reg.tsk_eu_typ == 23:
        attr = dict(round_mode=reg.opd2_n_str)

    operands = [get_value(**x, offset=memmap[MType.R][0]) for x in (opd0, opd1)]
    results = [get_value(**res0, offset=memmap[MType.R][0])]

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


@regitstry_opparam_converter("sAR")
def _converter(reg):
    n, c, h, w = (reg[f"res0_{d}"] for d in "nchw")
    # round mm
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=(reg.opd0_prec, reg.opd0_sign),
        shape=(n, c, h, w),
        stride=tuple(reg[f"opd0_{d}_str"] for d in "nchw"),
        layout=reg.opd0_str,
        is_const=reg.opd0_const,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=(reg.res0_prec, reg.opd0_sign or reg.opd1_sign),
        shape=(n, c, h, w),
        stride=tuple(reg[f"res0_{d}_str"] for d in "nchw"),
        layout=reg.res0_str,
    )
    opd1 = dict(
        address=reg.opd1_addr,
        dtype=(reg.opd1_prec, reg.opd1_sign),
        shape=(n, c, h, w),
        stride=tuple(reg[f"opd1_{d}_str"] for d in "nchw"),
        layout=reg.opd1_str,
        is_const=reg.opd1_const,
    )
    opd2 = dict(
        address=reg.opd2_addr,
        dtype=(reg.opd2_prec, reg.opd2_sign),
        shape=(1, c, 1, 1),
        layout=Layout.compact,
        is_const=reg.opd2_const,
    )

    if reg.tsk_eu_typ == 17:  # clamp
        opd2["shape"] = (1, c, 1, 2)
    elif reg.tsk_eu_typ == 28:  # copy_mb
        opd0["shape"] = (1, c, h, w)

    elif reg.tsk_eu_typ == 14:  # DATA_CONVERT
        res0["dtype"] = (reg.res0_prec, reg.opd2_sign)

    if reg.tsk_eu_typ == 12:
        attr = dict(iter=reg.opd2_n_str + 1)
    else:
        attr = dict(round_mode=reg.opd2_n_str)

    opd0["shape"] = restore_org_shape(opd0)
    opd1["shape"] = restore_org_shape(opd1)

    opd_num = reg.tsk_opd_num
    operands = [
        get_value(**x, offset=memmap[MType.R][0]) for x in (opd0, opd1, opd2)[:opd_num]
    ]
    results = [get_value(**res0, offset=memmap[MType.R][0])]

    return (results, attr, operands)


@regitstry_opparam_converter("sPorD")
def _converter(reg):
    n, c, h, w = (reg[f"res0_{d}"] for d in "nchw")
    # round mm
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=(reg.opd0_prec, reg.opd0_sign),
        shape=(n, c, reg.opd0_h, reg.opd0_w),
        layout=Layout.alignEU,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=(reg.res0_prec, reg.opd0_sign),
        shape=(n, c, h, w),
        layout=Layout.alignEU,
    )
    opd1 = dict(
        address=reg.opd1_addr,
        dtype=(reg.opd0_prec, reg.opd1_sign),
        shape=(1, c, reg.opd1_h, reg.opd1_w),
        layout=Layout.compact,
        is_const=reg.opd1_const,
    )
    opd2 = dict(
        address=reg.opd2_addr,
        dtype=(reg.opd0_prec, reg.opd2_sign),
        shape=(1, c, 1, 1),
        layout=Layout.compact,
        is_const=reg.opd2_const,
    )
    # padding
    opd3 = dict(
        address=reg.opd3_addr,
        dtype=(reg.opd0_prec, reg.opd0_sign),
        shape=(1, c, 1, 2),
        layout=Layout.compact,
        is_const=reg.opd3_const,
    )

    opds = []

    if reg.tsk_eu_typ in [1, 4]:  # avg, max
        opds = [opd0, opd3]
    elif reg.tsk_eu_typ in [0, 2]:  # depthwise, depthwise_relu
        res0["dtype"] = (reg.res0_prec, reg.opd0_sign or reg.opd1_sign or reg.opd2_sign)
        if reg.opd0_prec == DType.i8:
            opd2["dtype"] = DType.si32
        else:
            opd2["dtype"] = opd0["dtype"]
        if reg.tsk_eu_typ == 2:
            res0["dtype"] = (reg.res0_prec, reg.res0_prec == DType.i32)
        opds = [opd0, opd1, opd2, opd3]
    elif reg.tsk_eu_typ in [6, 7]:
        opd3["shape"] = (1, reg.res0_w, 1, 4)
        opd3["dtype"] = DType.si16
        opds = [opd0, opd3]
    elif reg.tsk_eu_typ == 5:
        opd3["shape"] = (1, reg.res0_w, 1, 4)
        opd3["dtype"] = DType.si16
        opds = [opd0, opd1, opd3]

    attr = dict(
        kernel=[reg.opd1_h, reg.opd1_w],
        stride=[reg.res_op_y_str, reg.res_op_x_str],
        in_zero=[reg.opd0_y_ins0, reg.opd0_x_ins0],
        ke_zero=[reg.opd1_y_ins0, reg.opd1_x_ins0],
        kernel_rotate=bool(reg.kernel_rotate),
        pad_mode=reg.pad_mode,
        pad=[reg[f"opd0_{x}_pad"] for x in ("up", "dn", "lf", "rt")],
        round_mode=reg.opd2_n_str,
        shift=np.uint32([reg.res1_addr]).view(np.int8)[0],
    )

    operands = [get_value(**x, offset=memmap[MType.R][0]) for x in opds]
    results = [get_value(**res0, offset=memmap[MType.R][0])]

    return (results, attr, operands)


def cw_tans_reg_format(reg):
    n, c, h, w = (reg[f"res0_{d}"] for d in "nchw")
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=DType(reg.res0_prec),
        shape=(n, w, h, c),
        layout=Layout.alignEU,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=DType(reg.res0_prec),
        shape=(n, c, h, w),
        layout=Layout.alignEU,
    )

    if reg.tsk_eu_typ == 0:  # cw_ts
        opd0["layout"] = Layout.T3

    operands = [get_value(**opd0, offset=memmap[MType.R][0])]
    results = [get_value(**res0, offset=memmap[MType.R][0])]

    return (results, {}, operands)


@regitstry_opparam_converter("sRQ&sDQ")
def _converter(reg):
    n, c, h, w = (reg[f"res0_{d}"] for d in "nchw")
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=(reg.opd0_prec, reg.opd0_sign),
        shape=(n, c, h, w),
        layout=Layout.alignEU,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=(reg.res0_prec, reg.opd2_sign),
        shape=(n, c, h, w),
        layout=Layout.alignEU,
    )
    opd1 = dict(
        address=reg.opd1_addr,
        is_const=reg.opd1_const,
        layout=Layout.alignEU,
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
            opd3["dtype"] = (DType.i16, reg.opd2_sign)
        opds = [opd0, opd1, opd2, opd3]
    elif reg.tsk_eu_typ == 3:  # dq_0
        opd1["dtype"] = (
            DType.i32,
            reg.opd0_sign,
        )  # If this is a bug, we should extend DType.
        opd1["shape"] = (1, c, 1, 1)
        opd2 = dict(opd1)
        opd2["address"] += 4
        opd2["shape"] = (1, c, 1, 1)
        opd2["dtype"] = DType.f32
        if opd1["is_const"]:
            opd2["address"] = reg.opd2_addr
        opds = [opd0, opd1, opd2]
    elif reg.tsk_eu_typ == 4:  # dq_1
        opd1["dtype"] = DType.si32  # zp
        opd1["shape"] = (1, c, 1, 1)
        opd2 = dict(opd1)  # scale
        opd2["address"] += 4
        opd3 = dict(opd2)  # shift
        opd3["address"] += 4
        opds = [opd0, opd1, opd2, opd3]
        if opd1["is_const"]:
            opd1["dtype"] = (DType.i16, reg.opd0_sign)  # zp
            opd2["address"] = reg.opd1_addr // 2**16  # shift
            opd2["dtype"] = DType.si16
            opd3["address"] = reg.opd2_addr  # scale
            opd3["dtype"] = DType.si32
            opds = [opd0, opd1, opd3, opd2]
    else:
        raise KeyError("Should not be here.")

    attr = dict(round_mode=reg.opd2_n_str)
    operands = [get_value(**x, offset=memmap[MType.R][0]) for x in opds]
    results = [get_value(**res0, offset=memmap[MType.R][0])]

    return (results, attr, operands)


@regitstry_opparam_converter("sSG")
def _converter(reg):
    n, c, h, w = (reg[f"res0_{d}"] for d in "nchw")
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=DType(reg.opd0_prec),
        layout=reg.opd0_str,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=DType(reg.res0_prec),
        shape=(n, c, h, w),
        layout=Layout.alignEU,
    )
    opd1 = dict(
        address=reg.opd0_addr,
        dtype=(reg.opd1_prec, 0),
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
        if reg.opd0_str != 0:
            opd0["layout"] = Layout.T4
        opd1["shape"] = (n, c, 1, reg.opd1_w)
        opd1["layout"] = Layout.alignEU
    elif reg.tsk_eu_typ == 2:
        opd0["shape"] = (n, c, reg.opd0_h, reg.opd0_w)
        opd1["shape"] = (1, reg.opd1_c, 1, 4)
        opd1["dtyp"] = DType.ui16
        opd1["layout"] = Layout.compact
        kh = reg.opd3_addr // 2**16
        kw = reg.opd3_addr % 2**16
        r_h = kh * kw
        res0["shape"] = (n, c, r_h, w)
        res0["layout"] = Layout.T3
    elif reg.tsk_eu_typ in [8, 15]:
        opd0["shape"] = (1, c, 1, reg.opd0_w)
        if reg.opd0_str != 0:
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
    elif reg.tsk_eu_typ == 10:
        opd1["shape"] = (n, c, 1, reg.opd1_w)
        opds = [opd1]
    elif reg.tsk_eu_typ == 7:
        opd0["shape"] = (4, c, 1, reg.opd0_w)
        if reg.opd0_str == 4:
            opd0["layout"] = Layout.T4
        else:
            opd0["layout"] = Layout.T5
        opd0["layout"] = Layout.stride
        opd1["shape"] = (1, c, 1, reg.opd1_w)
    else:
        raise KeyError("Should not be here.")
    attr = dict(
        limit_enable=bool(reg.opd3_const),
        fill_const=bool(reg.opd2_const),
        const_value=reg.opd2_addr,
    )
    operands = [get_value(**x, offset=memmap[MType.R][0]) for x in opds]
    results = [get_value(**x, offset=memmap[MType.R][0]) for x in rets]

    return (results, attr, operands)


@regitstry_opparam_converter("SGL")
def _converter(reg):
    n, c, h, w = (reg[f"res0_{d}"] for d in "nchw")
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=DType(reg.res0_prec),
        layout=Layout.stride,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=DType(reg.res0_prec),
        shape=(n, c, h, w),
        layout=Layout.stride,
    )
    opd1 = dict(
        address=reg.opd0_addr,
        dtype=(reg.opd1_prec, 0),
        layout=Layout.compact,
    )
    opd0["shape"] = (1, c, reg.opd0_h, w)
    if reg.opd0_str == 3:
        opd0["layout"] = Layout.T3
    else:
        opd0["layout"] = Layout.T4

    rets = [res0]
    if reg.tsk_eu_typ in [17, 18]:  # sliceToReverse
        opd1_h = reg.res0_h if reg.tsk_eu_typ == 17 else reg.opd0_h
        opd1["shape"] = (n, c, opd1_h, 1)
        res0["layout"] = Layout.T3
    elif reg.tsk_eu_typ == 19:
        opd0["shape"] = (1, c, reg.opd0_h, w)
        opd1["shape"] = (n, c, reg.opd0_h, 1)
        res0["layout"] = Layout.T3
        res1 = dict(
            address=reg.res1_addr,
            dtype=DType.si16,
            shape=(n, c, 1, 1),
            layout=Layout.compact,
        )
        rets = [res0, res1]
    else:
        raise KeyError("Should not be here.")

    attr = dict(
        limit_enable=bool(reg.opd3_const),
        fill_const=bool(reg.opd2_const),
        const_value=reg.opd2_addr,
    )

    operands = [get_value(**x, offset=memmap[MType.R][0]) for x in (opd0, opd1)]
    results = [get_value(**x, offset=memmap[MType.R][0]) for x in rets]

    return (results, attr, operands)


def bc_reg_format(reg):
    n, c, h, w = (reg[f"res0_{d}"] for d in "nchw")
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=DType(reg.res0_prec),
        shape=(n, c, h, w),
        layout=Layout.alignEU,
    )
    res0 = dict(
        address=reg.res0_addr,
        dtype=DType(reg.res0_prec),
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

    operands = [get_value(**opd0, offset=memmap[MType.R][0])]
    results = [get_value(**res0, offset=memmap[MType.R][0])]

    return (results, {}, operands)


@regitstry_opparam_converter("sTRANS&sBC")
def _converter(reg):
    if reg.tsk_eu_typ in (0, 1):
        return cw_tans_reg_format(reg)
    else:
        return bc_reg_format(reg)


def dma_addr(H, L):
    return H * 2**32 + L


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

    attr = dict(decompress=bool(reg.decompress_enable))

    if lane_mask != (2**64 - 1):
        attr["lane_mask"] = hex(lane_mask)

    if reg.fill_constant_en:
        attr = {}
        opd0 = dict(
            address=reg.constant_value, dtype=DType(reg.src_data_format), is_const=True
        )

    return res0, attr, opd0


@regitstry_opparam_converter("DMA_tensor（0x000）")
def _converter(reg):
    NONE = 0
    TRANS = 1  # NC Transpose or Matrix Transpose
    COLLECT = 2  # CW Transpose from lmem to gmem
    BROADCAST = 3
    DISTRIBUTE = 4  # CW Transpose from gmem to lmem
    BANK4_COPY = 5
    BANK4_BDC = 6
    res0, attr, opd0 = dma_reg_fmt_base(reg)
    if reg.nchw_copy:
        res0["shape"] = opd0["shape"]

    if reg.cmd_special_function in (BROADCAST, BANK4_BDC):  # broadcast
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

    if reg.cmd_special_function in (BANK4_BDC, BANK4_COPY):
        n, c, h, w = opd0["shape"]
        if reg.cmd_sepcial_function == BANK4_BDC:
            c = 0
        opd0["shape"] = (n, 0, h, w)
        res0["layout"] = Layout.DMA4Bank(res0["layout"].args[0])

    operands = [get_value(**opd0)]
    results = [get_value(**res0)]

    return (results, attr, operands)


@regitstry_opparam_converter("DMA_matrix")
def _converter(reg):
    """
    |--------+--------------+--------------|
    |        | src (local)  | des (ddr)    |
    |--------+--------------+--------------|
    | stride | [n, c, ?, x] | [?, ?, h, x] |
    | shape  | [n, c, ?, w] | [x, ?, h, w] |
    |--------+--------------+--------------|

    |--------+--------------+--------------|
    |        | src (ddr)    | des (local)  |
    |--------+--------------+--------------|
    | stride | [?, ?, h, x] | [n, c, ?, x] |
    | shape  | [?, ?, h, w] | [x, c, ?, w] |
    |--------+--------------+--------------|
    """
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


@regitstry_opparam_converter("DMA_masked_select")
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


@regitstry_opparam_converter("DMA_general")
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
    attr = dict(decompress=bool(reg.decompress_enable))
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


@regitstry_opparam_converter("DMA_cw_transpose")
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
        shape=(n, h, w, c),
        stride=(*(reg[f"dst_{d}stride"] for d in "nch"), 1),
        layout=Layout.DMAstride(lane_mask),
    )

    attr = dict(decompress=bool(reg.decompress_enable))

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


@regitstry_opparam_converter("DMA_nonzero")
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

    attr = dict(decompress=bool(reg.decompress_enable), base=reg.dst_nstride)

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


@regitstry_opparam_converter("DMA_gather")
def _converter(reg):
    return dma_gather_base(reg)


@regitstry_opparam_converter("DMA_scatter")
def _converter(reg):
    results, _, operands = dma_gather_base(reg)

    return (results, {}, operands)
