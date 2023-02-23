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
LEN_SIZE = BANK_SIZE * 16


class ExtEnum:
    """
    Add additional information to Enumerate.
    Enumerate is a constant type and a singleton.
    """

    def __init__(self, enum, *args, **kargs) -> None:
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
    R = 0  # local memory
    S = 1  # static memory
    L = 2  # L2 SRAM
    G = 3  # DDR
    UNKNOWN = 7

    def __call__(self, *args, **kargs):
        return ExtEnum(self, *args, **kargs)


class Layout(Enum):
    alignEU = 0  # 64 bytes aligment
    compact = 1
    offset = 2
    stride = 3
    matrix = 4
    matrix2 = 5
    _64IC = 10
    _32IC = 11
    _1IC = 12

    def __call__(self, *args, **kargs):
        return ExtEnum(self, *args, **kargs)

    def __hash__(self):
        return hash(self._name_)

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other

        if isinstance(other, Layout):
            return self.value == other.value

        if isinstance(other, ExtEnum):
            return self.value == other.value
        return False


# TPU1686/common/include/memmap.h
memmap = {
    MType.R: (int("0x8000000", 16), int("0x9000000", 16)),  # lmen_base 16M
    MType.S: (int("0x9000000", 16), int("0x9004000", 16)),  # static memory 16KB
    MType.L: (int("0x10000000", 16), int("0x10200000", 16)),  # L2 SRAM  2M
    MType.G: (int("0x100000000", 16), int("0x300000000", 16)),  # global memory
}


def get_mtype(address):
    # R : "npu_offset", "bank_index", "bank_offset", "r_addr"
    # G/S/L : "r_addr"
    for k, v in memmap.items():
        if address >= v[0] and address < v[1]:
            if k == MType.R:
                r_addr = address - memmap[MType.R][0]
                npu_offset = r_addr // LEN_SIZE
                addr_len = r_addr - npu_offset * LEN_SIZE
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


class DType(IntEnum):
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
    u8 = i8 + 8  # type: ignore
    u16 = i16 + 8  # type: ignore
    u32 = i32 + 8  # type: ignore
    # sign integer
    s8 = i8 + 16  # type: ignore
    s16 = i16 + 16  # type: ignore
    s32 = i32 + 16  # type: ignore

    def is_float(self):
        return self in (DType.f32, DType.f16, DType.bf16)

    def is_int(self):
        return not self.is_float()


def get_dtype(prec, sign=1):  # unsigned -> 0; sign -> 1
    if prec in (DType.f32, DType.bf16, DType.f16):
        return DType(prec)
    return DType(prec + 8 + (sign == 1) * 8)


to_np_dtype = {
    DType.s8: np.int8,
    DType.u8: np.uint8,
    DType.i8: np.uint8,
    DType.f16: np.float16,
    DType.f32: np.float32,
    DType.s16: np.int16,
    DType.u16: np.uint16,
    DType.i16: np.uint16,
    DType.s32: np.int32,
    DType.u32: np.uint32,
    DType.i32: np.uint32,
}


def local_layout_to_stride(memref):
    def get_aligenEU_stride():
        n, c, h, w = memref.shape
        align_type = 64 // memref.itemsize
        c_stride = (w * h + align_type - 1) // align_type * align_type
        n_stride = (c + memref.mtype.npu_offset + 63) // 64 * c_stride
        return (n_stride, c_stride, w, 1)

    def get_compact_stride():
        n, c, h, w = memref.shape
        c_stride = w * h
        n_stride = (c + memref.mtype.npu_offset + 63) // 64 * c_stride
        return (n_stride, c_stride, w, 1)

    def get_offset_stride():
        return (0, 1, 0, 0)

    layout = memref.layout
    if layout == Layout.alignEU:
        return get_aligenEU_stride()

    if layout == Layout.compact:
        return get_compact_stride()

    if layout == Layout.offset:
        return get_offset_stride()

    return None


class MemRef:
    def __init__(self, address, shape, dtype: DType, stride=None, layout=None):
        self.address = address
        self.shape = shape
        self.dtype = dtype
        self.layout = layout
        self.mtype = get_mtype(self.address)
        self.np_dtype = to_np_dtype[dtype]
        self.itemsize = self.np_dtype().itemsize
        self.stride = None
        if stride and any(stride):
            self.stride = stride
        elif layout:
            if self.mtype == MType.R:
                # TPU stride
                self.stride = local_layout_to_stride(self)
        # print information
        self.name = self.__name()
        self.type_str = self.__type_str()

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
        if self.stride and any((x != 0 for x in self.stride)):
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
    def __init__(self, LMEM, DDR) -> None:
        self.LMEM = LMEM.ravel()
        self.DDR = DDR.ravel()

    def _local_mem_to_numpy(self, memref):
        NPU_OFFSET = memref.mtype.npu_offset
        itemsize = memref.itemsize

        def data_view(shape, stride):
            offset = memref.mtype.r_addr + NPU_OFFSET * LEN_SIZE
            return np.lib.stride_tricks.as_strided(
                self.LMEM[offset : offset + 4].view(memref.np_dtype),
                shape,
                np.array(stride) * itemsize,
            )

        def get_stride_data_base(shape, stride):
            n, c, h, w = shape
            n_s, c_s, h_s, w_s = stride
            _shape = [n, (c + 63) // 64, 64, h, w]
            _stride = (n_s, c_s, LEN_SIZE // itemsize, h_s, w_s)
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
                LEN_SIZE // itemsize,
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
                LEN_SIZE // itemsize,
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
                LEN_SIZE // itemsize,
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
        }

        return get_data[memref.layout]()

    def _ddr_to_numpy(self, memref):
        assert memref.layout == None
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
        raise ValueError("unsupported memory view!")


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


def conv_reg_format(reg):
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


def mm_reg_format(reg):
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


def mm2_reg_format(reg):
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


def cmp_reg_format(reg):
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


def sfu_reg_format(reg):
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


def lin_reg_format(reg):
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


def vc_reg_format(reg):
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


def ar_reg_format(reg):
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
        address=reg.opd1_addr,
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


def pord_reg_format(reg):
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
            opd2["dtype"] = DType.s32
        else:
            opd2["dtype"] = opd0["dtype"]
        if reg.tsk_eu_typ == 2:
            res0["dtype"] = (reg.res0_prec, reg.res0_prec == DType.i32)
        opds = [opd0, opd1, opd2, opd3]
    elif reg.tsk_eu_typ in [6, 7]:
        opd3["shape"] = (1, reg.res0_w, 1, 4)
        opd3["dtype"] = DType.s16
        opds = [opd0, opd3]
    elif reg.tsk_eu_typ == 5:
        opd3["shape"] = (1, reg.res0_w, 1, 4)
        opd3["dtype"] = DType.s16
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


def t3_stride(opdef):
    _, c, h, w = opdef
    itemsize = to_np_dtype[opdef["dtype"]]().itemsize
    NPU_OFFSET = (opdef["address"] - memmap[MType.R][0]) // LEN_SIZE
    eu_num = 64 // itemsize
    h_stride = (w + eu_num - 1) / eu_num * eu_num
    c_stride = h * h_stride
    n_stride = c_stride * (c + NPU_OFFSET + 63) // 64
    return (n_stride, c_stride, h_stride, 1)


def t4_stride(opdef):
    _, _, _, w = opdef
    itemsize = to_np_dtype[opdef["dtype"]]().itemsize
    eu_num = 64 // itemsize
    h_stride = (w + eu_num - 1) / eu_num * eu_num
    return (0, 0, h_stride, 1)


def t5_stride(opdef):
    _, _, _, w = opdef
    itemsize = to_np_dtype[opdef["dtype"]]().itemsize
    eu_num = 64 // itemsize
    c_stride = (w + eu_num - 1) / eu_num * eu_num
    n_stride = LEN_SIZE // 8 // itemsize
    return (n_stride, c_stride, w, 1)


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
        opd0["stride"] = t3_stride(opd0)
        opd0["layout"] = Layout.stride

    operands = [get_value(**opd0, offset=memmap[MType.R][0])]
    results = [get_value(**res0, offset=memmap[MType.R][0])]

    return (results, {}, operands)


def rqdq_reg_format(reg):
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
        opd1["dtype"] = DType.s32
        opd1["shape"] = (1, c, 1, 1)
        opd2 = dict(opd1)
        opd2["address"] += 4
        opd3 = dict(opd2)
        opd3["address"] += 4
        if opd1["is_const"]:
            opd2["address"] = reg.opd2_addr
            opd2["dtype"] = DType.s8
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
        opd1["dtype"] = DType.s32  # zp
        opd1["shape"] = (1, c, 1, 1)
        opd2 = dict(opd1)  # scale
        opd2["address"] += 4
        opd3 = dict(opd2)  # shift
        opd3["address"] += 4
        opds = [opd0, opd1, opd2, opd3]
        if opd1["is_const"]:
            opd1["dtype"] = (DType.i16, reg.opd0_sign)  # zp
            opd2["address"] = reg.opd1_addr // 2**16  # shift
            opd2["dtype"] = DType.s16
            opd3["address"] = reg.opd2_addr  # scale
            opd3["dtype"] = DType.s32
            opds = [opd0, opd1, opd3, opd2]
    else:
        raise KeyError("Should not be here.")

    attr = dict(round_mode=reg.opd2_n_str)
    operands = [get_value(**x, offset=memmap[MType.R][0]) for x in opds]
    results = [get_value(**res0, offset=memmap[MType.R][0])]

    return (results, attr, operands)


def sg_reg_format(reg):
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
            opd0["layout"] = Layout.stride
            opd0["stride"] = t4_stride(opd0)
        opd1["shape"] = (n, c, 1, reg.opd1_w)
        opd1["layout"] = Layout.alignEU
    elif reg.tsk_eu_typ == 2:
        opd0["shape"] = (n, c, reg.opd0_h, reg.opd0_w)
        opd1["shape"] = (1, reg.opd1_c, 1, 4)
        opd1["dtyp"] = DType.u16
        opd1["layout"] = Layout.compact
        kh = reg.opd3_addr // 2**16
        kw = reg.opd3_addr % 2**16
        r_h = kh * kw
        res0["shape"] = (n, c, r_h, w)
        res0["stride"] = t3_stride(res0)
        res0["layout"] = Layout.stride
    elif reg.tsk_eu_typ in [8, 15]:
        opd0["shape"] = (1, c, 1, reg.opd0_w)
        if reg.opd0_str != 0:
            opd0["stride"] = t4_stride(opd0)
            opd0["layout"] = Layout.stride
        opd1["shape"] = (n, c, 1, reg.opd1_w)
        res1 = dict(
            address=reg.res1_addr,
            dtype=DType.s16,
            shape=(n, c, 1, 1),
            layout=Layout.compact,
        )
        rests = [res0, res1]
    elif reg.tsk_eu_typ in [9, 16]:
        opd0["shape"] = (n, c, 1, reg.opd0_w)
        opds = [opd0]
        res1 = dict(
            address=reg.res1_addr,
            dtype=DType.s16,
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
            opd0["stride"] = t4_stride(opd0)
        else:
            opd0["stride"] = t5_stride(opd0)
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


def sgl_reg_format(reg):
    n, c, h, w = (reg[f"res0_{d}"] for d in "nchw")
    opd0 = dict(
        address=reg.opd0_addr,
        dtype=DType(reg.opd0_prec),
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
        dtype=DType(reg.opd1_prec, 0),
        layout=Layout.compact,
    )
    opd0["shape"] = (1, c, reg.opd0_h, w)
    if reg.opd0_str == 3:
        opd0["stride"] = t3_stride(opd0)
    else:
        opd0["stride"] = t4_stride(opd0)

    rets = [res0]
    if reg.tsk_eu_typ in [17, 18]:
        opd1_h = reg.opd1_h if reg.tsk_eu_typ == 17 else reg.opd0_h
        opd1["shape"] = (n, c, opd1_h, 1)
        res0["stride"] = t3_stride(res0)
    elif reg.tsk_eu_typ == 19:
        opd0["shape"] = (1, c, reg.opd0_h, w)
        opd1["shape"] = (n, c, reg.opd0_h, 1)
        res0["stride"] = t3_stride(res0)
        res1 = dict(
            address=reg.res1_addr,
            dtype=DType.s16,
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
