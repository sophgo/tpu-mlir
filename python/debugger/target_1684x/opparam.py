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
# 1684x

import numpy as np
from typing import Callable, Dict, Union, Tuple, List, Any
from ..target_common import (
    MType,
    DType,
    Scalar,
    ExtEnum,
    Layout,
    get_dtype,
    atomic_reg,
)

from .regdef import *
from .cmodel import MemRef
from .memmap import *

# Value: MemRef | Scalar
# Scalar: Number
# Number: int | float
# MemRef: address, shape, Dtype, stride, offset?

ValueType = Union[MemRef, Scalar]

opparam_converter: Dict[
    str,
    Callable[[atomic_reg], Tuple[List[ValueType], Dict[str, Any], List[ValueType]]],
] = {}


def opparam_converter_regitstry(sheet_name):
    # original bind in _set_op()
    def add_converter(fun):
        if sheet_name in opparam_converter:
            raise KeyError(f"{sheet_name} have already registered.")

        assert sheet_name in op_class_dic, sheet_name

        opparam_converter[sheet_name] = fun
        return fun

    return add_converter


def get_value(
    address=-1,
    shape=None,
    stride=None,
    dtype=None,
    layout=None,
    is_const=False,
) -> Union[Scalar, MemRef]:
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
        if address < info.LMEM_SIZE:
            address += memmap[MType.R][0]
        return MemRef(address, shape, _dtype, stride, _layout)


# @opparam_converter_regitstry("__default__")
def default_converter(_):
    return ([],) * 3


@opparam_converter_regitstry("sCONV")
def sCONV_t_converter(reg: sCONV_reg):
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
        layout=Layout.alignIC,
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

    opd1["layout"] = Layout.alignIC
    if reg.opd0_prec == DType.i8:  # 8bits
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

    if reg.tsk_eu_typ in [1, 2]:  # conv_wrq
        assert opd2["is_const"] > 0
        assert opd3["is_const"] > 0

    operands = [get_value(**x) for x in (opd0, opd1, opd2, opd3)]
    results = [get_value(**res0)]

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


@opparam_converter_regitstry("sMM")
def sMM_t_converter(reg: sMM_reg):
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
        assert reg.opd0_prec == DType.i8
        opd2["dtype"] = (DType.i16, 1)  # bias
        attr["shift"] = reg.opd3_addr  # shift

    operands = [get_value(**x) for x in (opd0, opd1, opd2)[:opd_num]]
    results = [get_value(**res0)]

    return (results, attr, operands)


@opparam_converter_regitstry("sMM2")
def sMM2_t_converter(reg: sMM2_reg):
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

    operands = [get_value(**x) for x in (opd0, opd1, opd2)[:opd_num]]
    results = [get_value(**res0)]

    return (results, attr, operands)


@opparam_converter_regitstry("sCMP")
def sCMP_t_converter(reg: sCMP_reg):
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

    operands = [get_value(**x) for x in (opd0, opd1, opd2, opd3)]
    results = [get_value(**x) for x in rets]

    return (results, {}, operands)


@opparam_converter_regitstry("sSFU")
def sSFU_t_converter(reg: sSFU_reg):
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

    operands = [get_value(**x) for x in (opd0, opd1)[:opd_num]]
    results = [get_value(**res0)]

    return (results, attr, operands)


@opparam_converter_regitstry("sLIN")
def sLIN_t_converter(reg: sLIN_reg):
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

    operands = [get_value(**x) for x in (opd0, opd1, opd2)[:opd_num]]
    results = [get_value(**res0)]

    return (results, {}, operands)


@opparam_converter_regitstry("sVC")
def sVC_t_converter(reg: sVC_reg):
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
def sAR_t_converter(reg: sAR_reg):
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
    operands = [get_value(**x) for x in (opd0, opd1, opd2)[:opd_num]]
    results = [get_value(**res0)]

    return (results, attr, operands)


@opparam_converter_regitstry("sPorD")
def sPorD_t_converter(reg: sPorD_reg):
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

    operands = [get_value(**x) for x in opds]
    results = [get_value(**res0)]

    return (results, attr, operands)


def cw_trans_reg_format(reg: sTRANS_sBC_reg):
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
        opd0["layout"] = Layout.alignLine

    operands = [get_value(**opd0)]
    results = [get_value(**res0)]

    return (results, {}, operands)


@opparam_converter_regitstry("sRQ&sDQ")
def sRQ_sDQ_t_converter(reg: sRQ_sDQ_reg):
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
    operands = [get_value(**x) for x in opds]
    results = [get_value(**res0)]

    return (results, attr, operands)


@opparam_converter_regitstry("sSG")
def sSG_t_converter(reg: sSG_reg):
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
        res0["layout"] = Layout.alignLine
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
        rets = [res0, res1]
    elif reg.tsk_eu_typ in [9, 16]:
        opd0["shape"] = (n, c, 1, reg.opd0_w)
        opds = [opd0]
        res1 = dict(
            address=reg.res1_addr,
            dtype=DType.si16,
            shape=(n, c, 1, 1),
            layout=Layout.compact,
        )
        rets = [res0, res1]
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
    operands = [get_value(**x) for x in opds]
    results = [get_value(**x) for x in rets]

    return (results, attr, operands)


@opparam_converter_regitstry("SGL")
def SGL_t_converter(reg: SGL_reg):
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
        opd0["layout"] = Layout.alignLine
    else:
        opd0["layout"] = Layout.T4

    rets = [res0]
    if reg.tsk_eu_typ in [17, 18]:  # sliceToReverse
        opd1_h = reg.res0_h if reg.tsk_eu_typ == 17 else reg.opd0_h
        opd1["shape"] = (n, c, opd1_h, 1)
        res0["layout"] = Layout.alignLine
    elif reg.tsk_eu_typ == 19:
        opd0["shape"] = (1, c, reg.opd0_h, w)
        opd1["shape"] = (n, c, reg.opd0_h, 1)
        res0["layout"] = Layout.alignLine
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

    operands = [get_value(**x) for x in (opd0, opd1)]
    results = [get_value(**x) for x in rets]

    return (results, attr, operands)


def bc_reg_format(reg: sTRANS_sBC_reg):
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

    operands = [get_value(**opd0)]
    results = [get_value(**res0)]

    return (results, {}, operands)


@opparam_converter_regitstry("sTRANS&sBC")
def sTRANS_sBC_t_converter(reg: sTRANS_sBC_reg):
    if reg.tsk_eu_typ in (0, 1):
        return cw_trans_reg_format(reg)
    else:
        return bc_reg_format(reg)


def dma_addr(H, L):
    return H * 2**32 + L


def dma_reg_fmt_base(reg: Union[DMA_tensor_0x000__reg, DMA_matrix_reg]):
    # also used in DMA_matrix_reg
    lane_mask = reg.localmem_mask_h32 * 2**32 + reg.localmem_mask_l32

    if isinstance(reg, DMA_matrix_reg):
        address_iter = [
            (reg.src_start_addr_l8, reg.src_start_addr_h32),
            (reg.dst_start_addr_l8, reg.dst_start_addr_h32),
        ]
    elif isinstance(reg, DMA_tensor_0x000__reg):
        address_iter = [
            (reg.src_start_addr_h8, reg.src_start_addr_l32),
            (reg.dst_start_addr_h8, reg.dst_start_addr_l32),
        ]
    else:
        raise NotImplementedError()

    opd0 = dict(
        address=dma_addr(*address_iter[0]),
        dtype=DType(reg.src_data_format),
        shape=tuple(reg[f"src_{d}size"] for d in "nchw"),
        stride=tuple(reg[f"src_{d}stride"] for d in "nchw"),
        layout=Layout.DMAstride(lane_mask),
    )
    res0 = dict(
        address=dma_addr(*address_iter[1]),
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


@opparam_converter_regitstry("DMA_tensor（0x000）")
def DMA_tensor_0x000__t_converter(reg: DMA_tensor_0x000__reg):
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
        n, _, h, w = opd0["shape"]
        opd0["shape"] = (n, 1, h, w)
        n, _, h, w = res0["shape"]
        res0["shape"] = (n, reg.dst_csize, h, w)
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
        if reg.cmd_special_function == BANK4_BDC:
            c = 0
        opd0["shape"] = (n, 0, h, w)
        res0["layout"] = Layout.DMA4Bank(res0["layout"].args[0])

    operands = [get_value(**opd0)]
    results = [get_value(**res0)]

    return (results, attr, operands)


@opparam_converter_regitstry("DMA_matrix")
def DMA_matrix_t_converter(reg: DMA_matrix_reg):
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


@opparam_converter_regitstry("DMA_masked_select")
def DMA_masked_select_t_converter(reg: DMA_masked_select_reg):
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
def DMA_general_t_converter(reg: DMA_general_reg):
    copy_len = reg.src_cstride_move_length_
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
        res0["stride"] = (info.LANE_SIZE, 1)

    operands = [get_value(**opd0)]
    results = [get_value(**res0)]

    return (results, attr, operands)


@opparam_converter_regitstry("DMA_cw_transpose")
def DMA_cw_transpose_t_converter(reg: DMA_cw_transpose_reg):
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


@opparam_converter_regitstry("DMA_nonzero")
def DMA_nonzero_t_converter(reg: DMA_nonzero_reg):
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

    # TODO check dst_nstride_base_i_ or dst_nstride
    attr = dict(decompress=bool(reg.decompress_enable), base=reg.dst_nstride_base_i_)

    operands = [get_value(**opd0)]
    results = [get_value(**res0)]

    return (results, attr, operands)


def dma_gather_base(reg: DMA_gather_reg):
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
def DMA_gather_t_converter(reg: DMA_gather_reg):
    return dma_gather_base(reg)


@opparam_converter_regitstry("DMA_scatter")
def DMA_scatter_t_converter(reg: DMA_scatter_reg):
    results, _, operands = dma_gather_base(reg)

    return (results, {}, operands)
