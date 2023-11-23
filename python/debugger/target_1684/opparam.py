# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
# 1684

from typing import Callable, Dict, Union, Tuple, List, Any
from ..target_common import Scalar, atomic_reg
from .cmodel import MemRef

__all__ = ["opparam_converter"]

# Value: MemRef | Scalar
# Scalar: Number
# Number: int | float
# MemRef: address, shape, Dtype, stride, offset?

ValueType = Union[MemRef, Scalar]

opparam_converter: Dict[
    str, Callable[[atomic_reg], Tuple[List[ValueType], Dict[str, Any], List[ValueType]]]
] = {}


def opparam_converter_regitstry(class_name):
    def add_converter(fun):
        if class_name in opparam_converter:
            raise KeyError(f"{class_name} have already registered.")
        opparam_converter[class_name] = fun
        return fun

    return add_converter


def default_converter(_):
    return ([],) * 3


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
