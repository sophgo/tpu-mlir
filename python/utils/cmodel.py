#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import ctypes
import os
from ctypes import Structure, POINTER


class local_mem(Structure):
    _fields_ = [
        ("raw_ptr", POINTER(ctypes.c_char)),
        ("mem_arr", POINTER(POINTER(ctypes.c_uint32))),
        ("count", ctypes.c_int32),
        ("size_per_mem", ctypes.c_int32),
        ("align_num", ctypes.c_int32),
        ("need_free", ctypes.c_int32),
    ]


class _lib_wrapper(object):

    __slots__ = ["_lib", "_fntab"]

    def __init__(self, lib):
        self._lib = lib
        self._fntab = {}

    def __getattr__(self, name):
        try:
            return self._fntab[name]
        except KeyError:
            # Lazily wraps new functions as they are requested
            cfn = getattr(self._lib, name)
            wrapped = _lib_fn_wrapper(cfn)
            self._fntab[name] = wrapped
            return wrapped

    @property
    def _name(self):
        return self._lib._name

    @property
    def _handle(self):
        return self._lib._handle


class _lib_fn_wrapper(object):

    __slots__ = ["_cfn"]

    def __init__(self, cfn):
        self._cfn = cfn

    @property
    def argtypes(self):
        return self._cfn.argtypes

    @argtypes.setter
    def argtypes(self, argtypes):
        self._cfn.argtypes = argtypes

    @property
    def restype(self):
        return self._cfn.restype

    @restype.setter
    def restype(self, restype):
        self._cfn.restype = restype

    def __call__(self, *args, **kwargs):
        return self._cfn(*args, **kwargs)


_lib_name = "libcmodel_1684x.so"

try:
    lib_path = os.environ["LD_LIBRARY_PATH"]
    _lib_full_name = None
    for path in lib_path.split(":"):
        if os.path.isfile(os.path.join(path, _lib_name)):
            _lib_full_name = os.path.join(path, _lib_name)
            break
    if not _lib_full_name:
        raise OSError
    lib = ctypes.CDLL(_lib_full_name)
except OSError as e:
    msg = f"""Could not find/load shared object file: {_lib_name}
 Error was: {e}"""
    raise OSError(msg)


lib = _lib_wrapper(lib)


lib.cmodel_init.argtypes = [ctypes.c_int32, ctypes.c_int64]
lib.cmodel_init.restype = ctypes.c_int32
lib.cmodel_deinit.argtypes = [ctypes.c_int32]
lib.cmodel_multi_thread_cxt_deinit.argtypes = [ctypes.c_int32]

# local_mem
lib.get_local_mem.argtypes = [ctypes.c_int32]
lib.get_local_mem.restype = ctypes.POINTER(local_mem)
lib.clear_local_mem.argtypes = [ctypes.c_int32]
lib.fill_local_mem.argtypes = [ctypes.c_int32]

lib.get_l2_sram.argtypes = [ctypes.c_int32]
lib.get_l2_sram.restype = ctypes.c_void_p

lib.get_arrange_reg.argtypes = [ctypes.c_int32]
lib.get_arrange_reg.restype = ctypes.POINTER(ctypes.c_uint32)

lib.get_share_memaddr.argtypes = [ctypes.c_int32]
lib.get_share_memaddr.restype = ctypes.c_void_p

lib.get_global_memaddr.argtypes = [ctypes.c_int32]
lib.get_global_memaddr.restype = ctypes.c_void_p

lib.cmodel_get_global_mem_size.argtypes = [ctypes.c_int32]
lib.cmodel_get_global_mem_size.restype = ctypes.c_ulonglong

#
lib.execute_command.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint]

# atomic instructions (only for long commands)
lib.atomic_bc.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_conv.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_cw_transpose.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_depthwise.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_fused_cmp.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_fused_linear.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_gde.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_global_dma.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_mm.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_mm2.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_nms.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_pooling.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_pooling_depthwise.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_roi_depthwise.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_roi_pooling.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_rqdq.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_sequence_generate.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_sg.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_sgl.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_sort.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_special_func.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_sys.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_tensor_arithmetic.argtypes = [ctypes.c_int32, ctypes.c_void_p]
lib.atomic_vector_correlation.argtypes = [ctypes.c_int32, ctypes.c_void_p]
