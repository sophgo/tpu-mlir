# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

"""
cmodel.py provide a python interface of cmodel, which can compute cmd and access (global/local/smem) memory.
"""
import os
from numpy import ndarray
import warnings
import ctypes
from typing import Union
import numpy as np
from copy import copy
from ..target_common import *
from .opdef import TiuCmd, DmaCmd
from .memmap import *


class BM1684XRunner(DeviceRunner):
    lib_name = "libatomic_exec.so"

    def __init__(self, _):
        super().__init__()
        lib = lib_wrapper(open_lib(self.lib_name))

        self.lib = lib
        kernel_fn = os.path.join(
            os.environ["TPUC_ROOT"], "lib/libbm1684x_atomic_kernel.so"
        )
        lib.init_handle.restype = ctypes.c_void_p
        lib.init_handle_b.restype = ctypes.c_void_p

        runner = lib.init_handle(kernel_fn.encode(), 0)

        self.runner = ctypes.c_void_p(runner)

        self.lib.convert_addr.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint64,
        ]
        self.init_memory(_)
        self.reserved_offset = self.memory.reserved_offset

    def __del__(self):
        self.lib.deinit(self.runner)

    def _compute(self, cmd: BaseTpuCmd, engine_type):
        assert engine_type in {0, 1}
        reg = copy(cmd.reg)
        reg.cmd_id = 1
        reg.cmd_id_dep = 0

        if engine_type == 1:  # dma
            u32_buf = (ctypes.c_uint32 * (len(cmd.buf) // 4)).from_buffer_copy(reg)
            self.lib.convert_addr(u32_buf, self.reserved_offset)
            buf = bytes(u32_buf)
        else:
            buf = bytes(reg)

        return self.lib.launch_single_cmd(
            self.runner,
            ctypes.create_string_buffer(buf),
            engine_type,
            len(buf),
        )

    def init_memory(self, memory_size: int, _=None):
        self.memory = Memory(self.lib, self.runner)

    def tiu_compute(self, command: TiuCmd):
        return self._compute(command, 0)

    def dma_compute(self, command: DmaCmd):
        return self._compute(command, 1)


class PcieMemoryArray:
    def __getitem__(self, v):
        pass

    def __setitem__(self, k, v):
        pass


class Memory(DeviceMemory):
    """
    Memory agent. Extract/Set data from a give MemRef object.
    This class should handle all the tenors type in all kinds of storage.
    """

    device = Target.BM1684X

    def __init__(self, lib, runner_p) -> None:
        super().__init__(lib)
        self.lib = lib
        self.runner_p = runner_p
        self.reserved_offset = lib.get_reserved_mem(runner_p)
        print(f"use reserved memory {self.reserved_offset}")

    def _ddr_to_numpy(self, memref: MemRef):
        assert memref.shape is not None
        assert memref.stride is not None
        assert all(memref.shape)
        assert any(memref.stride)

        raw_data = np.zeros(memref.shape[0] * memref.stride[0], dtype=memref.np_dtype)
        # raw_data = np.zeros(memref.shape, dtype=memref.np_dtype)

        address = memref.address + self.reserved_offset
        # print(f"get from {address}")
        d2s_ret = self.lib.chip_d2s(
            self.runner_p,
            ctypes.c_uint64(address),
            raw_data.size * raw_data.dtype.itemsize,
            raw_data.ctypes.data_as(ctypes.c_void_p),
        )
        output = np.lib.stride_tricks.as_strided(
            raw_data[:4].view(memref.np_dtype),
            np.ctypeslib.as_array(memref.shape),
            np.ctypeslib.as_array(memref.stride) * memref.itemsize,
            writeable=False,
        )
        assert d2s_ret == 0
        if memref.dtype == DType.bf16:
            return bf16_to_fp32(output)
        return output

    def clear_memory(self):
        warnings.warn("pcie mode clear memory have no use")

    def get_data_from_address(self, address: int, data: np.ndarray) -> ndarray:
        address += self.reserved_offset
        d2s_ret = self.lib.chip_d2s(
            self.runner_p,
            ctypes.c_uint64(address),
            data.size * data.dtype.itemsize,
            data.ctypes.data_as(ctypes.c_void_p),
        )
        return d2s_ret

    def get_data(self, value: Union[Scalar, MemRef]):
        if isinstance(value, Scalar):
            return value.data

        assert isinstance(value, MemRef)
        if value.mtype == MType.G:
            return self._ddr_to_numpy(value)
        # if value.mtype == MType.R:
        #     return self._local_mem_to_numpy(value)
        raise ValueError(f"unsupported memory view: {value}")

    def set_data_to_address(self, address: int, data: np.ndarray):
        address += self.reserved_offset
        s2d_ret = self.lib.chip_s2d(
            self.runner_p,
            ctypes.c_uint64(address),
            data.size * data.dtype.itemsize,
            data.ctypes.data_as(ctypes.c_void_p),
        )
        assert s2d_ret == 0

    def set_data(self, value: MemRef, data: np.ndarray):
        m_type = value.mtype
        address = value.address + self.reserved_offset
        print(f"save to {address}({value.address})")
        if m_type == MType.G:
            assert data.dtype == value.np_dtype
            s2d_ret = self.lib.chip_s2d(
                self.runner_p,
                ctypes.c_uint64(address),
                data.size * data.dtype.itemsize,
                data.ctypes.data_as(ctypes.c_void_p),
            )
            assert s2d_ret == 0
        else:
            raise NotImplementedError(f"Not support setting {m_type} memory data.")

    def check_data(self, gd, address):
        actual = np.zeros_like(gd)

        self.lib.chip_d2s(
            self.runner_p,
            ctypes.c_uint64(address),
            actual.size * actual.dtype.itemsize,
            actual.ctypes.data_as(ctypes.c_void_p),
        )
        print((gd == actual).all(), gd.sum())
