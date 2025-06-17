# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
"""
soc.py provide a python interface of soc, which can compute cmd and access (global/local/smem) memory on soc device.
"""
import os
from numpy import ndarray
import warnings
import ctypes
from typing import Union
import numpy as np
from copy import copy
from debugger.target_1684x import *
from debugger.target_1684x.memmap import *
from debugger.target_common import *
from debugger.target_common.op_support import *


class BM1684XRunner(DeviceRunner):
    lib_name = "libatomic_exec_aarch64.so"
    soc_structs = []
    kernel_fn = os.path.join(
        os.environ["PROJECT_ROOT"],
        "debugger/lib/libbm1684x_atomic_kernel.so"  # do not use libbm1684x_kernel_module, which may cause nan error
    )

    def __init__(self, _):
        super().__init__()

    def init_runner(self):
        self.lib.init_handle.restype = ctypes.c_void_p
        self.lib.init_handle_b.restype = ctypes.c_void_p
        runner = self.lib.init_handle(self.kernel_fn.encode(), 0)
        self.runner = ctypes.c_void_p(runner)
        self.lib.convert_addr.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint64,
        ]
        self.init_memory()
        self.reserved_offset = self.memory.reserved_offset
        self.fast_checker = False

    def __del__(self):
        self.lib.deinit(self.runner)

    def init_memory(self):
        self.memory = Memory(self.lib, self.runner)

    def trans_cmds_to_buf_soc_consume(self, cmd_bufs, engine_type):
        buf_list = []
        for cmd_buf in cmd_bufs:
            if engine_type == 1:  # dma
                cmd_buf_array = (ctypes.c_uint32 * (len(cmd_buf) // 4))()
                ctypes.memmove(ctypes.addressof(cmd_buf_array), cmd_buf, len(cmd_buf))
                c_uint32_obj = (ctypes.c_uint32 *
                                (len(cmd_buf) // 4)).from_buffer_copy(cmd_buf_array)
                self.lib.convert_addr(c_uint32_obj, self.reserved_offset)
                cmd_buf = bytes(c_uint32_obj)
            buf_list.append(cmd_buf)
        return b"".join(buf_list)

    def checker_fast_compute(self, tiu_num, dma_num, tiu_buf, dma_buf):
        rt_tiu_buf = self.trans_cmds_to_buf_soc_consume(tiu_buf, 0)
        rt_dma_buf = self.trans_cmds_to_buf_soc_consume(dma_buf, 1)
        ret = self.lib.launch_cmds_in_pio(
            self.runner,
            ctypes.byref(ctypes.create_string_buffer(rt_tiu_buf)),
            ctypes.byref(ctypes.create_string_buffer(rt_dma_buf)),
            ctypes.c_size_t(len(rt_tiu_buf)),
            ctypes.c_size_t(len(rt_dma_buf)),
            ctypes.c_int(tiu_num),
            ctypes.c_int(dma_num),
        )
        assert ret == 0

    def fast_compute(self, tiu_num, dma_num, tiu_buf, dma_buf):
        assert tiu_num + dma_num == 1
        rt_tiu_buf = self.trans_cmds_to_buf_soc_consume(tiu_buf, 0)
        rt_dma_buf = self.trans_cmds_to_buf_soc_consume(dma_buf, 1)
        ret = self.lib.launch_cmd_in_pio(
            self.runner,
            ctypes.byref(ctypes.create_string_buffer(rt_tiu_buf)),
            ctypes.byref(ctypes.create_string_buffer(rt_dma_buf)),
            ctypes.c_size_t(len(rt_tiu_buf)),
            ctypes.c_size_t(len(rt_dma_buf)),
            ctypes.c_int(tiu_num),
            ctypes.c_int(dma_num),
        )
        assert ret == 0
        return 1


class Memory(DeviceMemory):
    """
    Memory agent. Extract/Set data from a give MemRef object.
    This class should handle all the tenors type in all kinds of storage.
    """

    # device = Target.BM1684X

    def __init__(self, lib, runner_p) -> None:
        super().__init__(lib)
        self.lib = lib
        self.runner_p = runner_p
        self.reserved_offset = lib.get_reserved_mem(runner_p)

        self.LMEM = np.zeros(16 * 1024 * 1024, dtype=np.uint8)

        print(f"use reserved memory {self.reserved_offset}")

    def _local_mem_to_numpy(self, memref: MemRef):
        NPU_OFFSET = memref.npu_offset
        itemsize = memref.itemsize
        l2s_ret = self.lib.chip_l2s(
            self.runner_p,
            self.LMEM.ctypes.data_as(ctypes.c_void_p),
        )
        assert l2s_ret == 0

        def data_view(shape, stride):
            offset = memref.r_addr - NPU_OFFSET * info.LANE_SIZE
            return np.lib.stride_tricks.as_strided(
                self.LMEM[offset:offset + 4].view(memref.np_dtype),
                shape,
                np.array(stride) * itemsize,
                writeable=False,
            )

        def get_stride_data_base(shape, stride):
            n, c, h, w = shape
            n_s, c_s, h_s, w_s = stride
            _shape = [n, (NPU_OFFSET + c + 63) // 64, 64, h, w]
            _stride = (n_s, c_s, info.LANE_SIZE // itemsize, h_s, w_s)
            return data_view(_shape, _stride).reshape(n, -1, h, w)[:n,
                                                                   NPU_OFFSET:NPU_OFFSET + c, :, :]

        def get_stride_data():
            return get_stride_data_base(memref.shape, memref.stride)

        def get_alignic_data():
            n, c, h, w = memref.shape
            cube_num = info.CUBE_NUM(memref.dtype)
            shape = (div_up(n, info.NPU_NUM), info.NPU_NUM, div_up(c, cube_num), cube_num, h, w)
            stride = (
                align_up(c, cube_num) * h * w,
                info.LANE_SIZE // itemsize,
                cube_num * h * w,
                1,
                cube_num * w,
                cube_num,
            )
            return data_view(shape, stride).reshape(shape[0] * shape[1], -1, h,
                                                    w)[:n, NPU_OFFSET:NPU_OFFSET + c, :, :]

        def get_matrix_data():
            r, c = memref.shape
            w = memref.layout.args[0]
            shape = (r, div_up(c, w), 1, w)
            _memref = copy.copy(memref)
            _memref.shape = shape
            _memref.layout = Layout.alignEU
            stride = local_layout_to_stride(_memref)
            return get_stride_data_base(shape, stride).reshape(r, -1)[:r, :c]

        def get_matrix2_data():
            r, c = memref.shape
            shape = (1, r, 1, c)
            _memref = copy(memref)
            _memref.shape = shape
            _memref.layout = Layout.alignEU
            stride = local_layout_to_stride(_memref)
            return get_stride_data_base(shape, stride).reshape(r, c)

        def _lane_mask_filter(c, lane_mask):
            lane_mask = np.unpackbits(np.uint64([lane_mask]).view(np.uint8), bitorder="little")
            _c = div_up(NPU_OFFSET + c, info.NPU_NUM)
            index = np.zeros(_c * info.NPU_NUM, bool)
            index[NPU_OFFSET:NPU_OFFSET + c] = True
            index = index.reshape(_c, info.NPU_NUM)
            index[:, lane_mask == 0] = False
            return index.flatten()

        def get_dma4bank_data():
            n, c, h, w = memref.shape
            shape = (4, n, div_up(NPU_OFFSET + c, info.NPU_NUM), info.NPU_NUM, h, w)
            n_s, c_s, h_s, w_s = memref.stride
            stride = (info.BANK_SIZE, n_s, c_s, info.LANE_SIZE // itemsize, h_s, w_s)
            index = _lane_mask_filter(c, memref.layout.args[0])
            return data_view(shape, stride).reshape(4, n, -1, h, w)[:, :, index, :, :]

        def get_dma_stride_data(_memref=memref):
            n, c, h, w = _memref.shape
            shape = (n, div_up(NPU_OFFSET + c, info.NPU_NUM), info.NPU_NUM, h, w)
            n_s, c_s, h_s, w_s = _memref.stride
            stride = (n_s, c_s, info.LANE_SIZE // itemsize, h_s, w_s)
            index = _lane_mask_filter(c, _memref.layout.args[0])
            return data_view(shape, stride).reshape(n, -1, h, w)[:, index, :, :]

        def get_dma_matrix_data():
            r, c = memref.shape
            w = memref.layout.args[1]
            shape = (r, div_up(c, w), 1, w)
            _memref = copy.copy(memref)
            _memref.shape = shape
            return get_dma_stride_data(_memref).reshape(r, -1)[:r, :c]

        def get_dma_linear_data():
            return data_view(memref.shape, memref.stride)

        get_lmem_data = {
            Layout.alignEU: get_stride_data,
            Layout.compact: get_stride_data,
            Layout.offset: get_stride_data,
            Layout.stride: get_stride_data,
            Layout.alignIC: get_alignic_data,
            Layout.matrix: get_matrix_data,
            Layout.matrix2: get_matrix2_data,
            Layout.alignLine: get_stride_data,
            Layout.T4: get_stride_data,
            Layout.T5: get_stride_data,
            Layout.DMA4Bank: get_dma4bank_data,
            Layout.DMAstride: get_dma_stride_data,
            Layout.DMAmatrix: get_dma_matrix_data,
            Layout.DMAlinear: get_dma_linear_data,
        }
        data = get_lmem_data[memref.layout]()
        if memref.dtype == DType.bf16:
            return bf16_to_fp32(data)
        return data

    def _ddr_to_numpy(self, memref: MemRef):
        assert memref.shape is not None
        assert memref.stride is not None
        assert all(memref.shape)
        assert any(memref.stride)

        for i in range(4):
            if memref.shape[i] != 0 and memref.stride[i] != 0:
                elements = memref.shape[i] * memref.stride[i]
                break

        raw_data = np.zeros(elements, dtype=memref.np_dtype)
        address = memref.address + self.reserved_offset
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

    def get_data(self, value_ref: ValueRef):
        value = value_ref.value
        if isinstance(value, Scalar):
            return value.data

        assert isinstance(value, MemRef)
        if value.mtype == MType.G:
            return self._ddr_to_numpy(value)
        if value.mtype == MType.R:
            return self._local_mem_to_numpy(value)
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
        if m_type == MType.G:
            assert data.dtype == value.np_dtype
            s2d_ret = self.lib.chip_s2d(
                self.runner_p,
                ctypes.c_uint64(address),
                data.size * data.dtype.itemsize,
                data.ctypes.data_as(ctypes.c_void_p),
            )
            assert s2d_ret == 0
            return True
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
