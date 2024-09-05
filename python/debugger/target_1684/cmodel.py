# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
# cmodel.py provide a python interface of cmodel, which can compute cmd and access (global/local/smem) memory.
import numpy as np
import ctypes
from typing import Union

from ..target_common import *
from .memmap import *


node_idx = 0


class BM1684Runner(CModelRunner):
    lib_name = "libcmodel_1684.so"

    def __init__(self, memory_size):
        super().__init__()

        lib = lib_wrapper(open_lib(self.lib_name))
        lib.cmodel_init.argtypes = [ctypes.c_int32, ctypes.c_int64]
        lib.cmodel_init.restype = ctypes.c_int32
        lib.cmodel_deinit.argtypes = [ctypes.c_int32]

        # local_mem
        lib.get_local_mem.argtypes = [ctypes.c_int32]
        lib.get_local_mem.restype = ctypes.POINTER(local_mem)
        lib.clear_local_mem.argtypes = [ctypes.c_int32]
        lib.fill_local_mem.argtypes = [ctypes.c_int32]

        lib.get_l2_sram.argtypes = [ctypes.c_int32]
        lib.get_l2_sram.restype = ctypes.POINTER(ctypes.c_char)

        lib.get_arrange_reg.argtypes = [ctypes.c_int32]
        lib.get_arrange_reg.restype = ctypes.POINTER(ctypes.c_uint32)

        lib.get_share_memaddr.argtypes = [ctypes.c_int32]
        lib.get_share_memaddr.restype = ctypes.c_void_p

        lib.get_global_memaddr.argtypes = [ctypes.c_int32]
        lib.get_global_memaddr.restype = ctypes.c_void_p

        lib.cmodel_get_global_mem_size.argtypes = [ctypes.c_int32]
        lib.cmodel_get_global_mem_size.restype = ctypes.c_ulonglong

        # computing function
        atomic_func_t = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p)
        lib.get_atomic_function.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        lib.get_atomic_function.restype = atomic_func_t

        # cmodel
        lib.bm_api_dynamic_fullnet.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.bm_api_dynamic_fullnet.restype = ctypes.c_int

        lib.cmodel_get_share_memory_addr.argtypes = [ctypes.c_uint32, ctypes.c_int]
        lib.cmodel_get_share_memory_addr.restype = ctypes.POINTER(ctypes.c_uint32)

        # async function
        lib.cmodel_call_atomic.argtypes = [
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]

        self.lib = lib
        self.init_memory(memory_size)

    def init_memory(self, memory_size):
        self.lib.cmodel_init(node_idx, memory_size)
        DDR = c_array_to_ndarray(self.lib.get_global_memaddr(0), memory_size)
        LMEM = c_array_to_ndarray(
            self.lib.get_local_mem(0).contents.raw_ptr, (64, 8, 1024 * 64)
        )
        L2SRAM = c_array_to_ndarray(self.lib.get_l2_sram(0), (4096 * 1024,))

        self.memory = Memory(LMEM, DDR, L2SRAM)

    def clear_memory(self):
        self.DDR.fill(0)
        self.LMEM.fill(0)
        lut = np.array(self.gen_lookup_table(), np.uint32).view(np.uint8)
        self.L2SRAM[: len(lut)] = lut[...]

    def __del__(self):
        self.lib.cmodel_deinit(0)

    def _compute(self, command: BaseTpuCmd, engine_type):
        command = np.frombuffer(command.buf, dtype=np.uint8)
        cmd_p = command.ctypes.data_as(ctypes.c_void_p)

        return self.lib.get_atomic_function(cmd_p, engine_type)(0, cmd_p)

    def tiu_compute(self, command: BaseTpuCmd):
        return self._compute(command, 0)

    def dma_compute(self, command: BaseTpuCmd):
        return self._compute(command, 1)

    def dynamic_compute(self, command: DynIrCmd, core_id=0):
        # ir_buf = np.frombuffer(command.ir_buffer, dtype=np.uint8)
        # buf_p = ir_buf.ctypes.data_as(ctypes.c_char_p)
        # breakpoint()
        # status = self.lib.bm_api_dynamic_fullnet(buf_p, command.ir_size)
        return 0

    @staticmethod
    def gen_lookup_table():
        return []


class Memory(CModelMemory):
    """
    Memory agent. Extract/Set data from a give MemRef object.
    This class should handle all the tenors type in all kinds of storage.
    """

    def clear_memory(self):
        self.DDR.fill(0)
        self.LMEM.fill(0)
        lut = np.array(BM1684Runner.gen_lookup_table(), np.uint32).view(np.uint8)
        self.SMEM[: len(lut)] = lut[...]

    def _local_mem_to_numpy(self, memref: MemRef):
        NPU_OFFSET = memref.npu_offset
        itemsize = memref.itemsize

        def data_view(shape, stride):
            offset = memref.r_addr - NPU_OFFSET * LANE_SIZE
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

        get_lmem_data = {
            Layout.alignEU: get_stride_data,
            Layout.compact: get_stride_data,
            Layout.stride: get_stride_data,
            Layout.alignEU_XN: get_4n_2n_data(True),
            Layout.compact_XN: get_4n_2n_data(False),
        }
        return get_lmem_data[memref.layout]()

    def _get_xn_shape_stride(self, memref: MemRef):
        assert memref.itemsize in (1, 2)
        xn = 4 // memref.itemsize
        n, *dims = memref.shape
        shape = (Ceil(n, xn), xn, *dims)
        stride = memref.stride
        stride = (stride[0], 1, *stride[1:])
        return np.uint64(shape), np.uint64(stride)

    def _ddr_to_numpy(self, memref: MemRef):
        assert memref.shape is not None
        assert memref.stride is not None
        assert all(memref.shape)
        assert any(memref.stride)
        offset = memref.r_addr

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

    def set_data(self, value: MemRef, data: np.ndarray):
        m_type = value.mtype
        if m_type == MType.G:
            assert data.dtype == value.np_dtype, f"{data.dtype} != {value.np_dtype}"
            offset = value.r_addr
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
        else:
            raise NotImplementedError(m_type)
