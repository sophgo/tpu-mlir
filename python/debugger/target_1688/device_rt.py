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

from debugger.target_common.op_support import StaticCmdGroup

from ..target_common import *
from .opdef import TiuCmd, DmaCmd
from .memmap import *
import platform
import struct

COEFF_TAG = 1
NEURON_TAG = 2

TIUBUF_TAG = 8
GDMABUF_TAG = 9
S2L_TAG = 10
L2S_TAG = 11
# LOCALBUF_TAG = 11


def _convert_gdma(buf):
    # print("{")
    # print("BEGIN_FAST_GEN_CMD(GDMA)")
    for index, i in enumerate(range(0, len(buf), 16)):
        low = struct.unpack("Q", buf[i:i + 8])[0]
        high = struct.unpack("Q", buf[i + 8:i + 16])[0]
        print(f"WRITE_CMD_GDMA({index,high,low})")
    #     print(f"WRITE_CMD_GDMA({index}, high, low);")
    # print("END_FAST_GEN_CMD(GDMA, pid_node)")
    # print("}")


def _convert_bdc(buf):
    # print("{")
    # print("BEGIN_FAST_GEN_CMD(BD)")
    for index, i in enumerate(range(0, len(buf), 16)):
        low = struct.unpack("Q", buf[i:i + 8])[0]
        high = struct.unpack("Q", buf[i + 8:i + 16])[0]
        print(f"WRITE_CMD_BD({index}, {high}, {low});")
    # print("END_FAST_GEN_CMD(BD, pid_node)")
    # print("}")


def memcpy_addr_mask(addr):
    return addr & ((1 << 32) - 1)


def memtag(addr):
    return ((addr >> 36) & 0x7)


class soc_launch_struct:

    def __init__(self, tiu_num, dma_num, tiu_buf, dma_buf):
        self.tiu_num = tiu_num
        self.dma_num = dma_num
        self.tiu_buf = tiu_buf
        self.dma_buf = dma_buf
        self.tiu_buf_len = len(tiu_buf)
        self.dma_buf_len = len(dma_buf)


class BM1688Runner(DeviceRunner):
    lib_name = "libatomic_exec_bm1688.so" if platform.machine(
    ) == 'x86_64' else 'libatomic_exec_bm1688_aarch64.so'

    soc_structs = []
    memory: "Memory"
    kernel_fn = os.path.join(os.environ["TPUC_ROOT"], "lib/libbmtpulv60_kernel_module.so")

    def __init__(self, memory_size=None):
        super().__init__()

    def init_runner(self):
        self.lib.convert_addr.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.lib.init_memory.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong]
        self.lib.init_memory.restype = ctypes.c_ulonglong
        self.lib.init_handle.restype = ctypes.c_void_p
        self.lib.memcpy_l2s.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
        self.lib.init_handle_b.restype = ctypes.c_void_p

        runner = self.lib.init_handle(self.kernel_fn.encode(), 0, 0x1688)
        self.runner_p = ctypes.c_void_p(runner)
        self.max_core_num = self.lib.get_max_core_num(self.runner_p)

        self.memory = Memory(self.lib, self.runner_p)
        self.init_memory()

    def trans_cmds_to_buf(self, cmds, engine_type):
        buf_list = []
        for cmd in cmds:
            cmd.reg.cmd_id_dep = 0
            reg = copy(cmd.reg)
            if engine_type == 1:  # dma
                u32_buf = (ctypes.c_uint32 * (len(cmd.buf) // 4)).from_buffer_copy(reg)
                self.lib.convert_addr(self.runner_p, u32_buf)
                buf = bytes(u32_buf)
            else:
                buf = bytes(reg)
            buf_list.append(buf)
        return b"".join(buf_list)

    def __del__(self):
        self.lib.deinit(self.runner_p)

    def _compute(self, cmd: BaseTpuCmd, engine_type):
        assert engine_type in {0, 1}
        reg.cmd_id_dep = 0
        reg = copy(cmd.reg)

        if engine_type == 1:  # dma
            u32_buf = (ctypes.c_uint32 * (len(cmd.buf) // 4)).from_buffer_copy(reg)

            self.lib.convert_addr(self.runner_p, u32_buf)
            buf = bytes(u32_buf)
        else:
            buf = bytes(reg)

        return self.lib.launch_single_cmd(
            self.runner_p,
            ctypes.create_string_buffer(buf),
            engine_type,
            len(buf),
        )

    @property
    def coeff_offset(self):
        return self.memory.coeff_offset

    @property
    def neuron_offset(self):
        return self.memory.neuron_offset

    def init_memory(self):
        self.memory = Memory(self.lib, self.runner_p)

    def cmds_compute(self, cmd_group: StaticCmdGroup):
        if len(cmd_group.all) > 1:
            for cmd in cmd_group.all:
                if cmd.cmd_type == CMDType.tiu:
                    # _convert_bdc(bytes(cmd.reg))
                    self.cmds_compute(StaticCmdGroup([cmd], [], [cmd]))
                elif cmd.cmd_type == CMDType.dma:
                    self.cmds_compute(StaticCmdGroup([], [cmd], [cmd]))
            return len(cmd_group.all)

        tiu, dma = cmd_group.tiu, cmd_group.dma
        # breakpoint()
        tiu_buf = self.trans_cmds_to_buf(tiu, 0)
        dma_buf = self.trans_cmds_to_buf(dma, 1)

        core_ids = set()
        for cmd in cmd_group.all:
            core_ids.add(cmd.core_id)
        assert len(core_ids) == 1, len(core_ids)

        # from .regdef import DMA_tensor_0x000__reg
        # from .opparam import DMA_tensor_0x000__converter
        # from .context import BM1688Context
        # if len(dma) > 0 and isinstance(dma[0].reg, DMA_tensor_0x000__reg):
        #     nreg = DMA_tensor_0x000__reg.from_buffer(bytearray(dma_buf))
        #     ncmd = DmaCmd(nreg,
        #                   buf=memoryview(bytearray(dma_buf)),
        #                   cmd_id=0,
        #                   param_fn=lambda x: DMA_tensor_0x000__converter(
        #                       context=BM1688Context.get_instance(), reg=x))
        #     print(ncmd.reg)

        #     print(tiu_buf)
        #     print(dma_buf)
        #     print(ctypes.c_size_t(len(tiu_buf)))
        #     print(ctypes.c_size_t(len(dma_buf)))
        #     print(ctypes.c_int(len(tiu)))
        #     print(ctypes.c_int(len(dma)))
        #     print(list(core_ids)[0])

        ret = self.lib.debug_cmds(
            self.runner_p,
            ctypes.byref(ctypes.create_string_buffer(tiu_buf)),
            ctypes.byref(ctypes.create_string_buffer(dma_buf)),
            ctypes.c_size_t(len(tiu_buf)),
            ctypes.c_size_t(len(dma_buf)),
            ctypes.c_int(len(tiu)),
            ctypes.c_int(len(dma)),
            list(core_ids)[0],
        )
        assert ret == 0
        return len(tiu) + len(dma)

    def tiu_compute(self, command: TiuCmd):
        return self.cmds_compute(StaticCmdGroup([command], [], [command]))

    def dma_compute(self, command: DmaCmd):
        return self.cmds_compute(StaticCmdGroup([], [command], [command]))

    def get_stack_cmds(self, cur_cmd_point, cmditer):
        from ..plugins.common import FinalMlirIndexPlugin

        tiu = []
        dma = []
        df = FinalMlirIndexPlugin.data_frame
        buf_cmds = 0

        while cur_cmd_point + buf_cmds < len(cmditer):
            if cmditer[cur_cmd_point + buf_cmds].cmd_type == CMDType.tiu:
                tiu.append(cmditer[cur_cmd_point + buf_cmds])
            else:
                dma.append(cmditer[cur_cmd_point + buf_cmds])

            result_not_empty = (len(df.loc[df["executed_id"] == cur_cmd_point + 1 + buf_cmds,
                                           "results"].tolist()[0]) > 0)

            if result_not_empty:
                break
            buf_cmds += 1

        return tiu, dma


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

        self.l2s_offset = self.lib.init_memory(
            self.runner_p,
            TIUBUF_TAG,
            1024 * 1024,
        )
        self.s2l_offset = self.lib.init_memory(
            self.runner_p,
            GDMABUF_TAG,
            1024 * 1024,
        )

        self.l2s_offset = self.lib.init_memory(
            self.runner_p,
            L2S_TAG,
            info.LMEM_SIZE,
        )
        self.s2l_offset = self.lib.init_memory(
            self.runner_p,
            S2L_TAG,
            info.LMEM_SIZE,
        )
        self.neuron_offset = self.lib.init_memory(
            self.runner_p,
            NEURON_TAG,
            32,
        )
        self.coeff_offset = self.lib.init_memory(
            self.runner_p,
            COEFF_TAG,
            32,
        )

        self.LMEM = []
        self.SMEM = []
        for lmem in range(2):
            self.LMEM.append(np.zeros(info.LMEM_SIZE, dtype=np.uint8))
        for smem in range(2):
            self.SMEM.append(np.zeros(64 * 1024, dtype=np.uint8))

        # print(f"use reserved memory {self.reserved_offset}")

    def set_neuron_size(self, size):
        self.neuron_offset = self.lib.init_memory(
            self.runner_p,
            NEURON_TAG,
            size,
        )

    def set_coeff_size(self, size):
        self.coeff_offset = self.lib.init_memory(
            self.runner_p,
            COEFF_TAG,
            size,
        )

    def _load_local_mem(self, core_id):
        l2s_ret = self.lib.memcpy_l2s(
            self.runner_p,
            self.LMEM[core_id].ctypes.data_as(ctypes.c_void_p),
            core_id,
        )
        assert l2s_ret == 0

    def _set_local_mem(self, mem, core_id):
        self.lib.memcpy_s2l(
            self.runner_p,
            mem.ctypes.data_as(ctypes.c_void_p),
            core_id,
        )

    def _local_mem_to_numpy(self, memref: MemRef, core_id):
        NPU_OFFSET = memref.npu_offset
        itemsize = memref.itemsize
        self._load_local_mem(core_id)

        # referece: TPU1688/bm1688/cmodel/src/cmodel_common.cpp
        # improve me: use cmodel interface to get data
        def data_view_int4(shape, stride):
            result = np.zeros(shape, dtype=np.uint8).reshape([shape[0], shape[1], -1])
            laddr = memref.r_addr
            start_npu_idx = NPU_OFFSET
            start_offset = laddr % info.LANE_SIZE
            for nidx in range(0, shape[0]):
                n_offset = nidx * stride[0]
                for cidx in range(0, shape[1]):
                    npu_idx = (start_npu_idx + cidx) % info.NPU_NUM
                    LMEM = self.LMEM[self.core_id][npu_idx * info.LANE_SIZE:(npu_idx + 1) * info.LANE_SIZE]
                    c_offset = ((start_npu_idx + cidx) // info.NPU_NUM) * stride[1]
                    h_offset = np.arange(0, shape[2]) * stride[2]
                    w_offset = np.arange(0, shape[3]) * stride[3]
                    dst_offset = np.add.outer(
                        n_offset,
                        np.add.outer(c_offset, np.add.outer(h_offset, w_offset)),
                    ).ravel()
                    index = start_offset + (dst_offset >> 1)
                    values = LMEM[index].view(np.uint8)
                    result[nidx][cidx] = np.where(dst_offset & 1 == 0, values & 0xF, values >> 4)
            result.reshape(shape)
            if memref.dtype == DType.si4:
                return np.where(result > 7, result - 16, result).astype(np.int8)
            return result

        def data_view(shape, stride):
            offset = memref.r_addr - NPU_OFFSET * info.LANE_SIZE
            return np.lib.stride_tricks.as_strided(
                self.LMEM[self.core_id][offset:offset + 4].view(memref.np_dtype),
                shape,
                np.array(stride) * itemsize,
                writeable=False,
            )

        def get_stride_data_base(shape, stride):
            n, c, h, w = shape
            n_s, c_s, h_s, w_s = stride
            _shape = [n, (NPU_OFFSET + c + info.NPU_NUM - 1) // info.NPU_NUM, info.NPU_NUM, h, w]
            _stride = (n_s, c_s, info.LANE_SIZE // itemsize, h_s, w_s)
            return data_view(_shape, _stride).reshape(n, -1, h, w)[:n,
                                                                   NPU_OFFSET:NPU_OFFSET + c, :, :]

        def get_stride_data():
            if memref.dtype in (DType.i4, DType.si4, DType.ui4):
                return data_view_int4(memref.shape, memref.stride)
            return get_stride_data_base(memref.shape, memref.stride)

        def get_64ic_data():
            n, c, h, w = memref.shape
            shape = ((n + 63) // 64, 64, (c + 63) // 64, 64, h, w)
            stride = (
                (c + 63) // 64 * 64 * h * w,
                info.LANE_SIZE // itemsize,
                64 * h * w,
                1,
                64 * w,
                64,
            )
            return data_view(shape, stride).reshape(shape[0] * shape[1], -1, h,
                                                    w)[:n, NPU_OFFSET:NPU_OFFSET + c, :, :]

        def get_32ic_data():
            n, c, h, w = memref.shape
            shape = ((n + 63) // 64, 64, (c + 32) // 32, 32, h, w)
            stride = (
                (c + 32) // 32 * 32 * h * w,
                info.LANE_SIZE // itemsize,
                32 * h * w,
                1,
                32 * w,
                32,
            )
            return data_view(shape, stride).reshape(shape[0] * shape[1], -1, h,
                                                    w)[:n, NPU_OFFSET:NPU_OFFSET + c, :, :]

        def get_1ic_data():
            n, c, h, w = memref.shape
            shape = ((n + 63) // 64, 64, c, h, w)
            stride = (
                c * h * w,
                info.LANE_SIZE // itemsize,
                h * w,
                w,
                1,
            )
            return data_view(shape, stride).reshape(-1, c, h, w)[:n,
                                                                 NPU_OFFSET:NPU_OFFSET + c, :, :]

        def get_matrix_data():
            r, c = memref.shape
            w = memref.layout.args[0]
            shape = (r, (c + w - 1) // w, 1, w)
            _memref = copy(memref)
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
            _c = (NPU_OFFSET + c + 63) // 64
            index = np.zeros(_c * 64, bool)
            index[NPU_OFFSET:NPU_OFFSET + c] = True
            index = index.reshape(_c, 64)
            index[:, lane_mask == 0] = False
            return index.flatten()

        def get_dma4bank_data():
            n, c, h, w = memref.shape
            shape = (4, n, (NPU_OFFSET + c + 63) // 64, 64, h, w)
            n_s, c_s, h_s, w_s = memref.stride
            stride = (info.BANK_SIZE, n_s, c_s, info.LANE_SIZE // itemsize, h_s, w_s)
            index = _lane_mask_filter(c, memref.layout.args[0])
            return data_view(shape, stride).reshape(4, n, -1, h, w)[:, :, index, :, :]

        def get_dma_stride_data(_memref=memref):
            n, c, h, w = _memref.shape
            shape = (n, (NPU_OFFSET + c + info.NPU_NUM - 1) // info.NPU_NUM, info.NPU_NUM, h, w)
            n_s, c_s, h_s, w_s = _memref.stride
            stride = (n_s, c_s, info.LANE_SIZE // itemsize, h_s, w_s)
            index = _lane_mask_filter(c, _memref.layout.args[0])
            return data_view(shape, stride).reshape(n, -1, h, w)[:, index, :, :]

        def get_dma_matrix_data():
            r, c = memref.shape
            w = memref.layout.args[1]
            shape = (r, (c + w - 1) // w, 1, w)
            _memref = copy(memref)
            _memref.shape = shape
            return get_dma_stride_data(_memref).reshape(r, -1)[:r, :c]

        def get_dma_linear_data():
            return data_view(memref.shape, memref.stride)

        get_lmem_data = {
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
        self.core_id = core_id
        data = get_lmem_data[memref.layout]()
        if memref.dtype == DType.bf16:
            return bf16_to_fp32(data)
        return data

    def _ddr_to_numpy(self, memref: MemRef):

        def _cal_offset(offsets):
            if len(offsets) <= 1:
                return offsets[0]
            b = offsets.pop()
            a = offsets.pop()
            offsets.append(np.add.outer(a, b))
            return _cal_offset(offsets)

        def _ddr_to_numpy_int4(shape, stride):
            result = np.zeros(shape, dtype=np.uint8)
            offsets = [np.arange(shape[i]) * stride[i] for i in range(len(shape))]
            dst_offset = _cal_offset(offsets).ravel()
            index = memref.r_addr + (dst_offset >> 1)
            values = self.DDR[index].view(np.uint8)
            result = np.where(dst_offset & 1 == 0, values & 0xF, values >> 4).reshape(shape)
            if memref.dtype == DType.si4:
                return np.where(result > 7, result - 16, result).astype(np.int8)
            return result

        elements = 1
        # for i in range(4):
        #     if memref.shape[i] != 0 and memref.stride[i] != 0:
        #         elements = memref.shape[i] * memref.stride[i]
        #         break
        for i in range(4):
            elements += (memref.shape[i] - 1) * memref.stride[i]

        raw_data = np.zeros(elements, dtype=memref.np_dtype)

        self.lib.memcpy_d2s(
            self.runner_p,
            ctypes.c_uint64(memcpy_addr_mask(memref.address)),
            raw_data.size * raw_data.dtype.itemsize,
            raw_data.ctypes.data_as(ctypes.c_void_p),
            memtag(memref.address),
        )
        assert memref.shape is not None
        assert memref.stride is not None
        assert all(memref.shape)
        assert any(memref.stride)

        # offset = memref.r_addr
        if memref.dtype in (DType.i4, DType.si4, DType.ui4):
            return _ddr_to_numpy_int4(memref.shape, memref.stride)
        data = np.lib.stride_tricks.as_strided(
            raw_data[:4].view(memref.np_dtype),
            np.ctypeslib.as_array(memref.shape),
            np.ctypeslib.as_array(memref.stride) * memref.itemsize,
            writeable=False,
        )
        if memref.dtype == DType.bf16:
            return bf16_to_fp32(data)

        return data.copy()

    def clear_memory(self):
        warnings.warn("pcie mode clear memory have no use")

    def get_data_from_address(self, address: int, data: np.ndarray) -> ndarray:
        address += self.neuron_offset
        d2s_ret = self.lib.chip_d2s(
            self.runner_p,
            ctypes.c_uint64(address),
            data.size * data.dtype.itemsize,
            data.ctypes.data_as(ctypes.c_void_p),
        )
        return d2s_ret

    def get_data(self, value_ref: ValueRef):
        value = value_ref.value
        core_id = value_ref.get("core_id", 0)
        if isinstance(value, Scalar):
            return value.data
        assert isinstance(value, MemRef)
        if value.mtype == MType.G:
            return self._ddr_to_numpy(value)
        if value.mtype == MType.R:
            return self._local_mem_to_numpy(value, core_id)
        raise ValueError(f"unsupported memory view: {value}")

    def set_data_to_address(self, address: int, data: np.ndarray):
        s2d_ret = self.lib.memcpy_s2d(
            self.runner_p,
            ctypes.c_uint64(memcpy_addr_mask(address)),
            data.size * data.dtype.itemsize,
            data.ctypes.data_as(ctypes.c_void_p),
            memtag(address),
        )
        assert s2d_ret == 0

    def set_data(self, value: MemRef, data: np.ndarray):
        m_type = value.mtype
        if m_type == MType.G:
            assert data.dtype == value.np_dtype
            s2d_ret = self.lib.memcpy_s2d(
                self.runner_p,
                ctypes.c_uint64(memcpy_addr_mask(value.address)),
                data.size * data.dtype.itemsize,
                data.ctypes.data_as(ctypes.c_void_p),
                memtag(value.address),
            )
            assert s2d_ret == 0
        else:
            raise NotImplementedError(f"Not support setting {m_type} memory data.")

    def check_data(self, gd, address):
        actual = np.zeros_like(gd)
        self.lib.memcpy_d2s(
            self.runner_p,
            ctypes.c_uint64(memcpy_addr_mask(address)),
            actual.size * actual.dtype.itemsize,
            actual.ctypes.data_as(ctypes.c_void_p),
            memtag(address),
        )
        print((gd == actual).all(), gd.sum())
