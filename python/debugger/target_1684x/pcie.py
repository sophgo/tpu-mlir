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
from .opdef import TiuCmd, DmaCmd, BaseTpuCmd
from .memmap import *
from ..plugins.common import FinalMlirIndexPlugin


class soc_launch_struct:
    def __init__(self, tiu_num, dma_num, tiu_buf, dma_buf):
        self.tiu_num = tiu_num
        self.dma_num = dma_num
        self.tiu_buf = tiu_buf
        self.dma_buf = dma_buf
        self.tiu_buf_len = len(tiu_buf)
        self.dma_buf_len = len(dma_buf)


class BM1684XRunner(DeviceRunner):
    lib_name = "libatomic_exec.so"
    soc_structs = []

    def __init__(self, memory_size, use_pcie=False):
        super().__init__()
        lib = lib_wrapper(open_lib(self.lib_name))

        self.lib = lib
        self.kernel_fn = os.path.join(
            os.environ["TPUC_ROOT"], "lib/libbm1684x_atomic_kernel.so"
        )
        self.fast_checker = False
        if use_pcie:
            self.use_pcie()
        else:
            self.use_soc()

        # used for get_cur_op_cmds()
        self.op_start_execute_id = 1
        self.in_op = False
        self.cur_op_cmds = 0

    def use_pcie(self):
        self.is_pcie = True
        self.is_soc = False
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

    def use_soc(self):
        self.is_pcie = False
        self.is_soc = True

    def trans_cmds_to_buf(self, cmds, engine_type):
        buf_list = []
        for cmd in cmds:
            reg = copy(cmd.reg)
            if engine_type == 1:  # dma
                u32_buf = (ctypes.c_uint32 * (len(cmd.buf) // 4)).from_buffer_copy(reg)
                self.lib.convert_addr(u32_buf, self.reserved_offset)
                buf = bytes(u32_buf)
            else:
                buf = bytes(reg)
            buf_list.append(buf)
        return b"".join(buf_list)

    def trans_cmds_to_buf_soc(self, cmds, engine_type):
        buf_list = []
        for cmd in cmds:
            reg = copy(cmd.reg)
            if engine_type == 1:  # dma
                u32_buf = (ctypes.c_uint32 * (len(cmd.buf) // 4)).from_buffer_copy(reg)
                buf = bytes(u32_buf)
            else:
                buf = bytes(reg)
            buf_list.append(buf)
        return buf_list

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

    def init_memory(self):
        self.memory = Memory(self.lib, self.runner)

    def tiu_compute(self, command: TiuCmd):
        return self._compute(command, 0)

    def dma_compute(self, command: DmaCmd):
        return self._compute(command, 1)

    def get_stack_cmds(self, cur_cmd_point, cmditer):
        tiu = []
        dma = []
        df = FinalMlirIndexPlugin.data_frame
        buf_cmds = 0

        while cur_cmd_point + buf_cmds < len(cmditer):
            if cmditer[cur_cmd_point + buf_cmds].cmd_type == CMDType.tiu:
                tiu.append(cmditer[cur_cmd_point + buf_cmds])
            else:
                dma.append(cmditer[cur_cmd_point + buf_cmds])

            result_not_empty = (
                len(df.loc[df["executed_id"] == cur_cmd_point + 1 + buf_cmds, "results"].tolist()[0])> 0
            )

            if result_not_empty:
                break
            buf_cmds += 1

        return tiu, dma

    def get_cur_op_cmds(self, cur_cmd_point, cmditer):
        if not self.in_op:
            self.op_start_execute_id = cur_cmd_point + 1
            self.in_op = True
            tiu, dma = self.get_stack_cmds(cur_cmd_point, cmditer)
            self.cur_op_cmds = len(tiu) + len(dma)
            if self.cur_op_cmds == 1:
                self.in_op = False
        else:
            if cur_cmd_point + 1 == self.op_start_execute_id + self.cur_op_cmds - 1:
                self.in_op = False
        return self.cur_op_cmds

    def checker_fast_compute_soc(self, cmd_group: StaticCmdGroup):
        tiu, dma = cmd_group.tiu, cmd_group.dma
        tiu_buf = self.trans_cmds_to_buf_soc(tiu, 0)
        dma_buf = self.trans_cmds_to_buf_soc(dma, 1)

        BM1684XRunner.soc_structs.append(
            soc_launch_struct(len(tiu), len(dma), tiu_buf, dma_buf)
        )
        return len(tiu) + len(dma)

    def fast_compute_soc(self, command: BaseTpuCmd):
        tiu = []
        dma = []
        if command.cmd_type == CMDType.tiu:
            tiu.append(command)
            tiu_buf = self.trans_cmds_to_buf_soc(tiu, 0)
            dma_buf = b""
        else:
            dma.append(command)
            tiu_buf = b""
            dma_buf = self.trans_cmds_to_buf_soc(dma, 1)
        assert len(tiu) + len(dma) == 1

        BM1684XRunner.soc_structs.append(
            soc_launch_struct(len(tiu), len(dma), tiu_buf, dma_buf)
        )
        return 1

    def checker_fast_compute(self, cmd_group: StaticCmdGroup):
        tiu, dma = cmd_group.tiu, cmd_group.dma
        tiu_buf = self.trans_cmds_to_buf(tiu, 0)
        dma_buf = self.trans_cmds_to_buf(dma, 1)

        ret = self.lib.launch_cmds_in_pio(
            self.runner,
            ctypes.byref(ctypes.create_string_buffer(tiu_buf)),
            ctypes.byref(ctypes.create_string_buffer(dma_buf)),
            ctypes.c_size_t(len(tiu_buf)),
            ctypes.c_size_t(len(dma_buf)),
            ctypes.c_int(len(tiu)),
            ctypes.c_int(len(dma)),
        )
        assert ret == 0
        return len(tiu) + len(dma)

    def fast_compute(self, command: BaseTpuCmd):
        tiu = []
        dma = []
        if command.cmd_type == CMDType.tiu:
            tiu.append(command)
            tiu_buf = self.trans_cmds_to_buf(tiu, 0)
            dma_buf = b""
        else:
            dma.append(command)
            tiu_buf = b""
            dma_buf = self.trans_cmds_to_buf(dma, 1)
        assert len(tiu) + len(dma) == 1
        ret = self.lib.launch_cmd_in_pio(
            self.runner,
            ctypes.byref(ctypes.create_string_buffer(tiu_buf)),
            ctypes.byref(ctypes.create_string_buffer(dma_buf)),
            ctypes.c_size_t(len(tiu_buf)),
            ctypes.c_size_t(len(dma_buf)),
            ctypes.c_int(len(tiu)),
            ctypes.c_int(len(dma)),
        )
        assert ret == 0
        return 1

    def cmds_compute(self, commands: StaticCmdGroup):
        if self.is_pcie:
            if not self.fast_checker or len(commands.all) == 1:
                self.fast_compute(commands.all[0])
            else:
                self.checker_fast_compute(commands)
        elif self.is_soc:
            if not self.fast_checker or len(commands.all) == 1:
                self.fast_compute_soc(commands.all[0])
            else:
                self.checker_fast_compute_soc(commands)



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
                self.LMEM[offset : offset + 4].view(memref.np_dtype),
                shape,
                np.array(stride) * itemsize,
                writeable=False,
            )

        def get_stride_data_base(shape, stride):
            n, c, h, w = shape
            n_s, c_s, h_s, w_s = stride
            _shape = [n, (NPU_OFFSET + c + 63) // 64, 64, h, w]
            _stride = (n_s, c_s, info.LANE_SIZE // itemsize, h_s, w_s)
            return data_view(_shape, _stride).reshape(n, -1, h, w)[
                :n, NPU_OFFSET : NPU_OFFSET + c, :, :
            ]

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
            return data_view(shape, stride).reshape(shape[0] * shape[1], -1, h, w)[
                :n, NPU_OFFSET : NPU_OFFSET + c, :, :
            ]

        def get_matrix_data():
            r, c = memref.shape
            w = memref.layout.args[0]
            shape = (r, div_up(c, w), 1, w)
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
            lane_mask = np.unpackbits(
                np.uint64([lane_mask]).view(np.uint8), bitorder="little"
            )
            _c = div_up(NPU_OFFSET + c, info.NPU_NUM)
            index = np.zeros(_c * info.NPU_NUM, bool)
            index[NPU_OFFSET : NPU_OFFSET + c] = True
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
