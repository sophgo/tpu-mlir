# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import copy
import ctypes
from itertools import chain
from typing import Tuple
import numpy as np

from ..target_common.runner import CModelRunner
from ..target_common import *
from .memmap import *

from numpy import ndarray
from typing import List


class BM1688Runner(CModelRunner):
    lib_name = "libcmodel_1688.so"
    # tag, range from 0 to 31, set as defined in /nntoolchain/TPU1686/bm1686/firmware_base/src/fullnet/nodechip_multi_fullnet.c
    TAG_WEIGHT = 1  # coeff
    TAG_ACTIVATION = 2  # neuron
    TAG_LMEM = 3  # lmem

    # ENGINE code, must same as defined in /nntoolchain/TPU1686/sg2260/spec/include/engine_type.h
    ENGINE_GDMA = 1
    ENGINE_HAU = 2

    def __init__(self, memory_size, base_addr: Tuple[int, int]):
        # always init with 2 cores
        super().__init__()
        self.base_addr = base_addr
        self.core_num = 2
        lib = lib_wrapper(open_lib(self.lib_name))
        lib.cmodel_init.argtypes = [ctypes.c_int32, ctypes.c_int64]
        lib.cmodel_init.restype = ctypes.c_int32
        lib.cmodel_deinit.argtypes = [ctypes.c_int32]
        lib.cmodel_deinit.restype = ctypes.c_void_p

        # local_mem
        lib.get_local_mem.argtypes = [ctypes.c_int32]
        lib.get_local_mem.restype = ctypes.POINTER(local_mem)

        lib.get_static_memaddr_by_node.argtypes = [ctypes.c_int32]
        lib.get_static_memaddr_by_node.restype = ctypes.POINTER(ctypes.c_char)

        lib.get_l2_sram.argtypes = [ctypes.c_int32]
        lib.get_l2_sram.restype = ctypes.c_void_p

        lib.get_share_memaddr.argtypes = [ctypes.c_int32]
        lib.get_share_memaddr.restype = ctypes.c_void_p

        lib.get_global_memaddr.argtypes = [ctypes.c_int32]
        lib.get_global_memaddr.restype = ctypes.c_void_p

        lib.cmodel_get_global_mem_size.argtypes = [ctypes.c_int32]
        lib.cmodel_get_global_mem_size.restype = ctypes.c_ulonglong

        # computing function
        lib.execute_command.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint]

        lib.atomic_set_base_ddr.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int32,
            ctypes.c_uint,
        ]
        lib.atomic_set_base_ddr.restype = ctypes.c_void_p

        lib.set_cur_nodechip_idx.argtypes = [ctypes.c_int32]
        lib.set_cur_nodechip_idx.restype = ctypes.c_void_p

        self.lib = lib
        self.init_memory(memory_size)

    def __del__(self):
        base_idx = (ctypes.c_int32 * 3)(1, 2, 31)
        base_addr = (ctypes.c_int64 * 3)(0, 0, 0)
        for i in range(self.core_num):
            self.lib.set_cur_nodechip_idx(i)
            self.lib.atomic_set_base_ddr(base_idx, base_addr, 3, self.ENGINE_GDMA)
            self.lib.cmodel_deinit(i)

    def init_memory(self, memory_size: int):
        self.memory_list = []
        base_idx = (ctypes.c_int32 * 3)(
            self.TAG_WEIGHT, self.TAG_ACTIVATION, self.TAG_LMEM
        )  # config 2 register, the first one is TAG_WEIGHT, use the base_addr_regine{TAG_WEIGHT}. (base_addr_regine0~31)
        LMEM = []
        SMEM = []

        for i in range(self.core_num):
            base_addr = [
                self.base_addr[0],
                self.base_addr[1],
                self.base_addr[2](i),
            ]
            print("using base_addr:", base_addr)
            base_addr = (ctypes.c_uint64 * 3)(*base_addr)
            self.lib.cmodel_init(i, memory_size)
            self.lib.set_cur_nodechip_idx(i)
            self.lib.atomic_set_base_ddr(base_idx, base_addr, 3, self.ENGINE_GDMA)

            LMEM.append(c_array_to_ndarray(self.lib.get_local_mem(i).contents.raw_ptr, (info.NPU_NUM, info.BANK_NUM, info.BANK_SIZE)))
            SMEM.append(c_array_to_ndarray(self.lib.get_static_memaddr_by_node(i), (64 * 1024,)))
        DDR = c_array_to_ndarray(self.lib.get_global_memaddr(0), memory_size)
        self.memory = Memory(LMEM, DDR, SMEM)

    def _compute(self, command: BaseTpuCmd, engine_type):
        atomic = np.frombuffer(command.buf, dtype=np.uint8)
        assert isinstance(atomic, np.ndarray)
        assert atomic.dtype == np.uint8
        return self.lib.execute_command(
            command.core_id,
            atomic.ctypes.data_as(ctypes.c_void_p),
            engine_type,
        )

    def tiu_compute(self, command: BaseTpuCmd):
        return self._compute(command, 0)

    def dma_compute(self, command: BaseTpuCmd):
        return self._compute(command, 1)

    @staticmethod
    def gen_lookup_table():
        # fmt: off
        EXP_COEFF = [
            0x3F800000, 0x3F800000, 0x3F000000, 0x3E2AAAAB, 0x3D2AAAAB, 0x3C088889,
            0x3AB60B61, 0x39500D01, 0x37D00D01, 0x3638EF1D, 0x3493F27E, 0x32D7322B,
            0x310F76C7, 0x2F309231, 0x2D49CBA5, 0x2B573F9F, 0x29573F9F, 0x274A963C,
            0x253413C3, 0x2317A4DA, 0x20F2A15D, 0x1EB8DC78, 0x1C8671CB, 0x1A3B0DA1,
            0x17F96781, 0x159F9E67, 0x13447430, 0x10E8D58E, 0xE850C51, 0xC12CFCC,
            0x99C9963, 0x721A697,
        ]
        LOG_COEFF = [
            0x0, 0x3F800000, 0xBF000000, 0x3EAAAAAB, 0xBE800000, 0x3E4CCCCD,
            0xBE2AAAAB, 0x3E124925, 0xBE000000, 0x3DE38E39, 0xBDCCCCCD, 0x3DBA2E8C,
            0xBDAAAAAB, 0x3D9D89D9, 0xBD924925, 0x3D888889, 0xBD800000, 0x3D70F0F1,
            0xBD638E39, 0x3D579436, 0xBD4CCCCD, 0x3D430C31, 0xBD3A2E8C, 0x3D321643,
            0xBD2AAAAB, 0x3D23D70A, 0xBD1D89D9, 0x3D17B426, 0xBD124925, 0x3D0D3DCB,
            0xBD088889, 0x3D042108, 0xBD000000, 0x3CF83E10, 0xBCF0F0F1, 0x3CEA0EA1,
            0xBCE38E39, 0x3CDD67C9, 0xBCD79436, 0x3CD20D21, 0xBCCCCCCD, 0x3CC7CE0C,
            0xBCC30C31, 0x3CBE82FA, 0xBCBA2E8C, 0x3CB60B61, 0xBCB21643, 0x3CAE4C41,
            0xBCAAAAAB, 0x3CA72F05, 0xBCA3D70A, 0x3CA0A0A1, 0xBC9D89D9, 0x3C9A90E8,
            0xBC97B426, 0x3C94F209, 0xBC924925, 0x3C8FB824, 0xBC8D3DCB, 0x3C8AD8F3,
            0xBC888889, 0x3C864B8A, 0xBC842108, 0x3C820821
        ]
        ERF_COEFF = [
            0xBFA1FC4E, 0x3F8000C7, 0x3EBF88FB, 0x3DC636C9, 0xBE3EC24C, 0x3E8EC7CC,
            0xBF914E5D, 0x3FBE87B0, 0xBF527892, 0x3E2EF945
        ]
        SEQ_COEFF = [
            0x0, 0x1, 0x2, 0x3, 0x4, 0x5,
            0x6, 0x7, 0x8, 0x9, 0xA, 0xB,
            0xC, 0xD, 0xE, 0xF, 0x10, 0x11,
            0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
            0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D,
            0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23,
            0x24, 0x25, 0x26, 0x27, 0x28, 0x29,
            0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F,
            0x30, 0x31, 0x32, 0x33, 0x34, 0x35,
            0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B,
            0x3C, 0x3D, 0x3E, 0x3F
        ]
        SIN_COEFF = [
            0x40C90FDB, 0xC2255DE7, 0x42A335E3, 0xC2996966, 0x42283C1A, 0xC17183A8,
            0x40747A1A, 0xBF37D6DD, 0x3DD57619, 0xBC452021, 0x3A943B81, 0xB8B90AFC,
            0x36C2CE2D, 0xB4AF48D5, 0x32885A92, 0xB03938FD, 0x2DDD95AF, 0xAB6B3C3E,
            0x28DF1AB6, 0xA63E2E9B, 0x23927FCF, 0xA0CCF457, 0x1E02C4A4, 0x9B18D26F,
            0x18242AFD, 0x9522A9A3, 0x12151FD0, 0x8EFDB95A, 0xBC8D5A5, 0x8894490A,
            0x54CBB84, 0x82046EB1
        ]
        COS_COEFF = [
            0x3F800000, 0xC19DE9E6, 0x4281E0F8, 0xC2AAE9E4, 0x4270FA83, 0xC1D368F9,
            0x40FCE9C5, 0xBFDB7128, 0x3E906316, 0xBD15062D, 0x3B77B718, 0xB9A95721,
            0x37C1C6C8, 0xB5BC4EB3, 0x339D55C3, 0xB1647685, 0x2F117937, 0xACA3CB87,
            0x2A2439A9, 0xA7938EFB, 0x24EEFD65, 0xA22F5474, 0x1F6A23B1, 0x9C8EE4DC,
            0x19A008EB, 0x96A50A23, 0x139D3CA3, 0x908ACFA9, 0xD63BDFE, 0x8A2E0D71,
            0x6F87455, 0x83A5FBC7
        ]
        # numpy use little-endian to store bytes
        SIN_FP16_COEFF=[
            (0b0100011001001000)|(0b1101000100101011<<16),
            (0b0101010100011010)|(0b1101010011001011<<16),
            (0b0101000101000010)|(0b1100101110001100<<16),
            (0b0100001110100100)|(0b1011100110111111<<16),
            (0b0010111010101100)|(0b1010001000101001<<16),
            (0b0001010010100010)|(0b1000010111001000<<16),
            (0b0000000001100001)|(0b1000000000000101<<16),
            (0b0000000000000000)|(0b1000000000000000<<16)
        ]
        COS_FP16_COEFF=[
            (0b0011110000000000)|(0b1100110011101111<<16),
            (0b0101010000001111)|(0b1101010101010111<<16),
            (0b0101001110001000)|(0b1100111010011011<<16),
            (0b0100011111100111)|(0b1011111011011100<<16),
            (0b0011010010000011)|(0b1010100010101000<<16),
            (0b0001101110111110)|(0b1000110101001011<<16),
            (0b0000000110000100)|(0b1000000000011000<<16),
            (0b0000000000000001)|(0b1000000000000000<<16)
        ]
        SIN_BFP16_COEFF=[
            (0b0100000011001001)|(0b1100001000100101<<16),
            (0b0100001010100011)|(0b1100001010011001<<16),
            (0b0100001000101000)|(0b1100000101110010<<16),
            (0b0100000001110100)|(0b1011111100111000<<16),
            (0b0011110111010101)|(0b1011110001000101<<16),
            (0b0011101010010100)|(0b1011100010111001<<16),
            (0b0011011011000011)|(0b1011010010101111<<16),
            (0b0011001010001000)|(0b1011000000111001<<16)
        ]
        COS_BFP16_COEFF=[
            (0b0011111110000000)|(0b1100000110011110<<16),
            (0b0100001010000010)|(0b1100001010101011<<16),
            (0b0100001001110001)|(0b1100000111010011<<16),
            (0b0100000011111101)|(0b1011111111011011<<16),
            (0b0011111010010000)|(0b1011110100010101<<16),
            (0b0011101101111000)|(0b1011100110101001<<16),
            (0b0011011111000010)|(0b1011010110111100<<16),
            (0b0011001110011101)|(0b1011000101100100<<16)
        ]
        ARCSIN_COEFF = [
            0x3F800000, 0x3E2AAAAB, 0x3D99999A, 0x3D36DB6E, 0x3CF8E38E, 0x3CB745D1,
            0x3C8E2762, 0x3C64CCCD, 0x3C3D43C4, 0x3C1FEF28, 0x3C09779E, 0x3BEF9DEA,
            0x3BD3431F, 0x3BBC16ED, 0x3BA8DD18, 0x3B98B41E, 0x3B8AF74F, 0x3B7E57C8,
            0x3B69E954, 0x3B58137B, 0x3B4865BC, 0x3B3A86F3, 0x3B2E2FAB, 0x3B232605,
            0x3B193AAC, 0x3B10469B, 0x3B08295E, 0x3B00C7CB, 0x3AF415F3, 0x3AE7BEF5,
            0x3ADC6973, 0x3AD1F8F8, 0x3AC854FC, 0x3ABF683D, 0x3AB7203D, 0x3AAF6CD0,
            0x3AA83FCA, 0x3AA18CAE, 0x3A9B4873, 0x3A956950, 0x3A8FE693, 0x3A8AB879,
            0x3A85D810, 0x3A813F20, 0x3A79D01B, 0x3A719B9A, 0x3A69D79E, 0x3A627BE4,
            0x3A5B80ED, 0x3A54DFE7, 0x3A4E929B, 0x3A48935F, 0x3A42DD01, 0x3A3D6AC2,
            0x3A383845, 0x3A334188, 0x3A2E82DA, 0x3A29F8D4, 0x3A25A052, 0x3A21766C,
            0x3A1D7873, 0x3A19A3E8, 0x3A15F67A, 0x3A126E04
        ]
        TAN_COEFF = [
            0x40490FDB, 0x41255DE7, 0x422335E3, 0x4322FFFD, 0x4422FA39, 0x4522F998,
            0x4622F986, 0x4722F984, 0x4822F983, 0x4922F983, 0x4A22F983, 0x4B22F983,
            0x4C22F983, 0x4D22F983, 0x4E22F983, 0x4F22F983, 0x5022F983, 0x5122F983,
            0x5222F983, 0x5322F983, 0x5422F983, 0x5522F983, 0x5622F983, 0x5722F983,
            0x5822F983, 0x5922F983, 0x5A22F983, 0x5B22F983, 0x5C22F983, 0x5D22F983,
            0x5E22F983, 0x5F22F983
        ]
        EXP_FP16_COEFF = [
            (0x3C00)|(0x3C00<<16), (0x3800)|(0x3155<<16), (0x2955)|(0x2044<<16),
            (0x15B0)|(0xA80<<16), (0x1A0)|(0x2E<<16), (0x5)|(0x0<<16), (0x0)|(0x0<<16), (0x0)|(0x0<<16)]
        EXP_BF16_COEFF = [
            (0x3F80)|(0x3F80<<16), (0x3F00)|(0x3E2B<<16), (0x3D2B)|(0x3C09<<16), (0x3AB6)|(0x3950<<16),
            (0x37D0)|(0x3639<<16), (0x3494)|(0x32D7<<16), (0x310F)|(0x2F31<<16), (0x2D4A)|(0x2B57<<16)]
        ERF_FP16_COEFF = [
            (0xBD10)|(0x3C00<<16), (0x35FC)|(0x2E32<<16), (0xB1F6)|(0x3476<<16), (0xBC8A)|(0x3DF4<<16),
            (0xBA94)|(0x3178<<16), (0x00)|(0x00<<16), (0x00)|(0x00<<16), (0x00)|(0x00<<16)
        ]
        ERF_BF16_COEFF = [
            (0xBFA2)|(0x3F80<<16), (0x3EC0)|(0x3DC6<<16), (0xBE3F)|(0x3E8F<<16), (0xBF91)|(0x3FBF<<16),
            (0xBF52)|(0x3E2F<<16), (0x00)|(0x00<<16), (0x00)|(0x00<<16), (0x00)|(0x00<<16),
        ]
        LOG_FP16_COEFF = [
            (0x0)|(0x3c00<<16), (0xb800)|(0x3555<<16), (0xb400)|(0x3266<<16), (0xb155)|(0x3092<<16),
            (0xb000)|(0x2f1c<<16), (0xae66)|(0x2dd1<<16), (0xad55)|(0x2cec<<16), (0xac92)|(0x2c44<<16)
        ]
        LOG_BF16_COEFF = [
            (0x0)|(0x3f80<<16), (0xbf00)|(0x3eab<<16), (0xbe80)|(0x3e4d<<16), (0xbe2b)|(0x3e12<<16),
            (0xbe00)|(0x3de4<<16), (0xbdcd)|(0x3dba<<16), (0xbdab)|(0x3d9e<<16), (0xbd92)|(0x3d89<<16),
        ]

        table = (
                (EXP_COEFF, 32),
                (LOG_COEFF, 64),
                (ERF_COEFF, 16),
                (SEQ_COEFF, 64),
                (SIN_COEFF, 32),
                (COS_COEFF, 32),
                (ARCSIN_COEFF, 64),
                (TAN_COEFF, 32),
                (EXP_FP16_COEFF, 8),
                (EXP_BF16_COEFF, 8),
                (ERF_FP16_COEFF, 8),
                (ERF_BF16_COEFF, 8),
                (LOG_FP16_COEFF, 8),
                (LOG_BF16_COEFF, 16),
                (SIN_FP16_COEFF,16),
                (SIN_BFP16_COEFF,16),
                (COS_FP16_COEFF,16),
                (COS_BFP16_COEFF,16)
        )

        def align_to_bytes(x):
            x_len = len(x[0])
            space = x[1]
            padding = int(np.ceil(space / 16) * 16) - x_len
            return x[0] + [0] * padding

        return list(chain.from_iterable((align_to_bytes(x) for x in table)))


class Memory(CModelMemory):
    """
    Memory agent. Extract/Set data from a give MemRef object.
    This class should handle all the tenors type in all kinds of storage.
    """

    def __init__(self, LMEM: List[ndarray], DDR: ndarray, SMEM: List[ndarray]) -> None:
        self.DDR = DDR.ravel()
        self.LMEM = []
        self.SMEM = []
        for lmem in LMEM:
            self.LMEM.append(lmem.ravel())
        for smem in SMEM:
            self.SMEM.append(smem.ravel())

    def _local_mem_to_numpy(self, memref: MemRef, core_id):
        NPU_OFFSET = memref.npu_offset
        itemsize = memref.itemsize

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
                    LMEM = self.LMEM[self.core_id][
                        npu_idx * info.LANE_SIZE : (npu_idx + 1) * info.LANE_SIZE
                    ]
                    c_offset = ((start_npu_idx + cidx) / info.NPU_NUM) * stride[1]
                    h_offset = np.arange(0, shape[2]) * stride[2]
                    w_offset = np.arange(0, shape[3]) * stride[3]
                    dst_offset = np.add.outer(
                        n_offset,
                        np.add.outer(c_offset, np.add.outer(h_offset, w_offset)),
                    ).ravel()
                    index = start_offset + (dst_offset >> 1)
                    values = LMEM[index].view(np.uint8)
                    result[nidx][cidx] = np.where(
                        dst_offset & 1 == 0, values & 0xF, values >> 4
                    )
            result.reshape(shape)
            if memref.dtype == DType.si4:
                return np.where(result > 7, result - 16, result).astype(np.int8)
            return result

        def data_view(shape, stride):
            offset = memref.r_addr - NPU_OFFSET * info.LANE_SIZE
            return np.lib.stride_tricks.as_strided(
                self.LMEM[self.core_id][offset : offset + 4].view(memref.np_dtype),
                shape,
                np.array(stride) * itemsize,
                writeable=False,
            )

        def get_stride_data_base(shape, stride):
            n, c, h, w = shape
            n_s, c_s, h_s, w_s = stride
            _shape = [n, div_up(NPU_OFFSET + c, info.NPU_NUM), info.NPU_NUM, h, w]
            _stride = (n_s, c_s, int(info.LANE_SIZE / itemsize), h_s, w_s)
            return data_view(_shape, _stride).reshape(n, -1, h, w)[
                :n, NPU_OFFSET : NPU_OFFSET + c, :, :
            ]

        def get_stride_data():
            if memref.dtype in (DType.i4, DType.si4, DType.ui4):
                return data_view_int4(memref.shape, memref.stride)
            return get_stride_data_base(memref.shape, memref.stride)

        def get_alignic_data():
            n, c, h, w = memref.shape
            cube_num = info.CUBE_NUM(memref.dtype)
            shape = (div_up(n, info.NPU_NUM), info.NPU_NUM, div_up(c, cube_num), cube_num, h, w)
            stride = (
                align_up(c, cube_num) * h * w,
                int(info.LANE_SIZE / itemsize),
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
            stride = (info.BANK_SIZE, n_s, c_s, int(info.LANE_SIZE / itemsize), h_s, w_s)
            index = _lane_mask_filter(c, memref.layout.args[0])
            return data_view(shape, stride).reshape(4, n, -1, h, w)[:, :, index, :, :]

        def get_dma_stride_data(_memref=memref):
            n, c, h, w = _memref.shape
            shape = (n, div_up(NPU_OFFSET + c, info.NPU_NUM), info.NPU_NUM, h, w)
            n_s, c_s, h_s, w_s = _memref.stride
            stride = (n_s, c_s, int(info.LANE_SIZE / itemsize), h_s, w_s)
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
            result = np.where(dst_offset & 1 == 0, values & 0xF, values >> 4).reshape(
                shape
            )
            if memref.dtype == DType.si4:
                return np.where(result > 7, result - 16, result).astype(np.int8)
            return result

        assert memref.shape is not None
        assert memref.stride is not None
        assert all(memref.shape)
        assert any(memref.stride)
        offset = memref.r_addr
        if memref.dtype in (DType.i4, DType.si4, DType.ui4):
            return _ddr_to_numpy_int4(memref.shape, memref.stride)
        data = np.lib.stride_tricks.as_strided(
            self.DDR[offset : offset + 4].view(memref.np_dtype),
            np.ctypeslib.as_array(memref.shape),
            np.ctypeslib.as_array(memref.stride) * memref.itemsize,
            writeable=False,
        )
        if memref.dtype == DType.bf16:
            return bf16_to_fp32(data)
        return data

    def clear_memory(self):
        self.DDR.fill(0)
        for lmem in self.LMEM:
            lmem.fill(0)

        lut = np.array(BM1688Runner.gen_lookup_table(), np.uint32).view(np.uint8)
        for smem in self.SMEM:
            smem[: len(lut)] = lut[...]

    def get_data(self, value_ref: ValueRef):
        value = value_ref.value
        core_id = value_ref.get('core_id', 0)
        if isinstance(value, Scalar):
            return value.data
        assert isinstance(value, MemRef)
        if value.mtype == MType.G:
            return self._ddr_to_numpy(value)
        if value.mtype == MType.R:
            return self._local_mem_to_numpy(value, core_id)
        raise ValueError(f"unsupported memory view: {value}")

    def set_data(self, value, data: np.ndarray):
        m_type = value.mtype
        if m_type == MType.G:
            offset = value.r_addr
            assert data.dtype == value.np_dtype
            src_u8 = np.ascontiguousarray(data.flatten()).view(np.uint8)
            self.DDR[offset : offset + src_u8.size] = src_u8.flatten()
            return
        raise NotImplementedError(f"Not support setting {m_type} memory data.")
