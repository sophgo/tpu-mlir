# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import ctypes
import numpy as np
import os
from ctypes import Structure, POINTER
from itertools import chain


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


def open_lib(lib_name):
    try:
        lib_path = os.environ["LD_LIBRARY_PATH"]
        lib_full_name = None
        for path in lib_path.split(":"):
            if os.path.isfile(os.path.join(path, lib_name)):
                lib_full_name = os.path.join(path, lib_name)
                break
        if not lib_full_name:
            raise OSError
        return ctypes.CDLL(lib_full_name)
    except OSError as e:
        msg = f"""Could not find/load shared object file: {lib_name}
     Error was: {e}"""
        raise OSError(msg)


def c_array_to_ndarray(x, shape):
    if isinstance(x, int):
        x = ctypes.c_void_p(x)
    if isinstance(shape, int):
        shape = (shape,)
    try:
        p = ctypes.cast(x, ctypes.POINTER(ctypes.c_uint8))
    except:
        raise Exception(f"unsupported memory access: {x}")
    else:
        return np.ctypeslib.as_array(p, shape=shape)


class BM1684X:
    lib_name = "libcmodel_1684x.so"

    def __init__(self, memory_size):
        lib = _lib_wrapper(open_lib(self.lib_name))
        lib.cmodel_init.argtypes = [ctypes.c_int32, ctypes.c_int64]
        lib.cmodel_init.restype = ctypes.c_int32
        lib.cmodel_deinit.argtypes = [ctypes.c_int32]
        lib.cmodel_multi_thread_cxt_deinit.argtypes = [ctypes.c_int32]
        # local_mem
        lib.get_local_mem.argtypes = [ctypes.c_int32]
        lib.get_local_mem.restype = ctypes.POINTER(local_mem)
        lib.clear_local_mem.argtypes = [ctypes.c_int32]
        lib.fill_local_mem.argtypes = [ctypes.c_int32]

        lib.get_static_memaddr_by_node.argtypes = [ctypes.c_int32]
        lib.get_static_memaddr_by_node.restype = ctypes.POINTER(ctypes.c_char)

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

        # computing function
        lib.execute_command.argtypes = [
            ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint]

        self.lib = lib
        self.__setup(memory_size)

    def __setup(self, memory_size):
        self.lib.cmodel_init(0, memory_size)
        # self.lib.cmodel_multi_thread_cxt_deinit(0)
        self.DDR = c_array_to_ndarray(
            self.lib.get_global_memaddr(0), memory_size)
        self.LMEM = c_array_to_ndarray(
            self.lib.get_local_mem(0).contents.raw_ptr, (64, 16, 1024 * 16)
        )
        self.SMEM = c_array_to_ndarray(
            self.lib.get_static_memaddr_by_node(0), (16 * 1024,)
        )

    def clear_memory(self):
        self.DDR.fill(0)
        self.LMEM.fill(0)
        lut = np.array(self.gen_lookup_table(), np.uint32).view(np.uint8)
        self.SMEM[: len(lut)] = lut[...]

    def __del__(self):
        self.lib.cmodel_deinit(0)

    def compute(self, command, engine_type):
        command = np.frombuffer(command, dtype=np.uint8)
        assert isinstance(command, np.ndarray)
        assert command.dtype == np.uint8
        return self.lib.execute_command(
            0,
            command.ctypes.data_as(ctypes.c_void_p),
            engine_type,
        )

    def tiu_compute(self, command, core_id=0):
        return self.compute(command, 0)

    def dma_compute(self, command, core_id=0):
        return self.compute(command, 1)

    @staticmethod
    def gen_lookup_table():
        # firmware_runtime.c:340
        # fmt: off
        EXP_COEFF = [
            0x3F800000, 0x3F800000, 0x3F000000, 0x3E2AAAAB, 0x3D2AAAAB, 0x3C088889,
            0x3AB60B61, 0x39500D01, 0x37D00D01, 0x3638EF1D, 0x3493F27E, 0x32D7322B,
            0x310F76C7, 0x2F309231, 0x2D49CBA5, 0x2B573F9F, 0x29573F9F, 0x274A963C,
            0x253413C3, 0x2317A4DA, 0x20F2A15D, 0x1EB8DC78, 0x1C8671CB, 0x1A3B0DA1,
            0x17F96781, 0x159F9E67, 0x13447430, 0x10E8D58E, 0xE850C51, 0xC12CFCC,
            0x99C9963, 0x721A697,
        ]
        EXP_TABLE = [
            0x1, 0x4, 0xA, 0x1B, 0x48, 0xC4, 0x215, 0x5A9, 0xF64, 0x29D6, 0x71B9,
            0x13521, 0x3484B, 0x8EC28, 0x1840FC, 0x41EDC4, 0xB33687,
            0x1739362, 0x22586E0, 0x2E0F96D, 0x398E2CB, 0x44FCB22, 0x50D35D7,
            0x5BFECBA, 0x6826D27, 0x7314490, 0x7F0EE94, 0x8A3BAF0, 0x95E884F,
            0xA1739FB, 0xACD89C1, 0xB8BAD78, 0xC3DD771, 0xD0102BF, 0xDAF5800,
            0xE6E511E, 0xF21F3FE, 0xFDC1DF9, 0x109595C7, 0x114B4EA4, 0x120A295C,
            0x12BBC7F1, 0x137F388B, 0x142D70C9, 0x14EBBAEC, 0x15A031FC, 0x1659BA5A,
            0x1713F623, 0x17C919B9, 0x1888A975, 0x1939BE2B, 0x19FC7361, 0x1AAB8EDC,
            0x1B692BEB, 0x1C1E74DD, 0x1CD75D5D, 0x1D925B02, 0x1E46EAF1, 0x1F072DBA,
            0x1FB7BA0F, 0x2079B5EA, 0x2129B229, 0x21E6A405, 0x229CBC92, 0x235506F2,
            0x2410C457, 0x24C4C239, 0x2585B61D, 0x2635BB8D, 0x26F7000F, 0x27A7DAA4,
            0x28642328, 0x291B090F, 0x29D2B706, 0x2A8F3216, 0x2B429F81, 0x2C044295,
            0x2CB3C295, 0x2D7451BD, 0x2E26083C, 0x2EE1A93F, 0x2F995A46, 0x30506D87,
            0x310DA433, 0x31C082B8, 0x3282D314, 0x3331CF19, 0x33F1AADE, 0x34A43AE5,
            0x355F3638, 0x3617B02A, 0x36CE2A62, 0x378C1AA1, 0x383E6BCE, 0x39016791,
            0x39AFE108, 0x3A6F0B5D, 0x3B227290, 0x3BDCC9FF, 0x3C960AAE, 0x3D4BED86,
            0x3E0A9555, 0x3EBC5AB2, 0x3F800000, 0x402DF854, 0x40EC7326, 0x41A0AF2E,
            0x425A6481, 0x431469C5, 0x43C9B6E3, 0x44891443, 0x453A4F54, 0x45FD38AC,
            0x46AC14EE, 0x4769E224, 0x481EF0B3, 0x48D805AD, 0x4992CD62, 0x4A478665,
            0x4B07975F, 0x4BB849A4, 0x4C7A7910, 0x4D2A36C8, 0x4DE75844, 0x4E9D3710,
            0x4F55AD6E, 0x5011357A, 0x50C55BFE, 0x51861E9D, 0x52364993, 0x52F7C118,
            0x53A85DD2, 0x5464D572, 0x551B8238, 0x55D35BB3, 0x568FA1FE, 0x5743379A,
            0x5804A9F1, 0x58B44F11, 0x597510AD, 0x5A2689FE, 0x5AE2599A, 0x5B99D21F,
            0x5C51106A, 0x5D0E12E4, 0x5DC1192B, 0x5E833952, 0x5F325A0E, 0x5FF267BB,
            0x60A4BB3E, 0x615FE4A9, 0x621826B5, 0x62CECB81, 0x638C881F, 0x643F009E,
            0x6501CCB3, 0x65B06A7B, 0x666FC62D, 0x6722F184, 0x67DD768B, 0x68967FF0,
            0x694C8CE5, 0x6A0B01A3, 0x6ABCEDE5, 0x6B806408, 0x6C2E804A, 0x6CED2BEF,
            0x6DA12CC1, 0x6E5B0F2E, 0x6F14DDC1, 0x6FCA5487, 0x70897F64, 0x713AE0EE,
            0x71FDFE91, 0x72AC9B6A, 0x736A98EC, 0x741F6CE9, 0x74D8AE7F, 0x7593401C,
            0x76482254, 0x77080156, 0x77B8D9AA, 0x787B3CCF, 0x792ABBCE, 0x79E80D11,
            0x7A9DB1ED, 0x7B56546B, 0x7C11A6F5, 0x7CC5F63B, 0x7D86876D, 0x7E36D809,
            0x7EF882B7
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
            0x3C00, 0x3C00, 0x3800, 0x3155, 0x2955, 0x2044, 0x15B0, 0xA80, 0x1A0, 0x2E
        ]
        EXP_BF16_COEFF = [
            0x3f80, 0x3f80, 0x3f00, 0x3e2b, 0x3d2b, 0x3c09, 0x3ab6, 0x3950, 0x37d0, 0x3639
        ]
        LOG_FP16_COEFF = [
            0x0, 0x3c00, 0xb800, 0x3555, 0xb400, 0x3266, 0xb155, 0x3092, 0xb000,
            0x2f1c, 0xae66, 0x2dd1, 0xad55, 0x2cec, 0xac92, 0x2c44
        ]
        LOG_BF16_COEFF = [
            0x0, 0x3f80, 0xbf00, 0x3eab, 0xbe80, 0x3e4d, 0xbe2b, 0x3e12, 0xbe00,
            0x3de4, 0xbdcd, 0x3dba, 0xbdab, 0x3d9e, 0xbd92, 0x3d89,
        ]
        # fmt: on
        table = (
            # table, space BUT we have to align :(
            (EXP_TABLE, 256),
            (EXP_COEFF, 32),
            (LOG_COEFF, 64),
            (ERF_COEFF, 16),
            (SEQ_COEFF, 64),
            (SIN_COEFF, 32),
            (COS_COEFF, 32),
            (ARCSIN_COEFF, 64),
            (TAN_COEFF, 32),
            (EXP_FP16_COEFF, 16),
            (EXP_BF16_COEFF, 16),
            (LOG_FP16_COEFF, 16),
            (LOG_BF16_COEFF, 16)
        )

        def align_to_64bytes(x):
            x_len = len(x[0])
            space = x[1]
            padding = int(np.ceil(space / 16) * 16) - x_len
            return x[0] + [0] * padding

        return list(chain.from_iterable((align_to_64bytes(x) for x in table)))


class BM1684:
    lib_name = "libcmodel_1684.so"

    def __init__(self, memory_size):
        lib = _lib_wrapper(open_lib(self.lib_name))
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

        # async function
        lib.cmodel_call_atomic.argtypes = [
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]

        self.lib = lib
        self.__setup(memory_size)

    def __setup(self, memory_size):
        self.lib.cmodel_init(0, memory_size)
        self.DDR = c_array_to_ndarray(
            self.lib.get_global_memaddr(0), memory_size)
        self.LMEM = c_array_to_ndarray(
            self.lib.get_local_mem(0).contents.raw_ptr, (64, 8, 1024 * 64)
        )
        self.L2SRAM = c_array_to_ndarray(
            self.lib.get_l2_sram(0), (4096 * 1024,))

    def clear_memory(self):
        self.DDR.fill(0)
        self.LMEM.fill(0)
        lut = np.array(self.gen_lookup_table(), np.uint32).view(np.uint8)
        self.L2SRAM[: len(lut)] = lut[...]

    def __del__(self):
        self.lib.cmodel_deinit(0)

    def compute(self, command, engine_type):
        command = np.frombuffer(command, dtype=np.uint8)
        cmd_p = command.ctypes.data_as(ctypes.c_void_p)
        return self.lib.get_atomic_function(cmd_p, engine_type)(0, cmd_p)

    def tiu_compute(self, command, core_id=0):
        return self.compute(command, 0)

    def dma_compute(self, command, core_id=0):
        return self.compute(command, 1)

    @staticmethod
    def gen_lookup_table():
        return []


class BM1686:
    lib_name = "libcmodel_1686.so"
    ENGINE_GDMA = 1
    TAG_WEIGHT = 1
    ENGINE_HAU = 2
    TAG_ACTIVATION = 2
    # always init with 2 cores
    core_num = 2

    def __init__(self, memory_size, base_addr):
        lib = _lib_wrapper(open_lib(self.lib_name))
        lib.cmodel_init.argtypes = [ctypes.c_int32, ctypes.c_int64]
        lib.cmodel_init.restype = ctypes.c_int32
        lib.cmodel_deinit.argtypes = [ctypes.c_int32]

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
        lib.execute_command.argtypes = [
            ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint]

        lib.atomic_set_base_ddr.argtypes = [ctypes.POINTER(
            ctypes.c_int32), ctypes.POINTER(ctypes.c_uint64), ctypes.c_int32, ctypes.c_uint]
        lib.atomic_set_base_ddr.restype = ctypes.c_void_p

        lib.set_cur_nodechip_idx.argtypes = [ctypes.c_int32]
        lib.set_cur_nodechip_idx.restype = ctypes.c_void_p

        self.lib = lib
        self.__setup(memory_size, base_addr)

    def __setup(self, memory_size, base_addr):
        base_idx = (ctypes.c_int32 * 2)(self.TAG_WEIGHT, self.TAG_ACTIVATION)
        base_addr = (ctypes.c_uint64 * 2)(base_addr[0], base_addr[1])
        self.LMEM = []
        for i in range(self.core_num):
            self.lib.cmodel_init(i, memory_size)
            self.lib.set_cur_nodechip_idx(i)
            self.lib.atomic_set_base_ddr(
                base_idx, base_addr, 2, self.ENGINE_GDMA)
            self.LMEM.append(c_array_to_ndarray(
                self.lib.get_local_mem(i).contents.raw_ptr, (32, 16, 1024 * 8)
            ))
        self.DDR = c_array_to_ndarray(
            self.lib.get_global_memaddr(0), memory_size)
        self.SMEM = c_array_to_ndarray(
            self.lib.get_static_memaddr_by_node(0), (16 * 1024,)
        )

    def clear_memory(self):
        self.DDR.fill(0)
        for lmem in self.LMEM:
            lmem.fill(0)
        lut = np.array(self.gen_lookup_table(), np.uint32).view(np.uint8)
        self.SMEM[: len(lut)] = lut[...]

    def __del__(self):
        base_idx = (ctypes.c_int32 * 2)(1, 2)
        base_addr = (ctypes.c_uint64 * 2)(0, 0)
        for i in range(self.core_num):
            self.lib.set_cur_nodechip_idx(i)
            self.lib.atomic_set_base_ddr(
                base_idx, base_addr, 2, self.ENGINE_GDMA)
            self.lib.cmodel_deinit(i)

    def compute(self, command, engine_type, core_id=0):
        command = np.frombuffer(command, dtype=np.uint8)
        assert isinstance(command, np.ndarray)
        assert command.dtype == np.uint8
        return self.lib.execute_command(
            core_id,
            command.ctypes.data_as(ctypes.c_void_p),
            engine_type,
        )

    def tiu_compute(self, command, core_id=0):
        return self.compute(command, 0, core_id)

    def dma_compute(self, command, core_id=0):
        return self.compute(command, 1, core_id)

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
        ] + [0] * 6
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
            0x3C00, 0x3C00, 0x3800, 0x3155, 0x2955, 0x2044,
            0x15B0, 0xA80, 0x1A0, 0x2E, 0x5,0x0, 0x0, 0x0, 0x0, 0x0]
        EXP_BF16_COEFF = [
            0x3F80, 0x3F80, 0x3F00, 0x3E2B, 0x3D2B, 0x3C09, 0x3AB6, 0x3950,
            0x37D0, 0x3639, 0x3494, 0x32D7, 0x310F, 0x2F31, 0x2D4A, 0x2B57 ]
        ERF_FP16_COEFF = [
            0xBD10, 0x3C00, 0x35FC, 0x2E32, 0xB1F6, 0x3476, 0xBC8A, 0x3DF4,
            0xBA94, 0x3178, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        ]
        ERF_BF16_COEFF = [
            0xBFA2, 0x3F80, 0x3EC0, 0x3DC6, 0xBE3F, 0x3E8F, 0xBF91, 0x3FBF,
            0xBF52, 0x3E2F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]
        LOG_FP16_COEFF = [
            0x0, 0x3c00, 0xb800, 0x3555, 0xb400, 0x3266, 0xb155, 0x3092, 0xb000,
            0x2f1c, 0xae66, 0x2dd1, 0xad55, 0x2cec, 0xac92, 0x2c44
        ]
        LOG_BF16_COEFF = [
            0x0, 0x3f80, 0xbf00, 0x3eab, 0xbe80, 0x3e4d, 0xbe2b, 0x3e12, 0xbe00,
            0x3de4, 0xbdcd, 0x3dba, 0xbdab, 0x3d9e, 0xbd92, 0x3d89,
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
            (EXP_FP16_COEFF, 16),
            (EXP_BF16_COEFF, 16),
            (ERF_FP16_COEFF, 16),
            (ERF_BF16_COEFF, 16),
            (LOG_FP16_COEFF, 16),
            (LOG_BF16_COEFF, 16)
        )

        def align_to_64bytes(x):
            x_len = len(x[0])
            space = x[1]
            padding = int(np.ceil(space / 16) * 16) - x_len
            return x[0] + [0] * padding

        return list(chain.from_iterable((align_to_64bytes(x) for x in table)))