# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import platform
import os
import shutil
import numpy as np
import ctypes
from functools import lru_cache
from ctypes import Structure, POINTER
from numpy import ndarray
from .op_support import MemRefBase, Value, CpuCmd, get_type_str, ValueRef, StaticCmdGroup, CMDType
from typing import List
import tempfile


def c_array_to_ndarray(x, shape):
    if isinstance(x, int):
        x = ctypes.c_void_p(x)
    if isinstance(shape, int):
        shape = (shape,)
    try:
        p = ctypes.cast(x, ctypes.POINTER(ctypes.c_uint8))
    except Exception:
        raise Exception(f"unsupported memory access: {x}")
    else:
        return np.ctypeslib.as_array(p, shape=shape)


class local_mem(Structure):
    _fields_ = [
        ("raw_ptr", POINTER(ctypes.c_char)),
        ("mem_arr", POINTER(POINTER(ctypes.c_uint32))),
        ("count", ctypes.c_int32),
        ("size_per_mem", ctypes.c_int32),
        ("align_num", ctypes.c_int32),
        ("need_free", ctypes.c_int32),
    ]


class lib_wrapper:
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

    def __repr__(self) -> str:
        return f"wrap({self._lib})"

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


# @contextmanager
def temp_position(file):
    os.makedirs(os.path.expanduser("~/.cache/tpu-mlir"), exist_ok=True)
    tempdirname = tempfile.TemporaryDirectory(
        dir=os.path.expanduser("~/.cache/tpu-mlir")
    ).name
    # make sure
    os.makedirs(tempdirname, exist_ok=True)
    temp_fn = os.path.join(tempdirname, os.path.basename(file))
    shutil.copy(file, temp_fn)

    return temp_fn


def open_lib(lib_name):
    """
    The same library can only be loaded once;
    more precisely as long as you try to load a
    library with the same path it gets only loaded **once**
    in the process.

    see https://stackoverflow.com/questions/55312646/loading-two-dynamic-library-instances-in-python for details

    """

    try:
        lib_path = os.environ["LD_LIBRARY_PATH"]
        lib_full_name = None
        for path in lib_path.split(":"):
            if os.path.isfile(os.path.join(path, lib_name)):
                lib_full_name = os.path.join(path, lib_name)
                break
        if not lib_full_name:
            raise OSError

        # lib_temp_name = temp_position(lib_full_name)

        return ctypes.CDLL(lib_full_name)
    except OSError as e:
        msg = f"""Could not find/load shared object file: {lib_name}
     Error was: {e}"""
        raise OSError(msg)
    finally:
        # os.remove(lib_temp_name)
        pass


class MemoryBase:
    using_cmodel = True

    def __init_subclass__(cls) -> None:
        cls.CPU_MEM = {}

    def __init__(self):
        self.CPU_MEM = {}

    def _load_local_mem(self, index):
        pass

    def _set_local_mem(self, mem, index):
        pass

    def set_neuron_size(self, size):
        pass

    def set_coeff_size(self, size):
        pass

    def clear_memory(self):
        raise NotImplementedError()

    def get_data_from_address(self, address: int, size) -> np.ndarray:
        raise NotImplementedError()

    def get_data(self, value_ref: ValueRef) -> np.ndarray:
        raise NotImplementedError()

    def set_data(self, value: MemRefBase, data: np.ndarray):
        raise NotImplementedError()

    def set_data_to_address(self, address: int, data: np.ndarray):
        raise NotImplementedError()

    def set_cpu_data(self, cmd_id: int, data: List[np.ndarray]):

        self.CPU_MEM[cmd_id] = data

    def get_cpu_data(self, cmd_id: int) -> List[np.ndarray]:
        return self.CPU_MEM[cmd_id]

    def clear_cpu_data(self):
        self.CPU_MEM.clear()

    @classmethod
    def get_context(cls):
        pass


class Runner:
    memory: MemoryBase
    DDR: ndarray
    LMEM: ndarray
    SMEM: ndarray
    using_cmodel = True

    def cmds_compute(self, commands: StaticCmdGroup):
        for command in commands.all:
            assert command.cmd_type.is_static()
            # if command.cmd_type
            cmd_type = command.cmd_type
            if cmd_type == CMDType.tiu:
                self.tiu_compute(command)
            elif cmd_type == CMDType.dma:
                self.dma_compute(command)

    def tiu_compute(self, _):
        raise NotImplementedError()

    def dma_compute(self, _):
        raise NotImplementedError()

    @property
    @lru_cache()
    def cpu_processor(self):
        import pyruntime_bm

        cpu_processer = pyruntime_bm.CpuLayer()
        return cpu_processer

    def cpu_compute(self, command: CpuCmd, core_id=0):
        assert all(
            [command.input_memref, command.output_memref]
        ), "currently only support single cpuop for each subnet."

        input_tensors = []
        input_shapes = []
        output_tensors: List[np.ndarray] = []
        output_shapes = []
        for ipt in command.input_memref:
            input_tensors.append(
                self.memory.get_data(ipt.to_ref()).astype(np.float32).flatten().tolist()
            )
            input_shapes.append(ipt.shape)
        for opt in command.output_memref:
            output_tensors.append(np.zeros(opt.shape, dtype=np.float32))
            output_shapes.append(opt.shape)

        # TODO add python type check
        args = (
            command.op_type.value,  # int
            bytes(command.buf),  # bytes param
            len(command.buf),  # int param_size
            input_tensors,  # List[List[float]]
            input_shapes,  # List[List[int]]
            output_tensors,  # List[numpy.ndarray[numpy.float32]]
            output_shapes,  # List[List[int]]
        )

        try:
            new_output_shape = self.cpu_processor.forward(*args)
        except TypeError:
            base_expstr = """The following argument types are supported: \n(self: pyruntime_bm.CpuLayer, op_type: int, param: bytes, param_size: int,
input_tensors: List[List[float]], input_shapes: List[List[int]], output_tensors:
List[numpy.ndarray[numpy.float32]], output_shapes: List[List[int]]) ->
List[List[int]]"""

            failure_types = get_type_str(*args)
            raise TypeError(f"{base_expstr}\n\n but got: {failure_types}")

        for idx, opt in enumerate(command.output_memref):
            opt.shape = new_output_shape[
                idx
            ]  # hack replace new shape of cpu operation, or assert?
            data = np.array(output_tensors[idx], dtype=opt.dtype.np_dtype())
            self.memory.set_data(opt, data)
            self.memory.set_cpu_data(command.cmd_id, data)

    def dynamic_compute(self, command, core_id=0):
        """skip for no implementation target"""
        pass


class CModelRunner(Runner):
    using_cmodel = True

    @property
    def DDR(self):
        return self.memory.DDR

    @property
    def LMEM(self):
        return self.memory.LMEM

    @property
    def SMEM(self):
        return self.memory.SMEM


class CModelMemory(MemoryBase):
    using_cmodel = True

    def __init__(self, LMEM: ndarray, DDR: ndarray, SMEM: ndarray) -> None:
        super().__init__()
        self.LMEM = LMEM.ravel()
        self.DDR = DDR.ravel()
        self.SMEM = SMEM.ravel()


class DeviceRunner(Runner):
    lib_name = "libatomic_exec.so" if platform.machine(
    ) == 'x86_64' else 'libatomic_exec_aarch64.so'
    """
    TODO
    """

    kernel_fn: str
    using_cmodel = False
    runner_p: ctypes.c_void_p

    def __init__(self) -> None:
        super().__init__()

        lib = lib_wrapper(open_lib(self.lib_name))
        self.lib = lib
        self.init_runner()

    def init_runner(self):
        pass

    @property
    def DDR(self):
        return self.memory.DDR

    @property
    def LMEM(self):
        return self.memory.LMEM

    @property
    def SMEM(self):
        return self.memory.SMEM

    def tiu_compute(self, command):
        raise NotImplementedError()

    def dma_compute(self, command):
        raise NotImplementedError()


class DeviceMemory(MemoryBase):
    using_cmodel = False

    def __init__(self, lib) -> None:
        super().__init__()
