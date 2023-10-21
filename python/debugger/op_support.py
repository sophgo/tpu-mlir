# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from enum import Enum, IntEnum
import functools, ctypes
import numpy as np
from collections import OrderedDict

__all__ = [
    "InsBase",
    "TIUBase",
    "DMABase",
    "NamedDict",
    "MType",
    "DType",
    "Scalar",
    "Engine",
]

# ------------------------------------------------------------
# utility function


class Engine(Enum):
    TIU = "TIU"
    DMA = "DMA"
    HAU = "HAU"


def reg_decoder_factory(fileds):
    key, high_bits = zip(*fileds)
    # check reg define is 64bits align
    assert all(64 * x in high_bits for x in range(1, high_bits[-1] // 64 + 1))
    bits_width = np.diff(high_bits, prepend=0)

    class REG(ctypes.Structure):

        _fields_ = [(k, ctypes.c_uint64, v) for k, v in zip(key, bits_width)]

        def __repr__(self):
            return str(dict(self))

        def __iter__(self):
            for field in self._fields_:
                yield (field[0], getattr(self, field[0]))

        def __getitem__(self, key):
            return getattr(self, key)

        def __setitem__(self, key, value):
            setattr(self, key, value)

    return REG


def extract_buf(buffer, range):
    l, h = range
    bit_len = h - l
    byte_begin = l // 8
    byte_end = (l + bit_len + 7) // 8
    bit_offset = l % 8
    byte_slice = buffer[byte_begin:byte_end]
    total_val = 0
    for v in byte_slice[::-1]:
        total_val = total_val << 8 | v
    total_val = total_val >> bit_offset
    total_val = total_val & ((1 << bit_len) - 1)
    return total_val


def get_continuous_stride(shape):
    return np.cumprod([1] + list(shape[-1:0:-1]), dtype=int)[::-1]


def DIV_UP(x, a):
    return (x + a - 1) // a


def ALIGN(x, a):
    return DIV_UP(x, a) * a


# ------------------------------------------------------------
class InsBase:
    # shared information
    opcode = None
    length = 0  # the bits length of this instruction
    description = "This is a base op."
    opcode_bits = (0, 0)  # [opcode position)
    # register description
    reg_def = None
    # object information
    __slots__ = (
        "_cache",  # lazy compute: results, attribute, operands,
        "op_name",
        # raw reg info
        "cmd",
        "reg",
        "cmd_id",
        "cmd_id_dep",
        "core_id"
    )

    def _is_comp(self, cmd_buf):
        raise NotImplementedError(self.__class__)

    def _decode(self):
        raise NotImplementedError(self.__class__)

    def _set_op(self, reg):
        raise NotImplementedError(self.__class__)

    def __set_cache(self):
        self._cache = {
            k: v
            for k, v in zip(
                ("results", "attribute", "operands"), self._set_op(self.reg)
            )
        }

    @classmethod
    def is_comp(cls, cmd_buf):
        return cls._is_comp(cls, cmd_buf)

    @classmethod
    def decode(cls, cmd_buf):
        cls = cls()
        bytes_length = cls.length // 8
        cls.cmd = bytes(cmd_buf[:bytes_length])
        cls._cache = {}
        cls._decode()
        return cls

    @property
    def operands(self):
        if "operands" in self._cache:
            return self._cache["operands"]
        self.__set_cache()
        return self._cache["operands"]

    @property
    def attribute(self):
        if "attribute" in self._cache:
            return self._cache["attribute"]
        self.__set_cache()
        return self._cache["attribute"]

    @property
    def results(self):
        if "results" in self._cache:
            return self._cache["results"]
        self.__set_cache()
        return self._cache["results"]

    def __eq__(self, other):
        if not isinstance(other, InsBase):
            return False

        if len(self.cmd) != len(other.cmd):
            return False
        return (self.cmd == other.cmd).all()

    def __hash__(self):
        return hash(str(self.cmd))

    def __repr__(self):
        return self.description


class TIUBase(InsBase):
    pass


class DMABase(InsBase):
    pass


# ------------------------------------------------------------
class DType(IntEnum):
    """
    The numeric type of the data.
    """

    # Signless
    # Only the bits width is correct.
    i8 = 0
    f16 = 1
    f32 = 2
    i16 = 3
    i32 = 4
    bf16 = 5
    i4 = 6
    # unsign integer
    ui8 = i8 + 8  # type: ignore
    ui16 = i16 + 8  # type: ignore
    ui32 = i32 + 8  # type: ignore
    ui4 = i4 + 8  # type: ignore
    # sign integer
    si8 = i8 + 16  # type: ignore
    si16 = i16 + 16  # type: ignore
    si32 = i32 + 16  # type: ignore
    si4 = i4 + 16  # type: ignore

    def is_float(self):
        return self in (DType.f32, DType.f16, DType.bf16)

    def is_int(self):
        return not self.is_float()

    def __hash__(self):
        # This is an enumeration type, not an integer type, even though it behaves
        # like an integer. It's beneficial to distinguish between this type and
        # integers when using them together as keys in a dictionary.
        return id(self)

    @property
    def itemsize(self):
        if self in (DType.i4, DType.ui4, DType.si4):
            return 0.5
        return to_np_dtype[self]().itemsize


class ExtEnum:
    """
    Add additional information to Enumerate.
    Enumerate is a constant type and a singleton.
    """

    def __init__(self, enum, *args, **kargs) -> None:
        assert isinstance(enum, Enum)
        self.enum = enum
        self.args = args
        self.__dict__.update(kargs)
        self._member_ = kargs.keys()

    @property
    def name(self):
        return self.enum.name

    @property
    def value(self):
        return self.enum.value

    def __hash__(self):
        return hash(self.enum)

    def __eq__(self, other):
        return self.enum == other

    def __repr__(self) -> str:
        kargs = {k: self.__dict__[k] for k in self._member_}
        return repr(self.enum) + f"{self.args}{kargs}"


class MType(Enum):
    """
    The type of memory.
    """

    R = 0  # local memory
    S = 1  # static memory
    L = 2  # L2 SRAM
    G = 3  # DDR
    UNKNOWN = 7

    def __call__(self, *args, **kargs):
        return ExtEnum(self, *args, **kargs)


def get_dtype(prec, sign=1):  # unsigned -> 0; sign -> 1
    if prec in (DType.f32, DType.bf16, DType.f16):
        return DType(prec)
    return DType(prec + 8 + (sign == 1) * 8)


def bf16_to_fp32(d_bf16):
    assert d_bf16.dtype == np.uint16
    s = d_bf16.shape
    d_bf16 = d_bf16.ravel()
    d_fp32 = np.empty_like(d_bf16, dtype=np.float32)
    v_ui16 = d_fp32.view(np.uint16)
    v_ui16[1::2] = d_bf16
    return d_fp32.reshape(s)


to_np_dtype = {
    DType.si8: np.int8,
    DType.ui8: np.uint8,
    DType.f16: np.float16,
    DType.f32: np.float32,
    DType.si16: np.int16,
    DType.ui16: np.uint16,
    DType.si32: np.int32,
    DType.ui32: np.uint32,
    DType.i8: np.uint8,
    DType.i16: np.uint16,
    DType.i32: np.uint32,
    DType.bf16: np.uint16,
    DType.i4: np.uint8,
    DType.ui4: np.uint8,
    DType.si4: np.uint8,
}


class Layout(Enum):
    """
    Data layout type in Local memory.
    """

    # BM168X
    # Tensor alignment
    alignEU = 0
    compact = 1
    offset = 2
    stride = 3
    # Matrix alignment
    matrix = 10
    matrix2 = 11
    # Weight alignment
    _64IC = 20
    _32IC = 21
    _1IC = 22
    _16IC = 23
    # special alignment. TODO: give it a better name
    T3 = 30
    T4 = 31
    T5 = 32
    # GDMA special layout
    DMAstride = 40  # contains lane mask
    DMA4Bank = 41
    DMAmatrix = 42
    DMAlinear = 43

    # BM1684
    alignEU_XN = 50
    compact_XN = 51
    continuous_XN = 60

    def __call__(self, *args, **kargs):
        return ExtEnum(self, *args, **kargs)


class MemRef:
    """
    A description of tensor in memory.
    """

    def __init__(self, address, shape, dtype: DType, stride=None, layout=None):
        self.address = address
        self.mtype = self.get_mtype(address)  # memory type with extended enumerate
        self.shape = shape
        self.dtype = dtype
        self.layout = layout
        self.np_dtype = to_np_dtype[dtype]
        self.itemsize = self.np_dtype().itemsize
        self.stride = stride

    def get_mtype(self, address):
        raise NotImplementedError()

    @property
    @functools.lru_cache()
    def name(self):
        """
        use relative address
        """
        k = self.mtype
        if k == MType.UNKNOWN:
            return f"%?.{self.address}"
        if k == MType.R:
            # R{bank_index}.{bank_offset}.L{NPU_OFFSET}
            # R0.123.L0
            mem_str = f"%{k.name}{k.bank_index}"
            if k.bank_offset:
                mem_str += f".{k.bank_offset}"
            if k.npu_offset:
                mem_str += f".L{k.npu_offset}"
            return mem_str
        return f"%{k.name}{k.r_addr}"

    @property
    @functools.lru_cache()
    def type_str(self):
        s = [str(x) for x in self.shape]
        if self.stride is not None and any((x != 0 for x in self.stride)):
            stride = tuple(self.stride)
            return f"memref<{'x'.join(s)}x{self.dtype.name}, strides: [{str(stride)[1:-1]}]>"
        return f"memref<{'x'.join(s)}x{self.dtype.name}>"

    def __repr__(self):
        return f"{self.name}: {self.type_str}"


class Scalar:
    def __init__(self, value, dtype: DType):
        data = np.uint32([value])
        self.dtype = dtype
        if dtype != DType.bf16:
            np_dtype = to_np_dtype[dtype]
            self.data = data.view(np_dtype)[0]
        else:
            self.data = data.view(np.float32)[0]
        self.name = f"%C{self.data}"
        self.type_str = f"{self.dtype.name}"

    def __repr__(self):
        return f"{self.name}: {self.type_str}"


class NamedDict(OrderedDict):
    def __init__(self, _dict):
        super().__init__()
        super().update(_dict)

    def __getattr__(self, key):
        return super().__getitem__(key)

    def __setattr__(self, key, value):
        super().__setitem__(key, value)
