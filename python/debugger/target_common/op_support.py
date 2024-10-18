# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import struct
from typing import Union, List, Dict, NamedTuple, Callable, Any, Tuple, Type
from enum import Enum, IntEnum
import functools
import ctypes
import numpy as np
from dataclasses import dataclass

__all__ = [
    "MType",
    "DType",
    "Scalar",
    "Engine",
]

def div_up(x: int, y: int) -> int:
    return (x + y - 1) // y if y else 0

def align_up(x: int, y: int) -> int:
    return div_up(x, y) * y

# ./tpu-cpuop/include/bmcpu_common.h
class CpuLayerType(Enum):
    CPU_SSD_DETECTION_OUTPUT = 0  # CAFFE
    CPU_ANAKIN_DETECT_OUTPUT = 1  # ANAKIN
    CPU_RPN = 2
    CPU_USER_DEFINED = 3  # USER DEFINED LAYER
    CPU_ROI_POOLING = 4  # ROI Pooling Layer
    CPU_ROIALIGN = 5  # from MXNet
    CPU_BOXNMS = 6  # from MXNet
    CPU_YOLO = 7  # YOLO LAYER
    CPU_CROP_AND_RESIZE = 8  # CROP AND RESIZE LAYER
    CPU_GATHER = 9  # GATHER LAYER
    CPU_NON_MAX_SUPPRESSION = 10  # NON MAX SUPPRESSION LAYER
    CPU_ARGSORT = 11  # ARGSORT FROM MXNET
    CPU_GATHERND = 12  # GATHER_ND LAYER
    CPU_YOLOV3_DETECTION_OUTPUT = 13  # YOLO V3 DETECT OUT
    CPU_WHERE = 14  # WHERE LAYER
    CPU_ADAPTIVE_AVERAGE_POOL = 15  # ADAPTIVE AVERAGE POOLING
    CPU_ADAPTIVE_MAX_POOL = 16  # ADAPTIVE MAX POOLING
    CPU_TOPK = 17  # TOPK
    CPU_RESIZE_INTERPOLATION = 18  # CPU RESIZE INTERPOLATION
    CPU_GATHERND_TF = 19  # CPU GATHER_ND TENSORFLOW LAYER
    CPU_SORT_PER_DIM = 20  # CPU SORT_PER_DIM LAYER
    CPU_WHERE_SQUEEZE_GATHER = 21  # CPU WHERE_SQUEEZE_GATHER LAYER
    CPU_MASKED_SELECT = 22  # CPU MASKED_SELECT LAYER
    CPU_UNARY = 23  # CPU UNARY LAYER
    CPU_EMBEDDING = 24  # CPU EMBEDDING
    CPU_TOPK_MX = 25  # TOPK from MXNET
    CPU_INDEX_PUT = 26  # CPU INDEX PUT
    CPU_SCATTER_ND = 27  # CPU SCATTER ND
    CPU_RANDOM_UNIFORM = 28  # CPU RANDOM UNIFORM
    CPU_GATHER_PT = 29  # CPU GATHER FOR PYTORCH
    CPU_BINARY = 30  # CPU BINARY: MOD, DIV, ...
    CPU_TENSORFLOW_NMS_V5 = 31  # CPU TENSORFLOW NMS V5
    CPU_GENERATE_PROPOSALS = 32  # CPU GENERATE PROPOSALS
    CPU_BBOX_TRANSFORM = 33  # CPU BBOX TRANSFORM
    CPU_BOX_WITH_NMS_LIMIT = 34  # CPU BOX WITH NMS LIMIT
    CPU_COLLECT_RPN_PROPOSALS = 35  # CPU COLLECT RPN PROPOSALS
    CPU_DISTRIBUTE_FPN_PROPOSALS = 36  # CPU DISTRIBUTE FPN PROPOSALS
    CPU_DISTRIBUTE_FPN_PROPOSALS_ROI_ALIGN_CONCAT = 37
    CPU_PYTORCH_ROI_ALIGN = 38  # CPU PYTORCH ROI ALIGN
    CPU_AFFINE_GRID_GENERATOR = 39  # CPU AFFINE GRID GENERATOR
    CPU_GRID_SAMPLER = 40  # CPU GRID SAMPLER
    CPU_AFFINE_GRID_SAMPLER = 41  # CPU AFFINE GRID SAMPLER
    CPU_RANDOM_UNIFORM_INT = 42  # CPU RANDOM UNIFORM INT
    CPU_TOPK_ASCENDING = 43  # CPU TOPK BY ASCENDING ORDER
    CPU_PYTORCH_INDEX = 44  # CPU PYTORCH INDEX
    CPU_EMBEDDING_BAG = 45  # CPU EMBEDDINGBAG

    # // the following layers have not been tested on windows
    # ifdef __linux__
    CPU_ONNX_NMS = 46  # CPU ONNX NMS
    CPU_DEFORM_GATHER = 47  # CPU DEFORM GATHER
    CPU_DEFORM_PSROIPOOLING = 48  # CPU DEFORM PSROIPOOLING
    CPU_PADDLE_YOLO_BOX = 49  # CPU PADDLE YOLO BOX
    CPU_PADDLE_MULTICLASS_NMS = 50  # CPU PADDLE MULTICLASS NMS
    CPU_PADDLE_DEFORM_CONV = 51  # CPU PADDLE DEFORMABLE CONV
    CPU_PADDLE_MATRIX_NMS = 52  # CPU PADDLE MATRIX NMS
    CPU_REVERSE_SEQUENCE = 53  # CPU REVERSE SEQUENCE
    CPU_FULL_INDEX = 54  # from pytorch tensor::index
    CPU_ADAPTIVE_AVERAGE_POOL_3D = 55  # ADAPTIVE AVERAGE 3D POOLING
    CPU_TENSOR_SCATTER_OP = 56  # tensorflow TENSOR SCATTER UPDATE,ADD,MAX,MIN,SUB
    # endif
    CPU_REPEAT_INTERLEAVE = 57  # torch.repeat_interleave when repeat is a tensor
    CPU_PADDLE_DENSITY_PRIOR_BOX = 58  # CPU PADDLE PRIOR BOX
    CPU_PADDLE_BOX_CODER = 59

    CPU_LAYER_UNKNOW = 60


class Target(Enum):
    BM1684X = "BM1684X"
    BM1684 = "BM1684"
    BM1688 = "BM1688"
    BM1690 = "BM1690"
    SG2380 = "SG2380"
    MARS3 = "MARS3"


class Engine(Enum):
    TIU = "TIU"
    DMA = "DMA"
    HAU = "HAU"


def get_continuous_stride(shape):
    return np.cumprod([1] + list(shape[-1:0:-1]), dtype=int)[::-1]


def DIV_UP(x, a):
    return (x + a - 1) // a


def ALIGN(x, a):
    return DIV_UP(x, a) * a


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
    f8 = 7
    # unsign integer
    ui8 = i8 + 8  # type: ignore
    ui16 = i16 + 8  # type: ignore
    ui32 = i32 + 8  # type: ignore
    ui4 = i4 + 8  # type: ignore
    f8e5m2 = f8 + 8  # type: ignore
    # sign integer
    si8 = i8 + 16  # type: ignore
    si16 = i16 + 16  # type: ignore
    si32 = i32 + 16  # type: ignore
    si4 = i4 + 16  # type: ignore
    f8e4m3 = f8 + 16  # type: ignore

    def is_float(self):
        return self in (DType.f32, DType.f16, DType.bf16, DType.f8e4m3, DType.f8e5m2, DType.f8)

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

    def np_dtype(self):
        return to_np_dtype[self]


class ExtEnum:
    """
    Add additional information to Enumerate.
    Enumerate is a constant type and a singleton.
    """

    def __init__(self, enum: Enum, *args, **kargs) -> None:
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


def get_dtype(prec, sign=1):  # unsigned -> 0; sign -> 1
    if prec in (DType.f32, DType.bf16, DType.f16):
        return DType(prec)
    return DType(prec + 8 + (sign == 1) * 8)


def bf16_to_fp32(d_bf16):
    assert d_bf16.dtype == np.uint16
    s = d_bf16.shape
    d_bf16 = d_bf16.ravel()
    d_fp32 = np.zeros_like(d_bf16, dtype=np.float32)
    v_ui16 = d_fp32.view(np.uint16)
    v_ui16[1::2] = d_bf16
    return d_fp32.reshape(s)


def fp8e5m2_to_fp16(d_fp8):
    assert d_fp8.dtype == np.uint8
    s = d_fp8.shape
    d_fp8 = d_fp8.ravel()
    d_fp16 = np.zeros_like(d_fp8, dtype=np.float16)
    v_ui8 = d_fp16.view(np.uint8)
    v_ui8[1::2] = d_fp8
    return d_fp16.reshape(s)


def fp8e4m3_to_fp16(d_fp8):
    assert d_fp8.dtype == np.uint8
    s = d_fp8.shape
    d_fp8 = d_fp8.ravel().astype(np.uint16)
    d_fp16 = np.zeros_like(d_fp8, dtype=np.float16)

    # refer to fp8_to_fp16() in /nntoolchain/TPU1686/common/include/cast.h
    ur = d_fp8 << 8
    sign = ur & 0x8000
    exponent = ((ur & 0x7800) >> 1) + 0x2000
    mantissa = (ur & 0x0700) >> 1
    absx = 0x7F & d_fp8

    nan_mask = absx == 0x7F
    ur[nan_mask] = 0x7FFF | sign[nan_mask]

    exp_mask = ~nan_mask & (exponent == 0x2000)
    mantissa_mask = exp_mask & (mantissa != 0)
    mantissa[exp_mask] = np.where(mantissa[exp_mask] != 0, mantissa[exp_mask] << 1, mantissa[exp_mask])
    exponent[exp_mask] = np.where(mantissa[exp_mask] != 0, exponent[exp_mask], 0)

    while np.any((mantissa[mantissa_mask] & 0x0400) == 0):
        mantissa[mantissa_mask] = np.where((mantissa[mantissa_mask] & 0x0400) == 0,mantissa[mantissa_mask] << 1,mantissa[mantissa_mask])
        exponent[mantissa_mask] = np.where((mantissa[mantissa_mask] & 0x0400) == 0,exponent[mantissa_mask] - 0x0400,exponent[mantissa_mask])
    mantissa[mantissa_mask] &= 0x03FF

    ur[~nan_mask] = (sign[~nan_mask] | exponent[~nan_mask]) | mantissa[~nan_mask]
    d_fp16 = np.frombuffer(ur.astype(np.uint16).tobytes(), dtype=np.float16)
    return d_fp16.reshape(s)


to_np_dtype = {
    DType.si8: np.int8,
    DType.ui8: np.uint8,
    DType.f8: np.uint8,
    DType.f8e5m2: np.uint8,
    DType.f8e4m3: np.uint8,
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
    alignIC = 20
    # Gather/Scatter alignment
    alignLine = 30
    T4 = 31 # TODO: give it a better name
    T5 = 32 # TODO: give it a better name
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


class Value:
    is_scalar = None

    def to_ref(self, **kwargs):
        return ValueRef(self, **kwargs)


class MemRefBase(Value):
    """
    A description of tensor in memory.
    """

    device: Target = None
    is_scalar = False

    def __init__(self, address, shape, dtype: DType, stride=None, layout=None):
        self.address = address
        self.mtype = self.get_mtype(address)  # memory type with extended enumerate
        self.shape = [int(i) for i in shape]
        self.dtype: DType = dtype
        self.layout = layout
        self.np_dtype = to_np_dtype[dtype]
        self.itemsize = self.np_dtype().itemsize
        self.stride = stride

    def __init_subclass__(cls) -> None:
        assert cls.device is not None, cls
        return super().__init_subclass__()

    @property
    @functools.lru_cache()
    def r_addr(self):
        if self.mtype == MType.UNKNOWN:
            return self.address

        from .context import get_target_context_cls

        context = get_target_context_cls(self.device.name)
        r_addr = self.address - context.memmap[self.mtype][0]
        return r_addr

    def get_mtype(self, address) -> MType:
        from .context import get_target_context_cls

        context = get_target_context_cls(self.device.name)
        return context.get_memory_type(address)

    @property
    def npu_offset(self):
        raise NotImplementedError()

    @property
    def bank_index(self):
        raise NotImplementedError()

    @property
    def bank_offset(self):
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
            mem_str = f"%{k.name}{self.bank_index}"
            if self.bank_offset:
                mem_str += f".{self.bank_offset}"
            if self.npu_offset:
                mem_str += f".L{self.npu_offset}"
            return mem_str
        return f"%{k.name}{self.r_addr}"

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


class Scalar(Value):
    is_scalar = True

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


ValueType = Union[Scalar, MemRefBase]


class ValueRef:

    def __init__(self, value: ValueType, **kwargs) -> None:
        self.value = value
        self.kwargs = kwargs

    def get(self, k, default=None):
        return self.kwargs.get(k, default)


class OpInfo(NamedTuple):
    op_name: str
    eu_name: str


class atomic_reg(ctypes.Structure):
    OP_NAME: str
    length: int

    def __repr__(self):
        return str(dict(self))

    def __iter__(self):
        for field in self._fields_:
            yield (field[0], getattr(self, field[0]))

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    @classmethod
    def from_values(cls, values: List[int]) -> "atomic_reg":
        res = cls()
        assert len(values) == len(cls._fields_), f"{len(values)} != {len(cls._fields_)}"
        for (key, *_), val in zip(cls._fields_, values):
            setattr(res, key, val)
        return res


ParamConvertFnType = Callable[
    [atomic_reg], Tuple[List[ValueType], Dict[str, Any], List[ValueType]]
]


class CMDType(Enum):
    tiu = 0
    dma = 1
    cpu = 2
    dyn_ir = 8
    unknown = 9

    def is_static(self):
        return self == CMDType.tiu or self == CMDType.dma


class FileLineMixin:
    def pre_ln(self):
        raise NotImplementedError()

    def post_ln(self):
        raise NotImplementedError()

    def mid_ln(self):
        raise NotImplementedError()


class BaseCmd(FileLineMixin):
    buf: bytes
    name: str
    attribute: Dict
    operands: List[ValueType]
    results: List[ValueType]
    subnet_id: int
    cmd_id: int

    def __new__(cls, *args, **kwargs):
        self = object.__new__(cls)
        return self

    @property
    def cmd_type(self) -> CMDType:
        if isinstance(self, Tiu):
            return CMDType.tiu
        elif isinstance(self, Dma):
            return CMDType.dma
        elif isinstance(self, CpuCmd):
            return CMDType.cpu
        elif isinstance(self, DynIrCmd):
            return CMDType.dyn_ir
        else:
            return CMDType.unknown

    def pre_ln(self):
        return 0

    def post_ln(self):
        return 0

    def mid_ln(self):
        return 1


class BaseTpuCmd(BaseCmd):
    reg: atomic_reg
    cmd_id: int
    cmd_id_dep: int
    opparam_converter: Dict[str, ParamConvertFnType] = {}

    def __init__(
        self,
        reg: atomic_reg,
        *,
        buf: memoryview,
        subnet_id=0,
        core_id=0,
        param_fn=None,
        cmd_id=None,
    ) -> None:
        self.reg = reg
        self.buf = buf
        assert len(buf) <= 128
        self.subnet_id = subnet_id
        self.core_id = core_id

        if param_fn is None and self.opparam_converter is not None:
            param_fn = self.opparam_converter.get(reg.OP_NAME, None)

        if param_fn is None:
            self.results, self.attribute, self.operands = [], {}, []
        else:
            self.results, self.attribute, self.operands = param_fn(reg)

    def __repr__(self) -> str:
        return "abc.none"

    def __getitem__(self, k):
        return self.attribute[k]

    @property
    def tuple_key(self):
        if isinstance(self, Tiu):
            key = (self.subnet_id, self.cmd_id, None, self.core_id)
        elif isinstance(self, Dma):
            key = (self.subnet_id, None, self.cmd_id, self.core_id)
        else:
            raise NotImplementedError()
        return key


class StaticCmdGroup(NamedTuple):
    tiu: List[BaseTpuCmd]
    dma: List[BaseTpuCmd]
    all: List[BaseTpuCmd]


class Tiu:
    # Mixin for detect cmd type
    pass


class Dma:
    # Mixin for detect cmd type
    pass


class RegIndex:
    def __init__(self):
        self.storage = {}

    def __setitem__(self, keys, value):
        # flatten eu_types
        from collections.abc import Iterable
        from itertools import product

        _tuple = lambda x: x if isinstance(x, Iterable) else (x,)
        keys_itr = (_tuple(x) for x in keys)
        for key in product(*keys_itr):
            self.storage[key] = value

    def __getitem__(self, key):
        if key in self.storage:
            return self.storage[key]
        raise KeyError(f"can not find {key}, This object only has {self}")

    def __repr__(self):
        return str(self.storage)

    def get(self, key, default) -> Type[BaseTpuCmd]:
        if key in self.storage:
            return self.storage[key]
        return default


@dataclass
class CpuCmd(BaseCmd):
    op_type: CpuLayerType
    buf: bytes
    buf_size: int
    input_memref: List[MemRefBase]
    output_memref: List[MemRefBase]
    cmd_id: int = 0  # assigned in python interface
    subnet_id: int = 0
    core_id = 0

    @property
    def cmd(self):
        return self

    @property
    def name(self):
        return self.op_type.name

    def __repr__(self) -> str:
        return f"{self.op_type.name}() ({self.input_memref}) -> {self.output_memref}"

    @property
    def operands(self):
        return [i for i in self.input_memref]

    @property
    def results(self):
        return [i for i in self.input_memref]

    @property
    def attribute(self):
        return {}

    @property
    def tuple_key(self):
        return (self.cmd.subnet_id, self.cmd.cmd_id, "cpu", 0)


@dataclass
class DynIrCmd(BaseCmd):
    ir_buffer: bytes
    ir_size: int

    input_memref: List[MemRefBase]
    output_memref: List[MemRefBase]
    cmd_id: int = 0  # assigned in python interface
    subnet_id: int = 0

    ctx_addr: int = 0
    ctx_size: int = 0
    core_id = 0

    @property
    def cmd(self):
        return self

    @property
    def buf(self):
        return self.ir_buffer

    @property
    def name(self):
        return f"DYN{self.subnet_id}_FULLNET"

    def __repr__(self) -> str:
        return f"{self.name}() ({self.input_memref}) -> {self.output_memref}"

    @property
    def operands(self):
        return [i for i in self.input_memref]

    @property
    def results(self):
        return [i for i in self.input_memref]

    @property
    def attribute(self):
        return {}

    @property
    def tuple_key(self):
        return (self.cmd.subnet_id, self.cmd.cmd_id, "dynir", 0)


def get_type_str(*args) -> str:
    types = []
    for i in args:
        if isinstance(i, (int, float, bytes, bytearray)):
            type_str = str(i.__class__.__name__)
        elif isinstance(i, np.ndarray):
            type_str = f"numpy.ndarray[{i.dtype}]"
        elif isinstance(i, list):
            inner = ", ".join(set([get_type_str(ii) for ii in i]))
            type_str = f"List[{inner}]"
        else:
            type_str = str(type(i))
        types.append(type_str)
    outer = ", ".join(types)
    return f"{outer}"
