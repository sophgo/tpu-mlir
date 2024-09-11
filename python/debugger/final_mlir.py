# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
"""
decode final.mlir and tensor_loc.json
"""
import re
from functools import lru_cache
from pprint import pformat
from typing import List, Tuple, Dict
import json
import numpy as np
from .target_common import DType, CMDType, Layout, BModelContext, MType

to_dtype: Dict[str, DType] = {
    "f32": DType.f32,
    "f16": DType.f16,
    "bf16": DType.bf16,
    "i8": DType.i8,
    "si8": DType.si8,
    "ui8": DType.ui8,
    "u8": DType.ui8,
    "f8": DType.f8,
    "f8e5m2": DType.f8e5m2,
    "f8e4m3": DType.f8e4m3,
    "i16": DType.i16,
    "si16": DType.si16,
    "ui16": DType.ui16,
    "u16": DType.ui16,
    "i32": DType.i32,
    "si32": DType.si32,
    "ui32": DType.ui32,
    "u32": DType.ui32,
    "i4": DType.i4,
    "si4": DType.si4,
    "ui4": DType.ui4,
}


def parse_zp_scale(type_str: str):
    from .lazy_mlir_ir import parse_mlir_type
    from .lazy_mlir_ir import quant

    mlir_type = parse_mlir_type(type_str)

    zero_point = 0
    if isinstance(mlir_type, quant.UniformQuantizedType):
        zero_point = mlir_type.zero_point

    scale = 1
    if (
        isinstance(mlir_type, quant.CalibratedQuantizedType)
        and "f8E4M3" in type_str
    ):
        scale = mlir_type.max / 448

    if isinstance(mlir_type, quant.UniformQuantizedType):
        scale = mlir_type.scale
    return (zero_point, scale)


class TLValue:
    address: int
    layout: str
    name: str
    reshape: str
    slice: str
    dtype: np.dtype
    zero_point: float
    scale: float
    memory_type: str
    mlir_type_str: str
    raw_dict: dict

    @classmethod
    def from_tldic(cls, dic):
        self = cls()
        self.address = dic["address"]
        self.layout = dic["layout"]
        self.memory_type = dic["memory_type"]
        self.name = dic["name"]
        self.reshape = dic["reshape"]
        self.slice = dic["slice"]
        self.mlir_type_str = dic["type"]
        self.dtype = to_dtype[self.memory_type.strip(">").split("x")[-1]].np_dtype()
        self.zero_point, self.scale = parse_zp_scale(self.mlir_type_str)
        self.raw_dict = dic
        return self

    @staticmethod
    def get_shape_and_dtype(input_string: str):
        pattern = r"<(\d+(?:x\d+)*)x(\S+)>"
        matches = re.findall(pattern, input_string)
        shape = []
        dtype = []
        for match in matches:
            shape.extend(map(int, match[0].split("x")))
            dtype.append(match[1])

        dtype = to_dtype[dtype[0]]
        return shape, dtype

    def get_memref(self, context: BModelContext):
        address = self.address
        shape, dtype = self.get_shape_and_dtype(self.memory_type)
        layout = self.layout
        layout = {
            "eu_align": Layout.alignEU,
            "eu_align_group3d": Layout.alignEU,
            "eu_align_xn": Layout.alignEU_XN,
            "eu_align_xn_group3d": Layout.alignEU_XN,
            "compact": Layout.compact,
            "compact_group3d": Layout.compact,
            "compact_xn": Layout.compact_XN,
            "compact_xn_group3d": Layout.compact_XN,
            "continuous": None,
            "continuous_xn": None,  # 4N/2N
            "continuous_group3d": None,
            # "continuous_xn_group3d": None,
        }[layout]

        # local memory
        lmem_start_addr = context.memmap[MType.R][0]
        if address < lmem_start_addr:
            address += lmem_start_addr
        stride = None

        # global memory
        if layout is None:
            # lib/Dialect/Tpu/Interfaces/BM1684X/Load.cpp
            # load/store case
            reshape = self.reshape
            if reshape:
                _, c, d, h, w = [int(x) for x in self.reshape[1:-1].split("x")]
                _layout = self.layout
                if _layout in ("continuous", "continuous_xn"):
                    stride = (c * d * h * w, h * w, w, 1)
                elif _layout == "continuous_group3d":
                    stride = (h * w, d * h * w, w, 1)
                else:
                    raise ValueError(f"Not supported layout: {_layout}")
            else:
                # global layer
                stride = context.get_continuous_stride(shape)

            if self.layout == "continuous_xn":  # fix 2N/4N
                stride = np.int64(stride) * 4 // dtype.itemsize
                layout = Layout.continuous_XN
        return context.MemRef(address, shape, dtype, layout=layout, stride=stride)

    def __repr__(self) -> str:
        return f"@{self.address}({{name={self.name}, layout={self.layout}, slice={self.slice}, mlir_type={self.mlir_type_str}, memory_type={self.memory_type}}})"


class Pickled_Value:
    def __init__(
        self, value: TLValue, Memref, Type, Zero_point, Scale, file_line, cmd_point
    ):
        self.address: int = value.address
        self.layout: str = value.layout
        self.memory_type: str = value.memory_type
        self.name: str = value.name
        self.reshape: str = value.reshape
        self.slice: str = value.slice
        self._type: Type = value.dtype
        self.dtype = to_dtype[self.memory_type.strip(">").split("x")[-1]].np_dtype()
        self.file_line = file_line
        self.cmd_point = cmd_point

        self.memref = Memref
        self.mlir_type = str(Type)
        self.zero_point = Zero_point
        self.scale = Scale

    def __repr__(self) -> str:
        return f"@{self.address}({{name={self.name}, cmd_point={self.cmd_point}, file_line={self.file_line}, layout={self.layout}, slice={self.slice}, mlir_type={self.mlir_type}, memory_type={self.memory_type}}})"


class CMD:
    loc_index: int
    file_line: int
    subnet_id: int
    opcode: str
    core_id: int
    tiu_dma_id_before: Tuple[int, int]
    tiu_dma_id_after: Tuple[int, int]
    operands: List[TLValue]
    results: List[TLValue]
    slice_all: bool

    @classmethod
    def from_tldic(cls, dic: dict, index: int):
        self = cls()

        self.loc_index = index
        self.file_line = dic["file-line"]
        self.subnet_id = dic["subnet_id"]
        self.opcode = dic["opcode"]
        self.core_id = dic.get("core_id", 0)
        # dma_id before is not aligned with asm cmd
        # For example, yolov5 cmd[2] use B0 as cmd_id_dep
        # but tensor_loc.json has tiu_dma_id(before) = (1, 1)
        self.tiu_dma_id_before = dic["tiu_dma_id(before)"]
        self.tiu_dma_id_after = dic["tiu_dma_id(after)"]
        # None for top.None
        self.operands = [TLValue.from_tldic(i) if len(i) > 0 else None for i in dic["operands"]]
        # None for no usage results
        self.results = [TLValue.from_tldic(i) if len(i) > 0 else None for i in dic["results"]]

        operands_slice_all = results_slice_all = False
        operands_slice_all = all(
            operand.slice == "[...]" for operand in self.operands if self.operands if operand
        )
        results_slice_all = all(
            result.slice == "[...]" for result in self.results if self.results if result
        )
        self.slice_all = operands_slice_all and results_slice_all
        return self

    @property
    def cmd_type(self) -> CMDType:
        if self.opcode in {
            "tpu.Store",
            "tpu.Load",
        }:
            return CMDType.dma
        else:
            return CMDType.tiu

    @property
    def tuple_key_before(self):
        return (self.subnet_id, *self.tiu_dma_id_before, self.core_id)

    @property
    def tuple_key_after(self):
        return (self.subnet_id, *self.tiu_dma_id_after, self.core_id)

    def __repr__(self) -> str:
        return pformat(self.__dict__)

    __str__ = __repr__


class TensorLoc:
    tensor_loc: List[CMD]

    @classmethod
    def from_tl_file(cls, tensor_loc_file: str):
        self = cls()
        with open(tensor_loc_file) as r:
            self.tensor_loc = [
                CMD.from_tldic(dic, index) for index, dic in enumerate(json.load(r))
            ]
        return self

    def __repr__(self) -> str:
        return pformat(self.__dict__)

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            return self.tensor_loc[index]
        raise KeyError(index)

    def __len__(self):
        return len(self.tensor_loc)


def iter_operations(op):
    from lazy_mlir_ir import Module, OpView, FuncOp, Region, Block

    if isinstance(op, Module):
        for oop in op.body.operations:
            yield from iter_operations(oop)
    elif isinstance(op, OpView):
        if op.operation.name == "builtin.module":
            for region in op.regions:
                yield from iter_operations(region)
        elif isinstance(op, FuncOp):
            for region in op.regions:
                yield from iter_operations(region)
        else:
            raise NotImplementedError(op)
    elif isinstance(op, Region):
        for block in op.blocks:
            yield from iter_operations(block)
    elif isinstance(op, Block):
        for operation in op.operations:
            if isinstance(operation, FuncOp):
                yield from iter_operations(operation)
            else:
                yield operation.operation
    else:
        raise NotImplementedError(op)


class Location:
    def __init__(self, loc_id_str: str, loc_name: str, fused: List[str] = None) -> None:
        super().__init__()
        self.loc_ref = loc_id_str
        self.loc_name = loc_name
        self.fused = fused

    @property
    def isfused(self):
        return self.fused is not None

    @staticmethod
    def parse(location_token: str):
        """
        <>
        #loc = loc(unknown)
        #loc2 = loc("onnx::Conv_2949")
        """
        loc_id_str, _, name_pos = location_token.split(" ", maxsplit=2)

        if "unknown" in name_pos:
            loc_name = "unknown"
        elif "fused" in name_pos:
            fused_id = name_pos[10:-2].split(", ")
            return Location(
                loc_id_str=loc_id_str, loc_name=f"fused_{name_pos}", fused=fused_id
            )
        else:
            loc_name = name_pos[4:-1].strip('"')

        return Location(loc_id_str=loc_id_str, loc_name=loc_name)

    def dump(self):
        loc_name = f'"{self.loc_name}"' if self.loc_name != "unknown" else self.loc_name
        if self.fused is not None:
            fused_name = ", ".join([f"{i}" for i in self.fused])
            return f"{self.loc_ref} = loc(fused[{fused_name}])"
        else:
            return f"{self.loc_ref} = loc({loc_name})"


class FinalMlirIndex:
    loc_ref_match = re.compile(r"loc\((#loc[0-9]+)\)")
    lines: List[str]
    attributes: Dict
    loc: TensorLoc
    locname2locref: dict
    locref2locname: dict
    locref2fileline: dict
    fileline2locref: dict

    @classmethod
    def from_context(cls, mlir_file, tensor_loc_file=None):
        self = cls()
        from .lazy_mlir_ir import ir_context, mlir_ir, parse_attribute

        with open(mlir_file, "r") as f:
            context = f.read()
        self.lines = lines = [i for i in context.split("\n")]

        ctx = ir_context
        ctx.allow_unregistered_dialects = True
        module = mlir_ir.Module.parse(context, ctx)

        attributes = module.operation.attributes
        arr_map = {}
        for i in range(len(attributes)):
            arr_map.update(parse_attribute(attributes[i]))

        self.attributes = arr_map
        self.loc = TensorLoc.from_tl_file(tensor_loc_file)

        self.locname2locref = locname2locref = {}
        self.locref2locname = locref2locname = {}
        self.locref2fileline = locref2fileline = {}
        self.fileline2locref = fileline2locref = {}

        # operations = iter_operations(module)

        loc_ref_match = self.loc_ref_match
        for lino, line in enumerate(lines, start=1):
            if line.startswith("#loc"):
                loc = Location.parse(line)
                if loc.isfused:
                    continue
                locname2locref[loc.loc_name] = loc.loc_ref
                locref2locname[loc.loc_ref] = loc.loc_name
            else:
                match = loc_ref_match.search(line)
                if match and "tpu.Store" not in line:
                    loc_ref = match.group(1)
                    locref2fileline[loc_ref] = lino
                    fileline2locref[lino] = loc_ref
        return self

    def get_locname_by_fileline(self, lino: int):
        loc_name = self.fileline2locref[lino]
        return self.locref2locname[loc_name]
