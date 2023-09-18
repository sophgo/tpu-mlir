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
import mlir
import mlir.ir
from mlir.dialects.func import FuncOp
from mlir.dialects import quant
from mlir.ir import *
from .target_common import DType, CMDType


to_dtype: Dict[str, DType] = {
    "f32": DType.f32,
    "f16": DType.f16,
    "bf16": DType.bf16,
    "i8": DType.i8,
    "si8": DType.si8,
    "ui8": DType.ui8,
    "u8": DType.ui8,
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


def parse_attribute(attr: Attribute) -> dict:
    if isinstance(attr, NamedAttribute):
        return {attr.name: parse_attribute(attr.attr)}
    if isinstance(attr, (StringAttr, IntegerAttr, BoolAttr)):
        return attr.value

    if isinstance(attr, (ArrayAttr)):
        return [parse_attribute(i) for i in attr]


ir_context = mlir.ir.Context()


def parse_mlir_type(asm: str):
    element_type = Type.parse(asm, ir_context).element_type
    if quant.UniformQuantizedType.isinstance(element_type):
        quant_type = quant.UniformQuantizedType(element_type)
        return quant_type
    if quant.CalibratedQuantizedType.isinstance(element_type):
        quant_type = quant.CalibratedQuantizedType(element_type)
        return quant_type
    return RankedTensorType.parse(asm, ir_context)


class Value:
    def __init__(self, dic) -> None:
        self.address: int = dic["address"]
        self.layout: str = dic["layout"]
        self.memory_type: str = dic["memory_type"]
        self.name: str = dic["name"]
        self.reshape: str = dic["reshape"]
        self.slice: str = dic["slice"]
        self._type: Type = dic["type"]
        self.dtype = to_dtype[self.memory_type.strip(">").split("x")[-1]].np_dtype()

    @property
    @lru_cache()
    def type(self):
        return parse_mlir_type(self._type)

    @property
    def zero_point(self):
        if "uniform" not in self._type:
            return 0
        if isinstance(self.type, quant.UniformQuantizedType):
            return self.type.zero_point
        return 0

    @property
    def scale(self):
        if "uniform" not in self._type:
            return 1
        if isinstance(self.type, quant.UniformQuantizedType):
            return self.type.scale
        return 1

    def __repr__(self) -> str:
        return f"@{self.address}({{name={self.name}, layout={self.layout}, slice={self.slice}, mlir_type={self.type}}})"


class CMD:
    def __init__(self, dic) -> None:
        self.file_line: int = dic["file-line"]
        self.subnet_id: int = dic["subnet_id"]
        self.opcode: str = dic["opcode"]
        self.core_id: int = dic.get("core_id", 0)
        # dma_id before is not aligned with asm cmd
        # For example, yolov5 cmd[2] use B0 as cmd_id_dep
        # but tensor_loc.json has tiu_dma_id(before) = (1, 1)
        self.tiu_dma_id_before: Tuple[int, int] = dic["tiu_dma_id(before)"]
        self.tiu_dma_id_after: Tuple[int, int] = dic["tiu_dma_id(after)"]
        # None for top.None
        self.operands = [Value(i) if len(i) > 0 else None for i in dic["operands"]]
        # None for no usage results
        self.results = [Value(i) if len(i) > 0 else None for i in dic["results"]]

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
    def cmd_id(self):
        if self.cmd_type == CMDType.tiu:
            return self.tiu_dma_id_after[0]
        else:
            return self.tiu_dma_id_after[1]

    @property
    def loc_name(self):
        return self.results[0].name

    def __repr__(self) -> str:
        return pformat(self.__dict__)

    __str__ = __repr__


class TensorLoc:
    def __init__(self, tensor_loc_file) -> None:
        with open(tensor_loc_file) as r:
            self.tensor_loc: List[CMD] = [CMD(i) for i in json.load(r)]

        print("load loc len", len(self.tensor_loc))

    def __repr__(self) -> str:
        return pformat(self.__dict__)

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            return self.tensor_loc[index]
        raise KeyError(index)


def iter_operations(op):
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
        self.loc_id_str = loc_id_str
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
            return f"{self.loc_id_str} = loc(fused[{fused_name}])"
        else:
            return f"{self.loc_id_str} = loc({loc_name})"


class FinalMlirIndex:
    loc_ref_match = re.compile(r"loc\((#loc[0-9]+)\)")

    # cmd_id -> loc_name -> loc_index -> file-line
    def __init__(self, mlir_file, tensor_loc_file=None) -> None:
        with open(mlir_file, "r") as f:
            context = f.read()
        self.lines = lines = [i for i in context.split("\n")]
        self.ctx = mlir.ir.Context()
        self.ctx.allow_unregistered_dialects = True

        module = mlir.ir.Module.parse(context, self.ctx)

        attributes = module.operation.attributes
        arr_map = {}
        for i in range(len(attributes)):
            arr_map.update(parse_attribute(attributes[i]))

        self.attributes = arr_map
        self.module = module
        self.loc = TensorLoc(tensor_loc_file)

        self.locname2locindex = locname2locindex = {}
        self.locindex2fileline = locindex2fileline = {}

        loc_ref_match = self.loc_ref_match
        for index, line in enumerate(lines):
            if line.startswith("#loc"):
                loc = Location.parse(line)
                if loc.isfused:
                    continue
                locname2locindex[loc.loc_name] = loc.loc_id_str
            else:
                match = loc_ref_match.search(line)
                if match and "tpu.Store" not in line:
                    loc_id = match.group(1)
                    locindex2fileline[loc_id] = index

    def get_fileline_by_loc(self, loc_name: str):
        loc_ref = self.locname2locindex[loc_name]
        return self.locindex2fileline[loc_ref]

    def get_operations(self):
        return iter_operations(self.module)
