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
from typing import List, Tuple, Dict
from debugger.target_common import DType

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


class Pickled_Value:
    def __init__(self, value, Memref, Type, Zero_point, Scale, file_line, cmd_point):
        self.address: int = value.address
        self.layout: str = value.layout
        self.memory_type: str = value.memory_type
        self.name: str = value.name
        self.reshape: str = value.reshape
        self.slice: str = value.slice
        self._type: str = value._type
        self.dtype = to_dtype[self.memory_type.strip(">").split("x")[-1]].np_dtype()
        self.file_line = file_line
        self.cmd_point = cmd_point

        self.memref = Memref
        self.mlir_type = str(Type)
        self.zero_point = Zero_point
        self.scale = Scale

    def __repr__(self) -> str:
        return f"({{name={self.name}, cmd_point={self.cmd_point}, file_line={self.file_line}, slice={self.slice}, memory_type={self.memory_type}}})"
