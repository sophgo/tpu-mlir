//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_TPU_ATTRS
#define TPU_MLIR_TPU_ATTRS

include "mlir/IR/EnumAttr.td"
include "tpu_mlir/Dialect/Tpu/IR/TpuBase.td"

def Tpu_LayerGroupAttr :
    StructAttr<"LayerGroup", Tpu_Dialect, [
    StructFieldAttr<"out_addr", DefaultValuedAttr<I64Attr, "0">>,
    StructFieldAttr<"out_size", DefaultValuedAttr<I64Attr, "0">>,
    StructFieldAttr<"buffer_addr", DefaultValuedAttr<I64Attr, "0">>,
    StructFieldAttr<"buffer_size", DefaultValuedAttr<I64Attr, "0">>,
    StructFieldAttr<"eu_align", DefaultValuedAttr<BoolAttr, "false">>,
    StructFieldAttr<"h_idx", I64ArrayAttr>,
    StructFieldAttr<"h_slice", I64ArrayAttr>,
    StructFieldAttr<"n_idx", I64ArrayAttr>,
    StructFieldAttr<"n_slice", I64ArrayAttr>,
    StructFieldAttr<"timestep", I64Attr>,
]> {
  let summary = "Structure of layer group parameters";
}

def Tpu_DequantMode: I32EnumAttr<"DequantMode",
    "dequant mode supported by DequantOp",
    [
      I32EnumAttrCase<"Normal", 0>,
      I32EnumAttrCase<"TFlite", 1>
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_DequantModeAttr : EnumAttr<Tpu_Dialect, Tpu_DequantMode, "dq_mode">;

def Tpu_RequantMode: I32EnumAttr<"RequantMode",
    "requant mode supported by RequantOp",
    [
      I32EnumAttrCase<"TFlite_Lshift", 0>,
      I32EnumAttrCase<"TFlite", 1>,
      I32EnumAttrCase<"Normal", 2>,
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_RequantModeAttr : EnumAttr<Tpu_Dialect, Tpu_RequantMode, "rq_mode">;

#endif
