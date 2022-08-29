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

#endif
