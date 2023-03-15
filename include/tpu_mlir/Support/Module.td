//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

// =============================================================================
//
// Defines TOP Dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_MODULE_ATTR
#define TPU_MLIR_MODULE_ATTR

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinDialect.td"

def Module_State: I32EnumAttr<"State", "module state",[
      I32EnumAttrCase<"TOP_F32", 0>,
      I32EnumAttrCase<"TOP_CALIBRATED", 1>,
      I32EnumAttrCase<"TOP_QUANTIZED", 2>,
      I32EnumAttrCase<"TPU_LOWERED", 3>,
      I32EnumAttrCase<"TPU_REORDERED", 4>,
      I32EnumAttrCase<"TPU_DIVIDED", 5>,
      I32EnumAttrCase<"TPU_ADDRESSED", 6>,
      I32EnumAttrCase<"TOSA_F32", 7>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::module";
}

def Module_Chip: I32EnumAttr<"Chip", "module chip",[
      I32EnumAttrCase<"ALL", 0>,
      I32EnumAttrCase<"BM1684",  1000, "bm1684">,
      I32EnumAttrCase<"BM1684X", 1001, "bm1684x">,
      I32EnumAttrCase<"BM1686",  1002, "bm1686">,
      I32EnumAttrCase<"CV180x",  2000, "cv180x">,
      I32EnumAttrCase<"CV181x",  2001, "cv181x">,
      I32EnumAttrCase<"CV182x",  2002, "cv182x">,
      I32EnumAttrCase<"CV183x",  2003, "cv183x">
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::module";
}

def Module_Mode: I32EnumAttr<"Mode", "module mode",[
      I32EnumAttrCase<"INT8",  0>,
      I32EnumAttrCase<"UINT8", 1>,
      I32EnumAttrCase<"INT4",  2>,
      I32EnumAttrCase<"BF16",  3>,
      I32EnumAttrCase<"F16",   4>,
      I32EnumAttrCase<"F32",   5>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::module";
}

def Module_Platform: I32EnumAttr<"Platform", "module platform",[
      I32EnumAttrCase<"ONNX",    0>,
      I32EnumAttrCase<"TORCH",   1>,
      I32EnumAttrCase<"TFLITE",  2>,
      I32EnumAttrCase<"CAFFE",   3>,
      I32EnumAttrCase<"TPULANG", 4>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::module";
}

#endif // TPU_MLIR_MODULE_ATTR
