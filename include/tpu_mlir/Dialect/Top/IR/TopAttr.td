//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_TOP_ATTRS
#define TPU_MLIR_TOP_ATTRS

include "tpu_mlir/Dialect/Top/IR/TopBase.td"

//===----------------------------------------------------------------------===//
// Input preprocess parameters structure.
//===----------------------------------------------------------------------===//
def Top_PreprocessParamAttr : StructAttr<"PreprocessParam", Top_Dialect, [
    StructFieldAttr<"pixel_format", StrAttr>,
    StructFieldAttr<"channel_format", StrAttr>,
    StructFieldAttr<"resize_dims", I64ArrayAttr>,
    StructFieldAttr<"keep_aspect_ratio", BoolAttr>,
    StructFieldAttr<"pad_value", I64Attr>,
    StructFieldAttr<"pad_type", StrAttr>,
    StructFieldAttr<"scale", F64ArrayAttr>,
    StructFieldAttr<"mean", F64ArrayAttr>]> {
  let summary = "Structure of parameters for preprocess";
}


#endif
