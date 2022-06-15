//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
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
    StructFieldAttr<"resize_dims", I32ArrayAttr>,
    StructFieldAttr<"keep_aspect_ratio", BoolAttr>,
    StructFieldAttr<"scale", F32ArrayAttr>,
    StructFieldAttr<"mean", F32ArrayAttr>]> {
  let summary = "Structure of parameters for preprocess";
}


#endif
