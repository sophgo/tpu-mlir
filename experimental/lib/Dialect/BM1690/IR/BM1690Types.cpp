//===- bm1690Types.cpp - BM1690 Types  ------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "tpu-mlir/Dialect/BM1690/IR/BM1690.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace tpu_mlir::bm1690;

#define GET_TYPEDEF_CLASSES
#include "tpu-mlir/Dialect/BM1690/IR/BM1690Types.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "tpu-mlir/Dialect/BM1690/IR/BM1690AttrDefs.cpp.inc"

#include "tpu-mlir/Dialect/BM1690/IR/BM1690Enum.cpp.inc"

void BM1690Dialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "tpu-mlir/Dialect/BM1690/IR/BM1690Types.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tpu-mlir/Dialect/BM1690/IR/BM1690AttrDefs.cpp.inc"
      >();
}
