//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Top/IR/TopOpsDialect.cpp.inc"

void TopDialect::initialize() {
  registerTopDialectOperations(this);
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tpu_mlir/Dialect/Top/IR/TopAttr.cpp.inc"
      >();
}

#define GET_ATTRDEF_CLASSES
#include "tpu_mlir/Dialect/Top/IR/TopAttr.cpp.inc"
