//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Module.h"

#include <numeric>


using namespace tpu_mlir::top;


//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Top/IR/TopOpsDialect.cpp.inc"
#include "tpu_mlir/Dialect/Top/IR/TopAttr.cpp.inc"

void TopDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tpu_mlir/Dialect/Top/IR/TopAttr.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "tpu_mlir/Dialect/Top/IR/TopOps.cpp.inc"
      >();
  wFile = nullptr;
}

//===----------------------------------------------------------------------===//
// Top Operator Definitions.
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "tpu_mlir/Dialect/Top/IR/TopAttr.cpp.inc"

#define GET_OP_CLASSES
#include "tpu_mlir/Dialect/Top/IR/TopOps.cpp.inc"
