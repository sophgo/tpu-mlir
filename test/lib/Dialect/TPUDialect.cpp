//===- bm1690Dailect.cpp - BM1690 dialect  --------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "TPUDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

//===----------------------------------------------------------------------===//
// TestDialect
//===----------------------------------------------------------------------===//

#include "TPUOpsDialect.cpp.inc"

using namespace mlir;
using namespace tpu;

void TPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
      >();
  addOperations<
#define GET_OP_LIST
#include "TPUOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "TPUOps.cpp.inc"
