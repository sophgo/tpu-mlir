//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include <numeric>

using namespace mlir;
using namespace tpu_mlir::tpu;

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/IR/TpuOpsDialect.cpp.inc"
#include "tpu_mlir/Dialect/Tpu/IR/TpuAttr.cpp.inc"

void TpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Tpu Operator Definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.cpp.inc"
