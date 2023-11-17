//===- sg2260Dailect.cpp - SG2260 dialect  --------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu-mlir/Dialect/SG2260/IR/SG2260.h"

using namespace mlir;
using namespace tpu_mlir::sg2260;

#include "tpu-mlir/Dialect/SG2260/IR/SG2260Dialect.cpp.inc"

void SG2260Dialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tpu-mlir/Dialect/SG2260/IR/SG2260Ops.cpp.inc"
      >();
  registerTypes();
}
