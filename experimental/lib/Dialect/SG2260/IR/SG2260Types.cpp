//===- sg2260Types.cpp - SG2260 Types  ------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "tpu-mlir/Dialect/SG2260/IR/SG2260.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace tpu_mlir::sg2260;

#define GET_TYPEDEF_CLASSES
#include "tpu-mlir/Dialect/SG2260/IR/SG2260Types.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "tpu-mlir/Dialect/SG2260/IR/SG2260AttrDefs.cpp.inc"

#include "tpu-mlir/Dialect/SG2260/IR/SG2260Enum.cpp.inc"

void SG2260Dialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "tpu-mlir/Dialect/SG2260/IR/SG2260Types.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tpu-mlir/Dialect/SG2260/IR/SG2260AttrDefs.cpp.inc"
      >();
}
