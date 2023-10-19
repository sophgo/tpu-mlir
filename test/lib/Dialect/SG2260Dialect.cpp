//===- sg2260Dailect.cpp - SG2260 dialect  --------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "SG2260Dialect.h"

//===----------------------------------------------------------------------===//
// TestDialect
//===----------------------------------------------------------------------===//

#include "SG2260OpsDialect.cpp.inc"

using namespace mlir;
using namespace sg2260;

static LogicalResult
setPropertiesFromAttribute(DMARegDef &prop, Attribute attr,
                           InFlightDiagnostic *diagnostic) {
  return success();
};
static DictionaryAttr getPropertiesAsAttribute(MLIRContext *ctx,
                                               const DMARegDef &prop) {
  return {};
};

static llvm::hash_code computeHash(const DMARegDef &prop) {
  auto start = reinterpret_cast<const uint64_t *>(&prop);
  auto end = start + sizeof(DMARegDef);
  return llvm::hash_combine_range(start, end);
};

static LogicalResult
setPropertiesFromAttribute(Matrix2RegDef &prop, Attribute attr,
                           InFlightDiagnostic *diagnostic) {
  return success();
};

static DictionaryAttr getPropertiesAsAttribute(MLIRContext *ctx,
                                               const Matrix2RegDef &prop) {
  return {};
};
static llvm::hash_code computeHash(const Matrix2RegDef &prop) {
  auto start = reinterpret_cast<const uint64_t *>(&prop);
  auto end = start + sizeof(Matrix2RegDef);
  return llvm::hash_combine_range(start, end);
};
