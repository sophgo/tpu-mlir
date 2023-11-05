//===- sg2260Dailect.cpp - SG2260 dialect  --------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "tpu-mlir/Dialect/SG2260/IR/SG2260Dialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "tpu-mlir/Dialect/SG2260/IR/SG2260Types.h.inc"

#include "SG2260RegDef.h"
#define GET_OP_CLASSES
#include "tpu-mlir/Dialect/SG2260/IR/SG2260Ops.h.inc"
