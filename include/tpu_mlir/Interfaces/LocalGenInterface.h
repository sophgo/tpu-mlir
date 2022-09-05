//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace tpu_mlir {
typedef struct {
  int64_t out_addr;
  int64_t out_size;
  int64_t buffer_addr;
  int64_t buffer_size;
  int64_t n_idx;
  int64_t n_slice;
  int64_t h_idx;
  int64_t h_slice;
  int64_t id;
  int64_t stage;
  bool eu_align;
  bool overstepped;
} group_info_t;
} // namespace tpu_mlir

#include "tpu_mlir/Support/Helper/Module.h"
using namespace tpu_mlir::helper;

/// Include the ODS generated interface header files.
#include "tpu_mlir/Interfaces/LocalGenInterface.h.inc"
