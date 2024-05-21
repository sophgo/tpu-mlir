//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

// =========================================
// GlobalGenInterface
// =========================================

void tpu::GatherNDOp::codegen_global_cv18xx(int64_t layer_id) {
  UNREACHABLE_THIS("Not Implemented");
}
