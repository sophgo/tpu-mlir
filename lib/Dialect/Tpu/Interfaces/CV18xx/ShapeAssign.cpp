//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

using namespace tpu_mlir::backend;


// =========================================
// GlobalGenInterface
// =========================================
void tpu::ShapeAssignOp::codegen_global_cv18xx(int64_t layer_id) {
  llvm_unreachable("Only support dynamic codegen");
}
