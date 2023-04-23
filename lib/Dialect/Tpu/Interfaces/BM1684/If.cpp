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
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/WeightReorder.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;
using namespace tpu_mlir::bm1684x;

// =========================================
// GlobalGenInterface
// =========================================
void tpu::IfOp::codegen_global_bm1684() {
}

uint32_t tpu::IfOp::dyn_codegen_global_bm1684(void *buffer) {
  return 0;
}

int64_t tpu::IfOp::get_fw_type_bm1684() {
  return -1;
}
