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
void tpu::CumSumOp::codegen_global_bm1684() {
  llvm_unreachable("Not Implemented");
}

uint32_t tpu::CumSumOp::dyn_codegen_global_bm1684(void* ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}

int64_t tpu::CumSumOp::get_fw_type_bm1684() {
  return -1;
}
