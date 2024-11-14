//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================
void tpu::MatchTemplateOp::codegen_global_bm1684x() {
  llvm_unreachable("Not support now.");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::MatchTemplateOp::dyn_codegen_global_bm1684x(void *buffer) {
  llvm_unreachable("Not support now.");
  return -1;
}

int64_t tpu::MatchTemplateOp::get_fw_type_bm1684x() {
  llvm_unreachable("Not support now.");
  return -1;
}
