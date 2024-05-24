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

void tpu::Yuv2rgbFormulaOp::codegen_global_bm1684() {
  llvm_unreachable("Not Implemented");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
uint32_t tpu::Yuv2rgbFormulaOp::dyn_codegen_global_bm1684(void *buffer) {
  llvm_unreachable("Not Implemented");
}

int64_t tpu::Yuv2rgbFormulaOp::get_fw_type_bm1684() { return -1; }
