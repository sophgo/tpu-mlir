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
void tpu::ShapeTransposeOp::codegen_global_bm1684() {
  llvm_unreachable("Not supported now");
}

uint32_t tpu::ShapeTransposeOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  llvm_unreachable("Not supported now");
}

int64_t tpu::ShapeTransposeOp::get_fw_type_bm1684() { return -1; }