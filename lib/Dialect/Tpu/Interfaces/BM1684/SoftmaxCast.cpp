//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

void tpu::SoftmaxCastOp::codegen_global_bm1684() {
  llvm_unreachable("Not Implemented");
}

uint32_t tpu::SoftmaxCastOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
int64_t tpu::SoftmaxCastOp::get_fw_type_bm1684() { return -1; }
