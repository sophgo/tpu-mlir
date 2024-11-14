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
void tpu::RandnLikeOp::codegen_global_bm1684x() {
  llvm_unreachable("Not Interpreter");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::RandnLikeOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(randn_like_spec_t);
  randn_like_spec_t param = {0};
  auto shape = module::getShape(getOutput());
  auto dims = shape.size();
  for (int i = 0; i < dims; i++) {
    param.max_shape[i] = shape[i];
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::RandnLikeOp::get_fw_type_bm1684x() { return FW_BMNET_RANDNLIKE; }
