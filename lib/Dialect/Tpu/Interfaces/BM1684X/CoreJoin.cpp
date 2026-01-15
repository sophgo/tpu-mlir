//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"

using namespace tpu_mlir::backend;

void tpu::CoreJoinOp::codegen_global_bm1684x() { return; }

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::CoreJoinOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) {
    return sizeof(core_join_spec_t);
  }
  core_join_spec_t param{0};
  param.axis = getAxis();
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::CoreJoinOp::get_fw_type_bm1684x() { return FW_BMNET_CORE_JOIN; }
