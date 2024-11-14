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

// ======================================
// GlobalGenInterface
// ======================================
void tpu::ShapeCastOp::codegen_global_bm1684x() {
  llvm_unreachable("Not supported now");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ShapeCastOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(shape_cast_param_t);
  // TODO: fix it
  shape_cast_param_t param = {0};
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::ShapeCastOp::get_fw_type_bm1684x() { return FW_BMNET_SHAPE_CAST; }

mlir::Type tpu::ShapeCastOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  return do_nothing(mode);
}
