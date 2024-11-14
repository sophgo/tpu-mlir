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
void tpu::ShapeClipOp::codegen_global_bm1684x() {
  llvm_unreachable("Not supported now");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ShapeClipOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(shape_clip_param_t);
  shape_clip_param_t param = {0};
  param.min = static_cast<float>(getMin().convertToDouble());
  param.max = static_cast<float>(getMax().convertToDouble());
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::ShapeClipOp::get_fw_type_bm1684x() { return FW_BMNET_SHAPE_CLIP; }
