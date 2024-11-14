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
void tpu::ShapeReshapeOp::codegen_global_bm1684x() {
  llvm_unreachable("Not Interpreter");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ShapeReshapeOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(shape_reshape_param_t);
  shape_reshape_param_t param = {0};
  auto shape = module::getI64Array(getShape());
  param.dims = shape->size();
  for (int i = 0; i < param.dims; i++) {
    param.shape[i] = shape->at(i);
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::ShapeReshapeOp::get_fw_type_bm1684x() {
  return FW_BMNET_SHAPE_RESHAPE;
}