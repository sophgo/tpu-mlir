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
void tpu::ShapeTransposeOp::codegen_global_bm1684x() {
  llvm_unreachable("Not supported now");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ShapeTransposeOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(shape_transpose_param_t);
  shape_transpose_param_t param = {0};
  auto perm = module::getI64Array(getOrder());
  auto in_shape = module::getShape(getInput());
  int dims = in_shape.size();
  for (int i = 0; i < dims; i++) {
    param.order[i] = perm->at(i);
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::ShapeTransposeOp::get_fw_type_bm1684x() {
  return FW_BMNET_SHAPE_TRANSPOSE;
}