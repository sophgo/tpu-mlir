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

void tpu::ShapeScatterElementsOp::codegen_global_bm1684x() {
  llvm_unreachable("Not supported now");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ShapeScatterElementsOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(shape_scatterelements_spec_t);
  shape_scatterelements_spec_t param{0};
  auto data_shape = module::getShape(getInput());
  auto indices_shape = module::getShape(getIndices());
  auto updates_shape = module::getShape(getUpdates());
  param.data_dims = data_shape.size();
  param.indices_dims = indices_shape.size();
  param.updates_dims = updates_shape.size();
  param.axis = getAxis();
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::ShapeScatterElementsOp::get_fw_type_bm1684x() {
  return FW_BMNET_SHAPE_SCATTERELEMENTS;
}
