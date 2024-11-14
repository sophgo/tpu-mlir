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
void tpu::ShapeReduceOp::codegen_global_bm1684x() {
  llvm_unreachable("Not supported now");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ShapeReduceOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(shape_reduce_param_t);
  shape_reduce_param_t param = {0};
  const auto axis_list = module::getI64Array(getAxes());
  param.axis_num = axis_list->size();
  param.keep_dims = getKeepdims();
  param.reduce_method = BM168x::get_reduce_type(getMode());
  param.scale = getScale().convertToDouble();
  for (int i = 0; i < param.axis_num; i++) {
    param.axis_list[i] = axis_list->at(i);
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::ShapeReduceOp::get_fw_type_bm1684x() {
  return FW_BMNET_SHAPE_REDUCE;
}
