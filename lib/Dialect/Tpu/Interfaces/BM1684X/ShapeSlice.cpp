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
void tpu::ShapeSliceOp::codegen_global_bm1684x() {
  llvm_unreachable("Not supported now");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ShapeSliceOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(shape_slice_param_t);
  shape_slice_param_t param = {0};
  const std::vector<int64_t> input_shape = module::getShape(getInput());
  const std::vector<int64_t> output_shape = module::getShape(getOutput());
  param.begin_mask = 0;
  param.end_mask = 0;
  const int num_dims = input_shape.size();
  const auto offset = module::getI64Array(getOffset());
  const auto ends = module::getI64Array(getEnds());
  const auto steps = module::getI64Array(getSteps());
  param.shape_size = offset->size();
  param.ellipsis_mask = 0;
  param.new_axis_mask = 0;
  param.shrink_axis_mask = 0;
  param.is_dynamic = !module::isNone(getOffsetT());
  if (param.is_dynamic) {
    assert(!module::isNone(getEndsT()));
    assert(!module::isNone(getStepsT()));
  }
  for (int i = 0; i < num_dims; i++) {
    param.begin_index[i] = offset->at(i);
    param.stride[i] = steps->at(i);
    // TODO: fix canonicalizers and reactivate this
    // param.end_index[i] = ends->at(i);
    param.end_index[i] = output_shape[i] * steps->at(i) + offset->at(i);
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::ShapeSliceOp::get_fw_type_bm1684x() {
  return FW_BMNET_SHAPE_SLICE;
}
