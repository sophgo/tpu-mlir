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

void tpu::ScatterElementsOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  scatter_elements_global_spec_t param = {0};
  auto data_shape = module::getShape(getInput());
  auto indices_shape = module::getShape(getIndices());
  auto updates_shape = module::getShape(getUpdates());
  param.data_dims = data_shape.size();
  param.indices_dims = indices_shape.size();
  param.updates_dims = updates_shape.size();
  param.intermediate_buffer_global_addr = module::getAddress(getBuffer());
  param.axis = getAxis();
  param.replace_add = getReduction();
  BM168x::call_global_func("backend_api_scatter_elements_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ScatterElementsOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(scatter_elements_global_spec_t);
  scatter_elements_global_spec_t param{0};
  auto data_shape = module::getShape(getInput());
  auto indices_shape = module::getShape(getIndices());
  auto updates_shape = module::getShape(getUpdates());
  param.data_dims = data_shape.size();
  param.indices_dims = indices_shape.size();
  param.updates_dims = updates_shape.size();
  param.intermediate_buffer_global_addr = module::getAddress(getBuffer());
  param.axis = getAxis();
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::ScatterElementsOp::get_fw_type_bm1684x() {
  return FW_BMNET_SCATTERELEMENTS;
}
