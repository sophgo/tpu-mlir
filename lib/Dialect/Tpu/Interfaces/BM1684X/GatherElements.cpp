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

void tpu::GatherElementsOp::codegen_global_bm1684x() {
  auto op = getOperation();
  gather_elements_global_param_t param{0};
  param.axis = getAxis();
  param.index_is_coeff = false;
  param.intermediate_buffer_global_addr = module::getAddress(getBuffer());
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_gather_global", &param, sizeof(param),
                           input_spec->data(), output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::GatherElementsOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) {
    return sizeof(gather_elements_global_param_t);
  }
  auto op = getOperation();
  gather_elements_global_param_t param{0};
  param.axis = getAxis();
  param.index_is_coeff = false;
  param.intermediate_buffer_global_addr = module::getAddress(getBuffer());
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::GatherElementsOp::get_fw_type_bm1684x() {
  return FW_BMNET_GATHERELEMENTS;
}
