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

void tpu::LayerNormBwdOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  auto normalized_shape = module::getI64Array(getNormalizedShape());
  layernorm_backward_param_t param = {0};
  param.axis = input_spec->at(0).dims - normalized_shape->size();
  BM168x::call_global_func("backend_api_layernorm_backward_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

void tpu::LayerNormBwdOp::codegen_global_bm1684() {}

void tpu::LayerNormBwdOp::codegen_global_cv18xx(int64_t layer_id) {}

// // dynamic codegen
// int64_t tpu::LayerNormBwdOp::dyn_codegen_local_bm1684x(void *buffer) {
//     return 0;
// }

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::LayerNormBwdOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}

int64_t tpu::LayerNormBwdOp::get_fw_type_bm1684x() { return FW_BMNET_CONV; }
