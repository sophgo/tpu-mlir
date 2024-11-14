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

void tpu::LayerNormTrainOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  auto normalized_shape = module::getI64Array(getNormalizedShape());
  layernorm_train_param_t param = {0};
  param.axis = input_spec->at(0).dims - normalized_shape->size();
  param.eps = getEps().convertToDouble();
  BM168x::call_global_func("backend_api_layernorm_train_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

void tpu::LayerNormTrainOp::codegen_global_bm1684() {}

void tpu::LayerNormTrainOp::codegen_global_cv18xx(int64_t layer_id) {}
