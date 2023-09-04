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

  layer_norm_global_spec_t param = {0};
  const bool have_weight = !getWeight().getType().isa<NoneType>();
  const bool have_bias = !getBias().getType().isa<NoneType>();

  param.common.axis = (int)getAxis();
  param.common.eps = getEps().convertToDouble();
  param.common.affine = (have_weight << 0) + (have_bias << 1);
  param.common.need_mean = false;
  param.common.need_rstd = false;
  BM168x::call_global_func("backend_api_layer_norm_train_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

void tpu::LayerNormTrainOp::codegen_global_bm1684() {

}

void tpu::LayerNormTrainOp::codegen_global_cv18xx(int64_t layer_id) {

}
