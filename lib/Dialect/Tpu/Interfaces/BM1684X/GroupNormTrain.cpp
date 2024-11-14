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

void tpu::GroupNormTrainOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  group_norm_train_global_param_t param = {0};
  const bool have_weight = !module::isNone(getWeight());
  const bool have_bias = !module::isNone(getBias());
  param.axis = 1;
  param.common.group_num = (int)getNumGroups();
  param.common.eps = getEps().convertToDouble();
  param.common.affine = (have_weight << 0) + (have_bias << 1);
  BM168x::call_global_func("backend_api_group_norm_train_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

void tpu::GroupNormTrainOp::codegen_global_bm1684() {}

void tpu::GroupNormTrainOp::codegen_global_cv18xx(int64_t layer_id) {}
