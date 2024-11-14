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

void tpu::LogicalAndOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  LogicalAnd_param_t param = {0};
  int length = 1;
  auto shape = module::getShape(getInputs()[0]);
  for (int i = 0; i < shape.size(); ++i) {
    length *= shape[i];
  }
  param.length = length;
  BM168x::call_global_func("backend_api_logical_and_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

void tpu::LogicalAndOp::codegen_global_bm1684() {}

void tpu::LogicalAndOp::codegen_global_cv18xx(int64_t layer_id) {}
