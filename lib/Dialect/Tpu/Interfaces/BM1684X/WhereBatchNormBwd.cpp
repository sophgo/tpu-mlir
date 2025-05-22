//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::WhereBnbwdOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  where_batchnorm_backward_param_t param = {0};
  param.do_recompute = getDoRecompute();
  BM168x::call_global_func("backend_api_where_batchnorm_backward_global",
                           &param, sizeof(param), input_spec->data(),
                           output_spec->data());
}

void tpu::WhereBnbwdOp::codegen_global_cv18xx(int64_t layer_id) {}

// // dynamic codegen

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::WhereBnbwdOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::WhereBnbwdOp::get_fw_type_bm1684x() { return FW_LAYER_UNKNOWN; }
