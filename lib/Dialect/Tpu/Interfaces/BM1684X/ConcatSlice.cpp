//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
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
void tpu::ConcatSliceOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  concat_slice_spec_t param = {0};
  param.axis = getAxis();
  BM168x::call_ppl_global_func("api_concat_slice_global", &param, sizeof(param),
                               input_spec->data(), output_spec->data());
}

int64_t tpu::ConcatSliceOp::get_fw_type_bm1684x() {
  return PPL_FW_CONCAT_SLICE;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ConcatSliceOp::dyn_codegen_global_bm1684x(void *buffer) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  concat_slice_spec_t param = {0};
  param.axis = getAxis();
  return BM168x::call_ppl_dyn_func("api_dyn_concat_slice_global", &param,
                                   input_spec->data(), output_spec->data(),
                                   buffer);
}
