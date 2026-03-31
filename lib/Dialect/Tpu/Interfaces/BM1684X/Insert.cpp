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

void tpu::InsertOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  insert_spec_t param = {0};
  param.axis = getAxis();
  param.offset = getOffset();

  BM168x::call_ppl_global_func("api_insert_global", &param, sizeof(param),
                               input_spec->data(), output_spec->data());
  return;
}

int64_t tpu::InsertOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) {
    return sizeof(insert_spec_t);
  }
  insert_spec_t param = {0};
  param.axis = getAxis();
  param.offset = getOffset();
  return BM168x::dynamic_spec_to_buffer(buffer, &param);
}

int64_t tpu::InsertOp::get_fw_type_bm1684x() { return PPL_FW_INSERT_TENSOR; }
