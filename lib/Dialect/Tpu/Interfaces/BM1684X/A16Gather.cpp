//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
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
void tpu::A16GatherOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  a16_gather_spec_t param = {0};
  param.axis = getAxis();
  param.weight_bits = getWeightBits();
  param.q_group_size = getQGroupSize();
  BM168x::call_ppl_global_func("api_a16_gather_global", &param, sizeof(param),
                               input_spec->data(), output_spec->data());
}

int64_t tpu::A16GatherOp::get_fw_type_bm1684x() { return PPL_FW_A16_GATHER; }

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::A16GatherOp::dyn_codegen_global_bm1684x(void *buffer) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  a16_gather_spec_t param = {0};
  param.axis = getAxis();
  param.weight_bits = getWeightBits();
  param.q_group_size = getQGroupSize();
  return BM168x::call_ppl_dyn_func("api_dyn_a16_gather_global", &param,
                                   input_spec->data(), output_spec->data(),
                                   buffer);
}
