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
void tpu::ChunkGatedDeltaRuleOp::codegen_global_bm1684x() {
  // auto op = getOperation();
  // auto input_spec = BM168x::get_input_spec(op);
  // auto output_spec = BM168x::get_output_spec(op);

  // BM168x::call_ppl_global_func("api_ChunkGatedDeltaRule_global", &param,
  // sizeof(param),
  //                              input_spec->data(), output_spec->data());
  UNREACHABLE_THIS("Not Support static codegen now");
}

int64_t tpu::ChunkGatedDeltaRuleOp::get_fw_type_bm1684x() {
  return PPL_FW_CHUNK_GATED_DELTA_RULE;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ChunkGatedDeltaRuleOp::dyn_codegen_global_bm1684x(void *buffer) {
  // auto op = getOperation();
  // auto input_spec = BM168x::get_input_spec(op);
  // auto output_spec = BM168x::get_output_spec(op);
  // chunk_gated_delta_rule_global_spec_t param = {0};
  // auto &common = param.common;
  // common.high_precision = module::isHighPrecision();
  // if (buffer) {
  //   // get_param(op, common);
  // }
  // return BM168x::call_ppl_dyn_func("api_dyn_chunk_gated_delta_rule_global",
  // &param,
  //                                  input_spec->data(), output_spec->data(),
  //                                  buffer);
  // TODO: support dynamic codegen
  return 0;
}
