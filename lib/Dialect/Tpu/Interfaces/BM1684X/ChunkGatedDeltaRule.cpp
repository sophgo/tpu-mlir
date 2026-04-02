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
#if 0
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  chunk_gated_delta_rule_spec_t param = {0};
  param.num_k_heads = getNumKHeads();
  param.num_v_heads = getNumVHeads();
  param.d = getD();
  param.chunk_size = getChunkSize();
  param.use_qk_l2norm = getUseQkL2norm();
  param.scale = static_cast<float>(getScale().convertToDouble());
  BM168x::call_ppl_global_func("api_chunk_gated_delta_rule_global", &param,
                               sizeof(param), input_spec->data(),
                               output_spec->data());
#endif
  UNREACHABLE_THIS(
      "Not Support static global codegen for ChunkGatedDeltaRuleOp");
}

int64_t tpu::ChunkGatedDeltaRuleOp::get_fw_type_bm1684x() {
  return PPL_FW_CHUNK_GATED_DELTA_RULE;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ChunkGatedDeltaRuleOp::dyn_codegen_global_bm1684x(void *buffer) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  chunk_gated_delta_rule_spec_t param = {0};
  param.num_k_heads = getNumKHeads();
  param.num_v_heads = getNumVHeads();
  param.d = getD();
  param.chunk_size = getChunkSize();
  param.use_qk_l2norm = getUseQkL2norm();
  param.scale = static_cast<float>(getScale().convertToDouble());
  return BM168x::call_ppl_dyn_func("api_dyn_chunk_gated_delta_rule_global",
                                   &param, input_spec->data(),
                                   output_spec->data(), buffer);
}
