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
void tpu::FAttentionLseOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  flash_attention_global_spec_t param = {0};
  auto &common = param.common;
  common.batch = getBatch();
  common.q_head = getQHead();
  common.kv_head = getKvHead();
  common.mq = getMq();
  common.mk = getMk();
  common.dim = getDim();
  common.scale = getScale().convertToDouble();
  common.hasmask = !module::isNone(getMask());
  common.high_precision = module::isHighPrecision();
  common.keep_dim = getKeepDims();
  common.mask_size = getMaskSize();
  common.has_lse = true; // always emit lse for this op

  BM168x::call_ppl_global_func("api_fattention_lse_global", &param,
                               sizeof(param), input_spec->data(),
                               output_spec->data());
}

int64_t tpu::FAttentionLseOp::get_fw_type_bm1684x() {
  // lse only exists on the v2 (high-precision) path
  return PPL_FW_FLASH_ATTENTION_HEIGH_PRECISION;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::FAttentionLseOp::dyn_codegen_global_bm1684x(void *buffer) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  flash_attention_global_spec_t param = {0};
  auto &common = param.common;
  common.high_precision = module::isHighPrecision();
  common.hasmask = !module::isNone(getMask());
  common.mask_size = getMaskSize();
  common.has_lse = true;
  if (buffer) {
    common.batch = getBatch();
    common.q_head = getQHead();
    common.kv_head = getKvHead();
    common.mq = getMq();
    common.mk = getMk();
    common.dim = getDim();
    common.scale = getScale().convertToDouble();
    common.keep_dim = getKeepDims();
  }
  return BM168x::call_ppl_dyn_func("api_dyn_fattention_lse_global", &param,
                                   input_spec->data(), output_spec->data(),
                                   buffer);
}
