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
void tpu::FlexAttentionOp::codegen_global_bm1684x() {
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
  common.dim = getQkD(); // informational; flex uses qk_d/v_d below
  common.qk_d = getQkD();
  common.v_d = getVD();
  common.scale = getScale().convertToDouble();
  common.hasmask = !module::isNone(getMask());
  common.has_bitmap = !module::isNone(getBlockBitmap());
  common.has_lse = getHasLse();
  common.flex_block = getFlexBlock();
  common.high_precision = module::isHighPrecision();
  common.keep_dim = getKeepDims();
  common.mask_size = getMaskSize();

  BM168x::call_ppl_global_func("api_fattention_flex_global", &param,
                               sizeof(param), input_spec->data(),
                               output_spec->data());
}

int64_t tpu::FlexAttentionOp::get_fw_type_bm1684x() {
  // flex uses the v2 (high-precision) fattention kernel family
  return PPL_FW_FLASH_ATTENTION_HEIGH_PRECISION;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::FlexAttentionOp::dyn_codegen_global_bm1684x(void *buffer) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  flash_attention_global_spec_t param = {0};
  auto &common = param.common;
  common.high_precision = module::isHighPrecision();
  common.hasmask = !module::isNone(getMask());
  common.has_bitmap = !module::isNone(getBlockBitmap());
  common.has_lse = getHasLse();
  common.flex_block = getFlexBlock();
  common.mask_size = getMaskSize();
  if (buffer) {
    common.batch = getBatch();
    common.q_head = getQHead();
    common.kv_head = getKvHead();
    common.mq = getMq();
    common.mk = getMk();
    common.dim = getQkD();
    common.qk_d = getQkD();
    common.v_d = getVD();
    common.scale = getScale().convertToDouble();
    common.keep_dim = getKeepDims();
  }
  return BM168x::call_ppl_dyn_func("api_dyn_fattention_flex_global", &param,
                                   input_spec->data(), output_spec->data(),
                                   buffer);
}
