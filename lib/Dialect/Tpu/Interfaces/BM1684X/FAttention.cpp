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
void tpu::FAttentionOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  flash_attention_global_spec_t param = {0};
  auto &common = param.common;
  // get_param(op, common);
  // common.num_core = module::getCoreNum();
  common.batch = getBatch();
  common.q_head = getQHead();
  common.kv_head = getKvHead();
  common.mq = getMq();
  common.mk = getMk();
  common.dim = getDim();
  common.scale = getScale().convertToDouble();
  common.hasmask = !module::isNone(getMask());
  common.high_precision = module::isHighPrecision();

  BM168x::call_ppl_global_func("api_fattention_global", &param, sizeof(param),
                               input_spec->data(), output_spec->data());
}

int64_t tpu::FAttentionOp::get_fw_type_bm1684x() { return FW_BMNET_FATTENTION; }

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::FAttentionOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!module::isBM1684X()) {
    UNREACHABLE_THIS("Not Implemented");
  }
  if (!buffer)
    return sizeof(flash_attention_global_spec_t);
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  flash_attention_global_spec_t param = {0};
  auto &common = param.common;
  // get_param(op, common);
  common.batch = getBatch();
  common.q_head = getQHead();
  common.kv_head = getKvHead();
  common.mq = getMq();
  common.mk = getMk();
  common.dim = getDim();
  common.scale = getScale().convertToDouble();
  common.hasmask = !module::isNone(getMask());
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}
