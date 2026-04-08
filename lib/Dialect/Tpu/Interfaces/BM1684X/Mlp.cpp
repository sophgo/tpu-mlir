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

void tpu::MlpOp::codegen_global_bm1684x() {
  mlp_spec_t spec;
  spec.has_bias_gate = !module::isNone(getBiasGate());
  spec.R_trans_gate = getWTransposeGate();
  spec.has_zp_gate = !module::isNone(getZpGate());
  spec.has_bias_up = !module::isNone(getBiasUp());
  spec.R_trans_up = getWTransposeUp();
  spec.has_zp_up = !module::isNone(getZpUp());
  spec.has_bias_down = !module::isNone(getBiasDown());
  spec.R_trans_down = getWTransposeDown();
  spec.has_zp_down = !module::isNone(getZpDown());
  spec.quantized = getQuantized();
  spec.weight_bits = getWeightBits();
  spec.q_group_size = getQGroupSize();
  spec.is_expert = getIsExpert();
  spec.num_expert = getNumExpert();
  spec.num_expert_per_tok = getNumExpertPerTok();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  if (supportMultiCore(*this)) {
    spec.use_multi_core = 1;
  } else {
    spec.use_multi_core = 0;
  }
  spec.buffer_addr = module::getAddress(getBuffer());
  BM168x::call_global_func("backend_api_mlp_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::MlpOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(mlp_spec_t);
  mlp_spec_t spec;
  memset(&spec, 0, sizeof(mlp_spec_t));
  spec.has_bias_gate = !module::isNone(getBiasGate());
  spec.R_trans_gate = getWTransposeGate();
  spec.has_zp_gate = !module::isNone(getZpGate());
  spec.has_bias_up = !module::isNone(getBiasUp());
  spec.R_trans_up = getWTransposeUp();
  spec.has_zp_up = !module::isNone(getZpUp());
  spec.has_bias_down = !module::isNone(getBiasDown());
  spec.R_trans_down = getWTransposeDown();
  spec.has_zp_down = !module::isNone(getZpDown());
  spec.quantized = getQuantized();
  spec.weight_bits = getWeightBits();
  spec.q_group_size = getQGroupSize();
  spec.is_expert = getIsExpert();
  spec.num_expert = getNumExpert();
  spec.num_expert_per_tok = getNumExpertPerTok();
  if (supportMultiCore(*this)) {
    spec.use_multi_core = 1;
  } else {
    spec.use_multi_core = 0;
  }
  spec.buffer_addr = module::getAddress(getBuffer());
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::MlpOp::get_fw_type_bm1684x() { return FW_BMNET_MLP; }
