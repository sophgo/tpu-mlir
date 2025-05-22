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

void tpu::A16MatMulOp::codegen_global_bm1684x() {
  a16_matmul_spec_t spec;
  spec.has_bias = !module::isNone(getBias());
  spec.R_trans = getWTranspose();
  spec.sign = getSign();
  spec.weight_bits = getWeightBits();
  spec.has_zp = !module::isNone(getZp());
  spec.q_group_size = getQGroupSize();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  if (supportMultiCore(*this)) {
    spec.use_multi_core = 1;
  } else {
    spec.use_multi_core = 0;
  }
  BM168x::call_global_func("backend_api_a16_matmul_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::A16MatMulOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(a16_matmul_spec_t);
  a16_matmul_spec_t spec;
  memset(&spec, 0, sizeof(a16_matmul_spec_t));
  spec.has_bias = !module::isNone(getBias());
  spec.R_trans = getWTranspose();
  spec.sign = getSign();
  spec.weight_bits = getWeightBits();
  spec.has_zp = !module::isNone(getZp());
  spec.q_group_size = getQGroupSize();
  if (supportMultiCore(*this)) {
    spec.use_multi_core = 1;
  } else {
    spec.use_multi_core = 0;
  }
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::A16MatMulOp::get_fw_type_bm1684x() { return FW_BMNET_A16_MATMUL; }
