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

void tpu::Fp8MatMulOp::codegen_global_bm1684x() {
  // NOTE: The PPL w8a8_block_matmul kernel does not support bias.
  // If bias is present, it should be handled by a separate Add op.
  w8a8_block_matmul_spec_t spec;
  spec.block_size_k = getBlockSize();
  spec.block_size_n = getBlockSize();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_ppl_global_func("api_w8a8_block_matmul_global", &spec,
                               sizeof(spec), input_spec->data(),
                               output_spec->data());
}

int64_t tpu::Fp8MatMulOp::get_fw_type_bm1684x() {
  return PPL_FW_W8A8_BLOCK_MATMUL;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::Fp8MatMulOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(w8a8_block_matmul_spec_t);
  w8a8_block_matmul_spec_t spec;
  memset(&spec, 0, sizeof(w8a8_block_matmul_spec_t));
  spec.block_size_k = getBlockSize();
  spec.block_size_n = getBlockSize();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  return BM168x::call_ppl_dyn_func("api_dyn_w8a8_block_matmul_global", &spec,
                                   input_spec->data(), output_spec->data(),
                                   buffer);
}
