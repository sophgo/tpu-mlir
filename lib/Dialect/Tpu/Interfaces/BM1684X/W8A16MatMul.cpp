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

void tpu::W8A16MatMulOp::codegen_global_bm1684x() {
  w8a16_matmul_spec_t spec;
  spec.has_bias = !module::isNone(getBias());
  spec.R_trans = getWTranspose();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_w8a16_matmul_global", &spec,
                           sizeof(spec), input_spec->data(),
                           output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::W8A16MatMulOp::dyn_codegen_global_bm1684x(void *buffer) {
  llvm_unreachable("Not supported now");
}

int64_t tpu::W8A16MatMulOp::get_fw_type_bm1684x() { return -1; }
