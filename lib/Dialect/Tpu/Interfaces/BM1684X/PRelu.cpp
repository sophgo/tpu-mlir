//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;


void tpu::PReluOp::codegen_global_bm1684x() {
  prelu_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.is_channel_shared = false;
  spec.slope_val = 0.f;
  spec.rshift_bit = getRshift();
  spec.upper_limit = -1;
  spec.round_mode = ROUND_UP;
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_prelu_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}

int64_t tpu::PReluOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::PReluOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                         local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  prelu_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.is_channel_shared = false;
  spec.slope_val = 0.f;
  spec.rshift_bit = getRshift();
  spec.upper_limit = 0;
  spec.round_mode = ROUND_UP;

  BM168x::call_local_func("backend_api_prelu_local", &spec, sizeof(spec),
                          &sec_info, input_spec->data(), output_spec->data());
}

//dynamic codegen
int64_t tpu::PReluOp::dyn_codegen_local_bm1684x(void *buffer) {
return 0;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::PReluOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}
