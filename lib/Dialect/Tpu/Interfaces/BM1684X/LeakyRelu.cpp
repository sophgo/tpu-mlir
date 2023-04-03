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
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/DynCompileCommon.hpp"
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::LeakyReluOp::codegen_global_bm1684x() {
  prelu_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.is_channel_shared = true;
  spec.upper_limit = -1;
  spec.round_mode = ROUND_UP;
  if (module::isUniformQuantized(getInput())) {
    spec.slope_val = static_cast<float>(getMultiplier().value());
    spec.rshift_bit = getRshift().value();
  } else {
    spec.slope_val = static_cast<float>(getAlpha().value().convertToDouble());
    spec.rshift_bit = 0;
  }
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_prelu_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::LeakyReluOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_hslice, int64_t out_dslice, int64_t out_wslice,
    group_type_t group_type) {
  return 0;
}

void tpu::LeakyReluOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step, int64_t d_step, int64_t w_step,
                                             group_type_t group_type,
                                             local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);

  prelu_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.is_channel_shared = true;
  spec.upper_limit = -1;
  spec.round_mode = ROUND_UP;
  if (module::isUniformQuantized(getInput())) {
    spec.slope_val = static_cast<float>(getMultiplier().value());
    spec.rshift_bit = getRshift().value();
  } else {
    spec.slope_val = static_cast<float>(getAlpha().value().convertToDouble());
    spec.rshift_bit = 0;
  }

  BM168x::call_local_func("backend_api_prelu_local", &spec, sizeof(spec),
                          &sec_info, input_spec->data(), output_spec->data());
}

// dynamic codegen
int64_t tpu::LeakyReluOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer) return sizeof(prelu_spec_t);
  prelu_spec_t spec = {0};
  spec.is_channel_shared = true;
  spec.upper_limit = -1;
  spec.round_mode = ROUND_UP;
  if (module::isUniformQuantized(getInput())) {
    spec.slope_val = static_cast<float>(getMultiplier().value());
    spec.rshift_bit = getRshift().value();
  } else {
    spec.slope_val = static_cast<float>(getAlpha().value().convertToDouble());
    spec.rshift_bit = 0;
  }
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::LeakyReluOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) return sizeof(prelu_spec_t);
  prelu_spec_t spec = {0};
  spec.is_channel_shared = true;
  spec.upper_limit = -1;
  spec.round_mode = ROUND_UP;
  if (module::isUniformQuantized(getInput())) {
    spec.slope_val = static_cast<float>(getMultiplier().value());
    spec.rshift_bit = getRshift().value();
  } else {
    spec.slope_val = static_cast<float>(getAlpha().value().convertToDouble());
    spec.rshift_bit = 0;
  }
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::LeakyReluOp::get_fw_type_bm1684x() {
  return FW_BMNET_PRELU;
}
