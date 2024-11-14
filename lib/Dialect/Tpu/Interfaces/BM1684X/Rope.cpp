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

void tpu::RopeOp::codegen_global_bm1684x() {
  rope_param_t param{0};
  param.mul1_shift = getMul1Shift();
  param.mul2_shift = getMul2Shift();
  param.add_shift = getAddShift();
  param.mul1_round_mode = round_mode_convert(getMul1RoundMode());
  param.mul2_round_mode = round_mode_convert(getMul2RoundMode());
  param.add_round_mode = round_mode_convert(getAddRoundMode());
  param.mul1_saturation = getMul1Saturation();
  param.mul2_saturation = getMul2Saturation();
  param.add_saturation = getAddSaturation();
  param.is_permute_optimize = getIsPermuteOptimize();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_rope_global", &param, sizeof(param),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::RopeOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  auto out_type = module::getStorageType(getOutput());
  if (out_type.isInteger(8)) {
    return 4 * out_lmem_bytes;
  }
  if (out_type.isInteger(16)) {
    return 4 * out_lmem_bytes;
  }
  if (out_type.isInteger(32)) {
    return 2 * out_lmem_bytes;
  } else if (out_type.isBF16() || out_type.isF16() || out_type.isF32()) {
    return 2 * out_lmem_bytes;
  }
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

void tpu::RopeOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                        int64_t h_step, int64_t d_step,
                                        int64_t w_step, group_type_t group_type,
                                        local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  const auto &gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);

  rope_param_t param = {0};
  param.buffer_addr = gi.buffer_addr;
  param.mul1_shift = getMul1Shift();
  param.mul2_shift = getMul2Shift();
  param.add_shift = getAddShift();
  param.mul1_round_mode = round_mode_convert(getMul1RoundMode());
  param.mul2_round_mode = round_mode_convert(getMul2RoundMode());
  param.add_round_mode = round_mode_convert(getAddRoundMode());
  param.mul1_saturation = getMul1Saturation();
  param.mul2_saturation = getMul2Saturation();
  param.add_saturation = getAddSaturation();
  param.is_permute_optimize = getIsPermuteOptimize();

  BM168x::call_local_func("backend_api_rope_local", &param, sizeof(param),
                          &sec_info, input_spec->data(), output_spec->data());
}

// dynamic codegen
int64_t tpu::RopeOp::dyn_codegen_local_bm1684x(void *buffer) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::RopeOp::dyn_codegen_global_bm1684x(void *buffer) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::RopeOp::get_fw_type_bm1684x() {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}