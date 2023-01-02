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

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================

void tpu::MinOp::codegen_global_bm1684x() {
  bcbinary_common_spec_t param{0};
  param.binary_type = BINARY_MIN;
  param.if_relu = 0;
  param.relu_upper_limit = -1.0f;
  param.rshift_A = 0;
  param.rshift_B = 0;
  param.scale_A = 1;
  param.scale_B = 1;
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_bcbinary_global", &param, sizeof(param),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::MinOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  int64_t buffer_size = 0;
  return buffer_size;
}

void tpu::MinOp::assign_sec_info(int64_t n_step, int64_t h_step,
                                 void *sec_info_) {
  local_sec_info_t *sec_info = (local_sec_info_t *)sec_info_;
  memset(sec_info, 0, sizeof(local_sec_info_t));

  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step);
  auto in0_gi = LocalGenInterface::getGroupInfo(getInputs()[0], n_step, h_step);
  auto in1_gi = LocalGenInterface::getGroupInfo(getInputs()[1], n_step, h_step);
  sec_info->n_slice = gi.n_slice;
  sec_info->h_slice = in0_gi.h_slice;
  sec_info->w_slice = w;
  sec_info->out_n_slice = gi.n_slice;
  sec_info->is_h_split = !(gi.h_idx == 0 && gi.h_slice == h);
  sec_info->h_idx = in0_gi.h_idx;
  sec_info->out_h_idx = gi.h_idx;
  sec_info->out_h_slice = gi.h_slice;
  sec_info->is_w_split = false;
  sec_info->out_w_slice = w;
}

void tpu::MinOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                       void *sec_info_) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  auto gi = getGroupInfo(n_step, h_step);

  bcbinary_local_param_t param = {0};
  param.spec.common.binary_type = BINARY_MIN;
  param.spec.common.if_relu = 0;
  param.spec.common.relu_upper_limit = -1.0f;
  param.spec.common.rshift_A = 0;
  param.spec.common.rshift_B = 0;
  param.spec.common.scale_A = 1;
  param.spec.common.scale_B = 1;
  param.spec.buffer_addr = gi.buffer_addr;
  param.A_is_coeff = false;
  param.B_is_coeff = false;

  BM168x::call_local_func("backend_api_bcbinary_local", &param, sizeof(param),
                          sec_info_, input_spec->data(), output_spec->data());
}
