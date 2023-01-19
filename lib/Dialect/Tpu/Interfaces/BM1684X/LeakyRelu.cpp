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

typedef struct {
  uint64_t input_addr;
  uint64_t slope_addr;
  uint64_t output_addr;
  int32_t input_n;
  int32_t input_c;
  int32_t input_h;
  int32_t input_w;
  int32_t channel_shared;
  float slope_val;
  int32_t rshift_bit;
  float relu_limit;
  DATA_TYPE_T dtype;
} leakyrelu_param_t;

typedef struct {
  float upper_limit;
  float slope_val;
  int is_channel_shared;
  int rshift_bit;
  int round_mode;
} prelu_spec_t;
#ifdef __cplusplus
}
#endif

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
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::LeakyReluOp::assign_sec_info(int64_t n_step, int64_t h_step,
                                       local_sec_info_t &sec_info) {
  memset(&sec_info, 0, sizeof(local_sec_info_t));

  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  sec_info.n_slice = in_gi.n_slice;
  sec_info.d_slice = 1;
  sec_info.h_slice = in_gi.h_slice;
  sec_info.h_idx = in_gi.h_idx;
  sec_info.is_h_split = !(in_gi.h_idx == 0 && in_gi.h_slice == h);
  sec_info.w_slice = w;
  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_idx = gi.h_idx;
  sec_info.out_h_slice = gi.h_slice;
  sec_info.out_w_slice = w;
}

void tpu::LeakyReluOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                             local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

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

//dynamic codegen
int64_t tpu::LeakyReluOp::dyn_codegen_local_bm1684x(void *buffer) {
return 0;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::LeakyReluOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}
