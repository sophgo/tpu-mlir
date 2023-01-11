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

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

typedef struct {
  uint64_t input_addr;
  uint64_t output_addr;
  uint64_t requant_addr;
  uint32_t buffer_local_addr;
  int n;
  int c;
  int h;
  int w;
  bool is_perchannel;
  float scale_value;
  float offset_value;
  int input_dtype;
  int output_dtype;
  int mode;
  int round_mode;
} requant_fp_param_t;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::RequantFpOp::codegen_global_bm1684x() {
  requant_fp_param_t param = {0};
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  param.input_addr = module::getAddress(getInput());
  param.output_addr = module::getAddress(getOutput());
  param.n = (int)n;
  param.c = (int)c;
  param.h = (int)h;
  param.w = (int)w;

  auto oqtype = module::getUniformQuantizedType(getOutput());
  param.scale_value = getScale().convertToDouble();
  param.offset_value = oqtype.getZeroPoint();
  param.mode = static_cast<int>(getQuantMode()) / 2;
  param.input_dtype = BM168x::getDataType(getInput());
  param.output_dtype = BM168x::getDataType(getOutput());
  param.round_mode = ROUNDING_HALF_AWAY_FROM_ZERO;
  BM168x::call_global_func("backend_api_requant_float_global", &param,
                           sizeof(param));
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::RequantFpOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  int64_t buffer_size = 0;
  if (getQuantMode() != RequantMode::Normal) {
    buffer_size = in_lmem_bytes;
  }
  return buffer_size;
}

void tpu::RequantFpOp::assign_sec_info(int64_t n_step, int64_t h_step,
                                       void *sec_info_) {
  local_sec_info_t *sec_info = (local_sec_info_t *)sec_info_;
  memset(sec_info, 0, sizeof(local_sec_info_t));

  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  sec_info->n_slice = in_gi.n_slice;
  sec_info->d_slice = 1;
  sec_info->h_slice = in_gi.h_slice;
  sec_info->h_idx = in_gi.h_idx;
  sec_info->is_h_split = !(in_gi.h_idx == 0 && in_gi.h_slice == h);
  sec_info->w_slice = w;
  sec_info->out_n_slice = gi.n_slice;
  sec_info->out_h_idx = gi.h_idx;
  sec_info->out_h_slice = gi.h_slice;
  sec_info->out_w_slice = w;
}

void tpu::RequantFpOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                             void *sec_info_) {
  local_sec_info_t *sec_info = (local_sec_info_t *)sec_info_;
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);

  requant_fp_param_t param = {0};
  param.input_addr = (uint32_t)in_gi.out_addr;
  param.output_addr = (uint32_t)gi.out_addr;
  param.buffer_local_addr = (uint32_t)gi.buffer_addr;
  param.n = sec_info->out_n_slice;
  param.c = c;
  param.h = sec_info->out_h_slice;
  param.w = w;

  auto oqtype = module::getUniformQuantizedType(getOutput());
  param.scale_value = getScale().convertToDouble();
  param.offset_value = oqtype.getZeroPoint();
  param.input_dtype = BM168x::getDataType(getInput());
  param.output_dtype = BM168x::getDataType(getOutput());
  param.mode = static_cast<int>(getQuantMode()) / 2;
  param.round_mode = ROUNDING_HALF_AWAY_FROM_ZERO;
  BM168x::call_local_func("backend_api_requant_float_local", &param,
                          sizeof(param));
}

//dynamic codegen
int64_t tpu::RequantFpOp::dyn_codegen_local_bm1684x(void *buffer) {
return 0;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::RequantFpOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}
