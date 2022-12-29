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

// =========================================
// GlobalGenInterface
// =========================================

void tpu::RequantIntAxisOp::codegen_global_bm1684x() {
  requant_int_param_t param = {0};
  int64_t n, c, h, w;
  module::getNCHW(input(), n, c, h, w);
  param.input_addr = module::getAddress(input());
  param.requant_addr = module::getAddress(quant());
  param.output_addr = module::getAddress(output());
  param.n = (int)n;
  param.c = (int)c;
  param.h = (int)h;
  param.w = (int)w;

  param.is_perchannel = true;
  param.reshaped_coeff = false;
  param.mode = static_cast<int>(quant_mode());
  param.input_dtype = BM168x::getDataType(input());
  param.output_dtype = BM168x::getDataType(output());
  param.round_mode = quant_mode() == tpu::RequantMode::Normal
                         ? ROUNDING_HALF_UP
                         : ROUNDING_HALF_AWAY_FROM_ZERO;
  BM168x::call_global_func("backend_api_requant_int_global", &param,
                           sizeof(param));
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::RequantIntAxisOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  int64_t buffer_size = 0;
  int64_t n, c, h, w;
  module::getNCHW(input(), n, c, h, w);
  auto chip = module::getChip();
  if (quant_mode() == tpu::RequantMode::TFlite_Lshift) {
    buffer_size = in_lmem_bytes;
    buffer_size += ceiling_func(c, BM168x::NPU_NUM) * BM168x::EU_BYTES;
  } else if (quant_mode() == tpu::RequantMode::TFlite) {
    buffer_size = in_lmem_bytes;
  }
  return buffer_size;
}

void tpu::RequantIntAxisOp::assign_sec_info(int64_t n_step, int64_t h_step,
                                            void *sec_info_) {
  local_sec_info_t *sec_info = (local_sec_info_t *)sec_info_;
  memset(sec_info, 0, sizeof(local_sec_info_t));

  int64_t n, c, h, w;
  module::getNCHW(input(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
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

void tpu::RequantIntAxisOp::codegen_local_bm1684x(int64_t n_step,
                                                  int64_t h_step,
                                                  void *sec_info_) {
  local_sec_info_t *sec_info = (local_sec_info_t *)sec_info_;
  int64_t n, c, h, w;
  module::getNCHW(input(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto quant_gi = LocalGenInterface::getGroupInfo(quant(), n_step, h_step);

  requant_int_param_t param = {0};
  param.input_addr = (uint32_t)in_gi.out_addr;
  param.requant_addr = (uint32_t)quant_gi.out_addr;
  param.output_addr = (uint32_t)gi.out_addr;
  param.buffer_local_addr = (uint32_t)gi.buffer_addr;
  param.n = sec_info->n_slice;
  param.c = c;
  param.h = sec_info->h_slice;
  param.w = w;
  param.is_perchannel = true;
  param.reshaped_coeff = false;
  if (module::isUniformQuantized(input())) {
    auto iqtype = module::getUniformQuantizedType(input());
    param.zx_value = iqtype.getZeroPoint();
  }
  param.input_dtype = BM168x::getDataType(input());
  param.output_dtype = BM168x::getDataType(output());
  param.mode = static_cast<int>(quant_mode());
  param.round_mode = quant_mode() == tpu::RequantMode::Normal
                         ? ROUNDING_HALF_UP
                         : ROUNDING_HALF_AWAY_FROM_ZERO;
  BM168x::call_local_func("backend_api_requant_int_local", &param,
                          sizeof(param));
}
