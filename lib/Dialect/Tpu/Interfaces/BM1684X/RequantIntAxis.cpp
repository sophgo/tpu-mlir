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

void tpu::RequantIntAxisOp::codegen_global_bm1684x() {
  requant_int_param_t param = {0};
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  param.input_addr = module::getAddress(getInput());
  param.requant_addr = module::getAddress(getQuant());
  param.output_addr = module::getAddress(getOutput());
  param.n = (int)n;
  param.c = (int)c;
  param.h = (int)h;
  param.w = (int)w;

  param.is_perchannel = true;
  param.reshaped_coeff = false;
  param.mode = static_cast<int>(getQuantMode());
  param.input_dtype = BM168x::getDataType(getInput());
  param.output_dtype = BM168x::getDataType(getOutput());
  param.round_mode = round_mode_convert(getRoundMode());
  BM168x::call_global_func("backend_api_requant_int_global", &param,
                           sizeof(param));
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::RequantIntAxisOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  int64_t buffer_size = 0;
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  if (getQuantMode() == tpu::RequantMode::TFLite_LShift) {
    buffer_size = in_lmem_bytes;
    buffer_size += ceiling_func(c, BM168x::NPU_NUM) * BM168x::EU_BYTES;
  } else if (getQuantMode() == tpu::RequantMode::TFLite) {
    buffer_size = in_lmem_bytes;
  }
  return buffer_size;
}

void tpu::RequantIntAxisOp::codegen_local_bm1684x(
    int64_t n_step, int64_t c_step, int64_t h_step, int64_t d_step,
    int64_t w_step, group_type_t group_type, local_sec_info_t &sec_info) {
  int64_t n, c, d, h, w;
  module::getNCDHW(getInput(), n, c, d, h, w, group_type);
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step,
                                               d_step, w_step, c_step);
  auto quant_gi = LocalGenInterface::getGroupInfo(getQuant(), n_step, h_step,
                                                  d_step, w_step, c_step);

  requant_int_param_t param = {0};
  param.input_addr = (uint32_t)in_gi.out_addr;
  param.requant_addr = (uint32_t)quant_gi.out_addr;
  param.output_addr = (uint32_t)gi.out_addr;
  param.buffer_local_addr = (uint32_t)gi.buffer_addr;
  param.n = sec_info.n_slice * in_gi.d_slice;
  param.c = c;
  param.h = sec_info.h_slice;
  param.w = sec_info.w_slice;
  param.is_perchannel = true;
  param.reshaped_coeff = false;
  if (module::isUniformQuantized(getInput())) {
    auto iqtype = module::getUniformQuantizedType(getInput());
    param.zx_value = iqtype.getZeroPoint();
  }
  param.input_dtype = BM168x::getDataType(getInput());
  param.output_dtype = BM168x::getDataType(getOutput());
  param.mode = static_cast<int>(getQuantMode());
  param.round_mode = round_mode_convert(getRoundMode());
  BM168x::call_local_func("backend_api_requant_int_local", &param,
                          sizeof(param));
}

// dynamic codegen
int64_t tpu::RequantIntAxisOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(dyn_requant_int_local_param_t);
  auto gi = getGroupInfo(0, 0, 0, 0, 0);

  dyn_requant_int_local_param_t param = {0};
  param.buffer_local_addr = (uint32_t)gi.buffer_addr;
  param.common.is_perchannel = true;
  param.common.reshaped_coeff = false;
  if (module::isUniformQuantized(getInput())) {
    auto iqtype = module::getUniformQuantizedType(getInput());
    param.common.zx_value = iqtype.getZeroPoint();
  }
  param.common.output_dtype = BM168x::getDataType(getOutput());
  param.common.mode = static_cast<int>(getQuantMode());
  param.common.round_mode = round_mode_convert(getRoundMode());
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::RequantIntAxisOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(dyn_requant_int_global_param_t);
  dyn_requant_int_global_param_t param = {0};

  param.common.is_perchannel = true;
  param.common.reshaped_coeff = false;
  param.common.mode = static_cast<int>(getQuantMode());
  param.common.output_dtype = BM168x::getDataType(getOutput());
  param.common.round_mode = round_mode_convert(getRoundMode());
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::RequantIntAxisOp::get_fw_type_bm1684x() {
  return FW_BMNET_REQUANT_INT;
}
