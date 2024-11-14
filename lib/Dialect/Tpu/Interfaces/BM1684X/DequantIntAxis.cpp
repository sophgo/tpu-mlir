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

void tpu::DequantIntAxisOp::codegen_global_bm1684x() {
  dequant_int_param_t param = {0};
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  param.input_addr = module::getAddress(getInput());
  param.dequant_addr = module::getAddress(getQuant());
  param.output_addr = module::getAddress(getOutput());
  param.n = (int)n;
  param.c = (int)c;
  param.h = (int)h;
  param.w = (int)w;

  param.is_perchannel = true;
  param.lshift = getLshift();
  param.mode = static_cast<int>(getQuantMode());
  param.input_dtype = BM168x::getDataType(getInput());
  param.output_dtype = BM168x::getDataType(getOutput());
  param.round_mode = round_mode_convert(getRoundMode());
  BM168x::call_global_func("backend_api_dequant_int_global", &param,
                           sizeof(param));
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::DequantIntAxisOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  if (getQuantMode() == DequantMode::TFLite) {
    return in_lmem_bytes;
  }
  return 0;
}

void tpu::DequantIntAxisOp::codegen_local_bm1684x(
    int64_t n_step, int64_t c_step, int64_t h_step, int64_t d_step,
    int64_t w_step, group_type_t group_type, local_sec_info_t &sec_info) {
  int64_t n, c, d, h, w;
  module::getNCDHW(getInput(), n, c, d, h, w, group_type);
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step,
                                               d_step, w_step, c_step);
  auto dequant_gi = LocalGenInterface::getGroupInfo(getQuant(), n_step, h_step,
                                                    d_step, w_step, c_step);

  dequant_int_param_t param = {0};
  param.input_addr = (uint32_t)in_gi.out_addr;
  param.dequant_addr = (uint32_t)dequant_gi.out_addr;
  param.output_addr = (uint32_t)gi.out_addr;
  param.buffer_local_addr = (uint32_t)gi.buffer_addr;
  param.n = sec_info.out_n_slice * in_gi.d_slice;
  param.c = c;
  param.h = sec_info.out_h_slice;
  param.w = sec_info.out_w_slice;
  param.is_perchannel = true;
  param.input_dtype = BM168x::getDataType(getInput());
  param.output_dtype = BM168x::getDataType(getOutput());
  param.lshift = getLshift();
  param.mode = static_cast<int>(getQuantMode());
  param.round_mode = round_mode_convert(getRoundMode());
  BM168x::call_local_func("backend_api_dequant_int_local", &param,
                          sizeof(param));
}

// dynamic codegen
int64_t tpu::DequantIntAxisOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(dyn_dequant_int_local_spec_t);
  auto gi = getGroupInfo(0, 0, 0, 0, 0);
  auto dequant_gi = LocalGenInterface::getGroupInfo(getQuant(), 0, 0);

  dyn_dequant_int_local_spec_t param = {0};
  param.dequant_addr = (uint32_t)dequant_gi.out_addr;
  param.buffer_local_addr = (uint32_t)gi.buffer_addr;
  param.common.is_perchannel = true;
  param.common.input_dtype = BM168x::getDataType(getInput());
  param.common.output_dtype = BM168x::getDataType(getOutput());
  param.common.lshift = getLshift();
  param.common.mode = static_cast<int>(getQuantMode());
  param.common.round_mode = round_mode_convert(getRoundMode());
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::DequantIntAxisOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(dyn_dequant_int_global_spec_t);
  dyn_dequant_int_global_spec_t param = {0};
  param.common.is_perchannel = true;
  param.common.lshift = getLshift();
  param.common.mode = static_cast<int>(getQuantMode());
  param.common.input_dtype = BM168x::getDataType(getInput());
  param.common.output_dtype = BM168x::getDataType(getOutput());
  param.common.round_mode = round_mode_convert(getRoundMode());
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::DequantIntAxisOp::get_fw_type_bm1684x() {
  return FW_BMNET_DEQUANT_INT;
}
