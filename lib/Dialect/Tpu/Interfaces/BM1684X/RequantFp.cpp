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
  param.scale_value = getScale().convertToDouble();
  param.offset_value = getOffset().convertToDouble();
  // if (module::getStorageType(getOutput()).isFloat8E4M3FN() ||
  // module::getStorageType(getOutput()).isFloat8E5M2()) {
  //   param.offset_value = 0.0;
  // } else {
  //   auto oqtype = module::getUniformQuantizedType(getOutput());
  //   param.offset_value = oqtype.getZeroPoint();
  // }
  if (getQuantMode() == RequantMode::MultiplierShift) {
    param.mode = 1;
  }
  param.input_dtype = BM168x::getDataType(getInput());
  param.output_dtype = BM168x::getDataType(getOutput());
  param.round_mode = round_mode_convert(getRoundMode());
  param.src_round_mode = round_mode_convert(getFirstRoundMode());
  BM168x::call_global_func("backend_api_requant_float_global", &param,
                           sizeof(param));
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::RequantFpOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  int64_t buffer_size = 0;
  if (getQuantMode() != RequantMode::MultiplierShift) {
    buffer_size = in_lmem_bytes;
  }
  return buffer_size;
}

void tpu::RequantFpOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                             int64_t h_step, int64_t d_step,
                                             int64_t w_step,
                                             group_type_t group_type,
                                             local_sec_info_t &sec_info) {
  int64_t n, c, d, h, w;
  module::getNCDHW(getInput(), n, c, d, h, w, group_type);
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step,
                                               d_step, w_step, c_step);
  requant_fp_param_t param = {0};
  param.input_addr = (uint32_t)in_gi.out_addr;
  param.output_addr = (uint32_t)gi.out_addr;
  param.buffer_local_addr = (uint32_t)gi.buffer_addr;
  param.n = sec_info.out_n_slice * in_gi.d_slice;
  param.c = c;
  param.h = sec_info.out_h_slice; // to do for int4  split
  param.w = sec_info.out_w_slice;

  param.scale_value = getScale().convertToDouble();
  param.offset_value = getOffset().convertToDouble();
  // if (module::getStorageType(getOutput()).isFloat8E4M3FN() ||
  // module::getStorageType(getOutput()).isFloat8E5M2()) {
  //   param.offset_value = 0.0f;
  // } else {
  //   auto oqtype = module::getUniformQuantizedType(getOutput());
  //   param.offset_value = oqtype.getZeroPoint();
  // }
  param.input_dtype = BM168x::getDataType(getInput());
  param.output_dtype = BM168x::getDataType(getOutput());
  if (getQuantMode() == RequantMode::MultiplierShift) {
    param.mode = 1;
  }
  param.round_mode = round_mode_convert(getRoundMode());
  param.src_round_mode = round_mode_convert(getFirstRoundMode());
  BM168x::call_local_func("backend_api_requant_float_local", &param,
                          sizeof(param));
}

// dynamic codegen
int64_t tpu::RequantFpOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(requant_fp_param_t);
  auto gi = getGroupInfo(0, 0, 0, 0, 0);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), 0, 0);
  requant_fp_param_t param = {0};
  param.input_addr = (uint32_t)in_gi.out_addr;
  param.output_addr = (uint32_t)gi.out_addr;
  param.buffer_local_addr = (uint32_t)gi.buffer_addr;
  param.scale_value = getScale().convertToDouble();
  param.offset_value = getOffset().convertToDouble();
  // if (module::getStorageType(getOutput()).isFloat8E4M3FN() ||
  // module::getStorageType(getOutput()).isFloat8E5M2()) {
  //   param.offset_value = 0.0f;
  // } else {
  //   auto oqtype = module::getUniformQuantizedType(getOutput());
  //   param.offset_value = oqtype.getZeroPoint();
  // }
  param.input_dtype = BM168x::getDataType(getInput());
  param.output_dtype = BM168x::getDataType(getOutput());
  if (getQuantMode() == RequantMode::MultiplierShift) {
    param.mode = 1;
  }
  param.round_mode = round_mode_convert(getRoundMode());
  param.src_round_mode = round_mode_convert(getFirstRoundMode());
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::RequantFpOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(requant_fp_param_t);
  requant_fp_param_t param = {0};
  param.input_addr = module::getAddress(getInput());
  param.output_addr = module::getAddress(getOutput());
  param.scale_value = getScale().convertToDouble();
  param.offset_value = getOffset().convertToDouble();
  // if (module::getStorageType(getOutput()).isFloat8E4M3FN() ||
  // module::getStorageType(getOutput()).isFloat8E5M2()) {
  //   param.offset_value = 0.0f;
  // } else {
  //   auto oqtype = module::getUniformQuantizedType(getOutput());
  //   param.offset_value = oqtype.getZeroPoint();
  // }
  if (getQuantMode() == RequantMode::MultiplierShift) {
    param.mode = 1;
  }
  param.input_dtype = BM168x::getDataType(getInput());
  param.output_dtype = BM168x::getDataType(getOutput());
  param.round_mode = round_mode_convert(getRoundMode());
  param.src_round_mode = round_mode_convert(getFirstRoundMode());
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::RequantFpOp::get_fw_type_bm1684x() {
  return FW_BMNET_REQUANT_FP32;
}
