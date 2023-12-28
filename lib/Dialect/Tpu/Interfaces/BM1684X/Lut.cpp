//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::LutOp::codegen_global_bm1684x() {
  lut_param_t p = {0};
  p.input_addr = module::getAddress(getInput());
  p.table_addr = module::getAddress(getTable());
  p.output_addr = module::getAddress(getOutput());
  p.input_dtype = BM168x::getDataType(getInput());
  p.table_dtype = BM168x::getDataType(getTable());
  p.output_dtype = BM168x::getDataType(getOutput());
  p.table_length = 256;
  p.is_local_layer = 0;
  p.shape_dim = 4;
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  p.shape[0] = n;
  p.shape[1] = c;
  p.shape[2] = h;
  p.shape[3] = w;
  BM168x::call_global_func("backend_api_lut", &p, sizeof(p));
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::LutOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  return 0;
}

void tpu::LutOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                       int64_t h_step, int64_t d_step,
                                       int64_t w_step, group_type_t group_type,
                                       local_sec_info_t &sec_info) {
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w, group_type);
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step,
                                               d_step, w_step, c_step);
  auto table_gi = LocalGenInterface::getGroupInfo(getTable(), n_step, h_step,
                                                  d_step, w_step, c_step);

  lut_param_t p = {0};
  p.input_addr = in_gi.out_addr;
  p.table_addr = table_gi.out_addr;
  p.output_addr = gi.out_addr;
  p.input_dtype = BM168x::getDataType(getInput());
  p.table_dtype = BM168x::getDataType(getTable());
  p.output_dtype = BM168x::getDataType(getOutput());
  p.table_length = 256;
  p.is_local_layer = 1;
  p.shape_dim = 4;
  p.shape[0] = sec_info.out_n_slice;
  p.shape[1] = sec_info.c_slice;
  p.shape[2] = sec_info.out_h_slice;
  p.shape[3] = sec_info.out_w_slice;
  BM168x::call_local_func("backend_api_lut", &p, sizeof(p));
}

// dynamic codegen
int64_t tpu::LutOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(dyn_lut_local_param_t);
  dyn_lut_local_param_t param = {0};
  param.common.output_dtype = BM168x::getDataType(getOutput());
  param.common.is_local_layer = 1;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::LutOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(dyn_lut_global_param_t);
  dyn_lut_global_param_t p = {0};
  p.common.output_dtype = BM168x::getDataType(getOutput());
  p.common.is_local_layer = 0;
  return BM168x::dynamic_spec_to_buffer(buffer, p);
}

int64_t tpu::LutOp::get_fw_type_bm1684x() { return FW_BMNET_LUT; }
