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

void tpu::PixelNormOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  const bool have_weight = !getWeight().getType().isa<NoneType>();
  const bool have_bias = !getBias().getType().isa<NoneType>();
  pixel_norm_global_spec_t param = {0};
  param.common.eps = getEps().convertToDouble();
  param.common.affine = (have_weight << 0) + (have_bias << 1);
  param.common.scale = 1.0f;
  if (module::isUniformQuantized(getInput())) {
    auto qtype = module::getUniformQuantizedType(getInput());
    param.common.scale = qtype.getScale();
  }
  BM168x::call_global_func("backend_api_pixel_norm_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::PixelNormOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  pixel_norm_local_spec_t param = {0};
  const bool have_weight = !getWeight().getType().isa<NoneType>();
  const bool have_bias = !getBias().getType().isa<NoneType>();
  param.common.eps = getEps().convertToDouble();
  param.common.affine = (have_weight << 0) + (have_bias << 1);
  if (module::isUniformQuantized(getInput())) {
    auto qtype = module::getUniformQuantizedType(getInput());
    param.common.scale = qtype.getScale();
  }
  local_sec_info_t sec_info;
  memset(&sec_info, 0, sizeof(local_sec_info_t));
  sec_info.group_type = group_type;
  // int64_t n, c, d, h, w, on, oc, od, oh, ow;
  // auto input = op->getOperand(0);
  // auto output = op->getResult(0);
  // module::getNCDHW(input, n, c, d, h, w, group_type);
  // module::getNCDHW(output, on, oc, od, oh, ow, group_type);
  sec_info.n_slice = in_nslice;
  sec_info.c_slice = in_cslice;
  sec_info.d_slice = in_dslice;
  sec_info.h_slice = in_hslice;
  sec_info.w_slice = in_wslice;
  sec_info.out_n_slice = out_nslice;
  sec_info.out_h_slice = out_hslice;
  sec_info.out_w_slice = out_wslice;
  return BM168x::call_local_bfsz_func("backend_api_pixel_norm_local_bfsz",
                                      &param, sizeof(param), &sec_info,
                                      input_spec->data(), output_spec->data());
}

void tpu::PixelNormOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                             int64_t h_step, int64_t d_step,
                                             int64_t w_step,
                                             group_type_t group_type,
                                             local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  pixel_norm_local_spec_t param = {0};
  const bool have_weight = !getWeight().getType().isa<NoneType>();
  const bool have_bias = !getBias().getType().isa<NoneType>();
  param.common.eps = getEps().convertToDouble();
  param.common.affine = (have_weight << 0) + (have_bias << 1);
  if (module::isUniformQuantized(getInput())) {
    auto qtype = module::getUniformQuantizedType(getInput());
    param.common.scale = qtype.getScale();
  }
  const auto &gi = getGroupInfo(0, 0, 0, 0, 0);
  param.buffer_addr = gi.buffer_addr;
  BM168x::call_local_func("backend_api_pixel_norm_local", &param, sizeof(param),
                          &sec_info, input_spec->data(), output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::PixelNormOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(pixel_norm_global_spec_t);
  const bool have_weight = !getWeight().getType().isa<NoneType>();
  const bool have_bias = !getBias().getType().isa<NoneType>();
  pixel_norm_global_spec_t param = {0};
  param.common.eps = getEps().convertToDouble();
  param.common.affine = (have_weight << 0) + (have_bias << 1);
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic LocalGenInterface
// ======================================
int64_t tpu::PixelNormOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(pixel_norm_local_spec_t);
  pixel_norm_local_spec_t param = {0};
  const bool have_weight = !getWeight().getType().isa<NoneType>();
  const bool have_bias = !getBias().getType().isa<NoneType>();
  param.common.eps = getEps().convertToDouble();
  param.common.affine = (have_weight << 0) + (have_bias << 1);
  const auto &gi = getGroupInfo(0, 0, 0, 0, 0);
  param.buffer_addr = gi.buffer_addr;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::PixelNormOp::get_fw_type_bm1684x() { return FW_BMNET_PIXEL_NORM; }
