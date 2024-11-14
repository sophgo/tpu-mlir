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
#include <stdio.h>

using namespace tpu_mlir::backend;

void tpu::MeanStdScaleOp::codegen_global_bm1684x() {
  auto std = module::getF64Array(getStd());
  auto scale = module::getF64Array(getScale());
  auto mean = module::getF64Array(getMean());
  auto zero_points = module::getF64Array(getZeroPoints());
  auto round_mode =
      round_mode_convert(symbolizeRoundMode(getRoundingMode()).value());
  auto rshift = module::getI32Array(getRshift());
  auto offset = module::getI32Array(getOffset());
  auto multi = module::getI32Array(getMulti());
  std::vector<int64_t> in_shape = module::getShape(getInput());

  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  mean_std_scale_param_t param = {0};
  param.num_of_chn = in_shape[1];
  memcpy(param.std, std->data(), sizeof(param.std));
  memcpy(param.mean, mean->data(), sizeof(param.mean));
  memcpy(param.scale, scale->data(), sizeof(param.scale));
  param.in_zp = zero_points->at(0);
  param.out_zp = zero_points->at(1);
  memcpy(param.multi, multi->data(), sizeof(param.multi));
  memcpy(param.rshift, rshift->data(), sizeof(param.rshift));
  memcpy(param.offset, offset->data(), sizeof(param.offset));
  param.round_mode = round_mode;

  BM168x::call_global_func("backend_api_mean_std_scale_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

int64_t tpu::MeanStdScaleOp::dyn_codegen_global_bm1684x(void *buffer) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::MeanStdScaleOp::get_fw_type_bm1684x() { return FW_LAYER_UNKNOWN; }

void tpu::MeanStdScaleOp::codegen_global_bm1684() {}

uint32_t tpu::MeanStdScaleOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}

int64_t tpu::MeanStdScaleOp::get_fw_type_bm1684() { return FW_LAYER_UNKNOWN; }

void tpu::MeanStdScaleOp::codegen_global_cv18xx(int64_t layer_id) {}

void tpu::MeanStdScaleOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                                int64_t h_step, int64_t d_step,
                                                int64_t w_step,
                                                group_type_t group_type,
                                                local_sec_info_t &sec_info) {
  auto zero_points = module::getF64Array(getZeroPoints());
  auto round_mode =
      round_mode_convert(symbolizeRoundMode(getRoundingMode()).value());

  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  auto f32_param_spec = LocalGenInterface::getGroupInfo(
      getF32Param(), n_step, h_step, d_step, w_step, c_step);
  auto grp_info = getGroupInfo(n_step, h_step, d_step, w_step, c_step);

  mean_std_scale_local_param_t param{0};
  param.buffer_addr = (uint32_t)grp_info.buffer_addr;
  param.f32_param_addr = (uint32_t)f32_param_spec.out_addr;
  param.in_zp = (int32_t)zero_points->at(0);
  param.out_zp = (int32_t)zero_points->at(1);
  param.round_mode = (int32_t)round_mode;

  BM168x::call_local_func("backend_api_mean_std_scale_local", &param,
                          sizeof(param), &sec_info, input_spec->data(),
                          output_spec->data());
}

int64_t tpu::MeanStdScaleOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  auto intype = module::getStorageType(getInput());
  int64_t in_dtype_len = intype.getIntOrFloatBitWidth() / 8;
  int64_t tensor_size = in_lmem_bytes / in_nslice;
  int64_t buffer_size = (tensor_size / in_dtype_len) * 9;
  buffer_size += align_up(5 * 4, Arch::EU_BYTES);
  return buffer_size;
}

void tpu::MeanStdScaleOp::codegen_local_bm1684(long, long, local_sec_info_t &) {
  llvm_unreachable("Not Implemented");
}

void tpu::MeanStdScaleOp::codegen_local_cv18xx(long, long, long, long,
                                               tpu_mlir::group_type_t,
                                               tpu_mlir::local_sec_info &,
                                               long) {
  llvm_unreachable("Not Implemented");
}

int64_t tpu::MeanStdScaleOp::getBufferSize_bm1684(long, long, long, long, long,
                                                  long) {
  return 0;
}

int64_t tpu::MeanStdScaleOp::getBufferSize_cv18xx(long, long, long, long, long,
                                                  long) {
  return 0;
}
