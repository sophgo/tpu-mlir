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
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/DynCompileCommon.hpp"
using namespace tpu_mlir::backend;


// =========================================
// GlobalGenInterface
// =========================================

void tpu::LayerNormOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  layer_norm_global_spec_t param = {0};
  const bool have_weight = !getWeight().getType().isa<NoneType>();
  const bool have_bias = !getBias().getType().isa<NoneType>();
  const bool need_mean = !getMean().getType().isa<NoneType>();
  const bool need_rstd = !getRstd().getType().isa<NoneType>();
  param.common.axis = (int)getAxis();
  param.common.eps = getEps().convertToDouble();
  param.common.affine = (have_weight << 0) + (have_bias << 1);
  param.common.need_mean = need_mean;
  param.common.need_rstd = need_rstd;
  BM168x::call_global_func("backend_api_layer_norm_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::LayerNormOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_hslice, int64_t out_dslice, int64_t out_wslice,
    group_type_t group_type) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  layer_norm_local_spec_t param = {0};
  const bool have_weight = !getWeight().getType().isa<NoneType>();
  const bool have_bias = !getBias().getType().isa<NoneType>();
  const bool need_mean = !getMean().getType().isa<NoneType>();
  const bool need_rstd = !getRstd().getType().isa<NoneType>();
  param.common.axis = (int)getAxis();
  if (group_type == GROUP_SMALL_C) {
    auto shape = module::getShape(getInput());
    param.common.axis = 2;
  }
  param.common.eps = getEps().convertToDouble();
  param.common.affine = (have_weight << 0) + (have_bias << 1);
  param.common.need_mean = need_mean;
  param.common.need_rstd = need_rstd;
  local_sec_info_t sec_info;
  memset(&sec_info, 0, sizeof(local_sec_info_t));
  sec_info.group_type = group_type;
  // int64_t n, c, d, h, w, on, oc, od, oh, ow;
  // auto input = op->getOperand(0);
  // auto output = op->getResult(0);
  // module::getNCDHW(input, n, c, d, h, w, group_type);
  // module::getNCDHW(output, on, oc, od, oh, ow, group_type);
  sec_info.n_slice = in_nslice;
  sec_info.d_slice = in_dslice;
  sec_info.h_slice = in_hslice;
  sec_info.w_slice = in_wslice;
  sec_info.out_n_slice = out_nslice;
  sec_info.out_h_slice = out_hslice;
  sec_info.out_w_slice = out_wslice;
  return BM168x::call_local_bfsz_func("backend_api_layer_norm_local_bfsz", &param, sizeof(param),
                                      &sec_info, input_spec->data(), output_spec->data());
}

void tpu::LayerNormOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step, int64_t d_step, int64_t w_step,
                                             group_type_t group_type,
                                             local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  layer_norm_local_spec_t param = {0};
  const bool have_weight = !getWeight().getType().isa<NoneType>();
  const bool have_bias = !getBias().getType().isa<NoneType>();
  const bool need_mean = !getMean().getType().isa<NoneType>();
  const bool need_rstd = !getRstd().getType().isa<NoneType>();
  param.common.axis = (int)getAxis();
  if (group_type == GROUP_SMALL_C) {
    auto shape = module::getShape(getInput());
    param.common.axis = 2;
  }
  param.common.eps = getEps().convertToDouble();
  param.common.affine = (have_weight << 0) + (have_bias << 1);
  param.common.need_mean = need_mean;
  param.common.need_rstd = need_rstd;
  const auto &gi = getGroupInfo(0, 0, 0, 0);
  param.buffer_addr = gi.buffer_addr;
  BM168x::call_local_func("backend_api_layer_norm_local", &param, sizeof(param),
                          &sec_info, input_spec->data(), output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::LayerNormOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) return sizeof(layer_norm_global_spec_t);
  layer_norm_global_spec_t param = {0};
  const bool have_weight = !getWeight().getType().isa<NoneType>();
  const bool have_bias = !getBias().getType().isa<NoneType>();
  const bool need_mean = !getMean().getType().isa<NoneType>();
  const bool need_rstd = !getRstd().getType().isa<NoneType>();
  param.common.axis = (int)getAxis();
  param.common.eps = getEps().convertToDouble();
  param.common.affine = (have_weight << 0) + (have_bias << 1);
  param.common.need_mean = need_mean;
  param.common.need_rstd = need_rstd;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic LocalGenInterface
// ======================================
int64_t tpu::LayerNormOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer) return sizeof(layer_norm_local_spec_t);
  layer_norm_local_spec_t param = {0};
  group_type_t group_type = GROUP_UNSUPPORT;
  auto op = getOperation();
  if (auto gOp = dyn_cast<GroupOp>(op->getParentOp())) {
    group_type = static_cast<group_type_t>(gOp.getGroupType());
  }
  assert(group_type < GROUP_UNSUPPORT);
  const bool have_weight = !getWeight().getType().isa<NoneType>();
  const bool have_bias = !getBias().getType().isa<NoneType>();
  const bool need_mean = !getMean().getType().isa<NoneType>();
  const bool need_rstd = !getRstd().getType().isa<NoneType>();
  param.common.axis = (int)getAxis();
  if (group_type == GROUP_SMALL_C) {
    auto shape = module::getShape(getInput());
    param.common.axis = 2;
  }
  param.common.eps = getEps().convertToDouble();
  param.common.affine = (have_weight << 0) + (have_bias << 1);
  param.common.need_mean = need_mean;
  param.common.need_rstd = need_rstd;
  const auto &gi = getGroupInfo(0, 0, 0, 0);
  param.buffer_addr = gi.buffer_addr;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::LayerNormOp::get_fw_type_bm1684x() {
  return FW_BMNET_LAYER_NORM;
}
