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
#include "tpu_mlir/Dialect/Tpu/Transforms/DynCompileCommon.hpp"
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
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice,
    group_type_t group_type) {
  // TODO: supports true3d case
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w, group_type);
  int num = in_nslice; // num = depth * nslice
  const int c_per_npu = ceiling_func(c, BM1684X::NPU_NUM);
  const int EU_NUM = BM1684X::EU_BYTES / 4;
  int mr_size = sizeof(float) * num * c_per_npu * EU_NUM;
  if (group_type == GROUP_NORMAL && getAxis() == 4) {
    mr_size *= in_hslice;
  }
  int tensor_size = sizeof(float) * num * c_per_npu *
                    align_up((int)in_hslice * (int)w, EU_NUM);
  const bool need_mean = !getMean().getType().isa<NoneType>();
  const bool need_rstd = !getRstd().getType().isa<NoneType>();
  bool need_new_mr = false;
  if (group_type == GROUP_NORMAL && getAxis() <= 1) {
    need_new_mr = need_new_mr || c_per_npu > 1;
  }
  int64_t buffer_size = tensor_size;
  if (!need_mean || need_new_mr) {
    buffer_size += mr_size;
  }
  if (!need_rstd || need_new_mr) {
    buffer_size += mr_size;
  }
  return buffer_size;
}

void tpu::LayerNormOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
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
  const auto &gi = getGroupInfo(0, 0);
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
  const auto &gi = getGroupInfo(0, 0);
  param.buffer_addr = gi.buffer_addr;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::LayerNormOp::get_layer_type() {
  return FW_BMNET_LAYER_NORM;
}
