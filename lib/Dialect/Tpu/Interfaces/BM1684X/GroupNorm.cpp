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

using namespace tpu_mlir::backend;


// =========================================
// GlobalGenInterface
// =========================================

void tpu::GroupNormOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  group_norm_global_param_t param = {0};
  const bool have_weight = !module::isNone(getWeight());
  const bool have_bias = !module::isNone(getBias());
  param.common.group_num = (int)getNumGroups();
  param.common.eps = getEps().convertToDouble();
  param.common.affine = (have_weight << 0) + (have_bias << 1);
  BM168x::call_global_func("backend_api_layer_norm_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::GroupNormOp::getBufferSize_bm1684x(
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
  int tensor_size = sizeof(float) * num * c_per_npu *
                    align_up((int)in_hslice * (int)w, EU_NUM);
  bool need_new_mr = false;
  if (group_type == GROUP_NORMAL) {
    need_new_mr = need_new_mr || c_per_npu > 1;
  }
  int64_t buffer_size = tensor_size;
  if (need_new_mr) {
    buffer_size += 2 * mr_size;
  }
  return buffer_size;
}

void tpu::GroupNormOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                             group_type_t group_type,
                                             local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  group_norm_local_param_t param = {0};
  const bool have_weight = !module::isNone(getWeight());
  const bool have_bias = !module::isNone(getBias());
  param.common.group_num = (int)getNumGroups();
  param.common.eps = getEps().convertToDouble();
  param.common.affine = (have_weight << 0) + (have_bias << 1);
  const auto &gi = getGroupInfo(0, 0);
  param.buffer_addr = gi.buffer_addr;
  BM168x::call_local_func("backend_api_layer_norm_local", &param, sizeof(param),
                          &sec_info, input_spec->data(), output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::GroupNormOp::dyn_codegen_global_bm1684x(void *buffer) { return 0; }

// ======================================
// Dynamic LocalGenInterface
// ======================================
int64_t tpu::GroupNormOp::dyn_codegen_local_bm1684x(void *buffer) { return 0; }
