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
// GloballGenInterface
// =========================================
void tpu::SoftmaxOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  bool has_table = !getTable().getType().isa<NoneType>();
  float in_scale = 1.0;
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    in_scale = in_qtype.getScale();
  }
  if (module::isUniformQuantized(getInput(), getOutput())) {
    if (getLog()) {
      llvm_unreachable("Not Implemented");
      return;
    }
    assert(has_table);
    auto out_qtype = module::getUniformQuantizedType(getOutput());
    softmax_tflite_fix8b_param_t param = {0};
    auto &common = param.common;
    common.begin_axis = getAxis();
    common.end_axis = getAxis();
    common.zero_point = out_qtype.getZeroPoint();
    common.scale_val = out_qtype.getScale();
    BM168x::call_global_func("backend_api_softmax_tflite_fix8b_global", &param,
                             sizeof(param), input_spec->data(),
                             output_spec->data());
  } else {
    softmax_global_param_t param = {0};
    auto &common = param.common;
    common.begin_axis = getAxis();
    common.end_axis = getAxis();
    common.scale_val = in_scale;
    common.log = getLog();
    BM168x::call_global_func("backend_api_softmax_global", &param,
                             sizeof(param), input_spec->data(),
                             output_spec->data());
  }
}

// =========================================
// LocalGenInterface
// =========================================
int64_t tpu::SoftmaxOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice,
    group_type_t group_type) {
  int64_t N, C, H, W;
  module::getNCHW(getInput(), N, C, H, W, group_type);

  int64_t buffer_size = 0;
  int c_per_npu = ceiling_func(C, BM168x::NPU_NUM);
  auto eu_num = BM168x::eu_num(sizeof(float));
  int64_t axis = group_type == GROUP_SMALL_C ? 2 : getAxis();
  if (axis == 2) {
    buffer_size += c_per_npu * align_up(W, eu_num) * sizeof(float);
  } else if (axis == 3) {
    buffer_size += c_per_npu * align_up(in_hslice, eu_num) * sizeof(float);
  }
  // 32 coeff and 192 table
  buffer_size += align_up((int64_t)32, eu_num) * sizeof(float);
  buffer_size += align_up((int64_t)192, eu_num) * sizeof(float);
  buffer_size +=
      c_per_npu * align_up(in_hslice * W, eu_num) * sizeof(float) * 2;
  if (getLog()) {
    buffer_size += c_per_npu * align_up(in_hslice * W, eu_num) * sizeof(float);
  }

  return buffer_size;
}

void tpu::SoftmaxOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                           group_type_t group_type,
                                           local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  const auto &gi = getGroupInfo(n_step, h_step);

  float in_scale = 1.0;
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    in_scale = in_qtype.getScale();
  }

  softmax_local_param_t param = {0};
  auto &common = param.common;
  param.buffer_addr = gi.buffer_addr;
  common.begin_axis = getAxis();
  common.end_axis = getAxis();
  if (group_type == GROUP_SMALL_C) {
    common.begin_axis = 2;
    common.end_axis = 2;
  }
  common.scale_val = in_scale;
  common.log = getLog();

  BM168x::call_local_func("backend_api_softmax_local", &param, sizeof(param),
                          &sec_info, input_spec->data(), output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::SoftmaxOp::dyn_codegen_global_bm1684x(void *buffer) { return 0; }
