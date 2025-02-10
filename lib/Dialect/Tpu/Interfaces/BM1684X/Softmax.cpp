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
      UNREACHABLE_THIS("Not Implemented");
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
#if 1
    if (support_multi_core()) {
      BM168x::call_global_func("backend_api_softmax_multicore_global", &param,
                               sizeof(param), input_spec->data(),
                               output_spec->data());
      return;
    }
    BM168x::call_global_func("backend_api_softmax_global", &param,
                             sizeof(param), input_spec->data(),
                             output_spec->data());
#else
    BM168x::call_ppl_global_func("api_softmax_global", &param, sizeof(param),
                                 input_spec->data(), output_spec->data());
#endif
  }
}

// =========================================
// LocalGenInterface
// =========================================
int64_t tpu::SoftmaxOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  int64_t N, C, H, W;
  module::getNCHW(getInput(), N, C, H, W, group_type);

  int64_t buffer_size = 0;
  int c_per_npu = ceiling_func(in_cslice, BM168x::NPU_NUM);
  auto in_type = BM168x::getDataType(getInput());
  auto in_type_len = BM168x::getFmtBytes(in_type);
  auto eu_num = BM168x::eu_num(in_type_len);
  auto stype = module::getStorageType(getInput().getType());
  int64_t axis = group_type == GROUP_SMALL_C ? 2 : getAxis();
  int32_t padding_flag = 0;

  // aligned with backend
  if (axis == 3 && !getLog() && in_wslice > eu_num && in_wslice % eu_num > 0) {
    in_wslice = align_up(in_wslice, eu_num);
    padding_flag = 1;
  }

#define SIZE                                                                   \
  ((stype.isF16() || stype.isBF16()) ? sizeof(int16_t) : sizeof(float))
  if (axis == 2) {
    buffer_size += c_per_npu * align_up(in_wslice, eu_num) * SIZE;
  } else if (axis == 3) {
    buffer_size += c_per_npu * align_up(in_hslice, eu_num) * SIZE;
  }
  // 32 coeff and 192 table
  buffer_size += align_up((int64_t)32, eu_num) * sizeof(float);
  buffer_size += align_up((int64_t)192, eu_num) * sizeof(float);
  buffer_size +=
      c_per_npu * align_up(in_hslice * in_wslice, eu_num) * sizeof(float) * 2;
  if (getLog()) {
    buffer_size += c_per_npu * align_up(in_hslice * in_wslice, eu_num) * SIZE;
  }

  if (padding_flag) {
    buffer_size += c_per_npu * align_up(in_hslice * in_wslice, eu_num) * SIZE;
  }

  return buffer_size;
}

void tpu::SoftmaxOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                           int64_t h_step, int64_t d_step,
                                           int64_t w_step,
                                           group_type_t group_type,
                                           local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type, n_step, h_step,
                                           d_step, w_step, c_step);
  auto output_spec = BM168x::get_output_spec(op, group_type, n_step, h_step,
                                             d_step, w_step, c_step);
  const auto &gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);

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
int64_t tpu::SoftmaxOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return (module::isUniformQuantized(getInput(), getOutput())
                ? sizeof(softmax_tflite_fix8b_param_t)
                : sizeof(softmax_global_param_t));
  bool has_table = !getTable().getType().isa<NoneType>();
  float in_scale = 1.0;
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    in_scale = in_qtype.getScale();
  }
  if (module::isUniformQuantized(getInput(), getOutput())) {
    if (getLog()) {
      UNREACHABLE_THIS("Not Implemented");
    }
    assert(has_table);
    auto out_qtype = module::getUniformQuantizedType(getOutput());
    softmax_tflite_fix8b_param_t param = {0};
    auto &common = param.common;
    common.begin_axis = getAxis();
    common.end_axis = getAxis();
    common.zero_point = out_qtype.getZeroPoint();
    common.scale_val = out_qtype.getScale();
    return BM168x::dynamic_spec_to_buffer(buffer, param);
  } else {
    softmax_global_param_t param = {0};
    auto &common = param.common;
    common.begin_axis = getAxis();
    common.end_axis = getAxis();
    common.scale_val = in_scale;
    common.log = getLog();
    return BM168x::dynamic_spec_to_buffer(buffer, param);
  }
}

int64_t tpu::SoftmaxOp::get_fw_type_bm1684x() { return FW_BMNET_SOFTMAX; }

// ======================================
// Dynamic LocalGenInterface
// ======================================
int64_t tpu::SoftmaxOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(softmax_local_param_t);
  softmax_local_param_t param{0};
  auto gi = getGroupInfo(0, 0, 0, 0, 0);
  param.buffer_addr = gi.buffer_addr;
  auto &common = param.common;
  common.zero_point = 0.f;
  common.scale_val = 1.f;
  common.begin_axis = getAxis();
  common.end_axis = getAxis();
  common.scale_val = 1.f;
  common.log = getLog();
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}
