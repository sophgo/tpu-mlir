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

void tpu::BinaryShiftOp::codegen_global_bm1684x() {
  binaryshift_global_param_t param{0};
  auto &spec = param.spec;
  spec.binary_op = BM168x::binary_mode(getMode());
  spec.rshift_num = -getShift();
  spec.b_is_const = false;
  spec.inversed = getIsReverse();
  spec.round_mode = round_mode_convert(getRoundMode());
  spec.is_saturate = getSaturation();
  param.a_is_coeff = false;
  param.b_is_coeff = false;
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_binary_shift_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::BinaryShiftOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  auto in0_type = module::getStorageType(getInput1());
  auto in1_type = module::getStorageType(getInput2());
  auto out_type = module::getStorageType(getOutput());
  int buffer_dsize;
  bool in_signed = in0_type.isSignedInteger() || in1_type.isSignedInteger() ||
                   getMode() == "Sub";
  if (in_signed && out_type.isUnsignedInteger()) {
    buffer_dsize = out_type.isInteger(8) ? 2 : 4;
  } else if (!in_signed && out_type.isSignedInteger()) {
    buffer_dsize = out_type.isInteger(8) ? 1 : (out_type.isInteger(16) ? 2 : 4);
  } else {
    return 0;
  }
  auto eu_num = BM168x::eu_num(buffer_dsize);
  int64_t buffer_size = ceiling_func(out_cslice, BM168x::NPU_NUM) *
                        align_up(out_hslice * out_dslice * out_wslice, eu_num) *
                        buffer_dsize;
  return buffer_size;
}

void tpu::BinaryShiftOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                               int64_t h_step, int64_t d_step,
                                               int64_t w_step,
                                               group_type_t group_type,
                                               local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  const auto &gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);

  binaryshift_local_param_t param = {0};
  auto &common = param.spec.common;
  common.binary_op = BM168x::binary_mode(getMode());
  common.rshift_num = -getShift();
  common.b_is_const = false;
  common.inversed = getIsReverse();
  common.round_mode = round_mode_convert(getRoundMode());
  common.is_saturate = getSaturation();
  param.spec.buffer = gi.buffer_addr;
  param.a_is_coeff = false;
  param.b_is_coeff = false;
  BM168x::call_local_func("backend_api_binary_shift_local", &param,
                          sizeof(param), &sec_info, input_spec->data(),
                          output_spec->data());
}

// dynamic codegen
int64_t tpu::BinaryShiftOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(binaryshift_local_param_t);
  binaryshift_local_param_t param = {0};
  auto gi = getGroupInfo(0, 0, 0, 0, 0);
  auto &common = param.spec.common;
  common.binary_op = BM168x::binary_mode(getMode());
  common.rshift_num = -getShift();
  common.b_is_const = false;
  common.inversed = getIsReverse();
  common.round_mode = round_mode_convert(getRoundMode());
  common.is_saturate = getSaturation();
  param.spec.buffer = gi.buffer_addr;
  param.a_is_coeff = false;
  param.b_is_coeff = false;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::BinaryShiftOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(binaryshift_global_param_t);
  binaryshift_global_param_t param{0};
  auto &spec = param.spec;
  spec.binary_op = BM168x::binary_mode(getMode());
  spec.rshift_num = -getShift();
  spec.b_is_const = false;
  spec.inversed = getIsReverse();
  spec.round_mode = round_mode_convert(getRoundMode());
  spec.is_saturate = getSaturation();
  param.a_is_coeff = false;
  param.b_is_coeff = false;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::BinaryShiftOp::get_fw_type_bm1684x() {
  return FW_BMNET_BINARY_SHIFT;
}
