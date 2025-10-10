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

void tpu::BinaryConstShiftOp::codegen_global_bm1684x() {
  binaryshift_global_param_t param{0};
  auto &spec = param.spec;
  spec.binary_op = BM168x::binary_mode(getMode());
  spec.rshift_num = -getShift();
  spec.b_is_const = true;
  spec.b_const_val = getScale();
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

int64_t tpu::BinaryConstShiftOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type,
    bool with_hw_margins) {
  auto in_type = module::getStorageType(getInput());
  auto out_type = module::getStorageType(getOutput());
  int buffer_dsize;
  if (in_type.isSignedInteger() && out_type.isUnsignedInteger()) {
    buffer_dsize = out_type.isInteger(8) ? 2 : 4;
  } else if (in_type.isUnsignedInteger() && out_type.isSignedInteger()) {
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

void tpu::BinaryConstShiftOp::codegen_local_bm1684x_kernel(
    std::vector<group_info_t> &in_group_infos,
    std::vector<group_info_t> &out_group_infos, local_sec_info_t &sec_info,
    std::shared_ptr<std::vector<tensor_spec_t>> input_spec,
    std::shared_ptr<std::vector<tensor_spec_t>> output_spec) {
  const auto &gi = out_group_infos[0];
  binaryshift_local_param_t param = {0};
  auto &common = param.spec.common;
  common.binary_op = BM168x::binary_mode(getMode());
  common.rshift_num = -getShift();
  common.b_is_const = true;
  common.b_const_val = getScale();
  common.inversed = getIsReverse();
  common.round_mode = round_mode_convert(getRoundMode());
  common.is_saturate = getSaturation();
  param.spec.buffer = gi.buffer_addr;
  param.a_is_coeff = false;
  param.b_is_coeff = false;
  // inception setting
  auto in_gi = in_group_infos[0];
  setHWMargins(input_spec->at(0).hw_margins, in_gi, gi);

  BM168x::call_local_func("backend_api_binary_shift_local", &param,
                          sizeof(param), &sec_info, input_spec->data(),
                          output_spec->data());
}

// dynamic codegen
int64_t tpu::BinaryConstShiftOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(binaryshift_local_param_t);
  binaryshift_local_param_t param = {0};
  const auto &gi = getGroupInfo(0, 0, 0, 0, 0);

  auto &common = param.spec.common;
  common.binary_op = BM168x::binary_mode(getMode());
  common.rshift_num = -getShift();
  common.b_is_const = true;
  common.b_const_val = getScale();
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
int64_t tpu::BinaryConstShiftOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(binaryshift_global_param_t);
  binaryshift_global_param_t param{0};
  auto &spec = param.spec;
  spec.binary_op = BM168x::binary_mode(getMode());
  spec.rshift_num = -getShift();
  spec.b_is_const = true;
  spec.b_const_val = getScale();
  spec.inversed = getIsReverse();
  spec.round_mode = round_mode_convert(getRoundMode());
  spec.is_saturate = getSaturation();
  param.a_is_coeff = false;
  param.b_is_coeff = false;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::BinaryConstShiftOp::get_fw_type_bm1684x() {
  return FW_BMNET_BINARY_SHIFT;
}
