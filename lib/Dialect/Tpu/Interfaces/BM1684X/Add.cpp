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

// int8
void tpu::AddOp::codegen_global_bm1684x() {
  std::vector<int64_t> multi_v(2, 1);
  std::vector<int64_t> rshift_v(2, 0);

  if (module::isUniformQuantized(getInputs()[0], getOutput()) ||
      (module::isUniformQuantized(getInputs()[1], getOutput()))) {
    auto m_v = module::getI64Array(getMultipliers(), 2, 1);
    auto r_v = module::getI64Array(getRshifts(), 2, 0);
    multi_v = *m_v.get();
    rshift_v = *r_v.get();
  }

  bcbinary_common_spec_t param{0};
  param.binary_type = BINARY_ADD;
  param.if_relu = getDoRelu();
  param.relu_upper_limit = getReluLimit().convertToDouble();
  param.rshift_A = rshift_v[0];
  param.rshift_B = rshift_v[1];
  param.scale_A = multi_v[0];
  param.scale_B = multi_v[1];
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_bcbinary_global", &param, sizeof(param),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::AddOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  auto out_type = module::getStorageType(getOutput());
  if (out_type.isInteger(8)) {
    // INT16 as middle result
    return 2 * out_lmem_bytes * sizeof(int16_t);
  }
  return 0;
}

void tpu::AddOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                       int64_t h_step, int64_t d_step,
                                       int64_t w_step, group_type_t group_type,
                                       local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);

  std::vector<int64_t> multi_v(2, 1);
  std::vector<int64_t> rshift_v(2, 0);
  if (module::isUniformQuantized(getInputs()[0], getOutput()) ||
      (module::isUniformQuantized(getInputs()[1], getOutput()))) {
    auto m_v = module::getI64Array(getMultipliers(), 2, 1);
    auto r_v = module::getI64Array(getRshifts(), 2, 0);
    multi_v = *m_v.get();
    rshift_v = *r_v.get();
  }

  bcbinary_local_param_t param = {0};
  param.spec.common.binary_type = BINARY_ADD;
  param.spec.common.if_relu = getDoRelu();
  param.spec.common.relu_upper_limit = getReluLimit().convertToDouble();
  param.spec.common.rshift_A = rshift_v[0];
  param.spec.common.rshift_B = rshift_v[1];
  param.spec.common.scale_A = multi_v[0];
  param.spec.common.scale_B = multi_v[1];
  param.spec.buffer_addr = gi.buffer_addr;
  param.A_is_coeff = false;
  param.B_is_coeff = false;

  BM168x::call_local_func("backend_api_bcbinary_local", &param, sizeof(param),
                          &sec_info, input_spec->data(), output_spec->data());
}

// dynamic codegen
int64_t tpu::AddOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(bcbinary_local_param_t);
  auto gi = getGroupInfo(0, 0, 0, 0, 0);
  std::vector<int64_t> multi_v(2, 1);
  std::vector<int64_t> rshift_v(2, 0);

  if (module::isUniformQuantized(getInputs()[0], getOutput()) ||
      (module::isWeight(getInputs()[0]) &&
       module::isUniformQuantized(getInputs()[1], getOutput()))) {
    auto m_v = module::getI64Array(getMultipliers(), 2, 1);
    auto r_v = module::getI64Array(getRshifts(), 2, 0);
    multi_v = *m_v.get();
    rshift_v = *r_v.get();
  }

  bcbinary_local_param_t param;
  memset(&param, 0, sizeof(param));
  param.spec.common.binary_type = BINARY_ADD;
  param.spec.common.if_relu = getDoRelu();
  param.spec.common.relu_upper_limit = getReluLimit().convertToDouble();
  param.spec.common.rshift_A = rshift_v[0];
  param.spec.common.rshift_B = rshift_v[1];
  param.spec.common.scale_A = multi_v[0];
  param.spec.common.scale_B = multi_v[1];
  param.spec.buffer_addr = gi.buffer_addr;
  param.A_is_coeff = false;
  param.B_is_coeff = false;

  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::AddOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) {
    return sizeof(bcbinary_common_spec_t);
  }
  bcbinary_common_spec_t spec = {0};
  spec.binary_type = BINARY_ADD;
  spec.if_relu = getDoRelu();
  spec.relu_upper_limit = getReluLimit().convertToDouble();
  auto m_v = module::getI64Array(getMultipliers(), 2, 1);
  auto r_v = module::getI64Array(getRshifts(), 2, 0);
  spec.rshift_A = r_v->at(0);
  spec.rshift_B = r_v->at(1);
  spec.scale_A = m_v->at(0);
  spec.scale_B = m_v->at(1);

  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::AddOp::get_fw_type_bm1684x() { return FW_BMNET_BROADCAST_BINARY; }
