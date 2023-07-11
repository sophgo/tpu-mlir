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

void tpu::ReciprocalOp::codegen_global_bm1684x() {
  assert(!module::isUniformQuantized(getInput()));

  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  constbinary_global_spec_t param = {0};
  param.common.binary_type = BINARY_DIV;
  param.common.if_relu = getDoRelu();
  param.common.relu_upper_limit = getReluLimit().convertToDouble();

  param.common.inversed = 0;
  param.common.scale_A = 1;
  param.common.rshift_A = 0;
  param.common.B_const_val = getConstVal().convertToDouble();
  param.common.B_dtype = input_spec->at(0).dtype;
  param.common.inversed = true;

  BM168x::call_global_func("backend_api_constbinary_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::ReciprocalOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  int64_t buffer_size = 0;
  auto dtype_A = BM168x::getDataType(getInput());
  if (dtype_A == DTYPE_INT8 || dtype_A == DTYPE_UINT8) {
    buffer_size = in_lmem_bytes * 2;
  }
  return buffer_size;
}

void tpu::ReciprocalOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                              int64_t h_step, int64_t d_step,
                                              int64_t w_step,
                                              group_type_t group_type,
                                              local_sec_info_t &sec_info) {
  assert(!module::isUniformQuantized(getInput()));
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);

  constbinary_local_spec_t param = {0};
  param.common.binary_type = BINARY_DIV;
  param.common.if_relu = getDoRelu();
  param.common.relu_upper_limit = getReluLimit().convertToDouble();
  param.common.inversed = true;
  param.common.scale_A = 1;
  param.common.rshift_A = 0;
  param.common.B_const_val = getConstVal().convertToDouble();
  param.common.B_dtype = input_spec->at(0).dtype;

  BM168x::call_local_func("backend_api_constbinary_local", &param,
                          sizeof(param), &sec_info, input_spec->data(),
                          output_spec->data());
}

// dynamic codegen
int64_t tpu::ReciprocalOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(constbinary_local_spec_t);
  assert(!module::isUniformQuantized(getInput()));
  group_type_t group_type = GROUP_UNSUPPORT;
  auto op = getOperation();
  if (auto gOp = dyn_cast<GroupOp>(op->getParentOp())) {
    group_type = static_cast<group_type_t>(gOp.getGroupType());
  }
  assert(group_type < GROUP_UNSUPPORT);
  auto input_spec = BM168x::get_input_spec(op, group_type);

  constbinary_local_spec_t param = {0};
  param.common.binary_type = BINARY_DIV;
  param.common.if_relu = getDoRelu();
  param.common.relu_upper_limit = getReluLimit().convertToDouble();
  param.common.inversed = true;
  param.common.scale_A = 1;
  param.common.rshift_A = 0;
  param.common.B_const_val = getConstVal().convertToDouble();
  param.common.B_dtype = input_spec->at(0).dtype;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ReciprocalOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(constbinary_global_spec_t);
  constbinary_global_spec_t param = {0};
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  param.common.binary_type = BINARY_DIV;
  param.common.if_relu = getDoRelu();
  param.common.relu_upper_limit = getReluLimit().convertToDouble();

  param.common.inversed = 0;
  param.common.scale_A = 1;
  param.common.rshift_A = 0;
  param.common.B_const_val = getConstVal().convertToDouble();
  param.common.B_dtype = input_spec->at(0).dtype;
  param.common.inversed = true;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::ReciprocalOp::get_fw_type_bm1684x() {
  return FW_BMNET_CONST_BINARY;
}
