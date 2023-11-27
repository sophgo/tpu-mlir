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
void tpu::MulConstOp::codegen_global_bm1684x() {
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  auto input_type = module::getStorageType(getInput());

  constbinary_global_spec_t param = {0};
  param.common.binary_type = BINARY_MUL;
  param.common.if_relu = getDoRelu();
  param.common.relu_upper_limit = getReluLimit().convertToDouble();
  param.common.inversed = 0;
  param.common.scale_A = 1;
  param.common.rshift_A = 0;
  param.common.f8_scale_A = 1.0;
  if (module::isUniformQuantized(getInput())) {
    param.common.B_const_val = 1; // coeff has been merge in multiplier&&rshift
    param.common.B_dtype = DTYPE_INT8;
    param.common.scale_A = getMultiplier();
    param.common.rshift_A = getRshift();
  } else {
    param.common.B_const_val = getConstVal().convertToDouble();
    param.common.B_dtype =
        input_type.isa<FloatType>() ? DTYPE_FP32 : DTYPE_INT32;
  }
  BM168x::call_global_func("backend_api_constbinary_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

static bool is_sign(DATA_TYPE_T dtype) {
  return !(dtype == DTYPE_UINT8 || dtype == DTYPE_UINT16 ||
           dtype == DTYPE_UINT32);
}

int64_t tpu::MulConstOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  int64_t buffer_size = 0;
  auto dtype_A = BM168x::getDataType(getInput());
  if (dtype_A == DTYPE_INT8 || dtype_A == DTYPE_UINT8) {
    buffer_size = in_lmem_bytes * 2;
  } else if ((dtype_A == DTYPE_F8E4M3 || dtype_A == DTYPE_F8E5M2) &&
             dtype_A == BM168x::getDataType(getOutput())) {
    buffer_size = in_lmem_bytes * 2;
  }
  return buffer_size;
}

void tpu::MulConstOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                            int64_t h_step, int64_t d_step,
                                            int64_t w_step,
                                            group_type_t group_type,
                                            local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  auto input_type = module::getStorageType(getInput());
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);

  constbinary_local_spec_t param = {0};
  param.common.binary_type = BINARY_MUL;
  param.common.if_relu = getDoRelu();
  param.common.relu_upper_limit = getReluLimit().convertToDouble();
  param.common.inversed = 0;
  param.common.scale_A = 1;
  param.common.rshift_A = 0;
  param.common.f8_scale_A = 1.0;
  param.buffer_addr = gi.buffer_addr;
  if (module::isUniformQuantized(getInput())) {
    param.common.B_const_val = 1; // coeff has been merge in multiplier&&rshift
    param.common.B_dtype = DTYPE_INT8;
    param.common.scale_A = getMultiplier();
    param.common.rshift_A = getRshift();
  } else {
    param.common.B_const_val = getConstVal().convertToDouble();
    param.common.B_dtype =
        input_type.isa<FloatType>() ? DTYPE_FP32 : DTYPE_INT32;
  }

  BM168x::call_local_func("backend_api_constbinary_local", &param,
                          sizeof(param), &sec_info, input_spec->data(),
                          output_spec->data());
}

// dynamic codegen
int64_t tpu::MulConstOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(constbinary_local_param_t);
  constbinary_local_param_t param;
  memset(&param, 0, sizeof(param));

  auto input_type = module::getStorageType(getInput());
  param.spec.common.binary_type = BINARY_MUL;
  param.spec.common.if_relu = getDoRelu();
  param.spec.common.relu_upper_limit = getReluLimit().convertToDouble();
  param.spec.common.inversed = 0;
  param.spec.common.scale_A = 1;
  param.spec.common.rshift_A = 0;
  param.spec.common.f8_scale_A = 1.0;
  if (module::isUniformQuantized(getInput())) {
    param.spec.common.B_const_val =
        1; // coeff has been merge in multiplier&&rshift
    param.spec.common.B_dtype = DTYPE_INT8;
    param.spec.common.scale_A = getMultiplier();
    param.spec.common.rshift_A = getRshift();
  } else {
    param.spec.common.B_const_val = getConstVal().convertToDouble();
    param.spec.common.B_dtype =
        input_type.isa<FloatType>() ? DTYPE_FP32 : DTYPE_INT32;
  }

  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::MulConstOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(constbinary_common_spec_t);
  constbinary_common_spec_t param = {0};
  auto input_type = module::getStorageType(getInput());
  param.binary_type = BINARY_MUL;
  param.if_relu = getDoRelu();
  param.relu_upper_limit = getReluLimit().convertToDouble();
  param.inversed = 0;
  param.scale_A = 1;
  param.rshift_A = 0;
  param.f8_scale_A = 1.0;
  if (module::isUniformQuantized(getInput())) {
    param.B_const_val = 1; // coeff has been merge in multiplier&&rshift
    param.B_dtype = DTYPE_INT8;
    param.scale_A = getMultiplier();
    param.rshift_A = getRshift();
  } else {
    param.B_const_val = getConstVal().convertToDouble();
    param.B_dtype = input_type.isa<FloatType>() ? DTYPE_FP32 : DTYPE_INT32;
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::MulConstOp::get_fw_type_bm1684x() { return FW_BMNET_CONST_BINARY; }
