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
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/DynCompileCommon.hpp"
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::SubConstOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  auto input_type = module::getStorageType(getInput());
  constbinary_global_spec_t param = {0};
  param.common.binary_type = BINARY_SUB;
  param.common.if_relu = getDoRelu();
  param.common.relu_upper_limit = getReluLimit().convertToDouble();
  param.common.B_const_val = getConstVal().convertToDouble();
  param.common.inversed = getIsReverse();
  param.common.scale_A = 1;
  param.common.rshift_A = 0;
  if (module::isUniformQuantized(getInput())) {
    param.common.B_dtype = DTYPE_INT32;
    param.common.scale_A = getMultiplier();
    param.common.rshift_A = getRshift();
  } else {
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

int64_t tpu::SubConstOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_hslice, int64_t out_dslice, int64_t out_wslice,
    group_type_t group_type) {
  int64_t buffer_size = 0;
  auto dtype_A = BM168x::getDataType(getInput());
  if (dtype_A == DTYPE_INT8 || dtype_A == DTYPE_UINT8) {
    buffer_size = in_lmem_bytes * 2;
  }
  return buffer_size;
}

void tpu::SubConstOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step, int64_t d_step, int64_t w_step,
                                            group_type_t group_type,
                                            local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  auto input_type = module::getStorageType(getInput());

  constbinary_local_spec_t param = {0};
  param.common.binary_type = BINARY_SUB;
  param.common.if_relu = getDoRelu();
  param.common.relu_upper_limit = getReluLimit().convertToDouble();
  param.common.B_const_val = getConstVal().convertToDouble();
  param.common.inversed = getIsReverse();
  param.common.scale_A = 1;
  param.common.rshift_A = 0;
  if (module::isUniformQuantized(getInput())) {
    param.common.B_dtype = DTYPE_INT32;
    param.common.scale_A = getMultiplier();
    param.common.rshift_A = getRshift();
  } else {
    param.common.B_dtype =
        input_type.isa<FloatType>() ? DTYPE_FP32 : DTYPE_INT32;
  }

  BM168x::call_local_func("backend_api_constbinary_local", &param,
                          sizeof(param), &sec_info, input_spec->data(),
                          output_spec->data());
}

// dynamic codegen
int64_t tpu::SubConstOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(constbinary_local_spec_t);
  auto input_type = module::getStorageType(getInput());
  constbinary_local_spec_t param = {0};
  param.common.binary_type = BINARY_SUB;
  param.common.if_relu = getDoRelu();
  param.common.relu_upper_limit = getReluLimit().convertToDouble();
  param.common.B_const_val = getConstVal().convertToDouble();
  param.common.inversed = getIsReverse();
  param.common.scale_A = 1;
  param.common.rshift_A = 0;
  if (module::isUniformQuantized(getInput())) {
    param.common.B_dtype = DTYPE_INT32;
    param.common.scale_A = getMultiplier();
    param.common.rshift_A = getRshift();
  } else {
    param.common.B_dtype =
        input_type.isa<FloatType>() ? DTYPE_FP32 : DTYPE_INT32;
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::SubConstOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(constbinary_global_spec_t);
  auto input_type = module::getStorageType(getInput());
  constbinary_global_spec_t param = {0};
  param.common.binary_type = BINARY_SUB;
  param.common.if_relu = getDoRelu();
  param.common.relu_upper_limit = getReluLimit().convertToDouble();
  param.common.B_const_val = getConstVal().convertToDouble();
  param.common.inversed = getIsReverse();
  param.common.scale_A = 1;
  param.common.rshift_A = 0;
  if (module::isUniformQuantized(getInput())) {
    param.common.B_dtype = DTYPE_INT32;
    param.common.scale_A = getMultiplier();
    param.common.rshift_A = getRshift();
  } else {
    param.common.B_dtype =
        input_type.isa<FloatType>() ? DTYPE_FP32 : DTYPE_INT32;
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::SubConstOp::get_fw_type_bm1684x() {
  return FW_BMNET_CONST_BINARY;
}
