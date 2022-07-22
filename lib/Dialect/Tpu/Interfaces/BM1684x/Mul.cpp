//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct binary_common_spec {
    int32_t binary_type;
    int32_t if_relu;
    float relu_upper_limit;
    int32_t scale_A;
    int32_t scale_B;
    int32_t rshift_A;
    int32_t rshift_B;
} binary_common_spec_t ;

typedef struct binary_local_spec {
    binary_common_spec_t common;
    uint32_t buffer_addr;
} binary_local_spec_t;

typedef struct binary_local_param {
    binary_local_spec_t spec;
    int32_t A_is_coeff;
    int32_t B_is_coeff;
} binary_local_param_t;

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::MulOp::codegen_global_int8_bm1684x() {
  int input_num = inputs().size();
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
  auto op = getOperation();
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);
  binary_common_spec_t spec;
  memset(&spec, 0, sizeof(binary_common_spec_t));
  spec.binary_type = BM_BINARY_MUL;
  spec.if_relu = (int)do_relu();
  spec.scale_A = (int)multiplier();
  spec.scale_B = 1;
  spec.rshift_A = (int)rshift();
  spec.rshift_B = 0;
  BM1684x::instance().call_global_func("backend_api_eltbinary_global", &spec,
                                       sizeof(spec), input_spec->data(), output_spec->data());
}

// f32
void tpu::MulOp::codegen_global_float_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);
  binary_common_spec_t spec;
  memset(&spec, 0, sizeof(binary_common_spec_t));
  spec.binary_type = BM_BINARY_MUL;
  spec.if_relu = (int)do_relu();
  spec.relu_upper_limit = 0;
  spec.scale_A = 1;
  spec.scale_B = 1;
  spec.rshift_A = 0;
  spec.rshift_B = 0;
  BM1684x::instance().call_global_func("backend_api_eltbinary_global", &spec,
                                       sizeof(spec), input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

bool is_sign(DATA_TYPE_T dtype) {
  return !(dtype == DTYPE_UINT8 || dtype == DTYPE_UINT16 || dtype == DTYPE_UINT32);
}

int64_t tpu::MulOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  int64_t buffer_size = 0;
  auto dtype_A = BM168x::getDataType(inputs()[0]);
  auto dtype_B = BM168x::getDataType(inputs()[1]);
  auto dtype_O = BM168x::getDataType(output());
  if (dtype_A == DTYPE_INT8 || dtype_A == DTYPE_UINT8) {
    if (multiplier() != 1 || rshift() != 0) {
      buffer_size = in_lmem_bytes * 2;
    }
  } else if ((sizeof(dtype_A) > sizeof(dtype_O)) &&
             (is_sign(dtype_A) || is_sign(dtype_B)) &&
             (!is_sign(dtype_O))) {
      buffer_size = in_lmem_bytes;
  }
  return buffer_size;
}

void tpu::MulOp::codegen_local_int8_bm1684x(int64_t n_step, int64_t h_step) {
  int64_t n, c, h, w;
  Module::getNCHW(inputs()[0], n, c, h, w);
  auto op = getOperation();
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(inputs()[0], n_step, h_step);
  binary_local_param_t param;
  memset(&param, 0, sizeof(binary_local_param_t));
  param.spec.common.binary_type = BM_BINARY_MUL;
  param.spec.common.if_relu = (int)do_relu();
  param.spec.common.relu_upper_limit = 0;
  param.spec.common.scale_A = (int)multiplier();
  param.spec.common.scale_B = 1;
  param.spec.common.rshift_A = (int)rshift();
  param.spec.common.rshift_B = 0;
  param.spec.buffer_addr = gi.buffer_addr;
  param.A_is_coeff = 0;
  param.B_is_coeff = 0;

  local_sec_info_t sec_info;
  memset(&sec_info, 0, sizeof(sec_info));
  sec_info.n_slice = in_gi.n_slice;
  sec_info.d_slice = 1;
  sec_info.h_slice = in_gi.h_slice;
  sec_info.h_idx = in_gi.h_idx;
  sec_info.is_h_split = !(in_gi.h_idx == 0 && in_gi.h_slice == h);
  sec_info.w_slice = w;
  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_idx = gi.h_idx;
  sec_info.out_h_slice = gi.h_slice;
  sec_info.out_w_slice = w;

  BM1684x::instance().call_local_func("backend_api_eltbinary_local", &param,
                                      sizeof(param), &sec_info,
                                      input_spec->data(),
                                      output_spec->data());
}

void tpu::MulOp::codegen_local_float_bm1684x(int64_t n_step, int64_t h_step) {
  int64_t n, c, h, w;
  Module::getNCHW(inputs()[0], n, c, h, w);
  auto op = getOperation();
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(inputs()[0], n_step, h_step);
  binary_local_param_t param;
  memset(&param, 0, sizeof(binary_local_param_t));
  param.spec.common.binary_type = BM_BINARY_MUL;
  param.spec.common.if_relu = do_relu();
  param.spec.common.relu_upper_limit = 0;
  param.spec.common.scale_A = 1;
  param.spec.common.scale_B = 1;
  param.spec.common.rshift_A = 0;
  param.spec.common.rshift_B = 0;
  param.spec.buffer_addr = gi.buffer_addr;
  param.A_is_coeff = 0;
  param.B_is_coeff = 0;

  local_sec_info_t sec_info;
  memset(&sec_info, 0, sizeof(sec_info));
  sec_info.n_slice = in_gi.n_slice;
  sec_info.d_slice = 1;
  sec_info.h_slice = in_gi.h_slice;
  sec_info.h_idx = in_gi.h_idx;
  sec_info.is_h_split = !(in_gi.h_idx == 0 && in_gi.h_slice == h);
  sec_info.w_slice = w;
  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_idx = gi.h_idx;
  sec_info.out_h_slice = gi.h_slice;
  sec_info.out_w_slice = w;

  BM1684x::instance().call_local_func("backend_api_eltbinary_local", &param,
                                      sizeof(param), &sec_info,
                                      input_spec->data(),
                                      output_spec->data());
}
