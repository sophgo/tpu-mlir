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

// use for constbinary
typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
    int A_shape[MAX_SHAPE_DIMS];
    int shape_dim;
    int A_dtype;
    int B_dtype;
    int res_dtype;
    float B_const_val;
    int inversed;
    int binary_type;
    int if_relu;
    float relu_limit;
    int scale_A;
    int rshift_A;
} constbinary_global_param_t;

typedef struct {
    unsigned int input_local_addr;
    unsigned int output_local_addr;
    unsigned int buffer_addr;
    int A_shape[4];
    int A_dtype;
    int B_dtype;
    int res_dtype;
    float B_const_val;
    int inversed;
    int binary_type;
    int if_relu;
    float relu_limit;
    int scale_A;
    int rshift_A;
} constbinary_local_param_t;

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::MulConstOp::codegen_global_int8_bm1684x() {
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
  constbinary_global_param_t param = {0};
  param.input_global_addr = Module::getAddress(input());
  param.output_global_addr = Module::getAddress(output());
  param.A_shape[0] = n;
  param.A_shape[1] = c;
  param.A_shape[2] = h;
  param.A_shape[3] = w;
  param.shape_dim = 4;
  param.A_dtype = BM1684x::getDataType(input());
  param.B_dtype = DTYPE_INT8;
  param.res_dtype = BM1684x::getDataType(output());
  param.B_const_val = 1; //static_cast<float>(coeffAttr().getValueAsDouble());
  param.binary_type = BM_BINARY_MUL;
  param.if_relu = do_relu();
  param.relu_limit = 0;
  param.scale_A = multiplier();
  param.rshift_A = rshift();
  BM1684x::instance().call_global_func("backend_api_constbinary_global",
                                       &param, sizeof(param));
}

// f32
void tpu::MulConstOp::codegen_global_float_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  constbinary_global_param_t param = {0};
  param.input_global_addr = Module::getAddress(input());
  param.output_global_addr = Module::getAddress(output());
  param.A_shape[0] = n;
  param.A_shape[1] = c;
  param.A_shape[2] = h;
  param.A_shape[3] = w;
  param.shape_dim = 4;
  param.A_dtype = BM1684x::getDataType(input());
  param.B_dtype = DTYPE_FP32;
  param.res_dtype = BM1684x::getDataType(output());
  param.B_const_val = const_val().convertToDouble();
  param.inversed = 0;
  param.binary_type = BM_BINARY_MUL;
  param.if_relu = do_relu();
  param.relu_limit = 0;
  BM1684x::instance().call_global_func("backend_api_constbinary_global",
                                          &param, sizeof(param));
}

// =========================================
// LocalGenInterface
// =========================================

static bool is_sign(DATA_TYPE_T dtype) {
  return !(dtype == DTYPE_UINT8 || dtype == DTYPE_UINT16 || dtype == DTYPE_UINT32);
}

int64_t tpu::MulConstOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  int64_t buffer_size = 0;
  auto dtype_A = BM1684x::getDataType(input());
  if (dtype_A == DTYPE_INT8 || dtype_A == DTYPE_UINT8) {
    buffer_size = in_lmem_bytes * 2;
  }
  return buffer_size;
}

void tpu::MulConstOp::codegen_local_int8_bm1684x(int64_t n_step, int64_t h_step) {
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  auto op = getOperation();
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  constbinary_local_param_t param = {0};
  param.input_local_addr = in_gi.out_addr;
  param.output_local_addr = gi.out_addr;
  param.A_shape[0] = in_gi.n_slice;
  param.A_shape[1] = c;
  param.A_shape[2] = in_gi.h_slice;
  param.A_shape[3] = w;
  param.A_dtype = BM1684x::getDataType(input());
  param.B_dtype = DTYPE_INT8;
  param.res_dtype = BM1684x::getDataType(output());
  param.B_const_val = 1; //coeff has been merge in multiplier&&rshift
  param.inversed = 0;
  param.binary_type = BM_BINARY_MUL;
  param.if_relu = do_relu();
  param.scale_A = multiplier();
  param.rshift_A = rshift();
  BM1684x::instance().call_local_func("backend_api_constbinary_local",
                                       &param, sizeof(param));
}

void tpu::MulConstOp::codegen_local_float_bm1684x(int64_t n_step, int64_t h_step) {
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  auto op = getOperation();
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  constbinary_local_param_t param = {0};
  param.input_local_addr = in_gi.out_addr;
  param.output_local_addr = gi.out_addr;
  param.buffer_addr = gi.buffer_addr;
  param.A_shape[0] = in_gi.n_slice;
  param.A_shape[1] = c;
  param.A_shape[2] = in_gi.h_slice;
  param.A_shape[3] = w;
  param.A_dtype = BM1684x::getDataType(input());
  param.B_dtype = DTYPE_FP32; // assume coeff is fp32
  param.res_dtype = BM1684x::getDataType(output());
  param.B_const_val = const_val().convertToFloat();
  param.inversed = 0;
  param.binary_type = BM_BINARY_MUL;
  param.if_relu = do_relu();
  param.relu_limit = relu_limit().convertToDouble();
  BM1684x::instance().call_local_func("backend_api_constbinary_local",
                                       &param, sizeof(param));
}
