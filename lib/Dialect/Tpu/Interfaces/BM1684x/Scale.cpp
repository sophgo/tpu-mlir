//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long scale_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long shift_global_addr;
  unsigned long long output_global_addr;
  int input_n;
  int input_c;
  int input_h;
  int input_w;
  int axis;
  int axis_num;
  int has_bias;
  int if_relu;
  float relu_upper_limit;
  int input_sign;
  int scale_sign;
  int bias_sign;
  int merge_weight_bias;
  ROUND_MODE_T round_mode;
  DATA_TYPE_T idtype;
  int version;
} scale_global_param_t;

typedef struct {
  unsigned int input_local_addr;
  unsigned int scale_local_addr;
  unsigned int bias_local_addr;
  unsigned int output_local_addr;
  const int *input_shape;
  const int *scale_shape;
  int has_bias;
  int if_relu;
  float relu_upper_limit;
  int is_scale_coeff;
  int is_bias_coeff;
  DATA_TYPE_T dtype;
} scale_float_local_param_t;

typedef struct {
  unsigned int input_local_addr;
  unsigned int scale_local_addr;
  unsigned int bias_local_addr;
  unsigned int shift_local_addr;
  unsigned int output_local_addr;
  unsigned int buffer_local_addr;
  const int *input_shape;
  int if_relu;
  float relu_upper_limit;
  int is_scale_coeff;
  int is_bias_coeff;
  int is_shift_coeff;
  DATA_TYPE_T idtype;
  DATA_TYPE_T sdtype;
  DATA_TYPE_T bdtype;
  ROUND_MODE_T round_mode;
  int version;
} scale_fixed_local_param_t;

#ifdef __cplusplus
}
#endif
// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::ScaleOp::codegen_global_bm1684x() {
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
  scale_global_param_t p = {0};
  p.input_global_addr = Module::getAddress(input());
  p.scale_global_addr = Module::getAddress(scale());
  p.bias_global_addr = Module::getAddress(bias());
  p.output_global_addr = Module::getAddress(output());
  p.input_n = (int)n;
  p.input_c = (int)c;
  p.input_h = (int)h;
  p.input_w = (int)w;
  p.axis = 1;
  p.axis_num = 1;
  p.has_bias = true;
  p.if_relu = do_relu();
  p.relu_upper_limit = relu_limit().convertToDouble();
  p.merge_weight_bias = 0;
  p.round_mode = ROUND_UP;
  p.idtype = BM168x::getDataType(input());
  if (Quant::isUniformQuantized(input())) {
    p.shift_global_addr = Module::getAddress(lshift());
    p.input_sign = Module::getStorageType(input()).isSignedInteger();
    p.scale_sign = Module::getStorageType(scale()).isSignedInteger();
    p.bias_sign = Module::getStorageType(bias()).isSignedInteger();
    p.version = 10;
    BM1684x::instance().call_global_func("backend_api_scale_global", &p,
                                         sizeof(scale_global_param_t));
  } else {
    BM1684x::instance().call_global_func("backend_api_scale_global", &p,
                                         sizeof(scale_global_param_t));
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::ScaleOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  auto out_type = Module::getStorageType(output());
  if (out_type.isInteger(8)) {
    // INT16 as middle result
    return 2 * out_lmem_bytes * sizeof(int16_t);
  } else if (out_type.isBF16() || out_type.isF16()) {
    return out_lmem_bytes;
  }
  return 0;
}

void tpu::ScaleOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step) {
  // out_zp is should be passed to backend
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto scale_gi = LocalGenInterface::getGroupInfo(scale(), n_step, h_step);
  auto bias_gi = LocalGenInterface::getGroupInfo(bias(), n_step, h_step);

  auto gi = getGroupInfo(n_step, h_step);
  llvm::SmallVector<int32_t, 4> input_shape = {(int)gi.n_slice, (int)c,
                                               (int)gi.h_slice, (int)w};
  llvm::SmallVector<int32_t, 4> scale_shape = {1, (int)c, 1, 1};
  if (Quant::isUniformQuantized(input())) {
    auto lshift_gi = LocalGenInterface::getGroupInfo(lshift(), n_step, h_step);
    scale_fixed_local_param_t p = {0};
    p.input_local_addr = in_gi.out_addr;
    p.scale_local_addr = scale_gi.out_addr;
    p.bias_local_addr = bias_gi.out_addr;
    p.shift_local_addr = lshift_gi.out_addr;
    p.output_local_addr = gi.out_addr;
    p.buffer_local_addr = gi.buffer_addr;
    p.input_shape = input_shape.data();
    p.if_relu = do_relu();
    p.relu_upper_limit = relu_limit().convertToDouble();
    p.is_scale_coeff = 1;
    p.is_bias_coeff = 1;
    p.is_shift_coeff = 1;
    p.idtype = BM168x::getDataType(input());
    p.sdtype = BM168x::getDataType(scale());
    p.bdtype = BM168x::getDataType(bias());
    p.round_mode = ROUND_UP;
    p.version = 10;
    BM1684x::instance().call_local_func("backend_api_scale_fixed_local", &p,
                                        sizeof(scale_fixed_local_param_t));
  } else {
    scale_float_local_param_t p = {0};
    p.input_local_addr = in_gi.out_addr;
    p.scale_local_addr = scale_gi.out_addr;
    p.bias_local_addr = bias_gi.out_addr;
    p.output_local_addr = gi.out_addr;
    p.input_shape = input_shape.data();
    p.scale_shape = scale_shape.data();
    p.if_relu = do_relu();
    p.relu_upper_limit = relu_limit().convertToDouble();
    p.has_bias = 1;
    p.is_scale_coeff = 1;
    p.is_bias_coeff = 1;
    p.dtype = BM168x::getDataType(input());
    BM1684x::instance().call_local_func("backend_api_scale_float_local", &p,
                                        sizeof(scale_float_local_param_t));
  }
}
