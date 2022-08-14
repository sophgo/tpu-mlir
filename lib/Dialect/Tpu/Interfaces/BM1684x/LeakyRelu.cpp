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
  uint64_t input_addr;
  uint64_t slope_addr;
  uint64_t output_addr;
  int32_t input_n;
  int32_t input_c;
  int32_t input_h;
  int32_t input_w;
  int32_t channel_shared;
  float slope_val;
  int32_t rshift_bit;
  float relu_limit;
  DATA_TYPE_T dtype;
} leakyrelu_param_t;

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================

void tpu::LeakyReluOp::codegen_global_int8_bm1684x() {
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  leakyrelu_param_t param = {0};
  param.input_addr = Module::getAddress(input());
  param.slope_addr = -1;
  param.output_addr = Module::getAddress(output());
  param.input_n = static_cast<int32_t>(n);
  param.input_c = static_cast<int32_t>(c);
  param.input_h = static_cast<int32_t>(h);
  param.input_w = static_cast<int32_t>(w);
  param.channel_shared = 1;
  param.slope_val = static_cast<float>(multiplier().getValue());
  param.rshift_bit = rshift().getValue();
  param.relu_limit = -1;
  param.dtype = BM168x::getDataType(input());
  BM1684x::instance().call_global_func("backend_api_prelu_global", &param,
                                       sizeof(leakyrelu_param_t));
}

void tpu::LeakyReluOp::codegen_global_float_bm1684x() {
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  leakyrelu_param_t param = {0};
  param.input_addr = Module::getAddress(input());
  param.slope_addr = -1;
  param.output_addr = Module::getAddress(output());
  param.input_n = static_cast<int32_t>(n);
  param.input_c = static_cast<int32_t>(c);
  param.input_h = static_cast<int32_t>(h);
  param.input_w = static_cast<int32_t>(w);
  param.channel_shared = 1;
  param.slope_val = static_cast<float>(alphaAttr().getValueAsDouble());
  param.rshift_bit = 0;
  param.relu_limit = -1;
  param.dtype = BM168x::getDataType(input());
  BM1684x::instance().call_global_func("backend_api_prelu_global", &param,
                                       sizeof(leakyrelu_param_t));
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::LeakyReluOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::LeakyReluOp::codegen_local_int8_bm1684x(int64_t n_step,
                                                  int64_t h_step) {
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);

  leakyrelu_param_t param = {0};
  param.input_addr = in_gi.out_addr;
  param.slope_addr = -1;
  param.output_addr = gi.out_addr;
  param.input_n = static_cast<int32_t>(gi.n_slice);
  param.input_c = static_cast<int32_t>(c);
  param.input_h = static_cast<int32_t>(gi.h_slice);
  param.input_w = static_cast<int32_t>(w);
  param.channel_shared = 1;
  param.slope_val = static_cast<float>(multiplier().getValue());
  param.rshift_bit = rshift().getValue();
  param.relu_limit = -1;
  param.dtype = BM168x::getDataType(input());
  BM1684x::instance().call_local_func("backend_api_prelu_local", &param,
                                      sizeof(leakyrelu_param_t));
}

void tpu::LeakyReluOp::codegen_local_float_bm1684x(int64_t n_step,
                                                   int64_t h_step) {
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);

  leakyrelu_param_t param = {0};
  param.input_addr = in_gi.out_addr;
  param.slope_addr = -1;
  param.output_addr = gi.out_addr;
  param.input_n = static_cast<int32_t>(gi.n_slice);
  param.input_c = static_cast<int32_t>(c);
  param.input_h = static_cast<int32_t>(gi.h_slice);
  param.input_w = static_cast<int32_t>(w);
  param.channel_shared = 1;
  param.slope_val = static_cast<float>(alphaAttr().getValueAsDouble());
  param.rshift_bit = 0;
  param.relu_limit = -1;
  param.dtype = BM168x::getDataType(input());
  BM1684x::instance().call_local_func("backend_api_prelu_local", &param,
                                      sizeof(leakyrelu_param_t));
}
