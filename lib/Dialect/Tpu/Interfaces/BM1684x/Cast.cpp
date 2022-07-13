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
typedef struct {
  uint64_t input_addr;
  uint64_t output_addr;
  uint64_t requant_addr;
  uint32_t buffer_local_addr;
  int n;
  int c;
  int h;
  int w;
  bool is_perchannel;
  float scale_value;
  float offset_value;
  int input_dtype;
  int output_dtype;
  int mode;
} requant_fp_param_t;

typedef struct {
  uint64_t input_addr;
  uint64_t output_addr;
  uint64_t dequant_addr;
  int n;
  int c;
  int h;
  int w;
  bool is_perchannel;
  float scale_value;
  int offset_value;
  int input_dtype;
} dequant_fp_param_t;

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::CastOp::codegen_global_int8_bm1684x() {
  bool qInput = Quant::isUniformQuantized(input());
  bool qOutput = Quant::isUniformQuantized(output());
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  if (!qInput && qOutput) {
    auto qtype = Quant::getUniformQuantizedType(output());
    requant_fp_param_t param = {0};
    param.input_addr = Module::getAddress(input());
    param.output_addr = Module::getAddress(output());
    param.n = (int)n;
    param.c = (int)c;
    param.h = (int)h;
    param.w = (int)w;
    param.is_perchannel = false;
    param.scale_value = 1.0 / qtype.getScale();
    param.offset_value = qtype.getZeroPoint();
    param.input_dtype = BM168x::getDataType(input());
    param.output_dtype = BM168x::getDataType(output());
    param.mode = 0;
    BM1684x::instance().call_global_func("backend_api_requant_float_global",
                                         &param, sizeof(param));
  } else {
    auto qtype = Quant::getUniformQuantizedType(input());
    dequant_fp_param_t param = {0};
    param.input_addr = Module::getAddress(input());
    param.output_addr = Module::getAddress(output());
    param.n = (int)n;
    param.c = (int)c;
    param.h = (int)h;
    param.w = (int)w;
    param.is_perchannel = false;
    param.scale_value = qtype.getScale();
    param.offset_value = qtype.getZeroPoint();
    param.input_dtype = BM168x::getDataType(input());
    BM1684x::instance().call_global_func("backend_api_dequant_float_global",
                                         &param, sizeof(param));
  }
}

// f32
void tpu::CastOp::codegen_global_float_bm1684x() {
  // same with int8
  codegen_global_int8_bm1684x();
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::CastOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
                                           int64_t out_lmem_bytes,
                                           int64_t in_nslice, int64_t in_hslice,
                                           int64_t out_nslice,
                                           int64_t out_hslice) {
  if (input().hasOneUse()) {
    return 0;
  }
  if (Quant::isUniformQuantized(input())) {
    return 0;
  }
  return in_lmem_bytes;
}

void tpu::CastOp::codegen_local_int8_bm1684x(int64_t n_step, int64_t h_step) {
  bool qInput = Quant::isUniformQuantized(input());
  bool qOutput = Quant::isUniformQuantized(output());
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
  if (!qInput && qOutput) {
    auto qtype = Quant::getUniformQuantizedType(output());
    uint32_t buffer_addr =
        input().hasOneUse() ? in_gi.out_addr : gi.buffer_addr;
    requant_fp_param_t param = {0};
    param.input_addr = in_gi.out_addr;
    param.output_addr = gi.out_addr;
    param.requant_addr = 0;
    param.buffer_local_addr = buffer_addr;
    param.n = gi.n_slice;
    param.c = c;
    param.h = gi.h_slice;
    param.w = w;
    param.is_perchannel = false;
    param.scale_value = 1 / qtype.getScale();
    param.offset_value = qtype.getZeroPoint();
    param.input_dtype = BM168x::getDataType(input());
    param.output_dtype = BM168x::getDataType(output());
    param.mode = 0;
    BM1684x::instance().call_local_func("backend_api_requant_float_local",
                                        &param, sizeof(param));
  } else {
    auto qtype = Quant::getUniformQuantizedType(input());
    dequant_fp_param_t param = {0};
    param.input_addr = in_gi.out_addr;
    param.output_addr = gi.out_addr;
    param.dequant_addr = 0;
    param.n = gi.n_slice;
    param.c = c;
    param.h = gi.h_slice;
    param.w = w;
    param.is_perchannel = false;
    param.scale_value = qtype.getScale();
    param.offset_value = qtype.getZeroPoint();
    param.input_dtype = BM168x::getDataType(input());
    BM1684x::instance().call_local_func("backend_api_dequant_float_local",
                                        &param, sizeof(param));
  }
}

void tpu::CastOp::codegen_local_float_bm1684x(int64_t n_step, int64_t h_step) {
  codegen_local_int8_bm1684x(n_step, h_step);
}
