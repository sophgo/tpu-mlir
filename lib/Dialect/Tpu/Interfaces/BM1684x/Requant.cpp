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
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

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
  int mul_value;
  int shift_value;
  int offset_value;
  int input_dtype;
  int output_dtype;
  int mode;
  int reshaped_coeff;
  int zx_value;
} requant_int_param_t;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::RequantOp::codegen_global_int8_bm1684x() {
  requant_int_param_t param = {0};
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  param.input_addr = Module::getAddress(input());
  param.output_addr = Module::getAddress(output());
  param.n = (int)n;
  param.c = (int)c;
  param.h = (int)h;
  param.w = (int)w;

  if (quant().getType().isa<RankedTensorType>()) {
    param.is_perchannel = true;
    param.requant_addr = Module::getAddress(quant());
    param.reshaped_coeff = false;
  } else {
    auto oqtype = Quant::getUniformQuantizedType(output());
    auto iqtype = Quant::getUniformQuantizedType(input());
    param.mul_value = multiplier().getValue();
    param.shift_value = -rshift().getValue();
    param.offset_value = oqtype.getZeroPoint();
    param.zx_value = iqtype.getZeroPoint();
  }
  param.mode = quant_mode().getValue();
  param.input_dtype = BM168x::getDataType(input());
  param.output_dtype = BM168x::getDataType(output());
  BM1684x::instance().call_global_func("backend_api_requant_int_global", &param,
                                       sizeof(param));
}

void tpu::RequantOp::codegen_global_float_bm1684x() {
  codegen_global_int8_bm1684x();
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::RequantOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  // store INT16:(X - Zx)
  auto input_dtype = BM1684x::getDataType(input());
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  int64_t sz = 0;
  if (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8) {
    sz +=
        ceiling_func(c, BM1684x::instance().get_npu_num()) *
        align_up(in_hslice * w, BM1684x::instance().get_eu_num(sizeof(short))) *
        sizeof(short);
  }
  return sz;
}

void tpu::RequantOp::codegen_local_int8_bm1684x(int64_t n_step,
                                                int64_t h_step) {
  requant_int_param_t param = {0};
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  param.input_addr = (uint32_t)in_gi.out_addr;
  param.output_addr = (uint32_t)gi.out_addr;
  param.buffer_local_addr = (uint32_t)gi.buffer_addr;
  param.n = gi.n_slice;
  param.c = c;
  param.h = gi.h_slice;
  param.w = w;

  if (quant().getType().isa<RankedTensorType>()) {
    auto requant_gi = LocalGenInterface::getGroupInfo(quant(), n_step, h_step);
    param.requant_addr = (uint32_t)requant_gi.out_addr;
    param.is_perchannel = true;
    param.reshaped_coeff = false;
  } else {
    auto oqtype = Quant::getUniformQuantizedType(output());
    auto iqtype = Quant::getUniformQuantizedType(input());
    param.mul_value = multiplier().getValue();
    param.shift_value = -rshift().getValue();
    param.offset_value = oqtype.getZeroPoint();
    param.zx_value = iqtype.getZeroPoint();
  }
  param.input_dtype = BM168x::getDataType(input());
  param.output_dtype = BM168x::getDataType(output());
  param.mode = 2;
  BM1684x::instance().call_local_func("backend_api_requant_int_local", &param,
                                      sizeof(param));
}

void tpu::RequantOp::codegen_local_float_bm1684x(int64_t n_step,
                                                 int64_t h_step) {
  llvm_unreachable("support later");
}
