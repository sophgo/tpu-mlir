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

// =========================================
// GlobalGenInterface
// =========================================

void tpu::RequantOp::codegen_global_bm1684x() {
  requant_int_param_t param = {0};
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  param.input_addr = Module::getAddress(input());
  param.output_addr = Module::getAddress(output());
  param.n = (int)n;
  param.c = (int)c;
  param.h = (int)h;
  param.w = (int)w;

  auto iqtype = Quant::getUniformQuantizedType(input());
  auto oqtype = Quant::getUniformQuantizedType(output());
  param.mul_value = multiplier();
  param.shift_value = -rshift();
  param.offset_value = oqtype.getZeroPoint();
  param.zx_value = iqtype.getZeroPoint();
  param.mode = 2;
  param.input_dtype = BM168x::getDataType(input());
  param.output_dtype = BM168x::getDataType(output());
  BM1684x::instance().call_global_func("backend_api_requant_int_global", &param,
                                       sizeof(param));
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::RequantOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  auto input_dtype = BM1684x::getDataType(input());
  if (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8) {
    // store INT16:(X - Zx)
    return in_lmem_bytes * 2;
  }
  return 0;
}

void tpu::RequantOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step) {
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

  auto oqtype = Quant::getUniformQuantizedType(output());
  param.mul_value = multiplier();
  param.shift_value = -rshift();
  param.offset_value = oqtype.getZeroPoint();

  if (Quant::isUniformQuantized(input())) {
    auto iqtype = Quant::getUniformQuantizedType(input());
    param.zx_value = iqtype.getZeroPoint();
  }
  param.input_dtype = BM168x::getDataType(input());
  param.output_dtype = BM168x::getDataType(output());
  param.mode = 2;
  BM1684x::instance().call_local_func("backend_api_requant_int_local", &param,
                                      sizeof(param));
}
