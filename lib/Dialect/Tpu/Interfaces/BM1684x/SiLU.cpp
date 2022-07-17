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
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::SiLUOp::codegen_global_int8_bm1684x() {
  auto input_shape = Module::getShape(input());
  active_param_t p = {0};
  p.input_addr = Module::getAddress(input());
  p.output_addr = Module::getAddress(output());
  p.shape_dim = input_shape.size();
  for (int i = 0; i < p.shape_dim; i++) {
    p.shape[i] = input_shape[i];
  }
  p.active_type = ACTIVE_SILU;
  p.dtype = BM168x::getDataType(output());
  BM1684x::instance().call_global_func("backend_api_active_global", &p,
                                       sizeof(p));
}

void tpu::SiLUOp::codegen_global_float_bm1684x() {
  codegen_global_int8_bm1684x();
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::SiLUOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
                                           int64_t out_lmem_bytes,
                                           int64_t in_nslice, int64_t in_hslice,
                                           int64_t out_nslice,
                                           int64_t out_hslice) {
  auto stype = Module::getStorageType(input());
  // |    work1    |    work0    | exp coeff  | exp_table |
  // | tensor_size | tensor_size |     32     |    192    |
  int64_t bytes = in_lmem_bytes / in_nslice;
  int64_t buffer_size = 2 * align_up(bytes, 64l);
  int64_t dtype_len = stype.getIntOrFloatBitWidth() / 8;
  buffer_size += align_up(32 * dtype_len, 64l) + align_up(192 * dtype_len, 64l);
  return buffer_size;
}

void tpu::SiLUOp::codegen_local_int8_bm1684x(int64_t n_step, int64_t h_step) {
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  active_param_t p = {0};
  p.input_addr = in_gi.out_addr;
  p.output_addr = gi.out_addr;
  p.buffer_local_addr = gi.buffer_addr;
  p.shape[0] = gi.n_slice;
  p.shape[1] = c;
  p.shape[2] = gi.h_slice;
  p.shape[3] = w;
  p.shape_dim = 4;
  p.dtype = BM168x::getDataType(output());
  p.active_type = ACTIVE_SILU;
  BM1684x::instance().call_local_func("backend_api_active_local", &p,
                                      sizeof(p));
}

void tpu::SiLUOp::codegen_local_float_bm1684x(int64_t n_step, int64_t h_step) {
  codegen_local_int8_bm1684x(n_step, h_step);
}
