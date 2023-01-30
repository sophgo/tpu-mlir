//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::PReluOp::codegen_global_cv18xx(int64_t layer_id) {
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  gaddr_t ga_slope = module::getAddress(getSlope());
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  if (module::isUniformQuantized(getOutput())) {
    int LE_scale = getRshift();
    int rshift_pos = getRshiftPos().value();
    int m_i8_pos = getMultiplierPos().value();
    cvi_backend_tg_fixed_prelu_kernel(layer_id, ga_input, ga_output, ga_slope,
                                      n, c, h, w, rshift_pos, m_i8_pos,
                                      LE_scale);
  } else {
    cvi_backend_tg_bf16_prelu_kernel(layer_id, ga_input, ga_output, ga_slope, n,
                                     c, h, w);
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::PReluOp::getBufferSize_cv18xx(int64_t in_lmem_bytes,
                                           int64_t out_lmem_bytes,
                                           int64_t in_nslice, int64_t in_hslice,
                                           int64_t out_nslice,
                                           int64_t out_hslice) {
  return 0;
}

void tpu::PReluOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                        int64_t layer_id) {
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto out_gi = LocalGenInterface::getGroupInfo(getOutput(), n_step, h_step);
  auto slope_gi = LocalGenInterface::getGroupInfo(getSlope());
  n = out_gi.n_slice;
  h = out_gi.h_slice;

  laddr_t la_input = in_gi.out_addr;
  laddr_t la_output = out_gi.out_addr;
  laddr_t la_slope = slope_gi.out_addr;

  if (module::isUniformQuantized(getOutput())) {
    int8_t m_i8_pos = getMultiplierPos().value();
    int8_t r_i8_pos = getRshiftPos().value();
    int8_t r_i8_neg = getRshift();

    cvi_backend_tl_prelu(layer_id, la_input, la_output, la_slope, n, c, h, w,
                         r_i8_pos, m_i8_pos, r_i8_neg);
  } else {
    cvi_backend_tl_bf16_prelu(layer_id, la_input, la_output, la_slope, n, c, h,
                              w);
  }
}
