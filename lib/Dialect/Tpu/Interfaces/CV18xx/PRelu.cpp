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
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"



using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::PReluOp::codegen_global_cv18xx( int64_t layer_id) {

  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  gaddr_t ga_slope =  module::getAddress(this->getSlope());
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  if (module::isUniformQuantized(getOutput())) {
    int LE_scale = this->getRshift();
    int rshift_pos = this->getRshiftPos().value();
    int m_i8_pos = this->getMultiplierPos().value();
    cvi_backend_tg_fixed_prelu_kernel( layer_id, ga_input, ga_output, ga_slope,
                                        n, c, h, w, rshift_pos, m_i8_pos, LE_scale);
  } else {
    cvi_backend_tg_bf16_prelu_kernel( layer_id, ga_input, ga_output,
                                      ga_slope, n, c, h, w);
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

void tpu::PReluOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step) {
  llvm_unreachable("Not supported now");
}
