//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

// #include "tpu_mlir/Backend/BM168x/cv18xx.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::LeakyReluOp::codegen_global_cv18xx(int64_t layer_id) {

  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  if (module::isUniformQuantized(getOutput())) {
    auto pos_rshift = this->getRshift().value();
    auto pos_m = this->getMultiplier().value();
    auto neg_rshift = this->getRshiftNeg().value();
    auto neg_m = this->getMultiplierNeg().value();
    cvi_backend_tg_fixed_leakyrelu_kernel(layer_id, ga_input, ga_output, n, c,
                                          h, w, pos_rshift, neg_rshift, pos_m,
                                          neg_m);
  } else {
    float negative_slope =
        static_cast<float>(getAlpha().value().convertToDouble());
    cvi_backend_tg_bf16_leakyrelu_kernel(layer_id, ga_input, ga_output,
                                         negative_slope, n, c, h, w);
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::LeakyReluOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  llvm_unreachable("Not supported now");
  return 0;
}

void tpu::LeakyReluOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step) {
  llvm_unreachable("Not supported now");
}
