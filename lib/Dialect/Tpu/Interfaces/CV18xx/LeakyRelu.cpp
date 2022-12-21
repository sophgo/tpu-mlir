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

void tpu::LeakyReluOp::codegen_global_cv18xx( int64_t layer_id) {

  gaddr_t ga_input = Module::getAddress(input());
  gaddr_t ga_output = Module::getAddress(output());
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  if (Quant::isUniformQuantized(output())) {
    auto pos_rshift = this->rshift().value();
    auto pos_m = this->multiplier().value();
    auto neg_rshift = this->rshift_neg().value();
    auto neg_m = this->multiplier_neg().value();
    cvi_backend_tg_fixed_leakyrelu_kernel( layer_id, ga_input, ga_output,
                                          n, c, h, w, pos_rshift, neg_rshift, pos_m, neg_m);
  } else {
    float negative_slope = static_cast<float>(alphaAttr().getValueAsDouble());
    cvi_backend_tg_bf16_leakyrelu_kernel( layer_id, ga_input, ga_output,
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
