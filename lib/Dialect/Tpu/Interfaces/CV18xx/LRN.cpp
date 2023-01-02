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
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================
void tpu::LRNOp::codegen_global_cv18xx(int64_t layer_id) {
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t exp_gaddr = module::getAddress(getTable());
  gaddr_t mantissa_gaddr = module::getAddress(getMantissa());
  gaddr_t ga_output = module::getAddress(getOutput());
  int64_t n, c, h, w;
  int64_t local_size = getSize();
  double alpha = this->getAlpha().convertToDouble();
  double k = this->getBias().convertToDouble();

  module::getNCHW(this->getInput(), n, c, h, w);
  if (module::isUniformQuantized(getOutput())) {
    llvm_unreachable("Not supported now");
  } else {
    cvi_backend_tg_bf16_lrn_kernel(layer_id, ga_input, ga_output, exp_gaddr,
                                   mantissa_gaddr, n, c, h, w, local_size,
                                   alpha, k);
  }
}
