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

#include "tpu_mlir/Support/MathUtils.h"



using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::PoolMaskOp::codegen_global_cv18xx(int64_t layer_id) {

  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  if (module::isUniformQuantized(getOutput())) {
    cvi_backend_tg_pool_mask_kernel(layer_id, ga_input, ga_output, n, c, h, w,
                                    getScale(), CVK_FMT_I8);
  } else {
    cvi_backend_tg_pool_mask_kernel(layer_id, ga_input, ga_output, n, c, h, w,
                                    getScale(), CVK_FMT_BF16);
  }
}

// =========================================
// LocalGenInterface
// =========================================
