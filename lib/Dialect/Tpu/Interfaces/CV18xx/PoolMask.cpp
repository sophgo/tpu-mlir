//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"

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
