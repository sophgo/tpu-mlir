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

void tpu::PoolMaskOp::codegen_global_cv18xx(int64_t layer_id) {

  gaddr_t ga_input = Module::getAddress(input());
  gaddr_t ga_output = Module::getAddress(output());
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  if (Quant::isUniformQuantized(output())) {
    cvi_backend_tg_pool_mask_kernel(layer_id, ga_input, ga_output, n, c, h, w,
                                    scale(), CVK_FMT_I8);
  } else {
    cvi_backend_tg_pool_mask_kernel(layer_id, ga_input, ga_output, n, c, h, w,
                                    scale(), CVK_FMT_BF16);
  }
}

// =========================================
// LocalGenInterface
// =========================================
