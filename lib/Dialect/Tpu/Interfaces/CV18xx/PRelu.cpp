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

void tpu::PReluOp::codegen_global_cv18xx(void* ctx, int64_t layer_id) {
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  gaddr_t ga_input = Module::getAddress(input());
  gaddr_t ga_output = Module::getAddress(output());
  gaddr_t ga_slope =  Module::getAddress(this->slope());
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  if (Quant::isUniformQuantized(output())) {
    int LE_scale = this->rshift();
    cvi_backend_tg_fixed_prelu_kernel(*backend_ctx, layer_id, ga_input, ga_output, ga_slope,
                                        n, c, h, w, 0, 0, LE_scale);
  } else {
    cvi_backend_tg_bf16_prelu_kernel(*backend_ctx, layer_id, ga_input, ga_output,
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
