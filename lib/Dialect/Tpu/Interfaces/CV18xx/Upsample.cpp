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



using namespace tpu_mlir::backend;


// =========================================
// GlobalGenInterface
// =========================================
void tpu::UpsampleOp::codegen_global_cv18xx( int64_t layer_id) {

  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  auto scale_h = this->getScaleH();
  auto scale_w = this->getScaleW();
  if (module::isUniformQuantized(getOutput())) {
    cvi_backend_tg_upsample_kernel( layer_id, ga_input, ga_output, n,
                                  c, h, w, scale_h, scale_w, CVK_FMT_I8);
  } else {
    cvi_backend_tg_upsample_kernel( layer_id, ga_input, ga_output, n,
                                  c, h, w, scale_h, scale_w, CVK_FMT_BF16);
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::UpsampleOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::UpsampleOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step, int64_t layer_id) {
  llvm_unreachable("Not supported now");
}
