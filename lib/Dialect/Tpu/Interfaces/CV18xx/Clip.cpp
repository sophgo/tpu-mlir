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

// int8
void tpu::ClipOp::codegen_global_cv18xx(int64_t layer_id) {
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  assert(!module::isUniformQuantized(getOutput()) && "Not support int8 Clip.");
  float coeffs[2];
  coeffs[0] = this->getMax().convertToDouble();
  coeffs[1] = this->getMin().convertToDouble();

  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  gaddr_t ga_inputs[1];
  ga_inputs[0] = ga_input;

  cvi_backend_tg_bf16_eltwise_min_max_kernel(layer_id, ga_inputs, ga_output, 1,
                                             n, c, h, w, false, false, 0, 0,
                                             coeffs);
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::ClipOp::getBufferSize_cv18xx(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  llvm_unreachable("Not supported now");
  return 0;
}

void tpu::ClipOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step) {
  llvm_unreachable("Not supported now");
}
