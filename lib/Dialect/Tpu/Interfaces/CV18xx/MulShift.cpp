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
// #include "tpu_mlir/Backend/BM168x/cv18xx.h"

#include "tpu_mlir/Support/Module.h"



// using namespace tpu_mlir::backend;


void tpu::MulShiftOp::codegen_global_cv18xx( int64_t layer_id) {
  llvm_unreachable("Not supported now");
}

int64_t tpu::MulShiftOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  llvm_unreachable("Not supported now");
  return 0;
}

void tpu::MulShiftOp::codegen_local_cv18xx(int64_t n_step,
                                                 int64_t h_step) {
  llvm_unreachable("Not supported now");
}
