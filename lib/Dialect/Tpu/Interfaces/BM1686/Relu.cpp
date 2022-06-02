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
#include "tpu_mlir/Backend/BM168x/BM1686.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::ReluOp::codegen_global_int8_bm1686() {
  llvm_unreachable("Codegen to be supported");
}

void tpu::ReluOp::codegen_global_float_bm1686() {
  llvm_unreachable("Codegen to be supported");
}


// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::ReluOp::getBufferSize_bm1686(int64_t out_n, int64_t out_c,
                                          int64_t out_h, int64_t out_w,
                                          int64_t out_lmem_bytes) {
  return 0;
}

void tpu::ReluOp::codegen_local_int8_bm1686(int64_t n_step, int64_t h_step) {
  llvm_unreachable("support later");
}

void tpu::ReluOp::codegen_local_float_bm1686(int64_t n_step, int64_t h_step) {
  llvm_unreachable("support later");
}
