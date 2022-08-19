//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

void tpu::AvgPool3DOp::codegen_global_bm1684() {
  llvm_unreachable("not supported for bm1684 pool3d");
}

int64_t tpu::AvgPool3DOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::AvgPool3DOp::codegen_local_bm1684(int64_t n_step,
                                                 int64_t h_step) {
  llvm_unreachable("support later");
}

// =========================================
// MaxPoolInterface
// =========================================

void tpu::MaxPool3DOp::codegen_global_bm1684() {
  llvm_unreachable("not supported for bm1684 pool3d");
}

int64_t tpu::MaxPool3DOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::MaxPool3DOp::codegen_local_bm1684(int64_t n_step,
                                                 int64_t h_step) {
  llvm_unreachable("support later");
}
