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
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

void tpu::MulOp::codegen_global_int8_bm1684() {
  llvm_unreachable("Codegen to be supported");
}

int64_t tpu::MulOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                         int64_t out_lmem_bytes) {
  return 0;
}

void tpu::MulOp::codegen_local_int8_bm1684(int64_t n_step, int64_t h_step) {
  llvm_unreachable("Codegen to be supported");
}
