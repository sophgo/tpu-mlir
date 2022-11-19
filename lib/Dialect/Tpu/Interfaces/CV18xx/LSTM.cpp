//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

// #include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

// ======================================
// WeightReorderInterface
// ======================================

void tpu::LSTMOp::weight_reorder_bf16_cv18xx() {
  llvm_unreachable("Not supported now");
}
void tpu::LSTMOp::weight_reorder_int8_cv18xx() {
  llvm_unreachable("Not supported now");
}

// =========================================
// GlobalGenInterface
// =========================================
void tpu::LSTMOp::codegen_global_cv18xx( int64_t layer_id) {
  llvm_unreachable("Not supported now");
}
