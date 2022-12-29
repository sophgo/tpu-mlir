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
void tpu::InterpOp::codegen_global_cv18xx( int64_t layer_id) {
  llvm_unreachable("Not supported now");
}
