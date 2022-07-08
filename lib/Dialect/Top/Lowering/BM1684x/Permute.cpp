//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

Value top::PermuteOp::lowering_int8_bm1684x(bool asymmetric) {
  llvm_unreachable("PermuteOp to be supported");
  return nullptr;
}

Value top::PermuteOp::lowering_f32_bm1684x() {
  llvm_unreachable("PermuteOp to be supported");
  return nullptr;
}

Value top::PermuteOp::lowering_bf16_bm1684x() {
  llvm_unreachable("PermuteOp to be supported");
  return nullptr;
}

Value top::PermuteOp::lowering_f16_bm1684x() {
  llvm_unreachable("PermuteOp to be supported");
  return nullptr;
}

Value top::PermuteOp::lowering_quant_bm1684x() {
  llvm_unreachable("PermuteOp not support now");
}
