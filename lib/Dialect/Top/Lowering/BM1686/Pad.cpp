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

Value top::PadOp::lowering_int8_bm1686(bool asymetric) {
  llvm_unreachable("PadOp to be supported");
  return nullptr;
}

Value top::PadOp::lowering_f32_bm1686() {
  llvm_unreachable("PadOp to be supported");
  return nullptr;
}

Value top::PadOp::lowering_bf16_bm1686() {
  llvm_unreachable("PadOp to be supported");
  return nullptr;
}

Value top::PadOp::lowering_f16_bm1686() {
  llvm_unreachable("PadOp to be supported");
  return nullptr;
}
