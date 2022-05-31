//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

Value top::BatchNormOp::lowering_int8_bm1684() {
  llvm_unreachable("BatchNormOp to be supported");
  return nullptr;
}

Value top::BatchNormOp::lowering_f32_bm1684() {
  llvm_unreachable("BatchNormOp to be supported");
  return nullptr;
}
