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

Value top::SoftmaxOp::lowering_int8_bm1686() {
  llvm_unreachable("SoftmaxOp to be supported");
  return nullptr;
}

Value top::SoftmaxOp::lowering_fp(llvm::StringRef mode) {
  llvm_unreachable("SoftmaxOp to be supported");
  return nullptr;
}
