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

int64_t top::ConcatOp::getFLOPs() { return 0; }

LogicalResult top::ConcatOp::init(InferenceParameter &p) { return success(); }
void top::ConcatOp::deinit(InferenceParameter &p) {}

LogicalResult top::ConcatOp::inference(InferenceParameter &p) {
  llvm_unreachable("ConcatOp to be supported");
  return success();
}
