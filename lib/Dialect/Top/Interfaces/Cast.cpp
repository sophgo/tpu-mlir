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
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

int64_t top::CastOp::getFLOPs() { return Module::getNumElements(output()); }

LogicalResult top::CastOp::init(InferenceParameter &p) { return success(); }
void top::CastOp::deinit(InferenceParameter &p) {}

LogicalResult top::CastOp::inference(InferenceParameter &p) {
  llvm_unreachable("Not support now");
}
