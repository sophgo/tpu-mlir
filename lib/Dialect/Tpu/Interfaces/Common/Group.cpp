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
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

Operation* tpu::GroupOp::getRefOp(Value v) {
  auto result = v.cast<OpResult>();
  auto &op = body().front().back();
  auto yieldOp = dyn_cast<tpu::YieldOp>(op);
  assert(yieldOp);
  auto storeOp = yieldOp.getOperand(result.getResultNumber()).getDefiningOp<tpu::StoreOp>();
  return storeOp.input().getDefiningOp();
}
