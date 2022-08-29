//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
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
