//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

Value tpu::BufferOp::create(mlir::Operation *OwnerOp,
                            mlir::RankedTensorType &type) {
  OpBuilder builder(OwnerOp->getContext());
  builder.setInsertionPoint(OwnerOp);
  auto loc = module::getLocLike(OwnerOp, "buffer");
  auto newOp = builder.create<tpu::BufferOp>(loc, type);
  return newOp.getResult();
}
