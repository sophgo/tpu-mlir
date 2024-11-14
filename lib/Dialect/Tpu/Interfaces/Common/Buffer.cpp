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
                            mlir::RankedTensorType &type,
                            tpu::BufferType buffer_type) {
  auto ctx = OwnerOp->getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPoint(OwnerOp);
  auto loc = module::getLocLike(OwnerOp, "buffer");
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr(
      "buffer_type", tpu::BufferTypeAttr::get(ctx, buffer_type)));
  auto newOp = builder.create<tpu::BufferOp>(loc, type, ValueRange{}, attrs);
  return newOp.getResult();
}
