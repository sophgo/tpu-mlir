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
  auto ctx = OwnerOp->getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPoint(OwnerOp);
  std::string op_name = module::getName(OwnerOp).str();
  std::string buffer_name = op_name + "_buffer";
  auto nameAttr = builder.getStringAttr(buffer_name);
  auto newOp = builder.create<tpu::BufferOp>(NameLoc::get(nameAttr), type);
  return newOp.getResult();
}
