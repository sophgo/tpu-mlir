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
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Float16.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

Value tpu::BufferOp::create(mlir::Operation *OwnerOp,
                            mlir::RankedTensorType &type) {
  auto ctx = OwnerOp->getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPoint(OwnerOp);
  std::string op_name = Module::getName(OwnerOp).str();
  std::string buffer_name = op_name + "_buffer";
  auto nameAttr = builder.getStringAttr(buffer_name);
  auto newOp = builder.create<tpu::BufferOp>(NameLoc::get(nameAttr), type);
  return newOp.getResult();
}
