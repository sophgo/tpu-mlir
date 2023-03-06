//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "tpu_mlir/Interfaces/ShapeInterface.h"
#include "tpu_mlir/Interfaces/ShapeInterface.cpp.inc"
#include "tpu_mlir/Support/Module.h"

namespace tpu_mlir {

void common_shape_inference(mlir::Operation *op) {
  if (op->getNumResults() != 1) {
    op->dump();
    llvm_unreachable("input and output should be only one");
  }
  auto in = op->getOperand(0);
  auto out = op->getResult(0);
  auto in_shape = module::getShape(in);
  module::setShapeOrVerify(out, in_shape);
}

}; // namespace tpu_mlir
