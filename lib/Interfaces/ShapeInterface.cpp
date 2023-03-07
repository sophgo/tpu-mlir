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

void broadcast_shape_inference(mlir::Operation *op) {
  if (op->getNumResults() != 1 && op->getNumOperands() != 2) {
    op->dump();
    llvm_unreachable("Only supports one output and two inputs.");
  }
  auto lhs_shape = module::getShape(op->getOperand(0));
  auto rhs_shape = module::getShape(op->getOperand(1));
  auto out_shape = llvm::SmallVector<int64_t>();
  for (auto it :
       llvm::zip_longest(llvm::reverse(lhs_shape), llvm::reverse(rhs_shape))) {
    if (std::get<0>(it) && std::get<0>(it) != 1) {
      out_shape.push_back(std::get<0>(it).value());
    } else {
      if (std::get<1>(it))
        out_shape.push_back(std::get<1>(it).value());
      else
        out_shape.push_back(std::get<0>(it).value());
    }
  }
  out_shape = llvm::SmallVector<int64_t>(llvm::reverse(out_shape));
  auto out = op->getResult(0);
  module::setShapeOrVerify(out, out_shape);
}

}; // namespace tpu_mlir
