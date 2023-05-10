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
  if (op->getNumResults() != 1) {
    op->dump();
    llvm_unreachable("Only supports one output and two inputs.");
  }
  auto lhs_shape = module::getShape(op->getOperand(0));
  auto out_shape = llvm::SmallVector<int64_t>(lhs_shape);
  if (op->getNumOperands() > 1) {
    for (int i = 1; i < op->getNumOperands(); ++i) {
      if (module::isNone(op->getOperand(i)))
        continue;
      auto hs_shape = module::getShape(op->getOperand(i));
      auto tmp_shape = llvm::SmallVector<int64_t>();
      for (auto it :
        llvm::zip_longest(llvm::reverse(out_shape), llvm::reverse(hs_shape))) {
        if (std::get<0>(it) && std::get<0>(it) != 1) {
          tmp_shape.push_back(std::get<0>(it).value());
        } else {
          if (std::get<1>(it))
            tmp_shape.push_back(std::get<1>(it).value());
          else
            tmp_shape.push_back(std::get<0>(it).value());
        }
      }
      out_shape = llvm::SmallVector<int64_t>(llvm::reverse(tmp_shape));
    }
  }
  auto out = op->getResult(0);
  module::setShapeOrVerify(out, out_shape);
}

void broadcast_tensor_reshape(const mlir::Value &expect, mlir::Value &input) {
  // insert 1 at the begin of input if dim of input is not same with expect
  if (module::isWeight(input) && module::getNumElements(input) > 1 &&
      module::getShape(input).size() != module::getShape(expect).size()) {
    llvm::SmallVector<int64_t> shape(
        module::getShape(expect).size() - module::getShape(input).size(), 1);
    for (auto iter : module::getShape(input)) {
      shape.push_back(iter);
    }
    auto newType = RankedTensorType::get(shape, module::getElementType(input));
    input.setType(newType);
  }
}
}; // namespace tpu_mlir
