//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Interfaces/ShapeInterface.h"
#include "tpu_mlir/Interfaces/ShapeInterface.cpp.inc"
#include "tpu_mlir/Support/Module.h"

namespace tpu_mlir {

void common_shape_inference(mlir::Operation *op) {
  if (op->getNumResults() != 1) {
    UNREACHABLE_OP("input and output should be only one", op);
  }
  auto in = op->getOperand(0);
  auto out = op->getResult(0);
  auto in_shape = module::getShape(in);
  module::setShapeOrVerify(out, in_shape);
  auto pre_op = in.getDefiningOp();
  if (op->hasTrait<trait::ScalarConsumer>()) {
    auto context = op->getContext();
    mlir::Builder builder(context);
    auto is_scalar = module::isScalar(pre_op);
    op->setAttr("is_scalar", builder.getBoolAttr(is_scalar));
  }
}

llvm::SmallVector<int64_t> computer_broadcast_shape(mlir::Operation *op) {
  if (op->getNumResults() != 1) {
    UNREACHABLE_OP("Only supports one output and two inputs.", op);
  }
  auto lhs_shape = module::getShape(op->getOperand(0));
  auto out_shape = llvm::SmallVector<int64_t>(lhs_shape);
  if (op->getNumOperands() > 1) {
    for (int i = 1; i < op->getNumOperands(); ++i) {
      if (module::isNone(op->getOperand(i)))
        continue;
      auto hs_shape = module::getShape(op->getOperand(i));
      auto tmp_shape = llvm::SmallVector<int64_t>();
      for (auto it : llvm::zip_longest(llvm::reverse(out_shape),
                                       llvm::reverse(hs_shape))) {
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
  return out_shape;
}

void broadcast_shape_inference(mlir::Operation *op) {
  auto out_shape = computer_broadcast_shape(op);
  auto out = op->getResult(0);
  module::setShapeOrVerify(out, out_shape);
}

void broadcast_tensor_reshape(const mlir::Value &expect, mlir::Value input) {
  // insert 1 at the begin of input if dim of input is not same with expect
  if (module::isWeight(input) && module::getNumElements(input) > 1 &&
      module::getShape(input).size() != module::getShape(expect).size()) {
    auto expect_shape = module::getShape(expect);
    auto input_shape = module::getShape(input);
    llvm::SmallVector<int64_t> shape(module::getShape(expect).size(), 1);
    for (int expect_index = (int)expect_shape.size() - 1,
             input_index = (int)input_shape.size() - 1;
         expect_index >= 0; expect_index--) {
      if (input_index >= 0) {
        if (expect_shape[expect_index] == input_shape[input_index]) {
          shape[expect_index] = expect_shape[expect_index];
          input_index--;
        } else {
          if (input_shape[input_index] == 1) {
            shape[expect_index] = 1;
            input_index--;
          } else {
            shape[expect_index] = 1;
          }
        }
      } else {
        shape[expect_index] = 1;
      }
    }
    int64_t real_input_product =
        std::accumulate(input_shape.begin() + 1, input_shape.end(),
                        input_shape[0], std::multiplies<int64_t>());
    int64_t bcast_input_product = std::accumulate(
        shape.begin() + 1, shape.end(), shape[0], std::multiplies<int64_t>());
    if (real_input_product != bcast_input_product) {
      shape.clear();
      shape.resize(expect_shape.size() - input_shape.size(), 1);
      for (auto iter : module::getShape(input)) {
        shape.push_back(iter);
      }
    }

    real_input_product =
        std::accumulate(input_shape.begin() + 1, input_shape.end(),
                        input_shape[0], std::multiplies<int64_t>());
    bcast_input_product = std::accumulate(shape.begin() + 1, shape.end(),
                                          shape[0], std::multiplies<int64_t>());
    assert(real_input_product == bcast_input_product);
    assert(shape.size() == module::getShape(expect).size());
    auto newType = RankedTensorType::get(shape, module::getElementType(input));
    input.setType(newType);
  }
}

}; // namespace tpu_mlir
