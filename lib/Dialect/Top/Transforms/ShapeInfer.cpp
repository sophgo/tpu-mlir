//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"

using namespace llvm;
using namespace mlir;

namespace tpu_mlir {
namespace top {

class ShapeInferPass : public ShapeInferBase<ShapeInferPass> {
public:
  ShapeInferPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    for (auto func : mOp.getOps<FuncOp>()) {
      func.walk([&](ShapeInterface op) { op.shape_inference(); });
    }
    module::updateModuleTypes();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createShapeInferPass() {
  return std::make_unique<ShapeInferPass>();
}
} // namespace top
} // namespace tpu_mlir
