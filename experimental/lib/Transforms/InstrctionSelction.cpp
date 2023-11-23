//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu-mlir/Dialect/SG2260/IR/SG2260.h"
#include "tpu-mlir/Transforms/Passes.h"

using namespace mlir;

void printSG2260StructureOp(Operation *operation) {
  using namespace tpu_mlir;
  using namespace tpu_mlir::sg2260;

  auto context = operation->getContext();
  auto mm = registerTraits(context);
  for (auto [key, value] : mm) {
    llvm::errs() << key << "\n";
    value.getIndexingMaps().dump();
    value.getAccelerateMap().dump();
    value.getIteratorTypes().dump();
  }
}

namespace tpu_mlir {
class InstrctionSelctionqPass
    : public InstrctionSelctionBase<InstrctionSelctionqPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    sg2260::SG2260Dialect>();
  }
  void runOnOperation() override {
    printSG2260StructureOp(getOperation());
    getOperation().walk([](linalg::LinalgOp op) {
      op.getIndexingMaps().dump();
      llvm::outs() << "\n";
    });
  }
};

std::unique_ptr<Pass> createInstrctionSelctionPass() {
  return std::make_unique<InstrctionSelctionqPass>();
}
} // namespace tpu_mlir
