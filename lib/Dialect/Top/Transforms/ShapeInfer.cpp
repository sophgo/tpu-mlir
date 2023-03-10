//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/PatternMatch.h"

using namespace llvm;
using namespace mlir;

namespace tpu_mlir {
namespace top {

class TupleFusePattern : public OpRewritePattern<TupleOp> {
public:
  using OpRewritePattern<TupleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TupleOp op,
                                PatternRewriter &rewriter) const override {
    auto out = op.getOutput();
    for (auto user : op->getUsers()) {
      std::vector<Value> operands;
      for (auto opd : user->getOperands()) {
        if (opd == out) {
          for (auto v : op.getOperands()) {
            operands.push_back(v);
          }
        } else {
          operands.push_back(opd);
        }
      }
      user->setOperands(operands);
    }
    op.erase();
    return success();
  }
};

class ShapeInferPass : public ShapeInferBase<ShapeInferPass> {
public:
  ShapeInferPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    auto ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<TupleFusePattern>(ctx);
    applyPatternsAndFoldGreedily(mOp, std::move(patterns));
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
