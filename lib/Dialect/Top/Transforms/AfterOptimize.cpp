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

template <typename TyOp>
struct RemoveUnuseOutput : public OpRewritePattern<TyOp> {
  using OpRewritePattern<TyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TyOp op,
                                PatternRewriter &rewriter) const override {
    for (Value out : op.getResults()) {
      if (out.getUsers().empty()) {
        out.setType(mlir::NoneType::get(rewriter.getContext()));
      }
    }
    return success();
  }
};

class AfterOptimizePass : public AfterOptimizeBase<AfterOptimizePass> {
public:
  AfterOptimizePass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    MLIRContext *ctx = &getContext();
    // remove unuse output
    RewritePatternSet patterns(ctx);
    patterns.add<RemoveUnuseOutput<top::LSTMOp>, RemoveUnuseOutput<top::GRUOp>,
                 RemoveUnuseOutput<top::LayerNormOp>>(ctx);
    applyPatternsAndFoldGreedily(mOp, std::move(patterns));
    // mark flops
    int64_t flops = 0;
    for (auto func : mOp.getOps<FuncOp>()) {
      func.walk([&](FlopsInterface op) { flops += op.getFLOPs(); });
    }
    module::setFLOPs(flops);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAfterOptimizePass() {
  return std::make_unique<AfterOptimizePass>();
}
} // namespace top
} // namespace tpu_mlir
