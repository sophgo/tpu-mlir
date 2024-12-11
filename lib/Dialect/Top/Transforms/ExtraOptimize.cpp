//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "tpu_mlir/Support/Patterns.h"

using namespace llvm;

namespace tpu_mlir {
namespace top {
template <typename OpTy>
struct RemoveUnuseOutput : public OpRewriterPatternEx<OpTy> {
public:
  RemoveUnuseOutput(mlir::MLIRContext *context)
      : OpRewriterPatternEx<OpTy>(context) {}

protected:
  mlir::LogicalResult
  matchAndRewriteImpl(OpTy op, mlir::PatternRewriter &rewriter) const override {
    for (Value out : op.getResults()) {
      if (out.getUsers().empty() && !isa<tpu::TopKOp, top::TopKOp>(op)) {
        out.setType(mlir::NoneType::get(rewriter.getContext()));
      }
    }
    return success();
  }
};

class ExtraOptimizePass : public ExtraOptimizeBase<ExtraOptimizePass> {
public:
  ExtraOptimizePass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    MLIRContext *ctx = &getContext();
    // remove unuse output
    RewritePatternSet patterns(ctx);
    patterns.add<RemoveUnuseOutput<top::LSTMOp>, RemoveUnuseOutput<top::GRUOp>,
                 patterns::FuseSameOp, patterns::InputReshape>(ctx);
    applyPatternsAndFoldGreedily(mOp, std::move(patterns));
    // mark flops
    int64_t flops = 0;
    for (auto func : mOp.getOps<FuncOp>()) {
      func.walk([&](FlopsInterface op) { flops += op.getFLOPs(); });
    }
    // weight fold after top optimized
    for (auto func : mOp.getOps<FuncOp>()) {
      func.walk([&](InferenceInterface op) { WeightFolder(op); });
    }
    module::setFLOPs(flops);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createExtraOptimizePass() {
  return std::make_unique<ExtraOptimizePass>();
}
} // namespace top
} // namespace tpu_mlir
