//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"

using namespace mlir;
namespace tpu_mlir {
namespace tpu {

struct StripInputQuantPattern : public OpRewritePattern<tpu::CastOp> {
  StripInputQuantPattern(MLIRContext *context)
      : OpRewritePattern<tpu::CastOp>(context) {}
  LogicalResult matchAndRewrite(tpu::CastOp op,
                                PatternRewriter &rewriter) const override {
    if (auto inputOp = op.input().getDefiningOp<top::InputOp>()) {
      if (!inputOp.getResult().hasOneUse())
        return failure();
      inputOp.getResult().setType(op.getResult().getType());
      rewriter.replaceOp(op, inputOp.getResult());
      return success();
    }
    return failure();
  };
};

struct StripOutputQuantPattern : public OpRewritePattern<tpu::CastOp> {
  StripOutputQuantPattern(MLIRContext *context)
      : OpRewritePattern<tpu::CastOp>(context) {}
  LogicalResult matchAndRewrite(tpu::CastOp op,
                                PatternRewriter &rewriter) const override {

    if (op.output().hasOneUse() &&
        isa<ReturnOp>(op.output().use_begin().getUser())) {
      rewriter.replaceOp(op, op.input());
      return success();
    }
    return failure();
  };
};

class StripIOQuantPass : public StripIOQuantBase<StripIOQuantPass> {
public:
  StripIOQuantPass() {}
  void runOnOperation() override {
    auto func = getOperation();
    if (func.getName() != "main") {
      return;
    }
    auto ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    if (quant_input)
      patterns.add<StripInputQuantPattern>(ctx);
    if (quant_output)
      patterns.add<StripOutputQuantPattern>(ctx);
    applyPatternsAndFoldGreedily(func, std::move(patterns));
    Module::updateModuleTypes(Module::getModuleOp(func));
  }
};

std::unique_ptr<OperationPass<FuncOp>> createStripIOQuant() {
  return std::make_unique<StripIOQuantPass>();
}
} // namespace tpu
} // namespace tpu_mlir
