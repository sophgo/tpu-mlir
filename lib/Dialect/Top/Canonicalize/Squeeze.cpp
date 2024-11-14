//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace tpu_mlir::top;

// unsqueeze + squeeze && in == out
struct TopFuseSqueeze : public OpRewriterPatternEx<SqueezeOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  TopFuseSqueeze(mlir::MLIRContext *context)
      : OpRewriterPatternEx<SqueezeOp>(context, "TopFuseSqueeze") {}

  LogicalResult matchAndRewriteImpl(SqueezeOp op,
                                    PatternRewriter &rewriter) const override {
    auto in_op = op.getInput().getDefiningOp();
    if (in_op->hasOneUse() && isa<UnsqueezeOp>(in_op)) {
      auto former_op = dyn_cast<UnsqueezeOp>(in_op);
      auto shape0 = module::getShape(op.getOutput());
      auto shape1 = module::getShape(former_op.getInput());
      if (shape0 != shape1) {
        return failure();
      }
      op.getOutput().replaceAllUsesWith(former_op.getInput());
      rewriter.eraseOp(op);
      rewriter.eraseOp(former_op);
      return success();
    }
    return failure();
  }
};

// squeeze scalar [1] -> [1]
struct TopSqueezeErase : public OpRewriterPatternEx<SqueezeOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  TopSqueezeErase(mlir::MLIRContext *context)
      : OpRewriterPatternEx<SqueezeOp>(context, "TopSqueezeErase") {}

  LogicalResult matchAndRewriteImpl(SqueezeOp op,
                                    PatternRewriter &rewriter) const override {
    if (!op.getIsScalar()) {
      return failure();
    }
    auto shape0 = module::getShape(op.getOutput());
    auto shape1 = module::getShape(op.getInput());
    if (shape0 != shape1) {
      return failure();
    }
    op.getOutput().replaceAllUsesWith(op.getInput());
    rewriter.eraseOp(op);
    return success();
  }
};

void SqueezeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<TopFuseSqueeze, TopSqueezeErase>(context);
}
