//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"


using namespace tpu_mlir::top;

// reshape + reshape
struct Squeeze2Reshape : public OpRewritePattern<SqueezeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SqueezeOp op,
                                PatternRewriter &rewriter) const override {
    // rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getOutput().getType(),
    //                                       op->getOperands(),
    //                                       std::vector<NamedAttribute>());
    return failure();

  }
};

// unsqueeze + squeeze && in == out
struct TopFuseSqueeze : public OpRewritePattern<SqueezeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SqueezeOp op,
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

void SqueezeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<Squeeze2Reshape, TopFuseSqueeze>(context);
}
