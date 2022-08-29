//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

struct TopFuseRelu : public OpRewritePattern<ReluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp op,
                                PatternRewriter &rewriter) const override {
    auto formerOp = op.input().getDefiningOp();
    if (!formerOp->getResult(0).hasOneUse())
      return failure();

    if (false == formerOp->hasTrait<SupportFuseRelu>()) {
      return failure();
    }
    auto relu_limit = op.relu_limit().convertToDouble();
    if (formerOp->hasAttr("relu_limit")) {
      auto old_limit = formerOp->getAttr("relu_limit").cast<FloatAttr>().getValueAsDouble();
      if (old_limit > 0 && relu_limit > old_limit) {
          relu_limit = old_limit;
      }
    }
    formerOp->setAttr("do_relu", rewriter.getBoolAttr(true));
    formerOp->setAttr("relu_limit", rewriter.getF64FloatAttr(relu_limit));
    formerOp->setLoc(op.getLoc());
    // remove the relu Op
    rewriter.replaceOp(op, {op.input()});
    return success();
  }
};

void ReluOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<TopFuseRelu>(context);
}
