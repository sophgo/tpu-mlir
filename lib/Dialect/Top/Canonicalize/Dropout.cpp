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

struct TopDropout : public OpRewritePattern<DropoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DropoutOp op,
                                PatternRewriter &rewriter) const override {
    auto formerOp = op.input().getDefiningOp();
    // remove the dropout Op
    rewriter.replaceOp(op, {op.input()});
    return success();
  }
};

void DropoutOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<TopDropout>(context);
}
