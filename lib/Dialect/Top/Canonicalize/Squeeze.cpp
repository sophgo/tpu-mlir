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

using namespace mlir;
using namespace tpu_mlir::top;

// reshape + reshape
struct Squeeze2Reshape : public OpRewritePattern<SqueezeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SqueezeOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getOutput().getType(),
                                           op->getOperands(),
                                           std::vector<NamedAttribute>());
    return success();
  }
};

void SqueezeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<Squeeze2Reshape>(context);
}
