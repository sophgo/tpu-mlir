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
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

struct PReluToLeakRelu : public OpRewritePattern<PReluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PReluOp op,
                                PatternRewriter &rewriter) const override {
    if (module::isWeight(op.getSlope()) == false) {
      return failure();
    }
    auto num = module::getNumElements(op.getSlope());
    if (num != 1) {
      return failure();
    }
    auto slope_op = op.getSlope().getDefiningOp<top::WeightOp>();
    auto slope = slope_op.read<float>();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("alpha", rewriter.getF64FloatAttr(slope->at(0))));
    rewriter.replaceOpWithNewOp<LeakyReluOp>(op, op.getOutput().getType(),
                                             ValueRange{op.getInput()}, attrs);
    return success();
  }
};

void PReluOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<PReluToLeakRelu>(context);
}
