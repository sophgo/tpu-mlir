//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace tpu_mlir::top;

struct TransposeToPermutePattern : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    const auto &input = op.getInput();
    auto dim0_ = op.getDim0();
    auto dim1_ = op.getDim1();
    auto dims = module::getShape(input).size();
    if (dims < 2)
      return failure();
    std::vector<int64_t> order;
    for (int i = 0; i < dims; ++i) {
      if (dim0_ == i) {
        order.push_back(dim1_);
      } else if (dim1_ == i) {
        order.push_back(dim0_);
      } else {
        order.push_back(i);
      }
    }
    // rewrite
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order)));
    rewriter.replaceOpWithNewOp<PermuteOp>(
        op, op.getResult().getType(), ValueRange{input}, attrs);
    return success();
  }
};

struct TransposeFussPattern : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    const auto &input = op.getInput();
    auto dims = module::getShape(input).size();
    if (dims >= 2)
      return failure();
    op.getOutput().replaceAllUsesWith(input);
    return success();
  }
};

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<TransposeToPermutePattern, TransposeFussPattern>(context);
}
