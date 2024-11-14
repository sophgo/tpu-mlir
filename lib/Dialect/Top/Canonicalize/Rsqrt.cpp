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

struct RsqrtToSqrtRec : public OpRewriterPatternEx<RsqrtOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  RsqrtToSqrtRec(mlir::MLIRContext *context)
      : OpRewriterPatternEx<RsqrtOp>(context, "RsqrtToSqrtRec") {}

  LogicalResult matchAndRewriteImpl(RsqrtOp op,
                                    PatternRewriter &rewriter) const override {
    return failure(); // do not need rsqrt to sqrt
    auto name = module::getName(op.getOutput());
    auto sqrt_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_sqrt"));
    auto sqrt_op = rewriter.create<top::SqrtOp>(
        sqrt_loc, op.getOutput().getType(), ValueRange{op.getInput()});
    sqrt_op.shape_inference();
    auto rec_input = sqrt_op.getResult();
    auto reciprocal_loc = NameLoc::get(rewriter.getStringAttr(name));
    std::vector<Value> operands;
    operands.push_back(rec_input);
    auto reciprocal = rewriter.create<ReciprocalOp>(
        reciprocal_loc, op.getOutput().getType(), operands);
    op.getOutput().replaceAllUsesWith(reciprocal);
    rewriter.eraseOp(op);
    return success();
  }
};

void RsqrtOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<RsqrtToSqrtRec>(context);
}
