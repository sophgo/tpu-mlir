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
using namespace tpu_mlir::trait;

struct PReluToLeakRelu : public OpRewriterPatternEx<PReluOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  PReluToLeakRelu(mlir::MLIRContext *context)
      : OpRewriterPatternEx<PReluOp>(context, "PReluToLeakRelu") {}

  LogicalResult matchAndRewriteImpl(PReluOp op,
                                    PatternRewriter &rewriter) const override {
    if (module::isWeight(op.getSlope()) == false) {
      return failure();
    }
    auto num = module::getNumElements(op.getSlope());
    if (num != 1) {
      return failure();
    }
    auto slope_op = op.getSlope().getDefiningOp<top::WeightOp>();
    auto slope = slope_op.read_as_float();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("alpha", rewriter.getF64FloatAttr(slope->at(0))));
    rewriter.replaceOpWithNewOp<LeakyReluOp>(op, op.getOutput().getType(),
                                             ValueRange{op.getInput()}, attrs);
    return success();
  }
};

// [4, 3, 24, 24] * [3] => [4, 3, 24, 24] * [1, 3, 1, 1]
struct PReluReshape : public OpRewriterPatternEx<PReluOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  PReluReshape(mlir::MLIRContext *context)
      : OpRewriterPatternEx<PReluOp>(context, "PReluReshape") {}

  LogicalResult matchAndRewriteImpl(PReluOp op,
                                    PatternRewriter &rewriter) const override {
    if (module::isWeight(op.getSlope()) == false) {
      return failure();
    }
    auto slope = op.getSlope();
    auto num = module::getNumElements(slope);
    if (num == 1) {
      // to leakyrelu
      return failure();
    }
    auto in_shape = module::getShape(op.getInput());
    auto num_dims = in_shape.size();
    if (num_dims == 1 || num != in_shape[1]) {
      return failure();
    }
    auto slope_shape = module::getShape(slope);
    if (num_dims == slope_shape.size()) {
      return failure();
    }
    std::vector<int64_t> new_shape(num_dims, 1);
    new_shape[1] = num;
    module::setShape(slope, new_shape);
    return success();
  }
};

void PReluOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<PReluToLeakRelu, PReluReshape>(context);
}
