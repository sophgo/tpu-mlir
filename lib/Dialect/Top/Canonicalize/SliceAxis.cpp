//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir::top;

struct SliceAxisToStridedSlice : public OpRewritePattern<SliceAxisOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SliceAxisOp op,
                                PatternRewriter &rewriter) const override {

    auto out_shape = module::getShape(op.getOutput());
    auto in_shape = module::getShape(op.getInput());
    int64_t dims = in_shape.size();

    auto axis_op = op.getAxis().getDefiningOp<top::WeightOp>();
    auto axis = axis_op.read<float>()->at(0);
    if (axis < 0)
      axis += dims;
    auto start_op = op.getStart().getDefiningOp<top::WeightOp>();
    auto start = start_op.read<float>()->at(0);
    if (start < 0)
      start += in_shape[axis];
    auto step_op = op.getStep().getDefiningOp<top::WeightOp>();
    auto step = step_op.read<float>()->at(0);

    std::vector<int64_t> offset(dims, 0);
    std::vector<int64_t> steps(dims, 1);
    offset[axis] = start;
    steps[axis] = step;
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("offset", rewriter.getI64ArrayAttr(offset)));
    attrs.push_back(
        rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr(steps)));
    rewriter.replaceOpWithNewOp<SliceOp>(op, op.getResult().getType(),
                                         op.getInput(), attrs);
    return success();
  }
};

void SliceAxisOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SliceAxisToStridedSlice>(context);
}
