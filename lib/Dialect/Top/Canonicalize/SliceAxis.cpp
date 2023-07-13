//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"


using namespace tpu_mlir::top;

struct SliceAxisToStridedSlice : public OpRewritePattern<SliceAxisOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SliceAxisOp op,
                                PatternRewriter &rewriter) const override {

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
    auto end_op = op.getEnd().getDefiningOp<top::WeightOp>();
    auto end = end_op.read<float>()->at(0);
    if (end < 0)
      end += in_shape[axis];
    if (end > in_shape[axis])
      end = in_shape[axis];

    std::vector<int64_t> offset(dims, 0);
    std::vector<int64_t> steps(dims, 1);
    std::vector<int64_t> ends(dims, -1);
    offset[axis] = start;
    steps[axis] = step;
    ends[axis] = end;
    auto none = module::getNoneOp(op);
    std::vector<Value> operands;
    const auto& opd = op->getOperand(0);
    operands.push_back(opd);
    operands.push_back(none);
    operands.push_back(none);
    operands.push_back(none);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("offset", rewriter.getI64ArrayAttr(offset)));
    attrs.push_back(
        rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr(steps)));
    attrs.push_back(
        rewriter.getNamedAttr("ends", rewriter.getI64ArrayAttr(ends)));
    rewriter.replaceOpWithNewOp<SliceOp>(op, op.getResult().getType(),
                                         operands, attrs);
    return success();
  }
};

void SliceAxisOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SliceAxisToStridedSlice>(context);
}
