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
    std::vector<Value> operands;
    const auto& opd = op->getOperand(0);
    operands.push_back(opd);
    auto none = module::getNoneOp(op);
    auto axis_op = op.getAxis().getDefiningOp<top::WeightOp>();
    auto axis = axis_op.read<float>()->at(0);
    if (axis < 0)
      axis += dims;
    float start = 0, end = in_shape[axis], step = 1;
    if (module::isWeight(op.getStart())) {
      auto start_op = op.getStart().getDefiningOp<top::WeightOp>();
      start = start_op.read<float>()->at(0);
      if (start < 0)
        start += in_shape[axis];
      operands.push_back(none);
    } else {
      auto start_op = op.getStart();
      operands.push_back(start_op);
    }

    if (module::isWeight(op.getEnd())) {
      auto end_op = op.getEnd().getDefiningOp<top::WeightOp>();
      end = end_op.read<float>()->at(0);
      if (end < 0)
        end += in_shape[axis];
      if (end > in_shape[axis])
        end = in_shape[axis];
      operands.push_back(none);
    } else {
      auto end_op = op.getEnd();
      operands.push_back(end_op);
    }

    if (module::isWeight(op.getStep())) {
      auto step_op = op.getStep().getDefiningOp<top::WeightOp>();
      step = step_op.read<float>()->at(0);
      operands.push_back(none);
    } else {
      auto step_op = op.getStep();
      operands.push_back(step_op);
    }
    std::vector<int64_t> offset(dims, 0);
    std::vector<int64_t> steps(dims, 1);
    std::vector<int64_t> ends(dims, -1);
    std::vector<int64_t> axes(1, -1);
    offset[axis] = start;
    steps[axis] = step;
    ends[axis] = end;
    axes[0] = axis;
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("offset", rewriter.getI64ArrayAttr(offset)));
    attrs.push_back(
        rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr(steps)));
    attrs.push_back(
        rewriter.getNamedAttr("ends", rewriter.getI64ArrayAttr(ends)));
    attrs.push_back(
        rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr(axes)));
    rewriter.replaceOpWithNewOp<SliceOp>(op, op.getResult().getType(), operands,
                                         attrs);
    return success();
  }
};

void SliceAxisOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SliceAxisToStridedSlice>(context);
}
