//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace tpu_mlir::top;

struct SliceAxisToStridedSlice : public OpRewriterPatternEx<SliceAxisOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  SliceAxisToStridedSlice(mlir::MLIRContext *context)
      : OpRewriterPatternEx<SliceAxisOp>(context, "SliceAxisToStridedSlice") {}

  LogicalResult matchAndRewriteImpl(SliceAxisOp op,
                                    PatternRewriter &rewriter) const override {

    auto in_shape = module::getShape(op.getInput());
    int64_t dims = in_shape.size();
    std::vector<Value> operands;
    const auto &opd = op->getOperand(0);
    operands.push_back(opd);
    auto none_op = module::getNoneOp(op);
    auto axis_op = op.getAxis().getDefiningOp<top::WeightOp>();
    auto axis_dims = module::getShape(axis_op)[0];
    auto axis_data = axis_op.read_as_float();
    std::vector<int64_t> axis(axis_dims, 0);
    for (int i = 0; i < axis_dims; i++) {
      axis[i] = axis_data->at(i);
      if (axis[i] < 0)
        axis[i] += dims;
    }
    std::vector<int64_t> offset(dims, 0);
    std::vector<int64_t> steps(dims, 1);
    std::vector<int64_t> ends(dims, std::numeric_limits<int64_t>::max());
    if (!module::isNone(op.getStart())) {
      if (module::isWeight(op.getStart())) {
        auto start_op = op.getStart().getDefiningOp<top::WeightOp>();
        auto start_data = start_op.read_as_float();
        for (int i = 0; i < axis_dims; i++) {
          auto axis = axis_data->at(i);
          offset[axis] = start_data->at(i);
        }
        operands.push_back(none_op);
      } else {
        auto start_op = op.getStart();
        operands.push_back(start_op);
      }
    }

    if (!module::isNone(op.getEnd())) {
      if (module::isWeight(op.getEnd())) {
        auto end_op = op.getEnd().getDefiningOp<top::WeightOp>();
        auto end_data = end_op.read_as_float();
        for (int i = 0; i < axis_dims; i++) {
          auto axis = axis_data->at(i);
          ends[axis] = end_data->at(i);
        }
        operands.push_back(none_op);
      } else {
        auto end_op = op.getEnd();
        operands.push_back(end_op);
      }
    }

    if (!module::isNone(op.getStep())) {
      ASSERT_OP(module::isWeight(op.getStep()), op);
      auto step_op = op.getStep().getDefiningOp<top::WeightOp>();
      auto step_data = step_op.read_as_float();
      for (int i = 0; i < axis_dims; i++) {
        auto axis = axis_data->at(i);
        steps[axis] = step_data->at(i);
        ASSERT_OP(steps[axis] != 0, op);
      }
      operands.push_back(none_op);
    } else {
      auto step_op = op.getStep();
      operands.push_back(step_op);
    }
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("offset", rewriter.getI64ArrayAttr(offset)));
    attrs.push_back(
        rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr(steps)));
    attrs.push_back(
        rewriter.getNamedAttr("ends", rewriter.getI64ArrayAttr(ends)));
    if ((!module::isNone(op.getStart()) && !module::isWeight(op.getStart())) ||
        (!module::isNone(op.getEnd()) && !module::isWeight(op.getEnd())) ||
        (!module::isNone(op.getStep()) && !module::isWeight(op.getStep()))) {
      attrs.push_back(
          rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr(axis)));
    } else {
      std::vector<int64_t> axes(dims, 0);
      for (int i = 0; i < dims; i++) {
        axes[i] = i;
      }
      attrs.push_back(
          rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr(axes)));
    }
    rewriter.replaceOpWithNewOp<SliceOp>(op, op.getResult().getType(), operands,
                                         attrs);
    return success();
  }
};

void SliceAxisOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SliceAxisToStridedSlice>(context);
}
