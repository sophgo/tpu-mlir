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

struct TopSelectToStridedSlice : public OpRewritePattern<SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const override {

    auto out_shape = module::getShape(op.getOutput());
    auto in_shape = module::getShape(op.getInput());
    int64_t dims = in_shape.size();
    auto axis_ = op.getAxis();
    auto index_ = op.getIndex();

    int64_t b_mask = -1, e_mask = -1;
    b_mask &= ~(1LL << axis_);
    e_mask &= ~(1LL << axis_);
    std::vector<float_t> starts, ends, strides;
    starts.resize(dims, 0);
    ends.resize(dims, 0);
    strides.resize(dims, 1);
    starts[axis_] = index_;
    ends[axis_] = index_ + 1;
    std::vector<Value> operands;
    operands.push_back(op.getInput());
    auto r_type = RankedTensorType::get({dims}, rewriter.getF32Type());
    operands.push_back(WeightOp::create(op, "_starts", starts, r_type));
    operands.push_back(WeightOp::create(op, "_ends", ends, r_type));
    operands.push_back(WeightOp::create(op, "_strides", strides, r_type));

    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("begin_mask", rewriter.getI64IntegerAttr(b_mask)));
    attrs.push_back(
        rewriter.getNamedAttr("end_mask", rewriter.getI64IntegerAttr(e_mask)));
    attrs.push_back(
        rewriter.getNamedAttr("ellipsis_mask", rewriter.getI64IntegerAttr(0)));
    attrs.push_back(
        rewriter.getNamedAttr("new_axis_mask", rewriter.getI64IntegerAttr(0)));
    attrs.push_back(
        rewriter.getNamedAttr("shrink_axis_mask", rewriter.getI64IntegerAttr(0)));
    rewriter.replaceOpWithNewOp<StridedSliceOp>(op, op.getResult().getType(),
        operands, attrs);
    return success();
  }
};

void SelectOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<TopSelectToStridedSlice>(context);
}
