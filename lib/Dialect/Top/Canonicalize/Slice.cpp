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

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

struct SplitSlicePattern : public OpRewritePattern<SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SliceOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasOneUse()) {
      return failure();
    }
    const auto users = op->getUsers();
    for (const auto user: users) {
      if (!isa<SliceOp>(user)) {
        return failure();
      }
    }
    const auto& opd = op->getOperand(0);
    const auto& res = op.getResult();
    const auto& offset = op->getAttr("offset");
    const auto& steps = op->getAttr("steps");
    rewriter.setInsertionPointAfterValue(opd);
    for (const auto user: users) {
      const std::string name_slice = module::getName(user->getOperand(0)).str() + "_slice";
      const auto& loc_slice = NameLoc::get(rewriter.getStringAttr(name_slice));
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("offset", offset));
      attrs.push_back(rewriter.getNamedAttr("steps", steps));
      auto slice_op = rewriter.create<SliceOp>(loc_slice, res.getType(), ValueRange(opd), attrs);
      auto slice_result_var = slice_op.getResult();
      user->eraseOperand(0);
      user->insertOperands(0, slice_result_var);
    }
    return success();
  }
};

struct MergeSlicePattern : public OpRewritePattern<SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SliceOp op,
                                PatternRewriter &rewriter) const override {

    auto in_op = op.getInput().getDefiningOp();
    if (!isa<SliceOp>(in_op) || in_op->hasOneUse() == false) {
      return failure();
    }
    auto output_shape = module::getShape(op.getOutput());
    auto num_dims = output_shape.size();
    auto in_slice = cast<SliceOp>(in_op);
    auto cur_offset = module::getI64Array(op.getOffset());
    auto cur_steps = module::getI64Array(op.getSteps());
    auto in_offset = module::getI64Array(in_slice.getOffset());
    auto in_steps = module::getI64Array(in_slice.getSteps());

    std::vector<int64_t> new_offset(num_dims, 0);
    std::vector<int64_t> new_steps(num_dims, 1);
    for (int i = 0; i < num_dims; i++) {
      auto cur_off = cur_offset->at(i);
      auto cur_s = cur_steps->at(i);
      assert(cur_s > 0);
      auto in_off = in_offset->at(i);
      auto in_s = in_steps->at(i);
      assert(in_s > 0);
      new_offset[i] = in_off + cur_off * in_s;
      new_steps[i] = in_s * cur_s;
    }
    op->setAttr("offset", rewriter.getI64ArrayAttr(new_offset));
    op->setAttr("steps", rewriter.getI64ArrayAttr(new_steps));
    op->setOperand(0, in_slice.getInput());
    rewriter.eraseOp(in_op);
    return success();
  }
};

struct NoUseSlicePattern : public OpRewritePattern<SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SliceOp op,
                                PatternRewriter &rewriter) const override {

    auto in_shape = module::getShape(op.getInput());
    auto out_shape = module::getShape(op.getOutput());
    auto steps = module::getI64Array(op.getSteps());
    for (int i = 0; i < in_shape.size(); ++i) {
      if (steps->at(i) == -1)
        return failure();
    }
    if (in_shape.size() != out_shape.size()) {
      return failure();
    }
    for (auto it : llvm::zip(in_shape, out_shape)) {
      if (std::get<0>(it) != std::get<1>(it)) {
        return failure();
      }
    }
    op.getOutput().replaceAllUsesWith(op.getInput());
    return success();
  }
};

struct TopSliceToReverse : public OpRewritePattern<SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SliceOp op,
                                PatternRewriter &rewriter) const override {
    auto in_shape = module::getShape(op.getInput());
    auto output_shape = module::getShape(op.getOutput());
    auto in_dims = in_shape.size();
    auto Steps = op.getSteps();
    int reverse_count = 0;
    int reverse_dim = 0;
    auto steps = module::getI64Array(op.getSteps());
    for (int i = 0; i < in_dims; i++) {
      if (steps->at(i) == -1 && in_shape[i] == output_shape[i]) {
        reverse_count++;
        reverse_dim = i;
      }
    }
    if (reverse_count != 1)
      return failure();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("axis", rewriter.getI64IntegerAttr(reverse_dim)));
    rewriter.replaceOpWithNewOp<ReverseOp>(op, op.getResult().getType(),
                                           op.getInput(), attrs);
    return success();
  }
};

void SliceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<NoUseSlicePattern,
                 SplitSlicePattern,
                 MergeSlicePattern,
                 TopSliceToReverse>(context);
}
