//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"


using namespace tpu_mlir::top;

// reshape + reshape
struct TopFuseReshape : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto in_op = op.getInput().getDefiningOp();
    if (in_op->hasOneUse() && isa<ReshapeOp>(in_op)) {
      op->setOperand(0, in_op->getOperand(0));
      rewriter.eraseOp(in_op);
      return success();
    }
    return failure();
  }
};

// reshape (in == out)
struct TopFuseReshape2 : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto shape0 = module::getShape(op.getOutput());
    auto shape1 = module::getShape(op.getInput());
    if (shape0 != shape1) {
      return failure();
    }
    op.getOutput().replaceAllUsesWith(op.getInput());
    rewriter.eraseOp(op);
    return success();
  }
};

// add + reshape + add + reshape
struct TopFuseReshape3 : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto in = op.getInput();
    auto add_op = dyn_cast<AddOp>(in.getDefiningOp());
    if (!(add_op && add_op->hasOneUse() && in.hasOneUse())) {
      return failure();
    }
    if (add_op.getNumOperands() != 2) {
      return failure();
    }
    auto a_in = add_op.getInputs()[0];
    auto b_in = add_op.getInputs()[1];
    if (!module::isWeight(b_in)) {
      return failure();
    }
    if (!a_in.hasOneUse()) {
      return failure();
    }
    if (!b_in.hasOneUse()) {
      return failure();
    }
    if (!isa<ReshapeOp>(a_in.getDefiningOp())) {
      return failure();
    }
    std::vector<int64_t> shape0 = module::getShape(op.getInput());
    std::vector<int64_t> shape1 = module::getShape(op.getOutput());
    if (shape0.size() != 1 + shape1.size()) {
      return failure();
    }
    if (!std::equal(shape0.begin() + 1, shape0.end(), shape1.begin())) {
      return failure();
    }
    if (shape0[0] != 1) {
      return failure();
    }
    std::vector<int64_t> a_shape = module::getShape(a_in);
    std::vector<int64_t> b_shape = module::getShape(b_in);
    if (a_shape[0] != 1 || b_shape[0] != 1) {
      return failure();
    }
    a_shape.erase(a_shape.begin());
    b_shape.erase(b_shape.begin());
    shape0.erase(shape0.begin());
    auto b_type = RankedTensorType::get(b_shape, module::getElementType(b_in));
    b_in.setType(b_type);
    auto a_type = RankedTensorType::get(a_shape, module::getElementType(a_in));
    a_in.setType(a_type);
    auto in_type = RankedTensorType::get(shape0, module::getElementType(in));
    in.setType(in_type);
    return success();
  }
};

// reshape<(0,ng,-1)> + instance_norm -> group_norm<ng> + reshape
struct ReshapeInstanceNormPattern : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // check param
    auto output = op.getOutput();
    if (!output.hasOneUse())
      return failure();
    auto next_op_ = *output.getUsers().begin();
    if (!isa<InstanceNormOp>(next_op_))
      return failure();
    auto next_op = dyn_cast<InstanceNormOp>(next_op_);
    auto ishape = module::getShape(op.getInput());
    auto oshape = module::getShape(op.getOutput());
    if (ishape[0] != oshape[0])
      return failure();
    if (ishape[1] < oshape[1])
      return failure();
    // rewrite now !
    const auto num_groups = oshape[1];
    auto input = op.getInput();
    std::vector<NamedAttribute> attrs;
    next_op->setAttr("num_groups", rewriter.getI64IntegerAttr(num_groups));
    for (auto &attr : next_op->getAttrs()) {
      attrs.push_back(attr);
    }
    std::vector<Value> gn_opds = {input, next_op->getOperand(1), next_op->getOperand(2)};
    auto gn_out_type =
      RankedTensorType::get(ishape, module::getElementType(input));
    auto loc = NameLoc::get(
      rewriter.getStringAttr(module::getName(input).str() + "_GroupNorm"));
    rewriter.setInsertionPointAfterValue(input);
    auto gn_op = rewriter.create<GroupNormOp>(
      loc, gn_out_type, gn_opds, attrs);
    rewriter.replaceOp(op, {gn_op});
    auto gn_output = gn_op.getOutput();
    rewriter.setInsertionPointAfterValue(gn_output);
    auto new_reshape_out_type = next_op.getResult().getType();
    rewriter.replaceOpWithNewOp<ReshapeOp>(
      next_op, new_reshape_out_type, gn_output, std::vector<NamedAttribute>());
    return failure();
  }
};

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<TopFuseReshape,
                 TopFuseReshape2,
                 TopFuseReshape3,
                 ReshapeInstanceNormPattern>(context);
}
