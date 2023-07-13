//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/Patterns.h"


using namespace tpu_mlir::top;

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

// merge some tanh and power(x,3) comprised gelu to gelu, first found in pytorch traced gpt2
struct MergeGeluPattern : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    MulOp mul_op = dyn_cast<MulOp>(op.getInput().getDefiningOp());
    if (mul_op == NULL || !mul_op.getOutput().hasOneUse())
      return failure();

    MulConstOp mulconst_op = NULL;
    AddConstOp addconst_op = NULL;

    for (auto in:mul_op.getInputs()) {
      if (isa<MulConstOp>(in.getDefiningOp()))
        mulconst_op = dyn_cast<MulConstOp>(in.getDefiningOp());
      else if (isa<AddConstOp>(in.getDefiningOp()))
        addconst_op = dyn_cast<AddConstOp>(in.getDefiningOp());
      else
        return failure();
    }
    if (!mulconst_op.getOutput().hasOneUse() || !addconst_op.getOutput().hasOneUse())
      return failure();

    TanhOp tanh_op = NULL;
    if (!isa<TanhOp>(addconst_op.getInput().getDefiningOp()))
      return failure();
    else
      tanh_op = dyn_cast<TanhOp>(addconst_op.getInput().getDefiningOp());
    if (!tanh_op.getOutput().hasOneUse())
      return failure();

    MulConstOp mulconst_op1 = NULL;
    AddOp add_op = NULL;
    if (!isa<MulConstOp>(tanh_op.getInput().getDefiningOp()))
      return failure();
    else
      mulconst_op1 = dyn_cast<MulConstOp>(tanh_op.getInput().getDefiningOp());
    if (!isa<AddOp>(mulconst_op1.getInput().getDefiningOp()))
      return failure();
    else
      add_op = dyn_cast<AddOp>(mulconst_op1.getInput().getDefiningOp());
    if (!mulconst_op1.getOutput().hasOneUse() || !add_op.getOutput().hasOneUse())
      return failure();

    MulConstOp mulconst_op2 = NULL;
    PowOp pow_op = NULL;
    ReshapeOp reshape_op = NULL;
    for (auto in:add_op.getInputs()) {
      if (isa<MulConstOp>(in.getDefiningOp()))
        mulconst_op2 = dyn_cast<MulConstOp>(in.getDefiningOp());
      else if (isa<ReshapeOp>(in.getDefiningOp()))
        reshape_op = dyn_cast<ReshapeOp>(in.getDefiningOp());
      else
        return failure();
    }
    if (!isa<PowOp>(mulconst_op2.getInput().getDefiningOp()))
        return failure();
    else
        pow_op = dyn_cast<PowOp>(mulconst_op2.getInput().getDefiningOp());
    if (!mulconst_op2.getOutput().hasOneUse() || !pow_op.getOutput().hasOneUse())
      return failure();

    if (pow_op.getInput().getDefiningOp() != reshape_op || mulconst_op.getInput().getDefiningOp() != reshape_op)
      return failure();
    int cnt = 0;
    int all = 0;
    for (auto out:reshape_op.getOutput().getUsers()) {
      if (out == mulconst_op || out == pow_op || out == add_op)
        cnt ++;
      all ++;
    }
    if (cnt != 3 || all != 3)
      return failure();
    if (pow_op.getExponent().convertToDouble() != 3.0 || fabs(mulconst_op2.getConstVal().convertToDouble()-0.044714998453855515)>1e-4 ||
        addconst_op.getConstVal().convertToDouble() != 1.0 || fabs(mulconst_op1.getConstVal().convertToDouble()-0.79788458347320556)>1e-4 ||
        fabs(mulconst_op.getConstVal().convertToDouble()-0.5)>1e-4)
      return failure();
    rewriter.replaceOpWithNewOp<GELUOp>(op, op.getResult().getType(),
             ValueRange{reshape_op.getInput()});
    return success();
  }
};

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<patterns::FuseRepeatPattern<top::ReshapeOp>,
                 TopFuseReshape2,
                 TopFuseReshape3,
                 ReshapeInstanceNormPattern,
                 MergeGeluPattern>(context);
}
