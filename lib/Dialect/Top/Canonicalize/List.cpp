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

/**
 * List  => Reshape + Concat
 **/
struct TopListReplace : public OpRewriterPatternEx<ListOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  TopListReplace(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ListOp>(context, "TopListReplace") {}

  LogicalResult matchAndRewriteImpl(top::ListOp op,
                                    PatternRewriter &rewriter) const {
    int num_inputs = op.getInputs().size();
    auto op_name = module::getName(op.getOperation()).str();
    std::vector<Value> concat_operands;
    for (int i = 0; i < num_inputs; i++) {
      auto in_op = op.getInputs()[i];
      auto in_ele = module::getNumElements(in_op);
      std::vector<int64_t> in_shape = {in_ele};
      std::string in_name = module::getName(in_op).str();
      auto reshape_loc =
          NameLoc::get(rewriter.getStringAttr(in_name + "_r_reshape"));
      auto reshape_type =
          RankedTensorType::get(in_shape, module::getElementType(in_op));
      std::vector<NamedAttribute> reshape_attrs;
      auto reshape_op = rewriter.create<top::ReshapeOp>(
          reshape_loc, reshape_type, ValueRange{in_op}, reshape_attrs);
      concat_operands.emplace_back(reshape_op->getResult(0));
    }
    std::vector<NamedAttribute> concat_attrs;
    concat_attrs.emplace_back(
        rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(0)));
    auto out_ele = module::getNumElements(op.getOutput());
    std::vector<int64_t> out_shape = {out_ele};
    auto concat_type = RankedTensorType::get(
        out_shape, module::getElementType(op.getInputs()[0]));
    auto concat_loc =
        NameLoc::get(rewriter.getStringAttr(op_name + "_r_concat"));
    auto concat_op = rewriter.create<top::ConcatOp>(
        concat_loc, concat_type, concat_operands, concat_attrs);

    rewriter.setInsertionPointAfter(op);
    rewriter.replaceAllUsesWith(op, concat_op->getResult(0));
    rewriter.eraseOp(op);
    return success();
  }
};

void ListOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<TopListReplace>(context);
}
