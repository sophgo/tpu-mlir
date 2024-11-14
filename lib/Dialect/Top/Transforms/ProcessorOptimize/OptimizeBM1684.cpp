//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

namespace tpu_mlir {
namespace bm1684 {

class ConvertMultiInputAdd : public OpRewriterPatternEx<top::AddOp> {
public:
  ConvertMultiInputAdd(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::AddOp>(context, "ConvertMultiInputAdd") {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::AddOp op,
                      mlir::PatternRewriter &rewriter) const override {
    auto inputs = op.getInputs();
    auto name = module::getName(op.getOperation()).str();
    if (inputs.size() <= 2) {
      return failure();
    }

    // Start accumulating from the first input
    Value accumulated = inputs[0];
    auto coeffArrayAttr = op.getCoeffAttr().cast<ArrayAttr>();
    for (int i = 1; i < inputs.size(); ++i) {
      Location ori_loc = op.getLoc();
      if (i != inputs.size() - 1) {
        ori_loc =
            NameLoc::get(rewriter.getStringAttr(name + std::to_string(i)));
      }
      auto newCoeffArrayAttr =
          rewriter.getArrayAttr({coeffArrayAttr[i - 1], coeffArrayAttr[i]});
      accumulated = rewriter.create<top::AddOp>(
          ori_loc, accumulated.getType(), ValueRange{accumulated, inputs[i]},
          op.getDoReluAttr(), op.getReluLimitAttr(), newCoeffArrayAttr,
          op.getIsScalarAttr());
    }

    rewriter.replaceOp(op, accumulated);
    return success();
  }
};

} // namespace bm1684

class NoneZeroFixRowMajor : public OpRewriterPatternEx<tpu::NonZeroOp> {
public:
  NoneZeroFixRowMajor(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<tpu::NonZeroOp>(context, "NoneZeroFixRowMajor") {}

protected:
  LogicalResult
  matchAndRewriteImpl(tpu::NonZeroOp op,
                      mlir::PatternRewriter &rewriter) const override {
    const int order = op.getOrder().str() == "ColMajor" ? 0 : 1;
    if (order == 0) {
      return failure();
    }
    auto type = op.getResult().getType();
    op->setAttr("order", rewriter.getStringAttr("ColMajor"));
    auto out_shape = module::getShape(op.getOutput());
    std::vector<int64_t> new_out_shape = {out_shape[1], out_shape[0]};
    module::setShape(op.getOutput(), new_out_shape);

    rewriter.setInsertionPointAfter(op);
    auto permute_out = NameLoc::get(rewriter.getStringAttr(
        module::getName(op.getOutput()).str() + "_permute"));
    std::vector<NamedAttribute> attrs;
    std::vector<int64_t> Porder = {1, 0};
    attrs.push_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(Porder)));
    auto permute_op = rewriter.create<tpu::PermuteOp>(
        permute_out, type, ValueRange{op.getOutput(), module::getNoneOp(op)},
        attrs);
    op.getOutput().replaceAllUsesExcept(permute_op.getOutput(), permute_op);
    return success();
  }
};

namespace top {
using namespace bm1684;
void populateOptimizeBM1684Patterns(RewritePatternSet *patterns) {
  // add bm1684 optimize here
  patterns
      ->add<ConvertMultiInputAdd, ConcatToSwapDimInner, NoneZeroFixRowMajor>(
          patterns->getContext(), 8);
  patterns->add<patterns::SqueezeToReshapePattern>(patterns->getContext());
  patterns->add<patterns::UnsqueezeToReshapePattern>(patterns->getContext());
}
} // namespace top
} // namespace tpu_mlir
