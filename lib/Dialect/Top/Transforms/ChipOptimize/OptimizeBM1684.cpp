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

class ConvertMultiInputAdd : public OpRewritePattern<top::AddOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(top::AddOp op,
                                PatternRewriter &rewriter) const override {
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
          op.getDoReluAttr(), op.getReluLimitAttr(), newCoeffArrayAttr);
    }

    rewriter.replaceOp(op, accumulated);
    return success();
  }
};

} // namespace bm1684

namespace top {
using namespace bm1684;
void populateOptimizeBM1684Patterns(RewritePatternSet *patterns) {
  // add bm1684 optimize here
  patterns->add<patterns::ConvertPattern<top::SqueezeOp, top::ReshapeOp>,
                patterns::ConvertPattern<top::UnsqueezeOp, top::ReshapeOp>,
                ConvertScaleOp, ConvertMultiInputAdd>(patterns->getContext(),
                                                      8);
}

} // namespace top
} // namespace tpu_mlir
