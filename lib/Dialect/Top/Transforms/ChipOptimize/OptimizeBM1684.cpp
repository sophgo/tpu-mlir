//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace tpu_mlir {
namespace bm1684 {
class ConvertUnsqueezeOp : public OpRewritePattern<top::UnsqueezeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::UnsqueezeOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<top::ReshapeOp>(op, op.getOutput().getType(),
                                            op->getOperands(),
                                            std::vector<NamedAttribute>());
    return success();
  }
};

class ConvertSqueezeOp : public OpRewritePattern<top::SqueezeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::SqueezeOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<top::ReshapeOp>(op, op.getOutput().getType(),
                                            op->getOperands(),
                                            std::vector<NamedAttribute>());
    return success();
  }
};
} // namespace bm1684

namespace top {
using namespace bm1684;
void populateOptimizeBM1684Patterns(RewritePatternSet *patterns) {
  // add bm1684 optimize here
  patterns->add<
        ConvertSqueezeOp,
        ConvertUnsqueezeOp
        >(patterns->getContext());
}

} // namespace top
} // namespace tpu_mlir
