#include "sophgo/Dialect/Top/IR/TopOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace sophgo::top;

struct TopFuseReshape : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: support to convert batchnorm to scale
    return failure();
  }
};

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<TopFuseReshape>(context);
}
