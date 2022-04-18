#include "sophgo/Dialect/Top/IR/TopOps.h"

using namespace mlir;
using namespace sophgo::top;

struct TopFuseBatchNorm : public OpRewritePattern<BatchNormOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BatchNormOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: support to convert batchnorm to scale
    return failure();
  }
};

void BatchNormOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<TopFuseBatchNorm>(context);
}
