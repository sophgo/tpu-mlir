#include "sophgo/Dialect/Tops/IR/TopsOps.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::tops;

struct TopsFuseReshape : public OpRewritePattern<tops::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tops::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: support to convert batchnorm to scale
    return failure();
  }
};

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<TopsFuseReshape>(context);
}
