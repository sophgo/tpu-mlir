#include "sophgo/Dialect/Tpu/IR/TpuOps.h"

using namespace mlir;
using namespace sophgo::tpu;

struct SimplifyRedundantCast : public OpRewritePattern<CastOp> {
  SimplifyRedundantCast(mlir::MLIRContext *context)
      : OpRewritePattern<CastOp>(context, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(CastOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::Value castInput = op.getOperand();
    auto castInputOp = castInput.getDefiningOp<CastOp>();

    if (!castInputOp)
      return failure();

    if (op.getResult().getType() != castInputOp.input().getType())
      return failure();

    // We have a redundant cast. Use the rewriter.
    rewriter.replaceOp(op, {castInputOp.getOperand()});
    return success();
  }
};

void CastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<SimplifyRedundantCast>(context);
}
