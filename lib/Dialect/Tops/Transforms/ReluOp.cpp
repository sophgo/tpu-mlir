#include "sophgo/Dialect/Tops/IR/TopsOps.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::tops;

static bool supportFuseRelu(Operation *op) {
  if (matchPattern(op, m_Op<tops::ConvOp>()) ||
      matchPattern(op, m_Op<tops::MaxPoolOp>()) ||
      matchPattern(op, m_Op<tops::AvgPoolOp>()) ||
      matchPattern(op, m_Op<tops::MatMulOp>()) ||
      matchPattern(op, m_Op<tops::AddOp>())) {
    return true;
  }
  return false;
}

struct TopsFuseRelu : public OpRewritePattern<tops::ReluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tops::ReluOp op,
                                PatternRewriter &rewriter) const override {
    auto formerOp = op.input().getDefiningOp();
    if (!formerOp->getResult(0).hasOneUse())
      return failure();

    if (false == supportFuseRelu(formerOp)) {
      return failure();
    }

    formerOp->setAttr("do_relu", rewriter.getBoolAttr(true));
    formerOp->setAttr("name", op.nameAttr());
    // remove the relu Op
    rewriter.replaceOp(op, {op.input()});
    return success();
  }
};


void ReluOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<TopsFuseRelu>(context);
}
