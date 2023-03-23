//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir::top;

struct SplitToSlice : public OpRewritePattern<SplitOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SplitOp op,
                                PatternRewriter &rewriter) const override {
    auto in_shape = module::getShape(op.getInput());
    int64_t dims = in_shape.size();
    auto num = op.getNum();
    auto axis = op.getAxis();
    std::vector<int64_t> offset(dims, 0);
    std::vector<int64_t> steps(dims, 1);
    rewriter.setInsertionPointAfter(op);
    for (int i = 0; i < num; i++) {
      auto out = op.getResult(i);
      auto out_shape = module::getShape(out);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("offset", rewriter.getI64ArrayAttr(offset)));
      attrs.push_back(
          rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr(steps)));
      auto s_op = rewriter.create<SliceOp>(module::getLoc(out), out.getType(),
                                           op.getInput(), attrs);
      out.replaceAllUsesWith(s_op.getOutput());
      offset[axis] += out_shape[axis];
    }
    return success();
  }
};

void SplitOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<SplitToSlice>(context);
}
