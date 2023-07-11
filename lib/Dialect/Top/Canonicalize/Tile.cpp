//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"


using namespace tpu_mlir::top;

struct TopFuseTile : public OpRewritePattern<TileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileOp op,
                                PatternRewriter &rewriter) const override {

    auto next_op = *op->getUsers().begin();
    if (isa<AddOp, SubOp, MulOp, MinOp, MaxOp>(next_op)) {
      auto shape0 = module::getShape(op.getInput());
      auto shape1 = module::getShape(op.getOutput());
      for (int i = 0; i < shape0.size(); ++i) {
        if (shape0[i] != shape1[i] && std::min(shape0[i], shape1[i]) != 1)
          return failure();
      }
      // remove the Tile Op
      rewriter.replaceOp(op, {op.getInput()});
      return success();
    }
    return failure();
  }
};

void TileOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<TopFuseTile>(context);
}
