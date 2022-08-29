//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace tpu_mlir::tpu;

struct TpuConv : public OpRewritePattern<Conv2DOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Conv2DOp op,
                                PatternRewriter &rewriter) const override {
    if (op.multiplier() == ::llvm::None) {
      return failure();
    }
    auto mode = op.quant_mode().getValue();
    if (mode == 2) {
      return failure();
    }

    // op->setAttrs("with_quant", false);

    // auto new_op =

    // remove the relu Op
    // rewriter.replaceOp(op, {op.input()});
    return success();
  }
};

void Conv2DOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<TpuConv>(context);
}
