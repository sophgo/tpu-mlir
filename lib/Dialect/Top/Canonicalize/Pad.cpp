//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace tpu_mlir::top;

struct TopFusePad : public OpRewritePattern<PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PadOp op,
                                PatternRewriter &rewriter) const override {

    if (!op->hasOneUse())
      return failure();

    if (op->getOperands()[0]
            .getType()
            .dyn_cast<TensorType>()
            .getShape()
            .size() != 4)
      return failure();

    auto nextOp = *op->getUsers().begin();
    auto paddings = op->getAttr("paddings").dyn_cast<ArrayAttr>();

    if (isa<ConvOp, MaxPoolOp, AvgPoolOp>(nextOp)) {

      for (auto &pad : llvm::make_range(
               paddings.begin(),
               paddings.begin() +
                   4 /*the leading padding is batch*2 and channel*2*/)) {
        if (pad.cast<IntegerAttr>().getInt() != 0)
          return failure();
      }

      for (auto &pad : nextOp->getAttr("pads").dyn_cast<ArrayAttr>()) {
        if (pad.cast<IntegerAttr>().getInt() != 0)
          return failure();
      }
    } else
      return failure();

    nextOp->setAttr(
        "pads", rewriter.getArrayAttr({paddings.begin() + 4, paddings.end()}));
    // remove the pad Op
    rewriter.replaceOp(op, {op.input()});
    return success();
  }
};

void PadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<TopFusePad>(context);
}
