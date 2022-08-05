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
using namespace tpu_mlir::trait;

struct TopFuseRelu : public OpRewritePattern<ReluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp op,
                                PatternRewriter &rewriter) const override {
    auto formerOp = op.input().getDefiningOp();
    if (!formerOp->getResult(0).hasOneUse())
      return failure();

    if (false == formerOp->hasTrait<SupportFuseRelu>()) {
      return failure();
    }

    bool has_upper_limit = op->hasAttr("upper_limit") &&
                           op.upper_limitAttr().getValueAsDouble() > 0.f;
    if (has_upper_limit) {
      formerOp->setAttr("upper_limit", op.upper_limitAttr());
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
  results.insert<TopFuseRelu>(context);
}
