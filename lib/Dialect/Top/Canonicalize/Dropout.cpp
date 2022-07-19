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

struct TopDropout : public OpRewritePattern<DropoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DropoutOp op,
                                PatternRewriter &rewriter) const override {
    auto formerOp = op.input().getDefiningOp();

    formerOp->setAttr("name", op.nameAttr());
    // remove the dropout Op
    rewriter.replaceOp(op, {op.input()});
    return success();
  }
};

void DropoutOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<TopDropout>(context);
}
