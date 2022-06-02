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

using namespace mlir;
using namespace tpu_mlir::top;

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
