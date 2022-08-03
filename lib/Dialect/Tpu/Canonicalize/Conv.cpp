//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace tpu_mlir::tpu;

struct TpuConv : public OpRewritePattern<ConvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvOp op,
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

void ConvOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<TpuConv>(context);
}
