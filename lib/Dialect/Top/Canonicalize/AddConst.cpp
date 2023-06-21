//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

struct ToAbsAdd : public OpRewritePattern<AddConstOp> {
  using OpRewritePattern::OpRewritePattern;
  ToAbsAdd(MLIRContext *context, PatternBenefit benefit = 9)
      : OpRewritePattern<AddConstOp>(context, benefit) {}

  LogicalResult matchAndRewrite(AddConstOp op,
                                PatternRewriter &rewriter) const override {
    // Todo:
    // Impelemnt a pattern to simplify AbsOp->AddConstOp to AbsAddOp
    return success();
  }
};

void AddConstOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<ToAbsAdd>(context);
}
