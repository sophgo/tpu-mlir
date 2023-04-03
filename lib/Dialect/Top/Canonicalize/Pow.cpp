//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

struct PowToBinary : public OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter &rewriter) const override {
    auto exp = op.getExponent().convertToDouble();
    std::vector<NamedAttribute> attrs;
    if (exp == 2) {
      rewriter.replaceOpWithNewOp<MulOp>(op, op.getOutput().getType(),
                                         ValueRange{op.getInput(), op.getInput()},
                                         attrs);
      success();
    } else if (exp == 0.5) {
      rewriter.replaceOpWithNewOp<SqrtOp>(op, op.getOutput().getType(),
                                         ValueRange{op.getInput()},
                                         attrs);
      success();
    }
    return failure();
  }
};

void PowOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<PowToBinary>(context);
}
