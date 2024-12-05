//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

namespace tpu_mlir {
namespace tpu {

// const to large, to 10k
struct MaskedFillTooLarge : public OpRewriterPatternEx<tpu::MaskedFillOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  MaskedFillTooLarge(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::MaskedFillOp>(context, "MaskedFillTooLarge") {}

  LogicalResult matchAndRewriteImpl(tpu::MaskedFillOp op,
                                    PatternRewriter &rewriter) const override {
    double const_val = op.getConstVal().convertToDouble();
    if (const_val >= 1e10) {
      const_val = 10000;
    } else if (const_val <= -1e10) {
      const_val = -10000;
    } else {
      return failure();
    }
    op.setConstVal(APFloat(const_val));
    return success();
  }
};

void tpu::MaskedFillOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.insert<MaskedFillTooLarge>(context);
}

} // namespace tpu
} // namespace tpu_mlir
