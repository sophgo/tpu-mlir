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
struct BinaryConstShiftFuse
    : public OpRewriterPatternEx<tpu::BinaryConstShiftOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  BinaryConstShiftFuse(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::BinaryConstShiftOp>(context,
                                                     "BinaryConstShiftFuse") {}

  LogicalResult matchAndRewriteImpl(tpu::BinaryConstShiftOp op,
                                    PatternRewriter &rewriter) const override {
    bool is_type_match = module::getStorageType(op.getInput()) ==
                         module::getStorageType(op.getResult());
    bool is_identity =
        ((std::abs(op.getScale()) == 0 && op.getMode().str() == "Add") ||
         (std::abs(op.getScale()) == 0 && op.getMode().str() == "Sub" &&
          op.getIsReverse() == false) ||
         (op.getScale() == 1 && op.getMode().str() == "Mul")) &&
        op.getShift() == 0;

    if (is_type_match && is_identity) {
      rewriter.replaceOp(op, op.getInput());
      return success();
    }
    return failure();
  }
};

void tpu::BinaryConstShiftOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<BinaryConstShiftFuse>(context);
}

} // namespace tpu
} // namespace tpu_mlir
