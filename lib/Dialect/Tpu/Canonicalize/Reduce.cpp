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

/**
 * Reduce (f32) -> Cast(f32-f16) -> Cast(f16-f32) -> Active(f32)
 * Reduce (f32) -> Cast(f32-int8) -> Cast(int8-f32) -> Active(f32)
 */
struct ReduceFuse : public OpRewriterPatternEx<tpu::ReduceOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  ReduceFuse(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::ReduceOp>(context, "ReduceFuse") {}

  LogicalResult matchAndRewriteImpl(tpu::ReduceOp op,
                                    PatternRewriter &rewriter) const override {
    auto opt = op.getOutput();
    if (!opt.hasOneUse()) {
      return failure();
    }

    auto next_op_ = *opt.user_begin();
    auto castf16 = dyn_cast<tpu::CastOp>(next_op_);

    if (!castf16 ||
        (!module::getElementType(castf16.getInput()).isF32() &&
         !module::isCalibratedType(castf16.getInput()) /**for int8 cast op*/)) {
      return failure();
    }

    if (!castf16->hasOneUse()) {
      return failure();
    }
    next_op_ = *castf16.getOutput().user_begin();
    auto castf32 = dyn_cast<tpu::CastOp>(next_op_);
    if (!castf32 || (!module::getElementType(castf32.getOutput()).isF32() &&
                     !module::isUniformQuantized(
                         castf32.getOutput()) /**for int8 cast op*/)) {
      return failure();
    }

    if (!castf32->hasOneUse()) {
      return failure();
    }
    next_op_ = *castf32.getOutput().user_begin();
    auto active = dyn_cast<tpu::ActiveOp>(next_op_);
    if (!active) {
      return failure();
    }

    active.setOperand(opt);
    rewriter.replaceAllUsesWith(castf16.getOutput(), active.getInput());
    // erase reversed
    rewriter.eraseOp(castf32);
    rewriter.eraseOp(castf16);

    return success();
  }
};

void tpu::ReduceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.insert<ReduceFuse>(context);
}

} // namespace tpu
} // namespace tpu_mlir
