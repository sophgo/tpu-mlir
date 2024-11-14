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
#include "tpu_mlir/Support/Patterns.h"

namespace tpu_mlir {
namespace top {
class MergeScale2Conv : public OpRewriterPatternEx<top::ScaleOp> {
public:
  MergeScale2Conv(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::ScaleOp>(context, "MergeScale2Conv", benefit) {
  }

protected:
  mlir::LogicalResult
  matchAndRewriteImpl(top::ScaleOp op,
                      mlir::PatternRewriter &rewriter) const override;
};

class ConvertScaleOp : public OpRewriterPatternEx<top::ScaleOp> {
public:
  ConvertScaleOp(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::ScaleOp>(context, "ConvertScaleOp", benefit) {}

protected:
  mlir::LogicalResult
  matchAndRewriteImpl(top::ScaleOp op,
                      mlir::PatternRewriter &rewriter) const override;
};

class ConcatToSwapDimInner : public OpRewriterPatternEx<top::ConcatOp> {
public:
  ConcatToSwapDimInner(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::ConcatOp>(context, "ConcatToSwapDimInner",
                                           benefit) {}

protected:
  mlir::LogicalResult
  matchAndRewriteImpl(top::ConcatOp op,
                      mlir::PatternRewriter &rewriter) const override;
};

} // namespace top
} // namespace tpu_mlir
