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
namespace tpu {
class LargePadConvPattern : public OpRewriterPatternEx<tpu::Conv2DOp> {
public:
  LargePadConvPattern(mlir::MLIRContext *context, int benifit = 1)
      : OpRewriterPatternEx<tpu::Conv2DOp>(context, "LargePadConvPattern",
                                           benifit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(tpu::Conv2DOp op,
                      mlir::PatternRewriter &rewriter) const override;
};

class PermuteReorderPattern : public OpRewriterPatternEx<tpu::PermuteOp> {
public:
  PermuteReorderPattern(mlir::MLIRContext *context, int benifit = 1)
      : OpRewriterPatternEx<tpu::PermuteOp>(context, "PermuteReorderPattern",
                                            benifit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(tpu::PermuteOp op,
                      mlir::PatternRewriter &rewriter) const override;
};

struct PermutePadSwap : public OpRewriterPatternEx<tpu::PermuteOp> {
  PermutePadSwap(mlir::MLIRContext *context, int benifit = 1)
      : OpRewriterPatternEx<tpu::PermuteOp>(context, "PermutePadSwap",
                                            benifit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(tpu::PermuteOp op,
                      mlir::PatternRewriter &rewriter) const override;
};

Value createSplitQuantizedMLP(mlir::PatternRewriter &rewriter,
                              mlir::Operation *op, Value arg0);
Value createSplitQuantizedMLP2(mlir::PatternRewriter &rewriter,
                               mlir::Operation *op, Value arg0, int num_device);

struct RemoveReshape : public OpRewritePattern<tpu::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::ReshapeOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace tpu
} // namespace tpu_mlir
