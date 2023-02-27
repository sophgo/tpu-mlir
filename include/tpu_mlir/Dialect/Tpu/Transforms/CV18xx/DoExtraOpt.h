//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "mlir/IR/PatternMatch.h"

#include "tpu_mlir/Support/Module.h"
#include <cstdint>

using namespace llvm;

namespace tpu_mlir {

namespace cv18xx {

class SplitReducePattern : public OpRewritePattern<tpu::ReduceOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override;
};

class FuseLeakReluPattern : public OpRewritePattern<tpu::LeakyReluOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::LeakyReluOp leakyReluOp,
                                PatternRewriter &rewriter) const override;
};

class SplitReluLimitPattern : public RewritePattern {
public:
  SplitReluLimitPattern(MLIRContext *context)
     : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

class MoveConvStrideToEltwiseOpPattern : public RewritePattern {
public:
  MoveConvStrideToEltwiseOpPattern(MLIRContext *context)
     : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

void populateDoExtraOptPatterns(RewritePatternSet *patterns);
} // namespace cv18xx
} // namespace tpu_mlir
