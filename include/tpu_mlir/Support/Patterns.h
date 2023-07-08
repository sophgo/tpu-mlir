//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/PatternMatch.h"

// Common Patterns
namespace tpu_mlir {
namespace patterns {

// if op0 == op1, remove op1
struct FuseSameOp : public RewritePattern {
  FuseSameOp(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

// if op =  op + op, fuse to one op. such as top::Reshape
template <typename OpTy>
struct FuseRepeatPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override;
};

// convert op a to op b, not care attributes. such as top::Sequence to
// top::Reshape
template <typename From, typename To>
struct ConvertPattern : public OpRewritePattern<From> {
  using OpRewritePattern<From>::OpRewritePattern;
  LogicalResult matchAndRewrite(From op,
                                PatternRewriter &rewriter) const override;
};

} // namespace patterns
} // namespace tpu_mlir
