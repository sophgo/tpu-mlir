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
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"

#include "tpu_mlir/Support/Module.h"
#include <cstdint>

using namespace llvm;

namespace tpu_mlir {

namespace cv18xx {

class ConvertMaskedFillOp : public OpRewritePattern<top::MaskedFillOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::MaskedFillOp op,
                                PatternRewriter &rewriter) const override;
};

class ConvertConv1dOp : public OpRewritePattern<top::ConvOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::ConvOp op,
                                PatternRewriter &rewriter) const override;
};

class ConvertConvPading : public OpRewritePattern<top::ConvOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::ConvOp op,
                                PatternRewriter &rewriter) const override;
};

class convertMaxPool3D : public OpRewritePattern<top::MaxPoolOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::MaxPoolOp op,
                                PatternRewriter &rewriter) const override;
};

class ConvertConvDilation : public OpRewritePattern<top::ConvOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::ConvOp op,
                                PatternRewriter &rewriter) const override;
};

class ConvertConv2dToMatMul : public OpRewritePattern<top::ConvOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::ConvOp op,
                                PatternRewriter &rewriter) const override;
};

class ConvertWhereOp : public OpRewritePattern<top::WhereOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::WhereOp op,
                                PatternRewriter &rewriter) const override;
};

class ConvertGatherOp : public OpRewritePattern<top::GatherOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::GatherOp op,
                                PatternRewriter &rewriter) const override;
};

class ConvertAddConstOp : public OpRewritePattern<top::AddConstOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::AddConstOp op,
                                PatternRewriter &rewriter) const override;
};

class ConvertDivOp : public OpRewritePattern<top::DivOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::DivOp op,
                                PatternRewriter &rewriter) const override;
};

class ConvertMaxPoolWithMaskOp : public OpRewritePattern<top::MaxPoolWithMaskOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::MaxPoolWithMaskOp op,
                                PatternRewriter &rewriter) const override;
};

class ConvertMaxUnpoolOp : public OpRewritePattern<top::MaxUnpoolOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::MaxUnpoolOp op,
                                PatternRewriter &rewriter) const override;
};

class ConvertScaleOp : public OpRewritePattern<top::ScaleOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::ScaleOp op,
                                PatternRewriter &rewriter) const override;
};

class ConvertUpsampleOp : public OpRewritePattern<top::UpsampleOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::UpsampleOp op,
                                PatternRewriter &rewriter) const override;
};

class ConvertMatMulWithRightTranspose : public OpRewritePattern<top::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::MatMulOp op,
                                PatternRewriter &rewriter) const override;
};

void populateDoExtraConversionPatterns(RewritePatternSet *patterns);
} // namespace cv18xx
} // namespace tpu_mlir
