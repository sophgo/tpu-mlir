//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "mlir/IR/PatternMatch.h"

#include "tpu_mlir/Support/Module.h"
#include <cstdint>

using namespace llvm;

namespace tpu_mlir {

namespace bm1684x {
class ConvertMatMulWithRightTranspose : public OpRewritePattern<top::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::MatMulOp op,
                                PatternRewriter &rewriter) const override;
};

class ReshapeReorderPattern : public OpRewritePattern<top::ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::ReshapeOp op,
                                PatternRewriter &rewriter) const override;
};

void populateDoExtraConversionPatterns(RewritePatternSet *patterns);
}
}
