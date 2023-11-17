//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/Patterns.h"
#include "tpu_mlir/Support/MathUtils.h"

namespace tpu_mlir {
namespace top {
class MergeScale2Conv : public OpRewritePattern<top::ScaleOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::ScaleOp op,
                                PatternRewriter &rewriter) const override;
};

class ConvertScaleOp : public OpRewritePattern<top::ScaleOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::ScaleOp op,
                                PatternRewriter &rewriter) const override;
};
} // namespace top
} // namespace tpu_mlir
