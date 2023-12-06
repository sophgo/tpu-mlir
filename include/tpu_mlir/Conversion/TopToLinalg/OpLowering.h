//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Conversion/TopToLinalg/TopLowering.h"

namespace tpu_mlir {

void populateTopToLinalgConversionPatterns(RewritePatternSet *patterns);

#define OpLowering(OP)                                                         \
  struct OP##LoweringToLinalg : public TopLoweringToLinalg<top::OP##Op> {                \
    OP##LoweringToLinalg(MLIRContext *ctx) : TopLoweringToLinalg<top::OP##Op>(ctx) {}    \
    void Lowering(PatternRewriter &rewriter, top::OP##Op op) const override;   \
  };

OpLowering(MaxPoolWithMask)
OpLowering(AvgPool)
OpLowering(BatchNormTrain)
OpLowering(LayerNormTrain)
OpLowering(Conv)
OpLowering(Reduce)
OpLowering(Transpose)
OpLowering(Input)
OpLowering(Add)
OpLowering(Reshape)
OpLowering(Softmax)
OpLowering(Permute)
OpLowering(Split)
OpLowering(Slice)
OpLowering(MatMul)
OpLowering(Variance)
OpLowering(Unsqueeze)
OpLowering(Squeeze)
// OpLowering(Broadcast)
OpLowering(AddConst)
OpLowering(Div)
OpLowering(Rsqrt)
OpLowering(Sub)
OpLowering(Mul)
OpLowering(MulConst)
OpLowering(Exp)
OpLowering(Arg)
// clang-format on
} // namespace tpu_mlir
