//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Conversion/TopToTpu/TopLowering.h"

namespace tpu_mlir {
namespace bm1684x {

void populateTopToTpuConversionPatterns(RewritePatternSet *patterns);

#define LOWERING(OP)                                                           \
  struct OP##Lowering : public TopLowering<top::OP##Op> {                      \
    OP##Lowering(MLIRContext *ctx) : TopLowering<top::OP##Op>(ctx) {}          \
    void LoweringINT8(PatternRewriter &rewriter, top::OP##Op op,     \
                      bool asymmetric) const override;                         \
    void LoweringBF16(PatternRewriter &rewriter,                     \
                      top::OP##Op op) const override;                          \
    void LoweringF16(PatternRewriter &rewriter,                      \
                     top::OP##Op op) const override;                           \
    void LoweringF32(PatternRewriter &rewriter,                      \
                     top::OP##Op op) const override;                           \
    void LoweringQuantized(PatternRewriter &rewriter,                \
                           top::OP##Op op) const override;                     \
  };

LOWERING(Abs)
LOWERING(Add)
LOWERING(AvgPool)
LOWERING(Cast)
LOWERING(Concat)
LOWERING(Conv)
LOWERING(Deconv)
LOWERING(Depth2Space)
LOWERING(Div)
LOWERING(Gather)
LOWERING(LeakyRelu)
LOWERING(Log)
LOWERING(LSTM)
LOWERING(MatMul)
LOWERING(Max)
LOWERING(MaxPool)
LOWERING(MaxPoolWithMask)
LOWERING(MaxUnpool)
LOWERING(Min)
LOWERING(Mul)
LOWERING(MulConst)
LOWERING(Pad)
LOWERING(Permute)
LOWERING(Relu)
LOWERING(Reshape)
LOWERING(Scale)
LOWERING(Sigmoid)
LOWERING(SiLU)
LOWERING(Slice)
LOWERING(Softmax)
LOWERING(Squeeze)
LOWERING(Tile)
LOWERING(Upsample)

} // namespace bm1684x
} // namespace tpu_mlir
