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

#define LOWERING_BM1684X(OP)                                                   \
  struct OP##Lowering : public TopLowering<top::OP##Op> {                      \
    OP##Lowering(MLIRContext *ctx) : TopLowering<top::OP##Op>(ctx) {}          \
    void LoweringINT8(PatternRewriter &rewriter, top::OP##Op op,               \
                      bool asymmetric) const override;                         \
    void LoweringBF16(PatternRewriter &rewriter,                               \
                      top::OP##Op op) const override;                          \
    void LoweringF16(PatternRewriter &rewriter,                                \
                     top::OP##Op op) const override;                           \
    void LoweringF32(PatternRewriter &rewriter,                                \
                     top::OP##Op op) const override;                           \
    void LoweringQuantized(PatternRewriter &rewriter,                          \
                           top::OP##Op op) const override;                     \
  };

LOWERING_BM1684X(Abs)
LOWERING_BM1684X(Add)
LOWERING_BM1684X(AddConst)
LOWERING_BM1684X(AvgPool)
LOWERING_BM1684X(Cast)
LOWERING_BM1684X(Concat)
LOWERING_BM1684X(Conv)
LOWERING_BM1684X(Deconv)
LOWERING_BM1684X(Depth2Space)
LOWERING_BM1684X(Div)
LOWERING_BM1684X(Exp)
LOWERING_BM1684X(Gather)
LOWERING_BM1684X(GRU)
LOWERING_BM1684X(LeakyRelu)
LOWERING_BM1684X(Log)
LOWERING_BM1684X(LRN)
LOWERING_BM1684X(LSTM)
LOWERING_BM1684X(MatMul)
LOWERING_BM1684X(Max)
LOWERING_BM1684X(MaxPool)
LOWERING_BM1684X(MaxPoolWithMask)
LOWERING_BM1684X(MaxUnpool)
LOWERING_BM1684X(Min)
LOWERING_BM1684X(Mul)
LOWERING_BM1684X(MulConst)
LOWERING_BM1684X(Pad)
LOWERING_BM1684X(Permute)
LOWERING_BM1684X(PRelu)
LOWERING_BM1684X(Relu)
LOWERING_BM1684X(Reshape)
LOWERING_BM1684X(Scale)
LOWERING_BM1684X(Sigmoid)
LOWERING_BM1684X(SiLU)
LOWERING_BM1684X(Slice)
LOWERING_BM1684X(Softmax)
LOWERING_BM1684X(Squeeze)
LOWERING_BM1684X(Tile)
LOWERING_BM1684X(Upsample)
LOWERING_BM1684X(Interp)
LOWERING_BM1684X(StridedSlice)
LOWERING_BM1684X(Reduce)
LOWERING_BM1684X(Pack)
LOWERING_BM1684X(Unpack)
LOWERING_BM1684X(Split)
LOWERING_BM1684X(Sub)
LOWERING_BM1684X(Sqrt)
LOWERING_BM1684X(Reciprocal)
LOWERING_BM1684X(Where)
LOWERING_BM1684X(MaskedFill)
LOWERING_BM1684X(Compare)
LOWERING_BM1684X(CompareConst)
LOWERING_BM1684X(Erf)
} // namespace bm1684x
} // namespace tpu_mlir
