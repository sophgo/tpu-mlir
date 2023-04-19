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
namespace bm1684 {

void populateTopToTpuConversionPatterns(RewritePatternSet *patterns);

#define LOWERING_BM1684(OP)                                                    \
  struct OP##Lowering : public TopLowering<top::OP##Op> {                      \
    OP##Lowering(MLIRContext *ctx) : TopLowering<top::OP##Op>(ctx) {}          \
    void LoweringINT8(PatternRewriter &rewriter, top::OP##Op op,               \
                      bool asymmetric) const override;                         \
    void LoweringF32(PatternRewriter &rewriter,                                \
                     top::OP##Op op) const override;                           \
  };

LOWERING_BM1684(Abs)
LOWERING_BM1684(Add)
LOWERING_BM1684(Arg)
LOWERING_BM1684(AvgPool)
LOWERING_BM1684(Concat)
LOWERING_BM1684(Conv)
LOWERING_BM1684(Floor)
LOWERING_BM1684(GroupNorm)
LOWERING_BM1684(MatMul)
LOWERING_BM1684(MaxPool)
LOWERING_BM1684(Mul)
LOWERING_BM1684(Log)
LOWERING_BM1684(Nms)
LOWERING_BM1684(Permute)
LOWERING_BM1684(Relu)
LOWERING_BM1684(Reshape)
LOWERING_BM1684(Slice)
LOWERING_BM1684(Sigmoid)
LOWERING_BM1684(SiLU)
LOWERING_BM1684(Softmax)
LOWERING_BM1684(Sub)
LOWERING_BM1684(Tile)
LOWERING_BM1684(TopK)
LOWERING_BM1684(Upsample)
LOWERING_BM1684(Interp)
LOWERING_BM1684(Reduce)
LOWERING_BM1684(HardSigmoid)
LOWERING_BM1684(HardSwish)
LOWERING_BM1684(AddConst)
LOWERING_BM1684(MulConst)
LOWERING_BM1684(LayerNorm)
LOWERING_BM1684(LRN)
LOWERING_BM1684(Min)
LOWERING_BM1684(Max)
LOWERING_BM1684(Deconv)
LOWERING_BM1684(Exp)
LOWERING_BM1684(PRelu)
LOWERING_BM1684(LSTM)
LOWERING_BM1684(LeakyRelu)
LOWERING_BM1684(GELU)
} // namespace bm1684
} // namespace tpu_mlir
