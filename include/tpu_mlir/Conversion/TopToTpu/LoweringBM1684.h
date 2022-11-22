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
LOWERING_BM1684(AvgPool)
LOWERING_BM1684(Concat)
LOWERING_BM1684(Conv)
LOWERING_BM1684(MatMul)
LOWERING_BM1684(MaxPool)
LOWERING_BM1684(Mul)
LOWERING_BM1684(Relu)
LOWERING_BM1684(Reshape)
LOWERING_BM1684(Slice)
LOWERING_BM1684(Sub)
LOWERING_BM1684(Tile)
LOWERING_BM1684(Upsample)

} // namespace bm1684
} // namespace tpu_mlir
