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
namespace cv18xx {

void populateTopToTpuConversionPatterns(RewritePatternSet *patterns);

#define LOWERING_CV18XX(OP)                                                    \
  struct OP##Lowering : public TopLowering<top::OP##Op> {                      \
    OP##Lowering(MLIRContext *ctx) : TopLowering<top::OP##Op>(ctx) {}          \
    void LoweringINT8(PatternRewriter &rewriter, top::OP##Op op,               \
                      bool asymmetric) const override;                         \
    void LoweringBF16(PatternRewriter &rewriter,                               \
                      top::OP##Op op) const override;                          \
  };

LOWERING_CV18XX(Add)
LOWERING_CV18XX(AvgPool)
LOWERING_CV18XX(Cast)
LOWERING_CV18XX(Concat)
LOWERING_CV18XX(Conv)
LOWERING_CV18XX(MatMul)
LOWERING_CV18XX(MaxPool)
LOWERING_CV18XX(Reshape)
LOWERING_CV18XX(Permute)
LOWERING_CV18XX(SiLU)
LOWERING_CV18XX(Softmax)
LOWERING_CV18XX(Upsample)
} // namespace cv18xx
} // namespace tpu_mlir
