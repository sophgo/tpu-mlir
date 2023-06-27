//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Conversion/TopToTosa/TopLowering.h"

namespace tpu_mlir {

void populateTopToTosaConversionPatterns(RewritePatternSet *patterns);

#define OpLowering(OP)                                                         \
  struct OP##Lowering : public TopLoweringToTosa<top::OP##Op> {                \
    OP##Lowering(MLIRContext *ctx) : TopLoweringToTosa<top::OP##Op>(ctx) {}    \
    void Lowering(PatternRewriter &rewriter, top::OP##Op op) const override;   \
  };
// clang-format off
OpLowering(Input)
OpLowering(Add)
OpLowering(Conv)
OpLowering(AvgPool)
OpLowering(MaxPool)
OpLowering(Softmax)
OpLowering(Reshape)
OpLowering(MatMul)
// clang-format on
} // namespace tpu_mlir
