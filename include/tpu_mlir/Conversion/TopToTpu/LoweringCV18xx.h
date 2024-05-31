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

LOWERING_CV18XX(Abs)
LOWERING_CV18XX(Add)
LOWERING_CV18XX(Arg)
LOWERING_CV18XX(AvgPool)
LOWERING_CV18XX(Cast)
LOWERING_CV18XX(CompareConst)
LOWERING_CV18XX(Concat)
LOWERING_CV18XX(Conv)
LOWERING_CV18XX(Clip)
LOWERING_CV18XX(Copy)
LOWERING_CV18XX(Csc)
LOWERING_CV18XX(CumSum)
LOWERING_CV18XX(Custom)
LOWERING_CV18XX(Deconv)
LOWERING_CV18XX(Depth2Space)
LOWERING_CV18XX(DetectionOutput)
LOWERING_CV18XX(Elu)
LOWERING_CV18XX(Exp)
LOWERING_CV18XX(FrcnDetection)
LOWERING_CV18XX(Gather)
LOWERING_CV18XX(GatherElements)
LOWERING_CV18XX(GatherND)
LOWERING_CV18XX(GELU)
LOWERING_CV18XX(GRU)
LOWERING_CV18XX(GridSampler)
LOWERING_CV18XX(HardSigmoid)
LOWERING_CV18XX(HardSwish)
LOWERING_CV18XX(InstanceNorm)
LOWERING_CV18XX(Interp)
LOWERING_CV18XX(LayerNorm)
LOWERING_CV18XX(LeakyRelu)
LOWERING_CV18XX(Log)
LOWERING_CV18XX(LRN)
LOWERING_CV18XX(LSTM)
LOWERING_CV18XX(MatchTemplate)
LOWERING_CV18XX(MatMul)
LOWERING_CV18XX(Max)
LOWERING_CV18XX(MaxPool)
LOWERING_CV18XX(Min)
LOWERING_CV18XX(Mish)
LOWERING_CV18XX(Mul)
LOWERING_CV18XX(MulConst)
LOWERING_CV18XX(Reshape)
LOWERING_CV18XX(RetinaFaceDetection)
LOWERING_CV18XX(Pad)
LOWERING_CV18XX(Pow)
LOWERING_CV18XX(Preprocess)
LOWERING_CV18XX(Proposal)
LOWERING_CV18XX(Reduce)
LOWERING_CV18XX(Reverse)
LOWERING_CV18XX(Permute)
LOWERING_CV18XX(PoolMask)
LOWERING_CV18XX(PRelu)
LOWERING_CV18XX(Reciprocal)
LOWERING_CV18XX(Relu)
LOWERING_CV18XX(RMSNorm)
LOWERING_CV18XX(ROIPooling)
LOWERING_CV18XX(ShuffleChannel)
LOWERING_CV18XX(Scale)
LOWERING_CV18XX(ScaleLut)
LOWERING_CV18XX(ScatterND)
LOWERING_CV18XX(Sigmoid)
LOWERING_CV18XX(SiLU)
LOWERING_CV18XX(Slice)
LOWERING_CV18XX(Softmax)
LOWERING_CV18XX(Softplus)
LOWERING_CV18XX(Softsign)
LOWERING_CV18XX(Sub)
LOWERING_CV18XX(SubConst)
LOWERING_CV18XX(SwapChannel)
LOWERING_CV18XX(Tanh)
LOWERING_CV18XX(TopK)
LOWERING_CV18XX(Tile)
LOWERING_CV18XX(Upsample)
LOWERING_CV18XX(YoloDetection)
LOWERING_CV18XX(Shape)
LOWERING_CV18XX(Floor)
} // namespace cv18xx
} // namespace tpu_mlir
