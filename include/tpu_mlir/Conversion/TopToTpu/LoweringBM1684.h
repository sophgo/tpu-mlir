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

void populateTopShapeToTpuConversionPatterns(RewritePatternSet *patterns);

#define SHAPE_LOWERING_BM1684(OP)                                             \
  struct OP##TryLowering : public TopShapeLowering<top::OP##Op> {              \
    OP##TryLowering(MLIRContext *ctx) : TopShapeLowering<top::OP##Op>(ctx) {}  \
    void Lowering(PatternRewriter &rewriter,                                   \
                  top::OP##Op op) const override;                              \
  };

SHAPE_LOWERING_BM1684(ConstantFill)
SHAPE_LOWERING_BM1684(Shape)
SHAPE_LOWERING_BM1684(Slice)

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
LOWERING_BM1684(Arccos)
LOWERING_BM1684(Arctanh)
LOWERING_BM1684(AvgPool)
LOWERING_BM1684(Clip)
LOWERING_BM1684(Concat)
LOWERING_BM1684(Conv)
LOWERING_BM1684(Cos)
LOWERING_BM1684(Custom)
LOWERING_BM1684(DeformConv2D)
LOWERING_BM1684(Depth2Space)
LOWERING_BM1684(Elu)
LOWERING_BM1684(Floor)
LOWERING_BM1684(Gather)
LOWERING_BM1684(GatherND)
LOWERING_BM1684(GridSampler)
LOWERING_BM1684(GroupNorm)
LOWERING_BM1684(GRU)
LOWERING_BM1684(InstanceNorm)
LOWERING_BM1684(MatMul)
LOWERING_BM1684(MaxPool)
LOWERING_BM1684(Mul)
LOWERING_BM1684(Log)
LOWERING_BM1684(Nms)
LOWERING_BM1684(Pad)
LOWERING_BM1684(Permute)
LOWERING_BM1684(Relu)
LOWERING_BM1684(Remainder)
LOWERING_BM1684(Reshape)
LOWERING_BM1684(Reverse)
LOWERING_BM1684(RoiAlign)
LOWERING_BM1684(Scale)
LOWERING_BM1684(ScatterND)
LOWERING_BM1684(Slice)
LOWERING_BM1684(Sigmoid)
LOWERING_BM1684(Sign)
LOWERING_BM1684(SiLU)
LOWERING_BM1684(Sin)
LOWERING_BM1684(Softmax)
LOWERING_BM1684(Softplus)
LOWERING_BM1684(Sub)
LOWERING_BM1684(Sqrt)
LOWERING_BM1684(Tanh)
LOWERING_BM1684(Tile)
LOWERING_BM1684(TopK)
LOWERING_BM1684(Upsample)
LOWERING_BM1684(Interp)
LOWERING_BM1684(Reduce)
LOWERING_BM1684(Reciprocal)
LOWERING_BM1684(RMSNorm)
LOWERING_BM1684(HardSigmoid)
LOWERING_BM1684(HardSwish)
LOWERING_BM1684(AddConst)
LOWERING_BM1684(SubConst)
LOWERING_BM1684(MulConst)
LOWERING_BM1684(LayerNorm)
LOWERING_BM1684(SwapDimInner)
LOWERING_BM1684(ShuffleChannel)
LOWERING_BM1684(LRN)
LOWERING_BM1684(Min)
LOWERING_BM1684(MinConst)
LOWERING_BM1684(Max)
LOWERING_BM1684(MaxConst)
LOWERING_BM1684(Deconv)
LOWERING_BM1684(Exp)
LOWERING_BM1684(PRelu)
LOWERING_BM1684(LSTM)
LOWERING_BM1684(LeakyRelu)
LOWERING_BM1684(GELU)
LOWERING_BM1684(Pow)
LOWERING_BM1684(Pow2)
LOWERING_BM1684(Pow3)
LOWERING_BM1684(Div)
LOWERING_BM1684(Compare)
LOWERING_BM1684(CompareConst)
LOWERING_BM1684(Mish)
LOWERING_BM1684(Softsign)
LOWERING_BM1684(StridedSlice)
LOWERING_BM1684(MaskedFill)
LOWERING_BM1684(Where)
LOWERING_BM1684(GatherElements)
} // namespace bm1684
} // namespace tpu_mlir
