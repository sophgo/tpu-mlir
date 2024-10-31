//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

void populateTopShapeToTpuConversionPatterns(RewritePatternSet *patterns) {
  patterns->add<
      // clang-format off
      ConstantFillTryLowering,
      ShapeTryLowering,
      SliceTryLowering
      // clang-format on
      >(patterns->getContext());
}

void populateTopToTpuConversionPatterns(RewritePatternSet *patterns) {
  patterns->add<
      // clang-format off
      AbsLowering,
      AddLowering,
      ArgLowering,
      ArccosLowering,
      ArctanhLowering,
      AvgPoolLowering,
      ClipLowering,
      ConcatLowering,
      ConvLowering,
      CosLowering,
      CustomLowering,
      DeformConv2DLowering,
      Depth2SpaceLowering,
      EluLowering,
      FloorLowering,
      GatherLowering,
      GatherNDLowering,
      GridSamplerLowering,
      GroupNormLowering,
      GRULowering,
      InstanceNormLowering,
      MatMulLowering,
      MaxPoolLowering,
      MulLowering,
      LogLowering,
      NmsLowering,
      PadLowering,
      PermuteLowering,
      ReciprocalLowering,
      ReduceLowering,
      ReluLowering,
      RemainderLowering,
      ReshapeLowering,
      ReverseLowering,
      RoiAlignLowering,
      ScaleLowering,
      ScatterNDLowering,
      ShuffleChannelLowering,
      SliceLowering,
      SoftmaxLowering,
      SoftplusLowering,
      SigmoidLowering,
      SignLowering,
      SiLULowering,
      SinLowering,
      SqrtLowering,
      StridedSliceLowering,
      TanhLowering,
      TileLowering,
      TopKLowering,
      UpsampleLowering,
      InterpLowering,
      ReduceLowering,
      HardSigmoidLowering,
      HardSwishLowering,
      AddConstLowering,
      SubConstLowering,
      MulConstLowering,
      SubLowering,
      LayerNormLowering,
      SwapDimInnerLowering,
      LRNLowering,
      MinLowering,
      MinConstLowering,
      MaxLowering,
      MaxConstLowering,
      DeconvLowering,
      ExpLowering,
      PReluLowering,
      LSTMLowering,
      LeakyReluLowering,
      GELULowering,
      PowLowering,
      Pow2Lowering,
      Pow3Lowering,
      DivLowering,
      CompareLowering,
      CompareConstLowering,
      MishLowering,
      SoftsignLowering,
      MaskedFillLowering,
      WhereLowering,
      GatherElementsLowering,
      RMSNormLowering,
      ModLowering
      // clang-format on
      >(patterns->getContext());
}
} // namespace bm1684
} // namespace tpu_mlir
