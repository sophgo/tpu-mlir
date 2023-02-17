//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {

void populateTopToTpuConversionPatterns(RewritePatternSet *patterns) {
  patterns->add<
      // clang-format off
      AbsLowering,
      AddLowering,
      ArgLowering,
      AddConstLowering,
      AvgPoolLowering,
      CastLowering,
      ConcatLowering,
      ConvLowering,
      DeconvLowering,
      Depth2SpaceLowering,
      DivLowering,
      EluLowering,
      ExpLowering,
      FloorLowering,
      GatherLowering,
      GRULowering,
      GELULowering,
      LeakyReluLowering,
      LogLowering,
      LRNLowering,
      LSTMLowering,
      MatMulLowering,
      MaxLowering,
      MaxPoolLowering,
      MaxPoolWithMaskLowering,
      MaxUnpoolLowering,
      MinLowering,
      MulLowering,
      MulConstLowering,
      PadLowering,
      PermuteLowering,
      PReluLowering,
      PowLowering,
      ReciprocalLowering,
      ReluLowering,
      ReshapeLowering,
      ScaleLowering,
      ScatterNDLowering,
      SigmoidLowering,
      SiLULowering,
      SliceLowering,
      SoftmaxLowering,
      SoftplusLowering,
      SqueezeLowering,
      TileLowering,
      UpsampleLowering,
      InterpLowering,
      StridedSliceLowering,
      ReduceLowering,
      PackLowering,
      UnpackLowering,
      SplitLowering,
      SubLowering,
      SqrtLowering,
      SwapDimInnerLowering,
      WhereLowering,
      MaskedFillLowering,
      CompareLowering,
      CompareConstLowering,
      ErfLowering,
      HardSigmoidLowering,
      HardSwishLowering,
      LayerNormLowering,
      TanhLowering,
      TopKLowering,
      ReverseLowering,
      PixelNormLowering
      // clang-format on
      >(patterns->getContext());
}
} // namespace bm1684x
} // namespace tpu_mlir
