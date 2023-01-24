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
      AddConstLowering,
      AvgPoolLowering,
      CastLowering,
      ConcatLowering,
      ConvLowering,
      DeconvLowering,
      Depth2SpaceLowering,
      DivLowering,
      ExpLowering,
      GatherLowering,
      GRULowering,
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
      SigmoidLowering,
      SiLULowering,
      SliceLowering,
      SoftmaxLowering,
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
      WhereLowering,
      MaskedFillLowering,
      CompareLowering,
      CompareConstLowering,
      ErfLowering,
      HardSigmoidLowering,
      HardSwishLowering,
      LayerNormLowering,
      TanhLowering,
      TopKLowering
      // clang-format on
      >(patterns->getContext());
}
} // namespace bm1684x
} // namespace tpu_mlir
