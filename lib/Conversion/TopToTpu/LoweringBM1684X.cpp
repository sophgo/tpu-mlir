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
      AvgPoolLowering,
      CastLowering,
      ConcatLowering,
      ConvLowering,
      DeconvLowering,
      Depth2SpaceLowering,
      DivLowering,
      ExpLowering,
      GatherLowering,
      LeakyReluLowering,
      LogLowering,
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
      ReluLowering,
      ReshapeLowering,
      ScaleLowering,
      SigmoidLowering,
      SiLULowering,
      SliceLowering,
      SoftmaxLowering,
      SqueezeLowering,
      SubLowering,
      TileLowering,
      UpsampleLowering
      // clang-format on
      >(patterns->getContext());
}
} // namespace bm1684x
} // namespace tpu_mlir
