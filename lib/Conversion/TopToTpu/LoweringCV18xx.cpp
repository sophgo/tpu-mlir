//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

namespace tpu_mlir {
namespace cv18xx {

void populateTopToTpuConversionPatterns(RewritePatternSet *patterns) {
  patterns->add<
      // clang-format off
      AbsLowering,
      AddLowering,
      AvgPoolLowering,
      CastLowering,
      ConcatLowering,
      ConvLowering,
      ClipLowering,
      DeconvLowering,
      Depth2SpaceLowering,
      GatherLowering,
      InterpLowering,
      LeakyReluLowering,
      LogLowering,
      MatMulLowering,
      MaxLowering,
      MaxPoolLowering,
      MinLowering,
      MulLowering,
      MulConstLowering,
      ReshapeLowering,
      PadLowering,
      PermuteLowering,
      ReduceLowering,
      PReluLowering,
      ReluLowering,
      ReciprocalLowering,
      SigmoidLowering,
      SiLULowering,
      SliceLowering,
      SoftmaxLowering,
      TileLowering,
      UpsampleLowering
      // clang-format on
      >(patterns->getContext());
}
} // namespace cv18xx
} // namespace tpu_mlir
