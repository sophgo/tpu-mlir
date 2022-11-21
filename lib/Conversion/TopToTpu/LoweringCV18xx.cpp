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
      AddLowering,
      AvgPoolLowering,
      CastLowering,
      ConcatLowering,
      ConvLowering,
      ClipLowering,
      DeconvLowering,
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
      PReluLowering,
      ReluLowering,
      SigmoidLowering,
      SiLULowering,
      SliceLowering,
      SoftmaxLowering,
      UpsampleLowering
      // clang-format on
      >(patterns->getContext());
}
} // namespace cv18xx
} // namespace tpu_mlir
