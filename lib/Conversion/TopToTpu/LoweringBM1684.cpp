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

void populateTopToTpuConversionPatterns(RewritePatternSet *patterns) {
  patterns->add<
      // clang-format off
      AbsLowering,
      AddLowering,
      ArgLowering,
      AvgPoolLowering,
      ConcatLowering,
      ConvLowering,
      MatMulLowering,
      MaxPoolLowering,
      MulLowering,
      PermuteLowering,
      ReduceLowering,
      ReluLowering,
      ReshapeLowering,
      SliceLowering,
      SoftmaxLowering,
      SigmoidLowering,
      SiLULowering,
      TileLowering,
      UpsampleLowering,
      InterpLowering,
      ReduceLowering,
      HardSigmoidLowering,
      HardSwishLowering,
      AddConstLowering,
      MulConstLowering,
      SubLowering
      // clang-format on
      >(patterns->getContext());
}
} // namespace bm1684
} // namespace tpu_mlir
