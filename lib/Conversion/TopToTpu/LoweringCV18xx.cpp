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
      ArgLowering,
      AvgPoolLowering,
      CastLowering,
      CompareConstLowering,
      ConcatLowering,
      ConvLowering,
      ClipLowering,
      CopyLowering,
      CumSumLowering,
      CustomLowering,
      CscLowering,
      DeconvLowering,
      Mmap2RgbmapLowering,
      Depth2SpaceLowering,
      DetectionOutputLowering,
      EluLowering,
      ExpLowering,
      FrcnDetectionLowering,
      GatherLowering,
      GatherElementsLowering,
      GatherNDLowering,
      GELULowering,
      GRULowering,
      GridSamplerLowering,
      HardSigmoidLowering,
      HardSwishLowering,
      InstanceNormLowering,
      InterpLowering,
      LayerNormLowering,
      LeakyReluLowering,
      LogLowering,
      LRNLowering,
      LSTMLowering,
      MatchTemplateLowering,
      MatMulLowering,
      MaxLowering,
      MaxConstLowering,
      MaxPoolLowering,
      MinLowering,
      MinConstLowering,
      MishLowering,
      MulLowering,
      MulConstLowering,
      ReshapeLowering,
      PadLowering,
      PowLowering,
      PreprocessLowering,
      ProposalLowering,
      PermuteLowering,
      ROIPoolingLowering,
      RetinaFaceDetectionLowering,
      ReduceLowering,
      ReverseLowering,
      PReluLowering,
      PoolMaskLowering,
      ReluLowering,
      ReciprocalLowering,
      ScaleLowering,
      ScaleLutLowering,
      ScatterNDLowering,
      ShuffleChannelLowering,
      SigmoidLowering,
      SiLULowering,
      SliceLowering,
      SoftmaxLowering,
      SoftplusLowering,
      SoftsignLowering,
      SubLowering,
      SubConstLowering,
      SwapChannelLowering,
      TanhLowering,
      TopKLowering,
      TileLowering,
      UpsampleLowering,
      YoloDetectionLowering,
      RMSNormLowering,
      ShapeLowering,
      FloorLowering
      // clang-format on
      >(patterns->getContext());
}
} // namespace cv18xx
} // namespace tpu_mlir
