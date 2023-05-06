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
void populateTopCfOpToTpuConversionPatterns(RewritePatternSet &patterns,
                                            TypeConverter &typeConverter,
                                            MLIRContext *ctx) {
  patterns.insert<IfOpLowering>(typeConverter, ctx);
}

void populateTopShapeToTpuConversionPatterns(RewritePatternSet *patterns) {
  patterns->add<
      // clang-format off
      ShapeTryLowering,
      ConstantFillTryLowering,
      ConcatTryLowering,
      UnsqueezeTryLowering,
      SqueezeTryLowering
      // clang-format on
      >(patterns->getContext());
}


void populateTopToTpuConversionPatterns(RewritePatternSet *patterns) {
  patterns->add<
      // clang-format off
      AbsLowering,
      AddLowering,
      ArgLowering,
      AddConstLowering,
      AvgPoolLowering,
      CastLowering,
      ClipLowering,
      ConcatLowering,
      ConvLowering,
      CosLowering,
      CoshLowering,
      DeconvLowering,
      DeformConv2DLowering,
      Depth2SpaceLowering,
      DivLowering,
      EluLowering,
      ExpLowering,
      FloorLowering,
      GatherLowering,
      GatherElementsLowering,
      GridSamplerLowering,
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
      MishLowering,
      MulLowering,
      MulConstLowering,
      NonZeroLowering,
      PadLowering,
      PermuteLowering,
      PReluLowering,
      PreprocessLowering,
      PowLowering,
      ReciprocalLowering,
      ReluLowering,
      ReshapeLowering,
      RoiAlignLowering,
      ScaleLowering,
      ScaleLutLowering,
      ScatterElementsLowering,
      ScatterNDLowering,
      SinLowering,
      SinhLowering,
      SigmoidLowering,
      SiLULowering,
      SliceLowering,
      SoftmaxLowering,
      SoftplusLowering,
      SoftsignLowering,
      SwapChannelLowering,
      TileLowering,
      UnsqueezeLowering,
      UpsampleLowering,
      InterpLowering,
      StridedSliceLowering,
      ReduceLowering,
      PackLowering,
      SubLowering,
      SubConstLowering,
      SqrtLowering,
      SqueezeLowering,
      SwapDimInnerLowering,
      WhereLowering,
      MaskedFillLowering,
      CompareLowering,
      CompareConstLowering,
      ErfLowering,
      HardSigmoidLowering,
      HardSwishLowering,
      LayerNormLowering,
      TanLowering,
      TanhLowering,
      TopKLowering,
      AttentionLowering,
      ReverseLowering,
      PixelNormLowering,
      YoloDetectionLowering,
      InstanceNormLowering,
      GroupNormLowering,
      DetectionOutputLowering,
      ShuffleChannelLowering
      // clang-format on
      >(patterns->getContext());
}
} // namespace bm1684x
} // namespace tpu_mlir
