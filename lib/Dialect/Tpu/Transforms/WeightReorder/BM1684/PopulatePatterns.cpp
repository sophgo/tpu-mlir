//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "../WeightReorder.h"

namespace tpu_mlir {
namespace tpu {

using namespace bm1684;

void populateWeightReorderBM1684Patterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
    WeightReorder<tpu::AddOp, int8_t>,
    WeightReorder<tpu::Conv2DOp, int8_t>,
    WeightReorder<tpu::Conv2DOp, Float32Type>,
    WeightReorder<tpu::Conv3DOp, int8_t>,
    WeightReorder<tpu::Conv3DOp, Float32Type>,
    WeightReorder<tpu::DeconvOp, int8_t>,
    WeightReorder<tpu::DeconvOp, Float32Type>,
    WeightReorder<tpu::Deconv3DOp, Float32Type>,
    WeightReorder<tpu::GroupNormOp, Float32Type>,
    WeightReorder<tpu::GRUOp, Float32Type>,
    WeightReorder<tpu::LSTMOp, Float32Type>,
    WeightReorder<tpu::MulOp, int8_t>,
    WeightReorder<tpu::PReluOp, int8_t>,
    WeightReorder<tpu::ScaleOp, int8_t>,
    WeightReorder<tpu::SubOp, int8_t>>(patterns->getContext());
  // clang-format on
};

} // namespace tpu
} // namespace tpu_mlir
