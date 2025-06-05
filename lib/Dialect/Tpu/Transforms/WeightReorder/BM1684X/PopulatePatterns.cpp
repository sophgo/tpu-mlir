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
using namespace bm1684x;
void populateWeightReorderBM1684XPatterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
    WeightReorder<tpu::Conv2DOp, int8_t>,
    WeightReorder<tpu::Conv2DOp, Float8E4M3FNType>,
    WeightReorder<tpu::Conv2DOp, Float8E5M2Type>,
    WeightReorder<tpu::Conv2DOp, BFloat16Type>,
    WeightReorder<tpu::Conv2DOp, Float16Type>,
    WeightReorder<tpu::Conv2DOp, Float32Type>,
    WeightReorder<tpu::Conv3DOp, int8_t>,
    WeightReorder<tpu::Conv3DOp, BFloat16Type>,
    WeightReorder<tpu::Conv3DOp, Float16Type>,
    WeightReorder<tpu::Conv3DOp, Float32Type>,
    WeightReorder<tpu::DeconvOp, int8_t>,
    WeightReorder<tpu::DeconvOp, BFloat16Type>,
    WeightReorder<tpu::DeconvOp, Float16Type>,
    WeightReorder<tpu::DeconvOp, Float32Type>,
    WeightReorder<tpu::Deconv3DOp, int8_t>,
    WeightReorder<tpu::Deconv3DOp, BFloat16Type>,
    WeightReorder<tpu::Deconv3DOp, Float16Type>,
    WeightReorder<tpu::Deconv3DOp, Float32Type>,
    WeightReorder<tpu::GRUOp, Float32Type>,
    WeightReorder<tpu::LSTMOp, Float32Type>,
    WeightReorder<tpu::MatMulOp, int8_t>,
    WeightReorder<tpu::MatMulOp, BFloat16Type>,
    WeightReorder<tpu::MatMulOp, Float16Type>,
    WeightReorder<tpu::AttentionOp, int8_t>,
    WeightReorder<tpu::AttentionOp, BFloat16Type>,
    WeightReorder<tpu::AttentionOp, Float16Type>,
    WeightReorder<tpu::A16MatMulOp, Float16Type>,
    WeightReorder<tpu::A16MatMulOp, BFloat16Type>
  >(patterns->getContext());
  // clang-format on
};

} // namespace tpu
} // namespace tpu_mlir
