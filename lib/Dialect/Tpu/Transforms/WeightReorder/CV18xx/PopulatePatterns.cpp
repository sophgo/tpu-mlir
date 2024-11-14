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

using namespace cv18xx;

void populateWeightReorderCV18xxPatterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
    WeightReorder<tpu::Conv2DOp, int8_t>,
    WeightReorder<tpu::Conv2DOp, BFloat16Type>,
    WeightReorder<tpu::DeconvOp, int8_t>,
    WeightReorder<tpu::DeconvOp, BFloat16Type>,
    WeightReorder<tpu::Conv3DOp, BFloat16Type>,
    WeightReorder<tpu::GRUOp, int8_t>,
    WeightReorder<tpu::GRUOp, BFloat16Type>,
    WeightReorder<tpu::LSTMOp, BFloat16Type>
  >(patterns->getContext());
  // clang-format on
};

} // namespace tpu
} // namespace tpu_mlir
