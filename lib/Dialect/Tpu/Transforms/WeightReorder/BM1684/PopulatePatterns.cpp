//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "../WeightReorder.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"


namespace tpu_mlir {
namespace tpu {

using namespace bm1684;

void populateWeightReorderBM1684Patterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
    WeightReorder<tpu::Conv2DOp, int8_t>,
    WeightReorder<tpu::Conv2DOp, Float32Type>,
    WeightReorder<tpu::DeconvOp, int8_t>,
    WeightReorder<tpu::DeconvOp, Float32Type>,
    WeightReorder<tpu::LSTMOp, Float32Type>  >(patterns->getContext());
  // clang-format on
};

} // namespace tpu
} // namespace tpu_mlir
