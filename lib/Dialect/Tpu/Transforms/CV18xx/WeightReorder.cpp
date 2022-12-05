//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/CV18xx/WeightReorder.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/GmemAllocator.hpp"
#include "tpu_mlir/Support/Helper/Module.h"
#include <cstdint>

using namespace llvm;
using namespace mlir;
namespace tpu_mlir {

namespace cv18xx {

void populateWeightReorderPatterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
    WeightReorder<tpu::Conv2DOp, int8_t>,
    WeightReorder<tpu::Conv2DOp, BFloat16Type>,
    WeightReorder<tpu::DeconvOp, int8_t>,
    WeightReorder<tpu::DeconvOp, BFloat16Type>,
    WeightReorder<tpu::GRUOp, int8_t>,
    WeightReorder<tpu::GRUOp, BFloat16Type>,
    WeightReorder<tpu::LSTMCVIOp, BFloat16Type>
  >(patterns->getContext());
  // clang-format on
};

} // namespace cv18xx
} // namespace tpu_mlir
