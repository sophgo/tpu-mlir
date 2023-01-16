//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/BM1684/WeightReorder.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/WeightReorder.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/GmemAllocator.hpp"
#include "tpu_mlir/Support/Module.h"
#include <cstdint>

using namespace llvm;
using namespace mlir;
namespace tpu_mlir {
namespace bm1684x {

void populateWeightReorderPatterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
    WeightReorder<tpu::Conv1DOp, int8_t>,
    WeightReorder<tpu::Conv1DOp, BFloat16Type>,
    WeightReorder<tpu::Conv1DOp, Float16Type>,
    WeightReorder<tpu::Conv1DOp, Float32Type>,
    WeightReorder<tpu::Conv2DOp, int8_t>,
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
    WeightReorder<tpu::GRUOp, Float32Type>,
    WeightReorder<tpu::LSTMOp, Float32Type>,
    WeightReorder<tpu::MatMulOp, int8_t>
  >(patterns->getContext());
  // clang-format on
};

} // namespace bm1684x

namespace bm1684 {

void populateWeightReorderPatterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
    WeightReorder<tpu::Conv1DOp, int8_t>,
    WeightReorder<tpu::Conv2DOp, int8_t>,
    WeightReorder<tpu::DeconvOp, int8_t>,
    WeightReorder<tpu::Conv2DOp, Float32Type>
  >(patterns->getContext());
  // clang-format on
};

} // namespace bm1684

} // namespace tpu_mlir
