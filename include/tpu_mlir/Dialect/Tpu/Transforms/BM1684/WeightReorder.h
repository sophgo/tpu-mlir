//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/GmemAllocator.hpp"
#include "tpu_mlir/Support/Helper/Module.h"
#include <cstdint>

using namespace llvm;
using namespace mlir;
namespace tpu_mlir {

namespace bm1684 {

template <class Op, typename T>
class WeightReorder : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override;
};

void populateWeightReorderPatterns(RewritePatternSet *patterns);
} // namespace bm1684
} // namespace tpu_mlir
