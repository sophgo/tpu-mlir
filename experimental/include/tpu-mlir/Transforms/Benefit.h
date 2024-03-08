//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "tpu-mlir/Dialect/BM1690/IR/StructuredOpsInterfaces.h"
#include "tpu-mlir/Transforms/StructuredTransform.h"

namespace tpu_mlir {
using namespace mlir;

class TransformBenefit {
private:
  bm1690::TPUISATraits target;
  // map (d0, d1...) -> shape(num...)
  // compute "cycle"
  // unroll -> shape(d?)
  // dropSymbol -> 1
  // MergeDim -> reshape
  // Permutation -> shape change?
  // ExpandDim -> 1
  // DecomposeExpr -> 1
  // source computation -> target TPUISATraits
  // getShape
public:
  TransformBenefit(bm1690::TPUISATraits target) : target(target) {}

  SmallVector<int64_t> getCycle(Transforms &transforms,
                                SmallVector<int64_t> shape);

private:
  void getCycle(Transforms &transforms, SmallVector<int64_t> shape,
                SmallVector<int64_t> &cycles, int64_t cycle);
};
} // namespace tpu_mlir
