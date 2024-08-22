//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"

namespace tpu_mlir {
namespace tpu {

void doCoreParallelPattern(ModuleOp module);

void doSpecificPattern(ModuleOp module);

#ifndef CORE_PARALLEL_HPP
#define CORE_PARALLEL_HPP

// Your function and class definitions

// Generate index vector for a given shape, it treats this shape as a nested
// loop.
// eg: given a shape (2, 3), it will call the function with (0, 0) (0, 1) ...
// (1, 2)
template <typename T, class Func>
void invokeInIterationSpace(ArrayRef<T> shape, Func &&func,
                            SmallVector<T, 8> dims = {}) {
  auto dim = dims.size();
  SmallVector dimN(dims);
  dimN.push_back(0);
  for (T i = 0, n = shape[dim]; i < n; i++) {
    dimN.back() = i;
    if (dimN.size() < shape.size()) {
      invokeInIterationSpace(shape, func, dimN);
    } else {
      func(dimN);
    }
  }
}

// convert <(d0, d1) -> (d0)> to [1, 0] when the shape is 2 dimension.
// convert <(d0, d1) -> (d0)> to [1] when the shape is 1 dimension.
template <typename T>
auto getValidStride(Attribute indexMapAttr, ArrayRef<T> iterShape) {
  // filter valid shape
  auto indexMap = cast<AffineMapAttr>(indexMapAttr).getValue();
  SmallVector<T> strides(iterShape.size(), 0);
  if (indexMap.getNumResults() == 0)
    return strides;
  auto context = indexMap.getContext();
  for (int i = iterShape.size() - 1, stride = 1; i >= 0; i--) {
    if (auto outIndex =
            indexMap.getResultPosition(getAffineDimExpr(i, context))) {
      strides[i] = stride;
      stride *= iterShape[i];
    }
  }
  return strides;
}

template <typename T> T getValidIndex(ArrayRef<T> dims, ArrayRef<T> strides) {
  T index = 0;
  for (auto [a, b] : llvm::zip(dims, strides))
    index += a * b;
  return index;
}
#endif // CORE_PARALLEL_HPP

std::optional<SmallVector<Type>> getSplitTypes(Attribute valueMap, Value value,
                                               ArrayRef<int64_t> shapeParallel,
                                               int splitDim, int splitMax);
tpu::CoreParallelOp forAll(IndexingMapsInterface op, int offset, int num_core);
} // namespace tpu
} // namespace tpu_mlir
