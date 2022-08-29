//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinAttributes.h"

namespace tpu_mlir {

using namespace mlir;

template <typename attr_T, typename out_T>
SmallVector<out_T> ArrayAttrADP(ArrayAttr attr,
                                std::function<out_T(Attribute)> &&func) {
  return llvm::to_vector<8>(llvm::map_range(attr, func));
}

template <typename attr_T, typename out_T>
SmallVector<out_T> ArrayAttrADP(ArrayAttr attr);

template <>
inline SmallVector<double> ArrayAttrADP<FloatAttr, double>(ArrayAttr attr) {
  return ArrayAttrADP<FloatAttr, double>(attr, [](Attribute attr) -> double {
    return attr.cast<FloatAttr>().getValue().convertToDouble();
  });
}

template <>
inline SmallVector<float> ArrayAttrADP<FloatAttr, float>(ArrayAttr attr) {
  return ArrayAttrADP<FloatAttr, float>(attr, [](Attribute attr) -> float {
    return attr.cast<FloatAttr>().getValue().convertToFloat();
  });
}

template <>
inline SmallVector<int64_t> ArrayAttrADP<IntegerAttr, int64_t>(ArrayAttr attr) {
  return ArrayAttrADP<IntegerAttr, int64_t>(
      attr, [](Attribute attr) -> int64_t {
        return attr.cast<IntegerAttr>().getInt();
      });
}

template <>
inline SmallVector<uint64_t>
ArrayAttrADP<IntegerAttr, uint64_t>(ArrayAttr attr) {
  return ArrayAttrADP<IntegerAttr, uint64_t>(
      attr, [](Attribute attr) -> uint64_t {
        return attr.cast<IntegerAttr>().getUInt();
      });
}
} // namespace tpu_mlir
