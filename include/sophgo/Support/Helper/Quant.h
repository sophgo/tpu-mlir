//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Quant/QuantTypes.h"

using namespace mlir;

namespace sophgo {
namespace helper {
struct Quant {

  static constexpr double QMAX_INT8 = 127.0;
  static constexpr int BITS_INT8 = 8;
  struct Type {
    static constexpr llvm::StringRef INT8 = "INT8";
    static constexpr llvm::StringRef UINT8 = "UINT8";
    static constexpr llvm::StringRef BF16 = "BF16";
    static constexpr llvm::StringRef F16 = "F16";
    static constexpr llvm::StringRef F32 = "F32";
  };

  static bool isCalibratedType(Value v) {
    return v.getType()
        .cast<RankedTensorType>()
        .getElementType()
        .isa<quant::CalibratedQuantizedType>();
  }

  static inline bool isUniformQuantized(Value v) {
    return v.getType()
        .cast<RankedTensorType>()
        .getElementType()
        .isa<quant::UniformQuantizedType>();
  }

  static inline quant::CalibratedQuantizedType getCalibratedType(Value v) {
    return v.getType()
        .cast<RankedTensorType>()
        .getElementType()
        .cast<quant::CalibratedQuantizedType>();
  }

  static inline quant::UniformQuantizedType getUniformQuantizedType(Value v) {
    return v.getType()
        .cast<RankedTensorType>()
        .getElementType()
        .cast<quant::UniformQuantizedType>();
  }

  static inline double getThreshold(Value v) {
    auto type = getCalibratedType(v);
    assert(type.getMax() == -type.getMin());
    return type.getMax();
  }

  static void getScaleAndZeroPoint(int64_t qmin, int64_t qmax, double rmin,
                                   double rmax, double &scale,
                                   int64_t &zeroPoint);
  static void getScaleAndZeroPoint(Value v, double &scale, int64_t &zeropoint);

  template<typename T>
  static inline int8_t to_int8(T value) {
    auto v = std::round(value);
    return v > 127 ? 127 : v < -128 ? -128 : v;
  }

  static void setQuantInt8Type(Value v, bool asymmetric = false,
                               bool sighType = true);
  static void setQuantExpressType(Value v);
};
} // namespace helper
} // namespace sophgo
