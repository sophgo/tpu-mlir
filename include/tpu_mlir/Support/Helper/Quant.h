//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/Dialect/Quant/QuantTypes.h"

using namespace mlir;

namespace tpu_mlir {
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

  // clang-format off
  static inline bool isCalibratedType(mlir::Type type) {
    return type.cast<RankedTensorType>().getElementType().isa<quant::CalibratedQuantizedType>();
  }

  static inline bool isCalibratedType(Value v) {
    return isCalibratedType(v.getType());
  }

  template <typename... Args>
  static inline bool isCalibratedType(Value v, Args... args) {
    return isCalibratedType(v) && isCalibratedType(args...);
  }

  static inline bool isUniformQuantized(mlir::Type type) {
    return type.cast<RankedTensorType>().getElementType().isa<quant::UniformQuantizedType>();
  }

  static inline bool isUniformQuantized(Value v) {
    return isUniformQuantized(v.getType());
  }

  template <typename... Args>
  static inline bool isUniformQuantized(Value v, Args... args) {
    return isUniformQuantized(v) && isUniformQuantized(args...);
  }

  static inline quant::CalibratedQuantizedType getCalibratedType(Value v) {
    return v.getType().cast<RankedTensorType>().getElementType().cast<quant::CalibratedQuantizedType>();
  }

  static inline quant::UniformQuantizedType getUniformQuantizedType(Value v) {
    return v.getType().cast<RankedTensorType>().getElementType().cast<quant::UniformQuantizedType>();
  }

  // clang-format on

  static inline double getThreshold(Value v) {
    auto type = getCalibratedType(v);
    assert(type.getMax() == -type.getMin());
    return type.getMax();
  }

  // for asymmetric
  static void getScaleAndZeroPoint(double rmin, double rmax, double &scale,
                                   int64_t &zeroPoint);
  // for symmetric
  static double getScale(double threshold, bool sign);
  static void getScaleAndZeroPoint(Value v, double &scale, int64_t &zeropoint,
                                   bool asymmetric);
  static void getScaleAndZeroPoint(Value v, double &scale, int64_t &zeropoint,
                                   bool &sign, bool asymmetric);

  template <typename T> static inline int8_t to_int8(T value, bool bm_mode=true) {
    auto v = bm_mode ? std::round(value) : floor(value + 0.5);
    return v > 127 ? 127 : v < -128 ? -128 : v;
  }
  template <typename T> static inline uint8_t to_uint8(T value, bool bm_mode=true) {
    auto v = bm_mode ? std::round(value) : floor(value + 0.5);
    return v > 255 ? 255 : v < 0 ? 0 : v;
  }
  static mlir::Type getQuantInt8Type(Value v, bool asymmetric = false);
};
} // namespace helper
} // namespace tpu_mlir
