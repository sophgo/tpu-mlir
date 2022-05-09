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
    static constexpr llvm::StringRef FP16 = "FP16";
    static constexpr llvm::StringRef FP32 = "FP32";
  };

  static bool isCalibratedType(Value v) {
    return v.getType()
        .cast<RankedTensorType>()
        .getElementType()
        .isa<quant::CalibratedQuantizedType>();
  }

  static inline bool isUniformQuantized(Value v) {
    return v.getType().cast<RankedTensorType>().getElementType().isa<quant::UniformQuantizedType>();
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

  static inline double getMax(Value v) {
    auto type = getCalibratedType(v);
    return type.getMax();
  }

  static inline double getMin(Value v) {
    auto type = getCalibratedType(v);
    return type.getMin();
  }

  static inline int8_t clip_to_int8(int value) {
    return value > 127 ? 127 : value < -128 ? -128 : value;
  }

  static inline uint8_t clip_to_uint8(int value) {
    return value > 255 ? 255 : value < 0 ? 0 : value;
  }

  static void setQuantInt8Type(Value v, bool asymmetric = false,
                               bool sighType = true);
  static void setQuantExpressType(Value v);
};
} // namespace helper
} // namespace sophgo
