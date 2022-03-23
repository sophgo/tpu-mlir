#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Quant/QuantTypes.h"

using namespace mlir;

namespace sophgo {
namespace helper {
struct Quant {
  struct Type {
    static constexpr llvm::StringRef INT8 = "INT8";
    static constexpr llvm::StringRef UINT8 = "UINT8";
    static constexpr llvm::StringRef BF16 = "BF16";
    static constexpr llvm::StringRef FP16 = "FP16";
    static constexpr llvm::StringRef FP32 = "FP32";
  };

  template<typename QType>
  static  bool isQuantizedType(Value v) {
    return v.getType().cast<RankedTensorType>().getElementType().isa<QType>();
  }

  template<typename QType>
  static  QType getQuantizedType(Value v) {
    return v.getType().cast<RankedTensorType>().getElementType().cast<QType>();
  }

  static inline bool isUniformQuantized(Value v) {
    return isQuantizedType<quant::UniformQuantizedType>(v) ||
           isQuantizedType<quant::UniformQuantizedPerAxisType>(v);
  }

  static inline double getThreshold(Value v) {
    auto type = getQuantizedType<quant::CalibratedQuantizedType>(v);
    assert(type.getMax() == -type.getMin());
    return type.getMax();
  }

  static inline double getMax(Value v) {
    auto type = getQuantizedType<quant::CalibratedQuantizedType>(v);
    return type.getMax();
  }

  static inline double getMin(Value v) {
    auto type = getQuantizedType<quant::CalibratedQuantizedType>(v);
    return type.getMin();
  }

  static void setQuantInt8Type(Value v, bool asymmetric = false);
  static void setQuantExpressType(Value v);
};
} // namespace helper
} // namespace sophgo
