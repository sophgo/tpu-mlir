//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Helper/Quant.h"
#include "mlir/Dialect/Quant/FakeQuantSupport.h"
#include "mlir/IR/PatternMatch.h"

#include "float.h"
#include <map>
using namespace llvm;
using namespace mlir;
namespace tpu_mlir {
namespace helper {

constexpr double Quant::QMAX_INT8;
constexpr int Quant::BITS_INT8;
constexpr llvm::StringRef Quant::Type::INT8;
constexpr llvm::StringRef Quant::Type::BF16;
constexpr llvm::StringRef Quant::Type::F16;
constexpr llvm::StringRef Quant::Type::F32;

void Quant::getScaleAndZeroPoint(double rmin, double rmax, double &scale,
                                 int64_t &zeroPoint) {
  int qmin = rmin < 0 ? -128 : 0;
  int qmax = rmin < 0 ? 127 : 255;
  // Determine the scale.
  double qminDouble = qmin;
  double qmaxDouble = qmax;
  scale = (rmax - rmin) / (qmaxDouble - qminDouble);
  double zeroPointFromMin = qminDouble - rmin / scale;

  // Now nudge the zero point to be an integer.
  zeroPoint = round(zeroPointFromMin);
  if (zeroPointFromMin < qminDouble) {
    zeroPoint = qmin;
  } else if (zeroPointFromMin > qmaxDouble) {
    zeroPoint = qmax;
  }
}

double Quant::getScale(double threshold, bool sign) {
  if (sign) {
    return threshold / 127.0;
  } else {
    return threshold / 255.0;
  }
}

void Quant::getScaleAndZeroPoint(Value v, double &scale, int64_t &zeropoint,
                                 bool asymmetric) {
  bool sign;
  getScaleAndZeroPoint(v, scale, zeropoint, sign, asymmetric);
}

void Quant::getScaleAndZeroPoint(Value v, double &scale, int64_t &zeropoint,
                                 bool &sign, bool asymmetric) {
  if (isCalibratedType(v)) {
    auto qtype = getCalibratedType(v);
    auto max = qtype.getMax();
    auto min = qtype.getMin();
    sign = min < 0;
    if (asymmetric) {
      getScaleAndZeroPoint(min, max, scale, zeropoint);
    } else {
      zeropoint = 0;
      scale = getScale(max, sign);
    }
  } else if (isUniformQuantized(v)) {
    auto qtype = getUniformQuantizedType(v);
    scale = qtype.getScale();
    zeropoint = qtype.getZeroPoint();
    sign = qtype.isSigned();
  } else {
    v.dump();
    llvm_unreachable("can't get scale and zeropoint");
  }
}

mlir::Type Quant::getQuantInt8Type(Value v, bool asymmetric) {
  auto type = v.getType().cast<RankedTensorType>();
  auto ctx = v.getContext();
  auto cali_type = getCalibratedType(v);
  auto max = cali_type.getMax();
  auto min = cali_type.getMin();
  double scale;
  int64_t zeropoint = 0;
  getScaleAndZeroPoint(v, scale, zeropoint, asymmetric);
  int64_t qmin = -128, qmax = 127;
  uint32_t flag = quant::QuantizationFlags::Signed;
  if (min >= 0) {
    qmin = 0;
    qmax = 255;
    flag = 0;
  }
  auto qtype = quant::UniformQuantizedType::get(flag, IntegerType::get(ctx, 8),
                                                cali_type.getExpressedType(),
                                                scale, zeropoint, qmin, qmax);
  return RankedTensorType::get(type.getShape(), qtype);
}

int32_t Quant::to_int(float_t v, RoundingMode round_mode) {
  // round_mode:
  //   0 : HALF_DOWN for bm168x
  //   1 : ROUNDING_HALF_TO_EVEN for cv18xx
  //   2 : ROUNDING_DOWN (round to zero) for cv18xx
  int32_t i32_val;
  if (round_mode == ROUNDING_HALF_DOWN) {
    i32_val = std::round(v);
  } else if (round_mode == ROUNDING_DOWN) {
    i32_val = (int)v;
  } else if (round_mode == ROUNDING_HALF_TO_EVEN) {
    float fraction, integer;
    float abs_v = std::abs(v);
    fraction = std::modf(abs_v, &integer);
    i32_val = (int)integer;
    if (fraction > 0.5) {
      i32_val = i32_val + 1;
    } else if (fraction == 0.5) {
      if (i32_val & 0x01) {
        i32_val = i32_val + 1;
      }
    }
    if (v < 0) {
      i32_val = -i32_val;
    }
  } else if (round_mode == ROUNDING_HALF_UP) {
    i32_val = floor(v + 0.5);
  } else {
    llvm_unreachable("not support round_mode.");
  }
  return i32_val;
}

} // namespace helper
} // namespace tpu_mlir
