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
constexpr llvm::StringRef Quant::Type::INT4;
constexpr llvm::StringRef Quant::Type::BF16;
constexpr llvm::StringRef Quant::Type::F16;
constexpr llvm::StringRef Quant::Type::F32;

void Quant::getScaleAndZeroPoint(double rmin, double rmax, double &scale,
                                 int64_t &zeroPoint, int bitwidth) {
  int qmin = rmin < 0 ? -128 : 0;
  int qmax = rmin < 0 ? 127 : 255;
  if (bitwidth == 4) {
    qmin = rmin < 0 ? -8 : 0;
    qmax = rmin < 0 ? 7 : 15;
  }
  // Determine the scale.
  double qminDouble = qmin;
  double qmaxDouble = qmax;
  scale = (rmax - rmin) / (qmaxDouble - qminDouble);
  double zeroPointFromMin = qminDouble - rmin / scale;

  // Now nudge the zero point to be an integer.
  zeroPoint = round(zeroPointFromMin);
  if (zeroPointFromMin < qminDouble) {
    zeroPoint = qmin;
    scale = rmax / (qmaxDouble - zeroPoint);
  } else if (zeroPointFromMin > qmaxDouble) {
    zeroPoint = qmax;
    scale = rmin / (qminDouble - zeroPoint);
  }
}

double Quant::getScale(double threshold, bool sign, int bitwidth) {
  if (bitwidth == 8) {
    if (sign) {
      return threshold / 127.0;
    } else {
      return threshold / 255.0;
    }
  } else {
    if (sign) {
      return threshold / 7.0;
    } else {
      return threshold / 15.0;
    }
  }
}

void Quant::getScaleAndZeroPoint(Value v, double &scale, int64_t &zeropoint,
                                 bool asymmetric, int bitwidth) {
  bool sign;
  getScaleAndZeroPoint(v, scale, zeropoint, sign, asymmetric, bitwidth);
}

void Quant::getScaleAndZeroPoint(Value v, double &scale, int64_t &zeropoint,
                                 bool &sign, bool asymmetric, int bitwidth) {
  if (isCalibratedType(v)) {
    auto qtype = getCalibratedType(v);
    auto max = qtype.getMax();
    auto min = qtype.getMin();
    sign = min < 0;
    if (asymmetric) {
      getScaleAndZeroPoint(min, max, scale, zeropoint, bitwidth);
    } else {
      zeropoint = 0;
      scale = getScale(max, sign, bitwidth);
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
} // namespace helper
} // namespace tpu_mlir
