//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Helper/Quant.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Quant/FakeQuantSupport.h"
#include "mlir/Dialect/Quant/QuantizeUtils.h"
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

} // namespace helper
} // namespace tpu_mlir
