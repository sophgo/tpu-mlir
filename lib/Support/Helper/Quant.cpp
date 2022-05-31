//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "sophgo/Support/Helper/Quant.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Quant/FakeQuantSupport.h"
#include "mlir/Dialect/Quant/QuantizeUtils.h"
#include "float.h"
#include <map>
using namespace llvm;
using namespace mlir;
namespace sophgo {
namespace helper {

constexpr double Quant::QMAX_INT8;
constexpr int Quant::BITS_INT8;
constexpr llvm::StringRef Quant::Type::INT8;
constexpr llvm::StringRef Quant::Type::BF16;
constexpr llvm::StringRef Quant::Type::F16;
constexpr llvm::StringRef Quant::Type::F32;

void Quant::getScaleAndZeroPoint(int64_t qmin, int64_t qmax, double rmin,
                                 double rmax, double &scale,
                                 int64_t &zeroPoint) {
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

void Quant::getScaleAndZeroPoint(Value v, double &scale, int64_t &zeropoint) {
  if (isCalibratedType(v)) {
    auto qtype = getCalibratedType(v);
    getScaleAndZeroPoint(-128, 127, qtype.getMin(), qtype.getMax(), scale,
                         zeropoint);
  } else if (isUniformQuantized(v)) {
    auto qtype = getUniformQuantizedType(v);
    scale = qtype.getScale();
    zeropoint = qtype.getZeroPoint();
  } else {
    v.dump();
    llvm_unreachable("can't get scale and zeropoint");
  }
}

void Quant::setQuantInt8Type(Value v, bool asymmetric, bool signType) {
  auto type = v.getType().cast<RankedTensorType>();
  auto ctx = v.getContext();
  auto cali_type = getCalibratedType(v);
  auto max = cali_type.getMax();
  auto min = cali_type.getMin();
  double scale;
  int64_t zeropoint = 0;
  int64_t qmin = -128, qmax = 127;
  uint32_t flag = quant::QuantizationFlags::Signed;
  if (asymmetric) {
    if (signType == false) {
      flag = 0;
      qmin = 0;
      qmax = 255;
    }
    getScaleAndZeroPoint(qmin, qmax, min, max, scale, zeropoint);
  } else {
    assert(max == -min); //ufw 1686实现不是完全对称
    scale = max / 127.0;
  }
  auto qtype = quant::UniformQuantizedType::get(flag, IntegerType::get(ctx, 8),
                                                cali_type.getExpressedType(),
                                                scale, zeropoint, qmin, qmax);
  auto new_type = RankedTensorType::get(type.getShape(), qtype);
  v.setType(new_type);
}

void Quant::setQuantExpressType(Value v) {
  if (!isUniformQuantized(v)) {
    return;
  }
  auto type = v.getType().cast<RankedTensorType>();
  auto expresstype =
      type.getElementType().cast<quant::QuantizedType>().getExpressedType();
  auto new_type = RankedTensorType::get(type.getShape(), expresstype);
  v.setType(new_type);
}

} // namespace helper
} // namespace sophgo
