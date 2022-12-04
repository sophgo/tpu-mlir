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
    scale = rmax / (qmaxDouble - zeroPoint);
  } else if (zeroPointFromMin > qmaxDouble) {
    zeroPoint = qmax;
    scale = rmin / (qminDouble - zeroPoint);
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

mlir::Type Quant::getQuantBoolType(Value v) {
  auto type = v.getType().cast<RankedTensorType>();
  auto ctx = v.getContext();
  auto cali_type = getCalibratedType(v);
  int64_t qmin = -128, qmax = 127;
  uint32_t flag = quant::QuantizationFlags::Signed;
  if (cali_type.getMin() >= 0) {
    qmin = 0;
    qmax = 255;
    flag = 0;
  }
  auto qtype = quant::UniformQuantizedType::get(flag, IntegerType::get(ctx, 8),
                                                cali_type.getExpressedType(),
                                                1.0, 0, qmin, qmax);
  return RankedTensorType::get(type.getShape(), qtype);
}

uint16_t Quant::to_bf16(float src, bool rounding) {
  // To convert a float 32 to bfloat16, a float 32 can be viewed as 32 bits
  // with the following tags:
  //
  // Sign |  Exp (8 bits) | Frac (23 bits)
  //  S     EEEEEEEE         FFFFFFLRTTTTTTTTTTTTTTT
  //
  //  S: Sign bit.
  //  E: Exponent bits.
  //  F: First 6 bits of fraction.
  //  L: Least significant bit of resulting bfloat16 if we truncate away the
  //  rest of the float32. This is also the 7th bit of fraction
  //  R: Rounding bit, 8th bit of fraction.
  //  T: Sticky bits, rest of fraction, 15 bits.

  // At this point, src must be either a normal float, or +/-infinity or
  // zero.
  uint16_t u16_val;
  if (rounding) {
    // Fast rounding algorithm that rounds a half value to nearest even. This
    // reduces expected error when we convert a large number of floats.
    //
    // The fast converting algorithm simply adds lsb (L) to 0x7fff (15 bits of
    // 1s) as the rounding bias, adds the rounding bias to the input, then
    // truncates the last 16 bits away.
    uint32_t u32_val = *((uint32_t *)(&src));
    uint32_t lsb = (u32_val >> 16) & 1;
    u32_val += (0x7fff + lsb);
    u16_val = ((uint16_t *)(&u32_val))[1];
    /* HW behavior */
    // infinity set to max finite positive value
    u16_val = ((u16_val & 0x7f80) == 0x7f80) ? 0x7f7f : u16_val;
  } else {
    u16_val = ((uint16_t *)(&src))[1];
  }
  return u16_val;
}

void Quant::to_bf16(float *src, uint16_t *dst, size_t size, bool rounding) {
  for (size_t i = 0; i < size; i++) {
    dst[i] = to_bf16(src[i], rounding);
  }
}

float Quant::BF16(float src, bool rounding) {
  float dst = 0;
  uint16_t u16_val = to_bf16(src, rounding);
  uint16_t *p = (uint16_t *)(&dst);
  p[1] = u16_val;
  return dst;
}

void Quant::BF16(float *src, float *dst, size_t size, bool rounding) {
  for (size_t i = 0; i < size; i++) {
    dst[i] = BF16(src[i], rounding);
  }
}
} // namespace helper
} // namespace tpu_mlir
