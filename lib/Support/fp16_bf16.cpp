#include "sophgo/Support/fp16_bf16.h"
#include <cmath>

fp32 fp16_to_fp32(const fp16& half) {
  fp32 res;
  if (half.data.format.exp == 31 && half.data.format.frac != 0) {
    // NAN which had beed checked with IC
    res.bits = UINT32_C(0xFFC00000);
    return res;
  }
  res.bits = fp16_ieee_to_fp32_bits(half.data.bits);
  return res;
}

fp16 fp32_to_fp16(const fp32& single) {
  fp16 res;
  if (single.format.exp == 255 && single.format.frac != 0) {
    // NAN which had been checked with IC
    res.data.bits = UINT16_C(0x7FFF);
    return res;
  }
  res.data.bits = fp16_ieee_from_fp32_value(single.fval);
  return res;
}

fp32 bf16_to_fp32(const bf16& half) {
  fp32 res;
  // TODO(guoyue) NAN need check with IC
  res.bits = (uint32_t)(half.data.bits) << 16;
  return res;
}

bf16 fp32_to_bf16(const fp32& single) {
  bf16 res;
  if (single.format.exp == 255) {
    if (single.format.frac != 0) {
      // NAN which had been checked with IC
      res.data.bits = 0x7fff;
    } else {
      // INF
      res.data.bits = (uint16_t)(single.bits >> 16);
    }
  } else if (single.format.exp == 0) {
    // zero
    res.data.bits = 0x0;
    res.data.format.sign = single.format.sign;
  } else {
    const uint16_t sign_exp = (single.bits & UINT32_C(0xFF800000)) >> 16;
    const uint32_t mantissa = single.bits & UINT32_C(0x7FFFFF);
    // Use CPU FP32 add to do mantissa >> 16 and rounding
    float base = fp32_from_bits(UINT32_C(0x48000000));
    base = fp32_from_bits(UINT32_C(0x40000000) | mantissa) + base;
    // Get new mantissa
    uint16_t bf16_mantissa = fp32_to_bits(base) & UINT32_C(0X1FF);
    bf16_mantissa = bf16_mantissa - UINT16_C(0x80);
    // Get bf16 bits
    res.data.bits = sign_exp + bf16_mantissa;
  }
  return res;
}

fp16 fp16::operator++ () {
  *this += 1;
  return *this;
}

fp16 fp16::operator++ (int dummy) {
  fp16 ret = *this;
  ++(*this);
  return ret;
}

bf16 bf16::operator++ () {
  *this += 1;
  return *this;
}

bf16 bf16::operator++ (int dummy) {
  bf16 ret = *this;
  ++(*this);
  return ret;
}

bool operator== (const fp16& a, const fp16& b) {
  bool res = a.data.bits == b.data.bits;
  return res;
}

bool operator!= (const fp16& a, const fp16& b) {
  return !(a == b);
}

bool operator> (const fp16& a, const fp16& b) {
  fp32 a_fp32 = fp16_to_fp32(a);
  fp32 b_fp32 = fp16_to_fp32(b);
  return a_fp32.fval > b_fp32.fval;
}

bool operator< (const fp16& a, const fp16& b) {
  fp32 a_fp32 = fp16_to_fp32(a);
  fp32 b_fp32 = fp16_to_fp32(b);
  return a_fp32.fval < b_fp32.fval;
}

bool operator>= (const fp16& a, const fp16& b) {
  return !(a < b);
}

bool operator<= (const fp16& a, const fp16& b) {
  return !(a > b);
}

fp16 operator+(const fp16& a, const fp16& b) {
  fp32 a32, b32, res32;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  res32.fval = a32.fval + b32.fval;
  fp16 res16 = fp32_to_fp16(res32);
  return res16;
}

fp16 operator-(const fp16& a, const fp16& b) {
  fp32 a32, b32, res32;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  res32.fval = a32.fval - b32.fval;
  fp16 res16 = fp32_to_fp16(res32);
  return res16;
}

fp16 operator*(const fp16& a, const fp16& b) {
  fp32 a32, b32, res32;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  res32.fval = a32.fval * b32.fval;
  fp16 res16 = fp32_to_fp16(res32);
  return res16;
}

fp16 operator/(const fp16& a, const fp16& b) {
  fp32 a32, b32, res32;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  res32.fval = a32.fval / b32.fval;
  fp16 res16 = fp32_to_fp16(res32);
  return res16;
}

bool operator== (const bf16& a, const bf16& b) {
  bool res = a.data.bits == b.data.bits;
  return res;
}

bool operator!= (const bf16& a, const bf16& b) {
  return !(a == b);
}

bool operator> (const bf16& a, const bf16& b) {
  fp32 a_fp32 = bf16_to_fp32(a);
  fp32 b_fp32 = bf16_to_fp32(b);
  return a_fp32.fval > b_fp32.fval;
}

bool operator< (const bf16& a, const bf16& b) {
  fp32 a_fp32 = bf16_to_fp32(a);
  fp32 b_fp32 = bf16_to_fp32(b);
  return a_fp32.fval < b_fp32.fval;
}

bool operator>= (const bf16& a, const bf16& b) {
  return !(a < b);
}

bool operator<= (const bf16& a, const bf16& b) {
  return !(a > b);
}

fp16 operator-(const fp16& a) {
  return fp16(0) - a;
}

bf16 operator+(const bf16& a, const bf16& b) {
  fp32 a32, b32, res32;
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  res32.fval = a32.fval + b32.fval;
  bf16 res16 = fp32_to_bf16(res32);
  return res16;
}

bf16 operator-(const bf16& a, const bf16& b) {
  fp32 a32, b32, res32;
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  res32.fval = a32.fval - b32.fval;
  bf16 res16 = fp32_to_bf16(res32);
  return res16;
}

bf16 operator*(const bf16& a, const bf16& b) {
  fp32 a32, b32, res32;
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  res32.fval = a32.fval * b32.fval;
  bf16 res16 = fp32_to_bf16(res32);
  return res16;
}

bf16 operator/(const bf16& a, const bf16& b) {
  fp32 a32, b32, res32;
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  res32.fval = a32.fval / b32.fval;
  bf16 res16 = fp32_to_bf16(res32);
  return res16;
}

bf16 operator-(const bf16& a) {
  return bf16(0) - a;
}

std::ostream &operator<< (std::ostream &out, const fp16& val) {
  out << float(val);
  return out;
}

std::ostream &operator<< (std::ostream &out, const bf16& val) {
  out << float(val);
  return out;
}

namespace std {
/*
  fp16 ceil(const fp16& val) {
    return fp16(std::ceil(float(val)));
  }
  bf16 ceil(const bf16& val) {
    return bf16(std::ceil(float(val)));
  }
  fp16 floor(const fp16& val) {
    return fp16(std::floor(float(val)));
  }
  bf16 floor(const bf16& val) {
    return bf16(std::floor(float(val)));
  }
*/

#define REG_ELE_FUN_TO_STD(fun, T)         \
T fun(const T& val) {                      \
  return T(std::fun(float(val)));          \
}

#define REG_F16_ELE_FUN_TO_STD(fun)        \
REG_ELE_FUN_TO_STD(fun, fp16)              \
REG_ELE_FUN_TO_STD(fun, bf16)

REG_F16_ELE_FUN_TO_STD(ceil)
REG_F16_ELE_FUN_TO_STD(floor)
REG_F16_ELE_FUN_TO_STD(log)
REG_F16_ELE_FUN_TO_STD(exp)
REG_F16_ELE_FUN_TO_STD(tanh)
REG_F16_ELE_FUN_TO_STD(fabs)
REG_F16_ELE_FUN_TO_STD(sqrt)
REG_F16_ELE_FUN_TO_STD(sin)
REG_F16_ELE_FUN_TO_STD(cos)
REG_F16_ELE_FUN_TO_STD(abs)

#undef REG_F16_ELE_FUN_TO_STD
#undef REG_ELE_FUN_TO_STD

#define REG_BINARY_TO_STD(fun, T)          \
T fun(const T& a, const T& b) {            \
  return T(std::fun(float(a), float(b)));  \
}

#define REG_F16_BINARY_TO_STD(fun)         \
REG_BINARY_TO_STD(fun, fp16)               \
REG_BINARY_TO_STD(fun, bf16)

REG_F16_BINARY_TO_STD(min)
REG_F16_BINARY_TO_STD(max)
REG_F16_BINARY_TO_STD(pow)

#undef REG_F16_BINARY_TO_STD
#undef REG_BINARY_TO_STD

  bool isinf(const fp16& val) {
    return (val.data.bits & 0x7fff) == 0x7c00;
  }
  bool isnan(const fp16& val) {
    return (val.data.bits & 0x7fff) > 0x7c00;
  }
  bool isfinite(const fp16& val) {
    return !isinf(val) && !isfinite(val);
  }

  bool isinf(const bf16& val) {
    return (val.data.bits & 0x7fff) == 0x7f80;
  }
  bool isnan(const bf16& val) {
    return (val.data.bits & 0x7fff) > 0x7f80;
  }
  bool isfinite(const bf16& val) {
    return !isinf(val) && !isfinite(val);
  }
}
