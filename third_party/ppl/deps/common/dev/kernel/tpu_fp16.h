#ifndef TPU_FP16_H_
#define TPU_FP16_H_

#include "common.h"
#include "fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline bool fp32_lt(fp32 a, fp32 b) {
  bool res = true;
  if ((a.format.exp == 255 && a.format.frac != 0) ||
      (b.format.exp == 255 && b.format.frac != 0)) {
    // a or b is NAN
    res = (a.bits != b.bits);
  } else {
    res = a.fval < b.fval;
  }
  return res;
}

static inline bool fp32_gt(fp32 a, fp32 b) {
  bool res = true;
  if ((a.format.exp == 255 && a.format.frac != 0) ||
      (b.format.exp == 255 && b.format.frac != 0)) {
    // a or b is NAN
    res = false;
  } else {
    res= a.fval > b.fval;
  }
  return res;
}

static inline bool fp32_eq(fp32 a, fp32 b) {
  bool res = true;
  if ((a.format.exp == 255 && a.format.frac != 0) ||
      (b.format.exp == 255 && b.format.frac != 0)) {
    // a or b is NAN
    res = (a.bits == b.bits);
  } else {
    res= (a.fval == b.fval);
  }
  return res;
}

static inline fp32 fp32_max(fp32 a, fp32 b) {
  if (a.format.exp == 0) a.bits &= 0x80000000;
  if (b.format.exp == 0) b.bits &= 0x80000000;
  fp32 res32 = fp32_gt(a, b) ? a : b;
  return res32;
}

static inline fp32 fp32_min(fp32 a, fp32 b) {
  if (a.format.exp == 0) a.bits &= 0x80000000;
  if (b.format.exp == 0) b.bits &= 0x80000000;
  fp32 res32 = fp32_lt(a, b) ? a : b;
  return res32;
}

/*
 * Functions of fp16
 */

/// Cast fp16 data to fp32 data
static inline fp32 fp16_to_fp32(fp16 half) {
  fp32 res;
  if (half.format.exp == 31 && half.format.frac != 0) {
    // NAN which had beed checked with IC
    res.bits = UINT32_C(0xFFC00000);
    return res;
  }
  res.bits = fp16_ieee_to_fp32_bits(half.bits);
  return res;
}

/// Cast fp32 data to fp16 data
/// The round mode is the same with default CPU standard
/// Default round mode: round to nearest with tie to even
/// The round mode can be change through fesetround()
static inline fp16 fp32_to_fp16(fp32 single) {
  fp16 res;
  if (single.format.exp == 255 && single.format.frac != 0) {
    // NAN which had been checked with IC
    res.bits = UINT16_C(0x7FFF);
    return res;
  }
  res.bits = fp16_ieee_from_fp32_value(single.fval);
  return res;
}

/// a + b
static inline fp16 fp16_add(fp16 a, fp16 b) {
  fp32 a32, b32, res32;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  res32.fval = a32.fval + b32.fval;
  fp16 res16 = fp32_to_fp16(res32);
  return res16;
}

/// a - b
static inline fp16 fp16_sub(fp16 a, fp16 b) {
  fp32 a32, b32, res32;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  res32.fval = a32.fval - b32.fval;
  fp16 res16 = fp32_to_fp16(res32);
  return res16;
}

/// a * b
static inline fp16 fp16_mul(fp16 a, fp16 b) {
  fp32 a32, b32, res32;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  res32.fval = a32.fval * b32.fval;
  fp16 res16 = fp32_to_fp16(res32);
  return res16;
}

/// a > b
static inline bool fp16_gt(fp16 a, fp16 b) {
  bool res;
  fp32 a32, b32;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  if ((a.format.exp == 31 && a.format.frac != 0) ||
      (b.format.exp == 31 && b.format.frac != 0)) {
    res = false;
  } else {
    res = a32.fval > b32.fval;
  }
  return res;
}

/// a < b
static inline bool fp16_lt(fp16 a, fp16 b) {
  bool res;
  fp32 a32, b32;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  if ((a.format.exp == 31 && a.format.frac != 0) ||
      (b.format.exp == 31 && b.format.frac != 0)) {
    res = a.bits != b.bits;
  } else {
    res = a32.fval < b32.fval;
  }
  return res;
}

/// a == b
static inline bool fp16_eq(fp16 a, fp16 b) {
  bool res;
  fp32 a32, b32;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  if ((a.format.exp == 31 && a.format.frac != 0) ||
      (b.format.exp == 31 && b.format.frac != 0)) {
    res = a.bits == b.bits;
  } else {
    res = a32.fval == b32.fval;
  }
  return res;
}

/// a != b
static inline bool fp16_neq(fp16 a, fp16 b) {
  bool res;
  fp32 a32, b32;
  a32 = fp16_to_fp32(a);
  b32 = fp16_to_fp32(b);
  if ((a.format.exp == 31 && a.format.frac != 0) ||
      (b.format.exp == 31 && b.format.frac != 0)) {
    res = a.bits != b.bits;
  } else {
    res = a32.fval != b32.fval;
  }
  return res;
}

/// max(a, b)
static inline fp16 fp16_max(fp16 a, fp16 b) {
  fp16 res16 = fp16_gt(a, b) ? a : b;
  return res16;
}

/// min(a, b)
static inline fp16 fp16_min(fp16 a, fp16 b) {
  fp16 res16 = fp16_lt(a, b) ? a : b;
  return res16;
}

/*
 * Functions of bf16
 */

/// Cast bf16 data to fp32 data
static inline fp32 bf16_to_fp32(bf16 half) {
  fp32 res;
  // TODO(guoyue) NAN need check with IC
  res.bits = (uint32_t)(half.bits) << 16;
  return res;
}

/// Cast fp32 data to bf16 data
/// The round mode is the same with default CPU standard
/// Default round mode: round to nearest with tie to even
/// The round mode can be change through fesetround()
static inline bf16 fp32_to_bf16(fp32 single) {
  bf16 res;
  if (single.format.exp == 255) {
    if (single.format.frac != 0) {
      // NAN which had been checked with IC
      res.bits = 0x7fff;
    } else {
      // INF
      res.bits = (uint16_t)(single.bits >> 16);
    }
  } else if (single.format.exp == 0) {
    // zero
    res.bits = 0x0;
    res.format.sign = single.format.sign;
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
    res.bits = sign_exp + bf16_mantissa;
  }
  return res;
}

/// a + b
static inline bf16 bf16_add(bf16 a, bf16 b) {
  fp32 a32, b32, res32;
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  res32.fval = a32.fval + b32.fval;
  bf16 res16 = fp32_to_bf16(res32);
  if (res32.format.exp == 0) {
    Double tmp;
    tmp.double_val = (double)a32.fval + (double)b32.fval;
    if ((((tmp.bits>>52) & 0x7ff) == 0x380) && (((tmp.bits>>44) & 0xff) == 0xff)) {
      int sign = res16.format.sign;
      res16.bits = 0x80;
      res16.format.sign = sign;
    }
  }
  return res16;
}

/// a - b
static inline bf16 bf16_sub(bf16 a, bf16 b) {
  fp32 a32, b32, res32;
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  res32.fval = a32.fval - b32.fval;
  bf16 res16 = fp32_to_bf16(res32);
  if (res32.format.exp == 0) {
    Double tmp;
    tmp.double_val = (double)a32.fval - (double)b32.fval;
    if ((((tmp.bits>>52) & 0x7ff) == 0x380) && (((tmp.bits>>44) & 0xff) == 0xff)) {
      int sign = res16.format.sign;
      res16.bits = 0x80;
      res16.format.sign = sign;
    }
  }
  return res16;
}

/// a * b
static inline bf16 bf16_mul(bf16 a, bf16 b) {
  fp32 a32, b32, res32;
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  res32.fval = a32.fval * b32.fval;
  bf16 res16 = fp32_to_bf16(res32);
  if (res32.format.exp == 0) {
    Double tmp;
    tmp.double_val = (double)a32.fval * (double)b32.fval;
    if ((((tmp.bits>>52) & 0x7ff) == 0x380) && (((tmp.bits>>44) & 0xff) == 0xff)) {
      int sign = res16.format.sign;
      res16.bits = 0x80;
      res16.format.sign = sign;
    }
  }
  return res16;
}

/// a > b
static inline bool bf16_gt(bf16 a, bf16 b) {
  bool res;
  fp32 a32, b32;
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  if ((a.format.exp == 255 && a.format.frac != 0) ||
      (b.format.exp == 255 && b.format.frac != 0)) {
    res = false;
  } else {
    res = a32.fval > b32.fval;
  }
  return res;
}

/// a < b
static inline bool bf16_lt(bf16 a, bf16 b) {
  bool res;
  fp32 a32, b32;
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  if ((a.format.exp == 255 && a.format.frac != 0) ||
      (b.format.exp == 255 && b.format.frac != 0)) {
    res = a.bits != b.bits;
  } else {
    res = a32.fval < b32.fval;
  }
  return res;
}

/// a == b
static inline bool bf16_eq(bf16 a, bf16 b) {
  bool res;
  fp32 a32, b32;
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  if ((a.format.exp == 255 && a.format.frac != 0) ||
      (b.format.exp == 255 && b.format.frac != 0)) {
    res = a.bits == b.bits;
  } else {
    res = a32.fval == b32.fval;
  }
  return res;
}

/// a != b
static inline bool bf16_neq(bf16 a, bf16 b) {
  bool res;
  fp32 a32, b32;
  a32 = bf16_to_fp32(a);
  b32 = bf16_to_fp32(b);
  if ((a.format.exp == 255 && a.format.frac != 0) ||
      (b.format.exp == 255 && b.format.frac != 0)) {
    res = a.bits != b.bits;
  } else {
    res = a32.fval != b32.fval;
  }
  return res;
}

/// max(a, b)
static inline bf16 bf16_max(bf16 a, bf16 b) {
  if (a.format.exp == 0) a.bits &= 0x8000;
  if (b.format.exp == 0) b.bits &= 0x8000;
  bf16 res16 = bf16_gt(a, b) ? a : b;
  return res16;
}

/// min(a, b)
static inline bf16 bf16_min(bf16 a, bf16 b) {
  if (a.format.exp == 0) a.bits &= 0x8000;
  if (b.format.exp == 0) b.bits &= 0x8000;
  bf16 res16 = bf16_lt(a, b) ? a : b;
  return res16;
}

#ifdef __cplusplus
}
#endif

#endif
