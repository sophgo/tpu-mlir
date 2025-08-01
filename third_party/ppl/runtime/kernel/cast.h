#ifndef CAST_H_
#define CAST_H_

#include "common.h"
// #include "math.h"
#include "fp16.h"
// #include "base_def.h"
#include "limits.h"

#ifdef __cplusplus
extern "C" {
#endif
static long long Right_Shift_Round(long long src, int shift_num, ROUND_MODE round_mode)
{
  if (shift_num == 0) return src;
  if (shift_num > 63) shift_num = 63;
  long long val, res;
  val = src >> shift_num;
  res = val;
  long long lo_mask = (1ull << shift_num) - 1;
  long long mant = src & lo_mask;
  long long mant_0d5 = 1ull << (shift_num - 1);
  if (round_mode == ROUND_HALF_TO_EVEN) {
    if (mant == mant_0d5) {
      res = val + (val & 1);
    } else if (mant > mant_0d5) {
      res = val + 1;
    }
  } else if (round_mode == ROUND_HALF_AWAY_FROM_ZERO) {
    if (src >= 0 && mant >= mant_0d5) {
      res = val + 1;
    } else if (src < 0 && mant > mant_0d5) {
      res = val + 1;
    }
  } else if (round_mode == ROUND_TOWARDS_ZERO) {
    if (src < 0) res = val + (mant != 0);
  } else if (round_mode == ROUND_DOWN) {
    res = val;
  } else if (round_mode == ROUND_UP) {
    res = val + (mant != 0);
  } else if (round_mode == ROUND_HALF_UP) {
    if (mant >= mant_0d5) res = val + 1;
  } else if (round_mode == ROUND_HALF_DOWN) {
    if (mant > mant_0d5) res = val + 1;
  }
  return res;
}
static uint8_t fp32_to_fp8(const float val, bool is_e5m2, bool saturate, ROUND_MODE rd_mode) {
  fp32 single;
  single.fval = val;
  uint8_t res = 0;

  uint32_t FP8_EXP_BIAS = 0;
  uint32_t FP8_EXP_MASK = 0;
  uint32_t FP8_SIGNIFICAND_BITS = 0;
  uint32_t FP8_MAXNORM = 0;
  uint32_t FP8_MANTISSA_MASK = 0;
  uint32_t FP8_INVALID_MASK = 0;
  if (is_e5m2) {
    FP8_EXP_BIAS = 15;
    FP8_EXP_MASK = 0x1f;
    FP8_SIGNIFICAND_BITS = 3;
    FP8_MAXNORM = 0x7b;
    FP8_MANTISSA_MASK = 0x3;
    FP8_INVALID_MASK = 0x1fffff;
  } else {
    FP8_EXP_BIAS = 7;
    FP8_EXP_MASK = 0xf;
    FP8_SIGNIFICAND_BITS = 4;
    FP8_MAXNORM = 0x7e;
    FP8_MANTISSA_MASK = 0x7;
    FP8_INVALID_MASK = 0xfffff;
  }

  if (single.format.exp > (127 - FP8_EXP_BIAS) && single.format.exp < 0xff) {
   const uint32_t mantissa = single.format.frac;
    const int32_t shift_num = 24 - FP8_SIGNIFICAND_BITS;
    uint32_t tmp = Right_Shift_Round(single.bits, shift_num, rd_mode);
    if (rd_mode == ROUND_DOWN && single.format.sign == 1) {
      tmp += ((mantissa & FP8_INVALID_MASK) != 0);
    } else if (rd_mode == ROUND_UP && single.format.sign == 1) {
      tmp -= ((mantissa & FP8_INVALID_MASK) != 0);
    }
    tmp <<= shift_num;
    const uint32_t exp = ((tmp >> 23) & 0xff) - 127 + FP8_EXP_BIAS;
    const uint32_t frac =
        (tmp >> (24 - FP8_SIGNIFICAND_BITS)) & FP8_MANTISSA_MASK;
    if (exp > FP8_EXP_MASK ||
        (exp == FP8_EXP_MASK && (frac == FP8_MANTISSA_MASK || is_e5m2))) {
      if (saturate) {
        res = FP8_MAXNORM;
      } else {
        // Inf in E5M2, NaN in E4M3
        res = is_e5m2 ? 0x7c : 0x7f;
      }
    } else {
      res = (exp << (FP8_SIGNIFICAND_BITS - 1)) | frac;
    }
  } else if (single.format.exp > 0 &&
             single.format.exp <= (127 - FP8_EXP_BIAS)) {
    int32_t mantissa = (single.format.frac) + (1 << 23);
    mantissa = single.format.sign ? -mantissa : mantissa;
    const int shift_num = (127 - FP8_EXP_BIAS + 1) - single.format.exp +
                          (24 - FP8_SIGNIFICAND_BITS);
    mantissa = Right_Shift_Round(mantissa, shift_num, rd_mode);
    mantissa = single.format.sign ? -mantissa : mantissa;
    res = mantissa & 0x7f;
  } else if (single.format.exp == 0xff && single.format.frac != 0) {
    // Canonical NaN
    const uint32_t xbits = 0x7fffffff | (single.format.sign << 31);
    if (is_e5m2) {
      const uint32_t mantissa =
          (xbits >> (24 - FP8_SIGNIFICAND_BITS)) & FP8_MANTISSA_MASK;
      res = 0x7e | mantissa;
    } else {
      res = 0x7f;
    }
  } else if (single.format.exp == 0xff && single.format.frac == 0) {
    if (saturate) {
      res = FP8_MAXNORM;
    } else {
      // no Inf in E4M3 and use NaN, Inf in E5M2
      res = is_e5m2 ? 0x7c : 0x7f;
    }
  }
  res |= (single.format.sign << 7);
  return res;
}
static float fp8_to_fp32(uint8_t src, bool is_e5m2) {
  uint32_t FP8_SIGNIFICAND_BITS = 0;
  uint32_t FP8_MANTISSA_MASK = 0;
  uint32_t FP8_EXP_BIAS = 0;
  uint32_t FP8_MINNORM_EXP = 0;
  uint32_t FP8_EXP_MASK = 0;
  if (is_e5m2) {
    FP8_SIGNIFICAND_BITS = 3u;
    FP8_MANTISSA_MASK = 0x3u;
    FP8_EXP_BIAS = 15u;
    FP8_MINNORM_EXP = 14u;
    FP8_EXP_MASK = 0x1fu;
  } else {
    FP8_SIGNIFICAND_BITS = 4u;
    FP8_MANTISSA_MASK = 0x7u;
    FP8_EXP_BIAS = 7u;
    FP8_MINNORM_EXP = 6u;
    FP8_EXP_MASK = 0xfu;
  }
  uint32_t sign = (src >> 7u) & 0x1u;
  uint32_t exponent = (src >> (FP8_SIGNIFICAND_BITS - 1)) & FP8_EXP_MASK;
  uint32_t mantissa = (src & FP8_MANTISSA_MASK) << (24u - FP8_SIGNIFICAND_BITS);
  if ((exponent == 0x1fu && is_e5m2) || ((src & 0x7fu) == 0x7fu && !is_e5m2)) {
    exponent = 0xffu;
    if (mantissa != 0u) {
      // NaN
      mantissa = 0x7fffffu | (sign << 23);
    }
  } else if (exponent == 0u) {
    /* Denorm or Zero */
    if (mantissa != 0u) {
      uint32_t msb = 0;
      exponent = 127 - FP8_MINNORM_EXP;
      while (msb == 0u) {
        msb = mantissa & 0x400000u;
        mantissa <<= 1u; // normalize
        exponent--;
      };
      mantissa &= 0x7fffffu;
    }
  } else {
    exponent += (127 - FP8_EXP_BIAS);
  }
  fp32 res = {.bits = 0};
  res.bits = (sign << 31u) | (exponent << 23u) | (mantissa);
  return res.fval;
}
#ifdef __cplusplus
}
#endif

#endif
