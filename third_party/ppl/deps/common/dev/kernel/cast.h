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
/*
 * Functions of fp4
 */
static fp32 fp4_to_fp32(uint8_t single)
{
    fp32 res;
    uint32_t ur = (uint32_t)single & 0x0FU;
    ur = (uint32_t)(ur << 28U);
    uint32_t sign = ur & 0x80000000U;
    uint32_t exponent = (uint32_t)(((ur & 0x60000000U) >> 6U) + 0x3F000000U);
    uint32_t mantissa = (ur & 0x10000000U) >> 6U;

    if (exponent == 0x3F000000U) {
        // zero or denormal
        if (mantissa != 0U) {
            // normalize
            mantissa = mantissa << 1U;
            while ((mantissa & 0x800000U) == 0U) {
                mantissa = mantissa << 1U;
                exponent = exponent - 0x800000U;
            }
            // discard implicit leading bit
            mantissa &= 0x07FFFFFU;
        } else {  // Zero
            exponent = 0U;
        }
        ur = (sign | exponent) | mantissa;
    } else {
        ur = (sign | exponent) | mantissa;
    }

    res.bits = ur;
    return res;
}

static fp16 fp4_to_fp16(uint8_t single)
{
    fp16 res;
    uint16_t ur = (uint16_t)single & 0x0FU;
    ur = (uint16_t)(ur << 12U);
    uint16_t sign = ur & 0x8000U;
    uint16_t exponent = (uint16_t)(((ur & 0x6000U) >> 3U) + 0x3800U);
    uint16_t mantissa = (ur & 0x1000U) >> 3U;

    if (exponent == 0x3800U) {
        // zero or denormal
        if (mantissa != 0U) {
            // normalize
            mantissa = (uint16_t)(mantissa << 1U);
            while ((mantissa & 0x0400U) == 0U) {
                mantissa = (uint16_t)(mantissa << 1U);
                exponent = (uint16_t)(exponent - 0x0400U);
            }
            // discard implicit leading bit
            mantissa &= 0x03FFU;
        } else {  // Zero
            exponent = 0U;
        }
        ur = (sign | exponent) | mantissa;
    } else {
        ur = (sign | exponent) | mantissa;
    }

    res.bits = ur;
    return res;
}

static bf16 fp4_to_bf16(uint8_t single)
{
    bf16 res;
    uint16_t ur = (uint16_t)single & 0x0FU;
    ur = (uint16_t)(ur << 12U);
    uint16_t sign = ur & 0x8000U;
    uint16_t exponent = (uint16_t)(((ur & 0x6000U) >> 6U) + 0x3F00U);
    uint16_t mantissa = (ur & 0x1000U) >> 6U;

    if (exponent == 0x3F00U) {
        // zero or denormal
        if (mantissa != 0U) {
            // normalize
            mantissa = (uint16_t)(mantissa << 1U);
            while ((mantissa & 0x0080U) == 0U) {
                mantissa = (uint16_t)(mantissa << 1U);
                exponent = (uint16_t)(exponent - 0x0080U);
            }
            // discard implicit leading bit
            mantissa &= 0x007FU;
        } else {  // Zero
            exponent = 0U;
        }
        ur = (sign | exponent) | mantissa;
    } else {
        ur = (sign | exponent) | mantissa;
    }

    res.bits = ur;
    return res;
}

static uint8_t fp4_to_fp8(uint8_t single, bool is_e5m2)
{
    uint8_t ur = (uint8_t)single & 0x0FU;
    ur = (uint8_t)(ur << 4U);
    uint8_t sign = ur & 0x80U;

    uint8_t exponent = is_e5m2 ? (uint8_t)(((ur & 0x60U) >> 3U) + 0x38U)
                               : (uint8_t)(((ur & 0x60U) >> 2U) + 0x30U);
    uint8_t mantissa = is_e5m2 ? ((ur & 0x10U) >> 3U) : ((ur & 0x10U) >> 2U);

    if (is_e5m2) {
        if (exponent == 0x38U) {
            // zero or denormal
            if (mantissa != 0U) {
                // normalize
                mantissa = (uint8_t)(mantissa << 1U);
                while ((mantissa & 0x04U) == 0U) {
                    mantissa = (uint8_t)(mantissa << 1U);
                    exponent = (uint8_t)(exponent - 0x04U);
                }
                // discard implicit leading bit
                mantissa &= 0x03U;
            } else {  // Zero
                exponent = 0U;
            }
            ur = (sign | exponent) | mantissa;
        } else {
            ur = (sign | exponent) | mantissa;
        }
    } else {
        if (exponent == 0x30U) {
            // zero or denormal
            if (mantissa != 0U) {
                // normalize
                mantissa = (uint8_t)(mantissa << 1U);
                while ((mantissa & 0x08U) == 0U) {
                    mantissa = (uint8_t)(mantissa << 1U);
                    exponent = (uint8_t)(exponent - 0x08U);
                }
                // discard implicit leading bit
                mantissa &= 0x07U;
            } else {  // Zero
                exponent = 0U;
            }
            ur = (sign | exponent) | mantissa;
        } else {
            ur = (sign | exponent) | mantissa;
        }
    }
    return ur;
}

static uint8_t fp16_to_fp4(const fp16 single, ROUND_MODE rd_mode)
{
    uint8_t res = 0;

    uint16_t FP4_EXP_BIAS = 1;
    uint16_t FP4_EXP_MASK = 0x3;
    uint16_t FP4_SIGNIFICAND_BITS = 2;
    uint16_t FP4_MAXNORM = 0x7;
    uint16_t FP4_MINNORM = 0xf;
    uint16_t FP4_MANTISSA_MASK = 0x1;
    uint16_t FP4_INVALID_MASK = 0x1ff;

    uint16_t FP16_EXP_BIAS = 15;
    uint16_t FP16_EXP_MASK = 0x1f;
    uint16_t FP16_SIGNIFICAND_BITS = 11;

    if (single.format.exp > (FP16_EXP_BIAS - FP4_EXP_BIAS) &&
        single.format.exp < FP16_EXP_MASK) {
        const uint16_t mantissa = single.format.frac;
        const int16_t shift_num = FP16_SIGNIFICAND_BITS - FP4_SIGNIFICAND_BITS;
        uint16_t tmp = Right_Shift_Round(single.bits, shift_num, rd_mode);
        if (rd_mode == ROUND_DOWN && single.format.sign == 1) {
            tmp += ((mantissa & FP4_INVALID_MASK) != 0);
        } else if (rd_mode == ROUND_UP && single.format.sign == 1) {
            tmp -= ((mantissa & FP4_INVALID_MASK) != 0);
        }
        tmp <<= shift_num;
        const uint16_t exp =
            ((tmp >> (FP16_SIGNIFICAND_BITS - 1)) & FP16_EXP_MASK) -
            FP16_EXP_BIAS + FP4_EXP_BIAS;
        const uint16_t frac =
            (tmp >> (FP16_SIGNIFICAND_BITS - FP4_SIGNIFICAND_BITS)) &
            FP4_MANTISSA_MASK;
        if (exp > FP4_EXP_MASK) {
            res = FP4_MAXNORM;
        } else {
            res = (exp << (FP4_SIGNIFICAND_BITS - 1)) | frac;
        }
    } else if (single.format.exp > 0 &&
               single.format.exp <= (FP16_EXP_BIAS - FP4_EXP_BIAS)) {
        int16_t mantissa =
            (single.format.frac) + (1 << (FP16_SIGNIFICAND_BITS - 1));
        mantissa = single.format.sign ? -mantissa : mantissa;
        const int shift_num = (FP16_EXP_BIAS - FP4_EXP_BIAS + 1) -
                              single.format.exp +
                              (FP16_SIGNIFICAND_BITS - FP4_SIGNIFICAND_BITS);
        mantissa = Right_Shift_Round(mantissa, shift_num, rd_mode);
        mantissa = single.format.sign ? -mantissa : mantissa;
        res = mantissa & 0x7;
    } else if (single.format.exp == FP16_EXP_MASK && single.format.frac != 0) {
        // NaN
        res = 0x0;
    } else if (single.format.exp == FP16_EXP_MASK && single.format.frac == 0) {
        // INF
        res = single.format.sign ? FP4_MINNORM : FP4_MAXNORM;
    }else if (single.format.exp == 0 && single.format.frac != 0){
        if ((rd_mode == ROUND_DOWN && single.format.sign == 1) ||
            (rd_mode == ROUND_UP && single.format.sign == 0)) {
            res = 0x1;
        }
    }
    res |= (single.format.sign << 3);
    return res;
}

static uint8_t fp8_to_fp4(const uint8_t single, bool is_e5m2,
                          ROUND_MODE rd_mode)
{
    uint8_t res = 0;

    uint8_t FP4_EXP_BIAS = 1;
    uint8_t FP4_EXP_MASK = 0x3;
    uint8_t FP4_SIGNIFICAND_BITS = 2;
    uint8_t FP4_MAXNORM = 0x7;
    uint8_t FP4_MINNORM = 0xf;
    uint8_t FP4_MANTISSA_MASK = 0x1;
    uint8_t FP4_INVALID_MASK = is_e5m2 ? 0x1 : 0x3;

    uint8_t FP8_EXP_BIAS = is_e5m2 ? 15 : 7;
    uint8_t FP8_EXP_MASK = is_e5m2 ? 0x1f : 0xf;
    uint8_t FP8_MANTISSA_MASK = is_e5m2 ? 0x3 : 0x7;
    uint8_t FP8_SIGNIFICAND_BITS = is_e5m2 ? 3 : 4;

    uint8_t single_exp = (single >> (FP8_SIGNIFICAND_BITS - 1)) & FP8_EXP_MASK;
    uint8_t single_frac = single & FP8_MANTISSA_MASK;
    uint8_t single_sign = (single >> 7u) & 0x1u;
    if (single_exp > (FP8_EXP_BIAS - FP4_EXP_BIAS) &&
        (single_exp < FP8_EXP_MASK || (!is_e5m2 && single_exp == FP8_EXP_MASK &&
                                       single_frac != FP8_MANTISSA_MASK))) {
        const uint8_t mantissa = single_frac;
        const uint8_t shift_num = FP8_SIGNIFICAND_BITS - FP4_SIGNIFICAND_BITS;
        uint8_t tmp = Right_Shift_Round(single, shift_num, rd_mode);
        if (rd_mode == ROUND_DOWN && single_sign == 1) {
            tmp += ((mantissa & FP4_INVALID_MASK) != 0);
        } else if (rd_mode == ROUND_UP && single_sign == 1) {
            tmp -= ((mantissa & FP4_INVALID_MASK) != 0);
        }
        tmp <<= shift_num;
        const uint8_t exp =
            ((tmp >> (FP8_SIGNIFICAND_BITS - 1)) & FP8_EXP_MASK) -
            FP8_EXP_BIAS + FP4_EXP_BIAS;
        const uint8_t frac =
            (tmp >> (FP8_SIGNIFICAND_BITS - FP4_SIGNIFICAND_BITS)) &
            FP4_MANTISSA_MASK;
        if (exp > FP4_EXP_MASK) {
            res = FP4_MAXNORM;
        } else {
            res = (exp << (FP4_SIGNIFICAND_BITS - 1)) | frac;
        }
    } else if (single_exp > 0 && single_exp <= (FP8_EXP_BIAS - FP4_EXP_BIAS)) {
        int8_t mantissa = single_frac + (1 << (FP8_SIGNIFICAND_BITS - 1));
        mantissa = single_sign ? -mantissa : mantissa;
        const int shift_num = (FP8_EXP_BIAS - FP4_EXP_BIAS + 1) - single_exp +
                              (FP8_SIGNIFICAND_BITS - FP4_SIGNIFICAND_BITS);
        mantissa = Right_Shift_Round(mantissa, shift_num, rd_mode);
        mantissa = single_sign ? -mantissa : mantissa;
        res = mantissa & 0x7;
    } else if (single_exp == FP8_EXP_MASK) {
        if (is_e5m2) {
            // NaN
            if (single_frac != 0)
                res = 0x0;
            // INF
            else
                res = single_sign ? FP4_MINNORM : FP4_MAXNORM;
        } else {
            // NaN, no INF in E4M3
            if (single_frac == FP8_MANTISSA_MASK) {
                res = 0x0;
            }
        }
    } else if (single_exp == 0 && single_frac != 0) {
        if ((rd_mode == ROUND_DOWN && single_sign == 1) ||
            (rd_mode == ROUND_UP && single_sign == 0)) {
            res = 0x1;
        }
    }
    res |= (single_sign << 3);
    return res;
}

/*
 * Functions of fp6 (e3m2 or e2m3)
 */
static uint8_t fp32_to_fp6(const fp32 single, bool is_e3m2, ROUND_MODE rd_mode) {
  uint8_t res = 0;

  uint32_t FP6_EXP_BIAS = 0;
  uint32_t FP6_EXP_MASK = 0;
  uint32_t FP6_SIGNIFICAND_BITS = 0; //mantissa's num_bit + 1
  uint32_t FP6_MAXNORM = 0;
  uint32_t FP6_MANTISSA_MASK = 0;
  uint32_t FP6_INVALID_MASK = 0;
  if (is_e3m2) {
    FP6_EXP_BIAS = 3;
    FP6_EXP_MASK = 0x7;
    FP6_SIGNIFICAND_BITS = 3;
    FP6_MAXNORM = 0x1f;
    FP6_MANTISSA_MASK = 0x3;
    FP6_INVALID_MASK = 0x1fffff;
  } else {
    FP6_EXP_BIAS = 1;
    FP6_EXP_MASK = 0x3;
    FP6_SIGNIFICAND_BITS = 4;
    FP6_MAXNORM = 0x1f;
    FP6_MANTISSA_MASK = 0x7;
    FP6_INVALID_MASK = 0xfffff;
  }

  uint32_t FP32_EXP_BIAS = 127;
  uint32_t FP32_EXP_MASK = 0xff;
  uint32_t FP32_SIGNIFICAND_BITS = 24;

  if (single.format.exp > (FP32_EXP_BIAS - FP6_EXP_BIAS) && single.format.exp < FP32_EXP_MASK) {
    //this menas fp6 exp > 0
    const uint32_t mantissa = single.format.frac;
    const int32_t shift_num = FP32_SIGNIFICAND_BITS - FP6_SIGNIFICAND_BITS;
    //for round use
    uint32_t tmp = Right_Shift_Round(single.bits, shift_num, rd_mode);
    if (rd_mode == ROUND_DOWN && single.format.sign == 1) {
      tmp += ((mantissa & FP6_INVALID_MASK) != 0);
    } else if (rd_mode == ROUND_UP && single.format.sign == 1) {
      tmp -= ((mantissa & FP6_INVALID_MASK) != 0);
    }
    tmp <<= shift_num;
    //get every part bit
    const uint32_t exp = ((tmp >> (FP32_SIGNIFICAND_BITS - 1)) & FP32_EXP_MASK) - (FP32_EXP_BIAS - FP6_EXP_BIAS);
    const uint32_t frac =
        (tmp >> (FP32_SIGNIFICAND_BITS - FP6_SIGNIFICAND_BITS)) & FP6_MANTISSA_MASK;
    if (exp > FP6_EXP_MASK) {
      res = FP6_MAXNORM;
    } else {
      res = (exp << (FP6_SIGNIFICAND_BITS - 1)) | frac;
    }
  } else if (single.format.exp > 0 && single.format.exp <= (FP32_EXP_BIAS - FP6_EXP_BIAS)) {
    //this menas fp6 exp < 0, let exp = 0 and mantissa do rshift
    int32_t mantissa = (single.format.frac) + (1 << (FP32_SIGNIFICAND_BITS - 1));
    mantissa = single.format.sign ? -mantissa : mantissa;
    const int shift_num = (FP32_EXP_BIAS - FP6_EXP_BIAS + 1) - single.format.exp +
                          (FP32_SIGNIFICAND_BITS - FP6_SIGNIFICAND_BITS);
    mantissa = Right_Shift_Round(mantissa, shift_num, rd_mode);
    mantissa = single.format.sign ? -mantissa : mantissa;
    res = mantissa & 0x1f;
  } else if (single.format.exp == FP32_EXP_MASK && single.format.frac != 0) {
    // NaN
    res = 0x0;
  } else if (single.format.exp == FP32_EXP_MASK && single.format.frac == 0) {
    // INF
    res = FP6_MAXNORM;
  }
  res |= (single.format.sign << 5);
  return res;
}

static fp32 fp6_to_fp32(uint8_t single, bool is_e3m2)
{
    fp32 res;
    uint32_t ur = (uint32_t)single & 0x3FU;
    ur = (uint32_t)(ur << 26U);
    uint32_t sign = ur & 0x80000000U;
    uint32_t exponent = 0, mantissa = 0;
    uint32_t exp_gap_bit = 0;
    if (is_e3m2) {
      exp_gap_bit = 0x3E000000U;
      exponent = (uint32_t)(((ur & 0x70000000U) >> 5U) + exp_gap_bit);
      mantissa = (ur & 0xC000000U) >> 5U;
    } else {
      exp_gap_bit = 0x3F000000U;
      exponent = (uint32_t)(((ur & 0x60000000U) >> 6U) + exp_gap_bit);
      mantissa = (ur & 0x1C000000U) >> 6U;
    }

    if (exponent == exp_gap_bit) {
        // zero or denormal
        if (mantissa != 0U) {
            // normalize
            mantissa = mantissa << 1U;
            while ((mantissa & 0x800000U) == 0U) {
                mantissa = mantissa << 1U;
                exponent = exponent - 0x800000U;
            }
            // discard implicit leading bit
            mantissa &= 0x07FFFFFU;
        } else {  // Zero
            exponent = 0U;
        }
        ur = (sign | exponent) | mantissa;
    } else {
        ur = (sign | exponent) | mantissa;
    }
    res.bits = ur;
    return res;
}

#ifdef __cplusplus
}
#endif

#endif
