#include "limits.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

namespace tpu_mlir {

static long long Right_Shift_Round(long long src, int shift_num,
                                   RoundingMode round_mode) {
  if (shift_num == 0)
    return src;
  if (shift_num > 63)
    shift_num = 63;
  long long val, res;
  val = src >> shift_num;
  res = val;
  long long lo_mask = (1ull << shift_num) - 1;
  long long mant = src & lo_mask;
  long long mant_0d5 = 1ull << (shift_num - 1);
  if (round_mode == ROUNDING_HALF_TO_EVEN) {
    if (mant == mant_0d5) {
      res = val + (val & 1);
    } else if (mant > mant_0d5) {
      res = val + 1;
    }
  } else if (round_mode == ROUNDING_HALF_AWAY_FROM_ZERO) {
    if (src >= 0 && mant >= mant_0d5) {
      res = val + 1;
    } else if (src < 0 && mant > mant_0d5) {
      res = val + 1;
    }
  } else if (round_mode == ROUNDING_TOWARDS_ZERO) {
    if (src < 0)
      res = val + (mant != 0);
  } else if (round_mode == ROUNDING_DOWN) {
    res = val;
  } else if (round_mode == ROUNDING_UP) {
    res = val + (mant != 0);
  } else if (round_mode == ROUNDING_HALF_UP) {
    if (mant >= mant_0d5)
      res = val + 1;
  } else if (round_mode == ROUNDING_HALF_DOWN) {
    if (mant > mant_0d5)
      res = val + 1;
  }
  return res;
}

static uint8_t fp32_to_fp8(const fp32 single, bool is_e5m2, bool saturate,
                           RoundingMode rd_mode) {
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
    if (rd_mode == ROUNDING_DOWN && single.format.sign == 1) {
      tmp += ((mantissa & FP8_INVALID_MASK) != 0);
    } else if (rd_mode == ROUNDING_UP && single.format.sign == 1) {
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

uint8_t f32_to_f8e4m3(float src, bool satu) {
  fp32 tmp = {.fval = src};
  return fp32_to_fp8(tmp, false, satu, ROUNDING_HALF_TO_EVEN);
}

uint8_t f32_to_f8e5m2(float src, bool satu) {
  fp32 tmp = {.fval = src};
  // return fp32_to_fp8(tmp, true, satu, ROUNDING_HALF_TO_EVEN);
  return fp32_to_fp8(tmp, true, false, ROUNDING_HALF_TO_EVEN);
}

static fp32 fp8_to_fp32(uint8_t src, bool is_e5m2) {
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
      do {
        msb = mantissa & 0x400000u;
        mantissa <<= 1u; // normalize
        exponent--;
      } while (msb == 0u);
      mantissa &= 0x7fffffu;
    }
  } else {
    exponent += (127 - FP8_EXP_BIAS);
  }
  fp32 res = {.bits = 0};
  res.bits = (sign << 31u) | (exponent << 23u) | (mantissa);
  return res;
}

float f8e4m3_to_f32(uint8_t src) {
  fp32 tmp = fp8_to_fp32(src, false);
  return tmp.fval;
}

float f8e5m2_to_f32(uint8_t src) {
  fp32 tmp = fp8_to_fp32(src, true);
  return tmp.fval;
}

static fp16 fp8_to_fp16(uint8_t single, bool is_e5m2) {
  fp16 res;
  uint16_t ur = (uint16_t)single;
  ur = (uint16_t)(ur << 8U);
  uint16_t sign = ur & 0x8000U;

  if (is_e5m2) {
    if ((ur & 0x7FFFU) > 0x7C00U) {
      /* If NaN, return canonical NaN */
      ur = 0x7FFFU | sign;
    }
  } else {

    uint16_t exponent = (uint16_t)(((ur & 0x7800U) >> 1U) + 0x2000U);
    uint16_t mantissa = (ur & 0x0700U) >> 1U;
    uint8_t absx = 0x7FU & (uint8_t)single;

    if (absx == 0x7FU) // NaN
    {
      ur = 0x7FFFU | sign; // fp16 canonical NaN, discard sign
    } else if (exponent == 0x2000U) {
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
      } else { // Zero
        exponent = 0U;
      }
      ur = (sign | exponent) | mantissa;
    } else {
      ur = (sign | exponent) | mantissa;
    }
  }
  res.bits = ur;
  return res;
}

uint16_t f8e4m3_to_f16(uint8_t src) {
  fp16 tmp = fp8_to_fp16(src, false);
  return tmp.bits;
}

uint16_t f8e5m2_to_f16(uint8_t src) {
  fp16 tmp = fp8_to_fp16(src, true);
  return tmp.bits;
}

float get_f8e4m3_max() { return float(448.0); }

float get_f8e4m3_min() { return float(1.9531250E-03); }

float get_f8e5m2_max() { return float(57344.0); }

float get_f8e5m2_min() { return float(1.5258789E-05); }

float F8E4M3(float src, float step, bool satu) {
  return f8e4m3_to_f32(f32_to_f8e4m3(src / step, satu));
}

float F8E5M2(float src, float step, bool satu) {
  return f8e5m2_to_f32(f32_to_f8e5m2(src / step, satu));
}

void F8E4M3(const float *p_src, float *p_dst, int num, float step, bool satu) {
#pragma omp parallel for schedule(static, omp_schedule(num))
  for (int i = 0; i < num; i++) {
    p_dst[i] = F8E4M3(p_src[i], step, satu);
  }
}

void F8E5M2(const float *p_src, float *p_dst, int num, float step, bool satu) {
#pragma omp parallel for schedule(static, omp_schedule(num))
  for (int i = 0; i < num; i++) {
    p_dst[i] = F8E5M2(p_src[i], step, satu);
  }
}

} // namespace tpu_mlir
