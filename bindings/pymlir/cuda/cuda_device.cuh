//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "cuda_helper.h"
#include <cuda_fp16.h>
#include <cstdint>

namespace tpu_mlir {
namespace cuda {

// -------------------------------------------------------------------------
// ------- bfloat16 definition

struct bfloat16 {
  uint16_t value;

  __device__ bfloat16() : value(0) {}
  __device__ bfloat16(uint16_t v) : value(v) {}
  __device__ bfloat16(float val, bool half_up = true) {
    if (half_up) {
      uint32_t u32_val = *((uint32_t *)(&val));
      uint32_t lsb = (u32_val >> 16) & 1;
      u32_val += (0x7fff + lsb);
      value = ((uint16_t *)(&u32_val))[1];
      /* HW behavior */
      // infinity set to max finite positive value
      value = ((value & 0x7f80) == 0x7f80) ? 0x7f7f : value;
    } else {
      value = ((uint16_t *)(&val))[1];
    }
  }

  __device__ operator float() const {
    unsigned int expanded = value << 16;
    return *reinterpret_cast<float *>(&expanded);
  }
};

__device__ float d_BF16(float data, bool round_up = true) {
  bfloat16 in_bf16(data, round_up);
  return static_cast<float>(in_bf16);
}

__device__ float d_RawBF16(uint16_t data) {
  bfloat16 in_bf16(data);
  return static_cast<float>(in_bf16);
}

__device__ uint16_t d_BF16Raw(float data, bool round_up = true) {
  bfloat16 in_bf16(data, round_up);
  return in_bf16.value;
}

// -------------------------------------------------------------------------
// ------- float16 definition

struct float16 {
  uint16_t value;

  __device__ float16() : value(0) {}
  __device__ float16(uint16_t v) : value(v) {}
  __device__ float16(float val, rounding_mode_t rmode = RD_HALF_TO_EVEN) {
    if (rmode == RD_HALF_TO_EVEN) {
      __half data = __float2half_rn(val);
      value = *reinterpret_cast<uint16_t *>(&data);
    }
  }

  __device__ operator float() const {
    __half data = *(__half *)(&value);
    return __half2float(data);
  }
};

__device__ float d_F16(float data, rounding_mode_t rmode = RD_HALF_TO_EVEN) {
  float16 in_f16(data, rmode);
  return static_cast<float>(in_f16);
}

__device__ float d_RawF16(uint16_t data) {
  float16 in_f16(data);
  return static_cast<float>(in_f16);
}

__device__ uint16_t d_F16Raw(float data,
                             rounding_mode_t rmode = RD_HALF_TO_EVEN) {
  float16 in_f16(data, rmode);
  return in_f16.value;
}

// -------------------------------------------------------------------------
// ----- type convert

template <typename T>
__device__ T d_f32ToInt(float data, rounding_mode_t rmode) {
  switch (rmode) {
  case RD_HALF_AWAY_FROM_ZERO:
    data = roundf(data);
    break;
  case RD_HALF_UP:
    data = floor(data + 0.5f);
    break;
  case RD_TOWARDS_ZERO:
    data = truncf(data);
    break;
  case RD_HALF_TO_EVEN:
    float fraction, integer;
    float abs_v = std::abs(data);
    fraction = std::modf(abs_v, &integer);
    int32_t i32_val = (int32_t)integer;
    if (fraction > 0.5) {
      i32_val = i32_val + 1;
    } else if (fraction == 0.5) {
      if (i32_val & 0x01) {
        i32_val = i32_val + 1;
      }
    }
    if (data < 0) {
      i32_val = -i32_val;
    }
    data = static_cast<float>(i32_val);
    break;
  }
  if (std::is_same<T, int8_t>::value) {
    data = fmaxf(-128.0f, fminf(127.0f, data));
  } else if (std::is_same<T, uint8_t>::value) {
    data = fmaxf(0.0f, fminf(255.0f, data));
  }
  return static_cast<T>(data);
}

__device__ int64_t Right_Shift_Round(int64_t src, int shift_num,
                                   rounding_mode_t round_mode=RD_HALF_TO_EVEN) {
  if (shift_num == 0)
    return src;
  if (shift_num > 63)
    shift_num = 63;
  int64_t val, res;
  val = src >> shift_num;
  res = val;
  int64_t lo_mask = (1ull << shift_num) - 1;
  int64_t mant = src & lo_mask;
  int64_t mant_0d5 = 1ull << (shift_num - 1);
  if (round_mode == RD_HALF_TO_EVEN) {
    if (mant == mant_0d5) {
      res = val + (val & 1);
    } else if (mant > mant_0d5) {
      res = val + 1;
    }
  } else if (round_mode == RD_HALF_AWAY_FROM_ZERO) {
    if (src >= 0 && mant >= mant_0d5) {
      res = val + 1;
    } else if (src < 0 && mant > mant_0d5) {
      res = val + 1;
    }
  } else if (round_mode == RD_TOWARDS_ZERO) {
    if (src < 0)
      res = val + (mant != 0);
  } else if (round_mode == RD_DOWN) {
    res = val;
  } else if (round_mode == RD_UP) {
    res = val + (mant != 0);
  } else if (round_mode == RD_HALF_UP) {
    if (mant >= mant_0d5)
      res = val + 1;
  } else if (round_mode == RD_HALF_DOWN) {
    if (mant > mant_0d5)
      res = val + 1;
  }
  return res;
}

__device__ uint8_t fp32_to_fp8(const float single, bool is_e5m2=false, bool saturate=true,
                           rounding_mode_t rd_mode=RD_HALF_TO_EVEN) {
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

  uint32_t int_value = __float_as_int(single);
  uint32_t exp = (int_value >> 23) & 0xff;
  bool sign = int_value >> 31;
  uint32_t frac = (int_value & (1ul<<23-1));

  if (exp > (127 - FP8_EXP_BIAS) && exp < 0xff) {
    const uint32_t mantissa = frac;
    const int32_t shift_num = 24 - FP8_SIGNIFICAND_BITS;
    uint32_t tmp = Right_Shift_Round(int_value, shift_num, rd_mode);
    if (rd_mode == RD_DOWN && sign == 1) {
      tmp += ((mantissa & FP8_INVALID_MASK) != 0);
    } else if (rd_mode == RD_UP && sign == 1) {
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
  } else if (exp > 0 &&
             exp <= (127 - FP8_EXP_BIAS)) {
    int32_t mantissa = (frac) + (1 << 23);
    mantissa = sign ? -mantissa : mantissa;
    const int shift_num = (127 - FP8_EXP_BIAS + 1) - exp +
                          (24 - FP8_SIGNIFICAND_BITS);
    mantissa = Right_Shift_Round(mantissa, shift_num, rd_mode);
    mantissa = sign ? -mantissa : mantissa;
    res = mantissa & 0x7f;
  } else if (exp == 0xff && frac != 0) {
    // Canonical NaN
    const uint32_t xbits = 0x7fffffff | (sign << 31);
    if (is_e5m2) {
      const uint32_t mantissa =
          (xbits >> (24 - FP8_SIGNIFICAND_BITS)) & FP8_MANTISSA_MASK;
      res = 0x7e | mantissa;
    } else {
      res = 0x7f;
    }
  } else if (exp == 0xff && frac == 0) {
    if (saturate) {
      res = FP8_MAXNORM;
    } else {
      // no Inf in E4M3 and use NaN, Inf in E5M2
      res = is_e5m2 ? 0x7c : 0x7f;
    }
  }
  res |= (sign << 7);
  return res;
}

__device__ float f8_to_fp32(uint8_t src, float scale, bool is_e5m2=false) {
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

  uint32_t value = (sign << 31u) | (exponent << 23u) | (mantissa);
  float f_value = __int_as_float(value);
  return f_value * scale;
}

__device__ void d_copyElement(void *src, int sidx, void *dst, int didx,
                              int tbytes) {
  switch (tbytes) {
  case 1:
    static_cast<uint8_t *>(dst)[didx] = static_cast<uint8_t *>(src)[sidx];
    break;
  case 2:
    static_cast<uint16_t *>(dst)[didx] = static_cast<uint16_t *>(src)[sidx];
    break;
  case 4:
    static_cast<uint32_t *>(dst)[didx] = static_cast<uint32_t *>(src)[sidx];
    break;
  default:
    break;
  }
}

__device__ void d_setZero(void *dst, int didx, int tbytes) {
  switch (tbytes) {
  case 1:
    static_cast<uint8_t *>(dst)[didx] = 0;
    break;
  case 2:
    static_cast<uint16_t *>(dst)[didx] = 0;
    break;
  case 4:
    static_cast<uint32_t *>(dst)[didx] = 0;
    break;
  default:
    break;
  }
}

// -------------------------------------------------------------------------
// ----- cv18xx

__device__ uint16_t d_lutSlopeBF16(uint16_t input, uint16_t *base_table,
                                   uint16_t *slope_table, float scale,
                                   float offset) {
  float in_rescale = d_BF16(d_RawBF16(input) - offset);
  in_rescale = d_BF16(in_rescale * scale);
  int in_i8 = d_f32ToInt<int8_t>(in_rescale, RD_TOWARDS_ZERO);
  // get delta x (x - x0)
  float delta_x = d_BF16(in_rescale - static_cast<float>(in_i8));
  // get slope
  auto slope = slope_table[in_i8 & 0xff];
  // base y0 = f(x0)
  auto base = base_table[in_i8 & 0xff];
  float slope_f32 = d_RawBF16(slope);
  float base_f32 = d_RawBF16(base);
  float out = d_BF16(delta_x * slope_f32) + base_f32;
  return d_BF16Raw(out);
}

__device__ uint16_t d_lutMantissaBF16(uint16_t input, uint16_t *exp_table,
                                      uint16_t *mantissa_table, bool is_log) {
  float val = d_RawBF16(input);
  int exponentIndex;
  if (val == 0) {
    exponentIndex = 0;
  } else if (val >= 0) {
    exponentIndex = floor(log2(val));
    exponentIndex += 62 + 1; // 62 means start with 2^-62, index from 1
  } else {
    exponentIndex = floor(log2(-1 * val));
    exponentIndex += 62 + 129; // 62 means start with 2^-62, index from 129
  }
  float exponent = d_RawBF16(exp_table[exponentIndex]);
  float mantissa = d_RawBF16(mantissa_table[input & 0xff]);
  float out = is_log ? (exponent + mantissa) : (exponent * mantissa);
  return d_BF16Raw(out);
}

} // namespace cuda
} // namespace tpu_mlir
