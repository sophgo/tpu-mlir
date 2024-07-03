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
