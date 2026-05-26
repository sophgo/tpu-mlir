//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "cuda_device.cuh"
#include "cmath"
#include <algorithm>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace tpu_mlir {
namespace cuda {

__global__ void g_f32ScaleToInt8(float *input, void *output, float scale,
                                 int size, bool sign, rounding_mode_t rmode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float value = input[idx] * scale;
    if (sign) {
      static_cast<int8_t *>(output)[idx] = d_f32ToInt<int8_t>(value, rmode);
    } else {
      static_cast<uint8_t *>(output)[idx] = d_f32ToInt<uint8_t>(value, rmode);
    }
  }
}

__global__ void g_bf16ScaleToInt8(uint16_t *input, void *output, float scale,
                                  int size, bool sign, rounding_mode_t rmode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float value = d_BF16(d_RawBF16(input[idx]) * d_BF16(scale));
    if (sign) {
      static_cast<int8_t *>(output)[idx] = d_f32ToInt<int8_t>(value, rmode);
    } else {
      static_cast<uint8_t *>(output)[idx] = d_f32ToInt<uint8_t>(value, rmode);
    }
  }
}

__global__ void g_f16ScaleToInt8(uint16_t *input, void *output, float scale,
                                 int size, bool sign, rounding_mode_t rmode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float value = d_F16(d_RawF16(input[idx]) * d_F16(scale));
    if (sign) {
      static_cast<int8_t *>(output)[idx] = d_f32ToInt<int8_t>(value, rmode);
    } else {
      static_cast<uint8_t *>(output)[idx] = d_f32ToInt<uint8_t>(value, rmode);
    }
  }
}

__global__ void g_int8ScaleToF32(void *input, float *output, float scale,
                                 int size, bool sign) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    // Convert int8 to float32 and scale
    if (sign) {
      output[idx] = static_cast<float>(((int8_t *)input)[idx]) * scale;
    } else {
      output[idx] = static_cast<float>(((uint8_t *)input)[idx]) * scale;
    }
  }
}

__global__ void g_int8ScaleToBF16(void *input, uint16_t *output, float scale,
                                  int size, bool sign) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    // Convert int8 to bfloat16 and scale
    float value;
    if (sign) {
      value = static_cast<float>(((int8_t *)input)[idx]) * d_BF16(scale);
    } else {
      value = static_cast<float>(((uint8_t *)input)[idx]) * d_BF16(scale);
    }
    output[idx] = d_BF16Raw(value);
  }
}

__global__ void g_int8ScaleToF16(void *input, uint16_t *output, float scale,
                                 int size, bool sign) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    // Convert int8 to bfloat16 and scale
    float value;
    if (sign) {
      value = static_cast<float>(((int8_t *)input)[idx]) * d_F16(scale);
    } else {
      value = static_cast<float>(((uint8_t *)input)[idx]) * d_F16(scale);
    }
    output[idx] = d_F16Raw(value);
  }
}

__global__ void g_int16ScaleToF32(void *input, float *output, float scale,
                                 int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    // Convert int16 to f32 and scale
    float value;
    value = static_cast<float>(((int16_t *)input)[idx]) * scale;
    output[idx] = value;
  }
}

__global__ void g_int16ScaleToBF16(void *input, uint16_t *output, float scale,
                                 int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    // Convert int16 to f32 and scale
    float value;
    value = static_cast<float>(((int16_t *)input)[idx]) * d_BF16(scale);
    output[idx] = d_BF16Raw(value);
  }
}

__global__ void g_int16ScaleToF16(void *input, uint16_t *output, float scale,
                                 int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    // Convert int16 to f32 and scale
    float value;
    value = static_cast<float>(((int16_t *)input)[idx]) * d_F16(scale);
    output[idx] = d_F16Raw(value);
  }
}

template <typename T0, typename T1, typename T2>
__global__ void g_mulInt8(T0 *a, T1 *b, T2 *out, int32_t multiplier,
                          int32_t rshift, int size, bool qdm, bool relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int32_t value;
    if (qdm) {
      int64_t data =
          static_cast<int64_t>(a[idx]) * static_cast<int64_t>(b[idx]);
      data = data * static_cast<int64_t>(multiplier);
      data = (data + (1ll << 30)) >> 31;
      value = static_cast<int32_t>(data);
      // half away from zero
      int32_t offset = 1 << (rshift - 1);
      bool negative = value < 0;
      if (negative) {
        value = -value;
      }
      value = (value + offset) >> rshift;
      if (negative) {
        value = -value;
      }
    } else {
      value = static_cast<int32_t>(a[idx]) * static_cast<int32_t>(b[idx]) *
              multiplier;
      // half up
      value = (value + (1 << (rshift - 1))) >> rshift;
    }
    if (std::is_same<T2, int8_t>::value) {
      int32_t min_ = relu ? 0 : -128;
      value = max(min_, min(127, value));
      ((int8_t *)out)[idx] = static_cast<int8_t>(value);
    } else {
      value = max(0, min(255, value));
      ((uint8_t *)out)[idx] = static_cast<uint8_t>(value);
    }
  }
}

template <typename T0, typename T1, typename T2>
__global__ void g_mulInt8(T0 *a, T1 *b, T2 *out, int n0, int c0, int h0, int w0,
                          int n1, int c1, int h1, int w1, int n2, int c2,
                          int h2, int w2, int multiplier, int rshift, bool qdm,
                          bool relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (n2 * c2 * h2 * w2)) {
    int idx_n = idx / (c2 * h2 * w2);
    int idx_c = idx % (c2 * h2 * w2) / (h2 * w2);
    int idx_h = idx % (h2 * w2) / w2;
    int idx_w = idx % w2;
    int idx_out = ((idx_n * c2 + idx_c) * h2 + idx_h) * w2 + idx_w;
    int idx_n0 = idx_n >= n0 ? 0 : idx_n;
    int idx_c0 = idx_c >= c0 ? 0 : idx_c;
    int idx_h0 = idx_h >= h0 ? 0 : idx_h;
    int idx_w0 = idx_w >= w0 ? 0 : idx_w;
    int idx_a = ((idx_n0 * c0 + idx_c0) * h0 + idx_h0) * w0 + idx_w0;
    int idx_n1 = idx_n >= n1 ? 0 : idx_n;
    int idx_c1 = idx_c >= c1 ? 0 : idx_c;
    int idx_h1 = idx_h >= h1 ? 0 : idx_h;
    int idx_w1 = idx_w >= w1 ? 0 : idx_w;
    int idx_b = ((idx_n1 * c1 + idx_c1) * h1 + idx_h1) * w1 + idx_w1;
    int32_t value;
    if (qdm) {
      int64_t data =
          static_cast<int64_t>(a[idx_a]) * static_cast<int64_t>(b[idx_b]);
      data = data * static_cast<int64_t>(multiplier);
      data = (data + (1ll << 30)) >> 31;
      value = static_cast<int32_t>(data);
      // half away from zero
      int32_t offset = 1 << (rshift - 1);
      bool negative = value < 0;
      if (negative) {
        value = -value;
      }
      value = (value + offset) >> rshift;
      if (negative) {
        value = -value;
      }
    } else {
      value = static_cast<int32_t>(a[idx_a]) * static_cast<int32_t>(b[idx_b]) *
              multiplier;
      // half up
      value = (value + (1 << (rshift - 1))) >> rshift;
    }
    if (std::is_same<T2, int8_t>::value) {
      int32_t min_ = relu ? 0 : -128;
      value = max(min_, min(127, value));
      ((int8_t *)out)[idx_out] = static_cast<int8_t>(value);
    } else {
      value = max(0, min(255, value));
      ((uint8_t *)out)[idx_out] = static_cast<uint8_t>(value);
    }
  }
}

template <typename T0, typename T1, typename T2>
__global__ void g_add4DInt8(T0 *a, T1 *b, T2 *out, int32_t mul0, int32_t mul1,
                            int shift0, int shift1, bool relu, int n0, int c0,
                            int h0, int w0, int n1, int c1, int h1, int w1,
                            int on, int oc, int oh, int ow) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_n = dst_idx / (oc * oh * ow);
  int idx_c = dst_idx % (oc * oh * ow) / (oh * ow);
  int idx_h = dst_idx % (oh * ow) / ow;
  int idx_w = dst_idx % ow;
  if (idx_w < ow && idx_h < oh && idx_c < oc && idx_n < on) {
    int idx_n0 = idx_n % n0;
    int idx_c0 = idx_c % c0;
    int idx_h0 = idx_h % h0;
    int idx_w0 = idx_w % w0;
    int idx_0 = ((idx_n0 * c0 + idx_c0) * h0 + idx_h0) * w0 + idx_w0;
    int idx_n1 = idx_n % n1;
    int idx_c1 = idx_c % c1;
    int idx_h1 = idx_h % h1;
    int idx_w1 = idx_w % w1;
    int idx_1 = ((idx_n1 * c1 + idx_c1) * h1 + idx_h1) * w1 + idx_w1;
    int32_t a_data = static_cast<int32_t>(a[idx_0]) * mul0;
    a_data = (a_data + (1 << (shift0 - 1))) >> shift0;
    int32_t b_data = static_cast<int32_t>(b[idx_1]) * mul1;
    b_data = (b_data + (1 << (shift1 - 1))) >> shift1;
    a_data = a_data + b_data;
    if (std::is_same<T2, int8_t>::value) {
      int32_t min_ = relu ? 0 : -128;
      a_data = max(min_, min(127, a_data));
      out[dst_idx] = static_cast<int8_t>(a_data);
    } else {
      a_data = max(0, min(255, a_data));
      out[dst_idx] = static_cast<uint8_t>(a_data);
    }
  }
}

template <typename T0, typename T1, typename T2>
__global__ void g_add4DF32(T0 *a, float scale0, T1 *b, float scale1, T2 *out, bool relu, int n0, int c0,
                            int h0, int w0, int n1, int c1, int h1, int w1,
                            int on, int oc, int oh, int ow) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_n = dst_idx / (oc * oh * ow);
  int idx_c = dst_idx % (oc * oh * ow) / (oh * ow);
  int idx_h = dst_idx % (oh * ow) / ow;
  int idx_w = dst_idx % ow;
  if (idx_w < ow && idx_h < oh && idx_c < oc && idx_n < on) {
    int idx_n0 = idx_n % n0;
    int idx_c0 = idx_c % c0;
    int idx_h0 = idx_h % h0;
    int idx_w0 = idx_w % w0;
    int idx_0 = ((idx_n0 * c0 + idx_c0) * h0 + idx_h0) * w0 + idx_w0;
    int idx_n1 = idx_n % n1;
    int idx_c1 = idx_c % c1;
    int idx_h1 = idx_h % h1;
    int idx_w1 = idx_w % w1;
    int idx_1 = ((idx_n1 * c1 + idx_c1) * h1 + idx_h1) * w1 + idx_w1;
    float a_data = a[idx_0] * scale0;
    float b_data = b[idx_1] * scale1;
    a_data = a_data + b_data;
    if (relu)
      a_data = max(0.0, a_data);
    out[dst_idx] = a_data;
  }
}

__global__ void g_add4DInt32(int32_t *a, int32_t *b, int32_t *out,
                            int n0, int c0, int h0, int w0,
                            int n1, int c1, int h1, int w1,
                            int on, int oc, int oh, int ow) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_n = dst_idx / (oc * oh * ow);
  int idx_c = dst_idx % (oc * oh * ow) / (oh * ow);
  int idx_h = dst_idx % (oh * ow) / ow;
  int idx_w = dst_idx % ow;
  if (idx_w < ow && idx_h < oh && idx_c < oc && idx_n < on) {
    int idx_n0 = idx_n % n0;
    int idx_c0 = idx_c % c0;
    int idx_h0 = idx_h % h0;
    int idx_w0 = idx_w % w0;
    int idx_0 = ((idx_n0 * c0 + idx_c0) * h0 + idx_h0) * w0 + idx_w0;
    int idx_n1 = idx_n % n1;
    int idx_c1 = idx_c % c1;
    int idx_h1 = idx_h % h1;
    int idx_w1 = idx_w % w1;
    int idx_1 = ((idx_n1 * c1 + idx_c1) * h1 + idx_h1) * w1 + idx_w1;
    int32_t a_data = a[idx_0];
    int32_t b_data = b[idx_1];
    a_data = a_data + b_data;
    out[dst_idx] = a_data;
  }
}

template <typename T0, typename T1, typename T2>
__global__ void g_sub4DF32(T0 *a, T1 *b, T2 *out, bool relu, bool reverse, int n0, int c0,
                            int h0, int w0, int n1, int c1, int h1, int w1,
                            int on, int oc, int oh, int ow) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_n = dst_idx / (oc * oh * ow);
  int idx_c = dst_idx % (oc * oh * ow) / (oh * ow);
  int idx_h = dst_idx % (oh * ow) / ow;
  int idx_w = dst_idx % ow;
  if (idx_w < ow && idx_h < oh && idx_c < oc && idx_n < on) {
    int idx_n0 = idx_n % n0;
    int idx_c0 = idx_c % c0;
    int idx_h0 = idx_h % h0;
    int idx_w0 = idx_w % w0;
    int idx_0 = ((idx_n0 * c0 + idx_c0) * h0 + idx_h0) * w0 + idx_w0;
    int idx_n1 = idx_n % n1;
    int idx_c1 = idx_c % c1;
    int idx_h1 = idx_h % h1;
    int idx_w1 = idx_w % w1;
    int idx_1 = ((idx_n1 * c1 + idx_c1) * h1 + idx_h1) * w1 + idx_w1;
    float a_data = a[idx_0];
    float b_data = b[idx_1];
    if (reverse)
      a_data = b_data - a_data;
    else
      a_data = a_data - b_data;
    if (relu)
      a_data = max(0.0, a_data);
    out[dst_idx] = a_data;
  }
}

template <typename T0, typename T1, typename T2>
__global__ void g_sub4DInt8(T0 *a, int mul0, int shift0, T1 *b, int mul1, int shift1, T2 *out, bool relu, bool reverse, int n0, int c0,
                            int h0, int w0, int n1, int c1, int h1, int w1,
                            int on, int oc, int oh, int ow) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_n = dst_idx / (oc * oh * ow);
  int idx_c = dst_idx % (oc * oh * ow) / (oh * ow);
  int idx_h = dst_idx % (oh * ow) / ow;
  int idx_w = dst_idx % ow;
  if (idx_w < ow && idx_h < oh && idx_c < oc && idx_n < on) {
    int idx_n0 = idx_n % n0;
    int idx_c0 = idx_c % c0;
    int idx_h0 = idx_h % h0;
    int idx_w0 = idx_w % w0;
    int idx_0 = ((idx_n0 * c0 + idx_c0) * h0 + idx_h0) * w0 + idx_w0;
    int idx_n1 = idx_n % n1;
    int idx_c1 = idx_c % c1;
    int idx_h1 = idx_h % h1;
    int idx_w1 = idx_w % w1;
    int idx_1 = ((idx_n1 * c1 + idx_c1) * h1 + idx_h1) * w1 + idx_w1;
    int a_data = a[idx_0];
    int b_data = b[idx_1];
    a_data = (a_data*mul0)>>shift0;
    b_data = (b_data*mul1)>>shift1;
    if (reverse)
      a_data = b_data - a_data;
    else
      a_data = a_data - b_data;
    if (relu)
      a_data = max(0, a_data);
    a_data = max(-128, a_data);
    a_data = min(127, a_data);
    out[dst_idx] = (int8_t)a_data;
  }
}

template <typename T0, typename T1, typename T2>
__global__ void g_mulConst4DF32(T0 *a, T1 b, T2 *out, bool relu, int n0, int c0,
                            int h0, int w0) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_n = dst_idx / (c0 * h0 * w0);
  int idx_c = dst_idx % (c0 * h0 * w0) / (h0 * w0);
  int idx_h = dst_idx % (h0 * w0) / w0;
  int idx_w = dst_idx % w0;
  if (idx_w < w0 && idx_h < h0 && idx_c < c0 && idx_n < n0) {
    float a_data = a[dst_idx];
    a_data = a_data * b;
    if (relu)
      a_data = max(0.0, a_data);
    out[dst_idx] = a_data;
  }
}

__global__ void g_subConst4DF32(float *input, float const_v, float*output,
      bool do_relu, bool reverse, int n, int c, int h, int w){
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_n = dst_idx / (c * h * w);
  int idx_c = dst_idx % (c * h * w) / (h * w);
  int idx_h = dst_idx % (h * w) / w;
  int idx_w = dst_idx % w;
  if (idx_w < w && idx_h < h && idx_c < c && idx_n < n) {
    float a_data = input[dst_idx];
    if (reverse)
      a_data = const_v - a_data;
    else
      a_data = a_data - const_v;
    if (do_relu)
      a_data = max(0.0, a_data);
    output[dst_idx] = a_data;
  }
}

template <typename T0>
__global__ void g_subConst4DI8(T0 *input, int const_v, int8_t *output,
      bool do_relu, bool reverse, int multi, int shift, int n, int c, int h, int w){
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_n = dst_idx / (c * h * w);
  int idx_c = dst_idx % (c * h * w) / (h * w);
  int idx_h = dst_idx % (h * w) / w;
  int idx_w = dst_idx % w;
  if (idx_w < w && idx_h < h && idx_c < c && idx_n < n) {
    int a_data = (int)input[dst_idx];
    if (reverse)
      a_data = const_v - a_data*multi;
    else
      a_data = a_data*multi - const_v;
    int val = a_data >> shift;
    // using rounding half up
    if (shift > 0 ) {
      int mant = a_data & ((1ul << shift) - 1);
      if (mant >= (1ul << (shift-1)))
        val += 1;
    }
    if (do_relu)
      a_data = max(0, val);
    else
      a_data = val;
    a_data = max(-128, a_data);
    a_data = min(127, a_data);
    output[dst_idx] = (int8_t)a_data;
  }
}

template <typename T0, typename T1, typename T2>
__global__ void g_mul4DF32(T0 *a, T1 *b, T2 *out, bool relu, int n0, int c0,
                            int h0, int w0, int n1, int c1, int h1, int w1,
                            int on, int oc, int oh, int ow) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_n = dst_idx / (oc * oh * ow);
  int idx_c = dst_idx % (oc * oh * ow) / (oh * ow);
  int idx_h = dst_idx % (oh * ow) / ow;
  int idx_w = dst_idx % ow;
  if (idx_w < ow && idx_h < oh && idx_c < oc && idx_n < on) {
    int idx_n0 = idx_n % n0;
    int idx_c0 = idx_c % c0;
    int idx_h0 = idx_h % h0;
    int idx_w0 = idx_w % w0;
    int idx_0 = ((idx_n0 * c0 + idx_c0) * h0 + idx_h0) * w0 + idx_w0;
    int idx_n1 = idx_n % n1;
    int idx_c1 = idx_c % c1;
    int idx_h1 = idx_h % h1;
    int idx_w1 = idx_w % w1;
    int idx_1 = ((idx_n1 * c1 + idx_c1) * h1 + idx_h1) * w1 + idx_w1;
    float a_data = a[idx_0];
    float b_data = b[idx_1];
    a_data = a_data * b_data;
    if (relu)
      a_data = max(0.0, a_data);
    out[dst_idx] = a_data;
  }
}

template <typename T0, typename T1, typename T2>
__global__ void g_div4DF32(T0 *a, T1 *b, T2 *out, bool relu, int n0, int c0,
                            int h0, int w0, int n1, int c1, int h1, int w1,
                            int on, int oc, int oh, int ow) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_n = dst_idx / (oc * oh * ow);
  int idx_c = dst_idx % (oc * oh * ow) / (oh * ow);
  int idx_h = dst_idx % (oh * ow) / ow;
  int idx_w = dst_idx % ow;
  if (idx_w < ow && idx_h < oh && idx_c < oc && idx_n < on) {
    int idx_n0 = idx_n % n0;
    int idx_c0 = idx_c % c0;
    int idx_h0 = idx_h % h0;
    int idx_w0 = idx_w % w0;
    int idx_0 = ((idx_n0 * c0 + idx_c0) * h0 + idx_h0) * w0 + idx_w0;
    int idx_n1 = idx_n % n1;
    int idx_c1 = idx_c % c1;
    int idx_h1 = idx_h % h1;
    int idx_w1 = idx_w % w1;
    int idx_1 = ((idx_n1 * c1 + idx_c1) * h1 + idx_h1) * w1 + idx_w1;
    float a_data = a[idx_0];
    float b_data = b[idx_1];
    if (b_data == 0.0f) b_data = 1e-8f;
    a_data = a_data / b_data;
    if (relu)
      a_data = max(0.0f, a_data);
    out[dst_idx] = a_data;
  }
}

__global__ void g_clip4DF32(float *input, float *output, float min_val, float max_val,
                            int n, int c, int h, int w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n * c * h * w) {
    float val = input[idx];
    if (val < min_val) val = min_val;
    if (val > max_val) val = max_val;
    output[idx] = val;
  }
}

__global__ void g_addConst4DF32(float *input, float *output, float const_val,
                                bool do_relu, int n, int c, int h, int w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n * c * h * w) {
    float val = input[idx] + const_val;
    if (do_relu && val < 0.0f) val = 0.0f;
    output[idx] = val;
  }
}

template <typename T0, typename T1, typename T2, typename T3>
__global__ void g_scale4DF32(T0 *a, T1 *s, T2 *b, T3 *out, bool relu, int n0, int c0,
                            int h0, int w0, int n1, int c1, int h1, int w1,int n2, int c2, int h2, int w2,
                            int on, int oc, int oh, int ow) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_n = dst_idx / (oc * oh * ow);
  int idx_c = dst_idx % (oc * oh * ow) / (oh * ow);
  int idx_h = dst_idx % (oh * ow) / ow;
  int idx_w = dst_idx % ow;
  if (idx_w < ow && idx_h < oh && idx_c < oc && idx_n < on) {
    int idx_n0 = idx_n % n0;
    int idx_c0 = idx_c % c0;
    int idx_h0 = idx_h % h0;
    int idx_w0 = idx_w % w0;
    int idx_0 = ((idx_n0 * c0 + idx_c0) * h0 + idx_h0) * w0 + idx_w0;
    int idx_1 = idx_c0;
    float a_data = a[idx_0];
    float s_data = s[idx_1];
    float b_data = b[idx_1];
    a_data = a_data * s_data + b_data;
    if (relu)
      a_data = max(0.0, a_data);
    out[dst_idx] = a_data;
  }
}

template <typename T> __global__ void g_neg(T *input, T *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = -input[idx];
  }
}

__global__ void g_pad4D(void *input, void *output, int n, int c, int h, int w,
                        int pad_h_t, int pad_h_b, int pad_w_l, int pad_w_r,
                        int tbytes) {
  int oh = h + pad_h_t + pad_h_b;
  int ow = w + pad_w_l + pad_w_r;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (n * c * oh * ow)) {
    int idx_n = idx / (c * oh * ow);
    int idx_c = idx % (c * oh * ow) / (oh * ow);
    int idx_h = idx % (oh * ow) / ow;
    int idx_w = idx % ow;
    int out_idx = ((idx_n * c + idx_c) * oh + idx_h) * ow + idx_w;
    if (idx_h >= pad_h_t && idx_h < (pad_h_t + h) && idx_w >= pad_w_l &&
        idx_w < (pad_w_l + w)) {
      int idx_in_h = idx_h - pad_h_t;
      int idx_in_w = idx_w - pad_w_l;
      int in_idx = ((idx_n * c + idx_c) * h + idx_in_h) * w + idx_in_w;
      d_copyElement(input, in_idx, output, out_idx, tbytes);
    } else {
      d_setZero(output, out_idx, tbytes);
    }
  }
}

__global__ void g_permute6D(void *input, void *output, int n, int c, int d, int h,
                            int w, int d1, int o0, int o1, int o2, int o3, int o4, int o5, int tbytes) {
  int oldIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (oldIdx < n * c * d * h * w * d1) {
    int dims[6] = {n, c, d, h, w, d1};
    int newDims[6] = {dims[o0], dims[o1], dims[o2], dims[o3], dims[o4], dims[o5]};
    int ind[6];
    ind[0] = oldIdx / (c * d * h * w * d1);             // n index
    ind[1] = (oldIdx % (c * d * h * w * d1)) / (d * h * w * d1); // c index
    ind[2] = (oldIdx % (d* h * w * d1)) / (h * w * d1);           // d index
    ind[3] = oldIdx % (h * w * d1) / ( w * d1);                  // h index
    ind[4] = oldIdx % (w * d1) / d1;                             // w index
    ind[5] = oldIdx % d1;                                       // d1 index
    int newInd[6] = {ind[o0], ind[o1], ind[o2], ind[o3], ind[o4], ind[o5]};
    int newIdx =
        ((((newInd[0] * newDims[1] + newInd[1]) * newDims[2] + newInd[2]) *
            newDims[3] + newInd[3]) * newDims[4] + newInd[4]) * newDims[5] + newInd[5];
    d_copyElement(input, oldIdx, output, newIdx, tbytes);
  }
}

__global__ void g_slice6D(void *src, void *dst, int n, int c, int d, int h, int w, int d1,
                          int off0, int off1, int off2, int off3, int off4, int off5,
                          int s0, int s1, int s2, int s3, int s4, int s5,
                          int on, int oc, int od, int oh,
                          int ow, int od1, int tbytes) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_n = dst_idx / (oc * od * oh * ow * od1);
  int idx_c = dst_idx % (oc * od * oh * ow * od1) / (od * oh * ow * od1);
  int idx_d = dst_idx % (od * oh * ow * od1) / (oh * ow * od1);
  int idx_h = dst_idx % (oh * ow * od1 ) / (ow * od1);
  int idx_w = dst_idx % (ow * od1) / od1;
  int idx_d1 = dst_idx % od1;
  if (idx_w < ow && idx_h < oh && idx_c < oc && idx_n < on && idx_d < od && idx_d1 < od1) {
    idx_n = off0 + idx_n * s0;
    idx_c = off1 + idx_c * s1;
    idx_d = off2 + idx_d * s2;
    idx_h = off3 + idx_h * s3;
    idx_w = off4 + idx_w * s4;
    idx_d1 = off5 + idx_d1 * s5;

    if (idx_n < n && idx_c < c && idx_h < h && idx_w < w && idx_d < d && idx_d1 < od1) {
      int src_idx = ((((idx_n * c + idx_c) * d + idx_d) * h + idx_h) * w  + idx_w) * d1 + idx_d1;
      d_copyElement(src, src_idx, dst, dst_idx, tbytes);
    }
  }
}

__global__ void g_swapDimInner6D(void *src, void *dst, int outter, int shape, int offset, int inner, int tbytes){
  int src_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (src_idx < outter * shape * inner) {
    int outer_idx = src_idx / (inner*shape);
    int axis_idx = src_idx % (inner*shape) / inner;
    int inner_idx = src_idx % inner;
    int new_axis = (axis_idx-offset+shape)%shape;
    int dst_idx = outer_idx*(shape*inner) + new_axis*inner + inner_idx;
    d_copyElement(src, src_idx, dst, dst_idx, tbytes);
  }
}

__global__ void g_tile4D(void *src, void *dst, int n, int c, int h, int w,
                         int on, int oc, int oh, int ow, int tbytes) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_n = dst_idx / (oc * oh * ow);
  int idx_c = dst_idx % (oc * oh * ow) / (oh * ow);
  int idx_h = dst_idx % (oh * ow) / ow;
  int idx_w = dst_idx % ow;
  if (idx_w < ow && idx_h < oh && idx_c < oc && idx_n < on) {
    int in = idx_n % n;
    int ic = idx_c % c;
    int ih = idx_h % h;
    int iw = idx_w % w;
    int src_idx = ((in * c + ic) * h + ih) * w + iw;
    d_copyElement(src, src_idx, dst, dst_idx, tbytes);
  }
}

__global__ void g_GELU(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    float value = 0.5*input[i]*(1.0+erff(input[i]/sqrt(2.0)));
    output[i] = value;
  }
}

__global__ void g_ELU(float* input, float *output, float alpha, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    float val = input[i];
    output[i] = val > 0.0f ? val : alpha * (expf(val) - 1.0f);
  }
}

__global__ void g_ERF(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    output[i] = erff(input[i]);
  }
}

__global__ void g_EXP(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    output[i] = expf(input[i]);
  }
}

__global__ void g_copyAxis(void *src, void *dst, int outer_dim, int axis_dim,
                           int inner_dim, int offset, int num, int tbytes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer_dim * num * inner_dim;
  if (idx < total) {
    int out_idx = idx / (num * inner_dim);
    int axis_idx = (idx % (num * inner_dim)) / inner_dim;
    int inner_idx = idx % inner_dim;
    int dstIdx = out_idx * axis_dim * inner_dim +
                 (axis_idx + offset) * inner_dim + inner_idx;
    d_copyElement(src, idx, dst, dstIdx, tbytes);
  }
}

__global__ void g_mmF32(float *A, float *B, float *C, bool right_transpose, int m, int k, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_m = idx / n;
  int idx_n = idx % n;
  if (idx_m < m && idx_n < n) {
    float sum = 0.0;
    if (right_transpose) {
      for (int i = 0; i < k; i++) {
        sum += A[idx_m * k + i] * B[idx_n * k + i];
      }
    } else {
      for (int i = 0; i < k; i++) {
        sum += A[idx_m * k + i] * B[i * n + idx_n];
      }
    }
    C[idx_m * n + idx_n] = sum;
  }
}

template <typename T0, typename T1>
__global__ void g_mmInt8(T0 *A, T1 *B, int32_t *C, bool right_transpose, int m, int k, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_m = idx / n;
  int idx_n = idx % n;
  if (idx_m < m && idx_n < n) {
    int32_t sum = 0;
    if (right_transpose) {
      for (int i = 0; i < k; i++) {
        sum += ((int32_t)A[idx_m * k + i]) * ((int32_t)B[idx_n * k + i]);
      }
    } else {
      for (int i = 0; i < k; i++) {
        sum += A[idx_m * k + i] * B[i * n + idx_n];
      }
    }
    C[idx_m * n + idx_n] = sum;
  }
}

__global__ void g_requantInt8Perchannel(int32_t *input, void *output,
                                        int32_t *multipliers, int32_t *shifts,
                                        int n, int c, int h, int w,
                                        bool out_sign, bool qdm, bool relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (n * c * h * w)) {
    int idx_c = idx % (c * h * w) / (h * w);
    int32_t value;
    if (qdm == false) {
      // half up
      int64_t data = static_cast<int64_t>(input[idx]) *
                     static_cast<int64_t>(multipliers[idx_c]);
      int64_t round = (int64_t)(1ll << (shifts[idx_c] - 1));
      data = (data + round) >> shifts[idx_c];
      value = static_cast<int32_t>(data);
    } else {

      int64_t data = static_cast<int64_t>(input[idx]) *
                     static_cast<int64_t>(multipliers[idx_c]);
      data = (data + (1ll << 30)) >> 31;
      value = static_cast<int32_t>(data);
      // half away from zero
      int32_t offset = 1 << (shifts[idx_c] - 1);
      bool negative = value < 0;
      if (negative) {
        value = -value;
      }
      value = (value + offset) >> shifts[idx_c];
      if (negative) {
        value = -value;
      }
    }
    if (out_sign) {
      int32_t min_ = relu ? 0 : -128;
      value = max(min_, min(127, value));
      ((int8_t *)output)[idx] = static_cast<int8_t>(value);
    } else {
      value = max(0, min(255, value));
      ((uint8_t *)output)[idx] = static_cast<uint8_t>(value);
    }
  }
}

__global__ void g_requantInt8(int32_t *input, void *output, int32_t multiplier,
                              int32_t shift, int num, bool out_sign, bool qdm,
                              bool relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    int32_t value;
    if (qdm == false) {
      // half up
      int64_t data =
          static_cast<int64_t>(input[idx]) * static_cast<int64_t>(multiplier);
      int64_t round = 1ll << (shift - 1);
      data = (data + round) >> shift;
      value = static_cast<int32_t>(data);
    } else {
      int64_t data =
          static_cast<int64_t>(input[idx]) * static_cast<int64_t>(multiplier);
      data = (data + (1ll << 30)) >> 31;
      value = static_cast<int32_t>(data);
      // half away from zero
      int32_t offset = 1 << (shift - 1);
      bool negative = value < 0;
      if (negative) {
        value = -value;
      }
      value = (value + offset) >> shift;
      if (negative) {
        value = -value;
      }
    }
    if (out_sign) {
      int32_t min_ = relu ? 0 : -128;
      value = max(min_, min(127, value));
      ((int8_t *)output)[idx] = static_cast<int8_t>(value);
    } else {
      value = max(0, min(255, value));
      ((uint8_t *)output)[idx] = static_cast<uint8_t>(value);
    }
  }
}

__global__ void g_requantInt16(int32_t *input, void *output, int32_t multiplier,
                              int32_t shift, int num, bool relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    int32_t value;
    // half up
    int64_t data =
        static_cast<int64_t>(input[idx]) * static_cast<int64_t>(multiplier);
    int64_t round = 1ll << (shift - 1);
    data = (data + round) >> shift;
    value = static_cast<int32_t>(data);
    int32_t min_ = relu ? 0 : -32768;
    value = max(min_, min(32767, value));
    ((int16_t *)output)[idx] = static_cast<int16_t>(value);
  }
}

__global__ void g_requantInt16Perchannel(int32_t *input, void *output,
                                        int32_t *multipliers, int32_t *shifts,
                                        int n, int c, int h, int w, bool relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (n * c * h * w)) {
    int idx_c = idx % (c * h * w) / (h * w);
    int32_t value;
    // half up
    int64_t data = static_cast<int64_t>(input[idx]) *
                    static_cast<int64_t>(multipliers[idx_c]);
    int64_t round = (int64_t)(1ll << (shifts[idx_c] - 1));
    data = (data + round) >> shifts[idx_c];
    value = static_cast<int32_t>(data);
    int32_t min_ = relu ? 0 : -32768;
    value = max(min_, min(32767, value));
    ((int16_t *)output)[idx] = static_cast<int16_t>(value);
  }
}

__global__ void g_requantF8Perchannel(float *input, uint8_t *output,
                                        float *scales, int n, int c, int h, int w, bool relu, bool conv=true) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (n * c * h * w)) {
    int idx_c = idx % (c * h * w) / (h * w);
    if (!conv)
      idx_c = idx % w;
    // half up
    float value = static_cast<float>(input[idx]) *
                    static_cast<float>(scales[idx_c]);
    if (relu){
      value = fmaxf(0.0f, value);
    }
    uint8_t f8_value = fp32_to_fp8(value);
    output[idx] = f8_value;
  }
}

__global__ void g_requantF8(float *input, uint8_t *output,
                                        float scale, int n, int c, int h, int w, bool relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (n * c * h * w)) {
    // half up
    float value = static_cast<float>(input[idx]) * scale;
    if (relu){
      value = fmaxf(0.0f, value);
    }
    uint8_t f8_value = fp32_to_fp8(value);
    output[idx] = f8_value;
  }
}

template <typename T>
__global__ void g_mulShift(T *input, T *output, int multiplier, int shift,
                           int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int32_t value = static_cast<int32_t>(input[idx]) * multiplier;
    value = (value + (1 << (shift - 1))) >> shift; // half up
    if (std::is_same<T, int8_t>::value) {
      value = fmaxf(-128.0f, fminf(127.0f, value));
    } else if (std::is_same<T, uint8_t>::value) {
      value = fmaxf(0.0f, fminf(255.0f, value));
    }
    output[idx] = static_cast<T>(value);
  }
}

template <typename T>
__global__ void g_mulShiftFloat(float *input, T* output,
                                float multiplier, float shift, int size, rounding_mode_t rmode){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float value = static_cast<float>(input[idx]) * multiplier;
    value = value + shift;
    int i_value = 0;
    if (rmode == RD_HALF_TO_EVEN) {
      i_value = d_f32ToInt<int32_t>(value, RD_HALF_TO_EVEN); /// not implemented half to even
    } else if (rmode == RD_HALF_AWAY_FROM_ZERO) {
      i_value = round(value);
    }
    if (std::is_same<T, int8_t>::value) {
      i_value = max(-128, min(127, i_value));
    } else if (std::is_same<T, uint8_t>::value) {
      i_value = max(0, min(255, i_value));
    }
    output[idx] = static_cast<T>(i_value);
  }
}

template <typename T>
__global__ void g_intToF32(T *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = static_cast<float>(input[idx]);
  }
}

template <typename T>
__global__ void g_f32ToInt(float *input, T *output, int size,
                           rounding_mode_t rmode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = d_f32ToInt<T>(input[idx], rmode);
  }
}

__global__ void g_f32ToBF16(float *input, uint16_t *output, int size,
                            rounding_mode_t rmode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = d_BF16Raw(input[idx], rmode == RD_HALF_UP);
  }
}

__global__ void g_bf16ToF32(uint16_t *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = d_RawBF16(input[idx]);
  }
}

__global__ void g_f32ToF16(float *input, uint16_t *output, int size,
                           rounding_mode_t rmode = RD_HALF_TO_EVEN) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = d_F16Raw(input[idx], rmode);
  }
}

__global__ void g_f16ToF32(uint16_t *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = d_RawF16(input[idx]);
  }
}

__global__ void g_f32ToF8(float *input, float scale, uint8_t *output, int size, rounding_mode_t rmode = RD_HALF_TO_EVEN) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = fp32_to_fp8(input[idx]*scale);
  }
}

__global__ void g_f8ToF32(uint8_t *input, float scale, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = f8_to_fp32(input[idx], scale);
  }
}

template <typename T> __global__ void g_print(T *data, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    printf("Data[%d] = %g\n", idx, (float)data[idx]);
  }
}

__global__ void g_printBF16(uint16_t *data, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    printf("Data[%d] = %g\n", idx, d_RawBF16(data[idx]));
  }
}

__global__ void g_printF16(uint16_t *data, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    printf("Data[%d] = %g\n", idx, d_RawF16(data[idx]));
  }
}

template <typename T> __global__ void g_doRelu(T *data, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    data[idx] = max(static_cast<T>(0), data[idx]);
  }
}

template <typename T>
__global__ void g_maxAxis(T *input, T *output, int outer_dim, int axis_dim,
                          int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int inner_idx = idx % inner_dim;
  int outer_idx = idx /inner_dim;
  if (inner_idx < inner_dim && outer_idx < outer_dim) {
    int input_offset = outer_idx * axis_dim * inner_dim;
    // find max
    T max_v = input[input_offset + inner_idx];
    for (int i = 1; i < axis_dim; i++) {
      T v = input[input_offset + inner_idx + i * inner_dim];
      if (v > max_v) {
        max_v = v;
      }
    }
    output[outer_idx * inner_dim + inner_idx] = max_v;
  }
}

__global__ void g_maxAxisBF16(uint16_t *input, uint16_t *output, int outer_dim,
                              int axis_dim, int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (outer_dim * inner_dim)) {
    int outer_idx = idx / inner_dim;
    int inner_idx = idx % inner_dim;
    int outer_offset = outer_idx * axis_dim * inner_dim;
    // find max
    float max_v = d_RawBF16(input[outer_offset + inner_idx]);
    int max_idx = 0;
    for (int i = 1; i < axis_dim; i++) {
      int idx = outer_offset + inner_idx + i * inner_dim;
      float v = d_RawBF16(input[idx]);
      if (max_v < v) {
        max_v = v;
        max_idx = idx;
      }
    }
    output[outer_idx * inner_dim + inner_idx] = input[max_idx];
  }
}

template <typename T>
__global__ void g_sumAxis(T *input, T *output, int outer_dim, int axis_dim,
                          int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (outer_dim * inner_dim)) {
    int outer_idx = idx / inner_dim;
    int inner_idx = idx % inner_dim;
    int outer_offset = outer_idx * axis_dim * inner_dim;
    // sum up
    T sum = 0;
    for (int i = 0; i < axis_dim; i++) {
      sum += input[outer_offset + inner_idx + i * inner_dim];
    }
    output[outer_idx * inner_dim + inner_idx] = sum;
  }
}

__global__ void g_sumAxisBF16(uint16_t *input, uint16_t *output, int outer_dim,
                              int axis_dim, int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (outer_dim * inner_dim)) {
    int outer_idx = idx / inner_dim;
    int inner_idx = idx % inner_dim;
    int outer_offset = outer_idx * axis_dim * inner_dim;
    // find max
    float sum = 0.0f;
    for (int i = 0; i < axis_dim; i++) {
      sum += d_RawBF16(input[outer_offset + inner_idx + i * inner_dim]);
    }
    output[outer_idx * inner_dim + inner_idx] = d_BF16Raw(sum);
  }
}

template <typename T>
__global__ void g_subAxis(T *input, T *sub, T *output, int outer_dim,
                          int axis_dim, int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int outer_idx = idx / (axis_dim * inner_dim);
  int axis_idx = idx % (axis_dim * inner_dim) / inner_dim;
  int inner_idx = idx % inner_dim;
  if (inner_idx < inner_dim && outer_idx < outer_dim && axis_idx < axis_dim) {
    int sub_idx = outer_idx * inner_dim + inner_idx;
    output[idx] = input[idx] - sub[sub_idx];
  }
}

__global__ void g_subAxisBF16(uint16_t *input, uint16_t *sub, uint16_t *output,
                              int outer_dim, int axis_dim, int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int outer_idx = idx / (axis_dim * inner_dim);
  int axis_idx = idx % (axis_dim * inner_dim) / inner_dim;
  int inner_idx = idx % inner_dim;
  if (inner_idx < inner_dim && outer_idx < outer_dim && axis_idx < axis_dim) {
    int sub_idx = outer_idx * inner_dim + inner_idx;
    float out = d_RawBF16(input[idx]) - d_RawBF16(sub[sub_idx]);
    output[idx] = d_BF16Raw(out);
  }
}

template <typename T>
__global__ void g_addAxis(T *input, T *add, T *output, int outer_dim,
                          int axis_dim, int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (inner_dim * outer_dim * axis_dim)) {
    int outer_idx = idx / (axis_dim * inner_dim);
    int inner_idx = idx % inner_dim;
    int add_idx = outer_idx * inner_dim + inner_idx;
    output[idx] = input[idx] + add[add_idx];
  }
}

__global__ void g_addAxisBF16(uint16_t *input, uint16_t *add, uint16_t *output,
                              int outer_dim, int axis_dim, int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (inner_dim * outer_dim * axis_dim)) {
    int outer_idx = idx / (axis_dim * inner_dim);
    int inner_idx = idx % inner_dim;
    int add_idx = outer_idx * inner_dim + inner_idx;
    float out = d_RawBF16(input[idx]) + d_RawBF16(add[add_idx]);
    output[idx] = d_BF16Raw(out);
  }
}

template <typename T>
__global__ void g_mulAxis(T *input, T *mul, T *output, int outer_dim,
                          int axis_dim, int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int outer_idx = idx / (axis_dim * inner_dim);
  int axis_idx = idx % (axis_dim * inner_dim) / inner_dim;
  int inner_idx = idx % inner_dim;
  if (inner_idx < inner_dim && outer_idx < outer_dim && axis_idx < axis_dim) {
    int sub_idx = outer_idx * inner_dim + inner_idx;
    T val = input[idx] * mul[sub_idx];
    output[idx] = val;
  }
}

__global__ void g_mulAxisBF16(uint16_t *input, uint16_t *mul, uint16_t *output,
                              int outer_dim, int axis_dim, int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int outer_idx = idx / (axis_dim * inner_dim);
  int axis_idx = idx % (axis_dim * inner_dim) / inner_dim;
  int inner_idx = idx % inner_dim;
  if (inner_idx < inner_dim && outer_idx < outer_dim && axis_idx < axis_dim) {
    int mul_idx = outer_idx * inner_dim + inner_idx;
    float out = d_RawBF16(input[idx]) * d_RawBF16(mul[mul_idx]);
    output[idx] = d_BF16Raw(out);
  }
}

__global__ void g_layerNorm(float *input, float *output, int outer_dim,
                              int inner_dim, float *weight, float *bias, float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (outer_dim)) {
    float *base_ptr = input+ idx*inner_dim;
    float sum = 0.0f;
    for (int inner_idx = 0;inner_idx< inner_dim; inner_idx ++) {
      float val = base_ptr[inner_idx];
      sum += val;
    }
    float mean = sum / inner_dim;
    float rstd = 0.0f;
    for (int inner_idx = 0;inner_idx< inner_dim; inner_idx ++) {
      float diff = base_ptr[inner_idx] - mean;
      rstd += diff * diff;
    }
    rstd = rstd / inner_dim;
    rstd += eps;
    float inv_std = rsqrtf(rstd);
    for (int inner_idx = 0;inner_idx< inner_dim; inner_idx ++) {
      float val = base_ptr[inner_idx];
      float norm = (val - mean) * inv_std;
      if (weight != nullptr)
        norm = norm*weight[inner_idx];
      if (bias != nullptr)
        norm = norm + bias[inner_idx];
      output[idx * inner_dim + inner_idx] = norm;
    }
  }
}

__global__ void g_layerNormBF16(float *input, float *output, int outer_dim,
                              int inner_dim, float *weight, float *bias, float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (outer_dim)) {
    float *base_ptr = input+ idx*inner_dim;
    float mean = 0.0f;
    float scale = d_BF16(1.0f / inner_dim);
    for (int inner_idx = 0;inner_idx< inner_dim; inner_idx ++) {
      float val = base_ptr[inner_idx] * scale;
      mean += val;
    }
    mean = d_BF16(mean);
    float rstd = 0.0f;
    for (int inner_idx = 0;inner_idx< inner_dim; inner_idx ++) {
      float diff = d_BF16(base_ptr[inner_idx] - mean);
      rstd += d_BF16(d_BF16(diff * diff)*scale);
    }
    rstd = d_BF16(rstd + eps);
    float inv_std = d_BF16(rsqrtf(rstd));
    for (int inner_idx = 0;inner_idx< inner_dim; inner_idx ++) {
      float val = base_ptr[inner_idx];
      float norm = d_BF16(d_BF16(val - mean) * inv_std);
      if (weight != nullptr)
        norm = d_BF16(norm*weight[inner_idx]);
      if (bias != nullptr)
        norm = d_BF16(norm + bias[inner_idx]);
      output[idx * inner_dim + inner_idx] = d_BF16(norm);
    }
  }
}

template <typename T0, typename T1>
__global__ void g_lut256(T0 *src, T1 *table, T1 *dst, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    int32_t offset = static_cast<int32_t>(src[idx]);
    if (offset < 0) {
      offset += 256;
    }
    if (offset >= 0 && offset < 256) {
      dst[idx] = table[offset];
    }
  }
}

__global__ void g_upsample4D(void *input, void *output, int n, int c, int ih,
                             int iw, int scale_h, int scale_w, int tbytes) {
  int oh = ih * scale_h;
  int ow = iw * scale_w;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (n * c * oh * ow)) {
    int dst_n = idx / (c * oh * ow);
    int dst_c = idx % (c * oh * ow) / (oh * ow);
    int dst_h = idx % (oh * ow) / ow;
    int dst_w = idx % ow;
    int dst_idx = ((dst_n * c + dst_c) * oh + dst_h) * ow + dst_w;
    int src_w = dst_w / scale_w;
    int src_h = dst_h / scale_h;
    int src_idx = ((dst_n * c + dst_c) * ih + src_h) * iw + src_w;
    d_copyElement(input, src_idx, output, dst_idx, tbytes);
  }
}

__global__ void g_depth2Space(void *input, void *output, int in, int ic, int ih,
                              int iw, int on, int oc, int oh, int ow,
                              int instride, int icstride, int ihstride,
                              int iwstride, int onstride, int ocstride,
                              int ohstride, int owstride, int block_h,
                              int block_w, bool crd, bool swap_cr,
                              bool inversed, int tbytes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (in * ic * ih * iw)) {
    int n = idx / (ic * ih * iw);
    int c = idx % (ic * ih * iw) / (ih * iw);
    int h = idx % (ih * iw) / iw;
    int w = idx % iw;
    int new_c, new_h, new_w, left;
    if (crd) {
      new_c = c / (block_h * block_w);
      left = c % (block_h * block_w);
    } else {
      new_c = c % oc;
      left = c / oc;
    }
    if (swap_cr) {
      int64_t c1 = left / block_w;
      int64_t c2 = left % block_w;
      int64_t rleft = c2 * block_h + c1;
      if (crd) {
        c = new_c * (block_h * block_w) + rleft;
      } else {
        c = rleft * oc + new_c;
      }
    }
    new_h = h * block_h + left / block_w;
    new_w = w * block_w + left % block_w;
    int64_t i_index = n * instride + c * icstride + h * ihstride + w * iwstride;
    int64_t o_index =
        n * onstride + new_c * ocstride + new_h * ohstride + new_w * owstride;
    if (inversed) {
      d_copyElement(input, o_index, output, i_index, tbytes);
    } else {
      d_copyElement(input, i_index, output, o_index, tbytes);
    }
  }
}

template <typename T0, typename T1>
__global__ void g_gather(T0 *indices, T1 *embedding, T1 *output,
                         int num_indices, int embedding_dim, int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_indices) {
    int index = static_cast<int>(indices[idx]);
    if (index < embedding_dim && index >= 0) {
      for (int i = 0; i < inner_dim; i++) {
        output[idx * inner_dim + i] = embedding[index * inner_dim + i];
      }
    }
  }
}

// -------------------------------------------------------------------------
// ------- cv18xx functions
__global__ void g_cvInt8ScaleToF32(int8_t *input, float *output, float scale,
                                   int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float intermediate = static_cast<float>(input[idx]);
    output[idx] = d_BF16(intermediate * scale);
  }
}

__global__ void g_cvInt8ScaleToBF16(int8_t *input, uint16_t *output,
                                    float scale, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float intermediate = static_cast<float>(input[idx]);
    output[idx] = d_BF16Raw(intermediate * scale);
  }
}

__global__ void g_cvF32ScaleToInt8(float *input, int8_t *output, float scale,
                                   int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    auto out_bf16 = d_BF16(d_BF16(input[idx], false) * scale);
    output[idx] = d_f32ToInt<int8_t>(out_bf16, RD_HALF_TO_EVEN);
  }
}

__global__ void g_cvBF16ScaleToInt8(uint16_t *input, int8_t *output,
                                    float scale, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    auto out_bf16 = d_BF16(d_RawBF16(input[idx]) * scale);
    output[idx] = d_f32ToInt<int8_t>(out_bf16, RD_HALF_TO_EVEN);
  }
}

__global__ void g_cvAdd4DInt8(int8_t *a, int8_t *b, int8_t *out, int32_t mul0,
                              int32_t mul1, int shift, bool relu, int n0,
                              int c0, int h0, int w0, int n1, int c1, int h1,
                              int w1, int on, int oc, int oh, int ow) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_n = dst_idx / (oc * oh * ow);
  int idx_c = dst_idx % (oc * oh * ow) / (oh * ow);
  int idx_h = dst_idx % (oh * ow) / ow;
  int idx_w = dst_idx % ow;
  if (idx_w < ow && idx_h < oh && idx_c < oc && idx_n < on) {
    int idx_n0 = idx_n % n0;
    int idx_c0 = idx_c % c0;
    int idx_h0 = idx_h % h0;
    int idx_w0 = idx_w % w0;
    int idx_0 = ((idx_n0 * c0 + idx_c0) * h0 + idx_h0) * w0 + idx_w0;
    int idx_n1 = idx_n % n1;
    int idx_c1 = idx_c % c1;
    int idx_h1 = idx_h % h1;
    int idx_w1 = idx_w % w1;
    int idx_1 = ((idx_n1 * c1 + idx_c1) * h1 + idx_h1) * w1 + idx_w1;
    int32_t temp = (int32_t)a[idx_0] * mul0 + (int32_t)b[idx_1] * mul1;
    temp = (temp + (1 << (shift - 1))) >> shift;
    int32_t min_ = relu ? 0 : -128;
    temp = max(min_, min(127, temp));
    out[dst_idx] = static_cast<int8_t>(temp);
  }
}

__global__ void g_cvPReluInt8(int8_t *input, int8_t *slope, int8_t *output,
                              int outer_dim, int inner_dim, int num_slope,
                              int multi_pos, int shift_pos, int shift_neg) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < outer_dim * inner_dim) {
    int outer_idx = idx / inner_dim;
    int slope_idx = outer_idx % num_slope;
    int8_t data = input[idx];
    if (data < 0) {
      int32_t value = static_cast<int32_t>(data * slope[slope_idx]);
      value = (value + (1 << (shift_neg - 1))) >> shift_neg; // half up
      value = max(-128, min(127, value));
      output[idx] = static_cast<int8_t>(value);
    } else {
      int32_t value = static_cast<int32_t>(data) * multi_pos;
      value = (value + (1 << (shift_pos - 1))) >> shift_pos; // half up
      value = max(-128, min(127, value));
      output[idx] = static_cast<int8_t>(value);
    }
  }
}

__global__ void g_cvMulShiftInt8(int8_t *input, int8_t *output, int multiplier,
                                 int shift, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int32_t value = static_cast<int32_t>(input[idx]) * multiplier;
    value = (value + (1 << (shift - 1))) >> shift; // half up
    value = max(-128, min(127, value));
    output[idx] = static_cast<int8_t>(value);
  }
}

__global__ void g_cvLutSlope(uint16_t *input, uint16_t *output,
                             uint16_t *table0, uint16_t *table1, int num,
                             float scale, float offset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    output[idx] = d_lutSlopeBF16(input[idx], table0, table1, scale, offset);
  }
}

__global__ void g_bmExp(float *input, float *output, int outer_dim, int axis_dim, int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int out_idx = idx / (axis_dim * inner_dim);
  int axis_idx = idx % (axis_dim * inner_dim) / inner_dim;
  int inner_idx = idx % inner_dim;
  if (out_idx < outer_dim && axis_idx < axis_dim && inner_idx < inner_dim) {
    float value = __expf(input[idx]);
    output[idx] = value;
  }
}

__global__ void g_bmReciprocal(float *input, float *output, int outer_dim, int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int out_idx = idx / inner_dim;
  int inner_idx = idx % inner_dim;
  if (out_idx < outer_dim && inner_idx < inner_dim) {
    float value = 1.0/(input[idx]);
    output[idx] = value;
  }
}

__global__ void g_cvLutMantissa(uint16_t *input, uint16_t *output,
                                uint16_t *table0, uint16_t *table1, int num,
                                bool is_log) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    output[idx] = d_lutMantissaBF16(input[idx], table0, table1, is_log);
  }
}

template<typename T>
__global__ void g_depth2space(
    const T* input, T* output,
    int block_h, int block_w,
    bool inversed,
    bool swap_output_dims,
    int is_crd,
    int n, int c, int h, int w,
    int instride, int icstride, int ihstride, int iwstride,
    int on, int oc, int oh, int ow,
    int onstride, int ocstride, int ohstride, int owstride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx > n*c*h*w)
    return;
  int64_t idx_n = idx / (c * h * w);
  int64_t idx_c = (idx % (c * h * w)) / (h * w);
  int64_t idx_h = (idx % (h*w)) / (w);
  int64_t idx_w = (idx % (h*w)) % w;
  int64_t new_c, left;
  if (is_crd) { // oc, block_h, block_w
    new_c = idx_c / (block_h * block_w);
    left = idx_c % (block_h * block_w);
  } else { // bh, bw, oc
    new_c = idx_c % oc;
    left = idx_c / oc;
  }
  if (swap_output_dims) {
    int64_t c1 = left / block_w;
    int64_t c2 = left % block_w;
    int64_t rleft = c2 * block_h + c1;
    if (is_crd) {
      idx_c = new_c * (block_h * block_w) + rleft;
    } else {
      idx_c = rleft * oc + new_c;
    }
  }
  int64_t new_h = idx_h * block_h + left / block_w;
  int64_t new_w = idx_w * block_w + left % block_w;
  int64_t i_index =
      idx_n * instride + idx_c * icstride + idx_h * ihstride + idx_w * iwstride;
  int64_t o_index = idx_n * onstride + new_c * ocstride + new_h * ohstride +
                    new_w * owstride;
  if (inversed) {
    output[i_index] = input[o_index];
  } else {
    output[o_index] = input[i_index];
  }
}

template<typename T>
__global__ void depth_to_space_kernel(
    const T* input, T* output,
    int block_h, int block_w,
    bool swap_output_dims,  //
    int channel_order,      // 0:DCR, 1:CRD, 2:RCD
    int n, int c, int h, int w) {

    int block_total = block_h * block_w;
    int output_c = c / block_total;

    //
    int output_h = swap_output_dims ? w * block_w : h * block_h;
    int output_w = swap_output_dims ? h * block_h : w * block_w;

    int total_output = n * output_c * output_h * output_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_output) return;

    if (idx == 3)
      printf("DepthToSpace: block_h=%d, block_w=%d, swap_output_dims=%d, channel_order=%d, n=%d, c=%d, h=%d, w=%d, output_c=%d, output_h=%d, output_w=%d, total_output=%d\n",
             block_h, block_w, swap_output_dims, channel_order, n, c, h, w,
             output_c, output_h, output_w, total_output);
    //
    int n_idx = idx / (output_c * output_h * output_w);
    int remaining = idx % (output_c * output_h * output_w);
    int c_idx = remaining / (output_h * output_w);
    remaining %= (output_h * output_w);
    int h_idx = remaining / output_w;
    int w_idx = remaining % output_w;

    //
    int orig_h, orig_w;
    if (swap_output_dims) {
        orig_h = w_idx;
        orig_w = h_idx;
    } else {
        orig_h = h_idx;
        orig_w = w_idx;
    }

    //
    int block_row = orig_h % block_h;
    int block_col = orig_w % block_w;
    int input_h = orig_h / block_h;
    int input_w = orig_w / block_w;

    //
    int input_c;
    if (channel_order == 0) {
        // DCR: Depth-Column-Row
        input_c = c_idx * block_total + block_col * block_h + block_row;
    } else if (channel_order == 1) {
        // CRD: Column-Row-Depth
        input_c = block_col * (block_h * output_c) + block_row * output_c + c_idx;
    } else if (channel_order == 2) {
        // RCD: Row-Column-Depth
        input_c = block_row * (block_w * output_c) + block_col * output_c + c_idx;
    } else {
        //
        input_c = c_idx * block_total + block_col * block_h + block_row;
    }

    if (idx == 3)
      printf("d2s: n_idx=%d, c_idx=%d, h_idx=%d, w_idx=%d, orig_h=%d, orig_w=%d, \
        block_row=%d, block_col=%d, input_h=%d, input_w=%d, input_c=%d\n", \
             n_idx, c_idx, h_idx, w_idx, orig_h, orig_w, block_row, block_col, \
             input_h, input_w, input_c);
    //
    int input_idx = ((n_idx * c + input_c) * h + input_h) * w + input_w;
    if (idx == 3)
        printf("d2s: input_idx=%d\n", input_idx);
    output[idx] = input[input_idx];
}


template<typename T>
__global__ void space_to_depth_kernel(
    const T* input, T* output,
    int block_h, int block_w,
    bool swap_input_dims,
    int channel_order,
    int n, int c, int h, int w) {

    int block_total = block_h * block_w;
    int output_c = c * block_total;

    //
    int output_h = swap_input_dims ? w / block_w : h / block_h;
    int output_w = swap_input_dims ? h / block_h : w / block_w;

    int total_output = n * output_c * output_h * output_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_output) return;

    //
    int n_idx = idx / (output_c * output_h * output_w);
    int remaining = idx % (output_c * output_h * output_w);
    int c_idx = remaining / (output_h * output_w);
    remaining %= (output_h * output_w);
    int h_idx = remaining / output_w;
    int w_idx = remaining % output_w;

    if (idx == 3)
      printf("SpaceToDepth: block_h=%d, block_w=%d, swap_input_dims=%d, channel_order=%d, n=%d, c=%d, h=%d, w=%d, output_c=%d, output_h=%d, output_w=%d, total_output=%d\n",
             block_h, block_w, swap_input_dims, channel_order, n, c, h, w,
             output_c, output_h, output_w, total_output);

    //
    int depth, block_row, block_col;

    if (channel_order == 0) {
        // DCR: Depth-Column-Row
        depth = c_idx / block_total;
        int block_offset = c_idx % block_total;
        block_col = block_offset / block_h;
        block_row = block_offset % block_h;
    } else if (channel_order == 1) {
        // CRD: Column-Row-Depth
        block_col = c_idx / (block_h * output_c);
        int remaining = c_idx % (block_h * output_c);
        block_row = remaining / output_c;
        depth = remaining % output_c;
    } else if (channel_order == 2) {
        // RCD: Row-Column-Depth
        block_row = c_idx / (block_w * output_c);
        int remaining = c_idx % (block_w * output_c);
        block_col = remaining / output_c;
        depth = remaining % output_c;
    } else {
        // DCR
        depth = c_idx / block_total;
        int block_offset = c_idx % block_total;
        block_col = block_offset / block_h;
        block_row = block_offset % block_h;
    }

    //
    int input_h, input_w;
    if (swap_input_dims) {
        input_h = h_idx * block_w + block_col;
        input_w = w_idx * block_h + block_row;
    } else {
        input_h = h_idx * block_h + block_row;
        input_w = w_idx * block_w + block_col;
    }

    //
    int final_input_h = swap_input_dims ? input_w : input_h;
    int final_input_w = swap_input_dims ? input_h : input_w;

    //
    int input_c = depth;
    if (idx == 3)
      printf("s2d: n_idx=%d, c_idx=%d, h_idx=%d, w_idx=%d, depth=%d, block_row=%d, block_col=%d, input_h=%d, input_w=%d, final_input_h=%d, final_input_w=%d, input_c=%d\n",
             n_idx, c_idx, h_idx, w_idx, depth, block_row, block_col,
             input_h, input_w, final_input_h, final_input_w, input_c);
    //
    int input_idx = ((n_idx * c + input_c) * h + final_input_h) * w + final_input_w;
    if (idx == 3)
        printf("s2d: input_idx=%d\n", input_idx);
    output[idx] = input[input_idx];
}


enum ReductionMode {
    REDUCE_SUM = 0,
    REDUCE_MEAN,
    REDUCE_MAX,
    REDUCE_MIN,
    REDUCE_L2_NORM,
    REDUCE_L1_NORM,
    REDUCE_PROD,     // Product
    REDUCE_VAR,      // Variance
    REDUCE_STD,      // Standard deviation
    REDUCE_ANY,      // Logical OR (for boolean)
    REDUCE_ALL       // Logical AND (for boolean)
};

// Helper function to get initial value based on mode
template<typename T, ReductionMode Mode>
__device__ __inline__ T getInitialValue() {
    if (Mode == REDUCE_MAX) {
        return (T)-CUDART_INF_F;
    } else if (Mode == REDUCE_MIN) {
        return (T)CUDART_INF_F;
    } else if (Mode == REDUCE_PROD) {
        return T(1);
    } else if (Mode == REDUCE_ANY) {
        return T(0);
    } else if (Mode == REDUCE_ALL) {
        return T(1);
    } else {
        return T(0);
    }
}

// Helper function to combine two values based on mode
template<typename T, ReductionMode Mode>
__device__ __inline__ T combineValues(T a, T b) {
    if (Mode == REDUCE_MAX) {
        return max(a, b);
    } else if (Mode == REDUCE_MIN) {
        return min(a, b);
    } else if (Mode == REDUCE_SUM || Mode == REDUCE_MEAN ||
               Mode == REDUCE_L2_NORM || Mode == REDUCE_L1_NORM ||
               Mode == REDUCE_VAR || Mode == REDUCE_STD) {
        return a + b;
    } else if (Mode == REDUCE_PROD) {
        return a * b;
    } else if (Mode == REDUCE_ANY) {
        return a || b;
    } else if (Mode == REDUCE_ALL) {
        return a && b;
    }
    return a;  // Default, should not reach here
}

// Structure to hold tensor shape information
struct TensorShape {
    int dims[8];          // Support up to 8 dimensions
    int strides[8];       // Strides for each dimension
    int ndim;             // Number of dimensions (up to 8)

    __host__ __device__ TensorShape() : ndim(0) {}

    __host__ __device__ void init(int n_dim, const int*shape) {
        ndim = n_dim;
        for (int i = 0; i < ndim; i++) {
            dims[i] = shape[i];
        }
        for (int i = ndim; i < 8; i++) {
            dims[i] = 1;  // Fill remaining dimensions with 1
        }
        computeStrides();
    }

    __host__ __device__ void computeStrides() {
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
    }

    __host__ __device__ int totalElements() const {
        int total = 1;
        for (int i = 0; i < ndim; i++) {
            total *= dims[i];
        }
        return total;
    }

    __host__ __device__ int linearIndex(const int indices[8]) const {
        int idx = 0;
        for (int i = 0; i < ndim; i++) {
            idx += indices[i] * strides[i];
        }
        return idx;
    }

    __host__ __device__ void computeIndices(int linear_idx, int indices[8]) const {
        for (int i = 0; i < ndim; i++) {
            indices[i] = (linear_idx / strides[i]) % dims[i];
        }
    }
};

// Kernel for multi-axis reduction
template<typename T, ReductionMode Mode, int BlockSize = 256>
__global__ void multiAxisReductionKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    TensorShape input_shape,
    TensorShape output_shape,
    const int* __restrict__ reduce_mask  // Boolean mask indicating which axes to reduce
) {
    // Each thread handles one element in the output tensor
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (output_idx < output_shape.totalElements()) {
        // Compute indices in output tensor
        int output_indices[8];
        output_shape.computeIndices(output_idx, output_indices);

        // Convert to input indices (with reduce dimensions as 0)
        int input_indices[8];
        int reduce_idx = 0;
        for (int i = 0; i < input_shape.ndim; i++) {
            if (reduce_mask[i]) {
                // This dimension is being reduced, start with 0
                input_indices[i] = 0;
            } else {
                // Copy from output indices
                input_indices[i] = output_indices[reduce_idx++];
            }
        }

        // Initialize reduction value
        T myVal = getInitialValue<T, Mode>();

        // Calculate total elements to reduce
        int reduce_total = 1;
        for (int i = 0; i < input_shape.ndim; i++) {
            if (reduce_mask[i]) {
                reduce_total *= input_shape.dims[i];
            }
        }

        // Nested loops over reduction dimensions (optimized for up to 4 reduction dims)
        if (reduce_total > 0) {
            // Count reduction dimensions
            int reduce_dims[4];
            int num_reduce_dims = 0;
            for (int i = 0; i < input_shape.ndim; i++) {
                if (reduce_mask[i]) {
                    reduce_dims[num_reduce_dims++] = i;
                }
            }

            // Handle different numbers of reduction dimensions
            if (num_reduce_dims == 1) {
                int dim = reduce_dims[0];
                for (int i0 = 0; i0 < input_shape.dims[dim]; i0++) {
                    input_indices[dim] = i0;
                    int idx = input_shape.linearIndex(input_indices);
                    T element = input[idx];
                    myVal = combineValues<T, Mode>(myVal, element);
                }
            } else if (num_reduce_dims == 2) {
                int dim1 = reduce_dims[0];
                int dim2 = reduce_dims[1];
                for (int i0 = 0; i0 < input_shape.dims[dim1]; i0++) {
                    input_indices[dim1] = i0;
                    for (int i1 = 0; i1 < input_shape.dims[dim2]; i1++) {
                        input_indices[dim2] = i1;
                        int idx = input_shape.linearIndex(input_indices);
                        T element = input[idx];
                        myVal = combineValues<T, Mode>(myVal, element);
                    }
                }
            } else if (num_reduce_dims == 3) {
                int dim1 = reduce_dims[0];
                int dim2 = reduce_dims[1];
                int dim3 = reduce_dims[2];
                for (int i0 = 0; i0 < input_shape.dims[dim1]; i0++) {
                    input_indices[dim1] = i0;
                    for (int i1 = 0; i1 < input_shape.dims[dim2]; i1++) {
                        input_indices[dim2] = i1;
                        for (int i2 = 0; i2 < input_shape.dims[dim3]; i2++) {
                            input_indices[dim3] = i2;
                            int idx = input_shape.linearIndex(input_indices);
                            T element = input[idx];
                            myVal = combineValues<T, Mode>(myVal, element);
                        }
                    }
                }
            } else if (num_reduce_dims == 4) {
                int dim1 = reduce_dims[0];
                int dim2 = reduce_dims[1];
                int dim3 = reduce_dims[2];
                int dim4 = reduce_dims[3];
                for (int i0 = 0; i0 < input_shape.dims[dim1]; i0++) {
                    input_indices[dim1] = i0;
                    for (int i1 = 0; i1 < input_shape.dims[dim2]; i1++) {
                        input_indices[dim2] = i1;
                        for (int i2 = 0; i2 < input_shape.dims[dim3]; i2++) {
                            input_indices[dim3] = i2;
                            for (int i3 = 0; i3 < input_shape.dims[dim4]; i3++) {
                                input_indices[dim4] = i3;
                                int idx = input_shape.linearIndex(input_indices);
                                T element = input[idx];
                                myVal = combineValues<T, Mode>(myVal, element);
                            }
                        }
                    }
                }
            } else {
                // Generic case for more than 4 reduction dimensions
                // Use a while loop for arbitrary number of reduction dims
                int reduce_indices[8] = {0};
                bool done = false;

                while (!done) {
                    // Set indices for reduction dimensions
                    int reduce_idx = 0;
                    for (int i = 0; i < input_shape.ndim; i++) {
                        if (reduce_mask[i]) {
                            input_indices[i] = reduce_indices[reduce_idx++];
                        }
                    }

                    // Access element
                    int idx = input_shape.linearIndex(input_indices);
                    T element = input[idx];
                    myVal = combineValues<T, Mode>(myVal, element);

                    // Increment reduction indices
                    int carry = 1;
                    for (int i = num_reduce_dims - 1; i >= 0 && carry; i--) {
                        int dim = reduce_dims[i];
                        reduce_indices[i]++;
                        if (reduce_indices[i] >= input_shape.dims[dim]) {
                            reduce_indices[i] = 0;
                            carry = 1;
                        } else {
                            carry = 0;
                        }
                    }
                    done = carry;
                }
            }

            // Post-processing based on mode
            if (Mode == REDUCE_MEAN) {
                myVal /= reduce_total;
            } else if (Mode == REDUCE_L2_NORM) {
                myVal = sqrt(myVal);
            } else if (Mode == REDUCE_VAR || Mode == REDUCE_STD) {
                // Note: For variance, this computes sum of squares
                // Need to compute mean first, then variance
            }
        }

        // Write result to output
        output[output_idx] = myVal;
    }
}

// Specialized kernel for variance (requires two passes)
template<typename T, int BlockSize = 256>
__global__ void varianceReductionKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    TensorShape input_shape,
    TensorShape output_shape,
    const bool* __restrict__ reduce_mask,
    T* __restrict__ means_cache = nullptr  // Optional cache for means
) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (output_idx < output_shape.totalElements()) {
        // Similar to multiAxisReductionKernel but with variance calculation
        // This is simplified - actual implementation needs mean first
        T sum = T(0);
        T sum_sq = T(0);

        // Compute mean and sum of squares in one pass (numerically unstable but faster)
        int reduce_total = 1;
        for (int i = 0; i < input_shape.ndim; i++) {
            if (reduce_mask[i]) {
                reduce_total *= input_shape.dims[i];
            }
        }

        if (reduce_total > 0) {
            // Get indices and compute
            int output_indices[8];
            output_shape.computeIndices(output_idx, output_indices);

            int input_indices[8];
            int reduce_idx = 0;
            for (int i = 0; i < input_shape.ndim; i++) {
                if (reduce_mask[i]) {
                    input_indices[i] = 0;
                } else {
                    input_indices[i] = output_indices[reduce_idx++];
                }
            }

            // Iterate over reduction dimensions
            int reduce_dims[4];
            int num_reduce_dims = 0;
            for (int i = 0; i < input_shape.ndim; i++) {
                if (reduce_mask[i]) {
                    reduce_dims[num_reduce_dims++] = i;
                }
            }

            // Single pass for mean and sum of squares
            T mean_accum = T(0);
            T m2_accum = T(0);
            int count = 0;

            // Using Welford's online algorithm for numerical stability
            if (num_reduce_dims == 1) {
                int dim = reduce_dims[0];
                for (int i = 0; i < input_shape.dims[dim]; i++) {
                    input_indices[dim] = i;
                    int idx = input_shape.linearIndex(input_indices);
                    T x = input[idx];

                    count++;
                    T delta = x - mean_accum;
                    mean_accum += delta / count;
                    T delta2 = x - mean_accum;
                    m2_accum += delta * delta2;
                }
            }
            // ... similar for other dimensions

            if (count > 1) {
                T variance = m2_accum / (count - 1);  // Sample variance
                output[output_idx] = variance;
            } else {
                output[output_idx] = T(0);
            }
        }
    }
}

// Optimized kernel for contiguous reduction dimensions
template<typename T, ReductionMode Mode, int BlockSize = 256>
__global__ void contiguousAxisReductionKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int outer_size,      // Product of dimensions before reduction
    int reduce_size,     // Size of dimension being reduced
    int inner_size       // Product of dimensions after reduction
) {
    // This kernel is optimized when reducing a single contiguous axis

    // Each block handles inner_size * outer_size outputs
    int batch = blockIdx.x;
    int inner_idx = threadIdx.x;

    if (batch < outer_size && inner_idx < inner_size) {
        T myVal = getInitialValue<T, Mode>();

        // Reduction over the contiguous dimension
        for (int i = 0; i < reduce_size; i++) {
            int input_idx = (batch * reduce_size + i) * inner_size + inner_idx;
            T element = input[input_idx];
            myVal = combineValues<T, Mode>(myVal, element);
        }

        // Post-processing
        if (Mode == REDUCE_MEAN) {
            myVal /= reduce_size;
        } else if (Mode == REDUCE_L2_NORM) {
            myVal = sqrt(myVal);
        }

        // Write output
        int output_idx = batch * inner_size + inner_idx;
        output[output_idx] = myVal;
    }
}

__global__ void g_divConst4DF32(float *input, float *output, float const_val,
                                bool is_reverse, bool do_relu, int n, int c, int h, int w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n * c * h * w) {
    float val;
    if (is_reverse) {
      val = (input[idx] == 0.0f) ? (const_val / 1e-8f) : (const_val / input[idx]);
    } else {
      val = input[idx] / const_val;
    }
    if (do_relu && val < 0.0f)
      val = 0.0f;
    output[idx] = val;
  }
}

#define EINSUM_MAX_DIMS 6

__global__ void g_einsumF32(
    const float *lhs, const float *rhs, float *out,
    int lhs_shape[EINSUM_MAX_DIMS], int rhs_shape[EINSUM_MAX_DIMS],
    int out_shape[EINSUM_MAX_DIMS],
    int lhs_rank, int rhs_rank, int out_rank, int num_contract,
    int lhs_out_dim[EINSUM_MAX_DIMS], int rhs_out_dim[EINSUM_MAX_DIMS],
    int lhs_contract_dim[EINSUM_MAX_DIMS], int rhs_contract_dim[EINSUM_MAX_DIMS],
    int contract_shapes[EINSUM_MAX_DIMS],
    int total_out_elems, int total_contract_elems) {
  int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_idx >= total_out_elems)
    return;

  int out_multi[EINSUM_MAX_DIMS] = {0};
  int rem = out_idx;
  for (int i = out_rank - 1; i >= 0; i--) {
    out_multi[i] = rem % out_shape[i];
    rem /= out_shape[i];
  }

  float sum = 0.0f;

  for (int c_flat = 0; c_flat < total_contract_elems; c_flat++) {
    int contract_multi[EINSUM_MAX_DIMS] = {0};
    int crem = c_flat;
    for (int i = num_contract - 1; i >= 0; i--) {
      contract_multi[i] = crem % contract_shapes[i];
      crem /= contract_shapes[i];
    }

    int lhs_idx = 0;
    int stride = 1;
    for (int i = lhs_rank - 1; i >= 0; i--) {
      int val = 0;
      for (int j = 0; j < out_rank; j++) {
        if (lhs_out_dim[j] == i) {
          val = out_multi[j];
          break;
        }
      }
      for (int j = 0; j < num_contract; j++) {
        if (lhs_contract_dim[j] == i) {
          val = contract_multi[j];
          break;
        }
      }
      lhs_idx += val * stride;
      stride *= lhs_shape[i];
    }

    int rhs_idx = 0;
    stride = 1;
    for (int i = rhs_rank - 1; i >= 0; i--) {
      int val = 0;
      for (int j = 0; j < out_rank; j++) {
        if (rhs_out_dim[j] == i) {
          val = out_multi[j];
          break;
        }
      }
      for (int j = 0; j < num_contract; j++) {
        if (rhs_contract_dim[j] == i) {
          val = contract_multi[j];
          break;
        }
      }
      rhs_idx += val * stride;
      stride *= rhs_shape[i];
    }

    sum += lhs[lhs_idx] * rhs[rhs_idx];
  }

  out[out_idx] = sum;
}

__global__ void g_mask_rcnn_bbox_pooler(
    const float *__restrict__ feat0, const float *__restrict__ feat1,
    const float *__restrict__ feat2, const float *__restrict__ feat3,
    const float *__restrict__ rois, float *__restrict__ output,
    int feat0_h, int feat0_w, int feat1_h, int feat1_w,
    int feat2_h, int feat2_w, int feat3_h, int feat3_w,
    int batch_size, int C, int roi_slice, int roi_len,
    int PH, int PW, int num_levels) {

  int feat_h[4] = {feat0_h, feat1_h, feat2_h, feat3_h};
  int feat_w[4] = {feat0_w, feat1_w, feat2_w, feat3_w};
  int feat_size[4] = {feat0_h * feat0_w, feat1_h * feat1_w,
                      feat2_h * feat2_w, feat3_h * feat3_w};

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_rois = roi_slice * batch_size;
  int pooled_size = PH * PW;
  int per_roi_elems = C * pooled_size;
  int total_elems = total_rois * per_roi_elems;

  if (idx >= total_elems) return;

  int roi_idx = idx / per_roi_elems;
  int rem = idx % per_roi_elems;
  int c = rem / pooled_size;
  int p = rem % pooled_size;
  int ph = p / PW;
  int pw = p % PW;

  int batch_idx = (int)rois[roi_idx * roi_len + 0];
  float x1 = rois[roi_idx * roi_len + 1];
  float y1 = rois[roi_idx * roi_len + 2];
  float x2 = rois[roi_idx * roi_len + 3];
  float y2 = rois[roi_idx * roi_len + 4];

  float roi_w = x2 - x1;
  float roi_h = y2 - y1;
  float area = roi_w * roi_h;
  if (area < 1.0f) area = 1.0f;

  int level = (int)floorf(2.0f + log2f(sqrtf(area) / 224.0f));
  if (level < 0) level = 0;
  if (level >= num_levels) level = num_levels - 1;

  int stride = 4 << level;
  float spatial_scale = 1.0f / (float)stride;

  float sx1 = x1 * spatial_scale;
  float sy1 = y1 * spatial_scale;
  float sx2 = x2 * spatial_scale;
  float sy2 = y2 * spatial_scale;

  float sroi_w = sx2 - sx1;
  float sroi_h = sy2 - sy1;
  if (sroi_w < 1.0f) sroi_w = 1.0f;
  if (sroi_h < 1.0f) sroi_h = 1.0f;

  float bin_h = sroi_h / PH;
  float bin_w = sroi_w / PW;

  int fh = feat_h[level];
  int fw = feat_w[level];

  float y = sy1 + (ph + 0.5f) * bin_h;
  float x = sx1 + (pw + 0.5f) * bin_w;

  if (y < -1.0f || y > fh || x < -1.0f || x > fw) {
    output[idx] = 0.0f;
    return;
  }

  y = fminf(fmaxf(y, 0.0f), fh - 1.0f);
  x = fminf(fmaxf(x, 0.0f), fw - 1.0f);

  int yl = (int)floorf(y);
  int yh = fminf(yl + 1, fh - 1);
  int xl = (int)floorf(x);
  int xh = fminf(xl + 1, fw - 1);

  float ly = y - yl;
  float lx = x - xl;
  float hy = 1.0f - ly;
  float hx = 1.0f - lx;

  const float *feat_ptrs[4] = {feat0, feat1, feat2, feat3};
  const float *feat = feat_ptrs[level] + batch_idx * C * feat_size[level] + c * feat_size[level];

  float val = hy * hx * feat[yl * fw + xl] +
              hy * lx * feat[yl * fw + xh] +
              ly * hx * feat[yh * fw + xl] +
              ly * lx * feat[yh * fw + xh];

  output[idx] = val;
}

__global__ void g_get_bbox_b_decode(
    const float *__restrict__ rois,
    const float *__restrict__ bbox,
    const float *__restrict__ scores,
    const float *__restrict__ max_val,
    float *__restrict__ cand_boxes,
    float *__restrict__ cand_scores,
    int *__restrict__ cand_indices,
    int *__restrict__ cand_count,
    int total_rois, int num_classes, int num_indexes,
    float delta2bbox_means, float delta2bbox_stds_0, float delta2bbox_stds_1,
    float threshold_score, float max_scalar_c,
    int max_candidates) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = total_rois * num_classes;
  if (idx >= total) return;

  int roi_idx = idx / num_classes;
  int cls_idx = idx % num_classes;

  float score = scores[roi_idx * num_classes + cls_idx];
  if (score < threshold_score) return;

  int bbox_base = roi_idx * num_classes * 4 + cls_idx * 4;
  float dx = bbox[bbox_base + 0] * delta2bbox_stds_0 + delta2bbox_means;
  float dy = bbox[bbox_base + 1] * delta2bbox_stds_0 + delta2bbox_means;
  float dw = bbox[bbox_base + 2] * delta2bbox_stds_1 + delta2bbox_means;
  float dh = bbox[bbox_base + 3] * delta2bbox_stds_1 + delta2bbox_means;

  float roi_x1 = rois[roi_idx * 5 + 1];
  float roi_y1 = rois[roi_idx * 5 + 2];
  float roi_x2 = rois[roi_idx * 5 + 3];
  float roi_y2 = rois[roi_idx * 5 + 4];
  float roi_w = roi_x2 - roi_x1;
  float roi_h = roi_y2 - roi_y1;
  if (roi_w < 1.0f) roi_w = 1.0f;
  if (roi_h < 1.0f) roi_h = 1.0f;
  float roi_ctr_x = roi_x1 + roi_w * 0.5f;
  float roi_ctr_y = roi_y1 + roi_h * 0.5f;

  float pred_ctr_x = dx * roi_w + roi_ctr_x;
  float pred_ctr_y = dy * roi_h + roi_ctr_y;
  dw = fminf(dw, max_scalar_c);
  dh = fminf(dh, max_scalar_c);
  float pred_w = expf(dw) * roi_w;
  float pred_h = expf(dh) * roi_h;

  float x1 = pred_ctr_x - pred_w * 0.5f;
  float y1 = pred_ctr_y - pred_h * 0.5f;
  float x2 = pred_ctr_x + pred_w * 0.5f;
  float y2 = pred_ctr_y + pred_h * 0.5f;

  int pos = atomicAdd(cand_count, 1);
  if (pos >= max_candidates) {
    atomicSub(cand_count, 1);
    return;
  }

  cand_boxes[pos * 4 + 0] = x1;
  cand_boxes[pos * 4 + 1] = y1;
  cand_boxes[pos * 4 + 2] = x2;
  cand_boxes[pos * 4 + 3] = y2;
  cand_scores[pos] = score;
  cand_indices[pos] = (roi_idx << 16) | cls_idx;
}

__global__ void g_get_bbox_b_collect(
    const float *__restrict__ cand_boxes,
    const float *__restrict__ cand_scores,
    const int *__restrict__ cand_indices,
    int num_candidates,
    float *__restrict__ out_bboxes,
    float *__restrict__ out_labels,
    int max_per_img,
    float nms_iou_thr,
    int *__restrict__ processed) {

  for (int i = threadIdx.x; i < max_per_img; i += blockDim.x) {
    out_bboxes[i * 5 + 0] = -1.0f;
    out_bboxes[i * 5 + 1] = 0.0f;
    out_bboxes[i * 5 + 2] = 0.0f;
    out_bboxes[i * 5 + 3] = 0.0f;
    out_bboxes[i * 5 + 4] = 0.0f;
    out_labels[i] = -1.0f;
  }
  __syncthreads();

  if (num_candidates <= 0) return;

  if (threadIdx.x == 0) {
    for (int i = 0; i < num_candidates; i++) processed[i] = 0;

    int collected = 0;
    while (collected < max_per_img) {
      float best_score = -1.0f;
      int best_idx = -1;
      for (int i = 0; i < num_candidates; i++) {
        if (!processed[i] && cand_scores[i] > best_score) {
          best_score = cand_scores[i];
          best_idx = i;
        }
      }
      if (best_idx < 0) break;

      float bx1 = cand_boxes[best_idx * 4 + 0];
      float by1 = cand_boxes[best_idx * 4 + 1];
      float bx2 = cand_boxes[best_idx * 4 + 2];
      float by2 = cand_boxes[best_idx * 4 + 3];
      float ba = (bx2 - bx1) * (by2 - by1);
      int cls_idx = cand_indices[best_idx] & 0xFFFF;

      out_bboxes[collected * 5 + 0] = 0.0f;
      out_bboxes[collected * 5 + 1] = bx1;
      out_bboxes[collected * 5 + 2] = by1;
      out_bboxes[collected * 5 + 3] = bx2;
      out_bboxes[collected * 5 + 4] = by2;
      out_labels[collected] = (float)cls_idx;
      collected++;
      processed[best_idx] = 1;

      for (int j = 0; j < num_candidates; j++) {
        if (processed[j]) continue;
        float ox1 = cand_boxes[j * 4 + 0];
        float oy1 = cand_boxes[j * 4 + 1];
        float ox2 = cand_boxes[j * 4 + 2];
        float oy2 = cand_boxes[j * 4 + 3];
        float oa = (ox2 - ox1) * (oy2 - oy1);

        float ix1 = fmaxf(bx1, ox1);
        float iy1 = fmaxf(by1, oy1);
        float ix2 = fminf(bx2, ox2);
        float iy2 = fminf(by2, oy2);
        float iw = fmaxf(0.0f, ix2 - ix1);
        float ih = fmaxf(0.0f, iy2 - iy1);
        float iarea = iw * ih;
        float uarea = ba + oa - iarea;
        float iou = (uarea > 0.0f) ? (iarea / uarea) : 0.0f;

        if (iou > nms_iou_thr) processed[j] = 1;
      }
    }
  }
  __syncthreads();
}

__global__ void g_mask_rcnn_mask_pooler(
    const float *__restrict__ feat0, const float *__restrict__ feat1,
    const float *__restrict__ feat2, const float *__restrict__ feat3,
    const float *__restrict__ bboxes,
    float *__restrict__ output,
    int feat0_h, int feat0_w, int feat1_h, int feat1_w,
    int feat2_h, int feat2_w, int feat3_h, int feat3_w,
    int batch_size, int C, int total_dets, int roi_len,
    int PH, int PW, int num_levels, float scale_factor) {

  int feat_h[4] = {feat0_h, feat1_h, feat2_h, feat3_h};
  int feat_w[4] = {feat0_w, feat1_w, feat2_w, feat3_w};
  int feat_size[4] = {feat0_h * feat0_w, feat1_h * feat1_w,
                      feat2_h * feat2_w, feat3_h * feat3_w};

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int pooled_size = PH * PW;
  int per_det_elems = C * pooled_size;
  int total_elems = total_dets * per_det_elems;

  if (idx >= total_elems) return;

  int det_idx = idx / per_det_elems;
  int rem = idx % per_det_elems;
  int c = rem / pooled_size;
  int p = rem % pooled_size;
  int ph = p / PW;
  int pw = p % PW;

  int batch_idx = (int)bboxes[det_idx * roi_len + 0];
  if (batch_idx < 0) {
    output[idx] = 0.0f;
    return;
  }

  float x1 = bboxes[det_idx * roi_len + 1] * scale_factor;
  float y1 = bboxes[det_idx * roi_len + 2] * scale_factor;
  float x2 = bboxes[det_idx * roi_len + 3] * scale_factor;
  float y2 = bboxes[det_idx * roi_len + 4] * scale_factor;

  float roi_w = x2 - x1;
  float roi_h = y2 - y1;
  float area = roi_w * roi_h;
  if (area < 1.0f) area = 1.0f;

  int level = (int)floorf(2.0f + log2f(sqrtf(area) / 224.0f));
  if (level < 0) level = 0;
  if (level >= num_levels) level = num_levels - 1;

  int stride = 4 << level;
  float spatial_scale = 1.0f / (float)stride;

  float sx1 = x1 * spatial_scale;
  float sy1 = y1 * spatial_scale;
  float sx2 = x2 * spatial_scale;
  float sy2 = y2 * spatial_scale;

  float sroi_w = sx2 - sx1;
  float sroi_h = sy2 - sy1;
  if (sroi_w < 1.0f) sroi_w = 1.0f;
  if (sroi_h < 1.0f) sroi_h = 1.0f;

  float bin_h = sroi_h / PH;
  float bin_w = sroi_w / PW;

  int fh = feat_h[level];
  int fw = feat_w[level];

  float y = sy1 + (ph + 0.5f) * bin_h;
  float x = sx1 + (pw + 0.5f) * bin_w;

  if (y < -1.0f || y > fh || x < -1.0f || x > fw) {
    output[idx] = 0.0f;
    return;
  }

  y = fminf(fmaxf(y, 0.0f), fh - 1.0f);
  x = fminf(fmaxf(x, 0.0f), fw - 1.0f);

  int yl = (int)floorf(y);
  int yh = fminf(yl + 1, fh - 1);
  int xl = (int)floorf(x);
  int xh = fminf(xl + 1, fw - 1);

  float ly = y - yl;
  float lx = x - xl;
  float hy = 1.0f - ly;
  float hx = 1.0f - lx;

  const float *feat_ptrs[4] = {feat0, feat1, feat2, feat3};
  const float *feat = feat_ptrs[level] + batch_idx * C * feat_size[level] + c * feat_size[level];

  float val = hy * hx * feat[yl * fw + xl] +
              hy * lx * feat[yl * fw + xh] +
              ly * hx * feat[yh * fw + xl] +
              ly * lx * feat[yh * fw + xh];

  output[idx] = val;
}

__global__ void g_maskedFill(
    const float *__restrict__ cond, const float *__restrict__ brn,
    float *__restrict__ output, float const_val, bool inversed,
    int num_elems) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elems) return;

  float cond_val = cond[idx];
  float brn_val = brn[idx];
  output[idx] = (cond_val != 0.0f) ? (inversed ? const_val : brn_val)
                                   : (inversed ? brn_val : const_val);
}

__global__ void g_matchTemplate(
    const float *__restrict__ input, const float *__restrict__ templ,
    float *__restrict__ output,
    int iH, int iW, int tH, int tW, int oH, int oW, int mode) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = oH * oW;
  if (idx >= total) return;

  int out_y = idx / oW;
  int out_x = idx % oW;
  int in_offset = out_y * iW + out_x;
  int patch_size = tH * tW;

  if (mode == 0) {
    float sum = 0.0f;
    for (int ty = 0; ty < tH; ty++) {
      for (int tx = 0; tx < tW; tx++) {
        float diff = input[in_offset + ty * iW + tx] - templ[ty * tW + tx];
        sum += diff * diff;
      }
    }
    output[idx] = sum;
  } else {
    float imean = 0.0f;
    for (int ty = 0; ty < tH; ty++) {
      for (int tx = 0; tx < tW; tx++) {
        imean += input[in_offset + ty * iW + tx];
      }
    }
    imean /= patch_size;

    float tmean = 0.0f;
    for (int i = 0; i < patch_size; i++) {
      tmean += templ[i];
    }
    tmean /= patch_size;

    float dividend = 0.0f, wndSum2 = 0.0f, templSum2 = 0.0f;
    for (int ty = 0; ty < tH; ty++) {
      for (int tx = 0; tx < tW; tx++) {
        float inp = input[in_offset + ty * iW + tx] - imean;
        float tpl = templ[ty * tW + tx] - tmean;
        dividend += inp * tpl;
        wndSum2 += inp * inp;
        templSum2 += tpl * tpl;
      }
    }
    float denom = sqrtf(wndSum2 * templSum2);
    output[idx] = (denom > 1e-8f) ? (dividend / denom) : 0.0f;
  }
}

__global__ void g_max(const float *a, const float *b, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = fmaxf(a[idx], b[idx]);
}

__global__ void g_maxConst(float *in, float *out, float const_val, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = fmaxf(in[idx], const_val);
}

__global__ void g_maxPoolWithMask(
    const float *__restrict__ input, float *__restrict__ output,
    float *__restrict__ mask, int n, int c, int ih, int iw,
    int oh, int ow, int kh, int kw, int sh, int sw,
    int pad_h, int pad_w) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * c * oh * ow;
  if (idx >= total) return;

  int ow_idx = idx % ow;
  int oh_idx = (idx / ow) % oh;
  int c_idx  = (idx / (ow * oh)) % c;
  int n_idx  = idx / (ow * oh * c);

  int hstart = oh_idx * sh - pad_h;
  int wstart = ow_idx * sw - pad_w;
  int hend = min(hstart + kh, ih);
  int wend = min(wstart + kw, iw);
  if (hstart < 0) hstart = 0;
  if (wstart < 0) wstart = 0;

  int in_base = (n_idx * c + c_idx) * ih * iw;
  float max_val = -3.402823e+38f;
  int max_idx = 0;

  for (int h = hstart; h < hend; h++) {
    for (int w = wstart; w < wend; w++) {
      int index = h * iw + w;
      float val = input[in_base + index];
      if (val > max_val) {
        max_val = val;
        max_idx = index;
      }
    }
  }

  output[idx] = max_val;
  mask[idx] = (float)max_idx;
}

__global__ void g_maxUnpool(const float *input, const float *mask,
                            float *output, int n, int c, int oh, int ow,
                            int scale_h, int scale_w, int out_h, int out_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * c * oh * ow;
  if (idx >= total) return;

  int c_idx  = (idx / (ow * oh)) % c;
  int n_idx  = idx / (ow * oh * c);
  int mask_idx = (int)mask[idx];
  int out_base = (n_idx * c + c_idx) * out_h * out_w;
  output[out_base + mask_idx] = input[idx];
}

__global__ void g_meanStdScale(const float *input, float *output,
                               const float *mean, const float *std,
                               const float *scale, const float *zero_point,
                               int n, int c, int h, int w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * c * h * w;
  if (idx >= total) return;

  int ci = (idx / (h * w)) % c;
  float val = (input[idx] - mean[ci]) / std[ci] * scale[ci] + zero_point[ci];
  output[idx] = val;
}

__global__ void g_maxPoolingIndicesBwd(const float *grad_output, const float *indices,
                                       float *grad_input, int num_elems) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elems) return;
  int flat_idx = (int)indices[idx];
  grad_input[flat_idx] = grad_output[idx];
}

__global__ void g_meanRstd(const float *input, float *mean_out, float *rstd_out,
                           float *running_mean, float *running_var,
                           const float *weight, const float *bias,
                           int n, int c, int hw, float eps, float momentum) {
  extern __shared__ float shared[];
  float *s_mean = shared;
  float *s_var  = shared + blockDim.x;

  int tid = threadIdx.x;
  int ci = blockIdx.x;
  if (ci >= c) return;

  int chw = c * hw;

  float sum = 0.0f;
  for (int i = tid; i < n * hw; i += blockDim.x) {
    int batch = i / hw;
    int spatial = i % hw;
    sum += input[batch * chw + ci * hw + spatial];
  }
  s_mean[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) s_mean[tid] += s_mean[tid + s];
    __syncthreads();
  }
  float mean_val = s_mean[0] / (n * hw);

  float var_sum = 0.0f;
  for (int i = tid; i < n * hw; i += blockDim.x) {
    int batch = i / hw;
    int spatial = i % hw;
    float diff = input[batch * chw + ci * hw + spatial] - mean_val;
    var_sum += diff * diff;
  }
  s_var[tid] = var_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) s_var[tid] += s_var[tid + s];
    __syncthreads();
  }
  float var_val = s_var[0] / (n * hw);
  float rstd_val = 1.0f / sqrtf(var_val + eps);

  if (tid == 0) {
    mean_out[ci] = mean_val;
    rstd_out[ci] = rstd_val;
    running_mean[ci] = (1.0f - momentum) * running_mean[ci] + momentum * mean_val;
    running_var[ci]  = (1.0f - momentum) * running_var[ci]  + momentum * var_val;
  }
}

__global__ void g_min(const float *a, const float *b, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = fminf(a[idx], b[idx]);
}
__global__ void g_minConst(float *in, float *out, float const_val, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = fminf(in[idx], const_val);
}
__global__ void g_mish(float *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float x = in[idx];
  float sp = logf(1.0f + expf(x));
  out[idx] = x * tanhf(sp);
}

__global__ void g_meshGrid(const float *input, float *output,
                           int total_elems, int stride, int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elems) return;
  int coord = (idx / stride) % dim;
  output[idx] = input[coord];
}

__global__ void g_mod(const float *a, const float *b, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float divisor = b[idx];
  float mid = a[idx] / divisor;
  out[idx] = a[idx] - floorf(mid) * divisor;
}

__global__ void g_swish(float *in, float *out, float beta, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float x = in[idx];
  out[idx] = x / (1.0f + expf(-x * beta));
}

__global__ void g_swapChannel(const float *in, float *out,
                              const int *order, int n, int c, int frame_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * c * frame_size;
  if (idx >= total) return;

  int fs = frame_size;
  int spatial = idx % fs;
  int c_out = (idx / fs) % c;
  int batch = idx / (c * fs);
  int c_in = order[c_out];

  out[batch * c * fs + c_out * fs + spatial] =
      in[batch * c * fs + c_in * fs + spatial];
}

__global__ void g_scatterElements(float *output, const float *updates,
                                  const int *flat_indices, int upd_num, bool add) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= upd_num) return;
  int fi = flat_indices[idx];
  if (add) atomicAdd(&output[fi], updates[idx]);
  else     output[fi] = updates[idx];
}

__global__ void g_scatterND(float *output, const float *updates,
                            const int *flat_indices, int upd_num, bool add) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= upd_num) return;
  int fi = flat_indices[idx];
  if (add) atomicAdd(&output[fi], updates[idx]);
  else     output[fi] = updates[idx];
}

__global__ void g_scaleLut(const float *input, float *output,
                           const float *scale, const float *bias,
                           int n, int c, int hw) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * c * hw;
  if (idx >= total) return;
  int c_idx = (idx / hw) % c;
  output[idx] = input[idx] * scale[c_idx] + bias[c_idx];
}

__global__ void g_sign(const float *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float v = in[idx];
  out[idx] = (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f);
}

__global__ void g_shuffleChannel(const float *in, float *out,
                                 int n, int c, int frame_size, int group) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * c * frame_size;
  if (idx >= total) return;

  int spatial = idx % frame_size;
  int c_out = (idx / frame_size) % c;
  int batch = idx / (c * frame_size);
  int gc = c / group;
  int c_in = (c_out % group) * gc + (c_out / group);

  out[batch * c * frame_size + c_out * frame_size + spatial] =
      in[batch * c * frame_size + c_in * frame_size + spatial];
}

__global__ void g_sin(const float *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return; out[idx] = sinf(in[idx]);
}
__global__ void g_sinh(const float *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return; out[idx] = sinhf(in[idx]);
}

__global__ void g_selectiveScan(
    const float *c_ptr, const float *deltaA, const float *deltaB_u,
    const float *u_ptr, const float *D_ptr,
    float *output, int Kcdim, int L, int Batch, int has_uD) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Cdim = Kcdim / 2;
  int total = Cdim * Batch;
  if (idx >= total) return;

  int k = idx / Batch;
  int b = idx % Batch;

  float x_up = 0.0f;
  for (int i = 0; i < L; i++) {
    int d_idx = k * L * Batch + i * Batch + b;
    x_up = deltaA[d_idx] * x_up + deltaB_u[d_idx];
    int c_idx = i * Kcdim * Batch + k * Batch + b;
    output[i * Kcdim * Batch + k * Batch + b] = x_up * c_ptr[c_idx];
  }

  float x_down = 0.0f;
  for (int i = 0; i < L; i++) {
    int ri = L - 1 - i;
    int d_idx = (Cdim + k) * L * Batch + ri * Batch + b;
    x_down = deltaA[d_idx] * x_down + deltaB_u[d_idx];
    int c_idx = ri * Kcdim * Batch + (Cdim + k) * Batch + b;
    output[ri * Kcdim * Batch + (Cdim + k) * Batch + b] = x_down * c_ptr[c_idx];
  }

  if (has_uD) {
    for (int l = 0; l < L; l++) {
      int idx = l * Kcdim * Batch + k * Batch + b;
      output[idx] += u_ptr[idx] * D_ptr[k];
      int idx2 = l * Kcdim * Batch + (Cdim + k) * Batch + b;
      output[idx2] += u_ptr[idx2] * D_ptr[Cdim + k];
    }
  }
}

__global__ void g_softplus(const float *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float x = in[idx];
  out[idx] = (x > 20.0f) ? x : logf(1.0f + expf(x));
}
__global__ void g_softsign(const float *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float x = in[idx];
  out[idx] = x / (1.0f + fabsf(x));
}

__global__ void g_sqrt(const float *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return; out[idx] = sqrtf(in[idx]);
}
__global__ void g_TAN(const float *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return; out[idx] = tanf(in[idx]);
}
__global__ void g_LN(const float *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return; out[idx] = logf(in[idx]);
}
__global__ void g_trilu(const float *in, float *out, int batch, int H, int W,
                        int row_stride, int diagonal, bool upper) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * H * W;
  if (idx >= total) return;
  int r = (idx % (H * W)) / W;
  int c = idx % W;
  bool keep;
  if (upper)
    keep = (c >= r + diagonal);
  else
    keep = (c <= r + diagonal);
  out[idx] = keep ? in[idx] : 0.0f;
}

__global__ void g_stridedSlice(const float *in, float *out,
                               const int *flat_indices, int out_num) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= out_num) return;
  out[idx] = in[flat_indices[idx]];
}

__global__ void g_nms(const float *__restrict__ boxes,
                      const float *__restrict__ scores, int *__restrict__ selected_buf,
                      int *__restrict__ count_buf, int batch, int num_classes,
                      int spatial_dim, int max_output_per_class,
                      float iou_threshold, float score_threshold) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_classes = batch * num_classes;
  if (tid >= total_classes) return;

  int b = tid / num_classes;
  int c = tid % num_classes;

  const float *box_base = boxes + b * spatial_dim * 4;
  const float *score_base = scores + (b * num_classes + c) * spatial_dim;
  int *my_selected = selected_buf + tid * max_output_per_class;
  int num_selected = 0;

  for (int iter = 0; iter < max_output_per_class; iter++) {
    float best_score = score_threshold;
    int best_idx = -1;

    for (int i = 0; i < spatial_dim; i++) {
      float s = score_base[i];
      if (s <= best_score) continue;

      bool overlaps = false;
      for (int j = 0; j < num_selected; j++) {
        int si = my_selected[j];
        const float *bi = box_base + i * 4;
        const float *bs = box_base + si * 4;

        float ymin_i = fminf(bi[0], bi[2]);
        float ymax_i = fmaxf(bi[0], bi[2]);
        float xmin_i = fminf(bi[1], bi[3]);
        float xmax_i = fmaxf(bi[1], bi[3]);
        float area_i = (ymax_i - ymin_i) * (xmax_i - xmin_i);
        if (area_i <= 0.0f) { overlaps = true; break; }

        float ymin_s = fminf(bs[0], bs[2]);
        float ymax_s = fmaxf(bs[0], bs[2]);
        float xmin_s = fminf(bs[1], bs[3]);
        float xmax_s = fmaxf(bs[1], bs[3]);
        float area_s = (ymax_s - ymin_s) * (xmax_s - xmin_s);
        if (area_s <= 0.0f) { overlaps = true; break; }

        float iy = fmaxf(0.0f, fminf(ymax_i, ymax_s) - fmaxf(ymin_i, ymin_s));
        if (iy == 0.0f) continue;
        float ix = fmaxf(0.0f, fminf(xmax_i, xmax_s) - fmaxf(xmin_i, xmin_s));
        if (ix == 0.0f) continue;

        float iou = (ix * iy) / (area_i + area_s - ix * iy);
        if (iou > iou_threshold) { overlaps = true; break; }
      }
      if (!overlaps) {
        best_score = s;
        best_idx = i;
      }
    }

    if (best_idx < 0) break;
    my_selected[num_selected++] = best_idx;
  }

  count_buf[tid] = num_selected;
}

} // namespace cuda
} // namespace tpu_mlir
