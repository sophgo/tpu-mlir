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
__global__ void g_add4DF32(T0 *a, T1 *b, T2 *out, bool relu, int n0, int c0,
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

} // namespace cuda
} // namespace tpu_mlir
