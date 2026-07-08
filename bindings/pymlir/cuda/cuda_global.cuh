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
                                 int size, bool sign, rounding_mode_t rmode,
                                 int zero_point) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float value = std::round(input[idx] * scale); // cpu behavior
    if (sign) {
      static_cast<int8_t *>(output)[idx] = d_f32ToInt<int8_t>(value + zero_point, rmode);
    } else {
      static_cast<uint8_t *>(output)[idx] = d_f32ToInt<uint8_t>(value + zero_point, rmode);
    }
  }
}

__global__ void g_bf16ScaleToInt8(uint16_t *input, void *output, float scale,
                                  int size, bool sign, rounding_mode_t rmode,
                                  int zero_point) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float value = d_BF16(d_RawBF16(input[idx]) * d_BF16(scale));
    if (sign) {
      static_cast<int8_t *>(output)[idx] = d_f32ToInt<int8_t>(value + zero_point, rmode);
    } else {
      static_cast<uint8_t *>(output)[idx] = d_f32ToInt<uint8_t>(value + zero_point, rmode);
    }
  }
}

__global__ void g_f16ScaleToInt8(uint16_t *input, void *output, float scale,
                                 int size, bool sign, rounding_mode_t rmode,
                                 int zero_point) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float value = d_F16(d_RawF16(input[idx]) * d_F16(scale));
    if (sign) {
      static_cast<int8_t *>(output)[idx] = d_f32ToInt<int8_t>(value + zero_point, rmode);
    } else {
      static_cast<uint8_t *>(output)[idx] = d_f32ToInt<uint8_t>(value + zero_point, rmode);
    }
  }
}

__global__ void g_int8ScaleToF32(void *input, float *output, float scale,
                                 int size, bool sign, float zero_point) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    // Convert int8 to float32 and scale
    if (sign) {
      output[idx] = (static_cast<float>(((int8_t *)input)[idx]) - zero_point) * scale;
    } else {
      output[idx] = (static_cast<float>(((uint8_t *)input)[idx]) - zero_point) * scale;
    }
  }
}

__global__ void g_int8ScaleToBF16(void *input, uint16_t *output, float scale,
                                  int size, bool sign, float zero_point) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    // Convert int8 to bfloat16 and scale
    float value;
    if (sign) {
      value = (static_cast<float>(((int8_t *)input)[idx]) - zero_point) * d_BF16(scale);
    } else {
      value = (static_cast<float>(((uint8_t *)input)[idx]) - zero_point) * d_BF16(scale);
    }
    output[idx] = d_BF16Raw(value);
  }
}

__global__ void g_int8ScaleToF16(void *input, uint16_t *output, float scale,
                                 int size, bool sign, float zero_point) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    // Convert int8 to bfloat16 and scale
    float value;
    if (sign) {
      value = (static_cast<float>(((int8_t *)input)[idx]) - zero_point) * d_F16(scale);
    } else {
      value = (static_cast<float>(((uint8_t *)input)[idx]) - zero_point) * d_F16(scale);
    }
    output[idx] = d_F16Raw(value);
  }
}

__global__ void g_int16ScaleToF32(void *input, float *output, float scale,
                                 int size, float zero_point) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    // Convert int16 to f32 and scale
    float value;
    value = (static_cast<float>(((int16_t *)input)[idx]) - zero_point) * scale;
    output[idx] = value;
  }
}

__global__ void g_int16ScaleToBF16(void *input, uint16_t *output, float scale,
                                 int size, float zero_point) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    // Convert int16 to f32 and scale
    float value;
    value = (static_cast<float>(((int16_t *)input)[idx]) - zero_point) * d_BF16(scale);
    output[idx] = d_BF16Raw(value);
  }
}

__global__ void g_int16ScaleToF16(void *input, uint16_t *output, float scale,
                                 int size, float zero_point) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    // Convert int16 to f32 and scale
    float value;
    value = (static_cast<float>(((int16_t *)input)[idx]) - zero_point) * d_F16(scale);
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
                          int h2, int w2, int multiplier, int rshift,
                          bool relu, int a_zp, int b_zp, int o_zp,
                          requant_mode_t qmode, rounding_mode_t rmode, bool is_cv18xx) {
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
    int64_t value = (static_cast<int64_t>(a[idx_a]) - a_zp) * (static_cast<int64_t>(b[idx_b]) - b_zp);
    if (qmode == MultiplierShift) {
      if (is_cv18xx) {
        value = d_f32ToInt<int32_t>((float)value * multiplier / (1<<rshift), rmode);
      } else {
        value = Right_Shift_Round(value * multiplier, rshift, rmode);
      }
    } else if (qmode == OnlyShift) {
      value = Right_Shift_Round(value, rshift, rmode);
    } else if (qmode == QDM || qmode == TFLite || qmode == TFLite_LShift) {
      int shift = is_cv18xx ? -rshift : rshift;
      int64_t tmp_value = shift > 0 ? value << shift : value;
      tmp_value = Right_Shift_Round(tmp_value * multiplier, 31, RD_HALF_UP);
      if (value > (1ll << 31) - 1) {
        value = (1ll << 31) - 1;
      } else if (value < -(1ll << 31)) {
        value = -(1ll << 31);
      } else {
        value = Right_Shift_Round(tmp_value, -shift, rmode);
      }
    }
    value += o_zp;
    if (std::is_same<T2, int8_t>::value) {
      int32_t min_ = relu ? o_zp : -128;
      value = max(min_, min(127, (int32_t)value));
      ((int8_t *)out)[idx_out] = static_cast<int8_t>(value);
    } else {
      int32_t min_ = relu ? o_zp : 0;
      value = max(min_, min(255, (int32_t)value));
      ((uint8_t *)out)[idx_out] = static_cast<uint8_t>(value);
    }
  }
}

template <typename T0, typename T1, typename T2>
__global__ void g_add6DInt8(T0 *a, T1 *b, T2 *out, int32_t mul0, int32_t mul1,
                            int shift0, int shift1, bool relu,
                            int i0, int i1, int i2, int i3, int i4, int i5,
                            int j0, int j1, int j2, int j3, int j4, int j5,
                            int o0, int o1, int o2, int o3, int o4, int o5,
                            int a_zp, int b_zp, int out_zp) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_0 = dst_idx / (o1 * o2 * o3 * o4 * o5);
  int idx_1 = dst_idx % (o1 * o2 * o3 * o4 * o5) / (o2 * o3 * o4 * o5);
  int idx_2 = dst_idx % (o2 * o3 * o4 * o5) / (o3 * o4 * o5);
  int idx_3 = dst_idx % (o3 * o4 * o5) / (o4 * o5);
  int idx_4 = dst_idx % (o4 * o5) / o5;
  int idx_5 = dst_idx % o5;
  if (idx_0 < i0 && idx_1 < i1 && idx_2 < i2 && idx_3 < i3 && idx_4 < i4 && idx_5 < i5) {
    int idx_i0 = idx_0 % i0;
    int idx_i1 = idx_1 % i1;
    int idx_i2 = idx_2 % i2;
    int idx_i3 = idx_3 % i3;
    int idx_i4 = idx_4 % i4;
    int idx_i5 = idx_5 % i5;
    int idx_0 = ((((idx_i0 * i1 + idx_i1) * i2 + idx_i2) * i3 + idx_i3) * i4 + idx_i4) * i5 + idx_i5;
    int idx_j0 = idx_0 % j0;
    int idx_j1 = idx_1 % j1;
    int idx_j2 = idx_2 % j2;
    int idx_j3 = idx_3 % j3;
    int idx_j4 = idx_4 % j4;
    int idx_j5 = idx_5 % j5;
    int idx_1 = ((((idx_j0 * j1 + idx_j1) * j2 + idx_j2) * j3 + idx_j3) * j4 + idx_j4) * j5 + idx_j5;
    int32_t a_data = static_cast<int32_t>(a[idx_0] - a_zp) * mul0;
    a_data = (a_data + (1 << (shift0 - 1))) >> shift0;
    int32_t b_data = (static_cast<int32_t>(b[idx_1]) - b_zp) * mul1;
    b_data = (b_data + (1 << (shift1 - 1))) >> shift1;
    a_data = a_data + b_data;
    if (std::is_same<T2, int8_t>::value) {
      int32_t min_ = relu ? out_zp : -128;
      a_data = max(min_, min(127, a_data + out_zp));
      out[dst_idx] = static_cast<int8_t>(a_data);
    } else {
      int32_t min_ = relu ? out_zp : 0;
      a_data = max(min_, min(255, a_data + out_zp));
      out[dst_idx] = static_cast<uint8_t>(a_data);
    }
  }
}

template <typename T0, typename T1, typename T2>
__global__ void g_add6DF32(T0 *a, float scale0, T1 *b, float scale1, T2 *out, bool relu,
                            int i0, int i1, int i2, int i3, int i4, int i5,
                            int j0, int j1, int j2, int j3, int j4, int j5,
                            int o0, int o1, int o2, int o3, int o4, int o5) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_0 = dst_idx / (o1 * o2 * o3 * o4 * o5);
  int idx_1 = dst_idx % (o1 * o2 * o3 * o4 * o5) / (o2 * o3 * o4 * o5);
  int idx_2 = dst_idx % (o2 * o3 * o4 * o5) / (o3 * o4 * o5);
  int idx_3 = dst_idx % (o3 * o4 * o5) / (o4 * o5);
  int idx_4 = dst_idx % (o4 * o5) / o5;
  int idx_5 = dst_idx % o5;
  if (idx_0 < o0 && idx_1 < o1 && idx_2 < o2 && idx_3 < o3 && idx_4 < o4 && idx_5 < o5) {
    int idx_i0 = idx_0 % i0;
    int idx_i1 = idx_1 % i1;
    int idx_i2 = idx_2 % i2;
    int idx_i3 = idx_3 % i3;
    int idx_i4 = idx_4 % i4;
    int idx_i5 = idx_5 % i5;
    int idx_0 = ((((idx_i0 * i1 + idx_i1) * i2 + idx_i2) * i3 + idx_i3) * i4 + idx_i4) * i5 + idx_i5;
    int idx_j0 = idx_0 % j0;
    int idx_j1 = idx_1 % j1;
    int idx_j2 = idx_2 % j2;
    int idx_j3 = idx_3 % j3;
    int idx_j4 = idx_4 % j4;
    int idx_j5 = idx_5 % j5;
    int idx_1 = ((((idx_j0 * j1 + idx_j1) * j2 + idx_j2) * j3 + idx_j3) * j4 + idx_j4) * j5 + idx_j5;
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
                            int on, int oc, int oh, int ow, int a_zp, int b_zp, int out_zp) {
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
    int a_data = a[idx_0] - a_zp;
    int b_data = b[idx_1] - b_zp;
    // half up
    a_data = ((a_data * mul0) + (1 << (shift0 - 1))) >> shift0;
    b_data = ((b_data * mul1) + (1 << (shift1 - 1))) >> shift1;
    if (reverse)
      a_data = b_data - a_data;
    else
      a_data = a_data - b_data;
    a_data += out_zp;
    if (relu)
      a_data = max(out_zp, a_data);
    a_data = max(-128, a_data);
    a_data = min(127, a_data);
    out[dst_idx] = (int8_t)a_data;
  }
}

template <typename T0, typename T1, typename T2>
__global__ void g_mulConst6DF32(T0 *a, T1 b, T2 *out, bool relu, int s0, int s1,
                            int s2, int s3, int s4, int s5) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_0 = dst_idx / (s1 * s2 * s3 * s4 * s5);
  int idx_1 = dst_idx % (s1 * s2 * s3 * s4 * s5) / (s2 * s3 * s4 * s5);
  int idx_2 = dst_idx % (s2 * s3 * s4 * s5) / (s3 * s4 * s5);
  int idx_3 = dst_idx % (s3 * s4 * s5) / (s4 * s5);
  int idx_4 = dst_idx % (s4 * s5) / s5;
  int idx_5 = dst_idx % s5;
  if (idx_0 < s0 && idx_1 < s1 && idx_2 < s2 && idx_3 < s3 && idx_4 < s4 && idx_5 < s5) {
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

template <typename T0, typename T1>
__global__ void g_subConst4DI8(T0 *input, int const_v, T1 *output, bool out_signed,
                               bool do_relu, bool reverse, int multi, int shift,
                               int n, int c, int h, int w, int output_zp) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_n = dst_idx / (c * h * w);
  int idx_c = dst_idx % (c * h * w) / (h * w);
  int idx_h = dst_idx % (h * w) / w;
  int idx_w = dst_idx % w;
  if (idx_w < w && idx_h < h && idx_c < c && idx_n < n) {
    int a_data = (int)input[dst_idx];
    if (reverse)
      a_data = const_v - a_data * multi;
    else
      a_data = a_data * multi - const_v;
    int val = a_data >> shift;
    // using rounding half up
    if (shift > 0) {
      int mant = a_data & ((1ul << shift) - 1);
      if (mant >= (1ul << (shift-1)))
        val += 1;
    }
    a_data = val + output_zp;
    if (do_relu)
      a_data = max(output_zp, a_data);
    if (out_signed) {
      a_data = max(-128, a_data);
      a_data = min(127, a_data);
      output[dst_idx] = (int8_t)a_data;
    } else {
      a_data = max(0, a_data);
      a_data = min(255, a_data);
      output[dst_idx] = (uint8_t)a_data;
    }
  }
}

template <typename T0, typename T1>
__global__ void g_addConstI8(T0 *input, int const_v, T1 *output,
  int multi, int shift, int input_zp, int output_zp, int size, bool do_relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int a_data = (int)input[idx] - input_zp;
    a_data = a_data * multi + const_v;
    if (shift > 0)
      a_data = (a_data + (1 << (shift - 1))) >> shift; // half up
    if (do_relu)
      a_data = max(0, a_data);
    if (std::is_same<T1, int8_t>::value) {
      a_data = max(-128, a_data);
      a_data = min(127, a_data);
      output[idx] = (int8_t)a_data;
    } else {
      a_data = max(0, a_data);
      a_data = min(255, a_data);
      output[idx] = (uint8_t)a_data;
    }
  }
}

template <typename T0, typename T1>
__global__ void g_maxConstI8(T0 *input, int const_v, T1 *output, int multi,
  int shift, int input_zp, int output_zp, int size, bool do_relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int a_data = (int)input[idx] - input_zp;
    a_data = max(a_data * multi, const_v);
    if (shift > 0)
      a_data = (a_data + (1 << (shift - 1))) >> shift; // half up
    if (do_relu)
      a_data = max(0, a_data);
    if (std::is_same<T1, int8_t>::value) {
      a_data = max(-128, a_data);
      a_data = min(127, a_data);
      output[idx] = max((int8_t)a_data, (int8_t)const_v);
    } else {
      a_data = max(0, a_data);
      a_data = min(255, a_data);
      output[idx] = max((uint8_t)a_data, (uint8_t)const_v);
    }
  }
}

template <typename T0, typename T1>
__global__ void g_minConstI8(T0 *input, int const_v, T1 *output, int multi,
  int shift, int input_zp, int output_zp, int size, bool do_relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int a_data = (int)input[idx] - input_zp;
    a_data = min(a_data * multi, const_v);
    if (shift > 0)
      a_data = (a_data + (1 << (shift - 1))) >> shift; // half up
    if (do_relu)
      a_data = max(0, a_data);
    if (std::is_same<T1, int8_t>::value) {
      a_data = max(-128, a_data);
      a_data = min(127, a_data);
      output[idx] = min((int8_t)a_data, (int8_t)const_v);
    } else {
      a_data = max(0, a_data);
      a_data = min(255, a_data);
      output[idx] = min((uint8_t)a_data, (uint8_t)const_v);
    }
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

__global__ void g_divConst4DF32(float *input, float const_v, float *output,
                                bool do_relu, bool reverse, int n, int c,
                                int h, int w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n * c * h * w) {
    float val = reverse ? (const_v / input[idx]) : (input[idx] / const_v);
    if (do_relu && val < 0.0f) {
      val = 0.0f;
    }
    output[idx] = val;
  }
}

template <typename T0, typename T1, typename T2>

__global__ void g_divMDF32(T0 *input0, T1 *input1, T2 *output,
                           int64_t *shape0, int64_t *shape1, int64_t *shape2,
                           int64_t *strides0, int64_t *strides1, int64_t *strides2,
                           int dims, int num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    int idx0 = 0, idx1 = 0;
    int tmp = idx;
    for (int i = dims - 1; i >= 0; --i) {
      int coord = tmp % shape2[i];
      tmp /= shape2[i];
      idx0 += (coord % shape0[i]) * strides0[i];
      idx1 += (coord % shape1[i]) * strides1[i];
    }
    float a_data = input0[idx0];
    float b_data = input1[idx1];
    output[idx] = a_data / b_data;
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
                        int tbytes, float pad_value) {
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
      d_setValue(output, out_idx, tbytes, pad_value);
    }
  }
}

__global__ void g_pad4D(void *input, void *output, int n, int c, int h, int w,
                        int pad_h_t, int pad_h_b, int pad_w_l, int pad_w_r,
                        int tbytes, bool is_edge) {
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
      if (is_edge) {
        int idx_in_h = min(max(idx_h - pad_h_t, 0), h - 1);
        int idx_in_w = min(max(idx_w - pad_w_l, 0), w - 1);
        int in_idx = ((idx_n * c + idx_c) * h + idx_in_h) * w + idx_in_w;
        d_copyElement(input, in_idx, output, out_idx, tbytes);
      } else { // reflect padding
        int idx_in_h = idx_h - pad_h_t;
        int idx_in_w = idx_w - pad_w_l;
        if (idx_in_h < 0)
          idx_in_h = -idx_in_h;
        else if (idx_in_h >= h)
          idx_in_h = 2 * h - idx_in_h - 2;
        if (idx_in_w < 0)
          idx_in_w = -idx_in_w;
        else if (idx_in_w >= w)
          idx_in_w = 2 * w - idx_in_w - 2;
        int in_idx = ((idx_n * c + idx_c) * h + idx_in_h) * w + idx_in_w;
        d_copyElement(input, in_idx, output, out_idx, tbytes);
      }
    }
  }
}

__global__ void g_insertZero4D(void *input, void *output, int n, int c, int h, int w,
                             int ins_h, int ins_w, int tbytes) {
  int oh = h + (h - 1) * ins_h;
  int ow = w + (w - 1) * ins_w;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (n * c * oh * ow)) {
    int idx_n = idx / (c * oh * ow);
    int idx_c = idx % (c * oh * ow) / (oh * ow);
    int idx_h = idx % (oh * ow) / ow;
    int idx_w = idx % ow;
    int out_idx = ((idx_n * c + idx_c) * oh + idx_h) * ow + idx_w;
    if (idx_h % (ins_h + 1) == 0 && idx_w % (ins_w + 1) == 0) {
      int idx_in_h = idx_h / (ins_h + 1);
      int idx_in_w = idx_w / (ins_w + 1);
      int in_idx = ((idx_n * c + idx_c) * h + idx_in_h) * w + idx_in_w;
      d_copyElement(input, in_idx, output, out_idx, tbytes);
    } else {
      d_setValue(output, out_idx, tbytes, 0);
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

__global__ void g_tile(void *src, void *dst, int64_t *in_shape, int64_t *out_shape, int num_dims, int tbytes) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_out = 1;
  for (int i = 0; i < num_dims; i++) {
    num_out *= out_shape[i];
  }
  if (dst_idx < num_out) {
    int src_idx = 0;
    int tmp = dst_idx;
    int src_stride = 1;
    for (int i = num_dims - 1; i >= 0; i--) {
      int out_dim_idx = tmp % out_shape[i];
      int in_dim_idx = out_dim_idx % in_shape[i];
      src_idx += in_dim_idx * src_stride;
      src_stride *= in_shape[i];
      tmp /= out_shape[i];
    }
    d_copyElement(src, src_idx, dst, dst_idx, tbytes);
  }
}

__global__ void g_ABSVAL(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    float input_i = input[i];
    output[i] = fabsf(input_i);
  }
}

__global__ void g_CEIL(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    float input_i = input[i];
    output[i] = ceilf(input_i);
  }
}

__global__ void g_ERF(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = erf(input_i);
  }
}

__global__ void g_EXP(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = exp(input_i);
  }
}

__global__ void g_LN(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = log(input_i);
  }
}

__global__ void g_LOG2(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = log2(input_i);
  }
}

__global__ void g_SQRT(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = sqrt(input_i);
  }
}

__global__ void g_RSQRT(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = rsqrt(input_i);
  }
}

__global__ void g_SQUARE(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = input_i * input_i;
  }
}

__global__ void g_SILU(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    double sigmoid = 1.0 / (1.0 + exp(-input_i));
    output[i] = input_i * sigmoid;
  }
}

__global__ void g_SIGMOID(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = 1.0 / (1.0 + exp(-input_i));
  }
}

__global__ void g_LOG_SIGMOID(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = log(1.0 + exp(-input_i));
  }
}

__global__ void g_ARCCOS(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = acos(input_i);
  }
}

__global__ void g_ARCTANH(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = atanh(input_i);
  }
}

__global__ void g_TAN(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = tan(input_i);
  }
}

__global__ void g_TANH(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = tanh(input_i);
  }
}

__global__ void g_GELU(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    double value = 0.5*input_i*(1.0+erf(input_i/sqrt(2.0)));
    output[i] = value;
  }
}

__global__ void g_TGELU(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = 0.5 * input_i * (1.0 + tanh(input_i * 0.7978845608 * (1.0 + 0.044715 * input_i * input_i)));
  }
}

__global__ void g_QGELU(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    double sigmoid = 1.0 / (1.0 + exp(-1.702 * input_i));
    output[i] = input_i * sigmoid;
  }
}

__global__ void g_SOFT_PLUS(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = input_i > 20 ? input_i : log(1.0 + exp(input_i));
  }
}

__global__ void g_FLOOR(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    float input_i = input[i];
    output[i] = floorf(input_i);
  }
}

__global__ void g_SOFT_SIGN(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = input_i / (1.0 + fabs(input_i));
  }
}

__global__ void g_MISH(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    double softplus = log(1.0 + exp(input_i));
    double tanh_sp = 2 / (1 + exp(-2 * softplus)) - 1;
    output[i] = input_i * tanh_sp;
  }
}

__global__ void g_COS(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = cos(input_i);
  }
}

__global__ void g_COSH(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = cosh(input_i);
  }
}

__global__ void g_SIN(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = sin(input_i);
  }
}

__global__ void g_SINH(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = sinh(input_i);
  }
}

__global__ void g_ROUND(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    float input_i = input[i];
    output[i] = roundf(input_i);
  }
}

__global__ void g_SIGN(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    float input_i = input[i];
    output[i] = (input_i > 0) - (input_i < 0);
  }
}

__global__ void g_HSWISH(float* input, float *output, int num) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    double hsigmoid = max(0.0, min(1.0, (input_i + 3.0) / 6.0));
    output[i] = input_i * hsigmoid;
  }
}

__global__ void g_SWISH(float* input, float *output, int num, double beta) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    double sigmoid = 1.0 / (1.0 + exp(-input_i * beta));
    output[i] = input_i * sigmoid;
  }
}

__global__ void g_ELU(float* input, float *output, int num, float alpha) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = input_i >= 0 ? input_i : alpha * (exp(input_i) - 1);
  }
}

__global__ void g_HSIGMOID(float* input, float *output, int num, double alpha, double beta) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<num){
    double input_i = input[i];
    output[i] = max(0.0, min(1.0, alpha * input_i + beta));
  }
}

__global__ void g_RELU(float* input, float *output, int num, double min_val, double max_val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < num){
    float input_i = max(input[i], (float)min_val);
    if (max_val > 0)
      input_i = min(input_i, (float)max_val);
    output[i] = input_i;
  }
}

__global__ void g_CLIP(float* input, float *output, int num, double min_val, double max_val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < num){
    output[i] = min(max(input[i], (float)min_val), (float)max_val);
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

__global__ void g_mmF32(float *A, float *B, float *C, int m, int k, int n,
    bool left_transpose, bool right_transpose, bool output_transpose,
    float left_zp, float right_zp) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_m = idx / n;
  int idx_n = idx % n;
  if (idx_m < m && idx_n < n) {
    float sum = 0.0;
    for (int i = 0; i < k; i++) {
      float left_val = left_transpose ? A[i * m + idx_m] : A[idx_m * k + i];
      float right_val = right_transpose ? B[idx_n * k + i] : B[i * n + idx_n];
      sum += (left_val - left_zp) * (right_val - right_zp);
    }
    int c_idx = output_transpose ? idx_n * m + idx_m : idx_m * n + idx_n;
    C[c_idx] = sum;
    // C[idx_m * n + idx_n] = sum;
    // C[idx_m * n + idx_n] = sum;
  }
}

template <typename T0, typename T1>
__global__ void g_mmInt8(T0 *A, T1 *B, int32_t *C, int m, int k, int n,
    bool left_transpose, bool right_transpose, bool output_transpose,
    int32_t left_zp, int32_t right_zp) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_m = idx / n;
  int idx_n = idx % n;
  if (idx_m < m && idx_n < n) {
    int32_t sum = 0;
    // if (right_transpose) {
    //   for (int i = 0; i < k; i++) {
    //     sum += ((int32_t)A[idx_m * k + i]) * ((int32_t)B[idx_n * k + i]);
    //   }
    // } else {
    //   for (int i = 0; i < k; i++) {
    //     sum += ((int32_t)A[idx_m * k + i]) * ((int32_t)B[i * n + idx_n]);
    //   }
    // }
    // C[idx_m * n + idx_n] = sum;
    for (int i = 0; i < k; i++) {
      int32_t left_value = left_transpose ? A[i * m + idx_m] : A[idx_m * k + i];
      int32_t right_value = right_transpose ? B[idx_n * k + i] : B[i * n + idx_n];
      sum += (left_value - left_zp) * (right_value - right_zp);
    }
    int c_idx = output_transpose ? idx_n * m + idx_m : idx_m * n + idx_n;
    C[c_idx] = sum;
  }
}

__global__ void g_mmIntDynamicQuantize(float *input, float *right, float *output,
    float *input_max_values, float *right_max_values, int m, int k, int n,
    bool left_transpose, bool right_transpose, bool output_transpose,
    int q_group_size, int quant_bits
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_m = idx / n;
  int idx_n = idx % n;
  if (idx_m >= m || idx_n >= n) return;
  float int_max = (1 << (quant_bits - 1)) - 1;
  int num_groups = (k + q_group_size - 1) / q_group_size;
  float sum = 0.0f;
  for (int g = 0; g < num_groups; g++) {
    int k_start = g * q_group_size;
    int k_end = min(k_start + q_group_size, k);

    float max_A = input_max_values[idx_m * num_groups + g];
    float max_B = right_max_values[idx_n * num_groups + g];

    float group_dot = 0;
    for (int i = k_start; i < k_end; i++) {
      float a_val = left_transpose ? input[i * m + idx_m] : input[idx_m * k + i];
      float b_val = right_transpose ? right[idx_n * k + i] : right[i * n + idx_n];

      float a_int = roundf(a_val / max_A * int_max);
      float b_int = roundf(b_val / max_B * int_max);
      group_dot += a_int * b_int;
    }
    sum += group_dot * (max_A / int_max) * (max_B / int_max);
  }
  int c_idx = output_transpose ? idx_n * m + idx_m : idx_m * n + idx_n;
  output[c_idx] = sum;
}

__global__ void g_mmIntDynamicQuantize(float *input, uint8_t *weight, float *output,
  float *input_max_values, float *weight_scale, float *weight_zp, int m, int k,
  int n, int group_size, int quant_bits) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_m = idx / n;
  int idx_n = idx % n;
  if (idx_m >= m || idx_n >= n) return;
  float int_max = (1 << (quant_bits - 1)) - 1;
  int num_groups = (k + group_size - 1) / group_size;
  float sum = 0.0f;
  for (int g = 0; g < num_groups; g++) {
    int k_start = g * group_size;
    int k_end = min(k_start + group_size, k);

    float input_max = input_max_values[idx_m * num_groups + g];
    float w_scale = weight_scale[idx_n * num_groups + g];
    float w_zp = weight_zp[idx_n * num_groups + g];

    float group_dot = 0;
    for (int k_idx = k_start; k_idx < k_end; k_idx++) {
      float a_val = input[idx_m * k + k_idx];
      float a_int = roundf(a_val * int_max / input_max);

      float w_int;
      if (quant_bits == 8) {
        w_int = (float)(int32_t)weight[idx_n * k + k_idx];
      } else { // quant_bits == 4
        int byte_idx = idx_n * ((k + 1) >> 1) + (k_idx >> 1);
        uint8_t byte_val = weight[byte_idx];
        w_int = (k_idx & 1) ? (float)(int32_t)(byte_val >> 4)
                            : (float)(int32_t)(byte_val & 0x0F);
      }
      w_int -= w_zp;
      group_dot += a_int * w_int;
    }
    sum += group_dot * (input_max / int_max) * w_scale;
  }

  output[idx_m * n + idx_n] = sum;
}

__global__ void g_mmF8DynamicQuantize(float *input, float *right, float *output,
    float *input_max_values, float *right_max_values, int m, int k, int n,
    bool left_transpose, bool right_transpose, bool output_transpose, int q_group_size
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_m = idx / n;
  int idx_n = idx % n;
  if (idx_m >= m || idx_n >= n) return;

  int num_groups = (k + q_group_size - 1) / q_group_size;
  float sum = 0.0f;
  for (int g = 0; g < num_groups; g++) {
    int k_start = g * q_group_size;
    int k_end = min(k_start + q_group_size, k);

    float max_A = input_max_values[idx_m * num_groups + g];
    float max_B = right_max_values[idx_n * num_groups + g];
    float step_A = max_A / 448.0f;
    float step_B = max_B / 448.0f;

    float group_dot = 0.0f;
    for (int i = k_start; i < k_end; i++) {
      float a_val = left_transpose ? input[i * m + idx_m] : input[idx_m * k + i];
      float b_val = right_transpose ? right[idx_n * k + i] : right[i * n + idx_n];

      uint8_t a_fp8 = fp32_to_fp8(a_val / step_A, false);
      uint8_t b_fp8 = fp32_to_fp8(b_val / step_B, false);
      float a_fp = f8_to_fp32(a_fp8, 1, false);
      float b_fp = f8_to_fp32(b_fp8, 1, false);
      group_dot += a_fp * b_fp;
    }
    sum += group_dot * step_A * step_B;
  }

  int c_idx = output_transpose ? idx_n * m + idx_m : idx_m * n + idx_n;
  output[c_idx] = sum;
}

__global__ void g_mmF8DynamicQuantize(float *input, uint8_t *weight, float *output,
  float *input_max_values, float *weight_scale, float *weight_zp, int m, int k,
  int n, int group_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_m = idx / n;
  int idx_n = idx % n;
  if (idx_m >= m || idx_n >= n) return;

  int num_groups = (k + group_size - 1) / group_size;
  float sum = 0.0f;
  for (int g = 0; g < num_groups; g++) {
    int k_start = g * group_size;
    int k_end = min(k_start + group_size, k);

    float max_A = input_max_values[idx_m * num_groups + g];
    float w_scale = weight_scale[idx_n * num_groups + g];

    float group_dot = 0.0f;
    for (int k_idx = k_start; k_idx < k_end; k_idx++) {
      float a_val = input[idx_m * k + k_idx];
      uint8_t a_fp8 = fp32_to_fp8(a_val * 448 / max_A, false);
      float a_fp = f8_to_fp32(a_fp8, 1, false);

      uint8_t w_fp8 = weight[idx_n * k + k_idx];
      float w_fp = f8_to_fp32(w_fp8, 1, false);
      group_dot += a_fp * w_fp;
    }
    sum += group_dot * (max_A / 448.0f) * w_scale;
  }

  output[idx_m * n + idx_n] = sum;
}

__global__ void g_mmF4DynamicQuantize(float *input, float *right, float *output,
    float *input_max_values, float *right_max_values, int m, int k, int n,
    bool left_transpose, bool right_transpose, bool output_transpose, int q_group_size
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_m = idx / n;
  int idx_n = idx % n;
  if (idx_m >= m || idx_n >= n) return;

  int num_groups = (k + q_group_size - 1) / q_group_size;
  float sum = 0.0f;
  for (int g = 0; g < num_groups; g++) {
    int k_start = g * q_group_size;
    int k_end = min(k_start + q_group_size, k);

    float max_A = input_max_values[idx_m * num_groups + g];
    float max_B = right_max_values[idx_n * num_groups + g];
    float group_dot = 0.0f;
    for (int i = k_start; i < k_end; i++) {
      float a_val = left_transpose ? input[i * m + idx_m] : input[idx_m * k + i];
      float b_val = right_transpose ? right[idx_n * k + i] : right[i * n + idx_n];
      group_dot += f32_to_f4e2m1(a_val, max_A / 6.0f) *
                   f32_to_f4e2m1(b_val, max_B / 6.0f);
    }
    sum += group_dot * (max_A / 6.0f) * (max_B / 6.0f);
  }

  int c_idx = output_transpose ? idx_n * m + idx_m : idx_m * n + idx_n;
  output[c_idx] = sum;
}

__global__ void g_mmF4DynamicQuantize(float *input, uint8_t *weight, float *output,
  float *input_max_values, float *weight_scale, float *weight_zp, int m, int k,
  int n, int group_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_m = idx / n;
  int idx_n = idx % n;
  if (idx_m >= m || idx_n >= n) return;

  int num_groups = (k + group_size - 1) / group_size;
  float sum = 0.0f;
  for (int g = 0; g < num_groups; g++) {
    int k_start = g * group_size;
    int k_end = min(k_start + group_size, k);

    float max_A = input_max_values[idx_m * num_groups + g];
    float w_scale = weight_scale[idx_n * num_groups + g];

    float group_dot = 0.0f;
    for (int k_idx = k_start; k_idx < k_end; k_idx++) {
      float a_val = input[idx_m * k + k_idx];
      float a_f4 = f32_to_f4e2m1(a_val, max_A / 6.0f);

      int byte_idx = idx_n * ((k + 1) >> 1) + (k_idx >> 1);
      uint8_t packed = weight[byte_idx];
      uint8_t nibble = (k_idx & 1) ? (packed >> 4) : (packed & 0x0F);
      int sign = nibble >> 3;
      int exp  = (nibble >> 1) & 0x3;
      int mant = nibble & 0x1;
      float w_val = (exp == 0) ? (mant ? 0.5f : 0.0f)
                               : (1.0f + mant * 0.5f) * (float)(1 << (exp - 1));
      float w_f4 = sign ? -w_val : w_val;

      group_dot += a_f4 * w_f4;
    }
    sum += group_dot * (max_A / 6.0f) * w_scale;
  }

  output[idx_m * n + idx_n] = sum;
}

__global__ void g_mmMXF4DynamicQuantize(float *input, float *right, float *output,
    float *input_max_values, float *right_max_values, int m, int k, int n,
    bool left_transpose, bool right_transpose, bool output_transpose, int q_group_size
  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_m = idx / n;
  int idx_n = idx % n;
  if (idx_m >= m || idx_n >= n) return;

  int num_groups = (k + q_group_size - 1) / q_group_size;
  float sum = 0.0f;
  for (int g = 0; g < num_groups; g++) {
    int k_start = g * q_group_size;
    int k_end = min(k_start + q_group_size, k);

    float max_A = input_max_values[idx_m * num_groups + g];
    float max_B = right_max_values[idx_n * num_groups + g];
    float step_A = max_A / 6.0f;
    float step_B = max_B / 6.0f;
    step_A = powf(2.0f, ceilf(log2f(step_A)));
    step_B = powf(2.0f, ceilf(log2f(step_B)));
    float group_dot = 0.0f;
    for (int i = k_start; i < k_end; i++) {
      float a_val = left_transpose ? input[i * m + idx_m] : input[idx_m * k + i];
      float b_val = right_transpose ? right[idx_n * k + i] : right[i * n + idx_n];
      group_dot += f32_to_f4e2m1(a_val, step_A) *
                   f32_to_f4e2m1(b_val, step_B);
    }
    sum += group_dot * step_A * step_B;
  }

  int c_idx = output_transpose ? idx_n * m + idx_m : idx_m * n + idx_n;
  output[c_idx] = sum;
}

__global__ void g_mmMXF4DynamicQuantize(float *input, uint8_t *weight, float *output,
  float *input_max_values, float *weight_scale, float *weight_zp, int m, int k,
  int n, int group_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_m = idx / n;
  int idx_n = idx % n;
  if (idx_m >= m || idx_n >= n) return;

  int num_groups = (k + group_size - 1) / group_size;
  float sum = 0.0f;
  for (int g = 0; g < num_groups; g++) {
    int k_start = g * group_size;
    int k_end = min(k_start + group_size, k);

    float max_A = input_max_values[idx_m * num_groups + g];
    float w_scale = weight_scale[idx_n * num_groups + g];
    float step_A = powf(2.0f, ceilf(log2f(max_A / 6.0f)));

    for (int k_idx = k_start; k_idx < k_end; k_idx++) {
      float a_val = input[idx_m * k + k_idx];
      float a_fp = f32_to_f4e2m1(a_val, step_A);

      int byte_idx = idx_n * ((k + 1) >> 1) + (k_idx >> 1);
      uint8_t packed = weight[byte_idx];
      uint8_t nibble = (k_idx & 1) ? (packed >> 4) : (packed & 0x0F);
      int sign = nibble >> 3;
      int exp  = (nibble >> 1) & 0x3;
      int mant = nibble & 0x1;
      float w_val = (exp == 0) ? (mant ? 0.5f : 0.0f)
                               : (1.0f + mant * 0.5f) * (float)(1 << (exp - 1));
      float w_fp = (sign ? -w_val : w_val) * w_scale;

      sum += a_fp * w_fp * step_A;
    }
  }

  output[idx_m * n + idx_n] = sum;
}

__global__ void g_groupAbsMax(float *input, float *output, int m, int n, int group_size, bool transpose) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_groups = (n + group_size - 1) / group_size;
  if (idx >= m * num_groups) return;

  int row = idx / num_groups;
  int group = idx % num_groups;
  int start = group * group_size;
  int end = min(start + group_size, n);

  float max_val = 1e-8f;
  for (int j = start; j < end; j++) {
    float val = transpose ? fabsf(input[j * m + row]) : fabsf(input[row * n + j]);
    max_val = fmaxf(max_val, val);
  }
  output[idx] = max_val;
}

__global__ void g_dequantA16MMWeight(int8_t *input, float *output, float *scale,
  float *zp, int num, int group_size, int bits) {
  int group_num = (num + group_size - 1) / group_size;
  int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (group_idx >= group_num) return;
  float s = scale[group_idx];
  float z = zp[group_idx];
  if (bits == 8) {
    int start = group_idx * group_size;
    int end = min(start + group_size, num);
    for (int i = start; i < end; i++) {
      int8_t val = input[i];
      output[i] = (val - z) * s;
    }
  } else if (bits == 4) {
    int start = group_idx * group_size / 2;
    int end = min((group_idx + 1) * group_size / 2, (num + 1) / 2);
    for (int i = start; i < end; i++) {
      uint8_t val = ((uint8_t *)input)[i];
      int8_t high = val >> 4;
      int8_t low = val & 0x0F;
      output[i * 2] = (low - z) * s;
      if (i * 2 + 1 < num) {
        output[i * 2 + 1] = (high - z) * s;
      }
    }
  }
}

__global__ void g_requantInt8Perchannel(int32_t *input, void *output,
                                        int32_t *multipliers, int32_t *shifts,
                                        int n, int c, int h, int w,
                                        bool out_sign, bool qdm, bool relu,
                                        int32_t zero_point) {
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
      int32_t min_ = relu ? zero_point : -128;
      value = max(min_, min(127, value + zero_point));
      ((int8_t *)output)[idx] = static_cast<int8_t>(value);
    } else {
      int32_t min_ = relu ? zero_point : 0;
      value = max(min_, min(255, value + zero_point));
      ((uint8_t *)output)[idx] = static_cast<uint8_t>(value);
    }
  }
}

__global__ void g_requantInt8Perchannel(int32_t *input, void *output,
                                        int32_t *multipliers, int32_t *shifts,
                                        int n, int c, int h, int w,
                                        bool out_sign, bool relu,
                                        int32_t zero_point, bool is_cv18xx,
                                        requant_mode_t qmode,
                                        rounding_mode_t rmode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (n * c * h * w)) {
    int idx_c = idx % (c * h * w) / (h * w);
    int32_t value;
    if (qmode == MultiplierShift) {
      if (is_cv18xx) {
        value = d_f32ToInt<int32_t>((float)input[idx]*multipliers[idx_c]/(1<<shifts[idx_c]), rmode);
      } else {
        value = Right_Shift_Round((int64_t)input[idx]*multipliers[idx_c], shifts[idx_c], rmode);
      }
    } else if (qmode == OnlyShift) {
      value = Right_Shift_Round((int64_t)input[idx], shifts[idx_c], rmode);
    } else if (qmode == QDM || qmode == TFLite || qmode == TFLite_LShift) {
      int shift = shifts[idx_c];
      if (is_cv18xx) {
        shift = -shifts[idx_c];
      }
      int64_t tmp_value = shift > 0 ? input[idx] << shift : input[idx];
      tmp_value = Right_Shift_Round(tmp_value * multipliers[idx_c], 31, RD_HALF_UP);
      if (value > (1ll << 31) - 1) {
        value = (1ll << 31) - 1;
      } else if (value < -(1ll << 31)) {
        value = -(1ll << 31);
      } else {
        value = Right_Shift_Round(tmp_value, -shift, rmode);
      }
    }
    if (out_sign) {
      int32_t min_ = relu ? zero_point : -128;
      value = max(min_, min(127, value + zero_point));
      ((int8_t *)output)[idx] = static_cast<int8_t>(value);
    } else {
      int32_t min_ = relu ? zero_point : 0;
      value = max(min_, min(255, value + zero_point));
      ((uint8_t *)output)[idx] = static_cast<uint8_t>(value);
    }
  }
}

__global__ void g_requantInt8Perchannel(int32_t *input, void *output,
                                        int32_t *multipliers, int32_t *shifts,
                                        int n, int c, int h, int w,
                                        bool out_sign, bool relu,
                                        int32_t* zero_points, bool is_cv18xx,
                                        requant_mode_t qmode,
                                        rounding_mode_t rmode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (n * c * h * w)) {
    int idx_c = idx % (c * h * w) / (h * w);
    int32_t value;
    int32_t zero_point = zero_points[idx_c];
    if (qmode == MultiplierShift) {
      if (is_cv18xx) {
        value = d_f32ToInt<int32_t>((float)input[idx]*multipliers[idx_c]/(1<<shifts[idx_c]), rmode);
      } else {
        value = Right_Shift_Round((int64_t)input[idx]*multipliers[idx_c], shifts[idx_c], rmode);
      }
    } else if (qmode == OnlyShift) {
      value = Right_Shift_Round((int64_t)input[idx], shifts[idx_c], rmode);
    } else if (qmode == QDM || qmode == TFLite || qmode == TFLite_LShift) {
      int shift = shifts[idx_c];
      if (is_cv18xx) {
        shift = -shifts[idx_c];
      }
      int64_t tmp_value = shift > 0 ? input[idx] << shift : input[idx];
      tmp_value = Right_Shift_Round(tmp_value * multipliers[idx_c], 31, RD_HALF_UP);
      if (value > (1ll << 31) - 1) {
        value = (1ll << 31) - 1;
      } else if (value < -(1ll << 31)) {
        value = -(1ll << 31);
      } else {
        value = Right_Shift_Round(tmp_value, -shift, rmode);
      }
    }
    if (out_sign) {
      int32_t min_ = relu ? zero_point : -128;
      value = max(min_, min(127, value + zero_point));
      ((int8_t *)output)[idx] = static_cast<int8_t>(value);
    } else {
      int32_t min_ = relu ? zero_point : 0;
      value = max(min_, min(255, value + zero_point));
      ((uint8_t *)output)[idx] = static_cast<uint8_t>(value);
    }
  }
}

__global__ void g_requantInt8(int32_t *input, void *output, int32_t multiplier,
                              int32_t shift, int num, bool out_sign, bool qdm,
                              bool relu, int32_t zero_point) {
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
      int32_t min_ = relu ? zero_point : -128;
      value = max(min_, min(127, value + zero_point));
      ((int8_t *)output)[idx] = static_cast<int8_t>(value);
    } else {
      int32_t min_ = relu ? zero_point : 0;
      value = max(min_, min(255, value + zero_point));
      ((uint8_t *)output)[idx] = static_cast<uint8_t>(value);
    }
  }
}

__global__ void g_requantInt16(int32_t *input, void *output, int32_t multiplier,
                              int32_t shift, int num, bool relu, int32_t zero_point) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    int32_t value;
    // half up
    int64_t data =
        static_cast<int64_t>(input[idx]) * static_cast<int64_t>(multiplier);
    int64_t round = 1ll << (shift - 1);
    data = (data + round) >> shift;
    value = static_cast<int32_t>(data) + zero_point;
    int32_t min_ = relu ? zero_point : -32768;
    value = max(min_, min(32767, value));
    ((int16_t *)output)[idx] = static_cast<int16_t>(value);
  }
}

__global__ void g_requantInt16Perchannel(int32_t *input, void *output,
                                        int32_t *multipliers, int32_t *shifts,
                                        int n, int c, int h, int w, bool relu,
                                        int32_t zero_point) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (n * c * h * w)) {
    int idx_c = idx % (c * h * w) / (h * w);
    int32_t value;
    // half up
    int64_t data = static_cast<int64_t>(input[idx]) *
                    static_cast<int64_t>(multipliers[idx_c]);
    int64_t round = (int64_t)(1ll << (shifts[idx_c] - 1));
    data = (data + round) >> shifts[idx_c];
    value = static_cast<int32_t>(data) + zero_point;
    int32_t min_ = relu ? zero_point : -32768;
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
                                        float scale, int s0, int s1, int s2, int s3, int s4, int s5, bool relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (s0 * s1 * s2 * s3 * s4 * s5)) {
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
                           int size, int input_zp, int output_zp) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int32_t value = (static_cast<int32_t>(input[idx]) - input_zp) * multiplier;
    value = (value + (1 << (shift - 1))) >> shift; // half up
    value += output_zp;
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
      i_value = d_f32ToInt<int32_t>(value, RD_HALF_TO_EVEN);
    } else if (rmode == RD_HALF_AWAY_FROM_ZERO) {
      i_value = round(value);
    } else { // default round half up
      i_value = floor(value + 0.5f);
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
__global__ void g_mulShiftDouble(float *input, T* output,
                                double multiplier, double shift, int size, rounding_mode_t rmode){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float value = static_cast<double>(input[idx]) * multiplier + shift;
    int i_value = 0;
    if (rmode == RD_HALF_TO_EVEN) {
      i_value = d_f32ToInt<int32_t>(value, RD_HALF_TO_EVEN); /// not implemented half to even
    } else if (rmode == RD_HALF_AWAY_FROM_ZERO) {
      i_value = round(value);
    } else { // default round half up
      i_value = floor(value + 0.5f);
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
__global__ void g_intToBF16(T *input, uint16_t *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = d_BF16Raw(static_cast<float>(input[idx]));
  }
}

template <typename T>
__global__ void g_intToF16(T *input, uint16_t *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = d_F16Raw(static_cast<float>(input[idx]));
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
    output[idx] = d_BF16Raw(input[idx], rmode);
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

__global__ void g_f16(float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = d_RawF16(d_F16Raw(input[idx]));
  }
}

__global__ void g_bf16(float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = d_RawBF16(d_BF16Raw(input[idx]));
  }
}

template <typename T>
__global__ void g_bf16ToInt(uint16_t *input, T *output, int size, rounding_mode_t rmode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float f32_value = d_RawBF16(input[idx]);
    output[idx] = d_f32ToInt<T>(f32_value, rmode);
  }
}

template <typename T>
__global__ void g_f16ToInt(uint16_t *input, T *output, int size, rounding_mode_t rmode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float f32_value = d_RawF16(input[idx]);
    output[idx] = d_f32ToInt<T>(f32_value, rmode);
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

template <typename T> __global__ void g_doRelu(T *data, int size, int zero_point = 0) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    data[idx] = max(static_cast<T>(zero_point), data[idx]);
  }
}

__global__ void g_doReluF16(uint16_t *data, int size, int zero_point = 0) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    data[idx] = (data[idx] & 0x8000) ? zero_point : data[idx];
  }
}

__global__ void g_doReluF8(uint8_t *data, int size, uint8_t zero_point = 0) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    data[idx] = (data[idx] & 0x80) ? zero_point : data[idx];
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
                          int axis_dim, int inner_dim, bool log) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int outer_idx = idx / (axis_dim * inner_dim);
  int axis_idx = idx % (axis_dim * inner_dim) / inner_dim;
  int inner_idx = idx % inner_dim;
  if (inner_idx < inner_dim && outer_idx < outer_dim && axis_idx < axis_dim) {
    int sub_idx = outer_idx * inner_dim + inner_idx;
    T val = input[idx] * mul[sub_idx];
    output[idx] = log ? logf(val) : val;
  }
}

__global__ void g_mulAxisBF16(uint16_t *input, uint16_t *mul, uint16_t *output,
                              int outer_dim, int axis_dim, int inner_dim, bool log) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int outer_idx = idx / (axis_dim * inner_dim);
  int axis_idx = idx % (axis_dim * inner_dim) / inner_dim;
  int inner_idx = idx % inner_dim;
  if (inner_idx < inner_dim && outer_idx < outer_dim && axis_idx < axis_dim) {
    int mul_idx = outer_idx * inner_dim + inner_idx;
    float out = d_RawBF16(input[idx]) * d_RawBF16(mul[mul_idx]);
    output[idx] = d_BF16Raw(log ? logf(out) : out);
  }
}

__global__ void g_layerNorm(float *input, float *output, int outer_dim,
                              int inner_dim, float *weight, float *bias, float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (outer_dim)) {
    float *base_ptr = input + idx * inner_dim;
    float sum = 0.0f;
    for (int inner_idx = 0; inner_idx < inner_dim; inner_idx ++) {
      float val = base_ptr[inner_idx];
      sum += val;
    }
    float mean = sum / inner_dim;
    float rstd = 0.0f;
    for (int inner_idx = 0; inner_idx < inner_dim; inner_idx ++) {
      const float diff = base_ptr[inner_idx] - mean;
      rstd += diff * diff;
    }
    rstd = rstd / inner_dim + eps;
    float inv_std = 1.0f / sqrtf(rstd);
    for (int inner_idx = 0; inner_idx < inner_dim; inner_idx ++) {
      float norm = (base_ptr[inner_idx] - mean) * inv_std;
      if (weight != nullptr)
        norm = norm * weight[inner_idx];
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
    float *base_ptr = input + idx * inner_dim;
    float mean = 0.0f;
    float scale = d_BF16(1.0f / inner_dim);
    for (int inner_idx = 0; inner_idx < inner_dim; inner_idx ++) {
      float val = base_ptr[inner_idx] * scale;
      mean += val;
    }
    mean = d_BF16(mean);
    float rstd = 0.0f;
    for (int inner_idx = 0; inner_idx < inner_dim; inner_idx ++) {
      float diff = d_BF16(base_ptr[inner_idx] - mean);
      rstd += d_BF16(d_BF16(diff * diff) * scale);
    }
    rstd = d_BF16(rstd + eps);
    float inv_std = d_BF16(1.0f / d_BF16(sqrtf(rstd)));
    for (int inner_idx = 0; inner_idx < inner_dim; inner_idx ++) {
      float val = base_ptr[inner_idx];
      float norm = d_BF16(d_BF16(val - mean) * inv_std);
      if (weight != nullptr)
        norm = d_BF16(norm * weight[inner_idx]);
      if (bias != nullptr)
        norm = d_BF16(norm + bias[inner_idx]);
      output[idx * inner_dim + inner_idx] = d_BF16(norm);
    }
  }
}

__global__ void g_layerNormBF16(float *input, float *output, int outer_dim,
                              int inner_dim, float *weight, float *bias,
                              float *table, float *mtable, float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (outer_dim)) {
    float *base_ptr = input + idx * inner_dim;
    float mean = 0.0f;
    float scale = d_BF16(1.0f / inner_dim);
    for (int inner_idx = 0; inner_idx < inner_dim; inner_idx ++) {
      float val = base_ptr[inner_idx] * scale;
      mean += val;
    }
    mean = d_BF16(mean);
    float rstd = 0.0f;
    for (int inner_idx = 0; inner_idx < inner_dim; inner_idx ++) {
      float diff = d_BF16(base_ptr[inner_idx] - mean);
      rstd += d_BF16(d_BF16(diff * diff) * scale);
    }
    rstd = d_BF16(rstd + eps);
    float inv_std = d_lutMantissaBF16(rstd, table, mtable, false);
    for (int inner_idx = 0; inner_idx < inner_dim; inner_idx ++) {
      float val = base_ptr[inner_idx];
      float norm = d_BF16(d_BF16(val - mean) * inv_std);
      if (weight != nullptr)
        norm = d_BF16(norm * weight[inner_idx]);
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

template <typename T0, typename T1>
__global__ void g_gatherElements(T0 *indices, T1 *input, T1 *output,
                                 int64_t *input_shape, int64_t *indices_shape,
                                 int64_t *input_strides, int rank, int axis,
                                 int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    int tmp = idx;
    int input_offset = 0;
    for (int dim = rank - 1; dim >= 0; --dim) {
      int coord = tmp % indices_shape[dim];
      tmp /= indices_shape[dim];
      int input_coord = coord;
      if (dim == axis) {
        input_coord = static_cast<int>(indices[idx]);
        if (input_coord < 0) {
          input_coord += input_shape[dim];
        }
        input_coord = max(0, min(input_coord, static_cast<int>(input_shape[dim]) - 1));
      }
      input_offset += input_coord * input_strides[dim];
    }
    output[idx] = input[input_offset];
  }
}

template <typename T0, typename T1>
__global__ void g_cugather(T0 *indices, T1 *embedding, T1 *output,
                         int num_indices, int outer_dims, int ax_dim, int inner_dims) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int outer_idx = idx / num_indices;
  int indices_idx = idx % num_indices;
  if (outer_idx < outer_dims && indices_idx < num_indices) {
    int index = static_cast<int>(indices[indices_idx]);
    if (index < 0) {
      index += ax_dim;
    }
    int src_idx = outer_idx * ax_dim * inner_dims;
    int dst_idx = outer_idx * num_indices * inner_dims + indices_idx * inner_dims;
    for (int i = 0; i < inner_dims; i++) {
      output[dst_idx + i] = embedding[src_idx + index* inner_dims + i];
    }
  }
}

// -------------------------------------------------------------------------
// ------- cv18xx functions
__global__ void g_cvInt8ScaleToF32(int8_t *input, float *output, float scale,
                                   int size, float zero_point) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float intermediate = static_cast<float>(input[idx]) - zero_point;
    output[idx] = d_BF16(intermediate * scale);
  }
}

__global__ void g_cvInt8ScaleToBF16(int8_t *input, uint16_t *output,
                                    float scale, int size, float zero_point) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float intermediate = static_cast<float>(input[idx]) - zero_point;
    output[idx] = d_BF16Raw(intermediate * scale);
  }
}

__global__ void g_cvF32ScaleToInt8(float *input, int8_t *output, float scale,
                                   int size, int zero_point) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    auto out_bf16 = d_BF16(d_BF16(input[idx], RD_TOWARDS_ZERO) * scale);
    output[idx] = d_f32ToInt<int8_t>(out_bf16 + zero_point, RD_HALF_TO_EVEN);
  }
}

__global__ void g_cvBF16ScaleToInt8(uint16_t *input, int8_t *output,
                                    float scale, int size, int zero_point) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    auto out_bf16 = d_BF16(d_RawBF16(input[idx]) * scale);
    output[idx] = d_f32ToInt<int8_t>(out_bf16 + zero_point, RD_HALF_TO_EVEN);
  }
}

__global__ void g_cvAdd6DInt8(int8_t *a, int8_t *b, int8_t *out, int32_t mul0,
                              int32_t mul1, int shift, bool relu,
                              int i0, int i1, int i2, int i3, int i4, int i5,
                              int j0, int j1, int j2, int j3, int j4, int j5,
                              int o0, int o1, int o2, int o3, int o4, int o5,
                              int a_zp, int b_zp, int out_zp) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_0 = dst_idx / (o1 * o2 * o3 * o4 * o5);
  int idx_1 = dst_idx % (o1 * o2 * o3 * o4 * o5) / (o2 * o3 * o4 * o5);
  int idx_2 = dst_idx % (o2 * o3 * o4 * o5) / (o3 * o4 * o5);
  int idx_3 = dst_idx % (o3 * o4 * o5) / (o4 * o5);
  int idx_4 = dst_idx % (o4 * o5) / o5;
  int idx_5 = dst_idx % o5;
  if (idx_0 < i0 && idx_1 < i1 && idx_2 < i2 && idx_3 < i3 && idx_4 < i4 && idx_5 < i5) {
    int idx_i0 = idx_0 % i0;
    int idx_i1 = idx_1 % i1;
    int idx_i2 = idx_2 % i2;
    int idx_i3 = idx_3 % i3;
    int idx_i4 = idx_4 % i4;
    int idx_i5 = idx_5 % i5;
    int idx_0 = ((((idx_i0 * i1 + idx_i1) * i2 + idx_i2) * i3 + idx_i3) * i4 + idx_i4) * i5 + idx_i5;
    int idx_j0 = idx_0 % j0;
    int idx_j1 = idx_1 % j1;
    int idx_j2 = idx_2 % j2;
    int idx_j3 = idx_3 % j3;
    int idx_j4 = idx_4 % j4;
    int idx_j5 = idx_5 % j5;
    int idx_1 = ((((idx_j0 * j1 + idx_j1) * j2 + idx_j2) * j3 + idx_j3) * j4 + idx_j4) * j5 + idx_j5;
    int32_t temp;
    if (a_zp != 0 || b_zp != 0 || out_zp != 0) {
      int32_t left = (((int32_t)a[idx_0] - a_zp) * mul0 + (1 << (shift - 1))) >> shift;
      int32_t right = (((int32_t)b[idx_1] - b_zp) * mul1 + (1 << (shift - 1))) >> shift;
      temp = left + right;
    } else {
      temp = (int32_t)a[idx_0] * mul0 + (int32_t)b[idx_1] * mul1;
      temp = (temp + (1 << (shift - 1))) >> shift;
    }
    int32_t min_ = relu ? out_zp : -128;
    temp = max(min_, min(127, temp + out_zp));
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

__global__ void g_bmExp(float *input, float *output, int outer_dim, int axis_dim, int inner_dim, float *exp_table) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int out_idx = idx / (axis_dim * inner_dim);
  int axis_idx = idx % (axis_dim * inner_dim) / inner_dim;
  int inner_idx = idx % inner_dim;
  if (out_idx < outer_dim && axis_idx < axis_dim && inner_idx < inner_dim) {
    if (exp_table != nullptr) {
      int32_t table_idx = static_cast<int32_t>(-input[idx]);
      table_idx = max(0, min(255, table_idx));
      float value = exp_table[table_idx];
      output[idx] = value;
    } else {
      float value = __expf(input[idx]);
      output[idx] = value;
    }
  }
}

__global__ void g_bmReciprocal(float *input, float *output, int outer_dim, int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int out_idx = idx / inner_dim;
  int inner_idx = idx % inner_dim;
  if (out_idx < outer_dim && inner_idx < inner_dim) {
    float value = 1.0/input[idx];
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
               Mode == REDUCE_VAR || Mode == REDUCE_STD) {
        return a + b;
    } else if (Mode == REDUCE_L1_NORM) {
      return a + abs(b);
    } else if (Mode ==  REDUCE_L2_NORM) {
      return a + b * b;
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
    int inner_size,      // Product of dimensions after reduction
    bool is_cv18xx_quant       // Flag for CV18xx
) {
    // This kernel is optimized when reducing a single contiguous axis

    // Each block handles inner_size * outer_size outputs
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;

    if (outer_idx < outer_size && inner_idx < inner_size) {
        T myVal = getInitialValue<T, Mode>();

        // Reduction over the contiguous dimension
        for (int i = 0; i < reduce_size; i++) {
            int input_idx = (outer_idx * reduce_size + i) * inner_size + inner_idx;
            T element = input[input_idx];
            myVal = combineValues<T, Mode>(myVal, element);
        }

        // Post-processing
        if (Mode == REDUCE_MEAN && !is_cv18xx_quant) {
            myVal /= reduce_size;
        } else if (Mode == REDUCE_L2_NORM) {
            myVal = sqrt(myVal);
        }

        // Write output
        int output_idx = outer_idx * inner_size + inner_idx;
        output[output_idx] = myVal;
    }
}

// Rotate kernel weights spatially (180 degree flip)
// Input: [oc, ic, kh, kw] or [g, oc/g, ic/g, kh, kw]
// Output: [oc, ic, kh, kw] with kh, kw flipped
template <typename T>
__global__ void g_rotateKernelWeight(T *src, T *dst, int oc, int ic, int kh, int kw) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = oc * ic * kh * kw;

  if (idx < total) {
    int w_idx = idx % kw;
    int h_idx = (idx / kw) % kh;
    int ic_idx = (idx / (kw * kh)) % ic;
    int oc_idx = idx / (kw * kh * ic);

    // Flip spatially: (h, w) -> (kh-1-h, kw-1-w)
    int flipped_h = kh - 1 - h_idx;
    int flipped_w = kw - 1 - w_idx;

    int dst_idx = ((oc_idx * ic + ic_idx) * kh + flipped_h) * kw + flipped_w;
    dst[dst_idx] = src[idx];
  }
}

// Pad tensor for deconv: insert zeros between pixels (stride), apply dilation, and padding
template <typename T>
__global__ void g_padTensorForDeconv(T *dst, T *src, int n, int ic, int ih, int iw,
                                     int oh, int ow, int sh, int sw,
                                     int pad_top, int pad_left, T pad_value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * ic * oh * ow;

  if (idx < total) {
    int w_idx = idx % ow;
    int h_idx = (idx / ow) % oh;
    int c_idx = (idx / (ow * oh)) % ic;
    int n_idx = idx / (ow * oh * ic);

    // Calculate source position (considering padding and stride)
    int src_h = h_idx - pad_top;
    int src_w = w_idx - pad_left;

    // Check if this position corresponds to an original input pixel
    bool is_strided_position = (src_h >= 0 && src_h < ih * sh && src_h % sh == 0 &&
                                 src_w >= 0 && src_w < iw * sw && src_w % sw == 0);

    if (is_strided_position) {
      int orig_h = src_h / sh;
      int orig_w = src_w / sw;
      if (orig_h >= 0 && orig_h < ih && orig_w >= 0 && orig_w < iw) {
        int src_idx = ((n_idx * ic + c_idx) * ih + orig_h) * iw + orig_w;
        dst[idx] = src[src_idx];
      } else {
        dst[idx] = pad_value;
      }
    } else {
      dst[idx] = pad_value;
    }
  }
}

__global__ void g_PReluF32(float *input, float *slope, float *output, int outer_dim, int inner_dim,
                              int num_slope) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < outer_dim * inner_dim) {
    int outer_idx = idx / inner_dim;
    int slope_idx = outer_idx % num_slope;
    float data = input[idx];
    if (data < 0) {
      output[idx] = data * slope[slope_idx];
    } else {
      output[idx] = data;
    }
  }
}

__global__ void g_PReluInt8(int8_t *input, int8_t *slope, int shift, int8_t *output,
                            int outer_dim, int inner_dim, int num_slope) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < outer_dim * inner_dim) {
    int outer_idx = idx / inner_dim;
    int slope_idx = outer_idx % num_slope;
    float data = input[idx];
    if (data < 0) {
      output[idx] = Right_Shift_Round(data * slope[slope_idx], shift, RD_HALF_UP);
    } else {
      output[idx] = data;
    }
  }
}

template <typename T>
__global__ void g_RightBitShift(T *input, T *output, int shift) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (shift < 0) {
    output[idx] = input[idx] << (-shift);
  } else {
    output[idx] = input[idx] >> shift;
  }
}

__global__ void g_GridSample4DIndex(float *input, float *output, int nthreads, int h, int w,
                                    bool align_corners, grid_sample_padding_mode_t padding_mode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nthreads) {
    float x, y;
    if (align_corners) {
      x = (input[idx * 2] + 1.0f) * 0.5f * (w - 1);
      y = (input[idx * 2 + 1] + 1.0f) * 0.5f * (h - 1);
    } else {
      x = ((input[idx * 2] + 1.0f) * w - 1) * 0.5f;
      y = ((input[idx * 2 + 1] + 1.0f) * h - 1) * 0.5f;
    }
    if (padding_mode == BORDER) {
      x = max(0.0f, min(x, w - 1.0f));
      y = max(0.0f, min(y, h - 1.0f));
    } else if (padding_mode == REFLECTION) {
      int twice_w_low = align_corners ? 0 : -1;
      int twice_w_high = align_corners ? 2 * (w - 1) : 2 * w - 1;
      int twice_h_low = align_corners ? 0 : -1;
      int twice_h_high = align_corners ? 2 * (h - 1) : 2 * h - 1;
      if (twice_w_low == twice_w_high) {
        x = 0.0f;
      } else {
        float _min = twice_w_low / 2.0f;
        float _span = (twice_w_high - twice_w_low) / 2.0f;
        x = fabsf(x - _min);
        float _extra = fmod(x, _span);
        int flips = static_cast<int>(floorf(x / _span));
        if (flips % 2 == 1) {
          x = _span - _extra + _min;
        } else {
          x = _extra + _min;
        }
      }
      if (twice_h_low == twice_h_high) {
        y = 0.0f;
      } else {
        float _min = twice_h_low / 2.0f;
        float _span = (twice_h_high - twice_h_low) / 2.0f;
        y = fabsf(y - _min);
        float _extra = fmod(y, _span);
        int flips = static_cast<int>(floorf(y / _span));
        if (flips % 2 == 1) {
          y = _span - _extra + _min;
        } else {
          y = _extra + _min;
        }
      }
      x = max(0.0f, min(x, w - 1.0f));
      y = max(0.0f, min(y, h - 1.0f));
    }
    output[idx * 2] = x;
    output[idx * 2 + 1] = y;
  }
}

__global__ void g_GridSample4DCompute(float *input, float *grid, float *output,
                                      int n, int c, int h, int w, int out_h, int out_w,
                                      grid_sample_interpolation_mode_t interpolation_mode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n * out_h * out_w) {
    int n_idx = idx / (out_h * out_w);
    int remaining = idx % (out_h * out_w);
    int h_idx = remaining / out_w;
    int w_idx = remaining % out_w;

    float x = grid[idx * 2];
    float y = grid[idx * 2 + 1];

    if (interpolation_mode == NEAREST) {
      int x_nearest = roundf(x);
      int y_nearest = roundf(y);
      bool is_valid = (x_nearest >= 0 && x_nearest < w &&
                       y_nearest >= 0 && y_nearest < h);
      for (int c_idx = 0; c_idx < c; c_idx++) {
        if (is_valid) {
          output[((n_idx * c + c_idx) * out_h + h_idx) * out_w + w_idx] =
              input[((n_idx * c + c_idx) * h + y_nearest) * w + x_nearest];
        } else {
          output[((n_idx * c + c_idx) * out_h + h_idx) * out_w + w_idx] = 0.0f;
        }
      }
    } else if (interpolation_mode == BILINEAR) {
      int x0 = floorf(x);
      int y0 = floorf(y);
      float x_lerp = x - x0;
      float y_lerp = y - y0;
      float x0y0_weight = (1 - x_lerp) * (1 - y_lerp);
      float x1y0_weight = x_lerp * (1 - y_lerp);
      float x0y1_weight = (1 - x_lerp) * y_lerp;
      float x1y1_weight = x_lerp * y_lerp;
      bool is_x0_valid = (x0 >= 0 && x0 < w);
      bool is_x1_valid = (x0 + 1 >= 0 && x0 + 1 < w);
      bool is_y0_valid = (y0 >= 0 && y0 < h);
      bool is_y1_valid = (y0 + 1 >= 0 && y0 + 1 < h);
      for (int c_idx = 0; c_idx < c; c_idx++) {

        float res = 0;
        if (is_x0_valid && is_y0_valid)
          res += x0y0_weight * input[((n_idx * c + c_idx) * h + y0) * w + x0];
        if (is_x1_valid && is_y0_valid)
          res += x1y0_weight * input[((n_idx * c + c_idx) * h + y0) * w + (x0 + 1)];
        if (is_x0_valid && is_y1_valid)
          res += x0y1_weight * input[((n_idx * c + c_idx) * h + (y0 + 1)) * w + x0];
        if (is_x1_valid && is_y1_valid)
        res += x1y1_weight * input[((n_idx * c + c_idx) * h + (y0 + 1)) * w + (x0 + 1)];
        output[((n_idx * c + c_idx) * out_h + h_idx) * out_w + w_idx] = res;
      }
    }
  }
}

__global__ void g_argMax(float *input, float *output_idx, float *output_val,
                         int outer_dim, int axis_dim, int inner_dim, bool is_cv18xx) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < outer_dim * inner_dim) {
    int outer_idx = idx / inner_dim;
    int inner_idx = idx % inner_dim;
    if (is_cv18xx) {
      int tile_size = 256;
      int tile_num = (axis_dim + tile_size - 1) / tile_size;
      for (int t = 0; t < tile_num; t++) {
        float max_value = input[outer_idx * axis_dim * inner_dim + t * tile_size * inner_dim + inner_idx];
        for (int i = 1; i < tile_size && t * tile_size + i < axis_dim; i++) {
          float value = input[outer_idx * axis_dim * inner_dim + (t * tile_size + i) * inner_dim + inner_idx];
          if (value >= max_value) {
            max_value = value;
          }
        }
        output_idx[outer_idx * tile_num * inner_dim + t * inner_dim + inner_idx] = max_value;
      }
    } else {
      float max_index = 0;
      float max_value = input[outer_idx * axis_dim * inner_dim + 0 * inner_dim + inner_idx];
      for (int i = 1; i < axis_dim; i++) {
        float value = input[outer_idx * axis_dim * inner_dim + i * inner_dim + inner_idx];
        if (value >= max_value) {
          max_value = value;
          max_index = i;
        }
      }
      output_idx[idx] = max_index;
      if (output_val != nullptr) {
        output_val[idx] = max_value;
      }
    }
  }
}

__global__ void g_argMin(float *input, float *output_idx, float *output_val,
                         int outer_dim, int axis_dim, int inner_dim, bool is_cv18xx) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < outer_dim * inner_dim) {
    int outer_idx = idx / inner_dim;
    int inner_idx = idx % inner_dim;

    if (is_cv18xx) {
      int tile_size = 256;
      int tile_num = (axis_dim + tile_size - 1) / tile_size;
      for (int t = 0; t < tile_num; t++) {
        float min_value = input[outer_idx * axis_dim * inner_dim + t * tile_size * inner_dim + inner_idx];
        for (int i = 1; i < tile_size && t * tile_size + i < axis_dim; i++) {
          float value = input[outer_idx * axis_dim * inner_dim + (t * tile_size + i) * inner_dim + inner_idx];
          if (value <= min_value) {
            min_value = value;
          }
        }
        output_idx[outer_idx * tile_num * inner_dim + t * inner_dim + inner_idx] = min_value;
      }
    } else {
      float min_index = 0;
      float min_value = input[outer_idx * axis_dim * inner_dim + 0 * inner_dim + inner_idx];
      for (int i = 1; i < axis_dim; i++) {
        float value = input[outer_idx * axis_dim * inner_dim + i * inner_dim + inner_idx];
        if (value <= min_value) {
          min_value = value;
          min_index = i;
        }
      }
      output_idx[idx] = min_index;
      if (output_val != nullptr) {
        output_val[idx] = min_value;
      }
    }
  }
}

template <typename T>
__global__ void g_argMax(T *input, T *arg_values, float *output, int outer_dim,
                         int axis_dim, int inner_dim, float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < outer_dim * inner_dim) {
    int outer_idx = idx / inner_dim;
    int inner_idx = idx % inner_dim;
    int tile_size = 256;
    int tile_num = (axis_dim + tile_size - 1) / tile_size;
    float target_value = arg_values[idx];
    float target_tile = 0;
    for (int t = 1; t < tile_num; t++) {
      float value = arg_values[outer_idx * tile_num * inner_dim + t * inner_dim + inner_idx];
      if (value >= target_value) {
        target_value = value;
        target_tile = t;
      }
    }
    int offset = target_tile * tile_size;
    for (int i = 0; i < tile_size && offset + i < axis_dim; i++) {
      float value = input[outer_idx * axis_dim * inner_dim + (offset + i) * inner_dim + inner_idx];
      if (value == target_value) {
        output[idx] = offset + i;
        break;
      }
    }
  }
}

__global__ void g_interpGrid(int *grid_int, float *grid_float, int dim, int out_dim,
                             float scale, bool align_corners, bool half_pixel,
                             interp_platform_t platform) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < out_dim) {
    int gi;
    float gf;
    switch (platform) {
      case TENSORFLOW_NEAREST: {
        gf = half_pixel ? (idx + 0.5f) * scale : idx * scale;
        gf = min(max(gf, 0.0f), dim - 1.0f);
        gi = align_corners ? roundf(gf) : floorf(gf);
        break;
      }
      case PYTORCH_SUPPORT: {
        gf = align_corners ? idx * scale : (idx + 0.5f) * scale - 0.5f;
        gf = min(max(gf, 0.0f), dim - 1.0f);
        gi = floorf(gf);
        break;
      }
      case PYTORCH_NEAREST: {
        gf = idx * scale;
        gf = min(max(gf, 0.0f), dim - 1.0f);
        gi = floorf(gf);
        break;
      }
      case ONNX_NEAREST: {
        gf = half_pixel ? (idx + 0.5f) * scale - 0.5f : idx * scale;
        gf = min(max(gf, 0.0f), dim - 1.0f);
        gi = (half_pixel || !align_corners) ? floorf(gf) : roundf(gf);
        break;
      }
      case CAFFE_SUPPORT:
      case CAFFE_NEAREST: {
        gf = idx * scale;
        gf = min(max(gf, 0.0f), dim - 1.0f);
        gi = floorf(gf);
        break;
      }
    }
    grid_int[idx] = gi;
    if (grid_float != nullptr) {
      grid_float[idx] = gf;
    }
  }
}

__global__ void g_interpCompute(float *input, int *grid_y_int, float *grid_y_float,
                                int *grid_x_int, float *grid_x_float, float *output,
                                int n, int c, int h, int w, int out_h, int out_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n * out_h * out_w) {
    int n_idx = idx / (out_h * out_w);
    int remaining = idx % (out_h * out_w);
    int h_idx = remaining / out_w;
    int w_idx = remaining % out_w;

    int y0 = grid_y_int[h_idx];
    int x0 = grid_x_int[w_idx];
    if (grid_y_float != nullptr && grid_x_float != nullptr) { // bilinear
      float y_lerp = grid_y_float[h_idx] - y0;
      float x_lerp = grid_x_float[w_idx] - x0;

      for (int c_idx = 0; c_idx < c; c_idx++) {
        float v00 = input[((n_idx * c + c_idx) * h + y0) * w + x0];
        float v01 = input[((n_idx * c + c_idx) * h + y0) * w + min(x0 + 1, w - 1)];
        float v10 = input[((n_idx * c + c_idx) * h + min(y0 + 1, h - 1)) * w + x0];
        float v11 = input[((n_idx * c + c_idx) * h + min(y0 + 1, h - 1)) * w + min(x0 + 1, w - 1)];
        float tmp0 = v00 + (v10 - v00) * y_lerp;
        float tmp1 = v01 + (v11 - v01) * y_lerp;
        float res = tmp0 + (tmp1 - tmp0) * x_lerp;
        output[((n_idx * c + c_idx) * out_h + h_idx) * out_w + w_idx] = res;
      }
    } else { // nearest
      for (int c_idx = 0; c_idx < c; c_idx++) {
        output[((n_idx * c + c_idx) * out_h + h_idx) * out_w + w_idx] =
            input[((n_idx * c + c_idx) * h + y0) * w + x0];
      }
    }
  }
}

__global__ void g_GQA_mm(float *A, float *B, float *C, int batch, int mq, int mk,
                         int q_head, int k_head, int dim, bool is_qk) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (is_qk) {
    // A is Query: [batch, mq, q_head, dim]
    // B is Key: [batch, mk, k_head, dim]
    // C is QK^T: [batch, q_head, mq, mk]
    // q_head and k_head may be different
    int head_ratio = q_head / k_head;
    if (idx < batch * q_head * mq) {
      int batch_idx = idx / (q_head * mq);
      int remaining = idx % (q_head * mq);
      int q_head_idx = remaining / mq;
      int mq_idx = remaining % mq;
      int k_head_idx = q_head_idx / head_ratio;

      for (int mk_idx = 0; mk_idx < mk; mk_idx++) {
        float sum = 0;
        for (int d = 0; d < dim; d++) {
          float q_val = A[((batch_idx * mq + mq_idx) * q_head + q_head_idx) * dim + d];
          float k_val = B[((batch_idx * mk + mk_idx) * k_head + k_head_idx) * dim + d];
          sum += q_val * k_val;
        }
        C[((batch_idx * q_head + q_head_idx) * mq + mq_idx) * mk + mk_idx] = sum;
      }
    }
  } else {
    // A is Attention: [batch, q_head, mq, mk]
    // B is Value: [batch, mk, k_head, dim]
    // C is Output: [batch, mq, q_head * dim]
    // q_head and k_head may be different
    int head_ratio = q_head / k_head;
    if (idx < batch * mq * q_head) {
      int batch_idx = idx / (mq * q_head);
      int remaining = idx % (mq * q_head);
      int mq_idx = remaining / q_head;
      int q_head_idx = remaining % q_head;
      int k_head_idx = q_head_idx / head_ratio;

      for (int d = 0; d < dim; d++) {
        float sum = 0;
        for (int mk_idx = 0; mk_idx < mk; mk_idx++) {
          float attn_val = A[((batch_idx * q_head + q_head_idx) * mq + mq_idx) * mk + mk_idx];
          float v_val = B[((batch_idx * mk + mk_idx) * k_head + k_head_idx) * dim + d];
          sum += attn_val * v_val;
        }
        C[((batch_idx * mq + mq_idx) * q_head + q_head_idx) * dim + d] = sum;
      }
    }
  }
}

// Migrated CUDA kernels from legacy branch.
__global__ void g_reciprocal(float *input, float *output, int num,
                             float const_val, int do_relu, float relu_limit) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    output[i] = const_val / input[i];
    if (do_relu) {
      if (output[i] < 0) {
        output[i] = 0;
      }
      if (relu_limit > 0.f && output[i] > relu_limit) {
        output[i] = relu_limit;
      }
    }
  }
}





__global__ void g_abs(float *input, float *output, int num) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    output[i] = fabsf(input[i]);
  }
}

#define AP_TILE_H 8
#define AP_TILE_W 16

__global__ void g_adaptiveAvgPool2D(float *input, float *output,
                                     int n, int c, int ih, int iw,
                                     int oh, int ow) {
  int out_col = blockIdx.y * AP_TILE_W + threadIdx.x;
  int out_row = blockIdx.x * AP_TILE_H + threadIdx.y;
  int nc      = blockIdx.z;                      // N × C 展平

  if (out_row >= oh || out_col >= ow) return;

  int n_idx = nc / c;
  int c_idx = nc % c;

  int start_h = out_row * ih / oh;
  int end_h   = ((out_row + 1) * ih + oh - 1) / oh;
  int start_w = out_col * iw / ow;
  int end_w   = ((out_col + 1) * iw + ow - 1) / ow;
  int kh = end_h - start_h;
  int kw = end_w - start_w;

  // 输入基地址偏移
  float *in_base = input + (n_idx * c + c_idx) * ih * iw;

  float sum = 0.0f;
  for (int h = start_h; h < end_h; ++h) {
    float *row = in_base + h * iw;
    for (int w = start_w; w < end_w; ++w) {
      sum += row[w];
    }
  }
  int out_idx = (nc * oh + out_row) * ow + out_col;
  output[out_idx] = sum / (kh * kw);
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



__global__ void g_addConst4DF32(float *input, float const_v, float *output,
                                bool do_relu, int n, int c, int h, int w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size = n * c * h * w;
  if (idx < size) {
    float val = input[idx] + const_v;
    if (do_relu && val < 0.0f) val = 0.0f;
    output[idx] = val;
  }
}



__global__ void g_arccos(float *input, float *output, int num) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    output[i] = acosf(input[i]);
  }
}



__global__ void g_arctanh(float *input, float *output, int num) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    output[i] = atanhf(input[i]);
  }
}


__global__ void g_attentionPV(float *scores, float *V, float *context,
                               int B, int H, int Mq, int Mk, int d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * H * Mq * d;
  if (idx < total) {
    int di = idx % d;
    int mq = (idx / d) % Mq;
    int h = (idx / (d * Mq)) % H;
    int b = idx / (d * Mq * H);
    float sum = 0.0f;
    for (int k = 0; k < Mk; ++k) {
      sum += scores[((b * H + h) * Mq + mq) * Mk + k] *
             V[((b * H + h) * Mk + k) * d + di];
    }
    context[idx] = sum;
  }
}


__global__ void g_attentionQK(float *Q, float *K, float *scores,
                               int B, int H, int Mq, int Mk, int d,
                               float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * H * Mq * Mk;
  if (idx < total) {
    int mk = idx % Mk;
    int mq = (idx / Mk) % Mq;
    int h = (idx / (Mk * Mq)) % H;
    int b = idx / (Mk * Mq * H);
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
      sum += Q[((b * H + h) * Mq + mq) * d + i] *
             K[((b * H + h) * Mk + mk) * d + i];
    }
    scores[idx] = sum * scale;
  }
}



__global__ void g_batchNormBwdCompute(float *grad_out, float *input,
                                       float *save_mean, float *save_invstd,
                                       float *dxhut,
                                       float *dx2_tmp, float *dx3,
                                       float *dx,
                                       int n, int c, int spatial) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * c * spatial;
  if (idx >= total) return;

  int ci = (idx / spatial) % c;
  float rstd  = save_invstd[ci];
  float mean  = save_mean[ci];
  float M     = (float)(n * spatial);

  dx[idx] = (rstd / M) * (M * dxhut[idx] - dx2_tmp[ci] * (input[idx] - mean) - dx3[ci]);
}


__global__ void g_batchNormBwdStats(float *grad_out, float *input, float *gamma,
                                    float *save_mean, float *save_invstd,
                                    float *dxhut,
                                    float *dgamma, float *dbeta,
                                    float *dx2_tmp, float *dx3,
                                    int n, int c, int spatial) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * c * spatial;
  if (idx >= total) return;

  int ci = (idx / spatial) % c;
  float rstd  = save_invstd[ci];
  float mean  = save_mean[ci];
  float gam   = (gamma != nullptr) ? gamma[ci] : 1.0f;
  float dout  = grad_out[idx];
  float x_val = input[idx];

  float x_hat = (x_val - mean) * rstd;
  float dh    = dout * gam;

  dxhut[idx] = dh;

  atomicAdd(&dgamma[ci], dout * x_hat);
  atomicAdd(&dbeta[ci],  dout);
  float r2 = rstd * rstd;
  atomicAdd(&dx2_tmp[ci], r2 * dh * (x_val - mean));
  atomicAdd(&dx3[ci],     dh);
}



__global__ void g_batchNormInference(float *input, float *output, int n, int c,
                                     int spatial, float *gamma, float *beta,
                                     float *mean, float *var, float eps,
                                     bool do_relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * c * spatial;
  if (idx < total) {
    int ci = (idx / spatial) % c;
    float gamma_v = gamma != nullptr ? gamma[ci] : 1.0f;
    float beta_v = beta != nullptr ? beta[ci] : 0.0f;
    float value = (input[idx] - mean[ci]) * rsqrtf(var[ci] + eps);
    value = value * gamma_v + beta_v;
    if (do_relu) {
      value = fmaxf(value, 0.0f);
    }
    output[idx] = value;
  }
}



__global__ void g_batchNormTrainNormalize(float *input, float *output,
                                          float *gamma, float *beta,
                                          float *mean_out,
                                          float *saved_invstd, int n, int c,
                                          int spatial, bool do_relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * c * spatial;
  if (idx < total) {
    int ci = (idx / spatial) % c;
    float gamma_v = gamma != nullptr ? gamma[ci] : 1.0f;
    float beta_v = beta != nullptr ? beta[ci] : 0.0f;
    float value = (input[idx] - mean_out[ci]) * saved_invstd[ci];
    value = value * gamma_v + beta_v;
    if (do_relu) {
      value = fmaxf(value, 0.0f);
    }
    output[idx] = value;
  }
}



__global__ void g_batchNormTrainStats(float *input, float *mean_in,
                                      float *var_in, float *mean_out,
                                      float *saved_invstd,
                                      float *running_mean,
                                      float *running_var, int n, int c,
                                      int spatial, float eps, float momentum) {
  int ci = blockIdx.x * blockDim.x + threadIdx.x;
  if (ci < c) {
    int channel_size = n * spatial;
    float sum = 0.0f;
    for (int ni = 0; ni < n; ++ni) {
      int base = (ni * c + ci) * spatial;
      for (int si = 0; si < spatial; ++si) {
        sum += input[base + si];
      }
    }

    float cur_mean = sum / channel_size;
    mean_out[ci] = cur_mean;

    float var_sum = 0.0f;
    for (int ni = 0; ni < n; ++ni) {
      int base = (ni * c + ci) * spatial;
      for (int si = 0; si < spatial; ++si) {
        float diff = input[base + si] - cur_mean;
        var_sum += diff * diff;
      }
    }

    float cur_var = var_sum / channel_size;
    saved_invstd[ci] = rsqrtf(cur_var + eps);
    running_mean[ci] = (1.0f - momentum) * mean_in[ci] + momentum * cur_mean;
    running_var[ci] = (1.0f - momentum) * var_in[ci] + momentum * cur_var;
  }
}


__global__ void g_ceil(float *input, float *output, int num) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    output[i] = ceilf(input[i]);
  }
}



__global__ void g_clip(float *input, float *output, int num, float min_v, float max_v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    output[i] = fminf(max_v, fmaxf(min_v, input[i]));
  }
}



template <typename T0, typename T1>
__global__ void g_compare4DF32(T0 *lhs, T1 *rhs, float *out, int mode,
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
    float a_v = lhs[idx_0];
    float b_v = rhs[idx_1];
    bool result;
    if (mode == 0) {
      result = (a_v == b_v);
    } else if (mode == 1) {
      result = (a_v > b_v);
    } else if (mode == 2) {
      result = (a_v >= b_v);
    } else if (mode == 3) {
      result = (a_v < b_v);
    } else if (mode == 4) {
      result = (a_v <= b_v);
    } else if (mode == 5) {
      result = (a_v != b_v);
    } else if (mode == 6) {
      out[dst_idx] = a_v * b_v; return;          // And = binary_mul
    } else if (mode == 7) {
      result = (a_v == 0.0f);                     // Not
    } else if (mode == 8) {
      result = (a_v != b_v);                      // Xor = binary_ne
    } else {
      result = false;
    }
    out[dst_idx] = result ? 1.0f : 0.0f;
  }
}



__global__ void g_compareConst4DF32(float *input, float const_v, float *output,
                                    int mode, bool inversed,
                                    int n, int c, int h, int w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n * c * h * w) {
    float a = input[idx];
    float b = const_v;
    if (inversed) {
      float tmp = a; a = b; b = tmp;
    }
    bool result;
    if (mode == 0) {
      result = (a == b);
    } else if (mode == 1) {
      result = (a > b);
    } else if (mode == 2) {
      result = (a >= b);
    } else if (mode == 3) {
      result = (a < b);
    } else if (mode == 4) {
      result = (a <= b);
    } else if (mode == 5) {
      result = (a != b);
    } else if (mode == 6) {
      output[idx] = a * b; return;          // And = binary_mul
    } else if (mode == 7) {
      result = (a == 0.0f);                  // Not
    } else if (mode == 8) {
      result = (a != b);                     // Xor = binary_ne
    } else {
      result = false;
    }
    output[idx] = result ? 1.0f : 0.0f;
  }
}



__global__ void g_constantFill(float *output, float value, int num) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    output[i] = value;
  }
}



__global__ void g_copy(void *input, void *output, int n, int c, int h, int w,
                       int i_n, int i_c, int i_h, int i_w,
                       int o_n, int o_c, int o_h, int o_w, int tbytes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * c * h * w;
  if (idx < total) {
    int cn = idx / (c * h * w);
    int cc = (idx % (c * h * w)) / (h * w);
    int ch = (idx % (h * w)) / w;
    int cw = idx % w;
    int in_idx = cn * i_n + cc * i_c + ch * i_h + cw * i_w;
    int out_idx = cn * o_n + cc * o_c + ch * o_h + cw * o_w;
    d_copyElement(input, in_idx, output, out_idx, tbytes);
  }
}



__global__ void g_correlation(float *left, float *right, float *output,
                               int max_disp, int num_groups, int ic, int ih, int iw) {
  int spatial = ih * iw;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_groups * max_disp * spatial;
  if (idx < total) {
    int group = idx / (max_disp * spatial);
    int cut = (idx % (max_disp * spatial)) / spatial;
    int s = idx % spatial;
    int h = s / iw;
    int w = s % iw;

    int l_base = group * ic * spatial;
    int r_base = group * ic * spatial;
    float sum = 0.0f;
    if (w >= cut) {
      int wcut = w - cut;
      for (int ch = 0; ch < ic; ++ch) {
        sum += left[l_base + ch * spatial + h * iw + w] *
               right[r_base + ch * spatial + h * iw + wcut];
      }
    }
    output[idx] = sum / ic;
  }
}



__global__ void g_cos(float *input, float *output, int num) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    output[i] = cosf(input[i]);
  }
}



__global__ void g_cosh(float *input, float *output, int num) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    output[i] = coshf(input[i]);
  }
}



__global__ void g_cumSum(float *input, float *output, int outer_dim,
                         int axis_dim, int stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer_dim * stride;
  if (idx < total) {
    int outer = idx / stride;
    int s = idx % stride;
    int base = outer * axis_dim * stride + s;
    float sum = 0.0f;
    for (int a = 0; a < axis_dim; ++a) {
      sum += input[base + a * stride];
      output[base + a * stride] = sum;
    }
  }
}



__global__ void g_depackRaw(float *input, float *output,
                             int n, int ih, int iw, int ph, int pw,
                             float scale, float black_level,
                             int c0, int c1, int c2, int c3) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = ih * 2, ow = iw * 2;
  int total = n * oh * ow;
  if (idx < total) {
    int bn = idx / (oh * ow);
    int s = idx % (oh * ow);
    int h = s / ow, w = s % ow;
    int block_y = h % 2, block_x = w % 2;
    int ch = (block_y * 2 + block_x == 0) ? c0
           : (block_y * 2 + block_x == 1) ? c1
           : (block_y * 2 + block_x == 2) ? c2 : c3;
    int in_h = h / 2 + ph;
    int in_w = w / 2 + pw;
    int in_idx = ((bn * 4 + ch) * (ih + ph) + in_h) * (iw + pw) + in_w;
    float val = input[in_idx];
    output[idx] = (val - black_level) * scale;
  }
}


__global__ void g_elu(float *input, float *output, int num, float alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    float val = input[i];
    output[i] = val > 0 ? val : alpha * (expf(val) - 1.0f);
  }
}



// ==========================================================================
// DequantizeLinear
// ==========================================================================

template <typename T>
__global__ void g_dequantizeLinearPerTensor(T *input, float *output,
                                            float scale, int32_t zp, int num) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    output[i] = ((float)(int32_t)input[i] - (float)zp) * scale;
  }
}

template <typename T>
__global__ void g_dequantizeLinearPerChannel(T *input, float *output,
                                             float *scale, int32_t *zp,
                                             int outer_dim, int channel_dim,
                                             int inner_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer_dim * channel_dim * inner_dim;
  if (i < total) {
    int c = (i / inner_dim) % channel_dim;
    output[i] = ((float)(int32_t)input[i] - (float)zp[c]) * scale[c];
  }
}

// ==========================================================================
// DequantInt
// ==========================================================================

template <typename T>
__global__ void g_dequantIntPerTensor(T *input, float *output, int num,
                                      int64_t multiplier, int64_t shift,
                                      int64_t lshift, int32_t zp, int mode,
                                      rounding_mode_t rmode) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    int32_t val = (int32_t)input[i] - zp;
    if (mode == 0) {
      output[i] = d_applyMultiplierAndRShift(val, multiplier, -shift, false,
                                             rmode);
    } else {
      int64_t tmp = (int64_t)val * multiplier << lshift;
      tmp = Right_Shift_Round(tmp, 31, RD_HALF_UP);
      output[i] = (float)Right_Shift_Round(tmp, -shift, rmode);
    }
  }
}

template <typename T>
__global__ void g_dequantIntPerChannel(T *input, float *output,
                                       int outer_dim, int channel_dim,
                                       int inner_dim, int64_t *multiplier,
                                       int64_t *shift, int64_t lshift,
                                       int32_t zp, int mode,
                                       rounding_mode_t rmode) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer_dim * channel_dim * inner_dim;
  if (i < total) {
    int c = (i / inner_dim) % channel_dim;
    int32_t val = (int32_t)input[i] - zp;
    if (mode == 0) {
      output[i] = d_applyMultiplierAndRShift(val, multiplier[c], -shift[c],
                                             false, rmode);
    } else {
      int64_t tmp = (int64_t)val * multiplier[c] << lshift;
      tmp = Right_Shift_Round(tmp, 31, RD_HALF_UP);
      output[i] = (float)Right_Shift_Round(tmp, -shift[c], rmode);
    }
  }
}

__global__ void g_embDenseBwd(float *grad_output, float *indices, float *output,
                               int batch_size, int embed_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * embed_dim;
  if (idx >= total) return;

  int b = idx / embed_dim;
  int e = idx % embed_dim;
  int weight_idx = (int)indices[b];
  output[idx] = grad_output[weight_idx * embed_dim + e];
}


__global__ void g_erf(float *input, float *output, int num) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    output[i] = erff(input[i]);
  }
}


__global__ void g_expElm(float *input, float *output, int num) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    output[i] = expf(input[i]);
  }
}

// 输入 dim 为 1 时广播
__global__ void g_expand(float *input, float *output,
                          int in_n, int in_c, int in_h, int in_w,
                          int out_n, int out_c, int out_h, int out_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = out_n * out_c * out_h * out_w;
  if (idx >= total) return;

  int ow = idx % out_w;
  int oh = (idx / out_w) % out_h;
  int oc = (idx / (out_w * out_h)) % out_c;
  int on = idx / (out_w * out_h * out_c);

  int iw = (in_w == 1) ? 0 : ow;
  int ih = (in_h == 1) ? 0 : oh;
  int ic = (in_c == 1) ? 0 : oc;
  int in_idx = (in_n == 1) ? 0 : on;

  output[idx] = input[in_idx * in_c * in_h * in_w + ic * in_h * in_w + ih * in_w + iw];
}

// N 维索引 gather
__global__ void g_gatherND(float *input, float *indices, float *output,
                            int *in_shape, int *in_strides, int *idx_strides,
                            int indices_dim, int coord_dim, int batch_dims,
                            int out_total, int copy_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int out_idx = idx / copy_len;
  int copy_off = idx % copy_len;
  if (out_idx >= out_total) return;

  // 解码 out_idx 到 indices 的多维坐标（不含最后的 coord 维）
  // indices 的第 0 到 batch_dims-1 维是 batch，batch_dims 到倒数第二维是 middle
  // out_idx 遍历所有 batch+middles 的 flat 组合
  int batch_flat = out_idx;
  // 解码 batch 部分坐标
  int in_base = 0;
  for (int d = 0; d < batch_dims; d++) {
    int coord = batch_flat / idx_strides[d];
    batch_flat %= idx_strides[d];
    in_base += coord * in_strides[d];
  }

  // 读取 indices：flat 索引 out_idx 对应 indices[out_idx, :]
  float *idx_ptr = indices + out_idx * coord_dim;
  for (int d = 0; d < coord_dim; d++) {
    int coord = (int)idx_ptr[d];
    in_base += coord * in_strides[batch_dims + d];
  }

  output[idx] = input[in_base + copy_off];
}

// 1 线程处理 1 个 group 的全部 inner_dim 元素
__global__ void g_groupNorm(float *input, float *output,
                             float *weight, float *bias,
                             int outer_dim, int inner_dim,
                             int channel, int channel_per_group, float eps) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= outer_dim) return;

  // 计算 mean
  float mean = 0.0f;
  float *in_ptr = input + g * inner_dim;
  for (int j = 0; j < inner_dim; j++) {
    mean += in_ptr[j];
  }
  mean /= (float)inner_dim;

  // 计算 rstd = 1/sqrt(var + eps)
  float var = 0.0f;
  for (int j = 0; j < inner_dim; j++) {
    float d = in_ptr[j] - mean;
    var += d * d;
  }
  var /= (float)inner_dim;
  float rstd = 1.0f / sqrtf(var + eps);

  // 归一化
  float *out_ptr = output + g * inner_dim;
  for (int j = 0; j < inner_dim; j++) {
    out_ptr[j] = (in_ptr[j] - mean) * rstd;
  }

  // affine: weight * y + bias, 按 channel 索引
  int spatial = inner_dim / channel_per_group;
  int group_idx = g % (channel / channel_per_group);
  for (int c = 0; c < channel_per_group; c++) {
    int ch = group_idx * channel_per_group + c;
    float w = (weight != nullptr) ? weight[ch] : 1.0f;
    float b = (bias != nullptr) ? bias[ch] : 0.0f;
    float *ch_ptr = out_ptr + c * spatial;
    for (int s = 0; s < spatial; s++) {
      ch_ptr[s] = ch_ptr[s] * w + b;
    }
  }
}

// 额外输出 mean 和 rstd
__global__ void g_groupNormTrain(float *input, float *output,
                                  float *mean_out, float *rstd_out,
                                  float *weight, float *bias,
                                  int outer_dim, int inner_dim,
                                  int channel, int channel_per_group,
                                  float eps) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= outer_dim) return;

  float mean = 0.0f;
  float *in_ptr = input + g * inner_dim;
  for (int j = 0; j < inner_dim; j++) {
    mean += in_ptr[j];
  }
  mean /= (float)inner_dim;

  float var = 0.0f;
  for (int j = 0; j < inner_dim; j++) {
    float d = in_ptr[j] - mean;
    var += d * d;
  }
  var /= (float)inner_dim;
  float rstd = 1.0f / sqrtf(var + eps);

  mean_out[g] = mean;
  rstd_out[g] = rstd;

  float *out_ptr = output + g * inner_dim;
  for (int j = 0; j < inner_dim; j++) {
    out_ptr[j] = (in_ptr[j] - mean) * rstd;
  }

  int spatial = inner_dim / channel_per_group;
  int group_idx = g % (channel / channel_per_group);
  for (int c = 0; c < channel_per_group; c++) {
    int ch = group_idx * channel_per_group + c;
    float w = (weight != nullptr) ? weight[ch] : 1.0f;
    float b = (bias != nullptr) ? bias[ch] : 0.0f;
    float *ch_ptr = out_ptr + c * spatial;
    for (int s = 0; s < spatial; s++) {
      ch_ptr[s] = ch_ptr[s] * w + b;
    }
  }
}

// 状态更新
__global__ void g_gruCell(float *x_gi, float *x_gr, float *x_gh,
                           float *h_gi, float *h_gr, float *h_gh,
                           float *h_prev, float *h_out,
                           int total, bool linear_before_reset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  float r = 1.0f / (1.0f + expf(-(x_gr[idx] + h_gr[idx])));
  float z = 1.0f / (1.0f + expf(-(x_gi[idx] + h_gi[idx])));

  // linear_before_reset: n = tanh(x_gh + r * h_gh)
  float n_val = x_gh[idx];
  if (linear_before_reset) {
    n_val += r * h_gh[idx];
  } else {
    n_val += h_gh[idx] * r;  // simplified: same formula for now
  }
  float n = tanhf(n_val);

  float h_new = (1.0f - z) * n + z * h_prev[idx];
  h_out[idx] = h_new;
}


__global__ void g_hardsigmoid(float *input, float *output, int num,
                               float alpha, float beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    float val = alpha * input[i] + beta;
    output[i] = fminf(1.0f, fmaxf(0.0f, val));
  }
}


__global__ void g_hardswish(float *input, float *output, int num) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    float val = input[i];
    output[i] = val * fminf(1.0f, fmaxf(0.0f, val / 6.0f + 0.5f));
  }
}


__global__ void g_indexPut(float *input, float *indices, float *values,
                            float *output, int num_indices, int inner_dim,
                            bool accumulate) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_indices * inner_dim;
  if (idx >= total) return;

  int i_idx = idx / inner_dim;
  int j = idx % inner_dim;
  int dst_idx = (int)indices[i_idx] * inner_dim + j;
  if (accumulate) {
    output[dst_idx] += values[idx];
  } else {
    output[dst_idx] = values[idx];
  }
}

// 对的全部 spatial 元素
__global__ void g_instanceNorm(float *input, float *output,
                                float *weight, float *bias,
                                int outer_dim, int inner_dim,
                                int channel, float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= outer_dim) return;

  float *in_ptr = input + idx * inner_dim;
  float *out_ptr = output + idx * inner_dim;

  float mean = 0.0f;
  for (int j = 0; j < inner_dim; j++) mean += in_ptr[j];
  mean /= (float)inner_dim;

  float var = 0.0f;
  for (int j = 0; j < inner_dim; j++) {
    float d = in_ptr[j] - mean;
    var += d * d;
  }
  var /= (float)inner_dim;
  float rstd = 1.0f / sqrtf(var + eps);

  for (int j = 0; j < inner_dim; j++) {
    out_ptr[j] = (in_ptr[j] - mean) * rstd;
  }

  int c = idx % channel;
  float w = (weight != nullptr) ? weight[c] : 1.0f;
  float b = (bias != nullptr) ? bias[c] : 0.0f;
  for (int j = 0; j < inner_dim; j++) {
    out_ptr[j] = out_ptr[j] * w + b;
  }
}

// 1 线程 1 个 outer
__global__ void g_layerNormTrain(float *input, float *output,
                                  float *mean_out, float *rstd_out,
                                  float *weight, float *bias,
                                  int outer_dim, int inner_dim, float eps) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= outer_dim) return;

  float *in_ptr = input + i * inner_dim;
  float *out_ptr = output + i * inner_dim;

  float mean = 0.0f;
  for (int j = 0; j < inner_dim; j++) mean += in_ptr[j];
  mean /= (float)inner_dim;

  float var = 0.0f;
  for (int j = 0; j < inner_dim; j++) {
    float d = in_ptr[j] - mean;
    var += d * d;
  }
  var /= (float)inner_dim;
  float rstd = 1.0f / sqrtf(var + eps);

  mean_out[i] = mean;
  rstd_out[i] = rstd;

  for (int j = 0; j < inner_dim; j++) {
    float val = (in_ptr[j] - mean) * rstd;
    if (weight) val *= weight[j];
    if (bias)   val += bias[j];
    out_ptr[j] = val;
  }
}

// x : alpha * x
__global__ void g_leakyRelu(float *input, float *output, int num, float alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    float val = input[i];
    output[i] = val > 0.0f ? val : alpha * val;
  }
}


__global__ void g_log(float *input, float *output, int num) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    output[i] = logf(input[i]);
  }
}


__global__ void g_logB(float *input, float *output, int num, float log_base_inv) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    output[i] = logf(input[i]) * log_base_inv;
  }
}


__global__ void g_logicalAnd(float *lhs, float *rhs, float *output,
                              int l_n, int l_c, int l_h, int l_w,
                              int r_n, int r_c, int r_h, int r_w,
                              int o_n, int o_c, int o_h, int o_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = o_n * o_c * o_h * o_w;
  if (idx >= total) return;

  int ow = idx % o_w;
  int oh = (idx / o_w) % o_h;
  int oc = (idx / (o_w * o_h)) % o_c;
  int on = idx / (o_w * o_h * o_c);

  int li = (l_n == 1) ? 0 : on;
  int lc = (l_c == 1) ? 0 : oc;
  int lh = (l_h == 1) ? 0 : oh;
  int lw = (l_w == 1) ? 0 : ow;
  float lv = lhs[li * l_c * l_h * l_w + lc * l_h * l_w + lh * l_w + lw];

  int ri = (r_n == 1) ? 0 : on;
  int rc = (r_c == 1) ? 0 : oc;
  int rh = (r_h == 1) ? 0 : oh;
  int rw = (r_w == 1) ? 0 : ow;
  float rv = rhs[ri * r_c * r_h * r_w + rc * r_h * r_w + rh * r_w + rw];

  output[idx] = (lv != 0.0f && rv != 0.0f) ? 1.0f : 0.0f;
}

// beta
__global__ void g_lrn(float *input, float *output,
                       int n, int c, int h, int w,
                       int size, float alpha, float beta, float bias) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * c * h * w;
  if (idx >= total) return;

  int wi = idx % w;
  int hi = (idx / w) % h;
  int ci = (idx / (w * h)) % c;
  int ni = idx / (w * h * c);

  int half = size / 2;
  int c_start = max(0, ci - half);
  int c_end   = min(c - 1, ci + half);
  float scale = alpha / (float)size;

  float sum_sq = 0.0f;
  float *in_ptr = input + ni * c * h * w + hi * w + wi;
  for (int j = c_start; j <= c_end; j++) {
    float val = in_ptr[j * h * w];
    sum_sq += val * val;
  }

  output[idx] = input[idx] / powf(bias + scale * sum_sq, beta);
}


__global__ void g_lstmAddBias(float *gate, float *bias, int batch_size,
                               int hidden_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * hidden_size;
  if (idx >= total) return;
  int h = idx % hidden_size;
  gate[idx] += bias[h];
}

// c, 更新 cell state 和 hidden state
__global__ void g_lstmCell(float *x_i, float *x_o, float *x_f, float *x_c,
                            float *h_i, float *h_o, float *h_f, float *h_c,
                            float *cell_state, float *hidden_state,
                            int total, float cont) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  float gi = 0.5f * tanhf(0.5f * (x_i[idx] + cont * h_i[idx])) + 0.5f;
  float go = 0.5f * tanhf(0.5f * (x_o[idx] + cont * h_o[idx])) + 0.5f;
  float gf = 0.5f * tanhf(0.5f * (x_f[idx] + cont * h_f[idx])) + 0.5f;
  float gc = tanhf(x_c[idx] + cont * h_c[idx]);

  float c_new = cont * gf * cell_state[idx] + gi * gc;
  float h_new = go * tanhf(c_new);
  cell_state[idx] = c_new;
  hidden_state[idx] = h_new;
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

// add for conv3d

__global__ void g_pad5D(void *input, void *output, int n, int c,int d,int h, int w,
                        int pad_h_t,int pad_d_f,int pad_d_b,int pad_h_b, int pad_w_l, int pad_w_r,
                        int tbytes){
  //输出尺寸
  int od = d + pad_d_f + pad_d_b;
  int oh = h + pad_h_t + pad_h_b;
  int ow = w + pad_w_l + pad_w_r;

  long idx = (long)blockIdx.x *blockDim.x + threadIdx.x;
  long total = (long)n * c * od * oh * ow;
  if (idx >= total) return;

  // 反解码成5D坐标（从外到内）
  int idx_w = idx % ow;
  int idx_h = (idx / ow)         % oh;
  int idx_d = (idx / (ow * oh))  % od;
  int idx_c = (idx / (ow * oh * od)) % c;
  int idx_n =  idx / (ow * oh * od * c);

  // 输出位置的线性索引
  long long out_idx = (((long long)idx_n * c + idx_c) * od + idx_d) * oh * ow
                      + idx_h * ow + idx_w;


  bool inside = (idx_d >= pad_d_f && idx_d < pad_d_f + d) &&
                  (idx_h >= pad_h_t && idx_h < pad_h_t + h) &&
                  (idx_w >= pad_w_l && idx_w < pad_w_l + w);

  if (inside)
    {
      int id_in = idx_d - pad_d_f;
      int ih_in = idx_h - pad_h_t;
      int iw_in = idx_w - pad_w_l;

      long long in_idx = (((long long)idx_n * c + idx_c) * d + id_in) * h * w
                       + ih_in * w + iw_in;

      d_copyElement(input, in_idx, output, out_idx, tbytes);
    }
  else
    {
      d_setZero(output, out_idx, tbytes);
    }
}

// for attention head reordering
__global__ void g_permuteBMHD(float *src, float *dst,
                               int B, int M, int H, int d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * M * H * d;
  if (idx < total) {
    int di = idx % d;
    int rem = idx / d;
    int h_src = rem % H;
    int m = (rem / H) % M;
    int b = rem / (H * M);
    // dst[b, h, m, d] = src[b, m, h, d]
    int si = ((b * M + m) * H + h_src) * d + di;
    dst[idx] = src[si];
  }
}



__global__ void g_range(float *output, float start, float delta, int num) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    output[i] = start + (float)i * delta;
  }
}



__global__ void g_requantF8Perchannel_3d(float *input, uint8_t *output,
                                        float *scales, int n, int c, int d,int h, int w, bool relu, bool conv=true) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (n * c * d * h * w)) {
    int idx_c = idx % (c *d* h * w) / (d*h * w);
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



__global__ void g_requantInt16Perchannel_3d(int32_t *input, void *output,
                                        int32_t *multipliers, int32_t *shifts,
                                        int n, int c, int h, int w,int d, bool relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = n * c * h * w * d;
  if (idx < total_elements) {
    // 计算通道索引 idx_c
    // 对于 4D (d==1): idx % (c*h*w) / (h*w)  → 保持原有逻辑
    // 对于 5D: 跳过 D*H*W 部分后计算通道
    int elements_per_channel = d * h * w;
    int idx_c = (idx % (c * elements_per_channel)) / elements_per_channel;

    int32_t value;
    // half up rounding
    int64_t data = static_cast<int64_t>(input[idx]) *
                   static_cast<int64_t>(multipliers[idx_c]);
    int64_t round_val = (int64_t)(1ll << (shifts[idx_c] - 1));
    data = (data + round_val) >> shifts[idx_c];
    value = static_cast<int32_t>(data);

    int32_t min_val = relu ? 0 : -32768;
    value = max(min_val, min(32767, value));

    ((int16_t *)output)[idx] = static_cast<int16_t>(value);
  }
  // if (idx < (n * c * h * w)) {
  //   int idx_c = idx % (c * h * w) / (h * w);
  //   int32_t value;
  //   // half up
  //   int64_t data = static_cast<int64_t>(input[idx]) *
  //                   static_cast<int64_t>(multipliers[idx_c]);
  //   int64_t round = (int64_t)(1ll << (shifts[idx_c] - 1));
  //   data = (data + round) >> shifts[idx_c];
  //   value = static_cast<int32_t>(data);
  //   int32_t min_ = relu ? 0 : -32768;
  //   value = max(min_, min(32767, value));
  //   ((int16_t *)output)[idx] = static_cast<int16_t>(value);
  // }
}


__global__ void g_requantInt8Perchannel_3d(int32_t *input, void *output,
                                        int32_t *multipliers, int32_t *shifts,
                                        int n, int c, int h, int w,int d,
                                        bool out_sign, bool qdm, bool relu) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = n * c * d * h * w;     // 支持 5D (NCDHW)

  if (idx < total_elements) {

    // ====================== 计算通道索引 idx_c ======================
    // d==1 时：行为与原来完全一致
    // d>1  时：正确处理 Conv3D 的深度维度
    int elements_per_channel = d * h * w;
    int idx_c = (idx % (c * elements_per_channel)) / elements_per_channel;

    int32_t value;

    if (qdm == false) {
      // half up rounding
      int64_t data = static_cast<int64_t>(input[idx]) *
                     static_cast<int64_t>(multipliers[idx_c]);
      int64_t round = (int64_t)(1ll << (shifts[idx_c] - 1));
      data = (data + round) >> shifts[idx_c];
      value = static_cast<int32_t>(data);
    }
    else {
      // qdm mode
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

    // ====================== 裁剪与输出 ======================
    if (out_sign) {
      int32_t min_ = relu ? 0 : -128;
      value = max(min_, min(127, value));
      ((int8_t *)output)[idx] = static_cast<int8_t>(value);
    }
    else {
      value = max(0, min(255, value));
      ((uint8_t *)output)[idx] = static_cast<uint8_t>(value);
    }
  }
}



__global__ void g_reverse(float *input, float *output, int outer_stride,
                           int axis_dim, int inner_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer_stride * axis_dim * inner_stride;
  if (i < total) {
    int o = i / (axis_dim * inner_stride);
    int a = (i / inner_stride) % axis_dim;
    int in = i % inner_stride;
    int src = o * axis_dim * inner_stride +
              (axis_dim - 1 - a) * inner_stride + in;
    output[i] = input[src];
  }
}



__global__ void g_rmsNorm(float *input, float *output, int outer_dim,
                           int inner_dim, float *gamma, float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < outer_dim) {
    float *in_row = input + idx * inner_dim;
    float *out_row = output + idx * inner_dim;
    float sum_sq = 0.0f;
    for (int j = 0; j < inner_dim; ++j) sum_sq += in_row[j] * in_row[j];
    float rms = sqrtf(sum_sq / inner_dim + eps);
    for (int j = 0; j < inner_dim; ++j) {
      float val = in_row[j] / rms;
      if (gamma) val *= gamma[j];
      out_row[j] = val;
    }
  }
}


__global__ void g_roiAlign(
    float *input, float *rois, float *output,
    int N, int C, int H, int W,
    int num_rois, int output_h, int output_w,
    int sampling_ratio, float spatial_scale,
    bool align_corners, bool avg_mode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_rois * C * output_h * output_w;
  if (idx >= total) return;

  int ow = idx % output_w;
  int oh = (idx / output_w) % output_h;
  int c = (idx / (output_w * output_h)) % C;
  int roi_idx = idx / (C * output_w * output_h);

  float batch_id_f = rois[roi_idx * 5 + 0];
  int batch_id = (int)batch_id_f;
  float x1 = rois[roi_idx * 5 + 1] * spatial_scale;
  float y1 = rois[roi_idx * 5 + 2] * spatial_scale;
  float x2 = rois[roi_idx * 5 + 3] * spatial_scale;
  float y2 = rois[roi_idx * 5 + 4] * spatial_scale;

  float offset = align_corners ? 0.5f : 0.0f;
  float start_h = y1 - offset;
  float start_w = x1 - offset;
  float end_h = y2 - offset;
  float end_w = x2 - offset;

  float roi_h = end_h - start_h;
  float roi_w = end_w - start_w;
  if (!align_corners) {
    roi_h = fmaxf(roi_h, 1.0f);
    roi_w = fmaxf(roi_w, 1.0f);
  }

  float bin_h = roi_h / output_h;
  float bin_w = roi_w / output_w;

  int sample_h = sampling_ratio > 0 ? sampling_ratio : (int)ceilf(bin_h);
  int sample_w = sampling_ratio > 0 ? sampling_ratio : (int)ceilf(bin_w);

  float result = 0.0f;
  bool first = true;
  int sample_count = sample_h * sample_w;

  for (int iy = 0; iy < sample_h; iy++) {
    float y = start_h + oh * bin_h + (iy + 0.5f) * bin_h / sample_h;
    for (int ix = 0; ix < sample_w; ix++) {
      float x = start_w + ow * bin_w + (ix + 0.5f) * bin_w / sample_w;

      float val;
      if (y < -1.0f || y > (float)H || x < -1.0f || x > (float)W) {
        val = 0.0f;
      } else {
        y = fminf(fmaxf(y, 0.0f), (float)(H - 1));
        x = fminf(fmaxf(x, 0.0f), (float)(W - 1));

        int yl = (int)y;
        int yh = min(yl + 1, H - 1);
        int xl = (int)x;
        int xh = min(xl + 1, W - 1);

        float ly = y - (float)yl;
        float lx = x - (float)xl;
        float hy = 1.0f - ly;
        float hx = 1.0f - lx;

        int base = batch_id * C * H * W + c * H * W;
        val = input[base + yl * W + xl] * hy * hx
            + input[base + yh * W + xl] * ly * hx
            + input[base + yl * W + xh] * hy * lx
            + input[base + yh * W + xh] * ly * lx;
      }

      if (avg_mode) {
        result += val;
      } else {
        if (first) { result = val; first = false; }
        else { result = fmaxf(result, val); }
      }
    }
  }

  if (avg_mode) {
    result /= (float)sample_count;
  }
  output[idx] = result;
}

// ties to even
__global__ void g_round(float *input, float *output, int num) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    output[i] = roundf(input[i]);
  }
}


__global__ void g_rsqrt(float *input, float *output, int num, float eps) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    output[i] = 1.0f / sqrtf(input[i] + eps);
  }
}
} // namespace cuda
} // namespace tpu_mlir
