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
                        int tbytes, int pad_value) {
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


}
}