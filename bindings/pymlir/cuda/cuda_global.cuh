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

} // namespace cuda
} // namespace tpu_mlir
