//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "cuda_helper.h"
#include "stdio.h"

#define CUDA_BLOCK_SIZE 256
#define CUDA_NUM_BLOCKS(n) ((n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE)

template <typename T> __device__ T kernelInt(float data, cuda_rmode_t rmode) {
  switch (rmode) {
  case CUDA_HALF_AWAY_FROM_ZERO:
    data = roundf(data);
    break;
  case CUDA_HALF_UP:
    data = floor(data + 0.5f);
    break;
  case CUDA_TOWARDS_ZERO:
    data = truncf(data);
    break;
  case CUDA_HALF_TO_EVEN:
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

__device__ void kernelCopyElement(void *src, int sidx, void *dst, int didx,
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

__device__ void kernelSetZero(void *dst, int didx, int tbytes) {
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

__global__ void kernelF32ToInt8(float *input, void *output, float scale,
                                int size, bool sign, cuda_rmode_t rmode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float value = input[idx] * scale;
    if (sign) {
      static_cast<int8_t *>(output)[idx] = kernelInt<int8_t>(value, rmode);
    } else {
      static_cast<uint8_t *>(output)[idx] = kernelInt<uint8_t>(value, rmode);
    }
  }
}

void cudaF32ToInt8(void *input, void *output, float scale, int size, bool sign,
                   cuda_rmode_t rmode) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  kernelF32ToInt8<<<num_blocks, block_size>>>((float *)input, output, scale,
                                              size, sign, rmode);
}

struct bfloat16 {
  uint16_t value;

  __device__ bfloat16() : value(0) {}
  __device__ bfloat16(uint16_t v) : value(v) {}
  __device__ bfloat16(float val, bool half_up = false) {
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

__device__ float kernel_BF16(float data, bool round_up = true) {
  bfloat16 in_bf16(data, round_up);
  return static_cast<float>(in_bf16);
}

__device__ float kernel_BF16(uint16_t data) {
  bfloat16 in_bf16(data);
  return static_cast<float>(in_bf16);
}

__global__ void kernelCVScaleToF32(int8_t *input, float *output, float scale,
                                   int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float intermediate = static_cast<float>(input[idx]);
    output[idx] = kernel_BF16(intermediate * scale);
  }
}

void cudaCVScaleToF32(void *input, void *output, float scale, int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  kernelCVScaleToF32<<<num_blocks, block_size>>>((int8_t *)input,
                                                 (float *)output, scale, size);
}

__global__ void kernelCVScaleToBF16(int8_t *input, uint16_t *output,
                                    float scale, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float intermediate = static_cast<float>(input[idx]);
    float out = kernel_BF16(intermediate * scale);
    bfloat16 out_bf16(out, false);
    output[idx] = out_bf16.value;
  }
}

void cudaCVScaleToBF16(void *input, void *output, float scale, int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  kernelCVScaleToBF16<<<num_blocks, block_size>>>(
      (int8_t *)input, (uint16_t *)output, scale, size);
}

__global__ void kernelInt8ToF32(void *input, float *output, float scale,
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

void cudaInt8ToF32(void *input, void *output, float scale, int size,
                   bool sign) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  kernelInt8ToF32<<<num_blocks, block_size>>>((int8_t *)input, (float *)output,
                                              scale, size, sign);
}

__global__ void kernelCVQuantInt8(float *input, int8_t *output, float scale,
                                  int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    auto out_bf16 = kernel_BF16(kernel_BF16(input[idx], false) * scale);
    output[idx] = kernelInt<int8_t>(out_bf16, CUDA_HALF_TO_EVEN);
  }
}

__global__ void kernelCVQuantInt8(uint16_t *input, int8_t *output, float scale,
                                  int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    auto out_bf16 = kernel_BF16(kernel_BF16(input[idx]) * scale);
    output[idx] = kernelInt<int8_t>(out_bf16, CUDA_HALF_TO_EVEN);
  }
}

void cudaCVQuantInt8(void *input, void *output, float scale, int size,
                     bool is_bf16) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  if (!is_bf16) {
    kernelCVQuantInt8<<<num_blocks, block_size>>>(
        (float *)input, (int8_t *)output, scale, size);
  } else {
    kernelCVQuantInt8<<<num_blocks, block_size>>>(
        (uint16_t *)input, (int8_t *)output, scale, size);
  }
}

__global__ void kernelCVAddInt8(int8_t *a, int8_t *b, int8_t *out, int32_t mul0,
                                int32_t mul1, int shift, int size, bool relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int32_t temp = (int32_t)a[idx] * mul0 + (int32_t)b[idx] * mul1;
    temp = (temp + (1 << (shift - 1))) >> shift;
    int32_t min_ = relu ? 0 : -128;
    temp = max(min_, min(127, temp));
    out[idx] = static_cast<int8_t>(temp);
  }
}

void cudaCVAddInt8(void *input0, void *input1, void *output, int mul0, int mul1,
                   int shift, int size, bool relu) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  kernelCVAddInt8<<<num_blocks, block_size>>>(
      (int8_t *)input0, (int8_t *)input1, (int8_t *)output, mul0, mul1, shift,
      size, relu);
}

template <typename T0, typename T1, typename T2>
__global__ void kernelMulInt8(T0 *a, T1 *b, T2 *out, int32_t multiplier,
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

void cudaMulInt8(void *a, void *b, void *o, bool a_sign, bool b_sign,
                 bool o_sign, int multiplier, int rshift, int size, bool qdm,
                 bool relu) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  if (a_sign && b_sign && o_sign) {
    kernelMulInt8<<<num_blocks, block_size>>>((int8_t *)a, (int8_t *)b,
                                              (int8_t *)o, multiplier, rshift,
                                              size, qdm, relu);
  } else if (!a_sign && !b_sign && !o_sign) {
    kernelMulInt8<<<num_blocks, block_size>>>((uint8_t *)a, (uint8_t *)b,
                                              (uint8_t *)o, multiplier, rshift,
                                              size, qdm, relu);
  } else if (a_sign && b_sign && !o_sign) {
    kernelMulInt8<<<num_blocks, block_size>>>((int8_t *)a, (int8_t *)b,
                                              (uint8_t *)o, multiplier, rshift,
                                              size, qdm, relu);
  }
}

template <typename T0, typename T1, typename T2>
__global__ void
kernelMulBinaryInt8(T0 *a, T1 *b, T2 *out, int n0, int c0, int h0, int w0,
                    int n1, int c1, int h1, int w1, int n2, int c2, int h2,
                    int w2, int multiplier, int rshift, bool qdm, bool relu) {
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

void cudaMulBinaryInt8(void *a, void *b, void *o, int n0, int c0, int h0,
                       int w0, int n1, int c1, int h1, int w1, int n2, int c2,
                       int h2, int w2, bool a_sign, bool b_sign, bool o_sign,
                       int multiplier, int rshift, bool qdm, bool relu) {
  int num_blocks = CUDA_NUM_BLOCKS(n2 * c2 * h2 * w2);
  int block_size = CUDA_BLOCK_SIZE;
  if (a_sign && b_sign && o_sign) {
    kernelMulBinaryInt8<<<num_blocks, block_size>>>(
        (int8_t *)a, (int8_t *)b, (int8_t *)o, n0, c0, h0, w0, n1, c1, h1, w1,
        n2, c2, h2, w2, multiplier, rshift, qdm, relu);
  } else if (!a_sign && !b_sign && !o_sign) {
    kernelMulBinaryInt8<<<num_blocks, block_size>>>(
        (uint8_t *)a, (uint8_t *)b, (uint8_t *)o, n0, c0, h0, w0, n1, c1, h1,
        w1, n2, c2, h2, w2, multiplier, rshift, qdm, relu);
  } else if (a_sign && b_sign && !o_sign) {
    kernelMulBinaryInt8<<<num_blocks, block_size>>>(
        (int8_t *)a, (int8_t *)b, (uint8_t *)o, n0, c0, h0, w0, n1, c1, h1, w1,
        n2, c2, h2, w2, multiplier, rshift, qdm, relu);
  } else if (a_sign && !b_sign && o_sign) {
    kernelMulBinaryInt8<<<num_blocks, block_size>>>(
        (int8_t *)a, (uint8_t *)b, (int8_t *)o, n0, c0, h0, w0, n1, c1, h1, w1,
        n2, c2, h2, w2, multiplier, rshift, qdm, relu);
  } else if (!a_sign && b_sign && o_sign) {
    kernelMulBinaryInt8<<<num_blocks, block_size>>>(
        (uint8_t *)a, (int8_t *)b, (int8_t *)o, n0, c0, h0, w0, n1, c1, h1, w1,
        n2, c2, h2, w2, multiplier, rshift, qdm, relu);
  } else if (a_sign && !b_sign && !o_sign) {
    kernelMulBinaryInt8<<<num_blocks, block_size>>>(
        (int8_t *)a, (uint8_t *)b, (uint8_t *)o, n0, c0, h0, w0, n1, c1, h1, w1,
        n2, c2, h2, w2, multiplier, rshift, qdm, relu);
  } else if (!a_sign && b_sign && !o_sign) {
    kernelMulBinaryInt8<<<num_blocks, block_size>>>(
        (uint8_t *)a, (int8_t *)b, (uint8_t *)o, n0, c0, h0, w0, n1, c1, h1, w1,
        n2, c2, h2, w2, multiplier, rshift, qdm, relu);
  }
}

template <typename T0, typename T1, typename T2>
__global__ void kernelAddInt8(T0 *a, T1 *b, T2 *out, int32_t mul0, int32_t mul1,
                              int shift0, int shift1, int size, bool relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int32_t a_data = static_cast<int32_t>(a[idx]) * mul0;
    a_data = (a_data + (1 << (shift0 - 1))) >> shift0;
    int32_t b_data = static_cast<int32_t>(b[idx]) * mul1;
    b_data = (b_data + (1 << (shift1 - 1))) >> shift1;
    a_data = a_data + b_data;
    if (std::is_same<T2, int8_t>::value) {
      int32_t min_ = relu ? 0 : -128;
      a_data = max(min_, min(127, a_data));
      out[idx] = static_cast<int8_t>(a_data);
    } else {
      a_data = max(0, min(255, a_data));
      out[idx] = static_cast<uint8_t>(a_data);
    }
  }
}

void cudaAddInt8(void *input0, void *input1, void *output, int mul0, int mul1,
                 int shift0, int shift1, bool a_sign, bool b_sign, bool o_sign,
                 int size, bool relu) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  if (a_sign && b_sign && o_sign) {
    kernelAddInt8<<<num_blocks, block_size>>>(
        (int8_t *)input0, (int8_t *)input1, (int8_t *)output, mul0, mul1,
        shift0, shift1, size, relu);
  } else if (!a_sign && b_sign && o_sign) {
    kernelAddInt8<<<num_blocks, block_size>>>(
        (uint8_t *)input0, (int8_t *)input1, (int8_t *)output, mul0, mul1,
        shift0, shift1, size, relu);
  } else if (a_sign && !b_sign && o_sign) {
    kernelAddInt8<<<num_blocks, block_size>>>(
        (int8_t *)input0, (uint8_t *)input1, (int8_t *)output, mul0, mul1,
        shift0, shift1, size, relu);
  } else if (a_sign && b_sign && !o_sign) {
    kernelAddInt8<<<num_blocks, block_size>>>(
        (int8_t *)input0, (int8_t *)input1, (uint8_t *)output, mul0, mul1,
        shift0, shift1, size, relu);
  } else if (!a_sign && !b_sign && o_sign) {
    kernelAddInt8<<<num_blocks, block_size>>>(
        (uint8_t *)input0, (uint8_t *)input1, (int8_t *)output, mul0, mul1,
        shift0, shift1, size, relu);
  } else if (!a_sign && b_sign && !o_sign) {
    kernelAddInt8<<<num_blocks, block_size>>>(
        (uint8_t *)input0, (int8_t *)input1, (uint8_t *)output, mul0, mul1,
        shift0, shift1, size, relu);
  } else if (a_sign && !b_sign && !o_sign) {
    kernelAddInt8<<<num_blocks, block_size>>>(
        (int8_t *)input0, (uint8_t *)input1, (uint8_t *)output, mul0, mul1,
        shift0, shift1, size, relu);
  } else if (!a_sign && !b_sign && !o_sign) {
    kernelAddInt8<<<num_blocks, block_size>>>(
        (uint8_t *)input0, (uint8_t *)input1, (uint8_t *)output, mul0, mul1,
        shift0, shift1, size, relu);
  }
}

__global__ void kernelPad4D(void *input, void *output, int n, int c, int h,
                            int w, int pad_h_t, int pad_h_b, int pad_w_l,
                            int pad_w_r, int tbytes) {
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
      kernelCopyElement(input, in_idx, output, out_idx, tbytes);
    } else {
      kernelSetZero(output, out_idx, tbytes);
    }
  }
}

void cudaPad4D(void *input, void *output, int n, int c, int h, int w,
               int pad_h_t, int pad_h_b, int pad_w_l, int pad_w_r, int tbytes) {
  int oh = h + pad_h_t + pad_h_b;
  int ow = w + pad_w_l + pad_w_r;
  int num_blocks = CUDA_NUM_BLOCKS(n * c * oh * ow);
  int block_size = CUDA_BLOCK_SIZE;
  kernelPad4D<<<num_blocks, block_size>>>(input, output, n, c, h, w, pad_h_t,
                                          pad_h_b, pad_w_l, pad_w_r, tbytes);
}

template <typename T>
__global__ void kernelNegative(T *input, T *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = -input[idx];
  }
}

void cudaNegative(void *input, void *output, int size, cudnnDataType_t type) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  switch (type) {
  case CUDNN_DATA_INT32:
    kernelNegative<<<num_blocks, block_size>>>((int32_t *)input,
                                               (int32_t *)output, size);
    break;
  case CUDNN_DATA_FLOAT:
    kernelNegative<<<num_blocks, block_size>>>((float *)input, (float *)output,
                                               size);
    break;
  case CUDNN_DATA_INT8:
    kernelNegative<<<num_blocks, block_size>>>((int8_t *)input,
                                               (int8_t *)output, size);
    break;
  default:
    break;
  }
}

__global__ void kernelPermute4D(void *input, void *output, int n, int c, int h,
                                int w, int o0, int o1, int o2, int o3,
                                int tbytes) {
  int oldIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (oldIdx < n * c * h * w) {
    int dims[4] = {n, c, h, w};
    int newDims[4] = {dims[o0], dims[o1], dims[o2], dims[o3]};
    int ind[4];
    ind[0] = oldIdx / (c * h * w);             // n index
    ind[1] = (oldIdx % (c * h * w)) / (h * w); // c index
    ind[2] = (oldIdx % (h * w)) / w;           // h index
    ind[3] = oldIdx % w;                       // w index
    int newInd[4] = {ind[o0], ind[o1], ind[o2], ind[o3]};
    int newIdx =
        ((newInd[0] * newDims[1] + newInd[1]) * newDims[2] + newInd[2]) *
            newDims[3] +
        newInd[3];
    kernelCopyElement(input, oldIdx, output, newIdx, tbytes);
  }
}

void cudaPermute4D(void *src, void *dst, int n, int c, int h, int w, int o0,
                   int o1, int o2, int o3, int tbytes) {
  int num = n * c * h * w;
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  kernelPermute4D<<<num_blocks, block_size>>>(src, dst, n, c, h, w, o0, o1, o2,
                                              o3, tbytes);
}

__global__ void kernelSlice4D(void *src, void *dst, int n, int c, int h, int w,
                              int off0, int off1, int off2, int off3, int s0,
                              int s1, int s2, int s3, int on, int oc, int oh,
                              int ow, int tbytes) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_n = dst_idx / (oc * oh * ow);
  int idx_c = dst_idx % (oc * oh * ow) / (oh * ow);
  int idx_h = dst_idx % (oh * ow) / ow;
  int idx_w = dst_idx % ow;
  if (idx_w < ow && idx_h < oh && idx_c < oc && idx_n < on) {
    idx_n = off0 + idx_n * s0;
    idx_c = off1 + idx_c * s1;
    idx_h = off2 + idx_h * s2;
    idx_w = off3 + idx_w * s3;
    if (idx_n < n && idx_c < c && idx_h < h && idx_w < w) {
      int src_idx = ((idx_n * c + idx_c) * h + idx_h) * w + idx_w;
      kernelCopyElement(src, src_idx, dst, dst_idx, tbytes);
    }
  }
}

void cudaSlice4D(void *src, void *dst, int n, int c, int h, int w, int off0,
                 int off1, int off2, int off3, int s0, int s1, int s2, int s3,
                 int on, int oc, int oh, int ow, int tbytes) {
  int num_blocks = CUDA_NUM_BLOCKS(on * oc * oh * ow);
  int block_size = CUDA_BLOCK_SIZE;
  kernelSlice4D<<<num_blocks, block_size>>>(src, dst, n, c, h, w, off0, off1,
                                            off2, off3, s0, s1, s2, s3, on, oc,
                                            oh, ow, tbytes);
}

__global__ void kernelCopyAxis(void *src, void *dst, int outer_dim,
                               int axis_dim, int inner_dim, int offset, int num,
                               int tbytes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer_dim * num * inner_dim;
  if (idx < total) {
    int out_idx = idx / (num * inner_dim);
    int axis_idx = (idx % (num * inner_dim)) / inner_dim;
    int inner_idx = idx % inner_dim;
    int dstIdx = out_idx * axis_dim * inner_dim +
                 (axis_idx + offset) * inner_dim + inner_idx;
    kernelCopyElement(src, idx, dst, dstIdx, tbytes);
  }
}

void cudaCopyAxis(void *src, void *dst, int outer_dim, int axis_dim,
                  int inner_dim, int offset, int num, int tbytes) {
  int total = outer_dim * num * inner_dim;
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;
  kernelCopyAxis<<<num_blocks, block_size>>>(src, dst, outer_dim, axis_dim,
                                             inner_dim, offset, num, tbytes);
}

__global__ void kernelMatMulF32(float *A, float *B, float *C, int m, int k,
                                int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (m * n)) {
    int row = idx / n;
    int col = idx % n;
    float sum = 0.0;
    for (int i = 0; i < k; i++) {
      sum += A[row * k + i] * B[i * n + col];
    }
    C[row * n + col] = sum;
  }
}

void cudaMatMulF32(void *input, void *right, void *output, int m, int k,
                   int n) {
  // Dimensions for blocks and grid
  int num_blocks = CUDA_NUM_BLOCKS(m * n);
  int block_size = CUDA_BLOCK_SIZE;
  kernelMatMulF32<<<num_blocks, block_size>>>((float *)input, (float *)right,
                                              (float *)output, m, k, n);
}

__global__ void kernelRequantInt8Perchannel(int32_t *input, void *output,
                                            int32_t *multipliers,
                                            int32_t *shifts, int n, int c,
                                            int h, int w, bool out_sign,
                                            bool qdm, bool relu) {
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

void cudaRequantInt8Perchannel(void *input, void *output, void *multipliers,
                               void *shifts, int n, int c, int h, int w,
                               bool out_sign, bool qdm, bool relu) {
  int num_blocks = CUDA_NUM_BLOCKS(n * c * h * w);
  int block_size = CUDA_BLOCK_SIZE;
  kernelRequantInt8Perchannel<<<num_blocks, block_size>>>(
      (int32_t *)input, output, (int32_t *)multipliers, (int32_t *)shifts, n, c,
      h, w, out_sign, qdm, relu);
}

__global__ void kernelRequantInt8(int32_t *input, void *output,
                                  int32_t multiplier, int32_t shift, int num,
                                  bool out_sign, bool qdm, bool relu) {
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

void cudaRequantInt8(void *input, void *output, int32_t multiplier,
                     int32_t shift, int num, bool out_sign, bool qdm,
                     bool relu) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  kernelRequantInt8<<<num_blocks, block_size>>>(
      (int32_t *)input, output, multiplier, shift, num, out_sign, qdm, relu);
}

__global__ void kernelCVMultiShiftInt8(int8_t *input, int8_t *output,
                                       int multiplier, int shift, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int32_t value = static_cast<int32_t>(input[idx]) * multiplier;
    value = (value + (1 << (shift - 1))) >> shift; // half up
    value = max(-128, min(127, value));
    output[idx] = static_cast<int8_t>(value);
  }
}

void cudaCVMultiShiftInt8(void *input, void *output, int multiplier, int shift,
                          int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  kernelCVMultiShiftInt8<<<num_blocks, block_size>>>(
      (int8_t *)input, (int8_t *)output, multiplier, shift, size);
}

template <typename T>
__global__ void kernelMulShift(T *input, T *output, int multiplier, int shift,
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

void cudaMulShift(void *input, void *output, int multiplier, int shift,
                  int size, cudnnDataType_t type) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  switch (type) {
  case CUDNN_DATA_INT8:
    kernelMulShift<<<num_blocks, block_size>>>(
        (int8_t *)input, (int8_t *)output, multiplier, shift, size);
    break;
  case CUDNN_DATA_UINT8:
    kernelMulShift<<<num_blocks, block_size>>>(
        (uint8_t *)input, (uint8_t *)output, multiplier, shift, size);
    break;
  }
}

__global__ void kernelInt32ToFloat(int32_t *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = static_cast<float>(input[idx]);
  }
}

__global__ void kernelFloatToInt32(float *input, int32_t *output, int size,
                                   cuda_rmode_t rmode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = kernelInt<int32_t>(input[idx], rmode);
  }
}

__global__ void kernelInt8ToFloat(int8_t *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = static_cast<float>(input[idx]);
  }
}

__global__ void kernelFloatToInt8(float *input, int8_t *output, int size,
                                  cuda_rmode_t rmode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = kernelInt<int8_t>(input[idx], rmode);
  }
}

__global__ void kernelUint8ToFloat(uint8_t *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = static_cast<float>(input[idx]);
  }
}

__global__ void kernelFloatToUint8(float *input, uint8_t *output, int size,
                                   cuda_rmode_t rmode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = kernelInt<uint8_t>(input[idx], rmode);
  }
}

__global__ void kernelFloatToBFloat16(float *input, uint16_t *output, int size,
                                      cuda_rmode_t rmode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    auto df16 = bfloat16(input[idx], rmode == CUDA_HALF_UP);
    output[idx] = df16.value;
  }
}

__global__ void kernelBFloat16ToFloat(uint16_t *input, float *output,
                                      int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    bfloat16 data;
    data.value = input[idx];
    output[idx] = static_cast<float>(data);
  }
}

cudaError_t cudaTransform(void *src, void *dst, int size,
                          cudnnDataType_t src_type, cudnnDataType_t dst_type,
                          cuda_rmode_t rmode) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  if (src_type == CUDNN_DATA_FLOAT && dst_type == CUDNN_DATA_INT32) {
    kernelFloatToInt32<<<num_blocks, block_size>>>((float *)src, (int32_t *)dst,
                                                   size, rmode);
  } else if (src_type == CUDNN_DATA_INT32 && dst_type == CUDNN_DATA_FLOAT) {
    kernelInt32ToFloat<<<num_blocks, block_size>>>((int32_t *)src, (float *)dst,
                                                   size);
  } else if (src_type == CUDNN_DATA_FLOAT && dst_type == CUDNN_DATA_INT8) {
    kernelFloatToInt8<<<num_blocks, block_size>>>((float *)src, (int8_t *)dst,
                                                  size, rmode);
  } else if (src_type == CUDNN_DATA_INT8 && dst_type == CUDNN_DATA_FLOAT) {
    kernelInt8ToFloat<<<num_blocks, block_size>>>((int8_t *)src, (float *)dst,
                                                  size);
  } else if (src_type == CUDNN_DATA_FLOAT && dst_type == CUDNN_DATA_UINT8) {
    kernelFloatToUint8<<<num_blocks, block_size>>>((float *)src, (uint8_t *)dst,
                                                   size, rmode);
  } else if (src_type == CUDNN_DATA_UINT8 && dst_type == CUDNN_DATA_FLOAT) {
    kernelUint8ToFloat<<<num_blocks, block_size>>>((uint8_t *)src, (float *)dst,
                                                   size);
  } else if (src_type == CUDNN_DATA_FLOAT && dst_type == CUDNN_DATA_BFLOAT16) {
    kernelFloatToBFloat16<<<num_blocks, block_size>>>(
        (float *)src, (uint16_t *)dst, size, rmode);
  } else if (src_type == CUDNN_DATA_BFLOAT16 && dst_type == CUDNN_DATA_FLOAT) {
    kernelBFloat16ToFloat<<<num_blocks, block_size>>>((uint16_t *)src,
                                                      (float *)dst, size);
  } else {
    // not implemented
    return cudaErrorNotSupported;
  }
  return cudaSuccess;
}

template <typename T> __global__ void kernelPrint(T *data, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    printf("Data[%d] = %g\n", idx, (float)data[idx]);
  }
}

__global__ void kernelPrintBF16(uint16_t *data, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    printf("Data[%d] = %g\n", idx, kernel_BF16(data[idx]));
  }
}

void cudaPrint(void *data, int size, cudnnDataType_t type) {
  switch (type) {
  case CUDNN_DATA_FLOAT:
    kernelPrint<<<(size + 256) / 256, 256>>>((float *)data, size);
    break;
  case CUDNN_DATA_INT32:
    kernelPrint<<<(size + 256) / 256, 256>>>((int32_t *)data, size);
    break;
  case CUDNN_DATA_INT8:
    kernelPrint<<<(size + 256) / 256, 256>>>((int8_t *)data, size);
    break;
  case CUDNN_DATA_UINT8:
    kernelPrint<<<(size + 256) / 256, 256>>>((uint8_t *)data, size);
    break;
  case CUDNN_DATA_BFLOAT16:
    kernelPrintBF16<<<(size + 256) / 256, 256>>>((uint16_t *)data, size);
    break;
  }
}

template <typename T> __global__ void kernelRelu(T *data, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    data[idx] = max(static_cast<T>(0), data[idx]);
  }
}

void cudaRelu(void *data, int size, cudnnDataType_t type) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  switch (type) {
  case CUDNN_DATA_FLOAT:
    kernelRelu<<<num_blocks, block_size>>>((float *)data, size);
    break;
  case CUDNN_DATA_INT32:
    kernelRelu<<<num_blocks, block_size>>>((int32_t *)data, size);
    break;
  case CUDNN_DATA_INT8:
    kernelRelu<<<num_blocks, block_size>>>((int8_t *)data, size);
    break;
  }
}

template <typename T>
__global__ void kernelMaxAxis(T *input, T *output, int outer_dim, int axis_dim,
                              int inner_dim) {
  int inner_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int outer_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (inner_idx < inner_dim && outer_idx < outer_dim) {
    int outer_offset = outer_idx * axis_dim * inner_dim;
    // find max
    T max_v = input[outer_offset + inner_idx];
    for (int i = 1; i < axis_dim; i++) {
      T v = input[outer_offset + inner_idx + i * inner_dim];
      if (v > max_v) {
        v = max_v;
      }
    }
    output[outer_idx * inner_dim + inner_idx] = max_v;
  }
}

__global__ void kernelMaxAxisBF16(uint16_t *input, uint16_t *output,
                                  int outer_dim, int axis_dim, int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (outer_dim * inner_dim)) {
    int outer_idx = idx / inner_dim;
    int inner_idx = idx % inner_dim;
    int outer_offset = outer_idx * axis_dim * inner_dim;
    // find max
    bfloat16 max_v(input[outer_offset + inner_idx]);
    for (int i = 1; i < axis_dim; i++) {
      bfloat16 v(input[outer_offset + inner_idx + i * inner_dim]);
      if (static_cast<float>(max_v) < static_cast<float>(v)) {
        max_v.value = v.value;
      }
    }
    output[outer_idx * inner_dim + inner_idx] = max_v.value;
  }
}

void cudaMaxAxis(void *input, void *output, int outer_dim, int axis_dim,
                 int inner_dim, cudnnDataType_t type) {
  int num_blocks = CUDA_NUM_BLOCKS(inner_dim * outer_dim);
  int block_size = CUDA_BLOCK_SIZE;
  if (type == CUDNN_DATA_BFLOAT16) {
    kernelMaxAxisBF16<<<num_blocks, block_size>>>(
        (uint16_t *)input, (uint16_t *)output, outer_dim, axis_dim, inner_dim);
  } else if (type == CUDNN_DATA_INT8) {
    kernelMaxAxis<<<num_blocks, block_size>>>((int8_t *)input, (int8_t *)output,
                                              outer_dim, axis_dim, inner_dim);
  } else if (type == CUDNN_DATA_UINT8) {
    kernelMaxAxis<<<num_blocks, block_size>>>(
        (uint8_t *)input, (uint8_t *)output, outer_dim, axis_dim, inner_dim);
  } else if (type == CUDNN_DATA_FLOAT) {
    kernelMaxAxis<<<num_blocks, block_size>>>((float *)input, (float *)output,
                                              outer_dim, axis_dim, inner_dim);
  } else if (type == CUDNN_DATA_INT32) {
    kernelMaxAxis<<<num_blocks, block_size>>>(
        (int32_t *)input, (int32_t *)output, outer_dim, axis_dim, inner_dim);
  } else {
  }
}

template <typename T>
__global__ void kernelSumAxis(T *input, T *output, int outer_dim, int axis_dim,
                              int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (outer_dim * inner_dim)) {
    int outer_idx = idx / inner_dim;
    int inner_idx = idx % inner_dim;
    int outer_offset = outer_idx * axis_dim * inner_dim;
    // find max
    T sum = 0;
    for (int i = 0; i < axis_dim; i++) {
      sum += input[outer_offset + inner_idx + i * inner_dim];
    }
    output[outer_idx * inner_dim + inner_idx] = sum;
  }
}

__global__ void kernelSumAxisBF16(uint16_t *input, uint16_t *output,
                                  int outer_dim, int axis_dim, int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (outer_dim * inner_dim)) {
    int outer_idx = idx / inner_dim;
    int inner_idx = idx % inner_dim;
    int outer_offset = outer_idx * axis_dim * inner_dim;
    // find max
    float sum = 0.0f;
    for (int i = 0; i < axis_dim; i++) {
      sum += kernel_BF16(input[outer_offset + inner_idx + i * inner_dim]);
    }
    bfloat16 sum_bf16(sum, true);
    output[outer_idx * inner_dim + inner_idx] = sum_bf16.value;
  }
}

void cudaSumAxis(void *input, void *output, int outer_dim, int axis_dim,
                 int inner_dim, cudnnDataType_t type) {
  int num_blocks = CUDA_NUM_BLOCKS(outer_dim * inner_dim);
  int block_size = CUDA_BLOCK_SIZE;
  if (type == CUDNN_DATA_BFLOAT16) {
    kernelSumAxisBF16<<<num_blocks, block_size>>>(
        (uint16_t *)input, (uint16_t *)output, outer_dim, axis_dim, inner_dim);
  } else if (type == CUDNN_DATA_FLOAT) {
    kernelSumAxis<<<num_blocks, block_size>>>((float *)input, (float *)output,
                                              outer_dim, axis_dim, inner_dim);
  } else if (type == CUDNN_DATA_INT32) {
    kernelSumAxis<<<num_blocks, block_size>>>(
        (int32_t *)input, (int32_t *)output, outer_dim, axis_dim, inner_dim);
  } else {
  }
}

template <typename T>
__global__ void kernelSubAxis(T *input, T *sub, T *output, int outer_dim,
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

__global__ void kernelSubAxisBF16(uint16_t *input, uint16_t *sub,
                                  uint16_t *output, int outer_dim, int axis_dim,
                                  int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int outer_idx = idx / (axis_dim * inner_dim);
  int axis_idx = idx % (axis_dim * inner_dim) / inner_dim;
  int inner_idx = idx % inner_dim;
  if (inner_idx < inner_dim && outer_idx < outer_dim && axis_idx < axis_dim) {
    int sub_idx = outer_idx * inner_dim + inner_idx;
    bfloat16 in_data(input[idx]);
    bfloat16 sub_data(sub[sub_idx]);
    float out = static_cast<float>(in_data) - static_cast<float>(sub_data);
    bfloat16 out_f16 = bfloat16(out, true);
    output[idx] = out_f16.value;
  }
}

void cudaSubAxis(void *input, void *sub, void *output, int outer_dim,
                 int axis_dim, int inner_dim, cudnnDataType_t type) {
  int num_blocks = CUDA_NUM_BLOCKS(outer_dim * axis_dim * inner_dim);
  int block_size = CUDA_BLOCK_SIZE;
  if (type == CUDNN_DATA_BFLOAT16) {
    kernelSubAxisBF16<<<num_blocks, block_size>>>(
        (uint16_t *)input, (uint16_t *)sub, (uint16_t *)output, outer_dim,
        axis_dim, inner_dim);
  } else if (type == CUDNN_DATA_INT8) {
    kernelSubAxis<<<num_blocks, block_size>>>((int8_t *)input, (int8_t *)sub,
                                              (int8_t *)output, outer_dim,
                                              axis_dim, inner_dim);
  } else if (type == CUDNN_DATA_UINT8) {
    kernelSubAxis<<<num_blocks, block_size>>>((uint8_t *)input, (uint8_t *)sub,
                                              (uint8_t *)output, outer_dim,
                                              axis_dim, inner_dim);
  } else if (type == CUDNN_DATA_FLOAT) {
    kernelSubAxis<<<num_blocks, block_size>>>((float *)input, (float *)sub,
                                              (float *)output, outer_dim,
                                              axis_dim, inner_dim);
  } else if (type == CUDNN_DATA_INT32) {
    kernelSubAxis<<<num_blocks, block_size>>>((int32_t *)input, (int32_t *)sub,
                                              (int32_t *)output, outer_dim,
                                              axis_dim, inner_dim);
  } else {
  }
}

template <typename T>
__global__ void kernelAddAxis(T *input, T *add, T *output, int outer_dim,
                              int axis_dim, int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (inner_dim * outer_dim * axis_dim)) {
    int outer_idx = idx / (axis_dim * inner_dim);
    int inner_idx = idx % inner_dim;
    int add_idx = outer_idx * inner_dim + inner_idx;
    output[idx] = input[idx] + add[add_idx];
  }
}

__global__ void kernelAddAxisBF16(uint16_t *input, uint16_t *add,
                                  uint16_t *output, int outer_dim, int axis_dim,
                                  int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (inner_dim * outer_dim * axis_dim)) {
    int outer_idx = idx / (axis_dim * inner_dim);
    int inner_idx = idx % inner_dim;
    int add_idx = outer_idx * inner_dim + inner_idx;
    bfloat16 in_data(input[idx]);
    bfloat16 add_data(add[add_idx]);
    float out = static_cast<float>(in_data) + static_cast<float>(add_data);
    bfloat16 out_f16 = bfloat16(out, true);
    output[idx] = out_f16.value;
  }
}

void cudaAddAxis(void *input, void *add, void *output, int outer_dim,
                 int axis_dim, int inner_dim, cudnnDataType_t type) {
  int num_blocks = CUDA_NUM_BLOCKS(outer_dim * axis_dim * inner_dim);
  int block_size = CUDA_BLOCK_SIZE;
  if (type == CUDNN_DATA_BFLOAT16) {
    kernelAddAxisBF16<<<num_blocks, block_size>>>(
        (uint16_t *)input, (uint16_t *)add, (uint16_t *)output, outer_dim,
        axis_dim, inner_dim);
  } else if (type == CUDNN_DATA_INT8) {
    kernelAddAxis<<<num_blocks, block_size>>>((int8_t *)input, (int8_t *)add,
                                              (int8_t *)output, outer_dim,
                                              axis_dim, inner_dim);
  } else if (type == CUDNN_DATA_UINT8) {
    kernelAddAxis<<<num_blocks, block_size>>>((uint8_t *)input, (uint8_t *)add,
                                              (uint8_t *)output, outer_dim,
                                              axis_dim, inner_dim);
  } else if (type == CUDNN_DATA_FLOAT) {
    kernelAddAxis<<<num_blocks, block_size>>>((float *)input, (float *)add,
                                              (float *)output, outer_dim,
                                              axis_dim, inner_dim);
  } else if (type == CUDNN_DATA_INT32) {
    kernelAddAxis<<<num_blocks, block_size>>>((int32_t *)input, (int32_t *)add,
                                              (int32_t *)output, outer_dim,
                                              axis_dim, inner_dim);
  } else {
  }
}

template <typename T>
__global__ void kernelMulAxis(T *input, T *mul, T *output, int outer_dim,
                              int axis_dim, int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int outer_idx = idx / (axis_dim * inner_dim);
  int axis_idx = idx % (axis_dim * inner_dim) / inner_dim;
  int inner_idx = idx % inner_dim;
  if (inner_idx < inner_dim && outer_idx < outer_dim && axis_idx < axis_dim) {
    int sub_idx = outer_idx * inner_dim * inner_idx;
    output[idx] = input[idx] + mul[sub_idx];
  }
}

__global__ void kernelMulAxisBF16(uint16_t *input, uint16_t *mul,
                                  uint16_t *output, int outer_dim, int axis_dim,
                                  int inner_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int outer_idx = idx / (axis_dim * inner_dim);
  int axis_idx = idx % (axis_dim * inner_dim) / inner_dim;
  int inner_idx = idx % inner_dim;
  if (inner_idx < inner_dim && outer_idx < outer_dim && axis_idx < axis_dim) {
    int sub_idx = outer_idx * inner_dim + inner_idx;
    bfloat16 in_data(input[idx]);
    bfloat16 sub_data(mul[sub_idx]);
    float out = static_cast<float>(in_data) * static_cast<float>(sub_data);
    bfloat16 out_f16 = bfloat16(out, true);
    output[idx] = out_f16.value;
  }
}

void cudaMulAxis(void *input, void *mul, void *output, int outer_dim,
                 int axis_dim, int inner_dim, cudnnDataType_t type) {
  int num_blocks = CUDA_NUM_BLOCKS(outer_dim * axis_dim * inner_dim);
  int block_size = CUDA_BLOCK_SIZE;
  if (type == CUDNN_DATA_BFLOAT16) {
    kernelMulAxisBF16<<<num_blocks, block_size>>>(
        (uint16_t *)input, (uint16_t *)mul, (uint16_t *)output, outer_dim,
        axis_dim, inner_dim);
  } else if (type == CUDNN_DATA_INT8) {
    kernelMulAxis<<<num_blocks, block_size>>>((int8_t *)input, (int8_t *)mul,
                                              (int8_t *)output, outer_dim,
                                              axis_dim, inner_dim);
  } else if (type == CUDNN_DATA_UINT8) {
    kernelMulAxis<<<num_blocks, block_size>>>((uint8_t *)input, (uint8_t *)mul,
                                              (uint8_t *)output, outer_dim,
                                              axis_dim, inner_dim);
  } else if (type == CUDNN_DATA_FLOAT) {
    kernelMulAxis<<<num_blocks, block_size>>>((float *)input, (float *)mul,
                                              (float *)output, outer_dim,
                                              axis_dim, inner_dim);
  } else if (type == CUDNN_DATA_INT32) {
    kernelMulAxis<<<num_blocks, block_size>>>((int32_t *)input, (int32_t *)mul,
                                              (int32_t *)output, outer_dim,
                                              axis_dim, inner_dim);
  } else {
  }
}

template <typename T0, typename T1>
__global__ void kernelLut256(T0 *src, T1 *table, T1 *dst, int size) {
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

void cudaLut256(void *src, void *table, void *dst, int size,
                cudnnDataType_t src_type, cudnnDataType_t dst_type) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  if (src_type == CUDNN_DATA_INT8 && dst_type == CUDNN_DATA_INT8) {
    kernelLut256<<<num_blocks, block_size>>>((int8_t *)src, (int8_t *)table,
                                             (int8_t *)dst, size);
  } else if (src_type == CUDNN_DATA_UINT8 && dst_type == CUDNN_DATA_UINT8) {
    kernelLut256<<<num_blocks, block_size>>>((uint8_t *)src, (uint8_t *)table,
                                             (uint8_t *)dst, size);
  } else if (src_type == CUDNN_DATA_INT8 && dst_type == CUDNN_DATA_UINT8) {
    kernelLut256<<<num_blocks, block_size>>>((int8_t *)src, (uint8_t *)table,
                                             (uint8_t *)dst, size);
  } else if (src_type == CUDNN_DATA_UINT8 && dst_type == CUDNN_DATA_INT8) {
    kernelLut256<<<num_blocks, block_size>>>((uint8_t *)src, (int8_t *)table,
                                             (int8_t *)dst, size);
  }
}

__global__ void kernelUpsample4D(void *input, void *output, int n, int c,
                                 int ih, int iw, int scale_h, int scale_w,
                                 int tbytes) {
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
    kernelCopyElement(input, src_idx, output, dst_idx, tbytes);
  }
}

void cudaUpsample4D(void *src, void *dst, int n, int c, int h, int w,
                    int scale_h, int scale_w, int tbytes) {
  int num_blocks = CUDA_NUM_BLOCKS(n * c * h * w * scale_h * scale_w);
  int block_size = CUDA_BLOCK_SIZE;
  kernelUpsample4D<<<num_blocks, block_size>>>(src, dst, n, c, h, w, scale_h,
                                               scale_w, tbytes);
}

__global__ void kernelDepth2Space(void *input, void *output, int in, int ic,
                                  int ih, int iw, int on, int oc, int oh,
                                  int ow, int instride, int icstride,
                                  int ihstride, int iwstride, int onstride,
                                  int ocstride, int ohstride, int owstride,
                                  int block_h, int block_w, bool crd,
                                  bool swap_cr, bool inversed, int tbytes) {
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
      kernelCopyElement(input, o_index, output, i_index, tbytes);
    } else {
      kernelCopyElement(input, i_index, output, o_index, tbytes);
    }
  }
}

void cudaDepth2Space(void *input, void *output, int in, int ic, int ih, int iw,
                     int on, int oc, int oh, int ow, int instride, int icstride,
                     int ihstride, int iwstride, int onstride, int ocstride,
                     int ohstride, int owstride, int block_h, int block_w,
                     bool crd, bool swap_cr, bool inversed, int tbytes) {
  int num_blocks = CUDA_NUM_BLOCKS(in * ic * ih * iw);
  int block_size = CUDA_BLOCK_SIZE;
  kernelDepth2Space<<<num_blocks, block_size>>>(
      input, output, in, ic, ih, iw, on, oc, oh, ow, instride, icstride,
      ihstride, iwstride, onstride, ocstride, ohstride, owstride, block_h,
      block_w, crd, swap_cr, inversed, tbytes);
}

__global__ void kernelCVSoftmax(uint16_t *input, uint16_t *output,
                                int outer_dim, int axis_dim, int inner_dim) {
  int inner_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int outer_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (inner_idx < inner_dim && outer_idx < outer_dim) {
    int outer_offset = outer_idx * axis_dim * inner_dim;
    // find max
    float max_bf16 = -__FLT_MAX__;
    for (int i = 0; i < axis_dim; i++) {
      bfloat16 data(input[outer_offset + inner_idx + i * inner_dim]);
      float v = static_cast<float>(data);
      if (v > max_bf16) {
        v = max_bf16;
      }
    }
    // sub max
    for (int i = 0; i < axis_dim; i++) {
      bfloat16 data(input[outer_offset + inner_idx + i * inner_dim]);
      float v = static_cast<float>(data);
      v -= max_bf16;
      bfloat16 out(v, true);
      output[outer_offset + inner_idx + i * inner_dim] = out.value;
    }
    //
  }
}

__device__ uint16_t kernel_bf16_lut_slope(uint16_t input, uint16_t *base_table,
                                          uint16_t *slope_table, float scale,
                                          float offset) {
  float in_bf16 = kernel_BF16(input);
  float in_rescale = kernel_BF16(in_bf16 - offset);
  in_rescale = kernel_BF16(in_rescale * scale);
  int in_i8 = kernelInt<int8_t>(in_rescale, CUDA_TOWARDS_ZERO);
  // get delta x (x - x0)
  float delta_x = kernel_BF16(in_rescale - static_cast<float>(in_i8));
  // get slope
  auto slope = slope_table[in_i8 & 0xff];
  // base y0 = f(x0)
  auto base = base_table[in_i8 & 0xff];
  float slope_f32 = kernel_BF16(slope);
  float base_f32 = kernel_BF16(base);
  float out = kernel_BF16(kernel_BF16(delta_x * slope_f32) + base_f32);
  bfloat16 out_bf16(out);
  return out_bf16.value;
}

__device__ uint16_t kernel_bf16_lut_mantissa(uint16_t input,
                                             uint16_t *exp_table,
                                             uint16_t *mantissa_table,
                                             bool is_log) {
  float val = kernel_BF16(input);
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
  float exponent = kernel_BF16(exp_table[exponentIndex]);
  float mantissa = kernel_BF16(mantissa_table[input & 0xff]);
  float out;
  if (is_log) {
    out = kernel_BF16(exponent + mantissa);
  } else {
    out = kernel_BF16(exponent * mantissa);
  }
  bfloat16 out_bf16(out, false);
  return out_bf16.value;
}

__global__ void kernelCVLutSlope(uint16_t *input, uint16_t *output,
                                 uint16_t *table0, uint16_t *table1, int num,
                                 float scale, float offset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    output[idx] =
        kernel_bf16_lut_slope(input[idx], table0, table1, scale, offset);
  }
}

void cudaCVLutSlope(void *input, void *output, void *table0, void *table1,
                    int num, float scale, float offset) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  kernelCVLutSlope<<<block_size, num_blocks>>>(
      (uint16_t *)input, (uint16_t *)output, (uint16_t *)table0,
      (uint16_t *)table1, num, scale, offset);
}

__global__ void kernelCVLutMantissa(uint16_t *input, uint16_t *output,
                                    uint16_t *table0, uint16_t *table1, int num,
                                    bool is_log) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    output[idx] = kernel_bf16_lut_mantissa(input[idx], table0, table1, is_log);
  }
}

void cudaCVLutMantissa(void *input, void *output, void *table0, void *table1,
                       int num, bool is_log) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  kernelCVLutMantissa<<<block_size, num_blocks>>>(
      (uint16_t *)input, (uint16_t *)output, (uint16_t *)table0,
      (uint16_t *)table1, num, is_log);
}

void cudaCVSoftmax(void *input, void *buffer, void *output, void *table0,
                   void *table1, void *table2, void *table3, int outer_dim,
                   int axis_dim, int inner_dim, float scale, float offset,
                   bool log) {
  // get max => buffer
  cudaMaxAxis(input, buffer, outer_dim, axis_dim, inner_dim,
              CUDNN_DATA_BFLOAT16);
  // sub max => output
  cudaSubAxis(input, buffer, output, outer_dim, axis_dim, inner_dim,
              CUDNN_DATA_BFLOAT16);

  // exp => output
  cudaCVLutSlope(output, output, table0, table1,
                 outer_dim * inner_dim * axis_dim, scale, offset);
  // sum => buffer
  cudaSumAxis(output, buffer, outer_dim, axis_dim, inner_dim,
              CUDNN_DATA_BFLOAT16);
  // 1/sum => buffer
  cudaCVLutMantissa(buffer, buffer, table2, table3, outer_dim * inner_dim, log);

  if (log) {
    cudaAddAxis(output, buffer, output, outer_dim, axis_dim, inner_dim,
                CUDNN_DATA_BFLOAT16);
  } else {
    cudaMulAxis(output, buffer, output, outer_dim, axis_dim, inner_dim,
                CUDNN_DATA_BFLOAT16);
  }
}
