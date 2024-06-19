#include "cuda_helper.h"

#define CUDA_BLOCK_SIZE 256
#define CUDA_NUM_BLOCKS(n) ((n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE)

__global__ void kernelQuantizeToInt8_0(float *input, int8_t *output,
                                       float scale, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float scaled_value = input[idx] * scale;
    // Apply 'half away from zero' rounding
    float rounded_value;
    if (scaled_value > 0) {
      rounded_value = floor(scaled_value + 0.5);
    } else {
      rounded_value = ceil(scaled_value - 0.5);
    }
    output[idx] = (int8_t)rounded_value;
  }
}

void cudaQuantizeToInt8_0(void *input, void *output, float scale, int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  kernelQuantizeToInt8_0<<<num_blocks, block_size>>>(
      (float *)input, (int8_t *)output, scale, size);
}

struct bfloat16 {
  uint16_t value;

  __device__ bfloat16() : value(0) {}
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

__device__ bfloat16 kernelBF16Mul(bfloat16 a, bfloat16 b) {
  float af = static_cast<float>(a);
  float bf = static_cast<float>(b);
  return bfloat16(af * bf, true);
}

__global__ void kernelCVScaleToF32(int8_t *input, float *output, float scale,
                                   int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    // Step 1: Convert int8 to FP32
    float intermediate = static_cast<float>(input[idx]);

    // Step 2: Multiply input by BF16 scale
    bfloat16 bf16_result =
        kernelBF16Mul(bfloat16(intermediate), bfloat16(scale));

    // Step 3: Convert BF16 back to FP32
    output[idx] = static_cast<float>(bf16_result);
  }
}

void cudaCVScaleToF32(void *input, void *output, float scale, int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  kernelCVScaleToF32<<<num_blocks, block_size>>>((int8_t *)input,
                                                 (float *)output, scale, size);
}

__global__ void kernelScaleToF32(int8_t *input, float *output, float scale,
                                 int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    // Convert int8 to float32 and scale
    output[idx] = static_cast<float>(input[idx]) * scale;
  }
}

void cudaScaleToF32(void *input, void *output, float scale, int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  kernelScaleToF32<<<num_blocks, block_size>>>((int8_t *)input, (float *)output,
                                               scale, size);
}

__global__ void kernelAddInt8(int8_t *a, int8_t *b, int8_t *out, int32_t mul0,
                              int32_t mul1, int shift, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int32_t temp = (int32_t)a[idx] * mul0 + (int32_t)b[idx] * mul1;
    temp = (temp + (1 << (shift - 1))) >> shift;
    out[idx] = (int8_t)temp;
  }
}

void cudaAddInt8(void *input0, void *input1, void *output, int mul0, int mul1,
                 int shift, int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  kernelAddInt8<<<num_blocks, block_size>>>((int8_t *)input0, (int8_t *)input1,
                                            (int8_t *)output, mul0, mul1, shift,
                                            size);
}
