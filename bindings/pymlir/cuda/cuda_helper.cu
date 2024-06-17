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
  unsigned short value;

  __device__ bfloat16() : value(0) {}
  __device__ bfloat16(float val) {
    unsigned int *pval = reinterpret_cast<unsigned int *>(&val);
    value = (*pval) >> 16;
  }

  __device__ operator float() const {
    unsigned int expanded = value << 16;
    return *reinterpret_cast<float *>(&expanded);
  }
};

__device__ bfloat16 kernelBF16Mul(bfloat16 a, bfloat16 b) {
  float af = static_cast<float>(a);
  float bf = static_cast<float>(b);
  return bfloat16(af * bf);
}

__global__ void kernelCVScaleToF32(int8_t *input, float *output, float scale,
                                   int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    // Step 1: Convert int8 to FP32
    float intermediate = static_cast<float>(input[idx]);

    // Step 2: Convert scale to BF16
    bfloat16 bf16_scale = bfloat16(scale);

    // Step 3: Multiply input by BF16 scale
    bfloat16 bf16_result = kernelBF16Mul(bfloat16(intermediate), bf16_scale);

    // Step 4: Convert BF16 back to FP32
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
