//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "cuda_global.cuh"
#include "../pycuda.h"    //这里包含函数的声明

namespace tpu_mlir {
namespace cuda {
#define CUDA_BLOCK_SIZE 256
#define CUDA_NUM_BLOCKS(n) ((n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE)

// -------------------------------------------------------------------------
// ------- type convert functions
size_t get_dtype_bytes(data_type_t type) {
  switch (type) {
  case DT_F64:
    return 8;
  case DT_F32:
  case DT_INT32:
    return 4;
  case DT_F16:
  case DT_BF16:
  case DT_UINT16:
  case DT_INT16:
    return 2;
  case DT_INT8:
  case DT_UINT8:
  case DT_F8E4M3:
    return 1;
  default:
    return 1;
  }
}

void f32ScaleToInt8(void *input, void *output, float scale, int size, bool sign,
                    rounding_mode_t rmode) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_f32ScaleToInt8<<<num_blocks, block_size>>>((float *)input, output, scale,
                                               size, sign, rmode);
}

void bf16ScaleToInt8(void *input, void *output, float scale, int size,
                     bool sign, rounding_mode_t rmode) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_bf16ScaleToInt8<<<num_blocks, block_size>>>((uint16_t *)input, output,
                                                scale, size, sign, rmode);
}

void f16ScaleToInt8(void *input, void *output, float scale, int size, bool sign,
                    rounding_mode_t rmode) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_f16ScaleToInt8<<<num_blocks, block_size>>>((uint16_t *)input, output, scale,
                                               size, sign, rmode);
}

void int8ScaleToF32(void *input, void *output, float scale, int size,
                    bool sign) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_int8ScaleToF32<<<num_blocks, block_size>>>(input, (float *)output, scale,
                                               size, sign);
}

void int8ScaleToBF16(void *input, void *output, float scale, int size,
                     bool sign) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_int8ScaleToBF16<<<num_blocks, block_size>>>(input, (uint16_t *)output,
                                                scale, size, sign);
}

void int8ScaleToF16(void *input, void *output, float scale, int size,
                    bool sign) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_int8ScaleToF16<<<num_blocks, block_size>>>(input, (uint16_t *)output, scale,
                                               size, sign);
}

void int16ScaleToF32(void *input, void *output, float scale, int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_int16ScaleToF32<<<num_blocks, block_size>>>(input, (float *)output, scale,
                                               size);
}

void int16ScaleToBF16(void *input, void *output, float scale, int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_int16ScaleToBF16<<<num_blocks, block_size>>>(input, (uint16_t *)output, scale,
                                               size);
}

void int16ScaleToF16(void *input, void *output, float scale, int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_int16ScaleToF16<<<num_blocks, block_size>>>(input, (uint16_t *)output, scale,
                                               size);
}

cudaError_t convertType(void *src, void *dst, int num_elem,
                        data_type_t src_type, data_type_t dst_type,
                        rounding_mode_t rmode) {
  int num_blocks = CUDA_NUM_BLOCKS(num_elem);
  int block_size = CUDA_BLOCK_SIZE;
  if (src_type == DT_F32 && dst_type == DT_INT32) {
    g_f32ToInt<<<num_blocks, block_size>>>((float *)src, (int32_t *)dst,
                                           num_elem, rmode);
  } else if (src_type == DT_INT32 && dst_type == DT_F32) {
    g_intToF32<<<num_blocks, block_size>>>((int32_t *)src, (float *)dst,
                                           num_elem);
  } else if (src_type == DT_F32 && dst_type == DT_INT8) {
    g_f32ToInt<<<num_blocks, block_size>>>((float *)src, (int8_t *)dst,
                                           num_elem, rmode);
  } else if (src_type == DT_INT8 && dst_type == DT_F32) {
    g_intToF32<<<num_blocks, block_size>>>((int8_t *)src, (float *)dst,
                                           num_elem);
  } else if (src_type == DT_F32 && dst_type == DT_UINT8) {
    g_f32ToInt<<<num_blocks, block_size>>>((float *)src, (uint8_t *)dst,
                                           num_elem, rmode);
  } else if (src_type == DT_UINT8 && dst_type == DT_F32) {
    g_intToF32<<<num_blocks, block_size>>>((uint8_t *)src, (float *)dst,
                                           num_elem);
  } else if (src_type == DT_F32 && dst_type == DT_BF16) {
    g_f32ToBF16<<<num_blocks, block_size>>>((float *)src, (uint16_t *)dst,
                                            num_elem, rmode);
  } else if (src_type == DT_BF16 && dst_type == DT_F32) {
    g_bf16ToF32<<<num_blocks, block_size>>>((uint16_t *)src, (float *)dst,
                                            num_elem);
  } else if (src_type == DT_F32 && dst_type == DT_F16) {
    g_f32ToF16<<<num_blocks, block_size>>>((float *)src, (uint16_t *)dst,
                                           num_elem, cuda::RD_HALF_TO_EVEN);
  } else if (src_type == DT_F16 && dst_type == DT_F32) {
    g_f16ToF32<<<num_blocks, block_size>>>((uint16_t *)src, (float *)dst,
                                           num_elem);
  } else if (src_type == DT_F32 && dst_type == DT_UINT16) {
    g_f32ToInt<<<num_blocks, block_size>>>((float *)src, (uint16_t *)dst,
                                           num_elem, rmode);
  } else if (src_type == DT_F32 && dst_type == DT_INT16) {
    g_f32ToInt<<<num_blocks, block_size>>>((float *)src, (int16_t *)dst,
                                           num_elem, rmode);
  } else if (src_type == DT_UINT16 && dst_type == DT_F32) {
    g_intToF32<<<num_blocks, block_size>>>((uint16_t *)src, (float *)dst,
                                           num_elem);
  } else if (src_type == DT_INT16 && dst_type == DT_F32) {
    g_intToF32<<<num_blocks, block_size>>>((int16_t *)src, (float *)dst,
                                           num_elem);
  } else if (src_type == DT_F8E4M3 && dst_type == DT_F32) {
    g_f8ToF32<<<num_blocks, block_size>>>((uint8_t *)src, 1.0, (float *)dst,
                                           num_elem);
  } else {
    // not implemented
    return cudaErrorNotSupported;
  }
  return cudaSuccess;
}

// -------------------------------------------------------------------------
// ------- binary functions
void mulInt8(void *a, void *b, void *o, bool a_sign, bool b_sign, bool o_sign,
             int multiplier, int rshift, int size, bool qdm, bool relu) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  if (a_sign && b_sign && o_sign) {
    g_mulInt8<<<num_blocks, block_size>>>((int8_t *)a, (int8_t *)b, (int8_t *)o,
                                          multiplier, rshift, size, qdm, relu);
  } else if (!a_sign && !b_sign && !o_sign) {
    g_mulInt8<<<num_blocks, block_size>>>((uint8_t *)a, (uint8_t *)b,
                                          (uint8_t *)o, multiplier, rshift,
                                          size, qdm, relu);
  } else if (a_sign && b_sign && !o_sign) {
    g_mulInt8<<<num_blocks, block_size>>>((int8_t *)a, (int8_t *)b,
                                          (uint8_t *)o, multiplier, rshift,
                                          size, qdm, relu);
  }
}

void mulInt8(void *a, void *b, void *o, int n0, int c0, int h0, int w0, int n1,
             int c1, int h1, int w1, int n2, int c2, int h2, int w2,
             bool a_sign, bool b_sign, bool o_sign, int multiplier, int rshift,
             bool qdm, bool relu) {
  int num_blocks = CUDA_NUM_BLOCKS(n2 * c2 * h2 * w2);
  int block_size = CUDA_BLOCK_SIZE;
  if (a_sign && b_sign && o_sign) {
    g_mulInt8<<<num_blocks, block_size>>>(
        (int8_t *)a, (int8_t *)b, (int8_t *)o, n0, c0, h0, w0, n1, c1, h1, w1,
        n2, c2, h2, w2, multiplier, rshift, qdm, relu);
  } else if (!a_sign && !b_sign && !o_sign) {
    g_mulInt8<<<num_blocks, block_size>>>(
        (uint8_t *)a, (uint8_t *)b, (uint8_t *)o, n0, c0, h0, w0, n1, c1, h1,
        w1, n2, c2, h2, w2, multiplier, rshift, qdm, relu);
  } else if (a_sign && b_sign && !o_sign) {
    g_mulInt8<<<num_blocks, block_size>>>(
        (int8_t *)a, (int8_t *)b, (uint8_t *)o, n0, c0, h0, w0, n1, c1, h1, w1,
        n2, c2, h2, w2, multiplier, rshift, qdm, relu);
  } else if (a_sign && !b_sign && o_sign) {
    g_mulInt8<<<num_blocks, block_size>>>(
        (int8_t *)a, (uint8_t *)b, (int8_t *)o, n0, c0, h0, w0, n1, c1, h1, w1,
        n2, c2, h2, w2, multiplier, rshift, qdm, relu);
  } else if (!a_sign && b_sign && o_sign) {
    g_mulInt8<<<num_blocks, block_size>>>(
        (uint8_t *)a, (int8_t *)b, (int8_t *)o, n0, c0, h0, w0, n1, c1, h1, w1,
        n2, c2, h2, w2, multiplier, rshift, qdm, relu);
  } else if (a_sign && !b_sign && !o_sign) {
    g_mulInt8<<<num_blocks, block_size>>>(
        (int8_t *)a, (uint8_t *)b, (uint8_t *)o, n0, c0, h0, w0, n1, c1, h1, w1,
        n2, c2, h2, w2, multiplier, rshift, qdm, relu);
  } else if (!a_sign && b_sign && !o_sign) {
    g_mulInt8<<<num_blocks, block_size>>>(
        (uint8_t *)a, (int8_t *)b, (uint8_t *)o, n0, c0, h0, w0, n1, c1, h1, w1,
        n2, c2, h2, w2, multiplier, rshift, qdm, relu);
  }
}
/*
void add4DInt8(
    void *input0,    // 输入 A（int8/uint8）
    void *input1,    // 输入 B（int8/uint8）
    void *output,    // 输出 C
    int mul0, int mul1,  // 两个输入的量化乘数
    int shift0, int shift1, // 两个输入的量化右移位数
    bool a_sign,      // 输入A是否是有符号(int8)
    bool b_sign,      // 输入B是否是有符号(int8)
    bool o_sign,      // 输出是否是有符号(int8)
    bool relu,        // 加法后是否加ReLU
    int n0,c0,h0,w0,  // 输入A的形状
    int n1,c1,h1,w1,  // 输入B的形状
    int n2,c2,h2,w2   // 输出C的形状
)

    1.判断 输入 A、输入 B、输出 是 int8 还是 uint8
    2.把指针转成正确的类型：(int8_t) 或 (uint8_t)**
    3.启动 GPU 核函数 g_add4DInt8 执行真正的加法

*/
void add4DInt8(void *input0, void *input1, void *output, int mul0, int mul1,
               int shift0, int shift1, bool a_sign, bool b_sign, bool o_sign,
               bool relu, int n0, int c0, int h0, int w0, int n1, int c1,
               int h1, int w1, int n2, int c2, int h2, int w2) {
  int size = n2 * c2 * h2 * w2;
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  if (a_sign && b_sign && o_sign) {
    g_add4DInt8<<<num_blocks, block_size>>>(
        (int8_t *)input0, (int8_t *)input1, (int8_t *)output, mul0, mul1,
        shift0, shift1, relu, n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
  } else if (!a_sign && b_sign && o_sign) {
    g_add4DInt8<<<num_blocks, block_size>>>(
        (uint8_t *)input0, (int8_t *)input1, (int8_t *)output, mul0, mul1,
        shift0, shift1, relu, n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
  } else if (a_sign && !b_sign && o_sign) {
    g_add4DInt8<<<num_blocks, block_size>>>(
        (int8_t *)input0, (uint8_t *)input1, (int8_t *)output, mul0, mul1,
        shift0, shift1, relu, n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
  } else if (a_sign && b_sign && !o_sign) {
    g_add4DInt8<<<num_blocks, block_size>>>(
        (int8_t *)input0, (int8_t *)input1, (uint8_t *)output, mul0, mul1,
        shift0, shift1, relu, n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
  } else if (!a_sign && !b_sign && o_sign) {
    g_add4DInt8<<<num_blocks, block_size>>>(
        (uint8_t *)input0, (uint8_t *)input1, (int8_t *)output, mul0, mul1,
        shift0, shift1, relu, n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
  } else if (!a_sign && b_sign && !o_sign) {
    g_add4DInt8<<<num_blocks, block_size>>>(
        (uint8_t *)input0, (int8_t *)input1, (uint8_t *)output, mul0, mul1,
        shift0, shift1, relu, n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
  } else if (a_sign && !b_sign && !o_sign) {
    g_add4DInt8<<<num_blocks, block_size>>>(
        (int8_t *)input0, (uint8_t *)input1, (uint8_t *)output, mul0, mul1,
        shift0, shift1, relu, n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
  } else if (!a_sign && !b_sign && !o_sign) {
    g_add4DInt8<<<num_blocks, block_size>>>(
        (uint8_t *)input0, (uint8_t *)input1, (uint8_t *)output, mul0, mul1,
        shift0, shift1, relu, n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
  }
}

void add4DF32(void *input0, float scale0, void *input1, float scale1, void *output,
               bool relu, int n0, int c0, int h0, int w0, int n1, int c1,
               int h1, int w1, int n2, int c2, int h2, int w2) {
  int size = n2 * c2 * h2 * w2;
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_add4DF32<<<num_blocks, block_size>>>(
      (float *)input0, scale0, (float *)input1, scale1, (float *)output,
      relu, n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
}

void add4DInt32(int32_t *input0, int32_t *input1, int32_t *output,
               int n0, int c0, int h0, int w0, int n1, int c1,
               int h1, int w1, int n2, int c2, int h2, int w2) {
  int size = n2 * c2 * h2 * w2;
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_add4DInt32<<<num_blocks, block_size>>>(
      (int32_t *)input0, (int32_t *)input1, (int32_t *)output,
      n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
}


void bmCompare4DF32(void *lhs, void *rhs, void *output, int mode,
                    int n0, int c0, int h0, int w0,
                    int n1, int c1, int h1, int w1,
                    int n2, int c2, int h2, int w2) {
  int size = n2 * c2 * h2 * w2;
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_compare4DF32<<<num_blocks, block_size>>>(
      (float *)lhs, (float *)rhs, (float *)output,
      mode, n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
}

void bmCompareConst4DF32(void *input, float const_v, void *output,
                          int mode, bool inversed, int n, int c, int h, int w) {
  int size = n * c * h * w;
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_compareConst4DF32<<<num_blocks, block_size>>>(
      (float *)input, const_v, (float *)output, mode, inversed, n, c, h, w);
}

void sub4DF32(void *input0, void *input1, void *output,
               bool relu, bool reverse, int n0, int c0, int h0, int w0, int n1, int c1,
               int h1, int w1, int n2, int c2, int h2, int w2) {
  int size = n2 * c2 * h2 * w2;
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_sub4DF32<<<num_blocks, block_size>>>(
      (float *)input0, (float *)input1, (float *)output,
      relu, reverse, n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
}

void sub4DInt8(void *input0, bool input0_unsigned, int mul0, int shift0, void *input1, bool input1_unsigned, int mul1, int shift1, void *output, bool output_unsigned,
               bool relu, bool reverse, int n0, int c0, int h0, int w0, int n1, int c1,
               int h1, int w1, int n2, int c2, int h2, int w2) {
  int size = n2 * c2 * h2 * w2;
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  if (input0_unsigned && input1_unsigned) {
    g_sub4DInt8<<<num_blocks, block_size>>>(
        (uint8_t *)input0, mul0, shift0, (uint8_t *)input1, mul1, shift1, (int8_t *)output,
        relu, reverse, n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
  } else if (input0_unsigned && !input1_unsigned) {
    g_sub4DInt8<<<num_blocks, block_size>>>(
        (uint8_t *)input0, mul0, shift0, (int8_t *)input1, mul1, shift1, (int8_t *)output,
        relu, reverse, n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
  } else if (!input0_unsigned && input1_unsigned) {
    g_sub4DInt8<<<num_blocks, block_size>>>(
        (int8_t *)input0, mul0, shift0, (uint8_t *)input1, mul1, shift1, (int8_t *)output,
        relu, reverse, n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
  } else {
    g_sub4DInt8<<<num_blocks, block_size>>>(
        (int8_t *)input0, mul0, shift0, (int8_t *)input1, mul1, shift1, (int8_t *)output,
        relu, reverse, n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
  }
}

void mulConst4DF32(void *input, float const_v, void *output, bool do_relu,
                  int n0, int c0, int h0, int w0) {
  int size = n0 * c0 * h0 * w0;
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_mulConst4DF32<<<num_blocks, block_size>>>(
      (float *)input, const_v, (float *)output,
      do_relu, n0, c0, h0, w0);
}

void addConst4DF32(void *input, float const_v, void *output,
                    bool do_relu, int n, int c, int h, int w) {
  int size = n * c * h * w;
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_addConst4DF32<<<num_blocks, block_size>>>(
      (float *)input, const_v, (float *)output,
      do_relu, n, c, h, w);
}

void div4DF32(void *a, void *b, void *output, bool relu, bool reverse,
              int n0, int c0, int h0, int w0,
              int n1, int c1, int h1, int w1,
              int n2, int c2, int h2, int w2) {
  int size = n2 * c2 * h2 * w2;
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_div4DF32<<<num_blocks, block_size>>>(
      (float *)a, (float *)b, (float *)output,
      relu, reverse, n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
}

void divConst4DF32(void *input, float const_v, void *output,
                   bool do_relu, bool reverse, int n, int c, int h, int w) {
  int size = n * c * h * w;
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_divConst4DF32<<<num_blocks, block_size>>>(
      (float *)input, const_v, (float *)output, do_relu, reverse, n, c, h, w);
}

void subConst4DF32(void *input, float const_v, void *output,
               bool do_relu, bool reverse, int n, int c, int h, int w) {
  int size = n * c * h * w;
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_subConst4DF32<<<num_blocks, block_size>>>(
      (float *)input, const_v, (float *)output,
      do_relu, reverse, n, c, h, w);
}

void subConst4DI8(void *input, bool in_signed, int const_v, void *output,
               bool do_relu, bool reverse, int multi, int shift, int n, int c, int h, int w){
  int size = n * c * h * w;
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  if (in_signed)
    g_subConst4DI8<<<num_blocks, block_size>>>(
        (int8_t *)input, const_v, (int8_t *)output,
        do_relu, reverse, multi, shift, n, c, h, w);
  else
    g_subConst4DI8<<<num_blocks, block_size>>>(
        (uint8_t *)input, const_v, (int8_t *)output,
        do_relu, reverse, multi, shift, n, c, h, w);
}

void mul4DF32(void *input0, void *input1, void *output, bool do_relu,
                  int n0, int c0, int h0, int w0,
                  int n1, int c1, int h1, int w1,
                  int n2, int c2, int h2, int w2) {
  int size = n2 * c2 * h2 * w2;
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_mul4DF32<<<num_blocks, block_size>>>(
      (float *)input0, (float *)input1, (float *)output,
      do_relu, n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
}

void copyAxis(void *src, void *dst, int outer_dim, int axis_dim, int inner_dim,
              int offset, int num, int tbytes) {
  int total = outer_dim * num * inner_dim;
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;
  g_copyAxis<<<num_blocks, block_size>>>(src, dst, outer_dim, axis_dim,
                                         inner_dim, offset, num, tbytes);
}

void maxAxis(void *input, void *output, int outer_dim, int axis_dim,
             int inner_dim, data_type_t type) {
  int num_blocks = CUDA_NUM_BLOCKS(inner_dim * outer_dim);
  int block_size = CUDA_BLOCK_SIZE;
  if (type == DT_BF16) {
    g_maxAxisBF16<<<num_blocks, block_size>>>(
        (uint16_t *)input, (uint16_t *)output, outer_dim, axis_dim, inner_dim);
  } else if (type == DT_INT8) {
    g_maxAxis<<<num_blocks, block_size>>>((int8_t *)input, (int8_t *)output,
                                          outer_dim, axis_dim, inner_dim);
  } else if (type == DT_UINT8) {
    g_maxAxis<<<num_blocks, block_size>>>((uint8_t *)input, (uint8_t *)output,
                                          outer_dim, axis_dim, inner_dim);
  } else if (type == DT_F32) {
    g_maxAxis<<<num_blocks, block_size>>>((float *)input, (float *)output,
                                          outer_dim, axis_dim, inner_dim);
  } else if (type == DT_INT32) {
    g_maxAxis<<<num_blocks, block_size>>>((int32_t *)input, (int32_t *)output,
                                          outer_dim, axis_dim, inner_dim);
  } else {
  }
}

void sumAxis(void *input, void *output, int outer_dim, int axis_dim,
             int inner_dim, data_type_t type) {
  int num_blocks = CUDA_NUM_BLOCKS(outer_dim * inner_dim);
  int block_size = CUDA_BLOCK_SIZE;
  if (type == DT_BF16) {
    g_sumAxisBF16<<<num_blocks, block_size>>>(
        (uint16_t *)input, (uint16_t *)output, outer_dim, axis_dim, inner_dim);
  } else if (type == DT_F32) {
    g_sumAxis<<<num_blocks, block_size>>>((float *)input, (float *)output,
                                          outer_dim, axis_dim, inner_dim);
  } else if (type == DT_INT32) {
    g_sumAxis<<<num_blocks, block_size>>>((int32_t *)input, (int32_t *)output,
                                          outer_dim, axis_dim, inner_dim);
  } else {
  }
}

void subAxis(void *input, void *sub, void *output, int outer_dim, int axis_dim,
             int inner_dim, data_type_t type) {
  int num_blocks = CUDA_NUM_BLOCKS(outer_dim * axis_dim * inner_dim);
  int block_size = CUDA_BLOCK_SIZE;
  if (type == DT_BF16) {
    g_subAxisBF16<<<num_blocks, block_size>>>(
        (uint16_t *)input, (uint16_t *)sub, (uint16_t *)output, outer_dim,
        axis_dim, inner_dim);
  } else if (type == DT_INT8) {
    g_subAxis<<<num_blocks, block_size>>>((int8_t *)input, (int8_t *)sub,
                                          (int8_t *)output, outer_dim, axis_dim,
                                          inner_dim);
  } else if (type == DT_UINT8) {
    g_subAxis<<<num_blocks, block_size>>>((uint8_t *)input, (uint8_t *)sub,
                                          (uint8_t *)output, outer_dim,
                                          axis_dim, inner_dim);
  } else if (type == DT_F32) {
    g_subAxis<<<num_blocks, block_size>>>((float *)input, (float *)sub,
                                          (float *)output, outer_dim, axis_dim,
                                          inner_dim);
  } else if (type == DT_INT32) {
    g_subAxis<<<num_blocks, block_size>>>((int32_t *)input, (int32_t *)sub,
                                          (int32_t *)output, outer_dim,
                                          axis_dim, inner_dim);
  } else {
  }
}

void addAxis(void *input, void *add, void *output, int outer_dim, int axis_dim,
             int inner_dim, data_type_t type) {
  int num_blocks = CUDA_NUM_BLOCKS(outer_dim * axis_dim * inner_dim);
  int block_size = CUDA_BLOCK_SIZE;
  if (type == DT_BF16) {
    g_addAxisBF16<<<num_blocks, block_size>>>(
        (uint16_t *)input, (uint16_t *)add, (uint16_t *)output, outer_dim,
        axis_dim, inner_dim);
  } else if (type == DT_INT8) {
    g_addAxis<<<num_blocks, block_size>>>((int8_t *)input, (int8_t *)add,
                                          (int8_t *)output, outer_dim, axis_dim,
                                          inner_dim);
  } else if (type == DT_UINT8) {
    g_addAxis<<<num_blocks, block_size>>>((uint8_t *)input, (uint8_t *)add,
                                          (uint8_t *)output, outer_dim,
                                          axis_dim, inner_dim);
  } else if (type == DT_F32) {
    g_addAxis<<<num_blocks, block_size>>>((float *)input, (float *)add,
                                          (float *)output, outer_dim, axis_dim,
                                          inner_dim);
  } else if (type == DT_INT32) {
    g_addAxis<<<num_blocks, block_size>>>((int32_t *)input, (int32_t *)add,
                                          (int32_t *)output, outer_dim,
                                          axis_dim, inner_dim);
  } else {
  }
}

void mulAxis(void *input, void *mul, void *output, int outer_dim, int axis_dim,
             int inner_dim, data_type_t type) {
  int num_blocks = CUDA_NUM_BLOCKS(outer_dim * axis_dim * inner_dim);
  int block_size = CUDA_BLOCK_SIZE;
  if (type == DT_BF16) {
    g_mulAxisBF16<<<num_blocks, block_size>>>(
        (uint16_t *)input, (uint16_t *)mul, (uint16_t *)output, outer_dim,
        axis_dim, inner_dim);
  } else if (type == DT_INT8) {
    g_mulAxis<<<num_blocks, block_size>>>((int8_t *)input, (int8_t *)mul,
                                          (int8_t *)output, outer_dim, axis_dim,
                                          inner_dim);
  } else if (type == DT_UINT8) {
    g_mulAxis<<<num_blocks, block_size>>>((uint8_t *)input, (uint8_t *)mul,
                                          (uint8_t *)output, outer_dim,
                                          axis_dim, inner_dim);
  } else if (type == DT_F32) {
    g_mulAxis<<<num_blocks, block_size>>>((float *)input, (float *)mul,
                                          (float *)output, outer_dim, axis_dim,
                                          inner_dim);
  } else if (type == DT_INT32) {
    g_mulAxis<<<num_blocks, block_size>>>((int32_t *)input, (int32_t *)mul,
                                          (int32_t *)output, outer_dim,
                                          axis_dim, inner_dim);
  } else {
  }
}

void neg(void *input, void *output, int size, data_type_t type) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  switch (type) {
  case DT_INT32:
    g_neg<<<num_blocks, block_size>>>((int32_t *)input, (int32_t *)output,
                                      size);
    break;
  case DT_F32:
    g_neg<<<num_blocks, block_size>>>((float *)input, (float *)output, size);
    break;
  case DT_INT8:
    g_neg<<<num_blocks, block_size>>>((int8_t *)input, (int8_t *)output, size);
    break;
  default:
    break;
  }
}

// -------------------------------------------------------------------------
// ------- nn functions
void pad4D(void *input, void *output, int n, int c, int h, int w, int pad_h_t,
           int pad_h_b, int pad_w_l, int pad_w_r, int tbytes) {
  int oh = h + pad_h_t + pad_h_b;
  int ow = w + pad_w_l + pad_w_r;
  int num_blocks = CUDA_NUM_BLOCKS(n * c * oh * ow);
  int block_size = CUDA_BLOCK_SIZE;
  g_pad4D<<<num_blocks, block_size>>>(input, output, n, c, h, w, pad_h_t,
                                      pad_h_b, pad_w_l, pad_w_r, tbytes);
}
//add for conv3d
void pad5D(void *input, void *output,int n, int c,int d,int h,int w,int pad_d_f,
           int pad_d_b,int pad_h_t, int pad_h_b, int pad_w_l, int pad_w_r, int tbytes){
  int od = d + pad_d_f + pad_d_b;
  int oh = h + pad_h_t + pad_h_b;
  int ow = w + pad_w_l + pad_w_r;

  int num_blocks = CUDA_NUM_BLOCKS(n * c * oh * ow * od);
  int block_size = CUDA_BLOCK_SIZE;

  g_pad5D<<<num_blocks, block_size>>>(input, output, n, c, d, h, w, pad_d_f, pad_d_b, pad_h_t,
                                      pad_h_b, pad_w_l, pad_w_r, tbytes);

}





void permute6D(void *src, void *dst, int n, int c, int d, int h, int w, int d1, int o0, int o1,
               int o2, int o3, int o4, int o5, int tbytes) {
  int num = n * c * d * h * w * d1;
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_permute6D<<<num_blocks, block_size>>>(src, dst, n, c, d, h, w, d1, o0, o1, o2, o3, o4, o5,
                                          tbytes);
}

void slice6D(void *src, void *dst, int n, int c, int d, int h, int w, int d1, int off0,
             int off1, int off2, int off3, int off4, int off5, int s0, int s1, int s2, int s3,
             int s4, int s5, int on, int oc, int od, int oh, int ow, int od1, int tbytes) {
  int num_blocks = CUDA_NUM_BLOCKS(on * oc * od * oh * ow * od1);
  int block_size = CUDA_BLOCK_SIZE;
  g_slice6D<<<num_blocks, block_size>>>(src, dst, n, c, d, h, w, d1, off0, off1, off2,
                                        off3, off4, off5, s0, s1, s2, s3, s4, s5, on, oc, od, oh, ow,
                                        od1, tbytes);
}

void swapDimInner6D(void *src, void *dst, int n, int c, int d, int h, int w, int d1, int off0,
             int off1, int off2, int off3, int off4, int off5, int tbytes) {
  int num_blocks = CUDA_NUM_BLOCKS(n * c * d * h * w * d1);
  int block_size = CUDA_BLOCK_SIZE;
  int offset[] = {off0, off1, off2, off3, off4, off5};
  int shape[] = {n, c, d, h, w, d1};
  int num_axis = 0;
  for (int i=0;i<6; i++) {
    if (offset[i] > 0 )
      num_axis ++;
  }
  void *buffer;
  cudaMalloc(&buffer, sizeof(float)*n*c*d*h*w*d1);
  void *output[] = {buffer, dst};
  int processing = 0;
  for (int i=0;i<6; i++) {
    if (offset[i] == 0)
      continue;
    int outter = 1;
    int inner = 1;
    for (int j=0;j<i;j++)
      outter *= shape[j];
    for (int j=i+1;j<6;j++)
      inner *= shape[j];
    void * out = output[((processing & 1) + (num_axis & 1)) % 2];
    void * in = output[((processing % 2) + (num_axis & 1)) % 2];
    g_swapDimInner6D<<<num_blocks, block_size>>>(processing==0?src:in, out, outter, shape[i], offset[i], inner, tbytes);
    processing += 1;
  }
  cudaFree(buffer);
}

void tile4D(void *src, void *dst, int n, int c, int h, int w, int on, int oc,
            int oh, int ow, int tbytes) {
  int num_blocks = CUDA_NUM_BLOCKS(on * oc * oh * ow);
  int block_size = CUDA_BLOCK_SIZE;
  g_tile4D<<<num_blocks, block_size>>>(src, dst, n, c, h, w, on, oc, oh, ow,
                                       tbytes);
}

void mmF32(void *input, void *right, void *output, bool right_transpose, int m, int k, int n) {
  // Dimensions for blocks and grid
  int num_blocks = CUDA_NUM_BLOCKS(m * n);
  int block_size = CUDA_BLOCK_SIZE;
  g_mmF32<<<num_blocks, block_size>>>((float *)input, (float *)right,
                                      (float *)output, right_transpose, m, k, n);
}

void mmInt8(void *input, bool left_signed, void *right, bool right_signed, void *output, bool right_transpose, int m, int k, int n) {
  // Dimensions for blocks and grid
  int num_blocks = CUDA_NUM_BLOCKS(m * n);
  int block_size = CUDA_BLOCK_SIZE;
  if (left_signed && right_signed) {
    g_mmInt8<<<num_blocks, block_size>>>((int8_t *)input, (int8_t *)right,
                                        (int32_t *)output, right_transpose, m, k, n);
    return;
  } else if (left_signed && !right_signed) {
    g_mmInt8<<<num_blocks, block_size>>>((int8_t *)input, (uint8_t *)right,
                                        (int32_t *)output, right_transpose, m, k, n);
    return;
  } else if (!left_signed && right_signed) {
    g_mmInt8<<<num_blocks, block_size>>>((uint8_t *)input, (int8_t *)right,
                                        (int32_t *)output, right_transpose, m, k, n);
    return;
  } else if (!left_signed && !right_signed) {
    g_mmInt8<<<num_blocks, block_size>>>((uint8_t *)input, (uint8_t *)right,
                                        (int32_t *)output, right_transpose, m, k, n);
    return;
  }
}

void gather(void *indices, void *embedding, void *output, int num_indices,
            int embedding_dim, int inner_dim, data_type_t ind_type,
            data_type_t embed_type) {
  int num_blocks = CUDA_NUM_BLOCKS(num_indices);
  int block_size = CUDA_BLOCK_SIZE;
  auto dbytes = get_dtype_bytes(embed_type);
  if (ind_type == DT_UINT16) {
    if (dbytes == 1) {
      g_gather<<<num_blocks, block_size>>>(
          (uint16_t *)indices, (uint8_t *)embedding, (uint8_t *)output,
          num_indices, embedding_dim, inner_dim);
    } else if (dbytes == 2) {
      g_gather<<<num_blocks, block_size>>>(
          (uint16_t *)indices, (uint16_t *)embedding, (uint16_t *)output,
          num_indices, embedding_dim, inner_dim);
    } else if (dbytes == 4) {
      g_gather<<<num_blocks, block_size>>>(
          (uint16_t *)indices, (uint32_t *)embedding, (uint32_t *)output,
          num_indices, embedding_dim, inner_dim);
    }
  } else if (ind_type == DT_INT32) {
    if (dbytes == 1) {
      g_gather<<<num_blocks, block_size>>>(
          (int32_t *)indices, (uint8_t *)embedding, (uint8_t *)output,
          num_indices, embedding_dim, inner_dim);
    } else if (dbytes == 2) {
      g_gather<<<num_blocks, block_size>>>(
          (int32_t *)indices, (uint16_t *)embedding, (uint16_t *)output,
          num_indices, embedding_dim, inner_dim);
    } else if (dbytes == 4) {
      g_gather<<<num_blocks, block_size>>>(
          (int32_t *)indices, (uint32_t *)embedding, (uint32_t *)output,
          num_indices, embedding_dim, inner_dim);
    }
  } else if (ind_type == DT_F32) {
    if (dbytes == 1) {
      g_gather<<<num_blocks, block_size>>>(
          (float *)indices, (uint8_t *)embedding, (uint8_t *)output,
          num_indices, embedding_dim, inner_dim);
    } else if (dbytes == 2) {
      g_gather<<<num_blocks, block_size>>>(
          (float *)indices, (uint16_t *)embedding, (uint16_t *)output,
          num_indices, embedding_dim, inner_dim);
    } else if (dbytes == 4) {
      g_gather<<<num_blocks, block_size>>>(
          (float *)indices, (uint32_t *)embedding, (uint32_t *)output,
          num_indices, embedding_dim, inner_dim);
    }
  }
}

void bmDepth2Space(void *input, void *output, bool inversed, bool swap_hw, bool crd, int block_h, int block_w,
  int n, int c, int h, int w, int ins, int ics, int ihs, int iws,
  int on, int oc, int oh, int ow, int ons, int ocs, int ohs, int ows, data_type_t type)
{
  int num_blocks = CUDA_NUM_BLOCKS(n * c * h * w);
  int block_size = CUDA_BLOCK_SIZE;

  if (type == DT_INT8 || type == DT_UINT8) {
    g_depth2space<<<num_blocks, block_size>>>(
        (uint8_t *)input, (uint8_t *)output, block_h, block_w, inversed, swap_hw, crd, n, c, h, w, ins, ics, ihs, iws, on, oc, oh, ow, ons, ocs, ohs, ows);
    return;
  } else if (type == DT_F16 || type == DT_BF16) {
    g_depth2space<<<num_blocks, block_size>>>(
        (uint16_t *)input, (uint16_t *)output, block_h, block_w, inversed, swap_hw, crd, n, c, h, w, ins, ics, ihs, iws, on, oc, oh, ow, ons, ocs, ohs, ows);
    return;
  } else if (type == DT_F32) {
    g_depth2space<<<num_blocks, block_size>>>(
        (float *)input, (float *)output, block_h, block_w, inversed, swap_hw, crd, n, c, h, w, ins, ics, ihs, iws, on, oc, oh, ow, ons, ocs, ohs, ows);
    return;
  }


  // if (!inversed) {
  //   if (type == DT_INT8 || type == DT_UINT8) {
  //     depth_to_space_kernel<<<num_blocks, block_size>>>(
  //         (uint8_t *)input, (uint8_t *)output, block_h, block_w, swap_hw, crd, n, c, h, w);
  //     return;
  //   } else if (type == DT_F16 || type == DT_BF16) {
  //     depth_to_space_kernel<<<num_blocks, block_size>>>(
  //         (uint16_t *)input, (uint16_t *)output, block_h, block_w, swap_hw, crd, n, c, h, w);
  //     return;
  //   } else if (type == DT_F32) {
  //     depth_to_space_kernel<<<num_blocks, block_size>>>(
  //         (float *)input, (float *)output, block_h, block_w, swap_hw, crd, n, c, h, w);
  //     return;
  //   }
  // } else {
  //   if (type == DT_INT8 || type == DT_UINT8) {
  //     space_to_depth_kernel<<<num_blocks, block_size>>>(
  //         (uint8_t *)input, (uint8_t *)output, block_h, block_w, swap_hw, crd, n, c, h, w);
  //     return;
  //   } else if (type == DT_F16 || type == DT_BF16) {
  //     space_to_depth_kernel<<<num_blocks, block_size>>>(
  //         (uint16_t *)input, (uint16_t *)output, block_h, block_w, swap_hw, crd, n, c, h, w);
  //     return;
  //   } else if (type == DT_F32) {
  //     space_to_depth_kernel<<<num_blocks, block_size>>>(
  //         (float *)input, (float *)output, block_h, block_w, swap_hw, crd, n, c, h, w);
  //     return;
  //   }
  // }
}

void requantInt8Perchannel(void *input, void *output, void *multipliers,
                           void *shifts, int n, int c, int h, int w,
                           bool out_sign, bool qdm, bool relu) {
  int num_blocks = CUDA_NUM_BLOCKS(n * c * h * w);
  int block_size = CUDA_BLOCK_SIZE;
  g_requantInt8Perchannel<<<num_blocks, block_size>>>(
      (int32_t *)input, output, (int32_t *)multipliers, (int32_t *)shifts, n, c,
      h, w, out_sign, qdm, relu);
  }

void requantInt8Perchannel_3d(void *input, void *output, void *multipliers,
                           void *shifts, int n, int c, int d,int h, int w,
                           bool out_sign, bool qdm, bool relu) {
  // int num_blocks = CUDA_NUM_BLOCKS(n * c * h * w);
  // int block_size = CUDA_BLOCK_SIZE;
  // g_requantInt8Perchannel<<<num_blocks, block_size>>>(
  //     (int32_t *)input, output, (int32_t *)multipliers, (int32_t *)shifts, n, c,
  //     h, w, out_sign, qdm, relu);

    int num_blocks = CUDA_NUM_BLOCKS(n * c * h * w * d);
    int block_size = CUDA_BLOCK_SIZE;
    g_requantInt8Perchannel_3d<<<num_blocks, block_size>>>(
      (int32_t *)input, output, (int32_t *)multipliers, (int32_t *)shifts, n, c,
      h, w, d, out_sign, qdm, relu);

}

void requantInt8(void *input, void *output, int32_t multiplier, int32_t shift,
                 int num, bool out_sign, bool qdm, bool relu) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_requantInt8<<<num_blocks, block_size>>>(
      (int32_t *)input, output, multiplier, shift, num, out_sign, qdm, relu);
}

void requantInt16(void *input, void *output, int32_t multiplier, int32_t shift,
                 int num, bool relu) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_requantInt16<<<num_blocks, block_size>>>(
      (int32_t *)input, output, multiplier, shift, num, relu);
}

void requantInt16Perchannel(void *input, void *output, void *multipliers,
                           void *shifts, int n, int c, int h, int w, bool relu) {
  int num_blocks = CUDA_NUM_BLOCKS(n * c * h * w);
  int block_size = CUDA_BLOCK_SIZE;
  g_requantInt16Perchannel<<<num_blocks, block_size>>>(
      (int32_t *)input, output, (int32_t *)multipliers, (int32_t *)shifts, n, c,
      h, w, relu);

}

void requantInt16Perchannel_3d(void *input, void *output, void *multipliers,
                           void *shifts, int n, int c,int d, int h, int w, bool relu) {

    int num_blocks = CUDA_NUM_BLOCKS(n * c * h * w * d);
    int block_size = CUDA_BLOCK_SIZE;
    g_requantInt16Perchannel_3d<<<num_blocks, block_size>>>(
      (int32_t *)input, output, (int32_t *)multipliers, (int32_t *)shifts, n, c,
      h, w, d, relu);


}


void requantF8(void *input, void *output, float scale,
                            int n, int c, int h, int w, bool relu){
  int num_blocks = CUDA_NUM_BLOCKS(n * c * h * w);
  int block_size = CUDA_BLOCK_SIZE;
  g_requantF8<<<num_blocks, block_size>>>(
      (float *)input, (uint8_t*)output, scale, n, c,
      h, w, relu);
}

void requantF8Perchannel(void *input, void *output, void *scales,
                            int n, int c, int h, int w, bool relu, bool conv=true){
  int num_blocks = CUDA_NUM_BLOCKS(n * c * h * w);
  int block_size = CUDA_BLOCK_SIZE;
  g_requantF8Perchannel<<<num_blocks, block_size>>>(
      (float *)input, (uint8_t*)output, (float *)scales, n, c,
      h, w, relu, conv);
}

void requantF8Perchannel_3d(void *input, void *output, void *scales,
                            int n, int c, int d, int h, int w, bool relu, bool conv=true){
  int num_blocks = CUDA_NUM_BLOCKS(n * c * d* h * w);
  int block_size = CUDA_BLOCK_SIZE;
  g_requantF8Perchannel_3d<<<num_blocks, block_size>>>(
      (float *)input, (uint8_t*)output, (float *)scales, n, c,
      d,h, w, relu, conv);
}

void mulShift(void *input, void *output, int multiplier, int shift, int size,
              data_type_t type) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  switch (type) {
  case DT_INT8:
    g_mulShift<<<num_blocks, block_size>>>((int8_t *)input, (int8_t *)output,
                                           multiplier, shift, size);
    break;
  case DT_UINT8:
    g_mulShift<<<num_blocks, block_size>>>((uint8_t *)input, (uint8_t *)output,
                                           multiplier, shift, size);
    break;
  }
}

void mulShiftFloat(void *input, void *output, float multiplier, float shift, rounding_mode_t round_mode, int size,
              data_type_t type) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  switch (type) {
  case DT_INT8:
    g_mulShiftFloat<<<num_blocks, block_size>>>((float *)input, (int8_t *)output,
                                           multiplier, shift, size, round_mode);
    break;
  case DT_UINT8:
    g_mulShiftFloat<<<num_blocks, block_size>>>((float *)input, (uint8_t *)output,
                                           multiplier, shift, size, round_mode);
    break;
  }
}

void quantF8(void *in_f32, void *out_f8, float scale_v, int size){
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_f32ToF8<<<num_blocks, block_size>>>((float *)in_f32, scale_v, (uint8_t *)out_f8, size);
}

void print(void *data, int size, data_type_t type) {
  switch (type) {
  case DT_F32:
    g_print<<<(size + 256) / 256, 256>>>((float *)data, size);
    break;
  case DT_INT32:
    g_print<<<(size + 256) / 256, 256>>>((int32_t *)data, size);
    break;
  case DT_INT8:
    g_print<<<(size + 256) / 256, 256>>>((int8_t *)data, size);
    break;
  case DT_UINT8:
    g_print<<<(size + 256) / 256, 256>>>((uint8_t *)data, size);
    break;
  case DT_INT16:
    g_print<<<(size + 256) / 256, 256>>>((int16_t *)data, size);
    break;
  case DT_UINT16:
    g_print<<<(size + 256) / 256, 256>>>((uint16_t *)data, size);
    break;
  case DT_BF16:
    g_printBF16<<<(size + 256) / 256, 256>>>((uint16_t *)data, size);
    break;
  case DT_F16:
    g_printF16<<<(size + 256) / 256, 256>>>((uint16_t *)data, size);
    break;
  }
}

void doRelu(void *data, int size, data_type_t type) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  switch (type) {
  case DT_F32:
    g_doRelu<<<num_blocks, block_size>>>((float *)data, size);
    break;
  case DT_INT32:
    g_doRelu<<<num_blocks, block_size>>>((int32_t *)data, size);
    break;
  case DT_INT8:
    g_doRelu<<<num_blocks, block_size>>>((int8_t *)data, size);
    break;
  }
}

void lut256(void *src, void *table, void *dst, int size, data_type_t src_type,
            data_type_t dst_type) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  if (src_type == DT_INT8 && dst_type == DT_INT8) {
    g_lut256<<<num_blocks, block_size>>>((int8_t *)src, (int8_t *)table,
                                         (int8_t *)dst, size);
  } else if (src_type == DT_UINT8 && dst_type == DT_UINT8) {
    g_lut256<<<num_blocks, block_size>>>((uint8_t *)src, (uint8_t *)table,
                                         (uint8_t *)dst, size);
  } else if (src_type == DT_INT8 && dst_type == DT_UINT8) {
    g_lut256<<<num_blocks, block_size>>>((int8_t *)src, (uint8_t *)table,
                                         (uint8_t *)dst, size);
  } else if (src_type == DT_UINT8 && dst_type == DT_INT8) {
    g_lut256<<<num_blocks, block_size>>>((uint8_t *)src, (int8_t *)table,
                                         (int8_t *)dst, size);
  } else if (src_type == DT_INT8 && dst_type == DT_F32) {
    g_lut256<<<num_blocks, block_size>>>((int8_t *)src, (float *)table,
                                         (float *)dst, size);
  } else if (src_type == DT_UINT8 && dst_type == DT_F32) {
    g_lut256<<<num_blocks, block_size>>>((uint8_t *)src, (float *)table,
                                         (float *)dst, size);
  } else if (src_type == DT_INT8 && dst_type == DT_F16) {
    g_lut256<<<num_blocks, block_size>>>((int8_t *)src, (uint16_t*)table,
                                         (uint16_t *)dst, size);
  } else if (src_type == DT_UINT8 && dst_type == DT_F16) {
    g_lut256<<<num_blocks, block_size>>>((uint8_t *)src, (uint16_t *)table,
                                         (uint16_t *)dst, size);
  }
}

void upsample4D(void *src, void *dst, int n, int c, int h, int w, int scale_h,
                int scale_w, int tbytes) {
  int num_blocks = CUDA_NUM_BLOCKS(n * c * h * w * scale_h * scale_w);
  int block_size = CUDA_BLOCK_SIZE;
  g_upsample4D<<<num_blocks, block_size>>>(src, dst, n, c, h, w, scale_h,
                                           scale_w, tbytes);
}

void depth2Space(void *input, void *output, int in, int ic, int ih, int iw,
                 int on, int oc, int oh, int ow, int instride, int icstride,
                 int ihstride, int iwstride, int onstride, int ocstride,
                 int ohstride, int owstride, int block_h, int block_w, bool crd,
                 bool swap_cr, bool inversed, int tbytes) {
  int num_blocks = CUDA_NUM_BLOCKS(in * ic * ih * iw);
  int block_size = CUDA_BLOCK_SIZE;
  g_depth2Space<<<num_blocks, block_size>>>(
      input, output, in, ic, ih, iw, on, oc, oh, ow, instride, icstride,
      ihstride, iwstride, onstride, ocstride, ohstride, owstride, block_h,
      block_w, crd, swap_cr, inversed, tbytes);
}

// -------------------------------------------------------------------------
// ------- cv18xx functions
void cvScaleToF32(void *input, void *output, float scale, int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_cvInt8ScaleToF32<<<num_blocks, block_size>>>((int8_t *)input,
                                                 (float *)output, scale, size);
}

void cvScaleToBF16(void *input, void *output, float scale, int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_cvInt8ScaleToBF16<<<num_blocks, block_size>>>(
      (int8_t *)input, (uint16_t *)output, scale, size);
}

void cvQuantInt8(void *input, void *output, float scale, int size,
                 bool is_bf16) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  if (!is_bf16) {
    g_cvF32ScaleToInt8<<<num_blocks, block_size>>>(
        (float *)input, (int8_t *)output, scale, size);
  } else {
    g_cvBF16ScaleToInt8<<<num_blocks, block_size>>>(
        (uint16_t *)input, (int8_t *)output, scale, size);
  }
}

void cvAdd4DInt8(void *input0, void *input1, void *output, int mul0, int mul1,
                 int shift, bool relu, int n0, int c0, int h0, int w0, int n1,
                 int c1, int h1, int w1, int on, int oc, int oh, int ow) {
  int size = on * oc * oh * ow;
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_cvAdd4DInt8<<<num_blocks, block_size>>>(
      (int8_t *)input0, (int8_t *)input1, (int8_t *)output, mul0, mul1, shift,
      relu, n0, c0, h0, w0, n1, c1, h1, w1, on, oc, oh, ow);
}

void cvPReluInt8(void *input, void *slope, void *output, int outer_dim,
                 int inner_dim, int num_slope, int multi_pos, int shift_pos,
                 int shift_neg) {
  int num_blocks = CUDA_NUM_BLOCKS(outer_dim * inner_dim);
  int block_size = CUDA_BLOCK_SIZE;
  g_cvPReluInt8<<<num_blocks, block_size>>>(
      (int8_t *)input, (int8_t *)slope, (int8_t *)output, outer_dim, inner_dim,
      num_slope, multi_pos, shift_pos, shift_neg);
}

void cvMulShiftInt8(void *input, void *output, int multiplier, int shift,
                    int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_cvMulShiftInt8<<<num_blocks, block_size>>>(
      (int8_t *)input, (int8_t *)output, multiplier, shift, size);
}

void cvLutSlope(void *input, void *output, void *table0, void *table1, int num,
                float scale, float offset) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_cvLutSlope<<<block_size, num_blocks>>>(
      (uint16_t *)input, (uint16_t *)output, (uint16_t *)table0,
      (uint16_t *)table1, num, scale, offset);
}

void bmExp(void *input, void *output, int outer_dim, int axis_dim, int inner_dim, data_type_t type) {
  int num_blocks = CUDA_NUM_BLOCKS(outer_dim*axis_dim*inner_dim);
  int block_size = CUDA_BLOCK_SIZE;
  g_bmExp<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, outer_dim, axis_dim, inner_dim);
}

void bmReciprocal(void *input, void *output, int outer_dim, int inner_dim, data_type_t type) {
  int num_blocks = CUDA_NUM_BLOCKS(outer_dim*inner_dim);
  int block_size = CUDA_BLOCK_SIZE;
  g_bmReciprocal<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, outer_dim, inner_dim);
}

void cvLutMantissa(void *input, void *output, void *table0, void *table1,
                   int num, bool is_log) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_cvLutMantissa<<<block_size, num_blocks>>>(
      (uint16_t *)input, (uint16_t *)output, (uint16_t *)table0,
      (uint16_t *)table1, num, is_log);
}

void cvSoftmax(void *input, void *buffer, void *output, void *table0,
               void *table1, void *table2, void *table3, int outer_dim,
               int axis_dim, int inner_dim, float scale, float offset,
               bool log) {
  // get max => buffer
  maxAxis(input, buffer, outer_dim, axis_dim, inner_dim, DT_BF16);
  // sub max => output
  subAxis(input, buffer, output, outer_dim, axis_dim, inner_dim, DT_BF16);

  // exp => output
  cvLutSlope(output, output, table0, table1, outer_dim * inner_dim * axis_dim,
             scale, offset);
  // sum => buffer
  sumAxis(output, buffer, outer_dim, axis_dim, inner_dim, DT_BF16);
  // 1/sum => buffer
  cvLutMantissa(buffer, buffer, table2, table3, outer_dim * inner_dim, log);

  if (log) {
    addAxis(output, buffer, output, outer_dim, axis_dim, inner_dim, DT_BF16);
  } else {
    mulAxis(output, buffer, output, outer_dim, axis_dim, inner_dim, DT_BF16);
  }
}

void bmSoftmax(void *input, void *buffer, void *output, int outer_dim,
               int axis_dim, int inner_dim, bool log) {
  // get max => buffer
  maxAxis(input, buffer, outer_dim, axis_dim, inner_dim, DT_F32);

  // sub max => output
  subAxis(input, buffer, output, outer_dim, axis_dim, inner_dim, DT_F32);

  // exp => output
  bmExp(output, output, outer_dim, axis_dim, inner_dim, DT_F32);

  // sum => buffer
  sumAxis(output, buffer, outer_dim, axis_dim, inner_dim, DT_F32);

  // 1/sum => buffer
  bmReciprocal(buffer, buffer, outer_dim, inner_dim, DT_F32);

  if (log) {
    addAxis(output, buffer, output, outer_dim, axis_dim, inner_dim, DT_F32);
  } else {
    mulAxis(output, buffer, output, outer_dim, axis_dim, inner_dim, DT_F32);
  }
}

void bmLayerNorm(void *input, void *output, int outer_dim,
               int inner_dim, void *weight, void *bias, float eps, data_type_t type) {
  int num_blocks = CUDA_NUM_BLOCKS(outer_dim);
  int block_size = CUDA_BLOCK_SIZE;
  if (type == DT_BF16) {
    g_layerNormBF16<<<num_blocks, block_size>>>(
        (float *)input, (float *)output, outer_dim, inner_dim, (float *)weight, (float *)bias, eps);
  } else if (type == DT_F32 || type == DT_F16) {
    g_layerNorm<<<num_blocks, block_size>>>(
        (float *)input, (float *)output, outer_dim, inner_dim, (float *)weight, (float *)bias, eps);
  } else {

  }
}

void bmClip(void *input, void *output, int size, float min_v, float max_v) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_clip<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, size, min_v, max_v);
}

void bmConstantFill(void *output, float value, int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_constantFill<<<num_blocks, block_size>>>((float *)output, value, size);
}

void bmCumSum(void *input, void *output, int outer_dim, int axis_dim,
              int stride) {
  int total = outer_dim * stride;
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;
  g_cumSum<<<num_blocks, block_size>>>((float *)input, (float *)output,
                                        outer_dim, axis_dim, stride);
}

void bmRMSNorm(void *input, void *output, int outer_dim, int inner_dim,
               void *gamma, float eps) {
  int num_blocks = CUDA_NUM_BLOCKS(outer_dim);
  int block_size = CUDA_BLOCK_SIZE;
  g_rmsNorm<<<num_blocks, block_size>>>((float *)input, (float *)output,
                                         outer_dim, inner_dim,
                                         (float *)gamma, eps);
}

void bmNonZeroFill(void *input, void *flat_idx, void *counter, int total) {
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;
  CHECK_CUDA(cudaMemset(counter, 0, sizeof(int)));
  g_nonZeroFill<<<num_blocks, block_size>>>(
      (float *)input, (int *)flat_idx, (int *)counter, total);
}

void bmRange(void *output, float start, float delta, int num) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_range<<<num_blocks, block_size>>>((float *)output, start, delta, num);
}

void copyToHost(float *dst, void *src, data_type_t type) {
  if (type == DT_F32) {
    CHECK_CUDA(cudaMemcpy(dst, src, sizeof(float), cudaMemcpyDeviceToHost));
  } else if (type == DT_INT32 || type == DT_UINT32) {
    int32_t val;
    CHECK_CUDA(cudaMemcpy(&val, src, sizeof(int32_t), cudaMemcpyDeviceToHost));
    *dst = (float)val;
  } else if (type == DT_INT8) {
    int8_t val;
    CHECK_CUDA(cudaMemcpy(&val, src, sizeof(int8_t), cudaMemcpyDeviceToHost));
    *dst = (float)val;
  } else if (type == DT_UINT8) {
    uint8_t val;
    CHECK_CUDA(cudaMemcpy(&val, src, sizeof(uint8_t), cudaMemcpyDeviceToHost));
    *dst = (float)val;
  } else if (type == DT_F16) {
    uint16_t val;
    CHECK_CUDA(cudaMemcpy(&val, src, sizeof(uint16_t), cudaMemcpyDeviceToHost));
    *dst = f16_to_f32(val);
  } else {
    llvm_unreachable("copyToHost unsupported type");
  }
}

void bmReciprocal(void *input, void *output, int num, float const_val,
                  bool do_relu, float relu_limit) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_reciprocal<<<num_blocks, block_size>>>((float *)input, (float *)output, num,
                                           const_val, (int)do_relu, relu_limit);
}

void bmRelu(void *input, void *output, int num, float relu_limit) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_relu<<<num_blocks, block_size>>>((float *)input, (float *)output, num,
                                      relu_limit);
}

void bmReverse(void *input, void *output, int outer_stride, int axis_dim,
               int inner_stride) {
  int total = outer_stride * axis_dim * inner_stride;
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;
  g_reverse<<<num_blocks, block_size>>>((float *)input, (float *)output,
                                         outer_stride, axis_dim, inner_stride);
}

void bmDepackRaw(void *input, void *output, int n, int ih, int iw,
                 int ph, int pw, float white_level, float black_level,
                 int c0, int c1, int c2, int c3) {
  int total = n * ih * 2 * iw * 2;
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;
  float scale = 255.0f / (white_level - black_level);
  g_depackRaw<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, n, ih, iw, ph, pw,
      scale, black_level, c0, c1, c2, c3);
}

// ==========================================================================
// DequantizeLinear
// ==========================================================================

template <typename T>
static void launchDequantizeLinearPerTensor(void *input, void *output,
                                              float scale, int32_t zp,
                                              int num) {
  int nb = CUDA_NUM_BLOCKS(num), bs = CUDA_BLOCK_SIZE;
  g_dequantizeLinearPerTensor<<<nb, bs>>>(
      (T *)input, (float *)output, scale, zp, num);
}

void bmDequantizeLinearPerTensor(void *input, void *output, float scale,
                                  int32_t zp, int num, data_type_t in_type) {
  if (in_type == DT_INT8)
    launchDequantizeLinearPerTensor<int8_t>(input, output, scale, zp, num);
  else if (in_type == DT_UINT8)
    launchDequantizeLinearPerTensor<uint8_t>(input, output, scale, zp, num);
  else
    launchDequantizeLinearPerTensor<int32_t>(input, output, scale, zp, num);
}

template <typename T>
static void launchDequantizeLinearPerChannel(
    void *input, void *output, float *scale, int32_t *zp,
    int outer_dim, int channel_dim, int inner_dim) {
  int total = outer_dim * channel_dim * inner_dim;
  int nb = CUDA_NUM_BLOCKS(total), bs = CUDA_BLOCK_SIZE;
  g_dequantizeLinearPerChannel<<<nb, bs>>>(
      (T *)input, (float *)output, scale, zp, outer_dim, channel_dim,
      inner_dim);
}

void bmDequantizeLinearPerChannel(void *input, void *output, float *scale,
                                   int32_t *zp, int outer_dim, int channel_dim,
                                   int inner_dim, data_type_t in_type) {
  if (in_type == DT_INT8)
    launchDequantizeLinearPerChannel<int8_t>(input, output, scale, zp,
                                              outer_dim, channel_dim, inner_dim);
  else if (in_type == DT_UINT8)
    launchDequantizeLinearPerChannel<uint8_t>(input, output, scale, zp,
                                               outer_dim, channel_dim, inner_dim);
  else
    launchDequantizeLinearPerChannel<int32_t>(input, output, scale, zp,
                                              outer_dim, channel_dim, inner_dim);
}

// ==========================================================================
// DequantInt
// ==========================================================================

template <typename T>
static void launchDequantIntPerTensor(void *input, void *output, int num,
                                       int64_t multiplier, int64_t shift,
                                       int64_t lshift, int32_t zp, int mode,
                                       rounding_mode_t rmode) {
  int nb = CUDA_NUM_BLOCKS(num), bs = CUDA_BLOCK_SIZE;
  g_dequantIntPerTensor<<<nb, bs>>>((T *)input, (float *)output, num,
                                     multiplier, shift, lshift, zp,
                                     mode, rmode);
}

void bmDequantIntPerTensor(void *input, void *output, int num,
                            int64_t multiplier, int64_t shift, int64_t lshift,
                            int32_t zp, int mode, rounding_mode_t rmode,
                            data_type_t in_type) {
  if (in_type == DT_INT8)
    launchDequantIntPerTensor<int8_t>(input, output, num, multiplier, shift,
                                       lshift, zp, mode, rmode);
  else
    launchDequantIntPerTensor<uint8_t>(input, output, num, multiplier, shift,
                                        lshift, zp, mode, rmode);
}

template <typename T>
static void launchDequantIntPerChannel(
    void *input, void *output, int outer_dim, int channel_dim, int inner_dim,
    int64_t *multiplier, int64_t *shift, int64_t lshift, int32_t zp,
    int mode, rounding_mode_t rmode) {
  int total = outer_dim * channel_dim * inner_dim;
  int nb = CUDA_NUM_BLOCKS(total), bs = CUDA_BLOCK_SIZE;
  g_dequantIntPerChannel<<<nb, bs>>>((T *)input, (float *)output,
                                      outer_dim, channel_dim, inner_dim,
                                      multiplier, shift, lshift, zp,
                                      mode, rmode);
}

void bmDequantIntPerChannel(void *input, void *output,
                             int outer_dim, int channel_dim, int inner_dim,
                             int64_t *multiplier, int64_t *shift,
                             int64_t lshift, int32_t zp, int mode,
                             rounding_mode_t rmode, data_type_t in_type) {
  if (in_type == DT_INT8)
    launchDequantIntPerChannel<int8_t>(input, output, outer_dim, channel_dim,
                                        inner_dim, multiplier, shift, lshift,
                                        zp, mode, rmode);
  else
    launchDequantIntPerChannel<uint8_t>(input, output, outer_dim, channel_dim,
                                         inner_dim, multiplier, shift, lshift,
                                         zp, mode, rmode);
}

void bmBatchNorm(void *input, void *output, int n, int c, int spatial,
                 void *gamma, void *beta, void *mean, void *var, float eps,
                 bool do_relu) {
  int num_blocks = CUDA_NUM_BLOCKS(n * c * spatial);
  int block_size = CUDA_BLOCK_SIZE;
  g_batchNormInference<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, n, c, spatial, (float *)gamma,
      (float *)beta, (float *)mean, (float *)var, eps, do_relu);
}

void bmBatchNormBwd(void *grad_out, void *input, void *gamma,
                     void *save_mean, void *save_invstd,
                     void *dxhut, void *dgamma, void *dbeta,
                     void *dx2_tmp, void *dx3, void *dx,
                     int n, int c, int spatial) {
  int num_blocks = CUDA_NUM_BLOCKS(n * c * spatial);
  int block_size = CUDA_BLOCK_SIZE;
  g_batchNormBwdStats<<<num_blocks, block_size>>>(
      (float *)grad_out, (float *)input, (float *)gamma,
      (float *)save_mean, (float *)save_invstd,
      (float *)dxhut,
      (float *)dgamma, (float *)dbeta,
      (float *)dx2_tmp, (float *)dx3,
      n, c, spatial);
  g_batchNormBwdCompute<<<num_blocks, block_size>>>(
      (float *)grad_out, (float *)input,
      (float *)save_mean, (float *)save_invstd,
      (float *)dxhut,
      (float *)dx2_tmp, (float *)dx3,
      (float *)dx,
      n, c, spatial);
}

void bmBatchNormTrain(void *input, void *mean, void *var, void *gamma,
                      void *beta, void *output, void *mean_out,
                      void *saved_invstd, void *running_mean,
                      void *running_var, int n, int c, int spatial, float eps,
                      float momentum, bool do_relu) {
  int block_size = CUDA_BLOCK_SIZE;
  int stat_blocks = CUDA_NUM_BLOCKS(c);
  g_batchNormTrainStats<<<stat_blocks, block_size>>>(
      (float *)input, (float *)mean, (float *)var, (float *)mean_out,
      (float *)saved_invstd, (float *)running_mean, (float *)running_var, n, c,
      spatial, eps, momentum);

  int norm_blocks = CUDA_NUM_BLOCKS(n * c * spatial);
  g_batchNormTrainNormalize<<<norm_blocks, block_size>>>(
      (float *)input, (float *)output, (float *)gamma, (float *)beta,
      (float *)mean_out, (float *)saved_invstd, n, c, spatial, do_relu);
}

void bmGELU(void *input, void *output, int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_GELU<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, size);
}

void bmAbs(void *input, void *output, int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_abs<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, size);
}

void bmArccos(void *input, void *output, int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_arccos<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, size);
}

void bmArctanh(void *input, void *output, int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_arctanh<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, size);
}

void bmCos(void *input, void *output, int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_cos<<<num_blocks, block_size>>>((float *)input, (float *)output, size);
}

void bmCosh(void *input, void *output, int size) {
  int num_blocks = CUDA_NUM_BLOCKS(size);
  int block_size = CUDA_BLOCK_SIZE;
  g_cosh<<<num_blocks, block_size>>>((float *)input, (float *)output, size);
}

void bmCopy(void *input, void *output, int n, int c, int h, int w,
            int i_n, int i_c, int i_h, int i_w,
            int o_n, int o_c, int o_h, int o_w, int tbytes) {
  int total = n * c * h * w;
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;
  g_copy<<<num_blocks, block_size>>>(input, output, n, c, h, w,
                                      i_n, i_c, i_h, i_w,
                                      o_n, o_c, o_h, o_w, tbytes);
}

void bmCorrelation(void *left, void *right, void *output,
                   int max_disp, int num_groups, int ic, int ih, int iw) {
  int spatial = ih * iw;
  int total = num_groups * max_disp * spatial;
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;
  g_correlation<<<num_blocks, block_size>>>(
      (float *)left, (float *)right, (float *)output,
      max_disp, num_groups, ic, ih, iw);
}

void bmArgMax(void *input, void *indices, int outer_dim, int axis_dim,
              int inner_dim, bool select_last) {
  int num_blocks = outer_dim * inner_dim;
  int block_size = axis_dim < ARG_BLOCK_SIZE ? axis_dim : ARG_BLOCK_SIZE;
  if (select_last) {
    g_argReduce<true, true><<<num_blocks, block_size>>>(
        (float *)input, (int *)indices,
        outer_dim, axis_dim, inner_dim);
  } else {
    g_argReduce<false, true><<<num_blocks, block_size>>>(
        (float *)input, (int *)indices,
        outer_dim, axis_dim, inner_dim);
  }
}

void bmArgMin(void *input, void *indices, int outer_dim, int axis_dim,
              int inner_dim, bool select_last) {
  int num_blocks = outer_dim * inner_dim;
  int block_size = axis_dim < ARG_BLOCK_SIZE ? axis_dim : ARG_BLOCK_SIZE;
  if (select_last) {
    g_argReduce<true, false><<<num_blocks, block_size>>>(
        (float *)input, (int *)indices,
        outer_dim, axis_dim, inner_dim);
  } else {
    g_argReduce<false, false><<<num_blocks, block_size>>>(
        (float *)input, (int *)indices,
        outer_dim, axis_dim, inner_dim);
  }
}

void bmAdaptiveAvgPool2D(void *input, void *output,
                          int n, int c, int ih, int iw, int oh, int ow) {
  dim3 block(AP_TILE_W, AP_TILE_H);
  dim3 grid((oh + AP_TILE_H - 1) / AP_TILE_H,
            (ow + AP_TILE_W - 1) / AP_TILE_W,
            n * c);
  g_adaptiveAvgPool2D<<<grid, block>>>(
      (float *)input, (float *)output, n, c, ih, iw, oh, ow);
}

void scale4D(void *src, void *scale, void * bias, void *dst, bool relu, int n, int c, int h, int w, int off0,
             int off1, int off2, int off3, int s0, int s1, int s2, int s3,
             int on, int oc, int oh, int ow) {
  int num_blocks = CUDA_NUM_BLOCKS(on * oc * oh * ow);
  int block_size = CUDA_BLOCK_SIZE;
  g_scale4DF32<<<num_blocks, block_size>>>((float*)src, (float*)scale, (float*)bias, (float*)dst, relu, n, c, h, w, off0, off1, off2, off3,
                                        s0, s1, s2, s3, on, oc, oh, ow);
}

void bmReduce(
  void *d_input,
  void *d_output,
  int shape_dim,
  void *input_shape,
  void *reduce_mask,
  int mode
) {
  enum ReductionMode mode_enum = static_cast<ReductionMode>(mode);
  TensorShape in_shape;
  in_shape.init(shape_dim, (int*)input_shape);
  TensorShape out_shape;
  int out_shape_idx = 0;
  int processed_axes_count = 0;
  int processed_axes[8]; // assuming max 8 dimensions
  for (int i = 0; i < in_shape.ndim; i++) {
      if (((int*)reduce_mask)[i] == 0) {
          out_shape.dims[out_shape_idx]= in_shape.dims[i];
          out_shape_idx ++;
      } else {
          processed_axes[processed_axes_count] = i;
          processed_axes_count ++;
      }
  }
  for (int i = out_shape_idx; i < 8; i++) {
      out_shape.dims[i] = 1;
  }
  out_shape.ndim = out_shape_idx;
  out_shape.computeStrides();
  cudaStream_t stream = 0;
  // Handle special cases
  if (processed_axes_count == 1) {
      // Single axis reduction - can use optimized kernel
      int axis = processed_axes[0];
      int outer_size = 1;
      for (int i = 0; i < axis; i++) {
          outer_size *= in_shape.dims[i];
      }
      int reduce_size = in_shape.dims[axis];
      int inner_size = 1;
      for (int i = axis + 1; i < in_shape.ndim; i++) {
          inner_size *= in_shape.dims[i];
      }

      // Launch optimized kernel
      dim3 blocks(outer_size);
      dim3 threads(min(1024, inner_size));
      switch (mode_enum) {
          case REDUCE_SUM:
              contiguousAxisReductionKernel<float, REDUCE_SUM><<<blocks, threads, 0, stream>>>(
                  (float *)d_input, (float *)d_output, outer_size, reduce_size, inner_size);
              break;
          case REDUCE_MEAN:
              contiguousAxisReductionKernel<float, REDUCE_MEAN><<<blocks, threads, 0, stream>>>(
                  (float *)d_input, (float *)d_output, outer_size, reduce_size, inner_size);
              break;
          case REDUCE_MAX:
              contiguousAxisReductionKernel<float, REDUCE_MAX><<<blocks, threads, 0, stream>>>(
                  (float *)d_input, (float *)d_output, outer_size, reduce_size, inner_size);
              break;
          case REDUCE_MIN:
              contiguousAxisReductionKernel<float, REDUCE_MIN><<<blocks, threads, 0, stream>>>(
                  (float *)d_input, (float *)d_output, outer_size, reduce_size, inner_size);
              break;
          case REDUCE_L2_NORM:
              contiguousAxisReductionKernel<float, REDUCE_L2_NORM><<<blocks, threads, 0, stream>>>(
                  (float *)d_input, (float *)d_output, outer_size, reduce_size, inner_size);
              break;
          case REDUCE_L1_NORM:
              contiguousAxisReductionKernel<float, REDUCE_L1_NORM><<<blocks, threads, 0, stream>>>(
                  (float *)d_input, (float *)d_output, outer_size, reduce_size, inner_size);
              break;
          case REDUCE_PROD:
              contiguousAxisReductionKernel<float, REDUCE_PROD><<<blocks, threads, 0, stream>>>(
                  (float *)d_input, (float *)d_output, outer_size, reduce_size, inner_size);
              break;
          case REDUCE_VAR:
              contiguousAxisReductionKernel<float, REDUCE_VAR><<<blocks, threads, 0, stream>>>(
                  (float *)d_input, (float *)d_output, outer_size, reduce_size, inner_size);
              break;
          default:
              break;
      }
  } else {
      // Launch kernel based on mode
      int blockSize = 256;
      int numBlocks = (out_shape.totalElements() + blockSize - 1) / blockSize;
      int * d_mask =nullptr;
      cudaMalloc(&d_mask, sizeof(int) * 8);
      cudaMemcpy(d_mask, reduce_mask, sizeof(int) * 8, cudaMemcpyHostToDevice);
      switch (mode) {
          case REDUCE_SUM:
              multiAxisReductionKernel<float, REDUCE_SUM><<<numBlocks, blockSize, 0, stream>>>(
                  (float *)d_input, (float *)d_output, in_shape, out_shape, d_mask);
              break;
          case REDUCE_MEAN:
              multiAxisReductionKernel<float, REDUCE_MEAN><<<numBlocks, blockSize, 0, stream>>>(
                  (float *)d_input, (float *)d_output, in_shape, out_shape, d_mask);
              break;
          case REDUCE_MAX:
              multiAxisReductionKernel<float, REDUCE_MAX><<<numBlocks, blockSize, 0, stream>>>(
                  (float *)d_input, (float *)d_output, in_shape, out_shape, d_mask);
              break;
          case REDUCE_MIN:
              multiAxisReductionKernel<float, REDUCE_MIN><<<numBlocks, blockSize, 0, stream>>>(
                  (float *)d_input, (float *)d_output, in_shape, out_shape, d_mask);
              break;
          case REDUCE_L2_NORM:
              multiAxisReductionKernel<float, REDUCE_L2_NORM><<<numBlocks, blockSize, 0, stream>>>(
                  (float *)d_input, (float *)d_output, in_shape, out_shape, d_mask);
              break;
          case REDUCE_L1_NORM:
              multiAxisReductionKernel<float, REDUCE_L1_NORM><<<numBlocks, blockSize, 0, stream>>>(
                  (float *)d_input, (float *)d_output, in_shape, out_shape, d_mask);
              break;
          case REDUCE_PROD:
              multiAxisReductionKernel<float, REDUCE_PROD><<<numBlocks, blockSize, 0, stream>>>(
                  (float *)d_input, (float *)d_output, in_shape, out_shape, d_mask);
              break;
          case REDUCE_VAR:
              multiAxisReductionKernel<float, REDUCE_VAR><<<numBlocks, blockSize, 0, stream>>>(
                  (float *)d_input, (float *)d_output, in_shape, out_shape, d_mask);
              break;
          default:
              break;
      }
      cudaFree(d_mask);
  }
  cudaStreamSynchronize(stream);
}



//add
// 在 type convert functions 附近或文件末尾

void affineSigmoid(
    void* input_ptr,
    void* output_ptr,
    float scale,
    float bias,
    bool do_log,
    int size,
    data_type_t dtype)
{
    if (size <= 0) return;

    int num_blocks = CUDA_NUM_BLOCKS(size);
    int block_size = CUDA_BLOCK_SIZE;

    // 直接调用 kernel
    g_affineSigmoidKernel<<<num_blocks, block_size>>>(
        static_cast<const float*>(input_ptr),
        output_ptr,
        scale,
        bias,
        do_log,
        size,
        dtype
    );

    CHECK_CUDA(cudaGetLastError());
}

void bmExpElm(void *input, void *output, int num) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_expElm<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, num);
}

void bmElu(void *input, void *output, int num, float alpha) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_elu<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, num, alpha);
}

void bmErf(void *input, void *output, int num) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_erf<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, num);
}

void bmExpand(void *input, void *output,
              int in_n, int in_c, int in_h, int in_w,
              int out_n, int out_c, int out_h, int out_w) {
  int total = out_n * out_c * out_h * out_w;
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;
  g_expand<<<num_blocks, block_size>>>(
      (float *)input, (float *)output,
      in_n, in_c, in_h, in_w,
      out_n, out_c, out_h, out_w);
}

void bmEmbDenseBwd(void *grad_output, void *indices, void *output,
                    int batch_size, int embed_dim) {
  int total = batch_size * embed_dim;
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;
  g_embDenseBwd<<<num_blocks, block_size>>>(
      (float *)grad_output, (float *)indices, (float *)output,
      batch_size, embed_dim);
}

void bmGatherElements(void *input, void *indices, void *output,
                       int *out_shape, int *in_strides, int *out_strides,
                       int rank, int axis) {
  int out_total = 1;
  for (int d = 0; d < rank; d++) out_total *= out_shape[d];

  int bytes = rank * sizeof(int);
  int *d_out_shape, *d_in_strides, *d_out_strides;
  cudaMalloc(&d_out_shape, bytes);
  cudaMalloc(&d_in_strides, bytes);
  cudaMalloc(&d_out_strides, bytes);
  cudaMemcpy(d_out_shape, out_shape, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in_strides, in_strides, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out_strides, out_strides, bytes, cudaMemcpyHostToDevice);

  int num_blocks = CUDA_NUM_BLOCKS(out_total);
  int block_size = CUDA_BLOCK_SIZE;
  g_gatherElements<<<num_blocks, block_size>>>(
      (float *)input, (float *)indices, (float *)output,
      d_out_shape, d_in_strides, d_out_strides, rank, axis, out_total);

  cudaDeviceSynchronize();
  cudaFree(d_out_shape);
  cudaFree(d_in_strides);
  cudaFree(d_out_strides);
}

void bmGatherND(void *input, void *indices, void *output,
                 int *in_shape, int *in_strides,
                 int *idx_shape, int *idx_strides,
                 int batch_dims, int indices_dim, int coord_dim,
                 int out_total, int copy_len) {
  int total = out_total * copy_len;
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;

  int max_bytes = 8 * sizeof(int);
  int *d_in_shape, *d_in_strides, *d_idx_strides;
  cudaMalloc(&d_in_shape, max_bytes);
  cudaMalloc(&d_in_strides, max_bytes);
  cudaMalloc(&d_idx_strides, max_bytes);
  cudaMemset(d_in_shape, 0, max_bytes);
  cudaMemset(d_in_strides, 0, max_bytes);
  cudaMemset(d_idx_strides, 0, max_bytes);
  cudaMemcpy(d_in_shape, in_shape, 8 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_in_strides, in_strides, 8 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_idx_strides, idx_strides, 8 * sizeof(int), cudaMemcpyHostToDevice);

  g_gatherND<<<num_blocks, block_size>>>(
      (float *)input, (float *)indices, (float *)output,
      d_in_shape, d_in_strides, d_idx_strides,
      indices_dim, coord_dim, batch_dims, out_total, copy_len);

  cudaDeviceSynchronize();
  cudaFree(d_in_shape);
  cudaFree(d_in_strides);
  cudaFree(d_idx_strides);
}

void bmGroupNorm(void *input, void *output, void *weight, void *bias,
                  int outer_dim, int inner_dim,
                  int channel, int channel_per_group, float eps) {
  int num_blocks = CUDA_NUM_BLOCKS(outer_dim);
  int block_size = CUDA_BLOCK_SIZE;
  g_groupNorm<<<num_blocks, block_size>>>(
      (float *)input, (float *)output,
      (float *)weight, (float *)bias,
      outer_dim, inner_dim, channel, channel_per_group, eps);
}

void bmGroupNormTrain(void *input, void *output, void *mean, void *rstd,
                       void *weight, void *bias,
                       int outer_dim, int inner_dim,
                       int channel, int channel_per_group, float eps) {
  int num_blocks = CUDA_NUM_BLOCKS(outer_dim);
  int block_size = CUDA_BLOCK_SIZE;
  g_groupNormTrain<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, (float *)mean, (float *)rstd,
      (float *)weight, (float *)bias,
      outer_dim, inner_dim, channel, channel_per_group, eps);
}

void bmGridSampler(void *input, void *grid, void *output,
                    int n, int c, int h, int w, int oh, int ow,
                    int mode, int padding_mode, bool align_corners) {
  int total = n * c * oh * ow;
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;
  g_gridSampler<<<num_blocks, block_size>>>(
      (float *)input, (float *)grid, (float *)output,
      n, c, h, w, oh, ow, mode, padding_mode, align_corners);
}

void bmGruCell(void *x_gi, void *x_gr, void *x_gh,
                void *h_gi, void *h_gr, void *h_gh,
                void *h_prev, void *h_out,
                int total, bool linear_before_reset) {
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;
  g_gruCell<<<num_blocks, block_size>>>(
      (float *)x_gi, (float *)x_gr, (float *)x_gh,
      (float *)h_gi, (float *)h_gr, (float *)h_gh,
      (float *)h_prev, (float *)h_out,
      total, linear_before_reset);
}

void bmFloor(void *input, void *output, int num) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_floor<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, num);
}

void bmRound(void *input, void *output, int num) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_round<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, num);
}

void bmAttentionQK(void *Q, void *K, void *scores,
                    int B, int H, int Mq, int Mk, int d, float scale) {
  int total = B * H * Mq * Mk;
  int nb = CUDA_NUM_BLOCKS(total), bs = CUDA_BLOCK_SIZE;
  g_attentionQK<<<nb, bs>>>((float *)Q, (float *)K, (float *)scores,
                              B, H, Mq, Mk, d, scale);
}

void bmAttentionPV(void *scores, void *V, void *context,
                    int B, int H, int Mq, int Mk, int d) {
  int total = B * H * Mq * d;
  int nb = CUDA_NUM_BLOCKS(total), bs = CUDA_BLOCK_SIZE;
  g_attentionPV<<<nb, bs>>>((float *)scores, (float *)V, (float *)context,
                              B, H, Mq, Mk, d);
}

void bmCeil(void *input, void *output, int num) {
  int nb = CUDA_NUM_BLOCKS(num), bs = CUDA_BLOCK_SIZE;
  g_ceil<<<nb, bs>>>((float *)input, (float *)output, num);
}

void bmPermuteBMHD(void *src, void *dst, int B, int M, int H, int d) {
  int total = B * M * H * d;
  int nb = CUDA_NUM_BLOCKS(total), bs = CUDA_BLOCK_SIZE;
  g_permuteBMHD<<<nb, bs>>>((float *)src, (float *)dst, B, M, H, d);
}

void bmSqrt(void *input, void *output, int num) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_sqrt<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, num);
}

void bmRsqrt(void *input, void *output, int num) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_rsqrt<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, num, 1e-5f);
}

void bmRoiAlign(void *input, void *rois, void *output,
                int N, int C, int H, int W,
                int num_rois, int output_h, int output_w,
                int sampling_ratio, float spatial_scale,
                bool align_corners, bool avg_mode) {
  int total = num_rois * C * output_h * output_w;
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;
  g_roiAlign<<<num_blocks, block_size>>>(
      (float *)input, (float *)rois, (float *)output,
      N, C, H, W,
      num_rois, output_h, output_w,
      sampling_ratio, spatial_scale,
      align_corners, avg_mode);
}

void bmLRN(void *input, void *output, int n, int c, int h, int w,
            int size, float alpha, float beta, float bias) {
  int total = n * c * h * w;
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;
  g_lrn<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, n, c, h, w, size, alpha, beta, bias);
}

void bmLSTMAddBias(void *gate, void *bias, int batch_size, int hidden_size) {
  int total = batch_size * hidden_size;
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;
  g_lstmAddBias<<<num_blocks, block_size>>>(
      (float *)gate, (float *)bias, batch_size, hidden_size);
}

void bmLSTMCell(void *x_i, void *x_o, void *x_f, void *x_c,
                 void *h_i, void *h_o, void *h_f, void *h_c,
                 void *cell_state, void *hidden_state,
                 int total, float cont) {
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;
  g_lstmCell<<<num_blocks, block_size>>>(
      (float *)x_i, (float *)x_o, (float *)x_f, (float *)x_c,
      (float *)h_i, (float *)h_o, (float *)h_f, (float *)h_c,
      (float *)cell_state, (float *)hidden_state, total, cont);
}

void bmIndexPut(void *input, void *indices, void *values, void *output,
                 int num_indices, int inner_dim, bool accumulate) {
  int total = num_indices * inner_dim;
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;
  g_indexPut<<<num_blocks, block_size>>>(
      (float *)input, (float *)indices, (float *)values, (float *)output,
      num_indices, inner_dim, accumulate);
}

void bmInterpBilinear(void *input, void *output,
                       int n, int c, int ih, int iw, int oh, int ow,
                       bool align_corners, bool half_pixel) {
  int total = n * c * oh * ow;
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;
  g_interpBilinear<<<num_blocks, block_size>>>(
      (float *)input, (float *)output,
      n, c, ih, iw, oh, ow, align_corners, half_pixel);
}

void bmInterpNearest(void *input, void *output,
                      int n, int c, int ih, int iw, int oh, int ow) {
  int total = n * c * oh * ow;
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;
  g_interpNearest<<<num_blocks, block_size>>>(
      (float *)input, (float *)output,
      n, c, ih, iw, oh, ow);
}

void bmInstanceNorm(void *input, void *output, void *weight, void *bias,
                     int outer_dim, int inner_dim, int channel, float eps) {
  int num_blocks = CUDA_NUM_BLOCKS(outer_dim);
  int block_size = CUDA_BLOCK_SIZE;
  g_instanceNorm<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, (float *)weight, (float *)bias,
      outer_dim, inner_dim, channel, eps);
}

void bmHardSigmoid(void *input, void *output, int num, float alpha, float beta) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_hardsigmoid<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, num, alpha, beta);
}

void bmLayerNormTrain(void *input, void *output, void *mean, void *rstd,
                       void *weight, void *bias,
                       int outer_dim, int inner_dim, float eps) {
  int num_blocks = CUDA_NUM_BLOCKS(outer_dim);
  int block_size = CUDA_BLOCK_SIZE;
  g_layerNormTrain<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, (float *)mean, (float *)rstd,
      (float *)weight, (float *)bias, outer_dim, inner_dim, eps);
}

void bmLogB(void *input, void *output, int num, float log_base_inv) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_logB<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, num, log_base_inv);
}

void bmLogicalAnd(void *lhs, void *rhs, void *output,
                   int l_n, int l_c, int l_h, int l_w,
                   int r_n, int r_c, int r_h, int r_w,
                   int o_n, int o_c, int o_h, int o_w) {
  int total = o_n * o_c * o_h * o_w;
  int num_blocks = CUDA_NUM_BLOCKS(total);
  int block_size = CUDA_BLOCK_SIZE;
  g_logicalAnd<<<num_blocks, block_size>>>(
      (float *)lhs, (float *)rhs, (float *)output,
      l_n, l_c, l_h, l_w, r_n, r_c, r_h, r_w, o_n, o_c, o_h, o_w);
}

void bmLeakyRelu(void *input, void *output, int num, float alpha) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_leakyRelu<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, num, alpha);
}

void bmLog(void *input, void *output, int num) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_log<<<num_blocks, block_size>>>((float *)input, (float *)output, num);
}

void bmHardSwish(void *input, void *output, int num) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_hardswish<<<num_blocks, block_size>>>(
      (float *)input, (float *)output, num);
}

} // namespace cuda
} // namespace tpu_mlir
