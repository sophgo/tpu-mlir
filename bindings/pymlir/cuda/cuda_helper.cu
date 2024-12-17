//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "cuda_global.cuh"

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
                                           num_elem, rmode);
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

void permute4D(void *src, void *dst, int n, int c, int h, int w, int o0, int o1,
               int o2, int o3, int tbytes) {
  int num = n * c * h * w;
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_permute4D<<<num_blocks, block_size>>>(src, dst, n, c, h, w, o0, o1, o2, o3,
                                          tbytes);
}

void slice4D(void *src, void *dst, int n, int c, int h, int w, int off0,
             int off1, int off2, int off3, int s0, int s1, int s2, int s3,
             int on, int oc, int oh, int ow, int tbytes) {
  int num_blocks = CUDA_NUM_BLOCKS(on * oc * oh * ow);
  int block_size = CUDA_BLOCK_SIZE;
  g_slice4D<<<num_blocks, block_size>>>(src, dst, n, c, h, w, off0, off1, off2,
                                        off3, s0, s1, s2, s3, on, oc, oh, ow,
                                        tbytes);
}

void tile4D(void *src, void *dst, int n, int c, int h, int w, int on, int oc,
            int oh, int ow, int tbytes) {
  int num_blocks = CUDA_NUM_BLOCKS(on * oc * oh * ow);
  int block_size = CUDA_BLOCK_SIZE;
  g_tile4D<<<num_blocks, block_size>>>(src, dst, n, c, h, w, on, oc, oh, ow,
                                       tbytes);
}

void mmF32(void *input, void *right, void *output, int m, int k, int n) {
  // Dimensions for blocks and grid
  int num_blocks = CUDA_NUM_BLOCKS(m * n);
  int block_size = CUDA_BLOCK_SIZE;
  g_mmF32<<<num_blocks, block_size>>>((float *)input, (float *)right,
                                      (float *)output, m, k, n);
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
  }
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

void requantInt8(void *input, void *output, int32_t multiplier, int32_t shift,
                 int num, bool out_sign, bool qdm, bool relu) {
  int num_blocks = CUDA_NUM_BLOCKS(num);
  int block_size = CUDA_BLOCK_SIZE;
  g_requantInt8<<<num_blocks, block_size>>>(
      (int32_t *)input, output, multiplier, shift, num, out_sign, qdm, relu);
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

} // namespace cuda
} // namespace tpu_mlir
