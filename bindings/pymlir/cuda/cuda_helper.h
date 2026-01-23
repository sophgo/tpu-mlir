//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>

namespace tpu_mlir {
namespace cuda {
// -------------------------------------------------------------------------
// --- definitions ---

// same RoundingMode defined in MathUtils.h
typedef enum {
  RD_HALF_AWAY_FROM_ZERO = 0, // 1.5 -> 2   -1.5 -> -2
  RD_HALF_UP = 1,             // 1.5 -> 2   -1.5 -> -1
  RD_HALF_DOWN = 2,           // 1.5 -> 1   -1.5 -> -2
  RD_HALF_TO_EVEN = 3,        // 1.5 -> 2    2.5 -> 2
  RD_HALF_TO_ODD = 4,         // 1.5 -> 1    0.5 -> 1
  RD_HALF_TOWARDS_ZERO = 5,   // 1.5 -> 1   -1.5 -> -1
  RD_TOWARDS_ZERO = 6,        // 1.6 -> 1   -1.6 -> -1
  RD_AWAY_FROM_ZERO = 7,      // 1.4 -> 2   -1.4 -> -2
  RD_UP = 8,                  // 1.4 -> 2   -1.6 -> -1
  RD_DOWN = 9,                // 1.6 -> 1   -1.4 -> -2
  RD_UNKNOWN = -1
} rounding_mode_t;

typedef enum {
  DT_INT8 = 0,      // 8-bit signed integer
  DT_UINT8 = 1,     // 8-bit unsigned integer
  DT_INT16 = 2,     // 16-bit signed integer
  DT_UINT16 = 3,    // 16-bit unsigned integer
  DT_INT32 = 4,     // 32-bit signed integer
  DT_UINT32 = 5,    // 32-bit unsigned integer
  DT_BF16 = 6,      // Bfloat16 floating-point
  DT_F16 = 7,       // 16-bit floating-point (Half precision)
  DT_F32 = 8,       // 32-bit floating-point (Single precision)
  DT_F64 = 9,       // 64-bit floating-point (Double precision)
  DT_F8E4M3 = 10,   // 8-bit float8 e4m3
  DT_UNKNOWN = 1000 // Unknown or invalid data type
} data_type_t;

size_t get_dtype_bytes(data_type_t type);
// -------------------------------------------------------------------------
// --- host functions ---

// float input * scale = int8 output，if !sign, uint8 output
void f32ScaleToInt8(void *input, void *output, float scale, int size, bool sign,
                    rounding_mode_t rmode);
void bf16ScaleToInt8(void *input, void *output, float scale, int size,
                     bool sign, rounding_mode_t rmode);
void f16ScaleToInt8(void *input, void *output, float scale, int size, bool sign,
                    rounding_mode_t rmode);
// int8 or uint8 * scale => float output
void int8ScaleToF32(void *input, void *output, float scale, int size,
                    bool sign);
void int8ScaleToBF16(void *input, void *output, float scale, int size,
                     bool sign);
void int8ScaleToF16(void *input, void *output, float scale, int size,
                    bool sign);
void int16ScaleToF32(void *input, void *output, float scale, int size);
void int16ScaleToBF16(void *input, void *output, float scale, int size);
void int16ScaleToF16(void *input, void *output, float scale, int size);

// mul: int8 * int8 * multiplier >> rshift => int8
void mulInt8(void *a, void *b, void *o, bool a_sign, bool b_sign, bool o_sign,
             int multiplier, int rshift, int size, bool qdm, bool relu);
// mul: support broadcast
void mulInt8(void *a, void *b, void *o, int n0, int c0, int h0, int w0, int n1,
             int c1, int h1, int w1, int n2, int c2, int h2, int w2,
             bool a_sign, bool b_sign, bool o_sign, int multiplier, int rshift,
             bool qdm, bool relu);

// add: i8 * i32 >> s0 + i8 * i32 >> s1 = int8 (half away from zero)
void add4DInt8(void *input0, void *input1, void *output, int mul0, int mul1,
               int shift0, int shift1, bool sign0, bool sign1, bool sign2,
               bool relu, int n0, int c0, int h0, int w0, int n1, int c1,
               int h1, int w1, int n2, int c2, int h2, int w2);
void add4DF32(void *input0, float scale0, void *input1, float scale1, void *output,
               bool relu, int n0, int c0, int h0, int w0, int n1, int c1,
               int h1, int w1, int n2, int c2, int h2, int w2);
void add4DInt32(int32_t *input0, int32_t *input1, int32_t *output,
                int n0, int c0, int h0, int w0,
                int n1, int c1, int h1, int w1,
                int n2, int c2, int h2, int w2);
void sub4DF32(void *input0, void *input1, void *output,
               bool relu, bool reverse, int n0, int c0, int h0, int w0, int n1, int c1,
               int h1, int w1, int n2, int c2, int h2, int w2);
void sub4DInt8(void *input0, bool input0_unsigned, int mul0, int shift0, void *input1, bool input1_unsigned, int mul1, int shift1, void *output, bool output_unsigned,
               bool relu, bool reverse, int n0, int c0, int h0, int w0, int n1, int c1,
               int h1, int w1, int n2, int c2, int h2, int w2);
void subConst4DF32(void *input, float const_v, void *output,
               bool relu, bool reverse, int n, int c, int h, int w);
void subConst4DI8(void *input, bool in_signed, int const_v, void *output,
               bool do_relu, bool reverse, int multi, int shift, int n, int c, int h, int w);
void mulConst4DF32(void *input, float const_v, void *output, bool do_relu,
                  int n0, int c0, int h0, int w0);
void mul4DF32(void *input0, void *input1, void *output, bool do_relu,
                  int n0, int c0, int h0, int w0,
                  int n1, int c1, int h1, int w1,
                  int n2, int c2, int h2, int w2);

void neg(void *input, void *output, int size, data_type_t type);
// zero pad
void pad4D(void *input, void *output, int n, int c, int h, int w, int pad_h_t,
           int pad_h_b, int pad_w_l, int pad_w_r, int tbytes);

void depth2Space(void *input, void *output, int in, int ic, int ih, int iw,
                 int on, int oc, int oh, int ow, int instride, int icstride,
                 int ihstride, int iwstride, int onstride, int ocstride,
                 int ohstride, int owstride, int block_h, int block_w, bool crd,
                 bool swap_cr, bool inversed, int tbytes);

void mmF32(void *input, void *right, void *output, bool right_transpose, int m, int k, int n);
void mmInt8(void *input, bool left_signed, void *right, bool right_signed, void *output, bool right_transpose, int m, int k, int n);
void requantInt8Perchannel(void *input, void *output, void *multipliers,
                           void *shifts, int n, int c, int h, int w,
                           bool out_sign, bool qdm = false, bool relu = false);

// requant from int32 to int8
void requantInt8(void *input, void *output, int32_t multiplier, int32_t shift,
                 int num, bool out_sign, bool qdm = false, bool relu = false);
void requantInt16(void *input, void *output, int32_t multiplier, int32_t shift,
                 int num, bool relu);
void requantInt16Perchannel(void *input, void *output, void *multipliers,
                           void *shifts, int n, int c, int h, int w, bool relu = false);
void requantF8Perchannel(void *input, void *output, void *scales,
                            int n, int c, int h, int w, bool relu, bool conv);
void requantF8(void *input, void *output, float scale,
                            int n, int c, int h, int w, bool relu);
// inplace relu
void doRelu(void *data, int size, data_type_t type);

// find max. input[outer_dim, axis_dim, inner_dim] =>
// output[outer_dim,1,inner_dim]
void maxAxis(void *input, void *output, int outer_dim, int axis_dim,
             int inner_dim, data_type_t type);

// input [outer_dim, axis_dim, inner_dim], sub[outer_dim,1,inner_dim] =>
// output[outer_dim, axis_dim, inner_dim]
void subAxis(void *input, void *sub, void *output, int outer_dim, int axis_dim,
             int inner_dim, data_type_t type);
void mulAxis(void *input, void *mul, void *output, int outer_dim, int axis_dim,
             int inner_dim, data_type_t type);
void addAxis(void *input, void *add, void *output, int outer_dim, int axis_dim,
             int inner_dim, data_type_t type);
// sum
void sumAxis(void *input, void *output, int outer_dim, int axis_dim,
             int inner_dim, data_type_t type);
void copyAxis(void *src, void *dst, int outer_dim, int axis_dim, int inner_dim,
              int offset, int num, int tbytes);

cudaError_t convertType(void *src, void *dst, int num_elem,
                        data_type_t src_type, data_type_t dst_type,
                        rounding_mode_t rmode = RD_TOWARDS_ZERO);
void permute6D(void *src, void *dst, int n, int c, int d, int h, int w, int d1, int o0, int o1,
               int o2, int o3, int o4, int o5, int tbytes);
void upsample4D(void *src, void *dst, int n, int c, int h, int w, int scale_h,
                int scale_w, int tbytes);
void print(void *data, int size, data_type_t type);

// input4 , offset4, step4 => output4
void slice6D(void *src, void *dst, int n, int c, int d, int h, int w, int d1, int off0,
             int off1, int off2, int off3, int off4, int off5, int s0, int s1, int s2, int s3,
             int s4, int s5, int on, int oc, int od, int oh, int ow, int od1, int tbytes);
void swapDimInner6D(void *src, void *dst, int n, int c, int d, int h, int w, int d1, int off0,
             int off1, int off2, int off3, int off4, int off5, int tbytes);
void tile4D(void *src, void *dst, int n, int c, int h, int w, int on, int oc,
            int oh, int ow, int tbytes);
void mulShift(void *input, void *output, int multiplier, int shift, int size,
              data_type_t type);
void mulShiftFloat(void *input, void *output, float multiplier, float shift, rounding_mode_t round_mode, int size,
              data_type_t type);
void quantF8(void *in_f32, void *out_f8, float scale_v, int size);
// src is i8, table has 256 value
void lut256(void *src, void *table, void *dst, int size, data_type_t src_type,
            data_type_t dst_type);
void gather(void *indices, void *embedding, void *output, int num_indices,
            int embedding_dim, int inner_dim, data_type_t ind_type,
            data_type_t embed_type);
void cudaGather(void *indices, void *embedding, void *output, int num_indices,
            int outer_dims, int ax_dim, int inner_dims, data_type_t ind_type,
            data_type_t embed_type);
void bmDepth2Space(void *input, void *output, bool inversed, bool swap_hw, bool crd,
            int block_h, int block_w, int n, int c, int h, int w,
            int ins, int ics, int ihs, int iws, int on, int oc, int oh, int ow,
            int ons, int ocs, int ohs, int ows, data_type_t type);

// -------------------------------------------------------------------------
// cv18xx only

// float * scale => int8 output
void cvQuantInt8(void *input, void *output, float scale, int size,
                 bool is_bf16);

// int8 or uint8 * scale => float output, cv18xx only
void cvScaleToF32(void *input, void *output, float scale, int size);
void cvScaleToBF16(void *input, void *output, float scale, int size);
// int8 * multi >> shift = i8 output
void cvMulShiftInt8(void *input, void *output, int multiplier, int shift,
                    int size);

// add: (int8 * int32 + int8 * int32) >> shift = int8 (half up)
void cvAdd4DInt8(void *input0, void *input1, void *output, int mul0, int mul1,
                 int shift, bool relu, int n0, int c0, int h0, int w0, int n1,
                 int c1, int h1, int w1, int on, int oc, int oh, int ow);

void cvPReluInt8(void *input, void *slope, void *output, int outer_dim,
                 int inner_dim, int num_slope, int multi_pos, int shift_pos,
                 int shift_neg);

void cvLutSlope(void *data, void *table0, void *table1, int num, float scale,
                float offset);
void cvLutMantissa(void *input, void *output, void *table0, void *table1,
                   int num, bool is_log);
// softmax by mantisa table
void cvSoftmax(void *input, void *buffer, void *output, void *table0,
               void *table1, void *table2, void *table3, int outer_dim,
               int axis_dim, int inner_dim, float scale, float offset,
               bool log);

void bmSoftmax(void *input, void *buffer, void *output,
               int outer_dim, int axis_dim, int inner_dim, bool log);
void bmGELU(void *input, void *output, int num);
void bmLayerNorm(void *input, void *output, int outer_dim,
               int inner_dim, void *weight, void *bias, float eps, data_type_t type);
void bmExp(void *input, void *output, int outer_dim, int axis_dim, int inner_dim, data_type_t type);
void bmReciprocal(void *input, void *output,  int outer_dim, int inner_dim, data_type_t type);

void scale4D(void *src, void *scale, void * bias, void *dst, bool relu, int n, int c, int h, int w, int off0,
             int off1, int off2, int off3, int s0, int s1, int s2, int s3,
             int on, int oc, int oh, int ow);
void bmReduce(void *d_input,void * d_output, int shape_dim, void *input_shape, void *reduce_mask, int mode);

} // namespace cuda
} // namespace tpu_mlir
