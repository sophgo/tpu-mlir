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

// -------------------------------------------------------------------------
// --- host functions ---

// float input * scale = int8 output，if !sign, uint8 output
void f32ScaleToInt8(void *input, void *output, float scale, int size, bool sign,
                    rounding_mode_t rmode);

// int8 or uint8 * scale => float output
void int8ScaleToF32(void *input, void *output, float scale, int size,
                    bool sign);

// mul: int8 * int8 * multiplier >> rshift => int8
void mulInt8(void *a, void *b, void *o, bool a_sign, bool b_sign, bool o_sign,
             int multiplier, int rshift, int size, bool qdm, bool relu);
// mul: support broadcast
void mulInt8(void *a, void *b, void *o, int n0, int c0, int h0, int w0, int n1,
             int c1, int h1, int w1, int n2, int c2, int h2, int w2,
             bool a_sign, bool b_sign, bool o_sign, int multiplier, int rshift,
             bool qdm, bool relu);

// add: i8 * i32 >> s0 + i8 * i32 >> s1 = int8 (half away from zero)
void addInt8(void *input0, void *input1, void *output, int mul0, int mul1,
             int shift0, int shift1, bool sign0, bool sign1, bool sign2,
             int size, bool relu);

void neg(void *input, void *output, int size, cudnnDataType_t type);
// zero pad
void pad4D(void *input, void *output, int n, int c, int h, int w, int pad_h_t,
           int pad_h_b, int pad_w_l, int pad_w_r, int tbytes);

void depth2Space(void *input, void *output, int in, int ic, int ih, int iw,
                 int on, int oc, int oh, int ow, int instride, int icstride,
                 int ihstride, int iwstride, int onstride, int ocstride,
                 int ohstride, int owstride, int block_h, int block_w, bool crd,
                 bool swap_cr, bool inversed, int tbytes);

void mmF32(void *input, void *right, void *output, int m, int k, int n);
void requantInt8Perchannel(void *input, void *output, void *multipliers,
                           void *shifts, int n, int c, int h, int w,
                           bool out_sign, bool qdm = false, bool relu = false);

// requant from int32 to int8
void requantInt8(void *input, void *output, int32_t multiplier, int32_t shift,
                 int num, bool out_sign, bool qdm = false, bool relu = false);

// inplace relu
void doRelu(void *data, int size, cudnnDataType_t type);

// find max. input[outer_dim, axis_dim, inner_dim] =>
// output[outer_dim,1,inner_dim]
void maxAxis(void *input, void *output, int outer_dim, int axis_dim,
             int inner_dim, cudnnDataType_t type);

// input [outer_dim, axis_dim, inner_dim], sub[outer_dim,1,inner_dim] =>
// output[outer_dim, axis_dim, inner_dim]
void subAxis(void *input, void *sub, void *output, int outer_dim, int axis_dim,
             int inner_dim, cudnnDataType_t type);
void mulAxis(void *input, void *mul, void *output, int outer_dim, int axis_dim,
             int inner_dim, cudnnDataType_t type);
void addAxis(void *input, void *add, void *output, int outer_dim, int axis_dim,
             int inner_dim, cudnnDataType_t type);
// sum
void sumAxis(void *input, void *output, int outer_dim, int axis_dim,
             int inner_dim, cudnnDataType_t type);
void copyAxis(void *src, void *dst, int outer_dim, int axis_dim, int inner_dim,
              int offset, int num, int tbytes);

cudaError_t convertType(void *src, void *dst, int size,
                        cudnnDataType_t src_type, cudnnDataType_t dst_type,
                        rounding_mode_t rmode = RD_TOWARDS_ZERO);
void permute4D(void *src, void *dst, int n, int c, int h, int w, int o0, int o1,
               int o2, int o3, int tbytes);
void upsample4D(void *src, void *dst, int n, int c, int h, int w, int scale_h,
                int scale_w, int tbytes);
void print(void *data, int size, cudnnDataType_t type);

// input4 , offset4, step4 => output4
void slice4D(void *src, void *dst, int n, int c, int h, int w, int off0,
             int off1, int off2, int off3, int s0, int s1, int s2, int s3,
             int on, int oc, int oh, int ow, int tbytes);
void mulShift(void *input, void *output, int multiplier, int shift, int size,
              cudnnDataType_t type);
// src is i8, table has 256 value
void lut256(void *src, void *table, void *dst, int size,
            cudnnDataType_t src_type, cudnnDataType_t dst_type);
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
void cvAddInt8(void *input0, void *input1, void *output, int mul0, int mul1,
               int shift, int size, bool relu);

void cvLutSlope(void *data, void *table0, void *table1, int num, float scale,
                float offset);
void cvLutMantissa(void *input, void *output, void *table0, void *table1,
                   int num, bool is_log);
// softmax by mantisa table
void cvSoftmax(void *input, void *buffer, void *output, void *table0,
               void *table1, void *table2, void *table3, int outer_dim,
               int axis_dim, int inner_dim, float scale, float offset,
               bool log);

} // namespace cuda
} // namespace tpu_mlir
