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
#include <type_traits>

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
  TFLite_LShift = 0,
  TFLite = 1,
  MultiplierShift = 2,
  OnlyShift = 3,
  QDM = 4,
  OnlyScale = 5,
} requant_mode_t;

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

typedef enum {
  ACTIVE_TANH = 0,
  ACTIVE_SIGMOID = 1,
  ACTIVE_RELU = 2,
  ACTIVE_EXP = 3,
  ACTIVE_ELU = 4,
  ACTIVE_SQRT = 5,
  ACTIVE_SQUARE = 6,
  ACTIVE_RSQRT = 7,
  ACTIVE_ABSVAL = 8,
  ACTIVE_LN = 9,
  ACTIVE_ROUND = 10,
  ACTIVE_CEIL = 11,
  ACTIVE_FLOOR = 12,
  ACTIVE_SIN = 13,
  ACTIVE_COS = 14,
  ACTIVE_IS_FINITE = 15,
  ACTIVE_MISH = 16,
  ACTIVE_SWISH = 17,
  ACTIVE_HSWISH = 18,
  ACTIVE_SILU = 19,
  ACTIVE_ARCSIN = 20,
  ACTIVE_ARCCOS = 21,
  ACTIVE_ARCSINH = 22,
  ACTIVE_ARCCOSH = 23,
  ACTIVE_ARCTANH = 24,
  ACTIVE_SINH = 25,
  ACTIVE_COSH = 26,
  ACTIVE_TAN = 27,
  ACTIVE_SIGN = 28,
  ACTIVE_GELU = 29,
  ACTIVE_ERF = 30,
  ACTIVE_HSIGMOID = 31,
  ACTIVE_LOG_SIGMOID = 32,
  ACTIVE_SOFT_PLUS = 33,
  ACTIVE_SOFT_SIGN = 34,
  ACTIVE_LOG2 = 35,
  ACTIVE_TGELU = 36,
  ACTIVE_QGELU = 37,
} active_mode_t;

typedef enum {
  BILINEAR = 0,
  NEAREST = 1
} grid_sample_interpolation_mode_t;

typedef enum {
  ZEROS = 0,
  BORDER = 1,
  REFLECTION = 2
} grid_sample_padding_mode_t;


typedef enum {
  CAFFE_SUPPORT = 0,
  TENSORFLOW_SUPPORT = 1,
  CAFFE_NEAREST = 2,
  TENSORFLOW_NEAREST = 3,
  PYTORCH_SUPPORT = 4,
  PYTORCH_NEAREST = 5,
  OPENCV_BILINEAR = 6,
  ONNX_NEAREST = 7,
} interp_platform_t;

size_t get_dtype_bytes(data_type_t type);
// -------------------------------------------------------------------------
// --- host functions ---

// float input * scale = int8 output，if !sign, uint8 output
void f32ScaleToInt8(void *input, void *output, float scale, int size, bool sign,
                    rounding_mode_t rmode, int zero_point = 0);
void bf16ScaleToInt8(void *input, void *output, float scale, int size,
                     bool sign, rounding_mode_t rmode, int zero_point = 0);
void f16ScaleToInt8(void *input, void *output, float scale, int size, bool sign,
                    rounding_mode_t rmode, int zero_point = 0);
// int8 or uint8 * scale => float output
void int8ScaleToF32(void *input, void *output, float scale, int size,
                    bool sign, float zero_point = 0.0f);
void int8ScaleToBF16(void *input, void *output, float scale, int size,
                     bool sign, float zero_point = 0.0f);
void int8ScaleToF16(void *input, void *output, float scale, int size,
                    bool sign, float zero_point = 0.0f);
void int16ScaleToF32(void *input, void *output, float scale, int size, float zero_point = 0.0f);
void int16ScaleToBF16(void *input, void *output, float scale, int size, float zero_point = 0.0f);
void int16ScaleToF16(void *input, void *output, float scale, int size, float zero_point = 0.0f);

// mul: int8 * int8 * multiplier >> rshift => int8
void mulInt8(void *a, void *b, void *o, bool a_sign, bool b_sign, bool o_sign,
             int multiplier, int rshift, int size, bool qdm, bool relu);
// mul: support broadcast
void mulInt8(void *a, void *b, void *o, int n0, int c0, int h0, int w0, int n1,
             int c1, int h1, int w1, int n2, int c2, int h2, int w2,
             bool a_sign, bool b_sign, bool o_sign, int multiplier, int rshift,
             bool relu, int a_zp = 0, int b_zp = 0, int o_zp = 0,
             requant_mode_t rqmode = MultiplierShift,
             rounding_mode_t rmode = RD_HALF_UP,
             bool is_cv18xx = false);

// add: i8 * i32 >> s0 + i8 * i32 >> s1 = int8 (half away from zero)
void add6DInt8(void *input0, void *input1, void *output, int mul0, int mul1,
              int shift0, int shift1, bool sign0, bool sign1, bool sign2,
              bool relu, int i0, int i1, int i2, int i3, int i4, int i5,
              int j0, int j1, int j2, int j3, int j4, int j5,
              int o0, int o1, int o2, int o3, int o4, int o5,
              int input0_zp = 0, int input1_zp = 0, int output_zp = 0);
void add6DF32(void *input0, float scale0, void *input1, float scale1, void *output,
               bool relu, int i0, int i1, int i2, int i3, int i4, int i5,
              int j0, int j1, int j2, int j3, int j4, int j5,
              int o0, int o1, int o2, int o3, int o4, int o5);
void add4DInt32(int32_t *input0, int32_t *input1, int32_t *output,
                int n0, int c0, int h0, int w0,
                int n1, int c1, int h1, int w1,
                int n2, int c2, int h2, int w2);
void sub4DF32(void *input0, void *input1, void *output,
               bool relu, bool reverse, int n0, int c0, int h0, int w0, int n1, int c1,
               int h1, int w1, int n2, int c2, int h2, int w2);
void sub4DInt8(void *input0, bool input0_unsigned, int mul0, int shift0, void *input1, bool input1_unsigned, int mul1, int shift1, void *output, bool output_unsigned,
               bool relu, bool reverse, int n0, int c0, int h0, int w0, int n1, int c1,
               int h1, int w1, int n2, int c2, int h2, int w2,
               int input0_zp = 0, int input1_zp = 0, int output_zp = 0);
void subConst4DF32(void *input, float const_v, void *output,
               bool relu, bool reverse, int n, int c, int h, int w);
void subConst4DI8(void *input, bool in_signed, int const_v, void *output, bool out_signed,
               bool do_relu, bool reverse, int multi, int shift,
               int n, int c, int h, int w, int output_zp = 0);
void addConstI8(void *input, int const_v, void *output, int multi, int shift,
                int input_zp, int output_zp, int size, bool in_signed,
                bool out_signed, bool do_relu);
void maxConstI8(void *input, int const_v, void *output, int multi, int shift,
                int input_zp, int output_zp, int size, bool in_signed,
                bool out_signed, bool do_relu);
void minConstI8(void *input, int const_v, void *output, int multi, int shift,
                int input_zp, int output_zp, int size, bool in_signed,
                bool out_signed, bool do_relu);
void mulConst6DF32(void *input, float const_v, void *output, bool do_relu,
                  int s0, int s1, int s2, int s3, int s4, int s5);
void mul4DF32(void *input0, void *input1, void *output, bool do_relu,
                  int n0, int c0, int h0, int w0,
                  int n1, int c1, int h1, int w1,
                  int n2, int c2, int h2, int w2);
void divMDF32(void *input0, void *input1, void *output, int64_t *shape0,
              int64_t *shape1, int64_t *shape2, int num_dims);
void divConst4DF32(void *input, float const_v, void *output,
                   bool do_relu, bool reverse, int n, int c, int h, int w);
void neg(void *input, void *output, int size, data_type_t type);
// zero pad
void pad4D(void *input, void *output, int n, int c, int h, int w, int pad_h_t,
           int pad_h_b, int pad_w_l, int pad_w_r, int tbytes, float pad_value = 0);
template <typename T,
          typename std::enable_if<std::is_arithmetic<T>::value &&
                                  !std::is_same<typename std::decay<T>::type, bool>::value,
                                  int>::type = 0>
inline void pad4D(void *input, void *output, int n, int c, int h, int w, int pad_h_t,
           int pad_h_b, int pad_w_l, int pad_w_r, int tbytes, T pad_value) {
  pad4D(input, output, n, c, h, w, pad_h_t, pad_h_b, pad_w_l, pad_w_r,
        tbytes, static_cast<float>(pad_value));
}
void pad4D(void *input, void *output, int n, int c, int h, int w, int pad_h_t,
           int pad_h_b, int pad_w_l, int pad_w_r, int tbytes, bool is_edge);
void insertZero4D(void *input, void *output, int n, int c, int h, int w,
                  int ins_h, int ins_w, int tbytes);
void depth2Space(void *input, void *output, int in, int ic, int ih, int iw,
                 int on, int oc, int oh, int ow, int instride, int icstride,
                 int ihstride, int iwstride, int onstride, int ocstride,
                 int ohstride, int owstride, int block_h, int block_w, bool crd,
                 bool swap_cr, bool inversed, int tbytes);

void mmF32(void *input, void *right, void *output, int m, int k, int n,
           bool left_transpose = false, bool right_transpose = false, bool output_transpose = false,
           float left_zp = 0.0f, float right_zp = 0.0f);
void mmInt8(void *input, bool left_signed, void *right, bool right_signed, void *output, int m, int k, int n,
            bool left_transpose = false, bool right_transpose = false, bool output_transpose = false,
            int left_zp = 0, int right_zp = 0);
void mmInt8DynamicQuantize(void *input, void *right, void *output, int m, int k, int n,
                           bool left_transpose = false, bool right_transpose = false, bool output_transpose = false,
                           int q_group_size = 32);
void mmInt8DynamicQuantize(void *input, void *weight, void *output, void *scale, void *zp,
                           int m, int k, int n, int q_group_size = 32);
void mmInt4DynamicQuantize(void *input, void *right, void *output, int m, int k, int n,
                           bool left_transpose = false, bool right_transpose = false, bool output_transpose = false,
                           int q_group_size = 32);
void mmInt4DynamicQuantize(void *input, void *weight, void *output, void *scale, void *zp,
                           int m, int k, int n, int q_group_size = 32);
void mmF8DynamicQuantize(void *input, void *right, void *output, int m, int k, int n,
                           bool left_transpose = false, bool right_transpose = false, bool output_transpose = false,
                           int q_group_size = 32);
void mmF8DynamicQuantize(void *input, void *weight, void *output, void *scale, void *zp,
                           int m, int k, int n, int q_group_size = 32);
void mmF4DynamicQuantize(void *input, void *right, void *output, int m, int k, int n,
                           bool left_transpose = false, bool right_transpose = false, bool output_transpose = false,
                           int q_group_size = 32);
void mmF4DynamicQuantize(void *input, void *weight, void *output, void *scale, void *zp,
                            int m, int k, int n, int q_group_size = 32);
void mmMXF4DynamicQuantize(void *input, void *right, void *output, int m, int k, int n,
                           bool left_transpose = false, bool right_transpose = false, bool output_transpose = false,
                           int q_group_size = 32);
void mmMXF4DynamicQuantize(void *input, void *weight, void *output, void *scale, void *zp,
                            int m, int k, int n, int q_group_size = 32);
void dequantA16MMWeight(void *input, void *output, void *scale, void *zp, int num, int group_size, int bits);
void requantInt8Perchannel(void *input, void *output, void *multipliers,
                           void *shifts, int n, int c, int h, int w,
                           bool out_sign, bool qdm = false, bool relu = false,
                           int zero_point = 0);
void requantInt8Perchannel(void *input, void *output, void *multipliers,
                           void *shifts, int n, int c, int h, int w,
                           bool out_sign, bool relu,
                           int zero_point, bool is_cv18xx,
                           requant_mode_t qmode,
                           rounding_mode_t rmode);
void requantInt8Perchannel(void *input, void *output, void *multipliers,
                           void *shifts, int n, int c, int h, int w,
                           bool out_sign, bool relu,
                           void *zero_points, bool is_cv18xx,
                           requant_mode_t qmode,
                           rounding_mode_t rmode);

// requant from int32 to int8
void requantInt8(void *input, void *output, int32_t multiplier, int32_t shift,
                 int num, bool out_sign, bool qdm = false, bool relu = false, int zero_point = 0);
void requantInt16(void *input, void *output, int32_t multiplier, int32_t shift,
                 int num, bool relu, int zero_point = 0);
void requantInt16Perchannel(void *input, void *output, void *multipliers,
                           void *shifts, int n, int c, int h, int w, bool relu = false,
                           int zero_point = 0);
void requantF8Perchannel(void *input, void *output, void *scales,
                            int n, int c, int h, int w, bool relu, bool conv);
void requantF8(void *input, void *output, float scale,
                            int s0, int s1, int s2, int s3, int s4, int s5, bool relu);
// inplace relu
void doRelu(void *data, int size, data_type_t type, int zero_point = 0);

// find max. input[outer_dim, axis_dim, inner_dim] =>
// output[outer_dim,1,inner_dim]
void maxAxis(void *input, void *output, int outer_dim, int axis_dim,
             int inner_dim, data_type_t type);

// input [outer_dim, axis_dim, inner_dim], sub[outer_dim,1,inner_dim] =>
// output[outer_dim, axis_dim, inner_dim]
void subAxis(void *input, void *sub, void *output, int outer_dim, int axis_dim,
             int inner_dim, data_type_t type);
void mulAxis(void *input, void *mul, void *output, int outer_dim, int axis_dim,
             int inner_dim, data_type_t type, bool log = false);
void addAxis(void *input, void *add, void *output, int outer_dim, int axis_dim,
             int inner_dim, data_type_t type);
void subAxis(void *input, void *sub, void *output, int outer_dim, int axis_dim,
             int inner_dim, data_type_t type);
// sum
void sumAxis(void *input, void *output, int outer_dim, int axis_dim,
             int inner_dim, data_type_t type);
void copyAxis(void *src, void *dst, int outer_dim, int axis_dim, int inner_dim,
              int offset, int num, int tbytes);

cudaError_t convertType(void *src, void *dst, int num_elem,
                        data_type_t src_type, data_type_t dst_type,
                        rounding_mode_t rmode = RD_HALF_TO_EVEN);
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
void tile(void *src, void *dst, int64_t *in_shape, int64_t *out_shape, int num_dims, int out_elems, int tbytes);
void mulShift(void *input, void *output, int multiplier, int shift, int size,
              data_type_t type, int input_zp = 0, int output_zp = 0);
void mulShiftFloat(void *input, void *output, float multiplier, float shift, rounding_mode_t round_mode, int size,
              data_type_t type);
void mulShiftDouble(void *input, void *output, double multiplier, double shift, rounding_mode_t round_mode, int size,
              data_type_t type);
void quantF8(void *in_f32, void *out_f8, float scale_v, int size);
// src is i8, table has 256 value
void lut256(void *src, void *table, void *dst, int size, data_type_t src_type,
            data_type_t dst_type);
void gather(void *indices, void *embedding, void *output, int num_indices,
            int embedding_dim, int inner_dim, data_type_t ind_type,
            data_type_t embed_type);
void gatherElements(void *indices, void *input, void *output,
                    const int64_t *input_shape, const int64_t *indices_shape,
                    int rank, int axis, data_type_t index_type,
                    data_type_t input_type);
void cudaGather(void *indices, void *embedding, void *output, int num_indices,
            int outer_dims, int ax_dim, int inner_dims, data_type_t ind_type,
            data_type_t embed_type);
void bmDepth2Space(void *input, void *output, bool inversed, bool swap_hw, bool crd,
            int block_h, int block_w, int n, int c, int h, int w,
            int ins, int ics, int ihs, int iws, int on, int oc, int oh, int ow,
            int ons, int ocs, int ohs, int ows, data_type_t type);

// Rotate kernel weights spatially (180 degree flip for deconv->conv)
void rotateKernelWeight(void *src, void *dst, int oc, int ic, int kh, int kw,
                        int group, int tbytes);

// Pad tensor for deconv (stride insertion, dilation, and padding)
void padTensorForDeconv(void *dst, void *src, int n, int ic, int ih, int iw,
                        int kh, int kw, int dh, int dw, int sh, int sw,
                        int pad_h, int pad_h_after, int pad_w, int pad_w_after,
                        int output_pad_h, int output_pad_w, float pad_value,
                        int tbytes);

void PReluF32(void *input, void *slope, void *output, int outer_dim, int inner_dim,
             int num_slope);

void PReluInt8(void *input, void *slope, int shift, void *output, int outer_dim,
               int inner_dim, int num_slope);
// -------------------------------------------------------------------------
// cv18xx only

// float * scale => int8 output
void cvQuantInt8(void *input, void *output, float scale, int size,
                 bool is_bf16, int zero_point = 0);

// int8 or uint8 * scale => float output, cv18xx only
void cvScaleToF32(void *input, void *output, float scale, int size, float zero_point = 0.0f);
void cvScaleToBF16(void *input, void *output, float scale, int size, float zero_point = 0.0f);
// int8 * multi >> shift = i8 output
void cvMulShiftInt8(void *input, void *output, int multiplier, int shift,
                    int size);

// add: (int8 * int32 + int8 * int32) >> shift = int8 (half up)
void cvAdd6DInt8(void *input0, void *input1, void *output, int mul0, int mul1,
                 int shift, bool relu, int i0, int i1, int i2, int i3, int i4, int i5,
                 int j0, int j1, int j2, int j3, int j4, int j5,
                 int o0, int o1, int o2, int o3, int o4, int o5,
                 int input0_zp = 0, int input1_zp = 0, int output_zp = 0);

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
void bmSoftmax(void *input, void *buffer, void *output,
               int outer_dim, int axis_dim, int inner_dim, void* exp_table,
               float scale, float zp);
void bmGELU(void *input, void *output, int num);
void bmABSVAL(void *input, void *output, int num);
void bmActive(void *input, void *output, int num, active_mode_t mode);
void bmActive(void *input, void *output, int num, active_mode_t mode, double coeff);
void bmActive(void *input, void *output, int num, active_mode_t mode, double coeff1, double coeff2);
void clip(void *input, void *output, int num, double min_val, double max_val);
void bmLayerNorm(void *input, void *output, int outer_dim,
               int inner_dim, void *weight, void *bias, float eps, data_type_t type);
void bmExp(void *input, void *output, int outer_dim, int axis_dim, int inner_dim, data_type_t type,
           void *exp_table = nullptr);
void bmReciprocal(void *input, void *output,  int outer_dim, int inner_dim, data_type_t type);

void scale4D(void *src, void *scale, void * bias, void *dst, bool relu, int n, int c, int h, int w, int off0,
             int off1, int off2, int off3, int s0, int s1, int s2, int s3,
             int on, int oc, int oh, int ow);
void bmReduce(void *d_input, void *d_output, int shape_dim, void *input_shape,
              void *reduce_mask, int mode, bool is_cv18xx = false);
void cvLayerNorm(void *input, void *output, int outer_dim,
               int inner_dim, void *weight, void *bias, void *table,
               void *mtable, float eps);
void RightBitShift(void *input, void *output, int shift, int size, int tbytes);
void GridSample4D(void *input, void *grid, void *output, int n, int c, int h,
                  int w, int out_h, int out_w, bool align_corners,
                  grid_sample_interpolation_mode_t interpolation_mode,
                  grid_sample_padding_mode_t padding_mode);
void argIndex(void *input, void *output_idx, void *output_val, int outer_dim, int axis_dim,
              int inner_dim, bool is_argmax, bool is_cv18xx = false);
void argIndex(void *input, void *arg_values, void *output_idx, int outer_dim, int axis_dim,
              int inner_dim, int input_bytes, float scale);
void interp(void *input, void *output, int n, int c, int h, int w, int out_h, int out_w,
            float scale_h, float scale_w, bool align_corners, bool half_pixel,
            interp_platform_t platform);
void GQA(void *Q, void *K, void *V, void *mask, void *output, int batch, int M_q, int M_k,
         int q_head, int kv_head, int dim, float scale, bool is_bf16);

// Migrated helper declarations from legacy branch.
void add4DF32(void *input0, float scale0, void *input1, float scale1, void *output,
               bool relu, int n0, int c0, int h0, int w0, int n1, int c1,
               int h1, int w1, int n2, int c2, int h2, int w2);
void bmCompare4DF32(void *lhs, void *rhs, void *output, int mode,
                    int n0, int c0, int h0, int w0,
                    int n1, int c1, int h1, int w1,
                    int n2, int c2, int h2, int w2);
void bmCompareConst4DF32(void *input, float const_v, void *output,
                          int mode, bool inversed, int n, int c, int h, int w);
void mulConst4DF32(void *input, float const_v, void *output, bool do_relu,
                  int n0, int c0, int h0, int w0);
void addConst4DF32(void *input, float const_v, void *output, bool do_relu,
                    int n, int c, int h, int w);
void pad5D(void *input, void *output,int n, int c,int d,int h,int w,int pad_d_f,
           int pad_d_b,int pad_h_t,int pad_h_b, int pad_w_l, int pad_w_r, int tbytes);
void requantInt8Perchannel_3d(void *input, void *output, void *multipliers,
                           void *shifts, int n, int c,int d, int h, int w,
                           bool out_sign, bool qdm = false, bool relu = false);
void requantInt16Perchannel_3d(void *input, void *output, void *multipliers,
                           void *shifts, int n, int c,int d, int h, int w, bool relu = false);
void requantF8Perchannel_3d(void *input, void *output, void *scales,
                            int n, int c, int d ,int h, int w, bool relu, bool conv);
void bmAbs(void *input, void *output, int num);
void bmCos(void *input, void *output, int num);
void bmCosh(void *input, void *output, int num);
void bmCopy(void *input, void *output, int n, int c, int h, int w,
            int i_n, int i_c, int i_h, int i_w,
            int o_n, int o_c, int o_h, int o_w, int tbytes);
void bmCorrelation(void *left, void *right, void *output,
                   int max_disp, int num_groups, int ic, int ih, int iw);
void bmArccos(void *input, void *output, int num);
void bmArctanh(void *input, void *output, int num);
void bmAdaptiveAvgPool2D(void *input, void *output,
                          int n, int c, int ih, int iw, int oh, int ow);
void bmBatchNorm(void *input, void *output, int n, int c, int spatial,
                 void *gamma, void *beta, void *mean, void *var, float eps,
                 bool do_relu);
void bmClip(void *input, void *output, int num, float min_v, float max_v);
void bmConstantFill(void *output, float value, int num);
void bmCumSum(void *input, void *output, int outer_dim, int axis_dim, int stride);
void bmRMSNorm(void *input, void *output, int outer_dim, int inner_dim,
               void *gamma, float eps);
void bmRange(void *output, float start, float delta, int num);
void bmReciprocal(void *input, void *output, int num, float const_val,
                  bool do_relu, float relu_limit);
void copyToHost(float *dst, void *src, data_type_t type);
void bmReverse(void *input, void *output, int outer_stride, int axis_dim,
               int inner_stride);
void bmDepackRaw(void *input, void *output, int n, int ih, int iw,
                 int ph, int pw, float white_level, float black_level,
                 int c0, int c1, int c2, int c3);
void bmDequantizeLinearPerTensor(void *input, void *output, float scale,
                                  int32_t zp, int num, data_type_t in_type);
void bmDequantizeLinearPerChannel(void *input, void *output, float *scale,
                                   int32_t *zp, int outer_dim, int channel_dim,
                                   int inner_dim, data_type_t in_type);
void bmDequantIntPerTensor(void *input, void *output, int num,
                            int64_t multiplier, int64_t shift, int64_t lshift,
                            int32_t zp, int mode, rounding_mode_t rmode,
                            data_type_t in_type);
void bmDequantIntPerChannel(void *input, void *output,
                             int outer_dim, int channel_dim, int inner_dim,
                             int64_t *multiplier, int64_t *shift,
                             int64_t lshift, int32_t zp, int mode,
                             rounding_mode_t rmode, data_type_t in_type);
void bmBatchNormBwd(void *grad_out, void *input, void *gamma,
                     void *save_mean, void *save_invstd,
                     void *dxhut, void *dgamma, void *dbeta,
                     void *dx2_tmp, void *dx3, void *dx,
                     int n, int c, int spatial);
void bmBatchNormTrain(void *input, void *mean, void *var, void *gamma,
                      void *beta, void *output, void *mean_out,
                      void *saved_invstd, void *running_mean,
                      void *running_var, int n, int c, int spatial, float eps,
                      float momentum, bool do_relu);
void bmExpElm(void *input, void *output, int num);
void bmElu(void *input, void *output, int num, float alpha);
void bmErf(void *input, void *output, int num);
void bmExpand(void *input, void *output,
              int in_n, int in_c, int in_h, int in_w,
              int out_n, int out_c, int out_h, int out_w);
void bmEmbDenseBwd(void *grad_output, void *indices, void *output,
                    int batch_size, int embed_dim);
void bmGatherND(void *input, void *indices, void *output,
                 int *in_shape, int *in_strides,
                 int *idx_shape, int *idx_strides,
                 int batch_dims, int indices_dim, int coord_dim,
                 int out_total, int copy_len);
void bmGroupNorm(void *input, void *output, void *weight, void *bias,
                  int outer_dim, int inner_dim,
                  int channel, int channel_per_group, float eps);
void bmGroupNormTrain(void *input, void *output, void *mean, void *rstd,
                       void *weight, void *bias,
                       int outer_dim, int inner_dim,
                       int channel, int channel_per_group, float eps);
void bmLRN(void *input, void *output, int n, int c, int h, int w,
            int size, float alpha, float beta, float bias);
void bmLSTMCell(void *x_i, void *x_o, void *x_f, void *x_c,
                 void *h_i, void *h_o, void *h_f, void *h_c,
                 void *cell_state, void *hidden_state,
                 int total, float cont);
void bmLSTMAddBias(void *gate, void *bias, int batch_size, int hidden_size);
void bmIndexPut(void *input, void *indices, void *values, void *output,
                 int num_indices, int inner_dim, bool accumulate);
void bmInstanceNorm(void *input, void *output, void *weight, void *bias,
                     int outer_dim, int inner_dim, int channel, float eps);
void bmHardSigmoid(void *input, void *output, int num, float alpha, float beta);
void bmLayerNormTrain(void *input, void *output, void *mean, void *rstd,
                       void *weight, void *bias,
                       int outer_dim, int inner_dim, float eps);
void bmLogB(void *input, void *output, int num, float log_base_inv);
void bmLogicalAnd(void *lhs, void *rhs, void *output,
                   int l_n, int l_c, int l_h, int l_w,
                   int r_n, int r_c, int r_h, int r_w,
                   int o_n, int o_c, int o_h, int o_w);
void bmLeakyRelu(void *input, void *output, int num, float alpha);
void bmLog(void *input, void *output, int num);
void bmHardSwish(void *input, void *output, int num);
void bmGruCell(void *x_gi, void *x_gr, void *x_gh,
                void *h_gi, void *h_gr, void *h_gh,
                void *h_prev, void *h_out,
                int total, bool linear_before_reset);
void bmRound(void *input, void *output, int num);
void bmAttentionQK(void *Q, void *K, void *scores,
                    int B, int H, int Mq, int Mk, int d, float scale);
void bmAttentionPV(void *scores, void *V, void *context,
                    int B, int H, int Mq, int Mk, int d);
void bmCeil(void *input, void *output, int num);
void bmPermuteBMHD(void *src, void *dst, int B, int M, int H, int d);
void bmRsqrt(void *input, void *output, int num);
void bmRoiAlign(void *input, void *rois, void *output,
                int N, int C, int H, int W,
                int num_rois, int output_h, int output_w,
                int sampling_ratio, float spatial_scale,
                bool align_corners, bool avg_mode);
} // namespace cuda
} // namespace tpu_mlir
