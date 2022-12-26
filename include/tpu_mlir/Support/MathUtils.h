//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace tpu_mlir {
// =======================
// round mode
// =======================
typedef enum {
  ROUNDING_HALF_AWAY_FROM_ZERO = 0, // 1.5 -> 2   -1.5 -> -2
  ROUNDING_HALF_UP = 1,             // 1.5 -> 2   -1.5 -> -1
  ROUNDING_HALF_DOWN = 2,           // 1.5 -> 1   -1.5 -> -2
  ROUNDING_HALF_TO_EVEN = 3,        // 1.5 -> 2    2.5 -> 2
  ROUNDING_HALF_TO_ODD = 4,         // 1.5 -> 1    0.5 -> 1
  ROUNDING_HALF_TOWARDS_ZERO = 5,   // 1.5 -> 1   -1.5 -> -1
  ROUNDING_TOWARDS_ZERO = 6,        // 1.6 -> 1   -1.6 -> -1
  ROUNDING_AWAY_FROM_ZERO = 7,      // 1.4 -> 2   -1.4 -> -2
  ROUNDING_UP = 8,
  /* CEIL */ // 1.4 -> 2   -1.6 -> -1
  ROUNDING_DOWN = 9,
  /* FLOOR */ // 1.6 -> 1   -1.4 -> -2
  ROUNDING_UNKNOWN = -1
} RoundingMode;

// =======================
// I8 multiplier type
// =======================
typedef enum {
  BM_QUANT = 0,
  BM_TFLITE_QUANT = 1,
  CVI_QUANT = 2,
  CVI_QDM_QUANT = 3, /* FLOOR */
  UNKNOWN = -1
} MultiplierType;

// =======================
// alignment function
// =======================
template <typename T> static inline T ceiling_func(T numerator, T denominator) {
  return (numerator + denominator - 1) / denominator;
}

template <typename T> static inline T align_up(T x, T a) {
  return ceiling_func(x, a) * a;
}

// =======================
// interfece for inference
// =======================
int omp_schedule(int count);

void function_relu(float *src, float *dst, int64_t size, float relu_limit = 0.f,
                   mlir::Type elem_type = nullptr);

// =======================
// interfece for quantization
// =======================
void get_scale_and_shift(float scale_f, int &scale, int &shift,
                         int bitwidth = 32);
void get_scale_and_shift_positive(float scale_f, int &scale, int &shift,
                                  int bitwidth = 32);
void get_scale_and_shift_positive_maxshift(float scale_f, int &scale,
                                           int &shift, int bitwidth,
                                           int max_shift = 8);
template <typename Dtype> float findMaxabs(const Dtype *pSrcData, int len);
template <typename Dtype>
void findMinMax(const Dtype *pSrcData, int len, Dtype *minVal, Dtype *maxVal);
int calRightShiftNum(float fmax, double thBottom, double thTop, int numBits);
template <typename T> void func_abs(int n, T *src, T *dst);
template <typename T> void func_log(int n, T *src, T *dst);
int calRightShiftNumUseCblas(float fmax, double thBottom, double thTop,
                             int numBits);
float func_log2(double dataInput);
void quantizeToInt32(const float *pSrc, int32_t *pDst, int len, double scale);
float quantizeToInt16(const float *pSrc, int16_t *pDst, int len, float scale,
                      int rshift = 0);
float quantizeToInt15(const float *pSrc, int16_t *pDst, int len, float scale,
                      int rshift = 0);
template <typename T>
void quantizeToInt8(const float *pSrc, T *pDst, int len, double scale,
                    RoundingMode round_mode = ROUNDING_HALF_AWAY_FROM_ZERO);
// to compitable with tflite
void QuantizeMultiplier(double double_multiplier, int64_t *quantized_multiplier,
                        int64_t *shift);
// cv18xx
double getQscaleForFilter(float max_filter, float threshold_y,
                          float threshold_x, int quant_bitwidth = 8);

double getQscaleForBias(float max_bias, float threshold_y);
void getRShiftAndMultiplierFromQScale(double double_multiplier,
                                      int64_t *quantized_multiplier,
                                      int64_t *shift, bool qdm = false,
                                      int64_t max_multiplier = 127);
int8_t getMultiplierI8FromQScaleAndRShift(double qscale, int8_t rshift);
void quantizeFilterRShiftAndMultiplier(const float *pSrc, int8_t *pDst, int len,
                                       float threshold_y, float threshold_x,
                                       int64_t rshift, int64_t multiplier,
                                       bool qdm = false);
void quantizeBiasRShiftAndMultiplier(const float *pSrc, int32_t *pDst, int len,
                                     float threshold_y, int64_t rshift,
                                     int64_t multiplier, bool qdm = false);

template <typename T>
T RightShiftRound(T src, int shift_num, RoundingMode round_mode);
// to compilable with tflite
int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t multiplier, int shift);
int64_t applyMultiplierAndRShift(int64_t v, int64_t multiplier, int64_t rshift,
                                 MultiplierType m_type = BM_QUANT);

void pad_tensor(float *p_after_pad, float *src, int n, int c, int h, int w,
                int pt, int pb, int pl, int pr, float pad_value);
void pad_tensor(float *p_after_pad, float *src, int n, int c, int d, int h,
                int w, int pdf, int pdb, int pht, int phb, int pwl, int pwr,
                float pad_value);
void pad_tensor_for_deconv(float *p_after_pad, float *src, int n, int c, int d,
                           int h, int w, int kd, int kh, int kw, int dd, int dh,
                           int dw, int sd, int sh, int sw, int pdf, int pdb,
                           int pht, int phb, int pwl, int pwr, float pad_value);
void tensor_sub_zp(float *tensor_after_zp, float *src, int64_t length,
                   float zero_point);
void tensor_hw_transpose(float *dst, float *src, int64_t N, int64_t C,
                         int64_t H, int64_t W);
void tensor_split(float *src_data, std::vector<std::vector<float>> &dst_data,
                  std::vector<int64_t> &shape, int slice_num, int axis);

int dnnl_mm(float *input, float *weight, float *bias, float *output, int m,
            int k, int n, bool transpose);

int8_t quantizeFilterRShift(float w, float threshold_y, float threshold_x,
                            uint32_t rshift);


static inline int32_t saturate(int32_t input, mlir::Type stype) {
  int32_t output;
  if (stype.isUnsignedInteger(8))
    output = input > 255 ? 255 : input < 0 ? 0 : input;
  else if (stype.isSignedInteger(8))
    output = input > 127 ? 127 : input < -128 ? -128 : input;
  else if (stype.isUnsignedInteger(16))
    output = input > 65535 ? 65535 : input < 0 ? 0 : input;
  else if (stype.isSignedInteger(16))
    output = input > 32767 ? 32767 : input < -32768 ? -32768 : input;
  else if (stype.isUnsignedInteger(4))
    output = input > 15 ? 15 : input < 0 ? 0 : input;
  else if (stype.isSignedInteger(4))
    output = input > 7 ? 7 : input < -8 ? -8 : input;
  else
    output = input;
  return output;
}

// to compilable with tflite stride slice
void stride_slice_gen_params(const int64_t *input_shape_, int input_dim_,
                             const float *begin_index_, const float *end_index_,
                             const float *strides_, int strides_size,
                             int begin_mask_, int end_mask_, int ellipsis_mask_,
                             int new_axis_mask_, int shrink_axis_mask_,
                             int *input_shape, int *input_dim, int *begin_index,
                             int *end_index, int *strides, int *begin_mask,
                             int *end_mask, int *shrink_axis_mask);
int StartForAxis(const int *start_indices, const int *strides, const int mask,
                 const int *shape, const int axis);
int StopForAxis(const int *stop_indices, const int *strides, const int mask,
                const int shrink_mask, const int *shape, const int axis,
                int start_for_axis);
std::vector<int64_t> shape_expand_dim(llvm::ArrayRef<int64_t> shape, int dims);

// compare
bool compare(float lhs, float rhs, llvm::StringRef mode);

// to compilable with gemmlowp
int32_t exp_on_negative_values(int input, int int_bits);

} // namespace tpu_mlir
