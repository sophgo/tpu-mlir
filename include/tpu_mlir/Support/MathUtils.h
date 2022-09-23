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
float quantizeToInt16(const float *pSrc, int16_t *pDst, int len, float scale,
                      int rshift = 0);
float quantizeToInt15(const float *pSrc, int16_t *pDst, int len, float scale,
                      int rshift = 0);
void quantizeToInt8(const float *pSrc, int8_t *pDst, int len, float scale);
// to compitable with tflite
void QuantizeMultiplier(double double_multiplier, int64_t *quantized_multiplier,
                        int64_t *shift);
// define round mode
typedef enum {
  ROUNDING_HALF_TO_EVEN = 0,
  ROUNDING_HALF_AWAY_FROM_ZERO = 1,
  ROUNDING_TOWARDS_ZERO = 2,
  ROUNDING_DOWN = 3, /* FLOOR */
  ROUNDING_UP = 4,   /* CEIL */
  ROUNDING_HALF_UP = 5,
  ROUNDING_HALF_DOWN = 6,
  ROUNDING_UNKNOWN = -1
} RoundingMode;
template <typename T>
T RightShiftRound(T src, int shift_num, RoundingMode round_mode);
// to compilable with tflite
int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t multiplier, int shift);
static inline int64_t applyMultiplierAndRShift(int64_t v, int64_t multiplier,
                                               int64_t rshift) {
  return RightShiftRound(v * multiplier, (int)rshift, ROUNDING_HALF_UP);
}
void pad_tensor(float *p_after_pad, float *src, int n, int c, int h, int w,
                int pt, int pb, int pl, int pr, float pad_value);
void pad_tensor(float *p_after_pad, float *src, int n, int c, int d, int h,
                int w, int pdf, int pdb, int pht, int phb, int pwl, int pwr,
                float pad_value);
void pad_tensor_for_deconv(float *p_after_pad, float *src, int n, int c, int d,
                           int h, int w, int kd, int kh, int kw, int dd, int dh,
                           int dw, int sd, int sh, int sw, int pdf, int pdb,
                           int pht, int phb, int pwl, int pwr,
                           float pad_value);
void tensor_sub_zp(float* tensor_after_zp, float* src, int64_t length, float zero_point);

int dnnl_mm(float *input, float *weight, float *bias, float *output, int m,
              int k, int n, bool transpose);

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
  else
    output = input;
  return output;
}
} // namespace tpu_mlir
