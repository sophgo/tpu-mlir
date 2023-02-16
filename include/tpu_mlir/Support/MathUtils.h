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
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

namespace tpu_mlir {

// =======================
// constant
// =======================
static constexpr double QMAX_INT8 = 127.0;
static constexpr int BITS_INT8 = 8;

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

void topk_indices(std::vector<std::pair<int, float>> &result,
                  const float *items, int num_elem, int k, bool largest);
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
void quantizeToInt32(const float *pSrc, int32_t *pDst, int len, float scale);
float quantizeToInt16(const float *pSrc, int16_t *pDst, int len, float scale,
                      int rshift = 0);
float quantizeToInt15(const float *pSrc, int16_t *pDst, int len, float scale,
                      int rshift = 0);
template <typename T>
void quantizeToInt8(const float *pSrc, T *pDst, int len, float scale,
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
int32_t MultiplyByQuantizedMultiplier(
    int32_t x, int32_t multiplier, int shift,
    RoundingMode rmode=ROUNDING_HALF_AWAY_FROM_ZERO);
int64_t applyMultiplierAndRShift(
    int64_t v, int64_t multiplier, int64_t rshift,
    tpu::RequantMode qmode = tpu::RequantMode::MultiplierShift,
    RoundingMode rmode=ROUNDING_HALF_UP);

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

std::shared_ptr<std::vector<float>>
binary_add(float *a, float *b, const llvm::ArrayRef<int64_t> &a_shape,
           const llvm::ArrayRef<int64_t> &b_shape,
           std::vector<int64_t> &o_shape);

int8_t quantizeFilterRShift(float w, float threshold_y, float threshold_x,
                            uint32_t rshift);

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

// reset pad to 4 dim
bool pad_reset(const std::vector<int64_t> &shape,
               const std::vector<int64_t> &pads, std::vector<int64_t> &shape_4,
               std::vector<int64_t> &pads_4);

// reset permtue to 4dim or 5dim
bool permute_reset(const std::vector<int64_t> &shape,
                   const std::vector<int64_t> &order,
                   std::vector<int64_t> &to_shape,
                   std::vector<int64_t> &to_order, int to_dim);
template <typename T>
void function_permute(T *from, T *to, const std::vector<int64_t> &shape_5,
                      const std::vector<int64_t> &order_5);

// compare
bool compare(float lhs, float rhs, llvm::StringRef mode);

// to compilable with gemmlowp
int32_t exp_on_negative_values(int input, int int_bits);

template <typename T> static int64_t to_int(T v, RoundingMode round_mode) {
  int64_t i64_val;
  if (round_mode == ROUNDING_HALF_AWAY_FROM_ZERO) {
    i64_val = std::round(v);
  } else if (round_mode == ROUNDING_DOWN) {
    i64_val = (int64_t)v;
  } else if (round_mode == ROUNDING_HALF_TO_EVEN) {
    float fraction, integer;
    float abs_v = std::abs(v);
    fraction = std::modf(abs_v, &integer);
    i64_val = (int64_t)integer;
    if (fraction > 0.5) {
      i64_val = i64_val + 1;
    } else if (fraction == 0.5) {
      if (i64_val & 0x01) {
        i64_val = i64_val + 1;
      }
    }
    if (v < 0) {
      i64_val = -i64_val;
    }
  } else if (round_mode == ROUNDING_HALF_UP) {
    i64_val = std::floor(v + 0.5);
  } else if (round_mode == ROUNDING_HALF_DOWN) {
    i64_val = std::ceil(v - 0.5);
  } else {
    llvm_unreachable("not support round_mode.");
  }
  return i64_val;
}

template <typename T>
static int64_t
saturate(T v, mlir::Type type,
         RoundingMode round_mode = ROUNDING_HALF_AWAY_FROM_ZERO) {
  auto itype = dyn_cast<mlir::IntegerType>(type);
  if (!itype) {
    type.dump();
    llvm_unreachable("not support type");
  }
  int64_t max, min;
  auto N = itype.getWidth();
  if (itype.isUnsigned()) {
    max = llvm::maxUIntN(N);
    min = 0;
  } else {
    max = llvm::maxIntN(N);
    min = llvm::minIntN(N);
  }
  v = to_int(v, round_mode);
  if (v > max) {
    v = max;
  } else if (v < min) {
    v = min;
  }
  return v;
}

template <typename T>
int8_t to_int8(T value,
               RoundingMode round_mode = ROUNDING_HALF_AWAY_FROM_ZERO) {
  auto v = to_int(value, round_mode);
  return v > 127 ? 127 : v < -128 ? -128 : v;
};

template <typename T>
uint8_t to_uint8(T value,
                 RoundingMode round_mode = ROUNDING_HALF_AWAY_FROM_ZERO) {
  auto v = to_int(value, round_mode);
  return v > 255 ? 255 : v < 0 ? 0 : v;
}

template <typename T>
int8_t to_int4(T value,
               RoundingMode round_mode = ROUNDING_HALF_AWAY_FROM_ZERO) {
  auto v = to_int(value, round_mode);
  return v > 7 ? 7 : v < -8 ? -8 : v;
};

template <typename T>
uint8_t to_uint4(T value,
                 RoundingMode round_mode = ROUNDING_HALF_AWAY_FROM_ZERO) {
  auto v = to_int(value, round_mode);
  return v > 15 ? 15 : v < 0 ? 0 : v;
}

void swap_dim_data(float *input, float *output, std::vector<int64_t> &ishape,
                   std::vector<int64_t> &offsets);

} // namespace tpu_mlir
