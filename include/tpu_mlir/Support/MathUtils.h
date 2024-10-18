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
#include <vector>
#include <unordered_map>
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
template <typename T>
static inline T abs_ceiling_func(T numerator, T denominator) {
  return (std::abs(numerator + denominator) - 1) / std::abs(denominator);
}

template <typename U, typename V>
static inline auto ceiling_func(U numerator, V denominator)
    -> decltype(numerator + denominator) {
  return (numerator + denominator - 1) / denominator;
}

template <typename U, typename V>
static inline auto align_up(U x, V a) -> decltype(x + a) {
  return ceiling_func(x, a) * a;
}

template <typename U, typename V>
static inline auto align_down(U x, V a) -> decltype(x - a) {
  return (x / a) * a;
}

template <typename U, typename V>
static inline auto align_nearest(U value, V multiple) {
  auto remainder = value % multiple;
  if (remainder == 0) {
    return value;
  }

  auto roundDown = value - remainder;
  auto roundUp = roundDown + multiple;

  return (value - roundDown < roundUp - value) ? roundDown : roundUp;
}

template <typename T>
static inline void get_factory(T num, std::vector<T> &factors, bool calc_set=false) {
  for (T d = 2; d*d <= num; d++) {
      if (num%d == 0) {
          factors.push_back(d);
          num /= d;
          while (num % d == 0)
          {
            if(!calc_set){
              factors.push_back(d);
            }
            num /= d;
          }
      }
  }
  if (num > 1) {
    factors.push_back(num);
  }
}

// =======================
// interfece for inference
// =======================
int omp_schedule(int count);

void function_relu(float *src, float *dst, int64_t size, float relu_limit = 0.f,
                   mlir::Type elem_type = nullptr);

template <typename T>
void topk_indices(std::vector<std::pair<int, T>> &result, const T *items,
                  int num_elem, int k, bool largest);
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
    RoundingMode rmode = ROUNDING_HALF_AWAY_FROM_ZERO);
int64_t applyMultiplierAndRShift(
    int64_t v, int64_t multiplier, int64_t rshift,
    tpu::RequantMode qmode = tpu::RequantMode::MultiplierShift,
    RoundingMode rmode = ROUNDING_HALF_UP);

void pad_tensor(float *p_after_pad, float *src, int n, int c, int h, int w,
                int pt, int pb, int pl, int pr, float pad_value);
void pad_tensor(float *p_after_pad, float *src, int n, int c, int d, int h,
                int w, int pdf, int pdb, int pht, int phb, int pwl, int pwr,
                float pad_value);
void pad_tensor_for_deconv(float *p_after_pad, float *src, int n, int c, int d,
                           int h, int w, int kd, int kh, int kw, int dd, int dh,
                           int dw, int sd, int sh, int sw, int pdf, int pdb,
                           int pht, int phb, int pwl, int pwr, int opd, int oph,
                           int opw, float pad_value);
void dilate_tensor(float *p_after_pad, float *src, int n, int c, int d, int h,
                   int w, int pdf, int pdb, int pht, int phb, int pwl, int pwr,
                   float pad_value, int ins_h, int ins_w, float ins_value);
void tensor_sub_zp(float *tensor_after_zp, float *src, int64_t length,
                   float zero_point);
template <typename T>
void tensor_hw_transpose(T *dst, T *src, int64_t N, int64_t C,
                         int64_t H, int64_t W) {
#pragma omp parallel for schedule(static, omp_schedule(N *C))
  for (int64_t nc = 0; nc < N * C; ++nc) {
    int64_t nc_offset = nc * H * W;
    for (int w = 0; w < W; ++w) {
      for (int h = 0; h < H; ++h) {
        int64_t d_offset = nc_offset + w * H + h;
        int64_t s_offset = nc_offset + h * W + w;
        dst[d_offset] = src[s_offset];
      }
    }
  }
}
template <typename T>
void tensor_hc_transpose(T *dst, T *src, int64_t N, int64_t C,
                         int64_t H, int64_t W) {
#pragma omp parallel for schedule(static, omp_schedule(N))
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t c = 0; c < C; ++c) {
        for (int64_t w = 0; w < W; ++w) {
          int64_t s_offset = w + h * W + c * H * W + n * C * H * W;
          int64_t d_offset = w + c * W + h * C * W + n * C * H * W;
          dst[d_offset] = src[s_offset];
        }
      }
    }
  }
}
void tensor_split(float *src_data, std::vector<std::vector<float>> &dst_data,
                  std::vector<int64_t> &shape, int slice_num, int axis);
template <typename T>
std::shared_ptr<std::vector<T>>
tensor_slice(T *src_data, const std::vector<int64_t> &shape, int64_t axis,
             int64_t offset, int64_t length, std::string mode);

int dnnl_mm(float *input, float *weight, float *bias, float *output, int m,
            int k, int n, bool transpose);

std::shared_ptr<std::vector<float>>
binary_add(float *a, float *b, const llvm::ArrayRef<int64_t> &a_shape,
           const llvm::ArrayRef<int64_t> &b_shape,
           std::vector<int64_t> &o_shape);
std::shared_ptr<std::vector<float>>
binary_mul(float *a, float *b, const llvm::ArrayRef<int64_t> &a_shape,
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
template <typename T>
std::vector<int64_t> shape_expand_dim(llvm::ArrayRef<T> shape, int dims);
template <typename T>
std::vector<int64_t> shape_expand_dim(const std::vector<T> &shape, int dims);
std::vector<int64_t> channel_expand_dim(llvm::ArrayRef<int64_t> shape,
                                        int dims);
template <typename T>
void tile(T *input, T *output, llvm::ArrayRef<int64_t> in_shape, int axis,
          int times);

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
void function_permute(T *from, T *to, const std::vector<int64_t> &shape,
                      const std::vector<int64_t> &order);

// compare
bool compare(float lhs, float rhs, llvm::StringRef mode);

// to compilable with gemmlowp
int32_t exp_on_negative_values(int input, int int_bits);

template <typename T> int64_t to_int(T v, RoundingMode round_mode);
template <typename T>
int64_t saturate(T v, mlir::Type type,
                 RoundingMode round_mode = ROUNDING_HALF_AWAY_FROM_ZERO);
extern template int64_t saturate<float>(float v, mlir::Type type,
                                        RoundingMode round_mode);
extern template int64_t saturate<double>(double v, mlir::Type type,
                                         RoundingMode round_mode);
template <typename T>
int16_t to_int16(T value,
                 RoundingMode round_mode = ROUNDING_HALF_AWAY_FROM_ZERO) {
  auto v = to_int(value, round_mode);
  return v > 32767 ? 32767 : v < -32768 ? -32768 : v;
};

template <typename T>
uint16_t to_uint16(T value,
                   RoundingMode round_mode = ROUNDING_HALF_AWAY_FROM_ZERO) {
  auto v = to_int(value, round_mode);
  return v > 65535 ? 65535 : v < 0 ? 0 : v;
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

// convert all data to int8 by scale
bool is_all_int8(const std::vector<float> &data, float scale = 1.0,
                 bool sign = true);
bool to_all_int8(const std::vector<float> &data, float &scale,
                 bool sign = true);

void swap_dim_data(float *input, float *output, std::vector<int64_t> &ishape,
                   std::vector<int64_t> &offsets);

void idx_to_list(int64_t idx, const std::vector<int64_t> &dim,
                 std::vector<int64_t> &idx_res);

// convert shape to index for gaven stride
int64_t list_to_idx(const std::vector<int64_t> &list,
                    const std::vector<int64_t> &stride);

// get the stride for the gaven shape
void get_stride(const std::vector<int64_t> &shape,
                std::vector<int64_t> &stride);

int getBcastIndex(int out_index, std::vector<int64_t> &output_shape,
                  std::vector<int64_t> &input_shape);

void set_auto_pad(llvm::StringRef mode, const std::vector<int64_t> &input_shape,
                  const std::vector<int64_t> &kernel_shape,
                  const std::vector<int64_t> &strides,
                  std::vector<int64_t> &pads);

typedef struct {
  int axis;
  int descending;
} sort_param_t;

void sort_per_dim(const sort_param_t& param, const int* shape, int dims, const float* input, float* sorted_values, float* sorted_indices);

RoundingMode round_mode_convert(tpu::RoundMode mode);

void distribute_elements(const std::vector<int64_t>& elements,
                         const std::vector<int64_t>& limits,
                         std::vector<std::vector<int64_t>>& result,
                         std::vector<int64_t>& current,
                         int index = 0);

std::vector<std::vector<int64_t>> find_distributions(const std::vector<int64_t>& elements,
                                                     const std::vector<int64_t>& limits);

} // namespace tpu_mlir
