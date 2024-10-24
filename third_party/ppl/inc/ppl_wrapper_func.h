#pragma once
#include "ppl_dma_func.h"
#include "ppl_func.h"
#include "ppl_hau_func.h"
#include "ppl_tiu_func.h"
#include "ppl_tpu.h"
#include "ppl_utils.h"

using namespace ppl;

template <typename T> auto getSignMask() {
  if constexpr (std::is_same<T, float>::value) {
    return static_cast<int32>(0x80000000);
  } else if constexpr (std::is_same<T, fp16>::value) {
    return static_cast<int16>(0x8000);
  } else if constexpr (std::is_same<T, bf16>::value) {
    return static_cast<int16>(0x8000);
  } else if constexpr (std::is_same<T, int8>::value) {
    return static_cast<int8>(0x80);
  } else if constexpr (std::is_same<T, uint8>::value) {
    return static_cast<uint8>(0x80);
  } else {
    static_assert(false, "Unsupported DataType");
  }
}

double log_approximation(double x) {
    if (x <= 0) {
        return -1;
    }
    double sum = 0.0;
    int n = 1000;
    double term = (x - 1) / (x + 1);
    double term_sq = term * term;
    double numerator = term;
    double denominator = 1.0;
    for (int i = 1; i <= n; i += 2) {
        sum += numerator / denominator;
        numerator *= term_sq;
        denominator += 2.0;
    }
    return 2 * sum;
}

void sigmoid_fp32(tensor<fp32> &local_output, tensor<fp32> &local_input,
                  dim4 *shape) {
  auto local_input_exp = tensor<fp32>(shape);
  tiu::fexp(local_input_exp, local_input);

  auto local_input_exp_reciprocal = tensor<fp32>(shape);
  tiu::fdiv(local_input_exp_reciprocal, 1, local_input_exp, 3);

  auto local_output_pre = tensor<fp32>(shape);
  tiu::fadd(local_output_pre, local_input_exp_reciprocal, 1);

  tiu::fdiv(local_output, 1, local_output_pre, 3);
}

template <typename DataType>
void prelu(tensor<DataType> &out, tensor<DataType> &in, float alpha) {
  if constexpr (std::is_same<DataType, uint32>::value ||
                std::is_same<DataType, uint16>::value ||
                std::is_same<DataType, uint8>::value) {
    dma::move(out, in);
  } else {
    float zero = 0;
    if constexpr (std::is_same<DataType, float>::value ||
                  std::is_same<DataType, fp16>::value ||
                  std::is_same<DataType, bf16>::value) {
      tiu::fmul(out, in, alpha);
    } else {
      tiu::mul(out, in, (int8)alpha, 0, RM_DOWN, true);
    }
    tiu::gt_select(out, in, zero, in, out);
  }
}

template <typename DataType>
void relu(tensor<DataType> &out, tensor<DataType> &in) {
  float zero = 0;
  tiu::max(out, in, zero);
}

template <typename DataType>
void fsqrt(tensor<DataType> &out, tensor<DataType> &in, dim4 *shape,
           dim4 *real_shape) {
  auto zero = getSignMask<DataType>();
  auto neg_zero = getSignMask<DataType>();
  if constexpr (std::is_same<DataType, float>::value) {
    zero = 0;
    neg_zero = 0x80000000;
  } else if constexpr (std::is_same<DataType, fp16>::value) {
    zero = 0;
    neg_zero = 0x8000;
  } else if constexpr (std::is_same<DataType, bf16>::value) {
    zero = 0;
    neg_zero = 0x8000;
  } else {
    static_assert(false, "Unsupported DataType");
  }
  auto rsqrt = make_tensor<DataType>(shape, real_shape);
  auto sel_0 = make_tensor<DataType>(shape, real_shape);
  auto sel_neg0 = make_tensor<DataType>(shape, real_shape);

  tiu::frsqrt(rsqrt, in);
  tiu::eq_select(sel_0, in, zero, zero, rsqrt);
  tiu::eq_select(sel_neg0, in, neg_zero, zero, sel_0);
  tiu::fmul(out, sel_neg0, in);
}

template <typename DataType>
void fsin(tensor<DataType> &out, tensor<DataType> &in, dim4 *shape,
          dim4 *real_shape) {
  const float PI = 3.14159265358979323846;
  float C = 1.0 / (2.0 * PI);
  auto work = make_tensor<DataType>(shape, real_shape);
  auto work1 = make_tensor<DataType>(shape, real_shape);
  auto work2 = make_tensor<DataType>(shape, real_shape);
  auto work3 = make_tensor<DataType>(shape, real_shape);
  tiu::fmul(work, in, C);
  tiu::round(work1, work, RM_HALF_AWAY_FROM_ZERO);
  tiu::fsub(work2, work, work1);
  tiu::fsin_base(out, work2);
}

template <typename DataType>
void fcos(tensor<DataType> &out, tensor<DataType> &in, dim4 *shape,
          dim4 *real_shape) {
  const float PI = 3.14159265358979323846;
  float C = 1.0 / (2.0 * PI);
  auto work = make_tensor<DataType>(shape, real_shape);
  auto work1 = make_tensor<DataType>(shape, real_shape);
  auto work2 = make_tensor<DataType>(shape, real_shape);
  tiu::fmul(work, in, C);
  tiu::round(work1, work, RM_HALF_AWAY_FROM_ZERO);
  tiu::fsub(work2, work, work1);
  tiu::fcos_base(out, work2);
}

template <typename DataType>
void farcsin(tensor<DataType> &out, tensor<DataType> &in, dim4 *shape,
             dim4 *real_shape) {
  auto sign_mask = getSignMask<DataType>();
  if constexpr (std::is_same<DataType, float>::value) {
    sign_mask = 0x80000000;
  } else if constexpr (std::is_same<DataType, fp16>::value) {
    sign_mask = 0x8000;
  } else if constexpr (std::is_same<DataType, bf16>::value) {
    sign_mask = 0x8000;
  } else {
    static_assert(false, "Unsupported DataType");
  }

  auto t_square = make_tensor<DataType>(shape, real_shape);
  auto t_mul025 = make_tensor<DataType>(shape, real_shape);
  auto t_025sub = make_tensor<DataType>(shape, real_shape);
  auto t_sqrt1 = make_tensor<DataType>(shape, real_shape);
  auto t_05sub = make_tensor<DataType>(shape, real_shape);
  auto t_sqrt2 = make_tensor<DataType>(shape, real_shape);
  auto t_arcsin = make_tensor<DataType>(shape, real_shape);
  auto t_mul2 = make_tensor<DataType>(shape, real_shape);
  auto t_sign = make_tensor<DataType>(shape, real_shape);

  tiu::fmul(t_square, in, in);
  tiu::fmul(t_mul025, t_square, 0.25);
  tiu::fsub(t_025sub, 0.25, t_mul025);
  tiu::fsqrt(t_sqrt1, t_025sub);
  tiu::fsub(t_05sub, 0.5, t_sqrt1);
  tiu::fsqrt(t_sqrt2, t_05sub);
  tiu::farcsin_base(t_arcsin, t_sqrt2);
  tiu::fmul(t_mul2, t_arcsin, 2.0);
  tiu::bitwise_and(t_sign, in, sign_mask);
  tiu::bitwise_or(out, t_sign, t_mul2);
}

template <typename DataType>
void farccos(tensor<DataType> &out, tensor<DataType> &in, dim4 *shape,
             dim4 *real_shape) {
  const float PI_05 = 3.14159265358979323846 / 2;
  farcsin(out, in, shape, real_shape);
  tiu::fsub(out, PI_05, out);
}

template <typename DataType>
void ftan(tensor<DataType> &out, tensor<DataType> &in, dim4 *shape,
          dim4 *real_shape) {
  const float PI = 3.14159265358979323846;
  float C = 1.0 / PI;
  float C025 = 0.25;
  float C1 = 1;
  float C2 = 2;

  auto sign_mask = getSignMask<DataType>();
  if constexpr (std::is_same<DataType, float>::value) {
    sign_mask = 0x80000000;
  } else if constexpr (std::is_same<DataType, fp16>::value) {
    sign_mask = 0x8000;
  } else if constexpr (std::is_same<DataType, bf16>::value) {
    sign_mask = 0x8000;
  } else {
    static_assert(false, "Unsupported DataType");
  }

  auto t_mul = make_tensor<DataType>(shape, real_shape);
  auto t_round = make_tensor<DataType>(shape, real_shape);
  auto t_rem = make_tensor<DataType>(shape, real_shape);
  auto t_abs = make_tensor<DataType>(shape, real_shape);
  auto t_sub = make_tensor<DataType>(shape, real_shape);
  auto t_tanx_sub_025pi = make_tensor<DataType>(shape, real_shape);
  auto t_c_div = make_tensor<DataType>(shape, real_shape);
  auto t_sub_c = make_tensor<DataType>(shape, real_shape);
  auto t_and = make_tensor<DataType>(shape, real_shape);
  auto t_square = make_tensor<DataType>(shape, real_shape);

  tiu::fmul(t_mul, in, C);
  tiu::round(t_round, t_mul, RM_HALF_TO_EVEN);
  tiu::fsub(t_rem, t_mul, t_round);
  tiu::abs(t_abs, t_rem);
  tiu::fsub(t_sub, t_abs, C025);
  tiu::ftan_base(t_tanx_sub_025pi, t_sub);
  tiu::fsub(t_sub, C1, t_tanx_sub_025pi);
  tiu::fdiv(t_c_div, C2, t_sub);
  tiu::fsub(t_sub_c, t_c_div, C1);
  tiu::bitwise_and(t_and, t_rem, sign_mask);
  tiu::bitwise_or(out, t_and, t_sub_c);
}

template <typename DataType>
void fcot(tensor<DataType> &out, tensor<DataType> &in, dim4 *shape,
          dim4 *real_shape) {
  const float PI = 3.14159265358979323846;
  float C = 1.0 / PI;
  float C05 = 0.5;
  float C025 = 0.25;
  float C1 = 1;
  float C2 = 2;

  auto sign_mask = getSignMask<DataType>();
  if constexpr (std::is_same<DataType, float>::value) {
    sign_mask = 0x80000000;
  } else if constexpr (std::is_same<DataType, fp16>::value) {
    sign_mask = 0x8000;
  } else if constexpr (std::is_same<DataType, bf16>::value) {
    sign_mask = 0x8000;
  } else {
    static_assert(false, "Unsupported DataType");
  }

  auto t_mul = make_tensor<DataType>(shape, real_shape);
  auto t_csub = make_tensor<DataType>(shape, real_shape);
  auto t_round = make_tensor<DataType>(shape, real_shape);
  auto t_rem = make_tensor<DataType>(shape, real_shape);
  auto t_abs = make_tensor<DataType>(shape, real_shape);
  auto t_sub = make_tensor<DataType>(shape, real_shape);
  auto t_tanx_sub_025pi = make_tensor<DataType>(shape, real_shape);
  auto t_c_div = make_tensor<DataType>(shape, real_shape);
  auto t_sub_c = make_tensor<DataType>(shape, real_shape);
  auto t_and = make_tensor<DataType>(shape, real_shape);
  auto t_square = make_tensor<DataType>(shape, real_shape);

  tiu::fmul(t_mul, in, C);
  tiu::fsub(t_csub, C05, t_mul);
  tiu::round(t_round, t_csub, RM_HALF_TO_EVEN);
  tiu::fsub(t_rem, t_csub, t_round);
  tiu::abs(t_abs, t_rem);
  tiu::fsub(t_sub, t_abs, C025);
  tiu::ftan_base(t_tanx_sub_025pi, t_sub);
  tiu::fsub(t_sub, C1, t_tanx_sub_025pi);
  tiu::fdiv(t_c_div, C2, t_sub);
  tiu::fsub(t_sub_c, t_c_div, C1);
  tiu::bitwise_and(t_and, t_rem, sign_mask);
  tiu::bitwise_or(out, t_and, t_sub_c);
}

template <typename DataType>
void exp_no_overflow(tensor<DataType> &out, tensor<DataType> &in, dim4 *shape,
                     dim4 *real_shape) {
  fp32 min_C = 0;
  if constexpr (std::is_same<DataType, fp32>::value) {
    min_C = -3.40282e35f;
  } else if (std::is_same<DataType, fp16>::value) {
    min_C = -45403.f;
  } else {
    min_C = -3.40282e35f;
  }

  auto maxc_tensor = make_tensor<DataType>(shape, real_shape);
  tiu::fmax(maxc_tensor, in, min_C);

  auto minc_tensor1 = make_tensor<DataType>(shape, real_shape);
  if (std::is_same<DataType, fp16>::value) {
    tiu::fmin(minc_tensor1, maxc_tensor, 45403.f);
  } else {
    tiu::move(minc_tensor1, maxc_tensor);
  }

  auto fp_mulc_tensor = make_tensor<DataType>(shape, real_shape);
  tiu::fmul(fp_mulc_tensor, minc_tensor1, 1.4426950f);

  auto fp_floor_tensor = make_tensor<DataType>(shape, real_shape);
  tiu::floor(fp_floor_tensor, fp_mulc_tensor);

  auto fp_mulc_tensor2 = make_tensor<DataType>(shape, real_shape);
  tiu::fmul(fp_mulc_tensor2, fp_floor_tensor, 0.69314718f);

  auto fp_sub = make_tensor<DataType>(shape, real_shape);
  tiu::fsub(fp_sub, maxc_tensor, fp_mulc_tensor2);

  if constexpr (std::is_same<DataType, fp32>::value) {
    auto cast_out = make_tensor<int16>(shape, real_shape);
    tiu::cast(cast_out, fp_floor_tensor, RM_HALF_AWAY_FROM_ZERO);

    auto minc_tensor = make_tensor<int16>(shape, real_shape);
    tiu::min(minc_tensor, cast_out, (int16)127);

    auto maxc_tensor2 = make_tensor<int16>(shape, real_shape);
    tiu::max(maxc_tensor2, minc_tensor, (int16)-127);

    auto add_intc_tensor = make_tensor<int32>(shape, real_shape);
    tiu::add(add_intc_tensor, maxc_tensor2, (int16)127, 23,
             RM_HALF_AWAY_FROM_ZERO, true);

    auto exp_out = make_tensor<fp32>(shape, real_shape);
    tiu::fexp(exp_out, fp_sub);

    auto cast_intc_tensor = add_intc_tensor.view<fp32>();

    tiu::fmul(out, exp_out, cast_intc_tensor);
  } else if constexpr (std::is_same<DataType, fp16>::value) {
    auto cast_out = make_tensor<int8>(shape, real_shape);
    tiu::cast(cast_out, fp_floor_tensor, RM_HALF_AWAY_FROM_ZERO);

    auto minc_tensor = make_tensor<int8>(shape, real_shape);
    tiu::min(minc_tensor, cast_out, (int16)15);

    auto maxc_tensor2 = make_tensor<int8>(shape, real_shape);
    tiu::max(maxc_tensor2, minc_tensor, (int16)-15);

    auto add_intc_tensor = make_tensor<int16>(shape, real_shape);
    tiu::add(add_intc_tensor, maxc_tensor2, (int16)15, 10,
             RM_HALF_AWAY_FROM_ZERO, true);

    auto exp_out = make_tensor<fp16>(shape, real_shape);
    tiu::fexp(exp_out, fp_sub);

    auto cast_intc_tensor = add_intc_tensor.view<fp16>();

    tiu::fmul(out, exp_out, cast_intc_tensor);
  } else if constexpr (std::is_same<DataType, bf16>::value) {
    auto cast_out = make_tensor<int16>(shape, real_shape);
    tiu::cast(cast_out, fp_floor_tensor, RM_HALF_AWAY_FROM_ZERO);

    auto minc_tensor = make_tensor<int16>(shape, real_shape);
    tiu::min(minc_tensor, cast_out, (int16)127);

    auto maxc_tensor2 = make_tensor<int16>(shape, real_shape);
    tiu::max(maxc_tensor2, minc_tensor, (int16)-127);

    auto add_intc_tensor = make_tensor<int16>(shape, real_shape);
    tiu::add(add_intc_tensor, maxc_tensor2, (int16)127, 7,
             RM_HALF_AWAY_FROM_ZERO, true);

    auto exp_out = make_tensor<bf16>(shape, real_shape);
    tiu::fexp(exp_out, fp_sub);

    auto cast_intc_tensor = add_intc_tensor.view<bf16>();

    tiu::fmul(out, exp_out, cast_intc_tensor);
  }
}

template <typename DataType>
void exp_no_overflow(tensor<DataType> &out, tensor<DataType> &in, dim4 *shape) {
  exp_no_overflow(out, in, shape, shape);
}

template <typename DataType>
void flog(tensor<DataType> &out, tensor<DataType> &in, dim4 *block_shape,
          dim4 *real_shape) {
  if constexpr (std::is_same_v<DataType, fp32>) {
    fp32 max_C = 1.175494351e-38;
    auto max_out = make_tensor<DataType>(block_shape, real_shape);
    tiu::max(max_out, in, max_C);
    auto exp_part_out = make_tensor<int32>(block_shape, real_shape);
    tiu::fexp_part(exp_part_out, max_out);
    int C = 254;
    uint8 frac = 23;
    auto sub_out = make_tensor<int32>(block_shape, real_shape);
    tiu::sub(sub_out, C, exp_part_out, frac, RM_HALF_AWAY_FROM_ZERO, true);
    auto mul_out = make_tensor<DataType>(block_shape, real_shape);
    tiu::fmul(mul_out, sub_out.view<DataType>(), max_out);
    auto sub_out2 = make_tensor<DataType>(block_shape, real_shape);
    tiu::fsub(sub_out2, mul_out, 1.f);
    auto taylor_out = make_tensor<DataType>(block_shape, real_shape);
    tiu::flog_base(taylor_out, sub_out2);
    auto eq_out = make_tensor<DataType>(block_shape, real_shape);
    int32 C3 = 0;
    int32 C4 = 0xFF800000;
    tiu::eq_select(eq_out, in, C3, C4, taylor_out);
    auto sub_out3 = make_tensor<int32>(block_shape, real_shape);
    int C2 = 127;
    tiu::sub(sub_out3, exp_part_out, C2, 0, RM_HALF_AWAY_FROM_ZERO, true);
    auto cast_out = make_tensor<DataType>(block_shape, real_shape);
    tiu::cast(cast_out, sub_out3);
    auto mul_out2 = make_tensor<DataType>(block_shape, real_shape);
    tiu::fmul(mul_out2, cast_out, 0.69314718056f);
    tiu::fadd(out, eq_out, mul_out2);
  } else if (std::is_same_v<DataType, fp16>) {
    fp32 max_C = 1.175494351e-38;
    auto max_out = make_tensor<DataType>(block_shape, real_shape);
    tiu::max(max_out, in, max_C);
    auto exp_part_out = make_tensor<int16>(block_shape, real_shape);
    tiu::fexp_part(exp_part_out, max_out);
    int C = 30;
    uint8 frac = 10;
    auto sub_out = make_tensor<int16>(block_shape, real_shape);
    tiu::sub(sub_out, C, exp_part_out, frac, RM_HALF_AWAY_FROM_ZERO, true);
    auto mul_out = make_tensor<DataType>(block_shape, real_shape);
    tiu::fmul(mul_out, sub_out.view<DataType>(), max_out);
    auto sub_out2 = make_tensor<DataType>(block_shape, real_shape);
    tiu::fsub(sub_out2, mul_out, 1.f);
    auto taylor_out = make_tensor<DataType>(block_shape, real_shape);
    tiu::flog_base(taylor_out, sub_out2);
    auto eq_out = make_tensor<DataType>(block_shape, real_shape);
    int32 C3 = 0;
    int32 C4 = 0xFF800000;
    tiu::eq_select(eq_out, in, C3, C4, taylor_out);
    auto sub_out3 = make_tensor<int16>(block_shape, real_shape);
    int C2 = 15;
    tiu::sub(sub_out3, exp_part_out, C2, 0, RM_HALF_AWAY_FROM_ZERO, true);
    auto cast_out = make_tensor<DataType>(block_shape, real_shape);
    tiu::cast(cast_out, sub_out3);
    auto mul_out2 = make_tensor<DataType>(block_shape, real_shape);
    tiu::fmul(mul_out2, cast_out, 0.69314718056f);
    tiu::fadd(out, eq_out, mul_out2);
  } else if (std::is_same_v<DataType, bf16>) {
    fp32 max_C = 1.175494351e-38;
    auto max_out = make_tensor<DataType>(block_shape, real_shape);
    tiu::max(max_out, in, max_C);
    auto exp_part_out = make_tensor<int16>(block_shape, real_shape);
    tiu::fexp_part(exp_part_out, max_out);
    int C = 254;
    uint8 frac = 7;
    auto sub_out = make_tensor<int16>(block_shape, real_shape);
    tiu::sub(sub_out, C, exp_part_out, frac, RM_HALF_AWAY_FROM_ZERO, true);
    auto mul_out = make_tensor<DataType>(block_shape, real_shape);
    tiu::fmul(mul_out, sub_out.view<DataType>(), max_out);
    auto sub_out2 = make_tensor<DataType>(block_shape, real_shape);
    tiu::fsub(sub_out2, mul_out, 1.f);
    auto taylor_out = make_tensor<DataType>(block_shape, real_shape);
    tiu::flog_base(taylor_out, sub_out2);
    auto eq_out = make_tensor<DataType>(block_shape, real_shape);
    int32 C3 = 0;
    int32 C4 = 0xFF800000;
    tiu::eq_select(eq_out, in, C3, C4, taylor_out);
    auto sub_out3 = make_tensor<int16>(block_shape, real_shape);
    int C2 = 127;
    tiu::sub(sub_out3, exp_part_out, C2, 0, RM_HALF_AWAY_FROM_ZERO, true);
    auto cast_out = make_tensor<DataType>(block_shape, real_shape);
    tiu::cast(cast_out, sub_out3);
    auto mul_out2 = make_tensor<DataType>(block_shape, real_shape);
    tiu::fmul(mul_out2, cast_out, 0.69314718056f);
    tiu::fadd(out, eq_out, mul_out2);
  } else {
    assert(0);
  }
}

void mish_f32(tensor<fp32> &out, tensor<fp32> &in, dim4 *shape,
              dim4 *real_shape) {
  auto t_exp = tensor<fp32>(shape);
  auto t_fill_max = tensor<fp32>(shape);
  auto t_sel_max = tensor<fp32>(shape);
  auto t_fill_min = tensor<fp32>(shape);
  auto t_sel_min = tensor<fp32>(shape);
  auto t_add_sqr = tensor<fp32>(shape);
  auto t_add_c = tensor<fp32>(shape);
  auto t_div = tensor<fp32>(shape);
  auto t_mul_c = tensor<fp32>(shape);

  float float_sqrt_max = 1.84467e+19f; // SQRT(FLT_MAX)
  float float_sqrt_min = 1.08420e-19f; // SQRT(FLT_MIN)
  float C1 = 1.0;
  float C_2 = -2.0;

  exp_no_overflow(t_exp, in, shape, real_shape);
  // tiu::fill(t_fill_max, float_sqrt_max_);
  tiu::gt_select(t_sel_max, t_exp, float_sqrt_max, float_sqrt_max, t_exp);
  // tiu::fill(t_fill_min, float_sqrt_min_);
  tiu::lt_select(t_sel_min, t_sel_max, float_sqrt_min, float_sqrt_min, t_sel_max);
  tiu::fadd_sqr(t_add_sqr, t_sel_min, C1);
  tiu::fadd(t_add_c, t_add_sqr, C1);
  tiu::fdiv(t_div, in, t_add_c);
  tiu::fmul(t_mul_c, t_div, C_2);
  tiu::fadd(out, in, t_mul_c);
}

void pow_f32(tensor<fp32> &out, tensor<fp32> &in1, tensor<fp32> &in2,
             dim4 *shape, dim4 *real_shape) {
  auto t_log = tensor<fp32>(shape);
  auto t_mul = tensor<fp32>(shape);
  flog(t_log, in1, shape, real_shape);
  tiu::fmul(t_mul, t_log, in2);
  exp_no_overflow(out, t_mul, shape, real_shape);
}

void pow_f32(tensor<fp32> &out, tensor<fp32> &in1, float in2, dim4 *shape,
             dim4 *real_shape) {
  auto t_log = tensor<fp32>(shape);
  auto t_mul = tensor<fp32>(shape);
  flog(t_log, in1, shape, real_shape);
  tiu::fmul(t_mul, t_log, in2);
  exp_no_overflow(out, t_mul, shape, real_shape);
}

void pow_f32(tensor<fp32> &out, float in1, tensor<fp32> &in2, dim4 *shape,
             dim4 *real_shape) {
  auto t_mul = tensor<fp32>(shape);
  float x = log_approximation(in1);
  tiu::fmul(t_mul, in2, x);
  exp_no_overflow(out, t_mul, shape, real_shape);
}

template <typename T>
void flogx(tensor<T> &out, tensor<T> &in, float x, dim4 *block_shape,
           dim4 *real_shape) {
  auto tmp = make_tensor<T>(block_shape, real_shape);
  flog(tmp, in, block_shape, real_shape);
  tiu::fmul(out, tmp, 1 / log(x));
}

void softplus_f32(tensor<fp32> &out, tensor<fp32> &in, float beta, dim4 *shape,
                  dim4 *real_shape) {
  auto zero = getSignMask<fp32>();
  auto t_mul = make_tensor<fp32>(shape, real_shape);
  auto t_exp = make_tensor<fp32>(shape, real_shape);
  auto t_add = make_tensor<fp32>(shape, real_shape);
  auto t_log = make_tensor<fp32>(shape, real_shape);
  auto t_sel = make_tensor<fp32>(shape, real_shape);

  float threshold = 20.0;
  if (beta != 1.f) {
    tiu::fmul(t_mul, in, beta);
  }
  exp_no_overflow(t_exp, beta != 1.f ? t_mul : in, shape, real_shape);
  tiu::fadd(t_add, t_exp, 1.0);
  flog(beta != 1.f ? t_log : t_sel, t_add, shape, real_shape);
  if (beta != 1.f) {
    tiu::fdiv(t_sel, t_log, (float)(beta));
  }
  tiu::gt_select(out, in, threshold, in, t_sel);
}

template <typename DataType>
void sinh_fp32(tensor<DataType> &out, tensor<DataType> &in, dim4 *shape,
               dim4 *real_shape) {
  auto exp_res = make_tensor<DataType>(shape, real_shape);
  auto div_res = make_tensor<DataType>(shape, real_shape);
  auto sub_res = make_tensor<DataType>(shape, real_shape);
  exp_no_overflow(exp_res, in, shape, real_shape);
  tiu::fdiv(div_res, 1.f, exp_res);
  tiu::fsub(sub_res, exp_res, div_res);
  tiu::fmul(out, sub_res, 0.5f);
}

template <typename DataType>
void arcsinh_fp32(tensor<DataType> &out, tensor<DataType> &in, dim4 *shape,
                  dim4 *real_shape) {
  auto mul_res = make_tensor<DataType>(shape, real_shape);
  auto add_res = make_tensor<DataType>(shape, real_shape);
  auto sqrt_res = make_tensor<DataType>(shape, real_shape);
  tiu::fmul(mul_res, in, in);
  tiu::fadd(add_res, mul_res, 1.0f);
  tiu::fsqrt(sqrt_res, add_res);
  tiu::fadd(add_res, sqrt_res, in);
  flog(out, add_res, shape, real_shape);
}

template <typename DataType>
void cosh_fp32(tensor<DataType> &out, tensor<DataType> &in, dim4 *shape,
               dim4 *real_shape) {
  auto exp_res = make_tensor<DataType>(shape, real_shape);
  auto div_res = make_tensor<DataType>(shape, real_shape);
  auto add_res = make_tensor<DataType>(shape, real_shape);
  exp_no_overflow(exp_res, in, shape, real_shape);
  tiu::fdiv(div_res, 1.f, exp_res);
  tiu::fadd(add_res, exp_res, div_res);
  tiu::fmul(out, add_res, 0.5f);
}

template <typename DataType>
void arccosh_fp32(tensor<DataType> &out, tensor<DataType> &in, dim4 *shape,
                  dim4 *real_shape) {
  auto mul_res = make_tensor<DataType>(shape, real_shape);
  auto sub_res = make_tensor<DataType>(shape, real_shape);
  auto sqrt_res = make_tensor<DataType>(shape, real_shape);
  auto add_res = make_tensor<DataType>(shape, real_shape);
  tiu::fmul(mul_res, in, in);
  tiu::fsub(sub_res, mul_res, 1.0f);
  tiu::fsqrt(sqrt_res, sub_res);
  tiu::fadd(add_res, sqrt_res, in);
  flog(out, add_res, shape, real_shape);
}

template <typename DataType>
void tanh_fp32(tensor<DataType> &out, tensor<DataType> &in, dim4 *shape,
               dim4 *real_shape) {
  auto add_res = make_tensor<DataType>(shape, real_shape);
  auto exp_res = make_tensor<DataType>(shape, real_shape);
  auto div_res = make_tensor<DataType>(shape, real_shape);
  tiu::fadd(add_res, in, in);
  exp_no_overflow(exp_res, add_res, shape, real_shape);
  tiu::fadd(add_res, exp_res, 1.f);
  tiu::fdiv(div_res, 2.f, add_res);
  tiu::fsub(out, 1.f, div_res);
}

template <typename DataType>
void arctanh_fp32(tensor<DataType> &out, tensor<DataType> &in, dim4 *shape,
                  dim4 *real_shape) {
  auto add_res = make_tensor<DataType>(shape, real_shape);
  auto sub_res = make_tensor<DataType>(shape, real_shape);
  auto div_res = make_tensor<DataType>(shape, real_shape);
  auto log_res = make_tensor<DataType>(shape, real_shape);
  tiu::fadd(add_res, in, 1.f);
  tiu::fsub(sub_res, 1.f, in);
  tiu::fdiv(div_res, add_res, sub_res);
  flog(log_res, div_res, shape, real_shape);
  tiu::fmul(out, log_res, 0.5f);
}

template <typename DataType>
void softsign_fp32(tensor<DataType> &out, tensor<DataType> &in, dim4 *shape,
                   dim4 *real_shape) {
  auto abs_res = make_tensor<DataType>(shape, real_shape);
  auto add_res = make_tensor<DataType>(shape, real_shape);
  tiu::abs(abs_res, in);
  tiu::fadd(add_res, abs_res, 1.f);
  tiu::fdiv(out, in, add_res);
}

// only support dtype=fp16/bf16/fp32, h == 1
template <typename DataType>
void quick_pooling(tensor<DataType> &out_tensor, tensor<DataType> &in_tensor,
                   dim4 *in_block_shape, dim4 *in_real_shape, float fill,
                   int mode, float scale = 1.0) {
  int n = in_real_shape->n;
  int c = in_real_shape->c;
  int w = in_real_shape->w;
  int h = 1;
  assert(in_real_shape->h == 1);
  int eu_num = get_eu_num<DataType>();
  int opti_w = eu_num;

  int align_w = align(w, eu_num);
  int slice = align_w / eu_num;
  if (align_w > w) {
    dim4 fill_offset = {0, 0, 0, w};
    dim4 fill_shape = {n, c, 1, align_w - w};
    auto fill_tensor = in_tensor.sub_view(fill_shape, fill_offset);
    tiu::fill(fill_tensor, fill);
    // only support h == 1
  }
  dim4 in_reduce_h = {n * h, c, slice, opti_w};
  dim4 out_reduce_h = {n * h, c, 1, opti_w};
  dim4 in_reduce_w = {n, c, h, opti_w};
  dim4 out_reduce_w = {n, c, h, 1};

  dim2 kernel = {slice, 1};
  padding_t pad = {0, 0, 0, 0};
  dim2 stride = {1, 1};
  dim2 dilation = {1, 1};
  dim4 tmp_block_shape = {in_block_shape->n, in_block_shape->c, 1, opti_w};
  // tensor<DataType> tmp_tensor(tmp_block_shape);
  auto tmp_tensor = tensor<DataType>(tmp_block_shape);
  if (mode == 0) {
    tiu::fpool_max(tmp_tensor.view(out_reduce_h), in_tensor.view(in_reduce_h),
                   &kernel, &pad, &stride, &dilation);
  } else {
    tiu::fpool_avg(tmp_tensor.view(out_reduce_h), in_tensor.view(in_reduce_h),
                   &kernel, &pad, &stride, &dilation, 1.0);
  }
  kernel.h = 1;
  kernel.w = opti_w;
  if (mode == 0) {
    tiu::fpool_max(out_tensor.view(out_reduce_w), tmp_tensor.view(in_reduce_w),
                   &kernel, &pad, &stride, &dilation);

  } else {
    tiu::fpool_avg(out_tensor.view(out_reduce_w), tmp_tensor.view(in_reduce_w),
                   &kernel, &pad, &stride, &dilation, scale);
  }
}

template <typename DataType>
void fp_avg_pool_2d_count_include_padding(tensor<DataType> &dst,
                                          tensor<DataType> &src, dim4 *oshape,
                                          dim4 *ishape, dim2 *kernel,
                                          padding_t *pad, dim2 *stride,
                                          dim2 *dilation,
                                          int avg_pooling_mode) {
  int in = ishape->n;
  int ic = ishape->c;
  int ih = ishape->h;
  int iw = ishape->w;

  int on = oshape->n;
  int oc = oshape->c;
  int oh = oshape->h;
  int ow = oshape->w;

  int kh = kernel->h;
  int kw = kernel->w;

  int up_pad_h = pad->top;
  int down_pad_h = pad->bottom;
  int left_pad_w = pad->left;
  int right_pad_w = pad->right;

  int stride_h = stride->h;
  int stride_w = stride->w;

  int dilation_h = dilation->h;
  int dilation_w = dilation->w;

  int down_pad_h_ori = down_pad_h;
  int right_pad_w_ori = right_pad_w;
  int pad_bottom = (oh - 1) * stride_h + kh - ih - up_pad_h;
  down_pad_h = pad_bottom > 0 ? pad_bottom : 0;
  int pad_right = (ow - 1) * stride_w + kw - iw - left_pad_w;
  right_pad_w = pad_right > 0 ? pad_right : 0;
  if (ih + up_pad_h + down_pad_h_ori - kh >= 0 &&
      (ih + up_pad_h + down_pad_h_ori - kh) / stride_h + 1 >= oh) {
    down_pad_h_ori = down_pad_h;
  }
  if (iw + left_pad_w + right_pad_w_ori - kw >= 0 &&
      (iw + left_pad_w + right_pad_w_ori - kw) / stride_w + 1 >= ow) {
    right_pad_w_ori = right_pad_w;
  }

  padding_t new_pad = {up_pad_h, down_pad_h, left_pad_w, right_pad_w};
  tiu::fpool_avg(dst, src, kernel, &new_pad, stride, dilation, 1.0 / kw / kw);
  if (avg_pooling_mode == 1) {
    // vertical pad compensate
    for (int y = 0; y < oh; y++) {
      int h = kh;
      int h0 = y * stride_h - up_pad_h;
      int h1 = h0 + kh;
      if (h0 < 0)
        h -= -h0;
      if (h1 > ih)
        h -= (h1 - ih);
      if (h != kh) {
        float scale1 = (float)kh / h;
        dim4 shape1 = {on, oc, 1, ow};
        dim4 offset1 = {0, 0, y, 0};
        auto sub_tensor = dst.sub_view(shape1, offset1);
        tiu::fmul(sub_tensor, sub_tensor, scale1);
      }
    }
    // horizontal pad compensate
    for (int x = 0; x < ow; x++) {
      int w = kw;
      int w0 = x * stride_w - left_pad_w;
      int w1 = w0 + kw;
      if (w0 < 0)
        w -= -w0;
      if (w1 > iw)
        w -= (w1 - iw);
      if (w != kw) {
        float scale2 = (float)kw / w;
        dim4 shape2 = {on, oc, oh, 1};
        dim4 offset2 = {0, 0, 0, x};
        auto sub_tensor = dst.sub_view(shape2, offset2);
        tiu::fmul(sub_tensor, sub_tensor, scale2);
      }
    }
  } else if (avg_pooling_mode == 0) {
    // Bottom compensate
    if (pad_bottom != down_pad_h_ori) {
      float scale3 = (float)kh / (kh - pad_bottom + down_pad_h_ori);
      dim4 shape3 = {on, oc, 1, ow};
      dim4 offset3 = {0, 0, oh - 1, 0};
      auto sub_tensor = dst.sub_view(shape3, offset3);
      tiu::fmul(sub_tensor, sub_tensor, scale3);
    }
    // Right compensate
    if (pad_right != right_pad_w_ori) {
      float scale4 = (float)kw / (kw - pad_right + right_pad_w_ori);
      dim4 shape4 = {on, oc, oh, 1};
      dim4 offset4 = {0, 0, 0, ow - 1};

      auto sub_tensor = dst.sub_view(shape4, offset4);
      tiu::fmul(sub_tensor, sub_tensor, scale4);
    }
  }
}
