#ifndef ATOMIC_TENSOR_ARITHMETIC
#define ATOMIC_TENSOR_ARITHMETIC
#include "checker_internel.h"
#ifdef __cplusplus
extern "C" {
#endif

// use for two_opd tensor_arithmetic
// include two_opd FP32/FP16/BFP16 AR
// can alse use for some two_opd fixed_point AR
void atomic_tensor_arithmetic_check(
    unsigned int A_addr,
    unsigned int B_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_B_stride,
    int * tensor_R_stride,
    int A_is_const,
    int B_is_const,
    int * Short_str, //len = 3, opd0, opd1, res
    int * Sign, //len = 3, opd0, opd1, opd2
    int sym_saturate,
    PREC * Prec, //len = 3, opd0, opd1, res
    AR_OP op);

void atomic_tensor_arithmetic_div_check(
    unsigned int A_addr,
    unsigned int B_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_B_stride,
    int * tensor_R_stride,
    int A_is_const,
    int B_is_const,
    int * Short_str, //len = 3, opd0, opd1, res
    PREC prec, // fp32, fp16, bf16
    int iter,
    int saturate);

void atomic_tensor_arithmetic_div_txp_check(
    unsigned int A_addr,
    unsigned int B_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_B_stride,
    int * tensor_R_stride,
    int A_is_const,
    int B_is_const,
    int * Short_str,
    int sign,   // distinguish between FP8E4M3 and FP8E5M2
    PREC prec,
    int iter,
    int saturate);

void atomic_tensor_arithmetic_div2_check(
    unsigned int A_addr,
    unsigned int B_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_B_stride,
    int * tensor_R_stride,
    int A_is_const,
    int B_is_const,
    int * Short_str,
    PREC prec,
    int saturate);

// use for ternary ternary tensor_arithmetic
void atomic_tensor_arithmetic_ternary_check(
    unsigned int A_addr,
    unsigned int B_addr,
    unsigned int C_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_B_stride,
    int * tensor_R_stride,
    int A_is_const,
    int B_is_const,
    int C_is_const,
    int * Short_str, // len = 3, opd0, opd1, res
    int * Sign, // len = 3, opd0, opd1, opd2
    int sym_saturate,
    PREC * Prec, // len = 4, opd0, opd1, opd2, res
    AR_OP op,
    ROUND_MODE round);

// use for SE/SG/SL
void atomic_tensor_arithmetic_select_check(
    unsigned int A_addr,
    unsigned int B_addr,
    unsigned int C_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_B_stride,
    int * tensor_R_stride,
    int A_is_const,
    int B_is_const,
    int * Short_str, //len = 3, opd0, opd1, res
    int * Sign, // len = 2, opd0/opd1, opd2/res
    PREC * Prec, //len = 2, opd0/opd1, opd2/res
    AR_OP op);

// use for two_opds with round(shift\mulDhr)
void atomic_tensor_arithmetic_with_round_check(
    unsigned int A_addr,
    unsigned int B_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_B_stride,
    int * tensor_R_stride,
    int A_is_const,
    int B_is_const,
    int * Short_str, //len = 3, opd0, opd1, res
    int * Sign, // len = 2
    PREC * Prec, //len = 3, opd0, opd1, res
    AR_OP op,
    ROUND_MODE round);

// use for dtype convert
void atomic_tensor_arithmetic_dtype_convert_check(
    unsigned int A_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_R_stride,
    int A_is_const,
    int * Short_str,
    int * Sign,
    int sym_saturate,
    PREC * Prec,
    ROUND_MODE round);

// copy/copy_mb/abs/not
//void atomic_tensor_arithmetic_copy_like_check(
void atomic_tensor_arithmetic_single_opd_check(
    unsigned int A_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_R_stride,
    int A_is_const,
    int * Short_str,
    int sign, // for abs
    PREC Prec,
    AR_OP op);

// use for get_first_zero/get_first_one
void atomic_tensor_arithmetic_get_first_check(
    unsigned int A_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_R_stride,
    int A_is_const,
    int * Short_str,
    PREC * Prec, // A && R can have different dtype
    int Sign,
    AR_OP op);

// use for fuse_mul_cast
void atomic_tensor_arithmetic_mul_cast_check(
    unsigned int A_addr,
    unsigned int B_addr,
    unsigned int R_addr,
    int N,
    int C,
    int H,
    int W,
    int * tensor_A_stride,
    int * tensor_B_stride,
    int * tensor_R_stride,
    int A_is_const,
    int B_is_const,
    int * Short_str, // num = 3, opd0, opd1, res0
    int *Sign, // num = 3, opd0, opd1, res0
    int saturate,
    PREC * Prec, // num = 3, opd0, opd1, res0
    ROUND_MODE round);

#ifdef __cplusplus
}
#endif

#endif  /* ATOMIC_TENSOR_ARITHMETIC */
