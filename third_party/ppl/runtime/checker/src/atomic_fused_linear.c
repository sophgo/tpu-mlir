#include "atomic_tiu.h"

void atomic_fused_linear_check(
    u32   A_addr,
    u32   B_addr,
    u32   C_addr,
    u32   R_addr,
    int   input_n,
    int   input_c,
    int   input_h,
    int   input_w,
    int   B_is_const,
    int   C_is_const,
    PREC  input_prec,
    PREC  output_prec,
    FP8_TYPE fp8_type,
    LIN_OP op,
    int saturate
) {

#ifdef USING_CMODEL
    ASSERT(A_addr % ALIGN_BYTES == 0);
    ASSERT(R_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
    if (!B_is_const) {
        ASSERT(get_npu_index(B_addr) == get_npu_index(R_addr));
        ASSERT(B_addr % get_bytesize(input_prec) == 0);
    }
    if (op == LIN_MAC && !C_is_const) {
        ASSERT(get_npu_index(R_addr) == get_npu_index(C_addr));
        ASSERT(C_addr % get_bytesize(input_prec) == 0);
    }
    ASSERT(op == LIN_MAC || op == LIN_ADD_SQR || op == LIN_SUB_SQR);
    ASSERT((input_n) < (((int)1) << 16) && ((input_n) > 0));
    ASSERT((input_c) < (((int)1) << 16) && ((input_c) > 0));
    ASSERT((input_h) < (((int)1) << 16) && ((input_h) > 0));
    ASSERT((input_w) < (((int)1) << 16) && ((input_w) > 0));
#ifdef __sg2262__
    ASSERT(input_prec == FP32 || input_prec == FP16 || input_prec == BFP16  || input_prec == FP8);
#else
    ASSERT(input_prec == FP32 || input_prec == FP16 || input_prec == BFP16);
#endif
    if (input_prec == FP32) {
        ASSERT(output_prec == FP32);
    } else if (input_prec == FP16) {
        ASSERT(output_prec == FP32 || output_prec == FP16);
    } else if (input_prec == BFP16) {
        ASSERT(output_prec == FP32 || output_prec == BFP16);
    }
#ifdef __sg2262__
ASSERT(saturate == 0 || saturate == 1);
    if (op == LIN_ADD_SQR || op == LIN_SUB_SQR) {
        ASSERT(C_is_const == 1 && C_addr == 0);
    }
#else
    if (input_prec == FP8) {
        ASSERT(fp8_type == FP8E5M2 || fp8_type == FP8E4M3);
        ASSERT(output_prec == FP32 || output_prec == FP16);
    }
#endif
#endif
}

void atomic_fused_madd_check(
    u32   A_addr,
    u32   B_addr,
    u32   C_addr,
    u32   R_addr,
    int   input_n,
    int   input_c,
    int   input_h,
    int   input_w,
    int   B_is_const,
    int   B_short_str,
    int   B_n,
    int   B_c,
    int   B_h,
    int   B_w,
    PREC  prec,
    int saturate
) {

#ifdef USING_CMODEL
#ifndef __sg2262__
    ASSERT_FS_INFO(0, "Not support");
#else
    ASSERT(A_addr % ALIGN_BYTES == 0);
    ASSERT(R_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
    ASSERT(B_is_const == 0 || B_is_const == 1);
    ASSERT(B_short_str == 0 || B_short_str == 1);
    if (!B_is_const) {
        ASSERT(B_c == input_c);
        ASSERT(get_npu_index(B_addr) == get_npu_index(R_addr));
        ASSERT(get_npu_index(C_addr) == get_npu_index(R_addr));
        if (B_short_str == 0) {
            ASSERT(B_addr % ALIGN_BYTES == 0);
            ASSERT(C_addr % ALIGN_BYTES == 0);
        } else {
            ASSERT(B_addr % get_bytesize(prec) == 0);
            ASSERT(C_addr % get_bytesize(prec) == 0);
        }
    }
    ASSERT((input_n) < (((int)1) << 16) && ((input_n) > 0));
    ASSERT((input_c) < (((int)1) << 16) && ((input_c) > 0));
    ASSERT((input_h) < (((int)1) << 16) && ((input_h) > 0));
    ASSERT((input_w) < (((int)1) << 16) && ((input_w) > 0));
    ASSERT(saturate == 0 || saturate == 1);
    ASSERT(prec == FP32 || prec == FP16 || prec == BFP16);
#endif
#endif
}
