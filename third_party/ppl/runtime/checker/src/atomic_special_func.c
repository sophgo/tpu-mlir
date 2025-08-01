#include "atomic_tiu.h"

void atomic_sfu_check(
    u32     A_addr,
    u32     Y_addr,
    u32     Y1_addr, //for frexp mantissa
    int     input_n,
    int     input_c,
    int     input_h,
    int     input_w,
    int     n,
    SFU_OP  sfu_op,
    u32     table_start_addr,
    PREC    res0_prec,
    PREC    opd0_prec
    ) {
#ifdef USING_CMODEL
    ASSERT(input_n != 0 && input_c != 0 && input_h != 0 && input_w != 0);
    u32 A_npu_idx = get_npu_index(A_addr);
    u32 Y_npu_idx = get_npu_index(Y_addr);
    u32 table_npu_idx = get_npu_index(table_start_addr);
    ASSERT(A_npu_idx == Y_npu_idx);
#ifdef __sg2262__
    u32 Y1_npu_idx = get_npu_index(Y1_addr);
    ASSERT(sfu_op == SFU_TAYLOR_4X || sfu_op == SFU_NORM ||
        sfu_op == SFU_RSQ || sfu_op == SFU_FREXP || sfu_op == SFU_RECIPROCAL);
#endif
    switch (sfu_op) {
        case SFU_TAYLOR_4X:
#ifndef __sg2262__
        case SFU_TAYLOR:
#endif
            ASSERT(((res0_prec == FP16) && (opd0_prec == FP16)) ||
                      ((res0_prec == BFP16) && (opd0_prec == BFP16)) ||
                      ((res0_prec == FP32) && (opd0_prec == FP32)));
            ASSERT(table_npu_idx == Y_npu_idx);
            ASSERT(table_start_addr % ALIGN_BYTES == 0);
#ifdef __sg2262__
            ASSERT(n >= 3);
#else
            ASSERT(n >= 2);
#endif
            break;
        case SFU_NORM:
#ifdef __sg2262__
            if(opd0_prec == FP16)
                ASSERT(res0_prec == INT16);
            else if(opd0_prec == BFP16)
                ASSERT(res0_prec == INT16);
            else if(opd0_prec == FP32)
                ASSERT(res0_prec == INT32);
#else
            if(opd0_prec == FP16)
                ASSERT(res0_prec == INT16 || res0_prec == FP16);
            else if(opd0_prec == BFP16)
                ASSERT(res0_prec == INT16 || res0_prec == BFP16);
            else if(opd0_prec == FP32)
                ASSERT(res0_prec == INT32 || res0_prec == FP32);
#endif
            else  ASSERT(0);
            break;
        case SFU_RSQ:
            ASSERT(1 <= n && n <= 4);
            ASSERT(is_float_prec(opd0_prec) && opd0_prec == res0_prec);
            break;
#ifdef __sg2262__
        case SFU_FREXP:
            ASSERT(Y1_npu_idx == Y_npu_idx);
            ASSERT(Y1_addr % ALIGN_BYTES == 0);
            ASSERT(opd0_prec == FP16 || opd0_prec == BFP16 || opd0_prec == FP32);
            ASSERT(res0_prec == opd0_prec);
            break;
        case SFU_RECIPROCAL:
            ASSERT(opd0_prec == FP16 || opd0_prec == BFP16 || opd0_prec == FP32);
            ASSERT(res0_prec == opd0_prec);
            break;
#endif
        default : ASSERT(0);
    }
#endif
}

//exp(clamp(A-B, range_start, range_end))
void atomic_fuse_exp_check(
    u32     A_addr,
    u32     B_addr,
    u32     table_start_addr,
    u32     Y_addr,
    u32     Y1_addr,
    int     input_n,
    int     input_c,
    int     input_h,
    int     input_w,
    int     iter_w,
    int     B_is_const,
    int     B_short_str,
    u32     *B_stride,
    PREC    prec,
    u32     range_start, //16bit[fp16/bf16] or 32bit[fp16/bf16]
    u32     range_end, //16bit[fp16/bf16] or 32bit[fp16/bf16]
    int     use_default_tcgr) {

#ifdef USING_CMODEL
#ifndef __sg2262__
    ASSERT_FS_INFO(0, "Not support");
#else
    ASSERT(input_n != 0 && input_c != 0 && input_h != 0 && input_w != 0);
    int A_npu_idx = get_npu_index(A_addr);
    int Y_npu_idx = get_npu_index(Y_addr);
    int Y1_npu_idx = get_npu_index(Y1_addr);
    int table_npu_idx = get_npu_index(table_start_addr);
    ASSERT(A_npu_idx == Y_npu_idx && A_npu_idx == table_npu_idx && A_npu_idx == Y1_npu_idx);
    ASSERT(B_is_const == 0 || B_is_const == 1);
    ASSERT(B_short_str == 0 || B_short_str == 3);
    if (!B_is_const) {
        ASSERT(get_npu_index(B_addr) == A_npu_idx);
        if (B_short_str == 0) {
            ASSERT(B_addr % ALIGN_BYTES == 0);
        } else {
            ASSERT(B_addr % get_bytesize(prec) == 0);
        }
    }
    ASSERT(prec == FP32 || prec == FP16 || prec == BFP16);
#endif
#endif
}
