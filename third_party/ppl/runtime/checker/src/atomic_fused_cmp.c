#include "atomic_tiu.h"

void atomic_fused_cmp_check(
    u32 tensorA_addr,
    u32 tensorB_addr,
    u32 tensorC_addr,
    u32 tensorD_addr,
    u32 tensorR0_addr,
    u32 tensorR1_addr,
    int N,
    int C,
    int H,
    int W,
    int A_is_constant,
    int B_is_constant,
    int C_is_constant,
    int D_is_constant,
    int sign,
    int side,
    int bin_w,
    int A_short_str, //normal(0:align, 3:tensor)
    int B_short_str, //normal(0:align, 3:tensor)
    PREC AB_dtype,
    PREC CD_dtype,
    PREC RES0_dtype,  // only used for srch bin
    CMP_OP op) {

#ifdef USING_CMODEL
    int AB_type_size = get_bytesize(AB_dtype);
    int A_align_num = (A_short_str == 0 ? (int)ALIGN_BYTES : AB_type_size);
    int B_align_num = (B_short_str == 0 ? (int)ALIGN_BYTES : AB_type_size);
    ASSERT((N) < (((int)1) << 16) && ((N) > 0));
    ASSERT((C) < (((int)1) << 16) && ((C) > 0));
    ASSERT((H) < (((int)1) << 16) && ((H) > 0));
    ASSERT((W) < (((int)1) << 16) && ((W) > 0));
    ASSERT((A_is_constant == 0) || (A_is_constant == 1));
    ASSERT((B_is_constant == 0) || (B_is_constant == 1));
    ASSERT((C_is_constant == 0) || (C_is_constant == 1));
    ASSERT((D_is_constant == 0) || (D_is_constant == 1));
    ASSERT((sign) == 0 || (sign) == 1);
    ASSERT((side) == 0 || (side) == 1);
#ifdef __sg2262__
    ASSERT(((op) == 22) || ((op) == 23) || ((op) == 24) || ((op) == 25) || ((op) == 26) || ((op) == 27));
#else
    ASSERT(((op) <= 27) && ((op) >= 22));
#endif

    ASSERT((A_short_str) == 0 || (A_short_str) == 3);
    ASSERT((B_short_str) == 0 || (B_short_str) == 3);
#ifdef __sg2262__
    ASSERT(AB_dtype == INT8 || AB_dtype == FP16 || AB_dtype == FP32 || AB_dtype == INT16 || AB_dtype == INT32 || AB_dtype == BFP16 || AB_dtype == FP8);
    ASSERT(CD_dtype == INT8 || CD_dtype == FP16 || CD_dtype == FP32 || CD_dtype == INT16 || CD_dtype == INT32 || CD_dtype == BFP16 || CD_dtype == FP8);
#else
    ASSERT((((int)AB_dtype) <= 7) && (((int)AB_dtype) >= 0));
    ASSERT(AB_dtype != INT4 && CD_dtype != INT4);
    ASSERT((((int)CD_dtype) <= 7) && (((int)CD_dtype) >= 0));
#endif
    if (op == CMP_SRCH_BIN) {
        ASSERT(RES0_dtype == INT8 || RES0_dtype == INT16 || RES0_dtype == INT32);
        ASSERT(B_is_constant == 0);
    }
    // check addr align according to dtype
    if (A_is_constant == 0) ASSERT(tensorA_addr % A_align_num == 0);
    if (B_is_constant == 0) ASSERT(tensorB_addr % B_align_num == 0);
    if (C_is_constant == 0) ASSERT(tensorC_addr % ALIGN_BYTES == 0);
    if (D_is_constant == 0) ASSERT(tensorD_addr % ALIGN_BYTES == 0);
    ASSERT(tensorR0_addr % ALIGN_BYTES == 0);
    if (op == CMP_GT_AND_SG || op == CMP_LT_AND_SL) {
        ASSERT(tensorR1_addr % ALIGN_BYTES == 0);
    }
#endif
}


