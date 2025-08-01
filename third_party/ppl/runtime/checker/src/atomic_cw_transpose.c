#include "atomic_tiu.h"


void atomic_cw_transpose_check(
    u32   A_addr,
    u32   Y_addr,
    int   input_n,
    int   input_c,
    int   input_h,
    int   input_w,
    PREC dtype,
    TRAN_OP op
) {
#ifdef USING_CMODEL
    int A_npu_idx = get_npu_index(A_addr);
    int Y_npu_idx = get_npu_index(Y_addr);
    ASSERT(A_addr % ALIGN_BYTES == 0);
    ASSERT(Y_addr % ALIGN_BYTES == 0);
    ASSERT((op == TRAN_C_W_TRANSPOSE  && A_npu_idx == 0) || (op == TRAN_W_C_TRANSPOSE && Y_npu_idx == 0));
    ASSERT((input_n) < (((int)1) << 16) && ((input_n) > 0));
    ASSERT((input_c) < (((int)1) << 16) && ((input_c) > 0));
    ASSERT((input_h) < (((int)1) << 16) && ((input_h) > 0));
    ASSERT((input_w) < (((int)1) << 16) && ((input_w) > 0));
  #ifndef __sg2262__
    ASSERT(op == TRAN_C_W_TRANSPOSE || op == TRAN_W_C_TRANSPOSE);
  #endif
    ASSERT(dtype == FP32 || dtype == FP16 || dtype == BFP16 || dtype == INT8 || dtype == FP8 || dtype == INT16 || dtype == INT32);
    ASSERT(op == TRAN_C_W_TRANSPOSE || op == TRAN_W_C_TRANSPOSE);
    ASSERT(A_addr != Y_addr);
#endif
}

