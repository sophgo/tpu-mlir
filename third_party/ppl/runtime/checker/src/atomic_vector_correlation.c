#include "atomic_tiu.h"

void atomic_vector_correlation_check(
        u32 A_addr,
        u32 B_addr,
        u32 R_addr,
        int A_len,
        int B_len,
        int A_w,
        int B_w,
        AR_OP op,
        PREC A_prec,
        PREC B_prec,
        PREC R_prec,
        ROUND_MODE round_mode,
        u32 select_val,
        int A_sign,
        int B_sign,
        int R_sign) {

#ifdef USING_CMODEL
    int A_c = (A_len + A_w - 1) / A_w;
    int B_c = (B_len + B_w - 1) / B_w;
    int A_w_last = A_len - (A_c - 1) * A_w;
    int opd2_n_str = (op == AR_DIV) ? 2 : 0;     // iter 3 for DIV OP
    ASSERT(A_addr % ALIGN_BYTES == 0);
    ASSERT(B_addr % ALIGN_BYTES == 0);
    ASSERT(R_addr % ALIGN_BYTES == 0);
    ASSERT(B_addr / LOCAL_MEM_SIZE == R_addr / LOCAL_MEM_SIZE);
    ASSERT(A_w > 0 && A_w < 65536);
    ASSERT(B_w > 0 && B_w < 65536);
    ASSERT(A_c > 0 && A_c < 65536);
    ASSERT(B_c > 0 && B_c < 65536);
    ASSERT(A_sign == 0 || A_sign == 1);
    ASSERT(B_sign == 0 || B_sign == 1);
#ifdef __sg2262__
    ASSERT(op == AR_MAX || op == AR_MIN);
    ASSERT(A_sign == B_sign);
    ASSERT(A_prec == B_prec && A_prec == R_prec);
#else
    ASSERT(op == AR_MUL || op == AR_ADD || op == AR_SUB ||
           op == AR_MAX || op == AR_MIN || op == AR_AND ||
           op == AR_OR  || op == AR_XOR || op == AR_SG  ||
           op == AR_SE  || op == AR_DIV || op == AR_SL  ||
           op == AR_ADD_SATU || op == AR_SUB_SATU ||
           op == AR_MUL_SATU);
    ASSERT(R_sign == 0 || R_sign == 1);
#endif
#endif
}
