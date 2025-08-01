#include "atomic_tensor_arithmetic.h"


#define CHECK_AR_STRIDE(p_stride) \
      ASSERT((p_stride[0] < (((int)1) << 18)) && (p_stride[0] >= 0)); \
      ASSERT((p_stride[1] < (((int)1) << 18)) && (p_stride[1] >= 0)); \
      ASSERT((p_stride[2] < (((int)1) << 18)) && (p_stride[2] >= 0)); \
      ASSERT((p_stride[3] < (((int)1) << 18)) && (p_stride[3] >= 0)); \

#define CHECK_AR_ZERO_DST_STRIDE(dst_stride, dst_shape) \
       if (dst_stride[0] == 0) ASSERT(dst_shape[0] == 1); \
       if (dst_stride[1] == 0) ASSERT(dst_shape[1] == 1); \
       if (dst_stride[2] == 0) ASSERT(dst_shape[2] == 1); \
       if (dst_stride[3] == 0) ASSERT(dst_shape[3] == 1); \

// NOTICE:
// /*****************************************/
// Short_str 0/1 can not support broadcast b
// if b need to broadcast, make Short_str = 3
// and set correspond dimension stride 0
// /*****************************************/

inline static int get_bit_width(PREC precision) {
  int bit_width = 8;
  if (precision == INT4) {
    bit_width = 4;
  } else if (precision == INT8 || precision == FP8) {
    bit_width = 8;
  } else if (precision == INT16 || precision == FP16 || precision == BFP16) {
    bit_width = 16;
  } else if (precision == INT32 || precision == FP32 || precision == TF32) {
    bit_width = 32;
  } else if (precision == FP20) {
    bit_width = 20;
  }
#ifdef __sg2262__
  else if (precision == FP4) {
    bit_width = 4;
  }
#endif
  else {
    ASSERT(0 && "invalid precision");
  }
  return bit_width;
}

static inline int ceiling_func(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}
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
    int * Short_str,
    int * Sign,
    int sym_saturate,
    PREC * Prec,
    AR_OP op
) {
#ifdef USING_CMODEL
#ifdef __sg2262__
    bool A_need_stride = Short_str[0] != 0;
    bool B_need_stride = Short_str[1] != 0;
    bool R_need_stride = Short_str[2] != 0;
#else
    bool A_need_stride = ((Short_str[0] != 0) && (Short_str[0] != 1));
    bool B_need_stride = ((Short_str[1] != 0) && (Short_str[1] != 1));
    bool R_need_stride = ((Short_str[2] != 0) && (Short_str[2] != 1));
#endif
    if (!A_is_const) {
       ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
      if (Short_str[0] == 0)
        ASSERT(A_addr % ALIGN_BYTES == 0);
      else if (Short_str[0] == 3)
        ASSERT(A_addr % get_bytesize(Prec[0]) == 0);
    }
    if (!B_is_const) {
       ASSERT(get_npu_index(B_addr) == get_npu_index(R_addr));
      if (Short_str[1] == 0)
        ASSERT(B_addr % ALIGN_BYTES == 0);
      else if (Short_str[1] == 3)
        ASSERT(B_addr % get_bytesize(Prec[1]) == 0);
    }
    if (Short_str[2] == 0)
      ASSERT(R_addr % ALIGN_BYTES == 0);
    else if (Short_str[2] == 3)
      ASSERT(R_addr % get_bytesize(Prec[2]) == 0);
    ASSERT(N < (((int)1) << 16) && (N > 0));
    ASSERT(C < (((int)1) << 16) && (C > 0));
    ASSERT(H < (((int)1) << 16) && (H > 0));
    ASSERT(W < (((int)1) << 16) && (W > 0));
    ASSERT((A_is_const == 0) || (A_is_const == 1));
    ASSERT((B_is_const == 0) || (B_is_const == 1));
    ASSERT((Short_str[0] == 0) || (Short_str[0] == 3));
    ASSERT((Short_str[1] == 0) || (Short_str[1] == 3));
    ASSERT((Short_str[2] == 0) || (Short_str[2] == 3));
    ASSERT((Sign[0] == 0) || (Sign[0] == 1));
    ASSERT((Sign[1] == 0) || (Sign[1] == 1));
#ifndef __sg2262__
    ASSERT((Sign[2] == 0) || (Sign[2] == 1));
#endif
    ASSERT((op != AR_NOT) && (op != AR_COPY) && (op != AR_DATA_CONVERT));
    ASSERT((op != AR_SG) && (op != AR_SE) && (op != AR_SL) && (op != AR_ABS));
    ASSERT((op != AR_GET_FIRST_ZERO) && (op != AR_GET_FIRST_ONE));
    ASSERT(sym_saturate == 0 || sym_saturate == 1);
#ifdef __sg2262__
    if (op == AR_MIN || op == AR_MAX) {
       ASSERT(Sign[0] == Sign[1] && Prec[0] == Prec[1] && Prec[0] == Prec[2]);
    }
    if (op == AR_ADD || op == AR_SUB || op == AR_MUL) {
       if (is_float_prec(Prec[0]) && Prec[0] != FP8 && Prec[1] != FP8 && Prec[0] != FP4) {
           ASSERT(Prec[0] == Prec[1] && Prec[0] == Prec[2]);
       } else if (Prec[0] == FP8) {
           if (op == AR_ADD || op == AR_MUL) {
              if (Prec[1] == FP8) {
                  ASSERT(Prec[2] == FP8 || Prec[2] == FP16 || Prec[2] == BFP16 || Prec[2] == FP32);
              } else if (Prec[1] == FP16) {
                  ASSERT(Prec[2] == FP16 || Prec[2] == FP32);
              } else if (Prec[1] == BFP16) {
                  ASSERT(Prec[2] == BFP16 || Prec[2] == FP32);
              } else if (Prec[1] == FP32) {
                  ASSERT(Prec[2] == FP32);
              } else {
              ASSERT(0);
              }
           } else {
              ASSERT(Prec[1] == FP8);
              ASSERT(Prec[2] == FP8 || Prec[2] == FP16 || Prec[2] == FP32);
           }
       } else if (Prec[0] == FP4) {
              ASSERT(op == AR_MUL);
              ASSERT(Prec[1] == FP8 || Prec[1] == FP16 || Prec[1] == BFP16);
              if (Prec[1] == FP8) {
                     ASSERT(Prec[2] == FP8 || Prec[2] == FP16 || Prec[2] == BFP16 || Prec[2] == FP32);
              } else if (Prec[1] == FP16) {
                     ASSERT(Prec[2] == FP16 || Prec[2] == FP32);
              } else {
                     ASSERT(Prec[2] == BFP16 || Prec[2] == FP32);
              }
       } else {
           ASSERT(is_fixed_prec(Prec[0]) && is_fixed_prec(Prec[1]) && is_fixed_prec(Prec[2]));
       }
    }
    if (op == AR_MAC && is_float_prec(Prec[0])) {
       ASSERT(Prec[0] != FP8 && (Prec[0] == Prec[1] && Prec[0] == Prec[2]));
    }
    if (A_need_stride) {
       ASSERT(Prec[0] != INT4 && Prec[0] != FP4);
       CHECK_AR_STRIDE(tensor_A_stride);
    }
    if (B_need_stride) {
       ASSERT(Prec[1] != INT4 && Prec[1] != FP4);
       CHECK_AR_STRIDE(tensor_B_stride);
    }
    if (R_need_stride) {
       ASSERT(Prec[0] != INT4 && Prec[1] != INT4 && Prec[2] != INT4 && Prec[0] != FP4 && Prec[1] != FP4 && Prec[2] != FP4);
       int dst_shape[4] = {N, C, H, W};
       CHECK_AR_ZERO_DST_STRIDE(tensor_R_stride, dst_shape);
    }
#else
    if ((op == AR_MIN || op == AR_MAX) && Prec[0] != FP8) {
       ASSERT(Sign[0] == Sign[1] && Prec[0] == Prec[1] && Prec[0] == Prec[2]);
    } else if ((op == AR_MIN || op == AR_MAX) && Prec[0] == FP8){
       ASSERT(Prec[0] == Prec[1] && Prec[0] == Prec[2]);
    }
    if (op == AR_ADD || op == AR_SUB || op == AR_MUL) {
       if (is_float_prec(Prec[0]) && Prec[0] != FP8 && Prec[1] != FP8) {
           ASSERT(Prec[0] == Prec[1] && Prec[0] == Prec[2]);
       } else if (Prec[0] == FP8 && Prec[1] == FP8){
           ASSERT(Prec[2] == FP8 || Prec[2] == FP16 || Prec[2] == FP32);
       } else if (Prec[0] == FP8 && Prec[1] == FP16 && op != AR_SUB){
           ASSERT(Prec[2] == FP16 || Prec[2] == FP32);
       } else if (Prec[0] == FP8 && Prec[1] == FP32 && op != AR_SUB){
           ASSERT(Prec[2] == FP32);
       } else if (Prec[0] == FP16 && Prec[1] == FP8 && op == AR_SUB){
           ASSERT(Prec[2] == FP16 || Prec[2] == FP32);
       } else if (Prec[0] == FP32 && Prec[1] == FP8 && op == AR_SUB){
           ASSERT(Prec[2] == FP32);
       } else if (is_fixed_prec(Prec[0])){
           ASSERT(is_fixed_prec(Prec[1]) && is_fixed_prec(Prec[2]));
       }
    }
    if (A_need_stride) {
       ASSERT(Prec[0] != INT4);
       CHECK_AR_STRIDE(tensor_A_stride);
    }
    if (B_need_stride) {
       ASSERT(Prec[1] != INT4);
       CHECK_AR_STRIDE(tensor_B_stride);
    }
    if (R_need_stride) {
       ASSERT(Prec[0] != INT4 && Prec[1] != INT4 && Prec[2] != INT4);
       CHECK_AR_STRIDE(tensor_R_stride);
       // int dst_shape[4] = {N, C, H, W};
       // CHECK_AR_ZERO_DST_STRIDE(tensor_R_stride, dst_shape);
    }
#endif
#endif
}

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
    int * Short_str,
    PREC prec,
    int iter,
    int saturate
) {

#ifdef USING_CMODEL
#ifdef __sg2262__
    ASSERT(saturate == 0 || saturate == 1);
    bool A_need_stride = Short_str[0] != 0;
    bool B_need_stride = Short_str[1] != 0;
    bool R_need_stride = Short_str[2] != 0;
#else
    bool A_need_stride = ((Short_str[0] != 0) && (Short_str[0] != 1));
    bool B_need_stride = ((Short_str[1] != 0) && (Short_str[1] != 1));
    bool R_need_stride = ((Short_str[2] != 0) && (Short_str[2] != 1));
#endif
    ASSERT(is_float_prec(prec));
    if (!A_is_const) {
       ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
      if (Short_str[0] == 0)
        ASSERT(A_addr % ALIGN_BYTES == 0);
      else if (Short_str[0] == 3)
        ASSERT(A_addr % get_bytesize(prec) == 0);
    }
    if (!B_is_const) {
       ASSERT(get_npu_index(B_addr) == get_npu_index(R_addr));
      if (Short_str[1] == 0)
        ASSERT(B_addr % ALIGN_BYTES == 0);
      else if (Short_str[1] == 3)
        ASSERT(B_addr % get_bytesize(prec) == 0);
    }
    if (Short_str[2] == 0)
      ASSERT(R_addr % ALIGN_BYTES == 0);
    else if (Short_str[2] == 3)
      ASSERT(R_addr % get_bytesize(prec) == 0);
    ASSERT(N < (((int)1) << 16) && (N > 0));
    ASSERT(C < (((int)1) << 16) && (C > 0));
    ASSERT(H < (((int)1) << 16) && (H > 0));
    ASSERT(W < (((int)1) << 16) && (W > 0));
    ASSERT((A_is_const == 0) || (A_is_const == 1));
    ASSERT((B_is_const == 0) || (B_is_const == 1));
    ASSERT((Short_str[0] == 0) || (Short_str[0] == 3));
    ASSERT((Short_str[1] == 0) || (Short_str[1] == 3));
    ASSERT((Short_str[2] == 0) || (Short_str[2] == 3));
#ifndef __sg2262__
    ASSERT(prec != INT4);
#endif
    if (A_need_stride) {
       CHECK_AR_STRIDE(tensor_A_stride);
    }
    if (B_need_stride) {
       CHECK_AR_STRIDE(tensor_B_stride);
    }
    if (R_need_stride) {
#ifndef __sg2262__
       CHECK_AR_STRIDE(tensor_R_stride);
#endif
       int dst_shape[4] = {N, C, H, W};
       CHECK_AR_ZERO_DST_STRIDE(tensor_R_stride, dst_shape);
    }
    ASSERT(iter >= 0 && iter <= 4);
#endif
}

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
  int sign,
  PREC prec,
  int iter,
  int saturate
) {

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  bool A_need_stride = ((Short_str[0] != 0) && (Short_str[0] != 1));
  bool B_need_stride = ((Short_str[1] != 0) && (Short_str[1] != 1));
  bool R_need_stride = ((Short_str[2] != 0) && (Short_str[2] != 1));
  ASSERT(is_float_prec(prec));
  if (!A_is_const) {
     ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
    if (Short_str[0] == 0)
      ASSERT(A_addr % ALIGN_BYTES == 0);
    else if (Short_str[0] == 3)
      ASSERT(A_addr % get_bytesize(prec) == 0);
  }
  if (!B_is_const) {
     ASSERT(get_npu_index(B_addr) == get_npu_index(R_addr));
    if (Short_str[1] == 0)
      ASSERT(B_addr % ALIGN_BYTES == 0);
    else if (Short_str[1] == 3)
      ASSERT(B_addr % get_bytesize(prec) == 0);
  }
  if (Short_str[2] == 0)
    ASSERT(R_addr % ALIGN_BYTES == 0);
  else if (Short_str[2] == 3)
    ASSERT(R_addr % get_bytesize(prec) == 0);
  ASSERT(N < (((int)1) << 16) && (N > 0));
  ASSERT(C < (((int)1) << 16) && (C > 0));
  ASSERT(H < (((int)1) << 16) && (H > 0));
  ASSERT(W < (((int)1) << 16) && (W > 0));
  ASSERT((A_is_const == 0) || (A_is_const == 1));
  ASSERT((B_is_const == 0) || (B_is_const == 1));
  ASSERT((Short_str[0] == 0) || (Short_str[0] == 3));
  ASSERT((Short_str[1] == 0) || (Short_str[1] == 3));
  ASSERT((Short_str[2] == 0) || (Short_str[2] == 3));
  ASSERT(prec != INT4);
  if (A_need_stride) {
     CHECK_AR_STRIDE(tensor_A_stride);
  }
  if (B_need_stride) {
     CHECK_AR_STRIDE(tensor_B_stride);
  }
  if (R_need_stride) {
     CHECK_AR_STRIDE(tensor_R_stride);
     int dst_shape[4] = {N, C, H, W};
     CHECK_AR_ZERO_DST_STRIDE(tensor_R_stride, dst_shape);
  }
  ASSERT(iter >= 0 && iter <= 4);
#endif
#endif
}

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
    int saturate
) {
#ifdef USING_CMODEL
#ifndef __sg2262__
    ASSERT_FS_INFO(0, "Not support");
#else
    bool A_need_stride = Short_str[0] != 0;
    bool B_need_stride = Short_str[1] != 0;
    bool R_need_stride = Short_str[2] != 0;
    ASSERT(saturate == 0 || saturate == 1);
    ASSERT(is_float_prec(prec));
    if (!A_is_const) {
       ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
      if (Short_str[0] == 0)
        ASSERT(A_addr % ALIGN_BYTES == 0);
      else if (Short_str[0] == 3)
        ASSERT(A_addr % get_bytesize(prec) == 0);
    }
    if (!B_is_const) {
       ASSERT(get_npu_index(B_addr) == get_npu_index(R_addr));
      if (Short_str[1] == 0)
        ASSERT(B_addr % ALIGN_BYTES == 0);
      else if (Short_str[1] == 3)
        ASSERT(B_addr % get_bytesize(prec) == 0);
    }
    if (Short_str[2] == 0)
      ASSERT(R_addr % ALIGN_BYTES == 0);
    else if (Short_str[2] == 3)
      ASSERT(R_addr % get_bytesize(prec) == 0);
    ASSERT(N < (((int)1) << 16) && (N > 0));
    ASSERT(C < (((int)1) << 16) && (C > 0));
    ASSERT(H < (((int)1) << 16) && (H > 0));
    ASSERT(W < (((int)1) << 16) && (W > 0));
    ASSERT((A_is_const == 0) || (A_is_const == 1));
    ASSERT((B_is_const == 0) || (B_is_const == 1));
    ASSERT((Short_str[0] == 0) || (Short_str[0] == 3));
    ASSERT((Short_str[1] == 0) || (Short_str[1] == 3));
    ASSERT((Short_str[2] == 0) || (Short_str[2] == 3));
    if (A_need_stride) {
       CHECK_AR_STRIDE(tensor_A_stride);
    }
    if (B_need_stride) {
       CHECK_AR_STRIDE(tensor_B_stride);
    }
    if (R_need_stride) {
       int dst_shape[4] = {N, C, H, W};
       CHECK_AR_ZERO_DST_STRIDE(tensor_R_stride, dst_shape);
    }
    ASSERT(saturate == 0 || saturate == 1);
#endif
#endif
}

// // this function used for ternary tensor_arithmetic
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
    int * Sign, // len = 2, opd0, opd1
    int sym_saturate,
    PREC * Prec, // len = 4, opd0, opd1, opd2, res
    AR_OP op,
    ROUND_MODE round) {

#ifdef USING_CMODEL
#ifdef __sg2262__
    bool A_need_stride = Short_str[0] != 0;
    bool B_need_stride = Short_str[1] != 0;
    bool R_need_stride = Short_str[2] != 0;
#else
    bool A_need_stride = ((Short_str[0] != 0) && (Short_str[0] != 1));
    bool B_need_stride = ((Short_str[1] != 0) && (Short_str[1] != 1));
    bool R_need_stride = ((Short_str[2] != 0) && (Short_str[2] != 1));
#endif
    if (!A_is_const) {
      ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
      if (Short_str[0] == 0)
        ASSERT(A_addr % ALIGN_BYTES == 0);
      else if (Short_str[0] == 3)
        ASSERT(A_addr % get_bytesize(Prec[0]) == 0);
    }
    if (!B_is_const) {
      ASSERT(get_npu_index(B_addr) == get_npu_index(R_addr));
      if (Short_str[1] == 0)
        ASSERT(B_addr % ALIGN_BYTES == 0);
      else if (Short_str[1] == 3)
        ASSERT(B_addr % get_bytesize(Prec[1]) == 0);
    }
    if (!C_is_const) {
      ASSERT(get_npu_index(C_addr) == get_npu_index(R_addr));
      ASSERT(C_addr % get_bytesize(Prec[2]) == 0);
    }
    if (Short_str[2] == 0)
      ASSERT(R_addr % ALIGN_BYTES == 0);
    else if (Short_str[2] == 3)
      ASSERT(R_addr % get_bytesize(Prec[3]) == 0);
    ASSERT(N < (((int)1) << 16) && (N > 0));
    ASSERT(C < (((int)1) << 16) && (C > 0));
    ASSERT(H < (((int)1) << 16) && (H > 0));
    ASSERT(W < (((int)1) << 16) && (W > 0));
    ASSERT((A_is_const == 0) || (A_is_const == 1));
    ASSERT((B_is_const == 0) || (B_is_const == 1));
    ASSERT((C_is_const == 0) || (C_is_const == 1));
    ASSERT((Short_str[0] == 0) || (Short_str[0] == 3));
    ASSERT((Short_str[1] == 0) || (Short_str[1] == 3));
    ASSERT((Short_str[2] == 0) || (Short_str[2] == 3));
    ASSERT((Sign[0] == 0) || (Sign[0] == 1));
    ASSERT((Sign[1] == 0) || (Sign[1] == 1));
#ifndef __sg2262__
    ASSERT((Sign[2] == 0) || (Sign[2] == 1));
    ASSERT(Prec[0] != INT4 && Prec[1] != INT4 && Prec[2] != INT4);
#endif
    ASSERT(sym_saturate == 0 || sym_saturate == 1);
    if (A_need_stride) {
       ASSERT(Prec[0] != INT4);
       CHECK_AR_STRIDE(tensor_A_stride);
    }
    if (B_need_stride) {
       ASSERT(Prec[1] != INT4);
       CHECK_AR_STRIDE(tensor_B_stride);
    }
    if (R_need_stride) {
       ASSERT(Prec[0] != INT4 && Prec[1] != INT4 && Prec[3] != INT4);
#ifndef __sg2262__
       CHECK_AR_STRIDE(tensor_R_stride);
#endif
       int dst_shape[4] = {N, C, H, W};
       CHECK_AR_ZERO_DST_STRIDE(tensor_R_stride, dst_shape);
    }
#ifdef __sg2262__
    ASSERT(op == AR_MAC || op == AR_ADD_SATU ||
           op == AR_SUB_SATU || op == AR_MUL_SATU ||
           op == AR_ADD || op == AR_SUB ||
           op == AR_MUL || op == AR_CLAMP);
    if (op == AR_MAC) {
       ASSERT((Prec[0] == INT8 && Prec[1] == INT8 && Prec[3] == INT16));
       ASSERT(Prec[2] == INT16 && C_is_const);
    } else if (op == AR_CLAMP) {
       ASSERT(Prec[0] != INT4 && Prec[0] != FP4);
       ASSERT(Prec[0] == Prec[1] && Prec[1] == Prec[2] && Prec[2] == Prec[3]);
       ASSERT(Sign[0] == Sign[1]);
       ASSERT(A_is_const == 0 && B_is_const == 1 && C_is_const == 1);
    } else {
       ASSERT(Prec[2] == INT8);
    }
#else
    ASSERT(op == AR_MAC || op == AR_ADD_SATU ||
           op == AR_SUB_SATU || op == AR_MUL_SATU ||
           op == AR_ADD || op == AR_SUB ||
           op == AR_MUL);
    if (op == AR_MAC) {
       ASSERT((Prec[0] == INT8 && Prec[1] == INT8 && Prec[3] == INT16) ||
              (Prec[0] == INT4 && Prec[1] == INT4 && Prec[3] == INT8));
       ASSERT(Prec[2] == INT16 && C_is_const);
    } else if (Prec[0] == INT4 || Prec[1] == INT4) {
       ASSERT(Prec[2] == INT8);
    }
#endif
#endif
}

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
    int * Sign, // len = 2, opd0, opd1
    PREC * Prec, //len = 2, opd0/opd1, opd2/res
    AR_OP op){

      //TENSOR_ARITHMETIC_GET_CYCLE(N, C, H, W, tensor_R_addr, op, pid_node);
#ifdef USING_CMODEL
#ifdef __sg2262__
    bool A_need_stride = Short_str[0] != 0;
    bool B_need_stride = Short_str[1] != 0;
    bool R_need_stride = Short_str[2] != 0;
#else
    bool A_need_stride = ((Short_str[0] != 0) && (Short_str[0] != 1));
    bool B_need_stride = ((Short_str[1] != 0) && (Short_str[1] != 1));
    bool R_need_stride = ((Short_str[2] != 0) && (Short_str[2] != 1));
#endif

    if (!A_is_const) {
      ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
      if (Short_str[0] == 0)
        ASSERT(A_addr % ALIGN_BYTES == 0);
      else if (Short_str[0] == 3)
        ASSERT(A_addr % get_bytesize(Prec[0]) == 0);
    }
    if (!B_is_const) {
      ASSERT(get_npu_index(B_addr) == get_npu_index(R_addr));
      if (Short_str[1] == 0)
        ASSERT(B_addr % ALIGN_BYTES == 0);
      else if (Short_str[1] == 3)
        ASSERT(B_addr % get_bytesize(Prec[0]) == 0);
    }
    if (Short_str[2] == 0)
      ASSERT(R_addr % ALIGN_BYTES == 0);
    else if (Short_str[2] == 3)
      ASSERT(R_addr % get_bytesize(Prec[1]) == 0);
    ASSERT(N < (((int)1) << 16) && (N > 0));
    ASSERT(C < (((int)1) << 16) && (C > 0));
    ASSERT(H < (((int)1) << 16) && (H > 0));
    ASSERT(W < (((int)1) << 16) && (W > 0));
    ASSERT((A_is_const == 0) || (A_is_const == 1));
    ASSERT((B_is_const == 0) || (B_is_const == 1));
    ASSERT((Short_str[0] == 0) || (Short_str[0] == 3));
    ASSERT((Short_str[1] == 0) || (Short_str[1] == 3));
    ASSERT((Short_str[2] == 0) || (Short_str[2] == 3));
    ASSERT((Sign[0] == 0) || (Sign[0] == 1));
    ASSERT((Sign[1] == 0) || (Sign[1] == 1));
#ifdef __sg2262__
    ASSERT(Prec[0] != FP4 && Prec[1] != FP4 && Prec[2] != FP4);
#else
    ASSERT((Sign[2] == 0) || (Sign[2] == 1));
#endif
    ASSERT(Prec[0] != INT4 && Prec[1] != INT4 && Prec[2] != INT4);
    ASSERT((op == AR_SG) || (op == AR_SE) || (op == AR_SL));
    if (A_need_stride) {
       ASSERT(Prec[0] != INT4);
       CHECK_AR_STRIDE(tensor_A_stride);
    }
    if (B_need_stride) {
       ASSERT(Prec[0] != INT4);
       CHECK_AR_STRIDE(tensor_B_stride);
    }
    if (R_need_stride) {
       ASSERT(Prec[0] != INT4 && Prec[1] != INT4);
#ifndef __sg2262__
       CHECK_AR_STRIDE(tensor_R_stride);
#endif
       int dst_shape[4] = {N, C, H, W};
       CHECK_AR_ZERO_DST_STRIDE(tensor_R_stride, dst_shape);
    }
//     ASSERT(get_bit_width(Prec[1]) <= get_bit_width(Prec[0]));
#endif
}

// use for two_opds with round ops
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
    int * Sign,
    PREC * Prec, //len = 3, opd0, opd1, res
    AR_OP op,
    ROUND_MODE round){

#ifdef USING_CMODEL
#ifdef __sg2262__
    bool A_need_stride = Short_str[0] != 0;
    bool B_need_stride = Short_str[1] != 0;
    bool R_need_stride = Short_str[2] != 0;
#else
    bool A_need_stride = ((Short_str[0] != 0) && (Short_str[0] != 1));
    bool B_need_stride = ((Short_str[1] != 0) && (Short_str[1] != 1));
    bool R_need_stride = ((Short_str[2] != 0) && (Short_str[2] != 1));
#endif
    if (!A_is_const) {
      ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
      if (Short_str[0] == 0)
        ASSERT(A_addr % ALIGN_BYTES == 0);
      else if (Short_str[0] == 3)
        ASSERT(A_addr % get_bytesize(Prec[0]) == 0);
    }
    if (!B_is_const) {
      ASSERT(get_npu_index(B_addr) == get_npu_index(R_addr));
      if (Short_str[1] == 0)
        ASSERT(B_addr % ALIGN_BYTES == 0);
      else if (Short_str[1] == 3)
        ASSERT(B_addr % get_bytesize(Prec[1]) == 0);
    }
    if (Short_str[2] == 0)
      ASSERT(R_addr % ALIGN_BYTES == 0);
    else if (Short_str[2] == 3)
      ASSERT(R_addr % get_bytesize(Prec[2]) == 0);
    ASSERT(N < (((int)1) << 16) && (N > 0));
    ASSERT(C < (((int)1) << 16) && (C > 0));
    ASSERT(H < (((int)1) << 16) && (H > 0));
    ASSERT(W < (((int)1) << 16) && (W > 0));
    ASSERT((A_is_const == 0) || (A_is_const == 1));
    ASSERT((B_is_const == 0) || (B_is_const == 1));
    ASSERT((Short_str[0] == 0) || (Short_str[0] == 3));
    ASSERT((Short_str[1] == 0) || (Short_str[1] == 3));
    ASSERT((Short_str[2] == 0) || (Short_str[2] == 3));
    ASSERT((Sign[0] == 0) || (Sign[0] == 1));
    ASSERT((Sign[1] == 0) || (Sign[1] == 1));
#ifdef __sg2262__
    ASSERT(Prec[1] == INT8);
#else
    ASSERT(Prec[0] != INT4 && Prec[1] != INT4 && Prec[2] != INT4);
#endif
    if (A_need_stride) {
       CHECK_AR_STRIDE(tensor_A_stride);
    }
    if (B_need_stride) {
       CHECK_AR_STRIDE(tensor_B_stride);
    }
    if (R_need_stride) {
#ifndef __sg2262__
       CHECK_AR_STRIDE(tensor_R_stride);
#endif
       int dst_shape[4] = {N, C, H, W};
       CHECK_AR_ZERO_DST_STRIDE(tensor_R_stride, dst_shape);
    }
    ASSERT(op == AR_LOGIC_SHIFT || op == AR_ARITH_SHIFT || op == AR_ROTATE_SHIFT);
    ASSERT(Prec[0] != INT4 && Prec[1] != INT4 && Prec[2] != INT4);
    ASSERT(Sign[1] == 1);
    ASSERT(get_bit_width(Prec[1]) <= get_bit_width(Prec[0])
           && is_fixed_prec(Prec[0]) && is_fixed_prec(Prec[1]));
    if (op == AR_ROTATE_SHIFT) {
       ASSERT(Prec[0] == Prec[2]);
    }
#endif
}

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
    int * Short_str, // num = 2, opd0, res0
    int * Sign, // num = 2, opd0, res0
    int sym_saturate,
    PREC * Prec, // num = 2, opd0, res0
    ROUND_MODE round){

    //TENSOR_ARITHMETIC_GET_CYCLE(N, C, H, W, tensor_R_addr, op, pid_node);
#ifdef USING_CMODEL
#ifdef __sg2262__
    bool A_need_stride = Short_str[0] != 0;
    bool R_need_stride = Short_str[1] != 0;
#else
    bool A_need_stride = ((Short_str[0] != 0) && (Short_str[0] != 1));
    bool R_need_stride = ((Short_str[1] != 0) && (Short_str[1] != 1));
#endif
    if (!A_is_const) {
      ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
      if (Short_str[0] == 0)
        ASSERT(A_addr % ALIGN_BYTES == 0);
      else if (Short_str[0] == 3)
        ASSERT(A_addr % get_bytesize(Prec[0]) == 0);
    }
    if (Short_str[1] == 0)
      ASSERT(R_addr % ALIGN_BYTES == 0);
    else if (Short_str[1] == 3)
      ASSERT(R_addr % get_bytesize(Prec[1]) == 0);
    ASSERT(N < (((int)1) << 16) && (N > 0));
    ASSERT(C < (((int)1) << 16) && (C > 0));
    ASSERT(H < (((int)1) << 16) && (H > 0));
    ASSERT(W < (((int)1) << 16) && (W > 0));
    ASSERT((A_is_const == 0) || (A_is_const == 1));
    ASSERT((Short_str[0] == 0) || (Short_str[0] == 3));
    ASSERT((Short_str[1] == 0) || (Short_str[1] == 3));
    ASSERT((Sign[0] == 0) || (Sign[0] == 1));
    ASSERT((Sign[1] == 0) || (Sign[1] == 1));
    ASSERT(sym_saturate == 0 || sym_saturate == 1);
    if (A_need_stride) {
#ifdef __sg2262__
       ASSERT(Prec[0] != INT4 && Prec[0] != FP4);
#else
       ASSERT(Prec[0] != INT4);
#endif
       CHECK_AR_STRIDE(tensor_A_stride);
    }
    if (R_need_stride) {
       ASSERT(Prec[0] != INT4 && Prec[1] != INT4);
#ifdef __sg2262__
       ASSERT(Prec[0] != FP4 && Prec[1] != FP4);
#else
       CHECK_AR_STRIDE(tensor_R_stride);
#endif
       int dst_shape[4] = {N, C, H, W};
       CHECK_AR_ZERO_DST_STRIDE(tensor_R_stride, dst_shape);
    }
    if(A_addr == R_addr) ASSERT(get_bytesize(Prec[0]) >= get_bytesize(Prec[1]));
#endif
}

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
    int * Short_str,// num = 2, opd0, res0
    int sign, // for abs and FP8
    PREC Prec,
    AR_OP op){

#ifdef USING_CMODEL
#ifdef __sg2262__
    bool A_need_stride = Short_str[0] != 0;
    bool R_need_stride = Short_str[1] != 0;
#else
    bool A_need_stride = ((Short_str[0] != 0) && (Short_str[0] != 1));
    bool R_need_stride = ((Short_str[1] != 0) && (Short_str[1] != 1));
#endif
    if (!A_is_const) {
      ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
      if (Short_str[0] == 0)
        ASSERT(A_addr % (get_eu_num(Prec) * get_bytesize(Prec)) == 0);
      else if (Short_str[0] == 3)
        ASSERT(A_addr % get_bytesize(Prec) == 0);
    }
    if (Short_str[1] == 0)
      ASSERT(R_addr % ceiling_func(get_eu_num(Prec) * get_bit_width(Prec), 8) == 0);
    else if (Short_str[1] == 3)
      ASSERT(R_addr % get_bytesize(Prec) == 0);
    ASSERT(N < (((int)1) << 16) && (N > 0));
    ASSERT(C < (((int)1) << 16) && (C > 0));
    ASSERT(H < (((int)1) << 16) && (H > 0));
    ASSERT(W < (((int)1) << 16) && (W > 0));
    ASSERT((A_is_const == 0) || (A_is_const == 1));
    ASSERT((Short_str[0] == 0) || (Short_str[0] == 3));
    ASSERT((Short_str[1] == 0) || (Short_str[1] == 3));
    ASSERT((op == AR_NOT) || (op == AR_COPY) || (op == AR_ABS));
#ifndef __sg2262__
    ASSERT(Prec != INT4 || op == AR_ABS);
#endif
    if (op == AR_NOT) {
      ASSERT(!is_float_prec(Prec) && Prec != INT4);
    }

    if (A_need_stride) {
      if (Prec == INT4) {
        ASSERT(op == AR_COPY);
        ASSERT(tensor_A_stride[3] == 1 && (tensor_A_stride[2] & 0x1) == 0 &&
               (tensor_A_stride[1] & 0x1) == 0 && (tensor_A_stride[0] & 0x1) == 0);
      }

      CHECK_AR_STRIDE(tensor_A_stride);
    }
    if (R_need_stride) {
      if (Prec == INT4) {
        ASSERT(op == AR_COPY);
        ASSERT(tensor_R_stride[3] == 1 && (tensor_R_stride[2] & 0x1) == 0 &&
               (tensor_R_stride[1] & 0x1) == 0 && (tensor_R_stride[0] & 0x1) == 0);
      }
#ifndef __sg2262__
      CHECK_AR_STRIDE(tensor_R_stride);
#endif
//       int dst_shape[4] = {N, C, H, W};
//       CHECK_AR_ZERO_DST_STRIDE(tensor_R_stride, dst_shape);
    }
#endif
}

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
    int * Short_str, // num = 2, opd0, res0
    PREC * Prec, // A && R can have different dtype
    int Sign,
    AR_OP op){

#ifdef USING_CMODEL
#ifdef __sg2262__
    bool A_need_stride = Short_str[0] != 0;
    bool R_need_stride = Short_str[1] != 0;
#else
    bool A_need_stride = ((Short_str[0] != 0) && (Short_str[0] != 1));
    bool R_need_stride = ((Short_str[1] != 0) && (Short_str[1] != 1));
#endif
    if (!A_is_const) {
      if (Short_str[0] == 0)
        ASSERT(A_addr % ALIGN_BYTES == 0);
      else if (Short_str[0] == 3)
        ASSERT(A_addr % get_bytesize(Prec[0]) == 0);
    }
    if (Short_str[1] == 0)
      ASSERT(R_addr % ALIGN_BYTES == 0);
    else if (Short_str[1] == 3)
      ASSERT(R_addr % get_bytesize(Prec[1]) == 0);
#ifdef __sg2262__
    ASSERT(Sign == 0 || Sign == 1);
#endif
    ASSERT(N < (((int)1) << 16) && (N > 0));
    ASSERT(C < (((int)1) << 16) && (C > 0));
    ASSERT(H < (((int)1) << 16) && (H > 0));
    ASSERT(W < (((int)1) << 16) && (W > 0));
    ASSERT((A_is_const == 0) || (A_is_const == 1));
    ASSERT((Short_str[0] == 0) || (Short_str[0] == 3));
    ASSERT((Short_str[1] == 0) || (Short_str[1] == 3));
    ASSERT((op == AR_GET_FIRST_ZERO) || (op == AR_GET_FIRST_ONE));
    ASSERT(Prec[0] != INT4 && Prec[1] != INT4);
    ASSERT(!is_float_prec(Prec[0]) && !is_float_prec(Prec[1]) && Prec[0] != INT4 && Prec[1] != INT4);
    ASSERT(get_bit_width(Prec[1]) <= get_bit_width(Prec[0]));
    if (A_need_stride) {
       CHECK_AR_STRIDE(tensor_A_stride);
    }
    if (R_need_stride) {
#ifndef __sg2262__
       CHECK_AR_STRIDE(tensor_R_stride);
#endif
       int dst_shape[4] = {N, C, H, W};
       CHECK_AR_ZERO_DST_STRIDE(tensor_R_stride, dst_shape);
    }
#endif
}

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
    ROUND_MODE round) {

#ifdef USING_CMODEL
#ifndef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
    bool A_need_stride = (Short_str[0] != 0 && !A_is_const);
    bool B_need_stride = (Short_str[1] != 0 && !B_is_const);
    bool R_need_stride = Short_str[2] != 0;
    ASSERT(N < (((int)1) << 16) && (N > 0));
    ASSERT(C < (((int)1) << 16) && (C > 0));
    ASSERT(H < (((int)1) << 16) && (H > 0));
    ASSERT(W < (((int)1) << 16) && (W > 0));
    ASSERT(A_is_const == 0 || A_is_const == 1);
    ASSERT(B_is_const == 0 || B_is_const == 1);
    ASSERT((Short_str[0] == 0) || (Short_str[0] == 1)|| (Short_str[0] == 3));
    ASSERT((Short_str[1] == 0) || (Short_str[1] == 1)|| (Short_str[1] == 3));
    ASSERT((Short_str[2] == 0) || (Short_str[2] == 1)|| (Short_str[2] == 3));
    if (A_is_const == 0) {
       ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
       if (Short_str[0] == 0) {
              ASSERT(A_addr % ALIGN_BYTES == 0);
       } else {
              ASSERT(A_addr % get_bytesize(Prec[0]) == 0);
       }
    }
    if (B_is_const == 0) {
       ASSERT(get_npu_index(B_addr) == get_npu_index(R_addr));
       if (Short_str[1] == 0) {
              ASSERT(B_addr % ALIGN_BYTES == 0);
       } else {
              ASSERT(B_addr % get_bytesize(Prec[1]) == 0);
       }
    }
    if (Short_str[2] == 0) {
       ASSERT(R_addr % ALIGN_BYTES == 0);
    } else {
       ASSERT(R_addr % get_bytesize(Prec[2]) == 0);
    }
    ASSERT(Sign[0] == 0 || Sign[0] == 1);
    ASSERT(Sign[1] == 0 || Sign[1] == 1);
    ASSERT(Sign[2] == 0 || Sign[2] == 1);
    ASSERT(Prec[0] == FP16 || Prec[0] == BFP16 || Prec[0] == FP32 || Prec[0] == FP8);
    if (Prec[0] == FP8) {
       ASSERT(Prec[1] == FP16 || Prec[1] == BFP16 || Prec[1] == FP32 || Prec[1] == FP8);
    } else {
       ASSERT(Prec[1] == Prec[0]);
    }
    ASSERT(Prec[2] == FP16 || Prec[2] == BFP16 || Prec[2] == FP32 || Prec[2] == FP8 || Prec[2] == INT8 || Prec[2] == INT16 || Prec[2] == INT32);
    if (A_need_stride) {
       CHECK_AR_STRIDE(tensor_A_stride);
    }
    if (B_need_stride) {
       CHECK_AR_STRIDE(tensor_B_stride);
    }
    if (R_need_stride) {
       CHECK_AR_STRIDE(tensor_B_stride);
    }
#endif
#endif
}
