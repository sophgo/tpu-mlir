#include "atomic_tiu.h"

/* requantize with fp32 scale and zero-point
 * A is aligned in local memory, support S8/U8/S16/U16/S32
 * B is fp32, and aligned in local memory,
 * B=[0,C,1,2], B(0,c,0,0) is scale, B(0,c,0,1) is zp
 * R is aligned in local memory, support S4/U4/S8/U8/S16/U16
 */
void atomic_rq_f32mode_check(
  u32 A_addr,
  u32 B_addr,
  u32 R_addr,
  int N,
  int C,
  int H,
  int W,
  int B_is_const,
  float scale_value,
  float zp_value,
  int A_sign,
  int R_sign,
  int sym_range,
  PREC A_prec,
  PREC R_prec,
  ROUND_MODE i2f_mode,
  ROUND_MODE f2i_mode)
{

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  ASSERT(B_is_const == 0 || B_is_const == 1);
  ASSERT(sym_range == 0 || sym_range == 1);
  ASSERT(R_sign == 0 || R_sign == 1);
  ASSERT(A_sign == 0 || A_sign == 1);
  ASSERT(A_prec == INT16 || A_prec == INT32 || A_prec == INT8);
  ASSERT(R_prec == INT8 || R_prec == INT16 || R_prec == INT4);
  ASSERT(N < (1 << 16) && N > 0);
  ASSERT(C < (1 << 16) && C > 0);
  ASSERT(H < (1 << 16) && H > 0);
  ASSERT(W < (1 << 16) && W > 0);
  ASSERT(i2f_mode < (1 << 3) && i2f_mode >= 0);
  ASSERT(f2i_mode < (1 << 3) && f2i_mode >= 0);
  ASSERT(i2f_mode != ROUND_HALF_UP && i2f_mode != ROUND_HALF_DOWN);
  ASSERT(f2i_mode != ROUND_HALF_UP && f2i_mode != ROUND_HALF_DOWN);
  ASSERT(A_addr % ALIGN_BYTES == 0 && R_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
  if (!B_is_const) {
    ASSERT(B_addr % (get_bytesize(INT32) * 2) == 0);
    ASSERT(get_npu_index(A_addr) == get_npu_index(B_addr));
  }

#endif
#endif
}

/* RQ_1 with int scale-factor, right-shift and zp,
 * A is aligned with shape=[N, C, H, W], support S32/S16/U16/S8/U8
 * B is aligned with shape=[1, C, 1, 3], storaged as S32
 * scale_factor=B(0,c,0,0) support S8/S16/S32,
 * shift_num=B(0,c,0,1) support S8,
 * zp=B(0,c,0,2) support U8/U4, zp_sign=res_sign
 * R is aligned with shape=[N, C, H, W], support S4/U4/S8/U8/S16/U16
 */
void atomic_rq_i32mode_check(
  u32 A_addr,
  u32 B_addr,
  u32 R_addr,
  int N,
  int C,
  int H,
  int W,
  int B_is_const,
  int scale_val,
  char shift_val, // negative: right shift, positive: left shift
  short zp_val,
  int A_sign,
  int R_sign,
  int sym_range,
  PREC A_prec,
  PREC R_prec,
  ROUND_MODE shift_rd)
{

#ifdef USING_CMODEL
  ASSERT(shift_rd < (1 << 3) && shift_rd >= 0);
  ASSERT(B_is_const == 0 || B_is_const == 1);
#ifdef __sg2262__
  ASSERT(sym_range == 0);
#else
  ASSERT(sym_range == 0 || sym_range == 1);
#endif
  ASSERT(A_sign == 0 || A_sign == 1);
  ASSERT(R_sign == 0 || R_sign == 1);
  ASSERT(A_prec == INT16 || A_prec == INT32 || A_prec == INT8);
  ASSERT(R_prec == INT4 || R_prec == INT8 || R_prec == INT16);
  ASSERT(N < (1 << 16) && N > 0);
  ASSERT(C < (1 << 16) && C > 0);
  ASSERT(H < (1 << 16) && H > 0);
  ASSERT(W < (1 << 16) && W > 0);
  ASSERT(A_addr % ALIGN_BYTES == 0 && R_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
  if (!B_is_const) {
    ASSERT(B_addr % (get_bytesize(INT32) * 2) == 0);
    ASSERT(get_npu_index(B_addr) == get_npu_index(A_addr));
  }
#endif
}

/* Dequantize with scale and output are float32
 * A=[N,C,H,W] with aligned in local memory, support INT4/INT8/INT16
 * B=[1,C,1,2] and B(0,c,0,0) with aligned in local memory, PREC is same with A,
 * R=[N,C,H,W] with aligned in local memory
 */
void atomic_dq_f32mode_check(
  u32 A_addr,
  u32 B_addr,
  u32 R_addr,
  int N,
  int C,
  int H,
  int W,
  int B_is_const,
  float scale_value,
  short zp_value, // INT8/INT16
  int A_sign,
  int R_sign,
  PREC A_prec,
  PREC R_prec,
  ROUND_MODE i2f_mode)
{

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT(A_prec == INT8 || A_prec == INT16);
  ASSERT(R_prec == FP32);
#else
  ASSERT(A_prec == INT4 || A_prec == INT8 || A_prec == INT16);
#endif
  ASSERT(A_sign == 0 || A_sign == 1);
  ASSERT(B_is_const == 0 || B_is_const == 1);
  ASSERT(N < (1 << 16) && N > 0);
  ASSERT(C < (1 << 16) && C > 0);
  ASSERT(H < (1 << 16) && H > 0);
  ASSERT(W < (1 << 16) && W > 0);
  ASSERT(i2f_mode < (1 << 3) && i2f_mode >= 0);
  ASSERT(i2f_mode != ROUND_HALF_UP && i2f_mode != ROUND_HALF_DOWN);
  ASSERT(A_addr % ALIGN_BYTES == 0 && R_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
  if (!B_is_const) {
    ASSERT(B_addr % (get_bytesize(INT32) * 2) == 0);
    ASSERT(get_npu_index(B_addr) == get_npu_index(A_addr));
  }
#endif
}

void atomic_dq_f16mode_check(
  u32 A_addr,
  u32 B_addr,
  u32 R_addr,
  int N,
  int C,
  int H,
  int W,
  int B_is_const,
  u16 scale_value, // fp16/bf16
  short zp_value, // INT8/INT16
  int A_sign,
  int R_sign,
  PREC A_prec,
  PREC R_prec,
  ROUND_MODE i2f_mode)
{

#ifdef USING_CMODEL
#ifndef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  ASSERT(A_prec == INT8 || A_prec == INT16);
  ASSERT(R_prec == FP16 || R_prec == BFP16);
  ASSERT(A_sign == 0 || A_sign == 1);
  ASSERT(B_is_const == 0 || B_is_const == 1);
  ASSERT(N < (1 << 16) && N > 0);
  ASSERT(C < (1 << 16) && C > 0);
  ASSERT(H < (1 << 16) && H > 0);
  ASSERT(W < (1 << 16) && W > 0);
  ASSERT(i2f_mode < (1 << 3) && i2f_mode >= 0);
  ASSERT(i2f_mode != ROUND_HALF_UP && i2f_mode != ROUND_HALF_DOWN);
  ASSERT(A_addr % ALIGN_BYTES == 0 && R_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
  if (!B_is_const) {
    ASSERT(B_addr % (get_bytesize(INT32)) == 0);
    ASSERT(get_npu_index(B_addr) == get_npu_index(A_addr));
  }
#endif
#endif
}

/* Dequantize for fixed output and zp ...
 * A=[N, C, H, W] with aligned in local memory, support INT4/INT8/INT16
 * B=[1, C, 1, 3] with aligned in local memory, B(0,c,0,0) is zp,
 * B(0,c,0,1) is scale_factor, B(0,c,0,2) is shift_num,
 * the prec is the same with A, but storaged as int32 in local mem
 * R=[N, C, H, W] with aligned in local memory, support S8/U8/S16/U16/S32
 */
void atomic_dq_i32mode_check(
  u32 A_addr,
  u32 B_addr,
  u32 R_addr,
  int N,
  int C,
  int H,
  int W,
  int B_is_const,
  short zp_value, // s16/u16
  int scale_factor, // S8/S16/S32
  char shift_num, // negative: right shift, positive: left shift
  int A_sign,
  int R_sign,
  int sym_range,
  PREC A_prec,
  PREC R_prec,
  ROUND_MODE shift_rd)
{

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  ASSERT(shift_rd < (1 << 3) && shift_rd >= 0);
  ASSERT(B_is_const == 0 || B_is_const == 1);
  ASSERT(sym_range == 0 || sym_range == 1);
  ASSERT(A_sign == 0 || A_sign == 1);
  ASSERT(R_sign == 0 || R_sign == 1);
  ASSERT(A_prec == INT4 || A_prec == INT8 || A_prec == INT16);
  ASSERT(R_prec == INT32 || R_prec == INT16 || R_prec == INT8);
  ASSERT(N < (1 << 16) && N > 0);
  ASSERT(C < (1 << 16) && C > 0);
  ASSERT(H < (1 << 16) && H > 0);
  ASSERT(W < (1 << 16) && W > 0);
  ASSERT(A_addr % ALIGN_BYTES == 0 && R_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
  if (!B_is_const) {
    ASSERT(B_addr % (get_bytesize(INT32) * 2) == 0);
    ASSERT(get_npu_index(A_addr) == get_npu_index(B_addr));
  }
#endif
#endif
}

/* Dequantize with scale and output are fp16/bfp16
 * A=[N,C,H,W] with aligned in local memory, support INT4/INT8
 * B=[N,C,H, DIV_UP(w, gsize)] with compress in local memory
 * B(n,c,h, gw[15:0]) is scale, B(n,c,h,gw[31:16]) is offset,
 * the prec is FP16/BF16, but storaged as int32 in local mem
 * R=[N,C,H,W] with aligned in local memory, support FP16/BF16
 */
void atomic_dq_half_prec_check(
  u32 A_addr,
  u32 B_addr,
  u32 R_addr,
  int N,
  int C,
  int H,
  int W,
  int gsize,  // 32,  64,  128,  256
  int A_sign,
  PREC A_prec,
  PREC R_prec)
{

#ifdef USING_CMODEL
#ifndef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  ASSERT(A_prec == INT4 || A_prec == INT8);
  ASSERT(R_prec == BFP16 || R_prec == FP16);
  ASSERT(A_sign == 0 || A_sign == 1);
  ASSERT(N < (1 << 16) && N > 0);
  ASSERT(C < (1 << 16) && C > 0);
  ASSERT(H < (1 << 16) && H > 0);
  ASSERT(W < (1 << 16) && W > 0);
  ASSERT(A_addr % ALIGN_BYTES == 0 && R_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
  ASSERT(B_addr % (get_bytesize(INT32)) == 0);
  ASSERT(get_npu_index(B_addr) == get_npu_index(A_addr));
  ASSERT(gsize == 32 || gsize == 64 || gsize == 128 || gsize == 256);
  ASSERT(W % gsize == 0);
  // 0: 32, 1: 64, 2: 128, 3: 256 for gsize
  u32 opd2_n_str = log2((double)gsize) - 5;
  ASSERT_FS_INFO( opd2_n_str >= 0 && opd2_n_str < 4, "invalid gsize");
#endif
#endif
}

/*
quant fp16/bf16 to fp8/int8
*/
void atomic_qt0_half_prec_check(
  u32 A_addr,
  u32 range,
  u32 R_addr,
  u32 R1_addr,
  int N,
  int C,
  int H,
  int W,
  int A_short_str,
  FP8_TYPE R_fp8_type,
  PREC A_prec,
  PREC R_prec,
  int saturate)
{

#ifdef USING_CMODEL
#ifndef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  ASSERT(A_prec == FP16 || A_prec == BFP16);
  ASSERT(R_prec == FP8 || R_prec == INT8);
  ASSERT(A_short_str == 0 || A_short_str == 6);
  ASSERT(R_fp8_type == 0 || R_fp8_type == 1);
  ASSERT(saturate == 0 || saturate == 1);
  ASSERT(N < (1 << 16) && N > 0);
  ASSERT(C < (1 << 16) && C > 0);
  ASSERT(H < (1 << 16) && H > 0);
  ASSERT(W < (1 << 16) && W > 0);
  if (A_short_str == 0) {
  ASSERT(A_addr % ALIGN_BYTES == 0 && R_addr % ALIGN_BYTES == 0);
  } else {
  ASSERT(A_addr % get_bytesize(A_prec) == 0);
  }
  ASSERT(R1_addr % get_bytesize(A_prec) == 0);
  ASSERT(get_npu_index(A_addr) == get_npu_index(R_addr));
  ASSERT(get_npu_index(A_addr) == get_npu_index(R1_addr));
#endif
#endif
}
