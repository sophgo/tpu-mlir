#include "atomic_tiu.h"

void atomic_mm2_check(
  u32 L_addr,
  u32 R_addr,
  u32 Y_addr,
  u32 Bias_addr,
  u32 RQ_addr, //save re_scale when const
  int L_row_num,
  int L_col_num,
  int R_col_num,
  int is_L_trans,
  int is_R_trans,
  int is_L_const,
  int is_R_const,
  int is_bias_const,
  int add_result,
  int do_relu,
  int do_rq,
  int is_rq_const,
  int saturate,
  PREC LR_prec,
  PREC B_prec,
  PREC Y_prec,
  FP8_TYPE L_fp8_type,
  FP8_TYPE R_fp8_type,
  FP8_TYPE Y_fp8_type,
  int tf32_mode)
{

#ifdef USING_CMODEL
  MM_OP mm_op = MM_NN;
  if (!is_L_trans && is_R_trans) mm_op = MM_NT;
  if (is_L_trans && is_R_trans) mm_op = MM_TT;
  if (tf32_mode != 0) {
    if (mm_op == MM_NN) mm_op = MM_NN_TF32;
    else if (mm_op == MM_NT) mm_op = MM_NT_TF32;
    else if (mm_op == MM_TT) mm_op = MM_TT_TF32;
  }
  if (!is_L_const) {
    ASSERT(L_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(L_addr) == 0);
  }
  if (!is_R_const) {
    ASSERT(R_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(R_addr) == 0);
  }
  if (!is_bias_const) {
    if (mm_op == MM_TT || mm_op == MM_TT_TF32) ASSERT(Bias_addr % get_bytesize(FP32) == 0);
    else ASSERT(Bias_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(Bias_addr) == 0);
  }
  ASSERT(Y_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(Y_addr) == 0);

  if (!is_rq_const) {
    ASSERT(RQ_addr % get_bytesize(FP32) == 0);
    ASSERT(get_npu_index(RQ_addr) == 0);
  }

  if (!is_R_trans) ASSERT(!is_L_trans);
  ASSERT(LR_prec == FP32 || LR_prec == FP16 ||
         LR_prec == BFP16 || LR_prec == FP8);
  if (LR_prec == FP32) {
    ASSERT(Y_prec == FP32);
    ASSERT(B_prec == FP32);
  }
  if (LR_prec == FP16) {
    ASSERT(Y_prec == FP16 || Y_prec == FP32);
  #ifndef __sg2262__
    ASSERT(B_prec == FP16 || B_prec == FP32);
  #endif
  }
  if (LR_prec == BFP16) {
    ASSERT(Y_prec == BFP16 || Y_prec == FP32);
    ASSERT(B_prec == BFP16 || B_prec == FP32);
  }
#ifdef __sg2262__
  if (LR_prec == FP8) {
    ASSERT(Y_prec == FP8 || Y_prec == FP16 || Y_prec == FP32 || Y_prec == BFP16);
    ASSERT(B_prec == FP16 || B_prec == FP32 || B_prec == BFP16);
  }
  if (mm_op == MM_NT && add_result) {
    ASSERT(Y_prec == FP32);
  }
#else
  if (LR_prec == FP8) {
    ASSERT(Y_prec == FP8 || Y_prec == FP16 || Y_prec == FP32);
    ASSERT(B_prec == FP16 || B_prec == FP32);
  }
#endif
  ASSERT(is_L_const < (1 << 1) && is_L_const >= 0);
  ASSERT(is_R_const < (1 << 1) && is_R_const >= 0);
  ASSERT(is_bias_const < (1 << 1) && is_bias_const >= 0);
  ASSERT(L_row_num < (1 << 16) && L_row_num > 0);
  ASSERT(L_col_num < (1 << 16) && L_col_num > 0);
  ASSERT(R_col_num < (1 << 16) && R_col_num > 0);
  ASSERT(add_result < (1 << 1) && add_result >= 0);
  ASSERT(do_relu < (1 << 1) && do_relu >= 0);
#ifdef __sg2262__
  ASSERT(saturate < (1 << 1) && saturate >= 0);
  ASSERT((do_rq == 0 && is_rq_const == 1) || do_rq == 1);
#else
  ASSERT(do_rq == 0);
#endif
  ASSERT(is_rq_const == 0 || is_rq_const == 1);

  ASSERT(L_fp8_type == FP8E4M3 || L_fp8_type == FP8E5M2);
  ASSERT(R_fp8_type == FP8E4M3 || R_fp8_type == FP8E5M2);
  ASSERT(Y_fp8_type == FP8E4M3 || Y_fp8_type == FP8E5M2);
#endif
}

/* if (!L_trans) zp=[1,NPU_NUM,1,R_col_num],and its stride is[0,0,W,1]
 * if (L_trans && R_trans) zp=[1,R_col_num,1,1], and it is compacted in local memory
 */
void atomic_mm2_fixed_check(
  u32 L_addr,
  u32 R_addr,
  u32 Y_addr,
  u32 rzp_addr,
  u32 Bias_addr,
  u32 RQ_addr,
  s8 shift_val,
  s16 yzp_val,
  int L_row_num,
  int L_col_num,
  int R_col_num,
  int is_L_trans,
  int is_R_trans,
  int is_L_const,
  int is_R_const,
  int is_zp_const,
  int L_sign,
  int R_sign,
  int add_result,
  int Res_sign,
  int Bias_sign,
  int is_bias_const,
  int is_rq_const,
  int do_relu,
  int sym_range,
  int do_rq,
  ROUND_MODE rshift_rd,
  PREC L_prec,
  PREC R_prec,
  PREC Y_prec)
{

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  MM_OP mm_op = MM_NN;
  if (!is_L_trans && is_R_trans) mm_op = MM_NT;
  if (is_L_trans && is_R_trans) mm_op = MM_TT;
  if (is_bias_const && Bias_addr == 0) Bias_sign = 0;
  if (!is_L_const) {
    ASSERT(L_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(L_addr) == 0);
  }
  if (!is_R_const) {
    ASSERT(R_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(R_addr) == 0);
  }
  if (!is_bias_const) {
    if (mm_op == MM_TT) ASSERT(Bias_addr % get_bytesize(INT32) == 0);
    else ASSERT(Bias_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(Bias_addr) == 0);
  }
  if (!is_zp_const) {
    if (mm_op == MM_TT) ASSERT(rzp_addr % get_bytesize(INT16) == 0);
    else ASSERT(rzp_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(rzp_addr) == 0);
  }
  if (do_rq && !is_rq_const) {
    if (mm_op == MM_TT) ASSERT(RQ_addr % (sizeof(int) * 2) == 0);
    else ASSERT(RQ_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(RQ_addr) == 0);
  }

  ASSERT(Y_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(Y_addr) == 0);

  ASSERT(L_prec == INT8);
  ASSERT(R_prec == INT8);
  ASSERT(Y_prec == INT8 || Y_prec == INT16 || Y_prec == INT32);
  if (!is_R_trans) ASSERT(!is_L_trans);
  ASSERT(L_sign < (1 << 1) && L_sign >= 0);
  ASSERT(R_sign < (1 << 1) && R_sign >= 0);
  ASSERT(Bias_sign < (1 << 1) && Bias_sign >= 0);
  ASSERT(is_bias_const < (1 << 1) && is_bias_const >= 0);
  ASSERT(is_zp_const < (1 << 1) && is_zp_const >= 0);
  ASSERT(is_rq_const < (1 << 1) && is_rq_const >= 0);
  ASSERT(is_L_const < (1 << 1) && is_L_const >= 0);
  ASSERT(is_R_const < (1 << 1) && is_R_const >= 0);
  ASSERT(L_row_num < (1 << 16) && L_row_num > 0);
  ASSERT(L_col_num < (1 << 16) && L_col_num > 0);
  ASSERT(R_col_num < (1 << 16) && R_col_num > 0);
  ASSERT(add_result < (1 << 1) && add_result >= 0);
  ASSERT(Res_sign < (1 << 1) && Res_sign >= 0);
  ASSERT(do_relu < (1 << 1) && do_relu >= 0);
  ASSERT(sym_range < (1 << 1) && sym_range >= 0);
  ASSERT(do_rq < (1 << 1) && do_rq >= 0);
  ASSERT(rshift_rd < 7 && rshift_rd >= 0);
#endif
#endif
}

void atomic_dq2_mm2_check(
  u32 L_addr, //activation, fp16/bf16/fp8
  u32 R_addr, //weight, fp4/int4
  u32 Y_addr,
  u32 DQ2_addr, //for weight dequant use
  u32 RQ_addr, //save re_scale when const
  int L_row_num,
  int L_col_num,
  int R_col_num,
  int gsize,
  int is_L_trans,
  int add_result,
  int do_rq,
  int is_rq_const,
  int saturate,
  PREC L_prec,
  PREC R_prec,
  PREC DQ2_prec, //8bit/16bit/32bit
  PREC Y_prec,
  FP8_TYPE L_fp8_type,
  FP8_TYPE Y_fp8_type,
  int R_sign)
{

#ifdef USING_CMODEL
#ifndef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  // 0: 32, 1: 64, 2: 128, 3: 256 for gsize
  u32 opd2_n_str = log2((double)gsize) - 5;
  MM_OP mm_op = DQ2_MM2_NT;
  if (is_L_trans) mm_op = DQ2_MM2_TT;
  ASSERT(L_addr % ALIGN_BYTES == 0 && get_npu_index(L_addr) == 0);
  ASSERT(R_addr % ALIGN_BYTES == 0 && get_npu_index(R_addr) == 0);
  ASSERT(Y_addr % ALIGN_BYTES == 0 && get_npu_index(Y_addr) == 0);
  if (!is_rq_const) {
    ASSERT(RQ_addr % get_bytesize(FP32) == 0);
    ASSERT(get_npu_index(RQ_addr) == 0);
  }
  ASSERT(L_prec == FP8 || L_prec == FP16 || L_prec == BFP16);
  ASSERT(R_prec == FP4 || R_prec == INT4);
  if (mm_op == DQ2_MM2_NT && add_result == 1) {
    ASSERT(Y_prec == FP32);
  }
  if (L_prec == FP8) {
    ASSERT(Y_prec == FP32 || Y_prec == FP16 || Y_prec == BFP16 || Y_prec == FP8);
    if (R_prec == INT4) {
      //dq2 is 16bit
      ASSERT(get_bytesize(DQ2_prec) == 2);
    } else {
      //dq2 is 8bit
      ASSERT(get_bytesize(DQ2_prec) == 1);
    }
  } else {
    ASSERT(Y_prec == FP32 || Y_prec == L_prec);
    if (R_prec == INT4) {
      //dq2 is 32bit
      ASSERT(get_bytesize(DQ2_prec) == 4);
    } else {
      //dq2 is 16bit
      ASSERT(get_bytesize(DQ2_prec) == 2);
    }
  }
  ASSERT(DQ2_addr % get_bytesize(DQ2_prec) == 0 && get_npu_index(DQ2_addr) == 0);
  ASSERT(L_col_num % gsize == 0);
  ASSERT(opd2_n_str >= 0 &&  opd2_n_str <= 3);
  ASSERT(L_row_num < (1 << 16) && L_row_num > 0);
  ASSERT(L_col_num < (1 << 16) && L_col_num > 0);
  ASSERT(R_col_num < (1 << 16) && R_col_num > 0);
  ASSERT(add_result < (1 << 1) && add_result >= 0);
  ASSERT(saturate < (1 << 1) && saturate >= 0);
  ASSERT(is_rq_const == 0 || is_rq_const == 1);
  ASSERT(L_fp8_type == FP8E4M3 || L_fp8_type == FP8E5M2);
  ASSERT(Y_fp8_type == FP8E4M3 || Y_fp8_type == FP8E5M2);
  ASSERT(R_sign == 0 || R_sign == 1);
  ASSERT((do_rq == 0 && is_rq_const == 1) || do_rq == 1);
#endif
#endif
}

void atomic_bq_mm2_check(
  u32 L_addr, //activation, fp8/fp4
  u32 R_addr, //weight, fp8/fp4
  u32 Y_addr,
  u32 L_bq_addr, //activation's dequant param addr
  u32 R_bq_addr, //weight's dequant param addr
  int L_row_num,
  int L_col_num,
  int R_col_num,
  int gsize,
  int add_result,
  int saturate,
  PREC L_prec,
  PREC R_prec,
  PREC BQ_prec,
  PREC Y_prec,
  FP8_TYPE L_fp8_type,
  FP8_TYPE R_fp8_type)
{
#ifdef USING_CMODEL
#ifndef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  // 0: 32, 1: 64, 2: 128, 3: 256 for gsize
  u32 opd2_n_str = log2((double)gsize) - 5;
  ASSERT(L_addr % ALIGN_BYTES == 0 && get_npu_index(L_addr) == 0);
  ASSERT(R_addr % ALIGN_BYTES == 0 && get_npu_index(R_addr) == 0);
  ASSERT(Y_addr % ALIGN_BYTES == 0 && get_npu_index(Y_addr) == 0);
  ASSERT(L_bq_addr % get_bytesize(BQ_prec) == 0 && get_npu_index(L_bq_addr) == 0);
  ASSERT(R_bq_addr % get_bytesize(BQ_prec) == 0 && get_npu_index(R_bq_addr) == 0);
  ASSERT(L_prec == FP8);
  ASSERT(R_prec == FP8 || R_prec == FP4);
  ASSERT(Y_prec == FP32 || Y_prec == FP16 || Y_prec == BFP16);
  ASSERT(BQ_prec == FP16 || BQ_prec == BFP16);
  ASSERT(L_col_num % gsize == 0);
  ASSERT(opd2_n_str >= 0 &&  opd2_n_str <= 3);
  ASSERT(L_row_num < (1 << 16) && L_row_num > 0);
  ASSERT(L_col_num < (1 << 16) && L_col_num > 0);
  ASSERT(R_col_num < (1 << 16) && R_col_num > 0);
  ASSERT(add_result < (1 << 1) && add_result >= 0);
  ASSERT(saturate < (1 << 1) && saturate >= 0);
  ASSERT(L_fp8_type == FP8E4M3 || L_fp8_type == FP8E5M2);
  ASSERT(R_fp8_type == FP8E4M3 || R_fp8_type == FP8E5M2);
#endif
#endif
}
