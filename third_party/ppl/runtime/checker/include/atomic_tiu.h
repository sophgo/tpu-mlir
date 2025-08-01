#ifndef ATOMIC_TIU_H
#define ATOMIC_TIU_H
#include "checker_internel.h"
#ifdef __cplusplus
extern "C" {
#endif

static inline u64 bd_get_lane_mask() {
  return 0xffffffffffffffffull;
}

static inline int bd_power_step() {
#ifdef ENABLE_POWER_CTRL
  return 0xf;
#else
  return 0x0;
#endif
}
static inline u64 gdma_get_cache_en() {
  return 0;
}

void atomic_conv_check(
    u32 input_addr,
    u32 weight_addr,
    u32 bias_addr,
    u32 pad_ins_addr,
    u32 rescale_addr,
    u32 output_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int output_c,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int ins_h,
    int ins_w,
    int dilation_h,
    int dilation_w,
    int pad_h_t,
    int pad_h_b,
    int pad_w_l,
    int pad_w_r,
    int kernel_is_const,
    int bias_is_const,
    int pad_ins_is_const,
    int kernel_rotate,
    int result_add,
    u32 ins_const_val,
    int *input_stride,
    int do_relu,
    int saturate,
    PREC input_prec,
    PREC output_prec,
    PREC bias_prec,
    int input_sign,
    int weight_sign,
    int res_sign,
    int bias_sign,
    int do_rescale,
    int rescale_is_const,
    PAD_MODE pad_mode);

void atomic_conv_quant_check(
    u32 input_addr,
    u32 weight_addr, // or weight const value
    u32 bias_addr, // or bias const value
    u32 pad_ins_addr, // pad const value
    u32 kzp_addr, // kzp const value
    u32 requant_addr, // multipiler const value
    u32 output_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int output_c,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int ins_h,
    int ins_w,
    int dilation_h,
    int dilation_w,
    int pad_h_t,
    int pad_h_b,
    int pad_w_l,
    int pad_w_r,
    int kernel_is_const,
    int bias_is_const,
    int pad_ins_is_const,
    int kzp_is_const,
    int kernel_rotate,
    int result_add,
    u32 ins_const_val,
    int input_sign,
    int weight_sign,
    int bias_sign,
    int res_sign,
    int *input_stride,
    int do_relu,
    int sym_saturate,
    int do_requant,
    int requant_is_const,
    int shift_num, // s8
    int ozp, // s16
    ROUND_MODE rm_mode,
    PREC input_prec,
    PREC weight_prec,
    PREC output_prec,
    PAD_MODE pad_mode);

void atomic_fused_linear_check(
    u32   A_addr,
    u32   B_addr,
    u32   C_addr,
    u32   Y_addr,
    int   input_n,
    int   input_c,
    int   input_h,
    int   input_w,
    int   B_is_const,
    int   C_is_const,
    PREC input_prec,
    PREC output_prec,
    FP8_TYPE fp8_type,
    LIN_OP op,
    int saturate
);

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
  );

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
    PREC AB_prec,
    PREC CD_prec,
    PREC RES0_dtype,
    CMP_OP op);

void atomic_mm_check(
  u32 L_addr,
  u32 R_addr,
  u32 Y_addr,
  u32 bias_addr,
  int L_tensor_W,
  int L_tensor_C,
  int R_tensor_W,
  int R_tensor_C,
  int L_row_num,
  int L_col_num,
  int is_L_trans,
  int is_L_const,
  int is_bias_const,
  int add_result,
  int do_relu);

void atomic_mm_fixed_check(
  u32 L_addr,
  u32 R_addr,
  u32 Y_addr,
  u32 bias_addr,
  int L_tensor_W,
  int L_tensor_C,
  int R_tensor_W,
  int R_tensor_C,
  int L_row_num,
  int L_col_num,
  int is_L_trans,
  int is_L_const,
  int L_sign,
  int R_sign,
  int bias_sign,
  int Res_sign,
  int is_bias_const,
  int add_result,
  int if_relu,
  int sym_range,
  int do_rq,
  s32 multiplier,
  s8 shift,
  s16 yzp,
  PREC Y_prec,
  PREC LR_prec,
  ROUND_MODE round_mode);

void atomic_mm2_check(
  u32 L_addr,
  u32 R_addr,
  u32 Y_addr,
  u32 Bias_addr,
  u32 RQ_addr,
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
  int tf32_mode);

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
  PREC Y_prec);

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
  int R_sign);

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
  FP8_TYPE R_fp8_type);

void atomic_lane_broad_check(
    u32 src_addr, // in local memory
    u32 dst_addr, // in local memory
    int N,
    int H,
    int W,
    int dst_C,
    u64 lane_mask,
    PREC prec);

void atomic_lane_broad_txp_check(
    u32 src_addr, // in local memory
    u32 dst_addr, // in local memory
    int N,
    int H,
    int W,
    int dst_C,
    PREC prec);

void atomic_lane_copy_check(
    u32 src_addr, // in local memory
    u32 dst_addr, // in local memory
    int N,
    int C,
    int H,
    int W,
    PREC prec);

void atomic_static_broad_check(
    u32 src_addr, // in static memory
    u32 dst_addr, // in local memory
    int C,
    int W,
    u64 lane_mask,
    PREC prec);

void atomic_static_broad_txp_check(
    u32 src_addr, // in static memory
    u32 dst_addr, // in local memory
    int C,
    int W,
    PREC prec);

void atomic_static_distribute_check(
    u32 src_addr, // in static memory
    u32 dst_addr, // in local memory
    int C,
    u64 lane_mask,
    PREC prec);

void atomic_static_distribute_txp_check(
    u32 src_addr, // in static memory
    u32 dst_addr, // in local memory
    int C,
    PREC prec);

void atomic_cw_transpose_check(
    u32   A_addr,
    u32   Y_addr,
    int   input_n,
    int   input_c,
    int   input_h,
    int   input_w,
    PREC dtype,
    TRAN_OP op
);

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
        int R_sign);

// scatter_gather_line
/* Note:tensorA is aligned in local memory, and its stride is
 * wstride=1,hstride=ceil(w,EU_NUM) * EU_NUM,
 * cstride = A_cstride_is0 ? 0 : h*hstride,
 * nstride = c_per_npu * cstride.
 * And tensorR stride is wstride=1, hstride=ceil(w,EU_NUM) * EU_NUM
 * cstride=h*hstride, nstride=c_per_npu * cstride.
 * And tensorB is aligned in local memory with normal storage.
 */
void atomic_sgl_check(
  u32 tensorA_addr,
  u32 tensorB_addr,
  u32 tensorR_addr,
  int tensorA_h,
  int tensorR_n,
  int tensorR_c,
  int tensorR_h,
  int tensorR_w,
  int A_cstride_is0,
  int if_fill_const,
  u32 fill_const_val,
  int limit_enable,
  PREC B_prec,
  PREC R_prec,
  SG_OP op);

/* PL_gather_d1coor :A=[N,C,1,Wa], B=[1,Wr,1,1], R=[N,C,1,Wr]
 * PL_scatter_d1coor:A=[N,C,1,Wa], B=[1,Wa,1,1], R=[N,C,1,Wr]
 * A and R is aligned in local memory, B is compacted in local mem
 */
void atomic_pl_sgd1_check(
  u32 tensorA_addr,
  u32 tensorB_addr,
  u32 tensorR_addr,
  int tensorA_w,
  int tensorR_n,
  int tensorR_c,
  int tensorR_w,
  int if_fill_const,
  u32 fill_const_val,
  int limit_enable,
  PREC B_prec,
  PREC R_prec,
  SG_OP op);

/* PL_gather_d2coor :A=[N,C,Ha,Wa], B=[1,Wr,1,1], R=[N,C,1,Wr]
 * PL_scatter_d2coor:A=[N,C,1,Wa], B=[1,Wa,1,1], R=[N,C,Hr,Wr]
 * A and R is aligned in local memory, B is compacted in local mem
 * opd1 is uint16, but storaged as INT32 with [h, w]
 */
void atomic_pl_sgd2_check(
  u32 tensorA_addr,
  u32 tensorB_addr,
  u32 tensorR_addr,
  int tensorA_h,
  int tensorA_w,
  int tensorR_n,
  int tensorR_c,
  int tensorR_h,
  int tensorR_w,
  int if_fill_const,
  u32 fill_const_val,
  int limit_enable,
  PREC R_prec,
  SG_OP op);

/* PE_S_gather_d1coor
 * PE_S_gather_hzd
 * PE_S_scatter_d1coor
 * PE_S_scatter_hzd
 */
/* A is aligned in local memory, if A_cstride is 0
 * A_wstride=1;A_hstride=ceil(w,EU_NUM),cstride=0;nstride=0
 * B and R is aligned in local memory
 */
void atomic_pes_sg_d1hzd_check(
  u32 tensorA_addr,
  u32 tensorB_addr,
  u32 tensorR_addr,
  int tensorA_w,
  int tensorR_n,
  int tensorR_c,
  int tensorR_w,
  int A_cstride_is0,
  int if_fill_const,
  u32 fill_const_val,
  int limit_enable,
  PREC B_prec,
  PREC R_prec,
  SG_OP op);

/* PE_S_mask_select: do not support bank confilict
 * PE_S_mask_selhzd: support bank confilict
 * A, B, R are aligned in local memory,
 * if A_cstride is 0, A_wstride=1;A_hstride=ceil(w,EU_NUM),cstride=0;nstride=0
 * B support uint8/uint16/uint32
 * and mask_num is compacted in local mem which only support uint16
 * A=[1,C,1,A_w], B=[N, C, 1, A_w], R=[N,C,1,R_w], mask_num=[N,C,1,1]
 */
void atomic_pes_mask_sel_check(
  u32 tensorA_addr,
  u32 tensorB_addr,
  u32 tensorR_addr,
  u32 mask_num_addr,
  int tensorA_w,
  int tensorB_n,
  int tensorB_c,
  int A_cstride_is0,
  PREC B_prec,
  PREC R_prec,
  SG_OP op);

/* PE_S_nonsero: do not support bank confilict
 * PE_S_nonzero_hzd: support bank confilict
 * A, R are aligned in local memory,
 * A support INT8/INT16/INT32, R support INT16/INT32
 * and mask_num is compacted in local mem which only support uint16
 * A=[N,C,1,W], R=[N,C,1,W],mask_num=[N,C,1,1]
 */
void atomic_pes_nonzero_check(
  u32 tensorA_addr,
  u32 tensorR_addr,
  u32 mask_num_addr,
  int tensorA_n,
  int tensorA_c,
  int tensorA_w,
  PREC A_prec,
  PREC R_prec,
  SG_OP op);

/*
 param  n : length of taylor table for TAYLOR or number of iterations of Newton's algorithm for RSQRT
*/
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
    PREC    opd0_prec);

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
    int     use_default_tcgr);

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
  ROUND_MODE f2i_mode);

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
  ROUND_MODE shift_rd);

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
  short zp_value, // S8/U8/S16/U16
  int A_sign,
  int R_sign,
  PREC A_prec,
  PREC R_prec,
  ROUND_MODE i2f_mode);

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
  ROUND_MODE i2f_mode);

void atomic_dq_i32mode_check(
  u32 A_addr,
  u32 B_addr,
  u32 R_addr,
  int N,
  int C,
  int H,
  int W,
  int B_is_const,
  short zp_value, // S8/U8/S16/U16
  int scale_factor, // S8/S16/S32
  char shift_num, // negative: right shift, positive: left shift
  int A_sign,
  int R_sign,
  int sym_range,
  PREC A_prec,
  PREC R_prec,
  ROUND_MODE shift_rd);

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
  PREC R_prec);

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
  int saturate);

#ifdef __cplusplus
}
#endif

#endif  /* ATOMIC_TIU_H */
