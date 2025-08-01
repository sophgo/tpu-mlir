#include "atomic_pooling_depthwise.h"

void atomic_max_min_pooling_check(
    u32 input_addr,
    u32 pad_ins_addr, // if pad_ins_is_const, store pad_value
    u32 output_addr,
    u32 index_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int ins_h,
    int ins_w,
    int dh,
    int dw,
    int pad_h_t,
    int pad_h_b,
    int pad_w_l,
    int pad_w_r,
    int pad_ins_is_const,
    int ins_const_val,
    int input_sign,
    PREC input_prec,
    PREC out_index_prec,
    PAD_MODE pad_mode,
    int do_relu,
    int saturate,
    PD_OP pool_op) {

#ifdef USING_CMODEL
  // compute the output_h, output_w
  int kh_ext = dh * (kh - 1) + 1;
  int kw_ext = dw * (kw - 1) + 1;
  int ih_ext = (input_h - 1) * (ins_h + 1) + pad_h_t + pad_h_b + 1;
  int iw_ext = (input_w - 1) * (ins_w + 1) + pad_w_l + pad_w_r + 1;
  int output_h = (ih_ext - kh_ext) / stride_h + 1;
  int output_w = (iw_ext - kw_ext) / stride_w + 1;
  u32 start_npu_idx = input_addr / LOCAL_MEM_SIZE;
  ASSERT(output_addr / LOCAL_MEM_SIZE == start_npu_idx);
  ASSERT((index_addr == 0xFFFFFFFF) || (index_addr / LOCAL_MEM_SIZE == start_npu_idx));
  ASSERT(input_addr % ALIGN_BYTES == 0);
  ASSERT(output_addr % ALIGN_BYTES == 0);
  ASSERT((index_addr == 0xFFFFFFFF) || (index_addr % ALIGN_BYTES == 0));
  ASSERT(input_prec == INT8 || input_prec == FP16 || input_prec == FP32 || input_prec == BFP16 || input_prec == FP8);
  ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
  ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
  ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
  ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
  ASSERT(output_h < (((int)1) << 16) && (output_h > 0));
  ASSERT(output_w < (((int)1) << 16) && (output_w > 0));
  ASSERT(kh < (((int)1) << 16) && (kh > 0));
  ASSERT(kw < (((int)1) << 16) && (kw > 0));
  ASSERT(stride_h > 0 && stride_h < 16);
  ASSERT(stride_w > 0 && stride_w < 16);
  ASSERT(ins_h >= 0 && ins_h < 8);
  ASSERT(ins_w >= 0 && ins_w < 8);
  ASSERT(dh > 0 && dh < 16);
  ASSERT(dw > 0 && dw < 16);
  ASSERT(pad_h_t >= 0 && pad_h_t < 16);
  ASSERT(pad_h_b >= 0 && pad_h_b < 16);
  ASSERT(pad_w_r >= 0 && pad_w_r < 16);
  ASSERT(pad_w_l >= 0 && pad_w_l < 16);
  ASSERT(do_relu == 0 || do_relu == 1);
#ifdef __sg2262__
  ASSERT(pad_ins_is_const == 1);
  ASSERT(input_prec == INT8 || input_prec == INT16 || input_prec == INT32 ||input_prec == FP16 || input_prec == FP32 || input_prec == BFP16 || input_prec == FP8);
  if (input_prec == INT8 || input_prec == INT16 || input_prec == INT32) {
    ASSERT(saturate == 0);
  } else {
    ASSERT(saturate == 0 || saturate == 1);
  }
#else
  ASSERT(pad_ins_is_const || pad_ins_addr / LOCAL_MEM_SIZE == start_npu_idx);
  ASSERT(pad_ins_is_const || pad_ins_addr % (2 * get_bytesize(input_prec)) == 0);
  ASSERT(input_prec == INT8 || input_prec == FP16 || input_prec == FP32 || input_prec == BFP16 || input_prec == FP8);
#endif
  ASSERT(pool_op == PD_MIN_POOLING || pool_op == PD_MAX_POOLING);
#endif
}

//only for float
void atomic_avg_pooling_check(
    u32 input_addr,
    u32 pad_ins_addr, // if pad_ins_is_const, store pad_value
    u32 rq_addr,
    u32 output_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int ins_h,
    int ins_w,
    int dh,
    int dw,
    int pad_h_t,
    int pad_h_b,
    int pad_w_l,
    int pad_w_r,
    int pad_ins_is_const,
    int kernel_const_val,
    int ins_const_val,
    int do_relu,
    int do_rq,
    int rq_is_const,
    int sym_range,
    float re_scale,
    PREC input_prec,
    PREC output_prec,
    FP8_TYPE input_fp8_prec,
    FP8_TYPE kernel_fp8_prec,
    FP8_TYPE output_fp8_prec,
    PAD_MODE pad_mode,
    ROUND_MODE round_mode) {

#ifdef USING_CMODEL
  // compute the output_h, output_w
  int kh_ext = dh * (kh - 1) + 1;
  int kw_ext = dw * (kw - 1) + 1;
  int ih_ext = (input_h - 1) * (ins_h + 1) + pad_h_t + pad_h_b + 1;
  int iw_ext = (input_w - 1) * (ins_w + 1) + pad_w_l + pad_w_r + 1;
  int output_h = (ih_ext - kh_ext) / stride_h + 1;
  int output_w = (iw_ext - kw_ext) / stride_w + 1;
  u32 start_npu_idx = input_addr / LOCAL_MEM_SIZE;
  ASSERT(output_addr / LOCAL_MEM_SIZE == start_npu_idx);
  ASSERT(input_addr % ALIGN_BYTES == 0);
  ASSERT(output_addr % ALIGN_BYTES == 0);
  ASSERT(input_prec == FP16 || input_prec == FP32 || input_prec == BFP16 || input_prec == FP8);
  ASSERT(output_prec == FP16 || output_prec == FP32 || output_prec == BFP16 || output_prec == FP8);
#ifdef __sg2262__
  ASSERT(pad_ins_is_const == 1);
#else
  ASSERT(pad_ins_is_const || pad_ins_addr / LOCAL_MEM_SIZE == start_npu_idx);
  ASSERT(pad_ins_is_const || pad_ins_addr % (2 * get_bytesize(input_prec)) == 0);
#endif
  if (input_prec == FP8) {
#ifdef __sg2262__
      ASSERT((do_rq && (output_prec == FP16 || output_prec == FP8 || output_prec == BFP16)) ||
             (!do_rq && (output_prec == FP16 || output_prec == FP32 || output_prec == BFP16)));
#else
      ASSERT((do_rq && (output_prec == FP16 || output_prec == FP8)) ||
             (!do_rq && (output_prec == FP16 || output_prec == FP32)));
#endif
  } else {
      ASSERT(output_prec == FP32 || output_prec == input_prec);
      ASSERT(do_rq == 0);
  }
  if (do_rq && !rq_is_const) {
    ASSERT(rq_addr / LOCAL_MEM_SIZE == start_npu_idx);
    ASSERT(rq_addr % get_bytesize(INT32) == 0);
  }
  ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
  ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
  ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
  ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
  ASSERT(output_h < (((int)1) << 16) && (output_h > 0));
  ASSERT(output_w < (((int)1) << 16) && (output_w > 0));
  ASSERT(kh < (((int)1) << 16) && (kh > 0));
  ASSERT(kw < (((int)1) << 16) && (kw > 0));
  ASSERT(stride_h > 0 && stride_h < 16);
  ASSERT(stride_w > 0 && stride_w < 16);
  ASSERT(ins_h >= 0 && ins_h < 8);
  ASSERT(ins_w >= 0 && ins_w < 8);
  ASSERT(dh > 0 && dh < 16);
  ASSERT(dw > 0 && dw < 16);
  ASSERT(pad_h_t >= 0 && pad_h_t < 16);
  ASSERT(pad_h_b >= 0 && pad_h_b < 16);
  ASSERT(pad_w_r >= 0 && pad_w_r < 16);
  ASSERT(pad_w_l >= 0 && pad_w_l < 16);
  ASSERT(do_relu == 0 || do_relu == 1);
  ASSERT(do_rq == 0 || do_rq == 1);
  ASSERT(sym_range == 0 || sym_range == 1);
  ASSERT(pad_mode >= 0 && pad_mode < (((int)1) << 2));
  ASSERT(round_mode == ROUND_HALF_TO_EVEN);
  ASSERT(input_fp8_prec == 0 || input_fp8_prec == 1);
  ASSERT(kernel_fp8_prec == 0 || kernel_fp8_prec == 1);
  ASSERT(output_fp8_prec == 0 || output_fp8_prec == 1);
#endif
}

//for fixed
void atomic_avg_pooling_fixed_check(
    u32 input_addr,
    u32 pad_ins_addr, // if pad_ins_is_const, store pad_value
    u32 output_addr,
    u32 rq_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int ins_h,
    int ins_w,
    int dh,
    int dw,
    int pad_h_t,
    int pad_h_b,
    int pad_w_l,
    int pad_w_r,
    int pad_ins_is_const,
    int kernel_const_val,
    int ins_const_val,
    int input_sign,
    int output_sign,
    int kernel_sign,
    int do_relu,
    int do_rq,
    int rq_is_const,
    int sym_range,
    int mul,
    s8  shift,
    s16 yzp,
    ROUND_MODE round_mode,
    PREC input_prec,
    PREC output_prec,
    PAD_MODE pad_mode) {

#ifdef USING_CMODEL
  // compute the output_h, output_w
  int kh_ext = dh * (kh - 1) + 1;
  int kw_ext = dw * (kw - 1) + 1;
  int ih_ext = (input_h - 1) * (ins_h + 1) + pad_h_t + pad_h_b + 1;
  int iw_ext = (input_w - 1) * (ins_w + 1) + pad_w_l + pad_w_r + 1;
  int output_h = (ih_ext - kh_ext) / stride_h + 1;
  int output_w = (iw_ext - kw_ext) / stride_w + 1;
  u32 start_npu_idx = input_addr / LOCAL_MEM_SIZE;
  ASSERT(output_addr / LOCAL_MEM_SIZE == start_npu_idx);
  if (do_rq && !rq_is_const) ASSERT(rq_addr / LOCAL_MEM_SIZE == start_npu_idx);
  ASSERT(input_addr % ALIGN_BYTES == 0);
#ifdef __sg2262__
  ASSERT(pad_ins_is_const == 1);
  ASSERT(sym_range == 0);
#else
  ASSERT(pad_ins_is_const == 0 || pad_ins_is_const == 1);
  ASSERT(pad_ins_is_const || pad_ins_addr / LOCAL_MEM_SIZE == start_npu_idx);
  ASSERT(pad_ins_is_const || pad_ins_addr % (2 * get_bytesize(input_prec)) == 0);
  ASSERT(sym_range == 0 || sym_range == 1);
#endif
  ASSERT(output_addr % ALIGN_BYTES == 0);
  if (do_rq && !rq_is_const) ASSERT(rq_addr % (2 * get_bytesize(INT32)) == 0);
  ASSERT(input_prec == INT8);
  ASSERT(output_prec == INT8 || output_prec == INT16 || output_prec == INT32);
  ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
  ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
  ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
  ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
  ASSERT(output_h < (((int)1) << 16) && (output_h > 0));
  ASSERT(output_w < (((int)1) << 16) && (output_w > 0));
  ASSERT(kh < (((int)1) << 16) && (kh > 0));
  ASSERT(kw < (((int)1) << 16) && (kw > 0));
  ASSERT(stride_h > 0 && stride_h < 16);
  ASSERT(stride_w > 0 && stride_w < 16);
  ASSERT(ins_h >= 0 && ins_h < 8);
  ASSERT(ins_w >= 0 && ins_w < 8);
  ASSERT(dh > 0 && dh < 16);
  ASSERT(dw > 0 && dw < 16);
  ASSERT(pad_h_t >= 0 && pad_h_t < 16);
  ASSERT(pad_h_b >= 0 && pad_h_b < 16);
  ASSERT(pad_w_r >= 0 && pad_w_r < 16);
  ASSERT(pad_w_l >= 0 && pad_w_l < 16);
  ASSERT(input_sign == 0 || input_sign == 1);
  ASSERT(output_sign == 0 || output_sign == 1);
  ASSERT(kernel_sign == 0 || kernel_sign == 1);
  ASSERT(do_relu == 0 || do_relu == 1);
  ASSERT(do_rq == 0 || do_rq == 1);
  ASSERT(rq_is_const == 0 || rq_is_const == 1);
  ASSERT(round_mode < 7 && round_mode >= 0);
  ASSERT(pad_mode >= 0 && pad_mode < (((int)1) << 2));
#endif
}

void atomic_depthwise_check(
    u32 input_addr,
    u32 weight_addr,  // if kernel_is_const, store weight value
    u32 bias_addr,    // if bias_is_const, store bias value
    u32 pad_ins_addr, // if pad_ins_is_const, store pad_value
    u32 rq_addr,
    u32 output_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int ins_h,
    int ins_w,
    int dh,
    int dw,
    int pad_h_t,
    int pad_h_b,
    int pad_w_l,
    int pad_w_r,
    int kernel_is_const,
    int bias_is_const,
    int pad_ins_is_const,
    int ins_const_val,
    int kernel_rotate,
    int do_relu,
    int saturate,
    int do_rq,
    int rq_is_const,
    PREC in_prec,
    PREC out_prec,
    FP8_TYPE input_type,
    FP8_TYPE kernel_type,
    FP8_TYPE res_type,
    PAD_MODE pad_mode)
{
#ifdef USING_CMODEL
      // compute the output_h, output_w
      int kh_ext = dh * (kh - 1) + 1;
      int kw_ext = dw * (kw - 1) + 1;
      int ih_ext = (input_h - 1) * (ins_h + 1) + pad_h_t + pad_h_b + 1;
      int iw_ext = (input_w - 1) * (ins_w + 1) + pad_w_l + pad_w_r + 1;
      int output_h = (ih_ext - kh_ext) / stride_h + 1;
      int output_w = (iw_ext - kw_ext) / stride_w + 1;
      int start_npu_idx = get_npu_index(input_addr);
      ASSERT(kernel_is_const || get_npu_index(weight_addr) == start_npu_idx);
      ASSERT(bias_is_const || get_npu_index(bias_addr) == start_npu_idx);
#ifdef __sg2262__
      ASSERT(pad_ins_is_const == 1);
#else
      ASSERT(pad_ins_is_const || get_npu_index(pad_ins_addr) == start_npu_idx);
      ASSERT(pad_ins_is_const || pad_ins_addr % (2 * get_bytesize(in_prec)) == 0);
      ASSERT(pad_ins_is_const >= 0 && pad_ins_is_const < 2);
#endif
      ASSERT(get_npu_index(output_addr) == start_npu_idx);
      ASSERT(input_addr % ALIGN_BYTES == 0);
      ASSERT(kernel_is_const || weight_addr % get_bytesize(in_prec) == 0);
      if (in_prec == FP8) {
#ifdef __sg2262__
          ASSERT(bias_is_const || (do_rq && (bias_addr % sizeof(float)) == 0) ||
                (!do_rq && (bias_addr % get_bytesize(out_prec)) == 0));
#else
          ASSERT(bias_is_const || (do_rq && (bias_addr % sizeof(float)) == 0)
                  || (!do_rq && (bias_addr % get_bytesize(out_prec)) == 0));
          ASSERT((!do_rq && (out_prec == FP16 || out_prec == FP32)) ||
                  (do_rq && (out_prec == FP16 || out_prec == FP8)));
          if (do_rq && !rq_is_const) {
                ASSERT(rq_addr % sizeof(float) == 0 && get_npu_index(rq_addr) == start_npu_idx);
          }
#endif
      } else {
            ASSERT(do_rq == 0);
            ASSERT(bias_is_const || bias_addr % get_bytesize(out_prec) == 0);
      }
      ASSERT(is_float_prec(in_prec) && is_float_prec(out_prec));
      ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
      ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
      ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
      ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
      ASSERT(output_h < (((int)1) << 16) && (output_h > 0));
      ASSERT(output_w < (((int)1) << 16) && (output_w > 0));
      ASSERT(kh < (((int)1) << 16) && (kh > 0));
      ASSERT(kw < (((int)1) << 16) && (kw > 0));
      ASSERT(stride_h > 0 && stride_h < 16);
      ASSERT(stride_w > 0 && stride_w < 16);
      ASSERT(ins_h >= 0 && ins_h < 8);
      ASSERT(ins_w >= 0 && ins_w < 8);
      ASSERT(dh > 0 && dh < 16);
      ASSERT(dw > 0 && dw < 16);
      ASSERT(pad_h_t >= 0 && pad_h_t < 16);
      ASSERT(pad_h_b >= 0 && pad_h_b < 16);
      ASSERT(pad_w_r >= 0 && pad_w_r < 16);
      ASSERT(pad_w_l >= 0 && pad_w_l < 16);
      ASSERT(kernel_is_const >= 0 && kernel_is_const < 2);
      ASSERT(bias_is_const >= 0 && bias_is_const < 2);
      ASSERT(kernel_rotate >= 0 && kernel_rotate < 2);
      ASSERT(do_relu >= 0 && do_relu < 2);
      ASSERT(do_rq >=0 && do_rq < 2);
      ASSERT(rq_is_const >=0 && rq_is_const < 2);
      ASSERT(input_type >=0 && input_type < 2);
      ASSERT(kernel_type >= 0 && kernel_type < 2);
      ASSERT(res_type >= 0 && res_type < 2);
#endif
}

void atomic_depthwise_quant_check(
    u32 input_addr,
    u32 weight_addr, // if kernel_is_const, store weight value
    u32 bias_addr, // if bias_is_const, store bias value
    u32 pad_ins_addr, // if pad_ins_is_const, store pad_value
    u32 requant_addr,
    u32 output_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int ins_h,
    int ins_w,
    int dh,
    int dw,
    int pad_h_t,
    int pad_h_b,
    int pad_w_l,
    int pad_w_r,
    int kernel_is_const,
    int bias_is_const,
    int pad_ins_is_const,
    int ins_const_val,
    int kernel_rotate,
    int input_sign,
    int weight_sign,
    int bias_sign,
    int output_sign,
    int do_relu,
    int sym_saturate,
    int do_requant,
    int requant_is_const,
    int shift_num, // s8
    int ozp, // s16
    ROUND_MODE rm_mode,
    PREC input_prec,
    PREC output_prec,
    PAD_MODE pad_mode) {

#ifdef USING_CMODEL
  // compute the output_h, output_w
  int kh_ext = dh * (kh - 1) + 1;
  int kw_ext = dw * (kw - 1) + 1;
  int ih_ext = (input_h - 1) * (ins_h + 1) + pad_h_t + pad_h_b + 1;
  int iw_ext = (input_w - 1) * (ins_w + 1) + pad_w_l + pad_w_r + 1;
  int output_h = (ih_ext - kh_ext) / stride_h + 1;
  int output_w = (iw_ext - kw_ext) / stride_w + 1;
#ifndef __sg2262__
  if (bias_is_const && bias_addr == 0) {
      bias_sign = 0;
  }
#endif
  int start_npu_idx = get_npu_index(input_addr);
  ASSERT(kernel_is_const || get_npu_index(weight_addr) == start_npu_idx);
  ASSERT(bias_is_const || get_npu_index(bias_addr) == start_npu_idx);
#ifdef __sg2262__
  ASSERT(pad_ins_is_const  == 1);
  ASSERT(bias_sign == 1 && bias_is_const < 2);
  ASSERT(sym_saturate == 0);
#else
  ASSERT(pad_ins_is_const || get_npu_index(pad_ins_addr) == start_npu_idx);
  ASSERT(pad_ins_is_const || pad_ins_addr % (2 * get_bytesize(input_prec)) == 0);
  ASSERT(pad_ins_is_const >= 0 && pad_ins_is_const < 2);
  ASSERT(bias_sign >= 0 && bias_is_const < 2);
  ASSERT(sym_saturate >= 0 && sym_saturate < 2);
#endif
  ASSERT(!do_requant || requant_is_const || get_npu_index(requant_addr) == start_npu_idx);
  ASSERT(get_npu_index(output_addr) == start_npu_idx);
  ASSERT(input_addr % ALIGN_BYTES == 0 && output_addr % ALIGN_BYTES == 0);
  ASSERT(kernel_is_const || weight_addr % get_bytesize(input_prec) == 0);
  ASSERT(bias_is_const || bias_addr % sizeof(int) == 0);
  ASSERT(!do_requant || requant_is_const || requant_addr % (sizeof(int) * 2) == 0);
  ASSERT(input_prec == INT8);
  ASSERT(output_prec == INT8 || output_prec == INT16 || output_prec == INT32);
  ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
  ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
  ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
  ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
  ASSERT(output_h < (((int)1) << 16) && (output_h > 0));
  ASSERT(output_w < (((int)1) << 16) && (output_w > 0));
  ASSERT(kh < (((int)1) << 16) && (kh > 0));
  ASSERT(kw < (((int)1) << 16) && (kw > 0));
  ASSERT(stride_h > 0 && stride_h < 16);
  ASSERT(stride_w > 0 && stride_w < 16);
  ASSERT(ins_h >= 0 && ins_h < 8);
  ASSERT(ins_w >= 0 && ins_w < 8);
  ASSERT(dh > 0 && dh < 16);
  ASSERT(dw > 0 && dw < 16);
  ASSERT(pad_h_t >= 0 && pad_h_t < 16);
  ASSERT(pad_h_b >= 0 && pad_h_b < 16);
  ASSERT(pad_w_r >= 0 && pad_w_r < 16);
  ASSERT(pad_w_l >= 0 && pad_w_l < 16);
  ASSERT(kernel_is_const >= 0 && kernel_is_const < 2);
  ASSERT(bias_is_const >= 0 && bias_is_const < 2);
  ASSERT(kernel_rotate >= 0 && kernel_rotate < 2);
  ASSERT(input_sign >= 0 && input_sign < 2);
  ASSERT(weight_sign >= 0 && weight_sign < 2);
  ASSERT(output_sign >= 0 && output_sign < 2);
  ASSERT(do_relu >= 0 && do_relu < 2);
  ASSERT(do_requant >= 0 && do_requant < 2);
  ASSERT(requant_is_const >= 0 && requant_is_const < 2);
  ASSERT(shift_num >= -128 && shift_num < 128);
  ASSERT(ozp >= -32768 && ozp < 32768);
#endif
}

void atomic_roi_max_min_pooling_check(
    u32 input_addr,
    u32 roi_addr, // roi pairs
    u32 output_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int output_w,
    int kh,
    int kw,
    int imm_const_val,
    int input_sign,
    PREC input_prec,
    int do_relu,
    PD_OP pool_op) {

#ifdef USING_CMODEL
  PREC output_prec = input_prec;
  u32 start_npu_idx = input_addr / LOCAL_MEM_SIZE;
  ASSERT(roi_addr / LOCAL_MEM_SIZE == 0);
  ASSERT(output_addr / LOCAL_MEM_SIZE == start_npu_idx);
  ASSERT(input_addr % ALIGN_BYTES == 0);
  ASSERT(roi_addr % (4 * get_bytesize(INT16)) == 0);
  ASSERT(output_addr % ALIGN_BYTES == 0);
  ASSERT(input_sign == 0 || input_sign == 1);
  ASSERT(input_prec == INT8 || input_prec == FP8 ||  input_prec == FP16 || input_prec == FP32 || input_prec == BFP16);
  ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
  ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
  ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
  ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
  ASSERT(output_w < (((int)1) << 16) && (output_w > 0));
  ASSERT(kh < (((int)1) << 16) && (kh > 0));
  ASSERT(kw < (((int)1) << 16) && (kw > 0));
  ASSERT(do_relu == 0 || do_relu == 1);
  ASSERT(pool_op == PD_ROI_MAX_POOLING || pool_op == PD_ROI_MIN_POOLING);
#endif
}

//only for float
void atomic_roi_avg_pooling_check(
    u32 input_addr,
    u32 roi_addr, // roi pairs
    u32 output_addr,
    u32 rq_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int output_w,
    int kh,
    int kw,
    int kernel_const_val,
    int imm_const_val,
    int do_rq,
    int rq_is_const,
    float re_scale,
    int sym_range,
    int do_relu,
    FP8_TYPE input_fp8_type,
    FP8_TYPE kernel_fp8_type,
    FP8_TYPE res_fp8_type,
    PREC input_prec,
    PREC output_prec,
    ROUND_MODE round_mode) {

#ifdef USING_CMODEL
  u32 start_npu_idx = input_addr / LOCAL_MEM_SIZE;
  ASSERT(roi_addr / LOCAL_MEM_SIZE == 0);
  ASSERT(output_addr / LOCAL_MEM_SIZE == start_npu_idx);
  ASSERT(input_addr % ALIGN_BYTES == 0);
  ASSERT(roi_addr % (4 * get_bytesize(INT16)) == 0);
  ASSERT(output_addr % ALIGN_BYTES == 0);
#ifndef __sg2262__
  if (do_rq && !rq_is_const) {
      ASSERT(rq_addr / LOCAL_MEM_SIZE == start_npu_idx);
      ASSERT(rq_addr % get_bytesize(INT32) == 0);
  }
#endif
  ASSERT(is_float_prec(input_prec));
  ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
  ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
  ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
  ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
  ASSERT(output_w < (((int)1) << 16) && (output_w > 0));
  ASSERT(kh < (((int)1) << 16) && (kh > 0));
  ASSERT(kw < (((int)1) << 16) && (kw > 0));
  ASSERT(do_relu == 0 || do_relu == 1);
  ASSERT(sym_range == 0 || sym_range == 1);
#ifdef __sg2262__
  ASSERT(input_fp8_type == 0 || input_fp8_type == 1);
  ASSERT(kernel_fp8_type == 0 || kernel_fp8_type == 1);
  if (input_prec == FP8) {
    ASSERT(output_prec == FP32);
  } else {
    ASSERT(input_prec == output_prec);
  }
#else
  ASSERT(do_rq == 0 || do_rq == 1);
  ASSERT(rq_is_const == 0 || rq_is_const == 1);
  ASSERT(round_mode == ROUND_HALF_TO_EVEN);
  ASSERT(input_fp8_type == 0 || input_fp8_type == 1);
  ASSERT(kernel_fp8_type == 0 || kernel_fp8_type == 1);
  ASSERT(res_fp8_type == 0 || res_fp8_type == 1);
  if (input_prec == FP8) {
      ASSERT((do_rq && (output_prec == FP16 || output_prec == FP8)) ||
             (!do_rq && (output_prec == FP16 || output_prec == FP32)));
  } else {
      ASSERT(output_prec == FP32 || output_prec == input_prec);
      ASSERT(do_rq == 0);
  }
#endif
#endif
}

//for fixed
void atomic_roi_avg_pooling_quant_check(
    u32 input_addr,
    u32 roi_addr, // roi pairs
    u32 output_addr,
    u32 rq_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int output_w,
    int kh,
    int kw,
    int kernel_const_val,
    int imm_const_val,
    int input_sign,
    int output_sign,
    int kernel_sign,
    PREC input_prec,
    PREC output_prec,
    int do_relu,
    int do_rq,
    int rq_is_const,
    int mul,
    s8 shift,
    s16 yzp,
    int sym_range,
    ROUND_MODE round_mode) {

#ifdef USING_CMODEL
  u32 start_npu_idx = input_addr / LOCAL_MEM_SIZE;
  ASSERT(roi_addr / LOCAL_MEM_SIZE == 0);
  ASSERT(output_addr / LOCAL_MEM_SIZE == start_npu_idx);
  ASSERT(input_addr % ALIGN_BYTES == 0);
  ASSERT(roi_addr % (4 * get_bytesize(INT16)) == 0);
  ASSERT(output_addr % ALIGN_BYTES == 0);
#ifdef __sg2262__
  ASSERT(do_rq == 0);
#else
  if (do_rq && !rq_is_const) {
      ASSERT(rq_addr / LOCAL_MEM_SIZE == start_npu_idx);
      ASSERT(rq_addr % (2 * get_bytesize(INT32)) == 0);
  }
#endif
  ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
  ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
  ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
  ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
  ASSERT(output_w < (((int)1) << 16) && (output_w > 0));
  ASSERT(kh < (((int)1) << 16) && (kh > 0));
  ASSERT(kw < (((int)1) << 16) && (kw > 0));
  ASSERT(input_sign == 0 || input_sign == 1);
  ASSERT(output_sign == 0 || output_sign == 1);
  ASSERT(kernel_sign == 0 || kernel_sign == 1);
  ASSERT(input_prec == INT8);
  ASSERT(do_relu == 0 || do_relu == 1);
#ifdef __sg2262__
  ASSERT(output_prec == INT32);
  ASSERT(rq_is_const == 1);
  ASSERT(sym_range == 0);
#else
  ASSERT(output_prec == INT8 || output_prec == INT16 || output_prec == INT32);
  ASSERT(do_relu == 0 || do_relu == 1);
  ASSERT(rq_is_const == 0 || rq_is_const == 1);
  ASSERT(sym_range == 0 || sym_range == 1);
#endif
  ASSERT(round_mode < 7 && round_mode >= 0);
#endif
}

void atomic_roi_depthwise_check(
    u32 input_addr,
    u32 weight_addr, // if kernel_is_const store weight value
    u32 roi_addr, // roi pairs
    u32 rq_addr,
    u32 output_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int output_w,
    int kh,
    int kw,
    int imm_const_val,
    int kernel_is_const,
    int kernel_rotate,
    int do_relu,
    int do_requant,
    int rq_is_const,
    PREC in_prec,
    PREC out_prec,
    FP8_TYPE in_type,
    FP8_TYPE kernel_type,
    FP8_TYPE res_type) {

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  int start_npu_idx = input_addr / LOCAL_MEM_SIZE;
  ASSERT(kernel_is_const || get_npu_index(weight_addr) == start_npu_idx);
  ASSERT(roi_addr / LOCAL_MEM_SIZE == 0);
  ASSERT(get_npu_index(output_addr) == start_npu_idx);
  ASSERT(input_addr % ALIGN_BYTES == 0);
  ASSERT(kernel_is_const || weight_addr % get_bytesize(in_prec) == 0);
  ASSERT(roi_addr % (4 * get_bytesize(INT16)) == 0);
  ASSERT(output_addr % ALIGN_BYTES == 0);
  ASSERT(is_float_prec(in_prec) && is_float_prec(out_prec));
  ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
  ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
  ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
  ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
  ASSERT(output_w < (((int)1) << 16) && (output_w > 0));
  ASSERT(kh < (((int)1) << 16) && (kh > 0));
  ASSERT(kw < (((int)1) << 16) && (kw > 0));
  ASSERT(kernel_is_const >= 0 && kernel_is_const < 2);
  ASSERT(kernel_rotate >= 0 && kernel_rotate < 2);
  ASSERT(do_relu >= 0 && do_relu < 2);
  ASSERT(do_requant == 0 || do_requant == 1);
  ASSERT(rq_is_const == 0 || rq_is_const == 1);
  ASSERT(in_type == 0 || in_type == 1);
  ASSERT(kernel_type == 0 || kernel_type == 1);
  ASSERT(res_type == 0 || res_type == 1);
  if (in_prec == FP8) {
      ASSERT((do_requant && (out_prec == FP16 || out_prec == FP8)) ||
             (!do_requant && (out_prec == FP16 || out_prec == FP32)));
      if (do_requant && !rq_is_const) {
            ASSERT(get_npu_index(rq_addr) == start_npu_idx && rq_addr % sizeof(float) == 0);
      }
  } else {
      ASSERT(out_prec == FP32 || out_prec == in_prec);
      ASSERT(do_requant == 0);
  }
#endif
#endif
}

void atomic_roi_depthwise_quant_check(
    u32 input_addr,
    u32 weight_addr, // if kernel_is_const store weight value
    u32 roi_addr, // roi pairs
    u32 requant_addr,
    u32 output_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int output_w,
    int kh,
    int kw,
    int imm_const_val,
    int kernel_is_const,
    int kernel_rotate,
    int input_sign,
    int kernel_sign,
    int res_sign,
    int do_relu,
    int sym_saturate,
    int do_requant,
    int requant_is_const,
    int shift_num, // s8
    int ozp, // s16
    ROUND_MODE rm_mode,
    PREC input_prec,
    PREC output_prec) {

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  int start_npu_idx = input_addr / LOCAL_MEM_SIZE;
  ASSERT(kernel_is_const || get_npu_index(weight_addr) == start_npu_idx);
  ASSERT(roi_addr / LOCAL_MEM_SIZE == 0);
  ASSERT(!do_requant || requant_is_const || get_npu_index(requant_addr) == start_npu_idx);
  ASSERT(get_npu_index(output_addr) == start_npu_idx);
  ASSERT(input_addr % ALIGN_BYTES == 0);
  ASSERT(kernel_is_const || weight_addr % get_bytesize(input_prec) == 0);
  ASSERT(roi_addr % (4 * get_bytesize(INT16)) == 0);
  ASSERT(!do_requant || requant_is_const || requant_addr % (2 * sizeof(int)) == 0);
  ASSERT(output_addr % ALIGN_BYTES == 0);
  ASSERT(input_prec == INT8);
  ASSERT(output_prec == INT8 || output_prec == INT16 || output_prec == INT32);
  ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
  ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
  ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
  ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
  ASSERT(output_w < (((int)1) << 16) && (output_w > 0));
  ASSERT(kh < (((int)1) << 16) && (kh > 0));
  ASSERT(kw < (((int)1) << 16) && (kw > 0));
  ASSERT(kernel_is_const >= 0 && kernel_is_const < 2);
  ASSERT(kernel_rotate >= 0 && kernel_rotate < 2);
  ASSERT(do_relu >= 0 && do_relu < 2);
  ASSERT(sym_saturate >= 0 && sym_saturate < 2);
  ASSERT(do_requant >= 0 && do_requant < 2);
  ASSERT(requant_is_const >= 0 && requant_is_const < 2);
  ASSERT(shift_num >= -128 && shift_num < 128);
  ASSERT(ozp >= -32768 && ozp < 32768);
  ASSERT(input_sign >= 0 && input_sign < 2);
  ASSERT(kernel_sign >= 0 && kernel_sign < 2);
  ASSERT(res_sign >= 0 && res_sign < 2);
#endif
#endif
}
