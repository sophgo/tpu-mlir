#include "atomic_tiu.h"
#include "atomic_conv_bw.h"

static int cal_conv2d_input_ext(int in, int insert, int pad_0, int pad_1) {
  return ((in - 1) * (insert + 1) + pad_0 + pad_1 + 1);
}

void atomic_conv_bw_check(
  u32 input_addr,   //opad0
  u32 grad_addr,    //opad1
  u32 pad_ins_addr, //opad2
  u32 res_addr,
  int n,
  int ic,
  int ih,
  int iw,
  int oc,
  int oh,
  int ow,
  int kh,
  int kw,
  int ins_h,
  int ins_w,
  int dh,
  int dw,
  int stride_h,
  int stride_w,
  int pad_h_t,
  int pad_h_b,
  int pad_w_l,
  int pad_w_r,
  int pad_ins_is_const,
  int result_add,
  u32 insert_const_val,
  int *input_stride,
  PAD_MODE pad_mode,
  PREC input_prec,
  PREC res_prec,
  FP8_TYPE input_fp8_type,
  FP8_TYPE grad_fp8_type,
  FP8_TYPE res_fp8_type) {

#ifdef USING_CMODEL
  if (input_stride) {
    ASSERT(input_addr % get_bytesize(input_prec) == 0);
  } else {
    ASSERT(input_addr % ALIGN_BYTES == 0);
  }

  ASSERT(get_npu_index(grad_addr) == get_npu_index(res_addr));
  ASSERT((grad_addr % ALIGN_BYTES == 0) && (res_addr % ALIGN_BYTES == 0));
#ifdef __sg2262__
  ASSERT(pad_ins_is_const == 1);
#else
  if (!pad_ins_is_const) {
    ASSERT(get_npu_index(pad_ins_addr) == get_npu_index(input_addr));
    ASSERT(pad_ins_addr % (get_bytesize(input_prec) * 2) == 0);
  }
  ASSERT(pad_ins_is_const == 0 || pad_ins_is_const == 1);
#endif
  ASSERT(n < (((int)1) << 16) && (n > 0));
  ASSERT(ic < (((int)1) << 16) && (ic > 0));
  ASSERT(ih < (((int)1) << 16) && (ih > 0));
  ASSERT(iw < (((int)1) << 16) && (iw > 0));
  int ih_ext = cal_conv2d_input_ext(ih, ins_h, pad_h_t, pad_h_b);
  int iw_ext = cal_conv2d_input_ext(iw, ins_w, pad_w_r, pad_w_l);
  ASSERT(ih_ext < (((int)1) << 16) && (ih_ext > 0));
  ASSERT(iw_ext < (((int)1) << 16) && (iw_ext > 0));
  ASSERT(oc < (((int)1) << 16) && (oc > 0));
  ASSERT(oh < (((int)1) << 16) && (oh > 0));
  ASSERT(ow < (((int)1) << 16) && (ow > 0));
  ASSERT(kh < (((int)1) << 16) && (kw > 0));
  ASSERT(kw < (((int)1) << 16) && (kw > 0));
  ASSERT(ins_h >= 0 && ins_h < 15);
  ASSERT(ins_w >= 0 && ins_w < 15);
  ASSERT(dh > 0 && dh < 16);
  ASSERT(dw > 0 && dw < 16);
  ASSERT(stride_h > 0 && stride_h < 16);
  ASSERT(stride_w > 0 && stride_w < 16);
  ASSERT(pad_h_t >= 0 && pad_h_t < 16);
  ASSERT(pad_h_b >= 0 && pad_h_b < 16);
  ASSERT(pad_w_r >= 0 && pad_w_r < 16);
  ASSERT(pad_w_l >= 0 && pad_w_l < 16);
  ASSERT(pad_mode >= 0 && pad_mode < 4);
  ASSERT((input_prec == FP32 || input_prec == TF32 || input_prec == FP16 || input_prec == BFP16 || input_prec == FP8) && res_prec == FP32);
  // only support FP32 output
  ASSERT(res_prec == FP32);
  ASSERT(input_fp8_type == FP8E5M2 || input_fp8_type == FP8E4M3);
  ASSERT(grad_fp8_type == FP8E5M2 || grad_fp8_type == FP8E4M3);
  ASSERT(res_fp8_type == 0 || res_fp8_type == 1);
  // do not support result_add
  ASSERT(result_add == 0);
#endif
}
