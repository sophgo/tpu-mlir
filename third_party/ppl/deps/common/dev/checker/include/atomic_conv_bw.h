#ifndef ATOMIC_CONV_BW_H
#define ATOMIC_CONV_BW_H

#include "checker_internel.h"

#ifdef __cplusplus
extern "C" {
#endif

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
  FP8_TYPE res_fp8_type
);

#ifdef __cplusplus
}
#endif
#endif  /* ATOMIC_CONV_BW_H */
