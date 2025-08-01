#include "atomic_tiu.h"

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
  int do_relu)
{

#ifdef USING_CMODEL
#ifdef __sg2262__
    ASSERT_FS_INFO(0, "Not support");
#else
  int L_last_W = (is_L_trans ? L_row_num : L_col_num) % L_tensor_W;
  if (L_last_W == 0) L_last_W = L_tensor_W;
  if (!is_L_const) {
    ASSERT(L_addr % ALIGN_BYTES == 0);
  }
  ASSERT(R_addr % ALIGN_BYTES == 0);
  ASSERT(Y_addr % ALIGN_BYTES == 0);
  if (!is_bias_const) {
    ASSERT(bias_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(Y_addr) == get_npu_index(bias_addr));
  }
  ASSERT(get_npu_index(Y_addr) == get_npu_index(R_addr));
  ASSERT(add_result < (1 << 1) && add_result >= 0);
  ASSERT(is_L_trans < (1 << 1) && is_L_trans >= 0);
  ASSERT(is_L_const < (1 << 1) && is_L_const >= 0);
  ASSERT(R_tensor_C < (1 << 16) && R_tensor_C > 0);
  ASSERT(R_tensor_W < (1 << 16) && R_tensor_W > 0);
  ASSERT(L_row_num < (1 << 16) && L_row_num > 0);
  ASSERT(L_tensor_C < (1 << 16) && L_tensor_C > 0);
  ASSERT(L_tensor_W < (1 << 16) && L_tensor_W > 0);
  ASSERT(L_col_num < (1 << 16) && L_col_num > 0);
  ASSERT(L_last_W < (1 << 16) && L_last_W > 0);
  ASSERT(do_relu < (1 << 1) && do_relu >= 0);
#endif
#endif
}

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
  ROUND_MODE round_mode)
{

#ifdef USING_CMODEL
#ifdef __sg2262__
    ASSERT_FS_INFO(0, "Not support");
#else
  int L_last_W = (is_L_trans ? L_row_num : L_col_num) % L_tensor_W;
  if (L_last_W == 0) L_last_W = L_tensor_W;
  if (is_bias_const && bias_addr == 0) bias_sign = 0;
  if (!is_L_const) {
    ASSERT(L_addr % ALIGN_BYTES == 0);
  }
  ASSERT(R_addr % ALIGN_BYTES == 0);
  ASSERT(Y_addr % ALIGN_BYTES == 0);
  if (!is_bias_const) {
    ASSERT(bias_addr % ALIGN_BYTES == 0);
    ASSERT(get_npu_index(bias_addr) == get_npu_index(Y_addr));
  }
    ASSERT(get_npu_index(R_addr) == get_npu_index(Y_addr));
  ASSERT(add_result < (1 << 1) && add_result >= 0);
  ASSERT(is_L_trans < (1 << 1) && is_L_trans >= 0);
  ASSERT(L_sign < (1 << 1) && L_sign >= 0);
  ASSERT(R_sign < (1 << 1) && R_sign >= 0);
  ASSERT(bias_sign < (1 << 1) && bias_sign >= 0);
  ASSERT(Res_sign < (1 << 1) && Res_sign >= 0);
  ASSERT(is_L_const < (1 << 1) && is_L_const >= 0);
  ASSERT(is_bias_const < (1 << 1) && is_bias_const >= 0);
  ASSERT(R_tensor_C < (1 << 16) && R_tensor_C > 0);
  ASSERT(R_tensor_W < (1 << 16) && R_tensor_W > 0);
  ASSERT(L_row_num < (1 << 16) && L_row_num > 0);
  ASSERT(L_tensor_C < (1 << 16) && L_tensor_C > 0);
  ASSERT(L_tensor_W < (1 << 16) && L_tensor_W > 0);
  ASSERT(L_col_num < (1 << 16) && L_col_num > 0);
  ASSERT(L_last_W < (1 << 16) && L_last_W > 0);
  ASSERT(if_relu == 0 || if_relu == 1);
  ASSERT(sym_range == 0 || sym_range == 1);
  ASSERT(do_rq == 0 || do_rq == 1);
  ASSERT(round_mode < 7 && round_mode >= 0);
  ASSERT(LR_prec == INT8 || LR_prec == INT16 || LR_prec == INT32);
  ASSERT(Y_prec == INT8 || Y_prec == INT16 || Y_prec == INT32);
#endif
#endif
}
