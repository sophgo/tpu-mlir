//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "tl_lrn"

namespace tpu_mlir {
namespace backend {
void cvi_backend_tl_lrn(uint32_t layer_id, laddr_t ifmap_laddr,
                        laddr_t ofmap_laddr, laddr_t sqr_lut_laddr,
                        laddr_t power_lut_laddr, laddr_t working_laddr,
                        int input_n, int input_c, int input_h, int input_w,
                        int size, int8_t sum_rshift_i8, int8_t lrn_rshift_i8,
                        int8_t *m_i8) {
  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "cvi_backend_tl_lrn:\n"
                 "    ifmap_laddr 0x%lx, ofmap_laddr 0x%lx"
                 "    power_lut_laddr 0x%lx, power_lut_laddr 0x%lx, "
                 "working_laddr 0x%lx"
                 "    in(%d, %d, %d, %d), size %d\n"
                 "    sum_right_shit_width %d, lrn_rshift_i8 %d\n",
                 ifmap_laddr, ofmap_laddr, sqr_lut_laddr, power_lut_laddr,
                 working_laddr, input_n, input_c, input_h, input_w, size,
                 sum_rshift_i8, lrn_rshift_i8));

  int move_counts = (size - 1) / 2;
  cvk_tl_shape_t lshape =
      CV18xx::tl_shape_t4(input_n, input_c, input_h, input_w);

  // cvk_tl_shape_t table_shape = {1, 32, 16, 16};
  cvk_tl_shape_t table_shape = CV18xx::tl_shape_t4(1, CV18xx::NPU_NUM, 16, 16);

  cvk_tl_t bottom;
  bottom.start_address = ifmap_laddr;
  bottom.fmt = CVK_FMT_U8;
  bottom.shape = lshape;
  bottom.stride = CV18xx::tl_default_stride(lshape, CVK_FMT_I8, 1);

  cvk_tl_t top;
  top.start_address = ofmap_laddr;
  top.fmt = CVK_FMT_U8;
  top.shape = lshape;
  top.stride = CV18xx::tl_default_stride(lshape, CVK_FMT_I8, 1);

  cvk_tl_t sqr_lut;
  sqr_lut.start_address = sqr_lut_laddr;
  sqr_lut.fmt = CVK_FMT_I8;
  sqr_lut.shape = table_shape;
  sqr_lut.stride = CV18xx::tl_default_stride(table_shape, CVK_FMT_I8, 1);

  cvk_tl_t pwr_lut;
  pwr_lut.start_address = power_lut_laddr;
  pwr_lut.fmt = CVK_FMT_I8;
  pwr_lut.shape = table_shape;
  pwr_lut.stride = CV18xx::tl_default_stride(table_shape, CVK_FMT_I8, 1);

  cvk_tl_t sum;
  sum.start_address = working_laddr;
  sum.fmt = CVK_FMT_U8;
  sum.shape = lshape;
  sum.stride = CV18xx::tl_default_stride(lshape, CVK_FMT_I8, 1);
  uint32_t sum_size =
      CV18xx::lmem_tensor_to_size(lshape.n, lshape.c, lshape.h, lshape.w);

  cvk_tl_t sum_high;
  sum_high.start_address = sum.start_address + sum_size; // after sum
  sum_high.fmt = CVK_FMT_U8;
  sum_high.shape = lshape;
  sum_high.stride = CV18xx::tl_default_stride(lshape, CVK_FMT_I8, 1);
  uint32_t sum_high_size =
      CV18xx::lmem_tensor_to_size(lshape.n, lshape.c, lshape.h, lshape.w);

  cvk_tl_t shift_sum;
  shift_sum.start_address =
      sum_high.start_address + sum_high_size; // after sum_high
  shift_sum.fmt = CVK_FMT_U8;
  shift_sum.shape = lshape;
  shift_sum.stride = CV18xx::tl_default_stride(lshape, CVK_FMT_I8, 1);
  uint32_t shift_sum_size =
      CV18xx::lmem_tensor_to_size(lshape.n, lshape.c, lshape.h, lshape.w);

  cvk_tl_t top_high;
  top_high.start_address =
      shift_sum.start_address + shift_sum_size; // after shift_sum
  top_high.fmt = CVK_FMT_U8;
  top_high.shape = lshape;
  top_high.stride = CV18xx::tl_default_stride(lshape, CVK_FMT_I8, 1);
  uint32_t top_high_size =
      CV18xx::lmem_tensor_to_size(lshape.n, lshape.c, lshape.h, lshape.w);

  cvk_tl_t top_high_high;
  top_high_high.start_address =
      top_high.start_address + top_high_size; // after top_high
  top_high_high.fmt = CVK_FMT_U8;
  top_high_high.shape = lshape;
  top_high_high.stride = CV18xx::tl_default_stride(lshape, CVK_FMT_I8, 1);
  uint32_t top_high_high_size =
      CV18xx::lmem_tensor_to_size(lshape.n, lshape.c, lshape.h, lshape.w);

  // Should no exceed local memory size
  assert((top_high_high.start_address + top_high_high_size) <=
         (uint32_t)(CV18xx::LMEM_BYTES));

  cvk_tiu_lookup_table_param_t p12 = {0};
  p12.ofmap = &top;
  p12.ifmap = &bottom;
  p12.table = &sqr_lut;
  p12.layer_id = layer_id;
  CV18xx::tiu_lookup_table(&p12);

  cvk_tiu_copy_param_t p11 = {0};
  p11.src = &top;
  p11.dst = &sum;
  p11.layer_id = layer_id;
  CV18xx::tiu_copy(&p11);
  CV18xx::tiu_zeros(layer_id, &sum_high); // sum_high initialize 0

  for (int step = 1; step <= move_counts && step < input_c; step++) {
    cvk_tdma_l2l_tensor_lrn_shift_param_t lrn_shift_p = {0};
    lrn_shift_p.dst = &shift_sum;
    lrn_shift_p.src = &top;
    lrn_shift_p.right_shift = false;
    lrn_shift_p.lrn_step = step;
    CV18xx::parallel_disable();
    CV18xx::tdma_l2l_tensor_lrn_shift(&lrn_shift_p);
    CV18xx::parallel_enable();

    cvk_tiu_mac_param_t p3 = {0};
    p3.res_high = &sum_high;
    p3.res_low = &sum;
    p3.res_is_int8 = 0;
    p3.a = &shift_sum;
    p3.b_const.val = 1;
    p3.b_is_const = 1;
    p3.b_const.is_signed = 0;
    p3.lshift_bits = 0;
    p3.rshift_bits = 0;
    p3.layer_id = layer_id;
    p3.relu_enable = 0;
    CV18xx::tiu_mac(&p3);

    lrn_shift_p.dst = &top_high;
    lrn_shift_p.src = &top;
    lrn_shift_p.right_shift = true;
    lrn_shift_p.lrn_step = step;
    CV18xx::parallel_disable();
    CV18xx::tdma_l2l_tensor_lrn_shift(&lrn_shift_p);
    CV18xx::parallel_enable();

    p3.res_high = &sum_high;
    p3.res_low = &sum;
    p3.res_is_int8 = 0;
    p3.a = &top_high;
    p3.b_const.val = 1;
    p3.b_is_const = 1;
    p3.b_const.is_signed = 0;
    p3.lshift_bits = 0;
    p3.rshift_bits = 0;
    p3.relu_enable = 0;
    CV18xx::tiu_mac(&p3);
  }
  // 16bits higher  8bits,
  cvk_tiu_mul_param_t p = {0};
  p.res_high = &top_high;
  p.res_low = &sum_high;
  p.a = &sum_high;
  p.b_const.val = m_i8[0];
  p.b_const.is_signed = 0;
  p.b_is_const = 1;
  p.rshift_bits = 0;
  p.relu_enable = 0;
  CV18xx::tiu_mul(&p);

  cvk_tiu_mac_param_t p3 = {0};
  p3.res_high = &top_high;
  p3.res_low = &sum_high;
  p3.res_is_int8 = true;
  p3.a = &sum;
  p3.b_const.val = m_i8[0];
  p3.b_is_const = 1;
  p3.b_const.is_signed = 0;
  p3.lshift_bits = 8;
  p3.rshift_bits = sum_rshift_i8;
  p3.layer_id = layer_id;
  p3.relu_enable = 0;
  CV18xx::tiu_mac(&p3);

  // scale=lut:(k+sum)^(-beta)
  p12.ofmap = &sum_high;
  p12.ifmap = &sum_high;
  p12.table = &pwr_lut;
  p12.layer_id = layer_id;
  CV18xx::tiu_lookup_table(&p12);

  // Y=x*scale*m_i8[1]>>lrn_rshift_i8
  cvk_tiu_mul_param_t p1 = {0};
  p1.res_high = &top_high;
  p1.res_low = &shift_sum;
  p1.a = &bottom;
  p1.b = &sum_high;
  p1.b_is_const = 0;
  p1.rshift_bits = 0;
  p1.layer_id = layer_id;
  p1.relu_enable = 0;
  CV18xx::tiu_mul(&p1);

  // 16bits higher  8bits,
  p.res_high = &top_high_high;
  p.res_low = &top_high;
  p.a = &top_high;
  p.b_const.val = m_i8[1];
  p.b_const.is_signed = 0;
  p.b_is_const = 1;
  p.rshift_bits = 0;
  p.relu_enable = 0;
  CV18xx::tiu_mul(&p);

  p3.res_high = &top_high_high;
  p3.res_low = &top_high;
  p3.res_is_int8 = true;
  p3.a = &shift_sum;
  p3.b_const.val = m_i8[1];
  p3.b_is_const = 1;
  p3.b_const.is_signed = 0;
  p3.lshift_bits = 8;
  p3.rshift_bits = lrn_rshift_i8;
  p3.relu_enable = 0;
  CV18xx::tiu_mac(&p3);

  cvk_tiu_min_param_t p7 = {0};
  p7.min = &top;
  p7.a = &top_high;
  p7.b_is_const = 1;
  p7.b_const.val = 127;
  p7.b_const.is_signed = 0;
  p7.layer_id = layer_id;
  CV18xx::tiu_min(&p7);
}

void cvi_backend_bf16_tl_lrn(uint32_t layer_id, laddr_t ifmap_laddr,
                             laddr_t ofmap_laddr, laddr_t power_exp_table,
                             laddr_t power_mantissa_table,
                             laddr_t working_laddr, int input_n, int input_c,
                             int input_h, int input_w, int size, float alpha,
                             float k) {
  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "cvi_backend_bf16_tl_lrn:\n"
                 "    ifmap_laddr 0x%lx, ofmap_laddr 0x%lx"
                 "    power_exp_table 0x%lx, power_mantissa_table 0x%lx, "
                 "working_laddr 0x%lx"
                 "    in(%d, %d, %d, %d), size %d\n"
                 "    alpha %f k %f\n",
                 ifmap_laddr, ofmap_laddr, power_exp_table,
                 power_mantissa_table, working_laddr, input_n, input_c, input_h,
                 input_w, size, alpha, k));

  int move_counts = (size - 1) / 2;
  cvk_tl_shape_t lshape =
      CV18xx::tl_shape_t4(input_n, input_c, input_h, input_w);

  // cvk_tl_shape_t table_shape = {1, 32, 32, 8};
  cvk_tl_shape_t table_shape = CV18xx::tl_shape_t4(1, CV18xx::NPU_NUM, 32, 8);

  cvk_tl_t bottom = {};
  bottom.start_address = ifmap_laddr;
  bottom.fmt = CVK_FMT_BF16;
  bottom.shape = lshape;
  bottom.stride = CV18xx::tl_default_stride(lshape, CVK_FMT_BF16, 1);

  cvk_tl_t top = {};
  cvk_tl_t *tmp = &top;
  top.start_address = ofmap_laddr;
  top.fmt = CVK_FMT_BF16;
  top.shape = lshape;
  top.stride = CV18xx::tl_default_stride(lshape, CVK_FMT_BF16, 1);

  cvk_tl_t power_exp_lut = {};
  power_exp_lut.start_address = power_exp_table;
  power_exp_lut.fmt = CVK_FMT_BF16;
  power_exp_lut.shape = table_shape;
  power_exp_lut.stride =
      CV18xx::tl_default_stride(table_shape, CVK_FMT_BF16, 1);

  cvk_tl_t power_mantissa_lut = {};
  power_mantissa_lut.start_address = power_mantissa_table;
  power_mantissa_lut.fmt = CVK_FMT_BF16;
  power_mantissa_lut.shape = table_shape;
  power_mantissa_lut.stride =
      CV18xx::tl_default_stride(table_shape, CVK_FMT_BF16, 1);

  cvk_tl_t sum = {};
  sum.start_address = working_laddr;
  sum.fmt = CVK_FMT_BF16;
  sum.shape = lshape;
  sum.stride = CV18xx::tl_default_stride(lshape, CVK_FMT_BF16, 1);

  int c_per_npu = ceiling_func(input_c, CV18xx::NPU_NUM);
  int csize_local = c_per_npu * bottom.stride.c;
  int working_size = input_n * csize_local;

  cvk_tl_t shift_sum = {};
  shift_sum.start_address = sum.start_address + working_size;
  shift_sum.fmt = CVK_FMT_BF16;
  shift_sum.shape = lshape;
  shift_sum.stride = CV18xx::tl_default_stride(lshape, CVK_FMT_BF16, 1);

  // y = x * ( k + sum(ax^2))^(-beta)
  // sum = x^2
  cvk_tiu_mul_param_t p0 = {0};
  p0.res_high = nullptr;
  p0.res_low = &sum;
  p0.a = &bottom;
  p0.b = &bottom;
  p0.b_is_const = 0;
  p0.rshift_bits = 0;
  p0.layer_id = layer_id;
  p0.relu_enable = 0;
  CV18xx::tiu_mul(&p0);

  // sum = a(x^2)
  cvk_tiu_mul_param_t p1 = {0};
  p1.res_high = nullptr;
  p1.res_low = &sum;
  p1.a = &sum;
  p1.b_const.val = CV18xx::convert_fp32_to_bf16(alpha / size);
  p1.b_const.is_signed = 1;
  p1.b_is_const = 1;
  p1.rshift_bits = 0;
  p1.layer_id = layer_id;
  p1.relu_enable = 0;
  CV18xx::tiu_mul(&p1);

  // tmp = sum * 1.0
  cvk_tiu_mul_param_t p2 = {0};
  p2.res_high = nullptr;
  p2.res_low = tmp;
  p2.a = &sum;
  p2.b_const.val = CV18xx::convert_fp32_to_bf16(1.0);
  p2.b_const.is_signed = 1;
  p2.b_is_const = 1;
  p2.rshift_bits = 0;
  p2.layer_id = layer_id;
  p2.relu_enable = 0;
  CV18xx::tiu_mul(&p2);

  // tmp = sum(ax^2)
  for (int step = 1; step <= move_counts && step < input_c; step++) {
    // sum shift c left -> shift_sum
    cvk_tdma_l2l_tensor_lrn_shift_param_t lrn_shift_p = {0};
    lrn_shift_p.dst = &shift_sum;
    lrn_shift_p.src = tmp;
    lrn_shift_p.right_shift = false;
    lrn_shift_p.lrn_step = step;
    CV18xx::parallel_disable();
    CV18xx::tdma_l2l_tensor_lrn_shift(&lrn_shift_p);
    CV18xx::parallel_enable();

    // sum = shift_sum + sum
    cvk_tiu_mac_param_t p3 = {0};
    p3.res_high = nullptr;
    p3.res_low = &sum;
    p3.res_is_int8 = 0;
    p3.a = &shift_sum;
    p3.b_const.val = CV18xx::convert_fp32_to_bf16(1.0);
    p3.b_is_const = 1;
    p3.b_const.is_signed = 1;
    p3.lshift_bits = 0;
    p3.rshift_bits = 0;
    p3.layer_id = layer_id;
    p3.relu_enable = 0;
    CV18xx::tiu_mac(&p3);

    // sum shift c right -> shift_sum
    lrn_shift_p.dst = &shift_sum;
    lrn_shift_p.src = tmp;
    lrn_shift_p.right_shift = true;
    lrn_shift_p.lrn_step = step;
    CV18xx::parallel_disable();
    CV18xx::tdma_l2l_tensor_lrn_shift(&lrn_shift_p);
    CV18xx::parallel_enable();

    // sum = shift_sum + sum
    cvk_tiu_mac_param_t p4 = {0};
    p4.res_high = nullptr;
    p4.res_low = &sum;
    p4.res_is_int8 = 0;
    p4.a = &shift_sum;
    p4.b_const.val = CV18xx::convert_fp32_to_bf16(1.0);
    p4.b_is_const = 1;
    p4.b_const.is_signed = 1;
    p4.lshift_bits = 0;
    p4.rshift_bits = 0;
    p4.relu_enable = 0;
    CV18xx::tiu_mac(&p4);
  }

  // sum = (k + sum(ax^2))
  cvk_tiu_add_param_t p5 = {0};
  p5.res_high = nullptr;
  p5.res_low = &sum;
  p5.a_high = nullptr;
  p5.a_low = &sum;
  p5.b_is_const = true;
  p5.b_const.val = CV18xx::convert_fp32_to_bf16(k);
  p5.rshift_bits = 0;
  p5.layer_id = layer_id;
  p5.relu_enable = false;
  CV18xx::tiu_add(&p5);

  // (k+sum(ax^2))^(-beta) = tmp^(-beta)
  // we find the exp and mantissa for tmp, and mul them
  // ==> lut: exp * lut: mantissa

  cvk_tdma_l2l_tensor_copy_param_t p6;
  // move high 8 bits to low 8 bit so
  // that we can do table lookup for exp
  memset(&p6, 0x00, sizeof(cvk_tdma_l2l_tensor_copy_param_t));
  p6.dst = &shift_sum;
  p6.src = &sum;
  p6.mv_lut_base = false;
  p6.mv_lut_idx = true;
  p6.layer_id = layer_id;
  CV18xx::parallel_disable();
  CV18xx::tdma_l2l_tensor_copy(&p6);
  CV18xx::parallel_enable();

  // tmp = lut: exp
  cvk_tiu_lookup_table_param_t p7 = {0};
  p7.ofmap = tmp;
  p7.ifmap = &shift_sum;
  p7.table = &power_exp_lut;
  p7.layer_id = layer_id;
  CV18xx::tiu_lookup_table(&p7);

  // shift_sum = lut: mantissa
  cvk_tiu_lookup_table_param_t p8 = {0};
  p8.ofmap = &shift_sum;
  p8.ifmap = &sum;
  p8.table = &power_mantissa_lut;
  p8.layer_id = layer_id;
  CV18xx::tiu_lookup_table(&p8);

  // (1+sum(ax^2))^(-beta) = tmp * shift_sum
  cvk_tiu_mul_param_t p9 = {0};
  p9.res_high = nullptr;
  p9.res_low = &top;
  p9.a = tmp;
  p9.b = &shift_sum;
  p9.b_is_const = 0;
  p9.rshift_bits = 0;
  p9.layer_id = layer_id;
  p9.relu_enable = 0;
  CV18xx::tiu_mul(&p9);

  // x * (1+sum(ax^2))^(-beta) = top * bottom
  cvk_tiu_mul_param_t p10 = {0};
  p10.res_high = nullptr;
  p10.res_low = &top;
  p10.a = &top;
  p10.b = &bottom;
  p10.b_is_const = 0;
  p10.rshift_bits = 0;
  p10.layer_id = layer_id;
  p10.relu_enable = 0;
  CV18xx::tiu_mul(&p10);
}

} // namespace backend
} // namespace tpu_mlir
