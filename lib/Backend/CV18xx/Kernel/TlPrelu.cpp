//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"

#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "tl_prelu"

namespace tpu_mlir {
namespace backend {
void cvi_backend_tl_prelu(uint32_t layer_id, laddr_t la_input,
                          laddr_t la_output, laddr_t la_slope, int n, int c,
                          int h, int w, int8_t r_i8_pos, int8_t m_i8_pos,
                          int8_t r_i8_neg) {

  LLVM_DEBUG(
      llvm::errs() << llvm::format("cvi_backend_tl_prelu:\n"
                                   "nchw = (%d, %d, %d, %d)\n"
                                   "la_input: %d, la_output: %d, la_slope:%d\n"
                                   "r_i8_pos: %d m_i8_pos: %d r_i8_neg:%d\n",
                                   n, c, h, w, la_input, la_output, la_slope,
                                   r_i8_pos, m_i8_pos, r_i8_neg));

  cvk_tl_t *tl_input = new cvk_tl_t;
  cvk_tl_t *tl_output = new cvk_tl_t;
  cvk_tl_t *tl_slope = new cvk_tl_t;

  tl_input->start_address = la_input;
  tl_input->fmt = CVK_FMT_I8;
  tl_input->shape = CV18xx::tl_shape_t4(n, c, h, w);
  tl_input->stride =
      CV18xx::tl_default_stride(tl_input->shape, CVK_FMT_I8, /*eu_align=*/1);

  tl_output->start_address = la_output;
  tl_output->fmt = CVK_FMT_I8;
  tl_output->shape = CV18xx::tl_shape_t4(n, c, h, w);
  tl_output->stride =
      CV18xx::tl_default_stride(tl_output->shape, CVK_FMT_I8, /*eu_align=*/1);

  tl_slope->start_address = la_slope;
  tl_slope->fmt = CVK_FMT_I8;
  tl_slope->shape = CV18xx::tl_shape_t4(1, c, 1, 1);
  tl_slope->stride =
      CV18xx::tl_default_stride(tl_slope->shape, CVK_FMT_I8, /*eu_align=*/1);

  // 0. relu = relu(bottom)
  // 1. relu = (relu * m_i8_pos) >> r_i8_pos
  // 2. neg = neg(0, botom)
  // 3. neg (n,c,h,w) = (neg(n,c,h,w) * slope(1,c,1,1)) >> r_i8_neg
  // 4. bottom = or relu, neg

  // 0. relu = relu(bottom)
  cvk_tiu_max_param_t p1 = {0};
  p1.max = tl_output;
  p1.a = tl_input;
  p1.b_is_const = 1;
  p1.b_const.is_signed = 1;
  p1.b_const.val = 0;
  p1.layer_id = layer_id;
  CV18xx::tiu_max(&p1);

  // 1. relu = (relu * m_i8_pos) >> r_i8_pos
  if (m_i8_pos != 0 && (m_i8_pos != 1 || r_i8_pos != 0)) {
    cvk_tiu_mul_param_t p2 = {0};
    p2.res_high = nullptr;
    p2.res_low = tl_output;
    p2.a = tl_output;
    p2.b_const.val = m_i8_pos;
    p2.b_const.is_signed = true;
    p2.b_is_const = 1;
    p2.rshift_bits = r_i8_pos;
    p2.layer_id = layer_id;
    p2.relu_enable = 0;
    CV18xx::tiu_mul(&p2);
  }

  // 2. neg = neg(0, botom)
  cvk_tiu_min_param_t p3 = {0};
  p3.min = tl_input;
  p3.a = tl_input;
  p3.b_is_const = 1;
  p3.b_const.val = 0;
  p3.b_const.is_signed = 1;
  p3.layer_id = layer_id;
  CV18xx::tiu_min(&p3);

  // 3. neg (n,c,h,w) = (neg(n,c,h,w) * slope(1,c,1,1)) >> r_i8_neg
  cvk_tiu_depthwise_pt_convolution_param_t p4 = {0};
  p4.ins_h = 0;
  p4.ins_last_h = 0;
  p4.ins_w = 0;
  p4.ins_last_w = 0;
  p4.pad_top = 0;
  p4.pad_bottom = 0;
  p4.pad_left = 0;
  p4.pad_right = 0;
  p4.stride_h = 1;
  p4.stride_w = 1;
  p4.dilation_h = 1;
  p4.dilation_w = 1;
  p4.ofmap = tl_input;
  p4.ifmap = tl_input;
  p4.weight = tl_slope;
  p4.bias = nullptr;
  p4.rshift_bits = r_i8_neg;
  p4.relu_enable = 0;
  p4.layer_id = layer_id;
  p4.ins_val = 0;                                // symmetric quantization
  p4.ins_fp = CV18xx::convert_fp32_to_bf16(0.0); // symmetric quantization
  CV18xx::tiu_pt_depthwise_convolution(&p4);

  // 4. bottom = or relu, neg
  cvk_tiu_or_int8_param_t p5 = {0};
  p5.res = tl_output;
  p5.a = tl_input;
  p5.b = tl_output;
  p5.layer_id = layer_id;
  CV18xx::tiu_or_int8(&p5);

  delete tl_slope;
  delete tl_output;
  delete tl_input;
}

void cvi_backend_tl_bf16_prelu(uint32_t layer_id, laddr_t la_input,
                               laddr_t la_output, laddr_t la_slope, int n,
                               int c, int h, int w) {

  LLVM_DEBUG(
      llvm::errs() << llvm::format("cvi_backend_tl_bf16_prelu:\n"
                                   "nchw = (%d, %d, %d, %d)\n"
                                   "la_input: %d, la_output: %d, la_slope:%d\n",
                                   n, c, h, w, la_input, la_output, la_slope));

  cvk_tl_t *tl_input = new cvk_tl_t;
  cvk_tl_t *tl_output = new cvk_tl_t;
  cvk_tl_t *tl_slope = new cvk_tl_t;

  tl_input->start_address = la_input;
  tl_input->fmt = CVK_FMT_BF16;
  tl_input->shape = CV18xx::tl_shape_t4(n, c, h, w);
  tl_input->stride =
      CV18xx::tl_default_stride(tl_input->shape, CVK_FMT_BF16, /*eu_align=*/1);

  tl_output->start_address = la_output;
  tl_output->fmt = CVK_FMT_BF16;
  tl_output->shape = CV18xx::tl_shape_t4(n, c, h, w);
  tl_output->stride =
      CV18xx::tl_default_stride(tl_output->shape, CVK_FMT_BF16, /*eu_align=*/1);

  tl_slope->start_address = la_slope;
  tl_slope->fmt = CVK_FMT_BF16;
  tl_slope->shape = CV18xx::tl_shape_t4(1, c, 1, 1);
  tl_slope->stride =
      CV18xx::tl_default_stride(tl_slope->shape, CVK_FMT_BF16, /*eu_align=*/1);

  // 1. neg = min(0, botom)
  // 2. neg (n,c,h,w) = (neg(n,c,h,w) * slope(1,c,1,1))
  // 3. relu = relu(bottom), dirty input
  // 4. bottom = or relu, neg

  // 0. neg = min(0, botom)
  cvk_tiu_min_param_t p1 = {0};
  p1.min = tl_output;
  p1.a = tl_input;
  p1.b_is_const = 1;
  p1.b_const.val = CV18xx::convert_fp32_to_bf16(0.0);
  p1.b_const.is_signed = 1;
  p1.layer_id = layer_id;
  CV18xx::tiu_min(&p1);

  // 2. neg (n,c,h,w) = (neg(n,c,h,w) * slope(1,c,1,1))
  cvk_tiu_depthwise_pt_convolution_param_t p2 = {0};
  p2.ins_h = 0;
  p2.ins_last_h = 0;
  p2.ins_w = 0;
  p2.ins_last_w = 0;
  p2.pad_top = 0;
  p2.pad_bottom = 0;
  p2.pad_left = 0;
  p2.pad_right = 0;
  p2.stride_h = 1;
  p2.stride_w = 1;
  p2.dilation_h = 1;
  p2.dilation_w = 1;
  p2.ofmap = tl_output;
  p2.ifmap = tl_output;
  p2.weight = tl_slope;
  p2.bias = nullptr;
  p2.rshift_bits = 0;
  p2.relu_enable = 0;
  p2.layer_id = layer_id;
  p2.ins_val = 0;                                // symmetric quantization
  p2.ins_fp = CV18xx::convert_fp32_to_bf16(0.0); // symmetric quantization
  CV18xx::tiu_pt_depthwise_convolution(&p2);

  // 3. relu = relu(bottom), dirty it
  cvk_tiu_max_param_t p3 = {0};
  p3.max = tl_input;
  p3.a = tl_input;
  p3.b_is_const = 1;
  p3.b_const.is_signed = 1;
  p3.b_const.val = CV18xx::convert_fp32_to_bf16(0.0);
  p3.layer_id = layer_id;
  CV18xx::tiu_max(&p3);

  // 4. bottom = or relu, neg
  cvk_tiu_add_param_t p4 = {0};
  p4.res_high = nullptr;
  p4.res_low = tl_output;
  p4.a_high = nullptr;
  p4.a_low = tl_input;
  p4.b_is_const = false;
  p4.b.high = nullptr;
  p4.b.low = tl_output;
  p4.rshift_bits = 0;
  p4.layer_id = layer_id;
  p4.relu_enable = 0;
  CV18xx::tiu_add(&p4);

  delete tl_slope;
  delete tl_output;
  delete tl_input;
}

} // namespace backend
} // namespace tpu_mlir
