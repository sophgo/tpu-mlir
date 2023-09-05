//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"

#define DEBUG_TYPE "tl_leay_relu"

namespace tpu_mlir {
namespace backend {
void cvi_backend_tl_leaky_relu(uint32_t layer_id, laddr_t input_laddr,
                               laddr_t output_laddr, int input_n, int input_c,
                               int input_h, int input_w,
                               int GT_right_shift_width,
                               int LE_right_shift_width, int GT_scale,
                               int LE_scale) {
  bool isIgnorePosPart =
      (GT_scale == 0 || (GT_scale == 1 && GT_right_shift_width == 0));
  bool isSlopeSmallerThanOne = ((LE_scale >> LE_right_shift_width) == 0);

  // input
  cvk_tl_shape_t tl_shape =
      CV18xx::tl_shape_t4(input_n, input_c, input_h, input_w);
  cvk_tl_t tl_input;
  tl_input.start_address = input_laddr;
  tl_input.fmt = CVK_FMT_I8;
  tl_input.shape = tl_shape;
  tl_input.stride = CV18xx::tl_default_stride(tl_shape, CVK_FMT_I8, 1);

  cvk_tl_t tl_output;
  tl_output.start_address = output_laddr;
  tl_output.fmt = CVK_FMT_I8;
  tl_output.shape = tl_shape;
  tl_output.stride = CV18xx::tl_default_stride(tl_shape, CVK_FMT_I8, 1);

  if (isIgnorePosPart && LE_scale >= 0) {
    cvk_tiu_mul_param_t p4 = {0};
    p4.res_high = nullptr;
    p4.res_low = &tl_output;
    p4.a = &tl_input;
    p4.b_const.val = LE_scale;
    p4.b_const.is_signed = true;
    p4.b_is_const = 1;
    p4.rshift_bits = LE_right_shift_width;
    p4.layer_id = layer_id;
    p4.relu_enable = 0;
    CV18xx::tiu_mul(&p4);

    if (isSlopeSmallerThanOne) {
      cvk_tiu_max_param_t p1 = {0};
      p1.max = &tl_output;
      p1.a = &tl_input;
      p1.b = &tl_output;
      p1.b_is_const = 0;
      p1.layer_id = layer_id;
      CV18xx::tiu_max(&p1);
    } else {
      cvk_tiu_min_param_t p1 = {0};
      p1.min = &tl_output;
      p1.a = &tl_input;
      p1.b = &tl_output;
      p1.b_is_const = 0;
      p1.layer_id = layer_id;
      CV18xx::tiu_min(&p1);
    }
  } else {
    // 0. relu = relu(bottom)
    cvk_tiu_max_param_t p13 = {0};
    p13.max = &tl_output;
    p13.a = &tl_input;
    p13.b_is_const = 1;
    p13.b_const.is_signed = 1;
    p13.b_const.val = 0;
    p13.layer_id = layer_id;
    CV18xx::tiu_max(&p13);

    if (!isIgnorePosPart) {
      // 1. relu = (relu * GT_scale) >> GT_right_shift_width
      cvk_tiu_mul_param_t p = {0};
      p.res_high = nullptr;
      p.res_low = &tl_output;
      p.a = &tl_output;
      p.b_const.val = GT_scale;
      p.b_const.is_signed = true;
      p.b_is_const = 1;
      p.rshift_bits = GT_right_shift_width;
      p.layer_id = layer_id;
      p.relu_enable = 0;
      CV18xx::tiu_mul(&p);
    }

    // 2. neg = neg(0, botom)
    cvk_tiu_min_param_t p7 = {0};
    p7.min = &tl_input;
    p7.a = &tl_input;
    p7.b_is_const = 1;
    p7.b_const.val = 0;
    p7.b_const.is_signed = 1;
    p7.layer_id = layer_id;
    CV18xx::tiu_min(&p7);

    // 3. neg (n,c,h,w) = (neg(n,c,h,w) * slope) >> LE_right_shift_width
    cvk_tiu_mul_param_t p8 = {0};
    p8.res_high = nullptr;
    p8.res_low = &tl_input;
    p8.a = &tl_input;
    p8.b_const.val = LE_scale;
    p8.b_const.is_signed = true;
    p8.b_is_const = 1;
    p8.rshift_bits = LE_right_shift_width;
    p8.layer_id = layer_id;
    p8.relu_enable = 0;
    CV18xx::tiu_mul(&p8);

    // 4. bottom = or relu, neg
    cvk_tiu_or_int8_param_t p9 = {0};
    p9.res = &tl_output;
    p9.a = &tl_output;
    p9.b = &tl_input;
    p9.layer_id = layer_id;
    CV18xx::tiu_or_int8(&p9);
  }
}

void cvi_backend_bf16_tl_leaky_relu(uint32_t layer_id, laddr_t input_laddr,
                                    laddr_t output_laddr, laddr_t work_laddr,
                                    int input_n, int input_c, int input_h,
                                    int input_w, float neg_slope) {

  // input
  cvk_tl_shape_t tl_shape =
      CV18xx::tl_shape_t4(input_n, input_c, input_h, input_w);
  cvk_tl_t tl_input, tl_output, tl_working;
  CV18xx::lmem_init_tensor(&tl_input, tl_shape, CVK_FMT_BF16, 1);
  CV18xx::lmem_init_tensor(&tl_output, tl_shape, CVK_FMT_BF16, 1);
  CV18xx::lmem_init_tensor(&tl_working, tl_shape, CVK_FMT_BF16, 1);

  tl_input.start_address = input_laddr;
  tl_output.start_address = output_laddr;
  tl_working.start_address = work_laddr;

  // 0. relu = relu(bottom)
  cvk_tiu_max_param_t p13 = {0};
  p13.max = &tl_working;
  p13.a = &tl_input;
  p13.b_is_const = 1;
  p13.b_const.is_signed = 1;
  p13.b_const.val = 0;
  p13.layer_id = layer_id;
  CV18xx::tiu_max(&p13);

  // 1. neg = neg(0, botom)
  cvk_tiu_min_param_t p7 = {0};
  p7.min = &tl_output;
  p7.a = &tl_input;
  p7.b_is_const = 1;
  p7.b_const.val = 0;
  p7.b_const.is_signed = 1;
  p7.layer_id = layer_id;
  CV18xx::tiu_min(&p7);

  // 3. neg (n,c,h,w) = (neg(n,c,h,w) * slope)
  cvk_tiu_mul_param_t p8 = {0};
  p8.res_high = nullptr;
  p8.res_low = &tl_output;
  p8.a = &tl_output;
  p8.b_const.val = CV18xx::convert_fp32_to_bf16(neg_slope);
  p8.b_const.is_signed = true;
  p8.b_is_const = 1;
  p8.rshift_bits = 0;
  p8.layer_id = layer_id;
  p8.relu_enable = 0;
  CV18xx::tiu_mul(&p8);

  // 4. bottom = or relu, neg
  cvk_tiu_add_param_t p9 = {0};
  p9.res_high = nullptr;
  p9.res_low = &tl_output;
  p9.a_high = nullptr;
  p9.a_low = &tl_working;
  p9.b_is_const = false;
  p9.b.high = nullptr;
  p9.b.low = &tl_output;
  p9.rshift_bits = 0;
  p9.layer_id = layer_id;
  p9.relu_enable = 0;
  CV18xx::tiu_add(&p9);
}

} // namespace backend
} // namespace tpu_mlir
