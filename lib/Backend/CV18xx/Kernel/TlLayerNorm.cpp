//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"

#define DEBUG_TYPE "tl_layernorm"

namespace tpu_mlir {
namespace backend {
void cvi_backend_tl_bf16_layernorm(uint32_t layer_id, laddr_t la_input,
                                   laddr_t la_output, laddr_t la_table,
                                   laddr_t la_mantissa_table, laddr_t la_scale,
                                   laddr_t la_bias, laddr_t la_working,
                                   bool has_scale, bool has_bias, float eps,
                                   int n, int c, int h, int w) {
  cvk_fmt_t fmt = CVK_FMT_BF16;
  cvk_tl_t tl_input = {0};
  cvk_tl_t tl_output = {0};
  cvk_tl_t tl_lut = {0};
  cvk_tl_t tl_mantissa = {0};
  cvk_tl_t tl_square = {0};
  cvk_tl_t tl_mean = {0};
  cvk_tl_t tl_buff = {0};
  cvk_tl_t tl_var = {0};
  cvk_tl_t tl_scale = {0};
  cvk_tl_t tl_bias = {0};
  auto lshape = CV18xx::tl_shape_t4(n, c, h, w);
  auto table_shape = CV18xx::lut_table_shape(fmt);
  auto mean_shape = CV18xx::tl_shape_t4(n, c, 1, 1);
  auto mean_size = CV18xx::lmem_tensor_to_size(mean_shape, fmt, 1);

  CV18xx::lmem_init_tensor(&tl_input, lshape, fmt, 1);
  tl_input.start_address = la_input;
  CV18xx::lmem_init_tensor(&tl_output, lshape, fmt, 1);
  tl_output.start_address = la_output;
  CV18xx::lmem_init_tensor(&tl_lut, table_shape, fmt, 1);
  tl_lut.start_address = la_table;
  CV18xx::lmem_init_tensor(&tl_mantissa, table_shape, fmt, 1);
  tl_mantissa.start_address = la_mantissa_table;
  CV18xx::lmem_init_tensor(&tl_square, lshape, fmt, 1);
  tl_square.start_address = la_working;
  CV18xx::lmem_init_tensor(&tl_mean, mean_shape, fmt, 1);
  tl_mean.start_address =
      tl_square.start_address + CV18xx::lmem_tensor_to_size(lshape, fmt, 1);
  CV18xx::lmem_init_tensor(&tl_buff, mean_shape, fmt, 1);
  tl_buff.start_address = tl_mean.start_address + mean_size;
  CV18xx::lmem_init_tensor(&tl_var, mean_shape, fmt, 1);
  tl_var.start_address = tl_buff.start_address + mean_size;
  CV18xx::lmem_init_tensor(&tl_scale, lshape, fmt, 1);
  CV18xx::lmem_init_tensor(&tl_bias, lshape, fmt, 1);
  tl_scale.start_address = la_scale;
  tl_scale.stride.n = 0;
  tl_scale.stride.c = 0;
  tl_bias.start_address = la_bias;
  tl_bias.stride.n = 0;
  tl_bias.stride.c = 0;

  // inupt => mean
  cvk_tiu_average_pooling_param_t p1 = {0};
  p1.ofmap = &tl_mean;
  p1.ifmap = &tl_input;
  p1.kh = h;
  p1.kw = w;
  p1.ins_h = 0;
  p1.ins_last_h = 0;
  p1.ins_w = 0;
  p1.ins_last_w = 0;
  p1.stride_h = 1;
  p1.stride_w = 1;
  p1.avg_pooling_const = CV18xx::convert_fp32_to_bf16(1.0);
  p1.ins_val = 0;
  p1.ins_fp = CV18xx::convert_fp32_to_bf16(0.0);
  p1.layer_id = layer_id;
  CV18xx::tiu_average_pooling(&p1);

  // output = input - mean
  // expand [n,c,1,1] =>[n, c,h,w]
  tl_mean.shape = tl_input.shape;
  tl_mean.stride.w = 0;
  tl_mean.stride.h = 0;
  cvk_tiu_sub_param_t p3 = {0};
  p3.res_high = 0;
  p3.res_low = &tl_output;
  p3.a_high = 0;
  p3.a_low = &tl_input;
  p3.b_high = 0;
  p3.b_low = &tl_mean;
  p3.rshift_bits = 0;
  p3.layer_id = layer_id;
  CV18xx::tiu_sub(&p3);

  // square = output * output
  cvk_tiu_mul_param_t p4 = {0};
  p4.res_high = nullptr;
  p4.res_low = &tl_square;
  p4.a = &tl_output;
  p4.b = &tl_output;
  p4.b_is_const = 0;
  p4.rshift_bits = 0;
  p4.layer_id = layer_id;
  p4.relu_enable = 0;
  CV18xx::tiu_mul(&p4);

  // mean = average(square)
  tl_mean.shape = mean_shape;
  tl_mean.stride = CV18xx::tl_default_stride(mean_shape, fmt, 1);
  cvk_tiu_average_pooling_param_t p5 = {0};
  p5.ofmap = &tl_mean;
  p5.ifmap = &tl_square;
  p5.kh = h;
  p5.kw = w;
  p5.ins_h = 0;
  p5.ins_last_h = 0;
  p5.ins_w = 0;
  p5.ins_last_w = 0;
  p5.stride_h = 1;
  p5.stride_w = 1;
  p5.avg_pooling_const = CV18xx::convert_fp32_to_bf16(1.0);
  p5.ins_val = 0;
  p5.ins_fp = CV18xx::convert_fp32_to_bf16(0.0);
  p5.layer_id = layer_id;
  CV18xx::tiu_average_pooling(&p5);

  // mean = mean + eps
  cvk_tiu_add_param_t p6 = {0};
  p6.res_high = nullptr;
  p6.res_low = &tl_mean;
  p6.a_high = nullptr;
  p6.a_low = &tl_mean;
  p6.b_is_const = true;
  p6.b_const.val = CV18xx::convert_fp32_to_bf16(eps);
  p6.rshift_bits = 0;
  p6.layer_id = layer_id;
  p6.relu_enable = 0;
  CV18xx::tiu_add(&p6);

  // var = 1/sqrt(mean)
  cvk_tiu_bf16_lookup_interp_table_param_t p7 = {0};
  p7.ifmap = &tl_mean;
  p7.buf = &tl_buff;
  p7.tbl_answer = &tl_lut;
  p7.tbl_answer_mantissa = &tl_mantissa;
  p7.ofmap = &tl_var;
  p7.is_scientific = 1;
  CV18xx::tiu_bf16_lookup_interp_table(&p7);

  // output = output * var
  tl_var.shape = tl_output.shape;
  tl_var.stride.w = 0;
  tl_var.stride.h = 0;
  cvk_tiu_mul_param_t p9 = {0};
  p9.res_high = nullptr;
  p9.res_low = &tl_output;
  p9.a = &tl_output;
  p9.b = &tl_var;
  p9.b_is_const = 0;
  p9.rshift_bits = 0;
  p9.layer_id = layer_id;
  p9.relu_enable = 0;
  CV18xx::tiu_mul(&p9);

  if (has_scale) {
    cvk_tiu_mul_param_t p10 = {0};
    p10.res_high = nullptr;
    p10.res_low = &tl_output;
    p10.a = &tl_output;
    p10.b = &tl_scale;
    p10.layer_id = layer_id;
    p10.relu_enable = 0;
    CV18xx::tiu_mul(&p10);
  }
  if (has_bias) {
    cvk_tiu_add_param_t p11 = {0};
    p11.res_high = nullptr;
    p11.res_low = &tl_output;
    p11.a_high = nullptr;
    p11.a_low = &tl_output;
    p11.b.high = nullptr;
    p11.b.low = &tl_bias;
    p11.rshift_bits = 0;
    p11.layer_id = layer_id;
    p11.relu_enable = 0;
    CV18xx::tiu_add(&p11);
  }
}
} // namespace backend
} // namespace tpu_mlir
