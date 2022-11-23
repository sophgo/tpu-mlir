//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"

namespace tpu_mlir {
namespace backend {
void cvi_backend_tg_fixed_conv_kernel(
    uint32_t layer_id, gaddr_t ga_ifmap, gaddr_t ga_ofmap, gaddr_t ga_weight,
    gaddr_t ga_bias, int input_n, int input_c, int input_h, int input_w,
    int groups, int output_c, uint16_t kh, uint16_t kw, uint16_t dilation_h,
    uint16_t dilation_w, uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left,
    uint8_t pad_right, uint8_t insert_h, uint8_t insert_w, uint8_t stride_h,
    uint8_t stride_w, int do_bias, int do_activation, float activation_arg[],
    int activation_gt_scale, int activation_gt_rshift, int activation_le_scale,
    int activation_le_rshift, int right_shift_width, bool do_chl_quan,
    bool do_ic_alignment, std::vector<uint8_t> *filter = nullptr,
    std::vector<uint8_t> *new_filter = nullptr, int pad_value = 0,
    gaddr_t ga_scale_lut = GA_INVALID);

void cvi_backend_tg_eltwise_abs_kernel(uint32_t layer_id, gaddr_t ga_inputs[],
                                       gaddr_t ga_output, int32_t operand_num,
                                       int32_t n, int32_t c, int32_t h,
                                       int32_t w, bool do_relu,
                                       bool do_early_stride, int32_t stride_h,
                                       int32_t stride_w, int32_t rshift,
                                       const int32_t *multipliers,
                                       const int32_t *coeffs, cvk_fmt_t fmt);

void cvi_backend_tg_fixed_eltwise_add_kernel(
    uint32_t layer_id, gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c, int32_t h, int32_t w,
    bool do_relu, bool do_early_stride, int32_t stride_h, int32_t stride_w,
    int32_t rshift, const int32_t *multipliers, const int32_t *coeffs);

void cvi_backend_tg_fixed_eltwise_max_kernel(
    uint32_t layer_id, gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c, int32_t h, int32_t w,
    bool do_relu, bool do_early_stride, int32_t stride_h, int32_t stride_w,
    int32_t rshift, const int32_t *multipliers, const int32_t *coeffs);

void cvi_backend_tg_fixed_eltwise_mul_kernel(
    uint32_t layer_id, gaddr_t ga_inputs[],
    gaddr_t ga_output, int32_t operand_num, int32_t n, int32_t c, int32_t h, int32_t w,
    bool do_relu, bool do_early_stride, int32_t stride_h, int32_t stride_w,
    int32_t rshift, const int32_t *multipliers, const int32_t *coeffs);

void cvi_backend_tg_fixed_eltwise_min_kernel(
    uint32_t layer_id, gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c, int32_t h, int32_t w,
    bool do_relu, bool do_early_stride, int32_t stride_h, int32_t stride_w,
    int32_t rshift, const int32_t *multipliers, const int32_t *coeffs);

void cvi_backend_tg_quant_kernel(uint32_t layer_id, cvk_fmt_t from,
                                 cvk_fmt_t to, gaddr_t bottom_gaddr,
                                 gaddr_t top_gaddr, int input_n, int input_c,
                                 int input_h, int input_w,
                                 float const_scale = 1.0, int offset = 0);

void cvi_backend_tg_fixed_avg_pooling_kernel(
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output, int n, int c, int h,
    int w, int kh, int kw, int pad_top, int pad_bot, int pad_left,
    int pad_right, int stride_h, int stride_w, bool do_relu, int rshift,
    int multiplier, bool ceil_mode);

void cvi_backend_tg_fixed_max_pooling_kernel(
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output, int n, int c, int h,
    int w, int kh, int kw, int pad_top, int pad_bot, int pad_left,
    int pad_right, int stride_h, int stride_w, bool do_relu, bool ceil_mode);

void cvi_backend_tg_fixed_fc_kernel(
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_weight, gaddr_t ga_bias,
    gaddr_t ga_output, int M, int K, int N, bool have_bias, bool do_relu,
    std::vector<int> rshift_width, std::vector<int> multiplier,
    const std::vector<uint8_t> *old_filter = nullptr,
    std::vector<uint8_t> *new_filter = nullptr, int batch_high = 1,
    int batch_low = 1, bool lstride = false, bool rstride = false,
    bool ostride = false);

// i8 leakyrelu
void cvi_backend_tg_fixed_leakyrelu_kernel(uint32_t layer_id, uint64_t ga_input,
                                           uint64_t ga_output, int n, int c,
                                           int h, int w, int GT_rshift,
                                           int LE_rshift, int GT_scale,
                                           int LE_scale);

void cvi_backend_tg_lut_kernel(uint32_t layer_id, gaddr_t bottom_gaddr,
                               gaddr_t top_gaddr, gaddr_t sg_lut_gaddr,
                               int input_n, int input_c, int input_h,
                               int input_w, cvk_fmt_t fmt);

// i8 prelu
void cvi_backend_tg_fixed_prelu_kernel(uint32_t layer_id, uint64_t ga_input,
                                       uint64_t ga_output,
                                       uint64_t negative_scope_gaddr, int n,
                                       int c, int h, int w, int GT_rshift,
                                       int GT_scale, int LE_rshift);

void cvi_backend_tg_upsample_kernel(uint32_t layer_id, gaddr_t ga_ifmap,
                                    gaddr_t ga_ofmap, uint32_t input_n,
                                    uint32_t input_c, uint32_t input_h,
                                    uint32_t input_w, uint32_t h_factor,
                                    uint32_t w_factor, cvk_fmt_t fmt);

void cvi_backend_tg_permute_kernel(uint32_t layer_id, gaddr_t ga_input,
                                   gaddr_t ga_output, int n, int c, int h,
                                   int w, int order_n, int order_c, int order_h,
                                   int order_w, cvk_fmt_t fmt);

//////////////// bf16 kernel API /////////////////
void cvi_backend_tg_bf16_conv_kernel(
    uint32_t layer_id, gaddr_t ga_ifmap, gaddr_t ga_ofmap, gaddr_t ga_weight,
    gaddr_t ga_bias, int input_n, int input_c, int input_h, int input_w,
    int groups, int output_c, uint16_t kh, uint16_t kw, uint16_t dilation_h,
    uint16_t dilation_w, uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left,
    uint8_t pad_right, uint8_t ins_h, uint8_t ins_w, uint8_t stride_h,
    uint8_t stride_w, int do_bias, int do_activation, bool fp32_output,
    std::vector<uint8_t> *filter = nullptr,
    std::vector<uint8_t> *new_filter = nullptr, bool do_quant = false,
    gaddr_t ga_scale = 0, gaddr_t ga_zeropoint = 0);

void cvi_backend_tg_bf16_eltwise_add_kernel(
    uint32_t layer_id, gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c, int32_t h, int32_t w,
    bool do_relu, bool do_early_stride, int32_t stride_h, int32_t stride_w,
    const float *coeffs);

void cvi_backend_tg_bf16_eltwise_max_kernel(
    uint32_t layer_id, gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c, int32_t h, int32_t w,
    bool do_relu, bool do_early_stride, int32_t stride_h, int32_t stride_w,
    const float coeffs[]);

void cvi_backend_tg_bf16_eltwise_min_kernel(
    uint32_t layer_id, gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c, int32_t h, int32_t w,
    bool do_relu, bool do_early_stride, int32_t stride_h, int32_t stride_w,
    const float coeffs[]);

void cvi_backend_tg_bf16_eltwise_min_max_kernel(
    uint32_t layer_id, gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c, int32_t h, int32_t w,
    bool do_relu, bool do_early_stride, int32_t stride_h, int32_t stride_w,
    const float coeffs[]);

void cvi_backend_tg_bf16_eltwise_mul_kernel(uint32_t layer_id, gaddr_t ga_inputs[],
                                            gaddr_t ga_output, int32_t operand_num,
                                            int32_t n, int32_t c, int32_t h, int32_t w,
                                            bool do_relu, bool do_early_stride,
                                            int32_t stride_h, int32_t stride_w,
                                            const float coeffs[]);

void cvi_backend_tg_bf16_fc_kernel(
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_weight, gaddr_t ga_bias,
    gaddr_t ga_output, int M, int K, int N, bool do_bias, bool do_relu,
    const std::vector<uint8_t> *old_filter = nullptr,
    std::vector<uint8_t> *new_filter = nullptr, int batch_high = 1,
    int batch_low = 1, bool lstride = false, bool rstride = false,
    bool ostride = false, bool do_quant_bf16 = 0, gaddr_t ga_scale = 0,
    gaddr_t ga_zeropoint = 0);

// bf16 leakyrelu
void cvi_backend_tg_bf16_leakyrelu_kernel(uint32_t layer_id, gaddr_t ga_input,
                                          gaddr_t ga_output,
                                          float negative_slope, int n, int c,
                                          int h, int w);

void cvi_backend_tg_bf16_lut_mantissa_kernel(
    uint32_t layer_id, gaddr_t bottom_gaddr, gaddr_t top_gaddr,
    gaddr_t exp_lut_table, gaddr_t mantissa_lut_table, int input_n, int input_c,
    int input_h, int input_w, int method);

void cvi_backend_tg_bf16_lut_slope_kernel(
    uint32_t layer_id, gaddr_t bottom_gaddr, gaddr_t top_gaddr,
    gaddr_t y0_table_gaddr, gaddr_t slope_gaddr, int input_n, int input_c,
    int input_h, int input_w, float range_min, float range_max);

void cvi_backend_tg_bf16_pooling_kernel(
    uint32_t layer_id, gaddr_t ifmap_gaddr, gaddr_t ofmap_gaddr,
    gaddr_t index_gaddr, gaddr_t o_findex_gaddr, int n, int c, int h, int w,
    int kh, int kw, int pad_top, int pad_bot, int pad_left, int pad_right,
    int stride_h, int stride_w, int is_avg_pooling, float avg_const,
    int do_relu, const bool ceil_mode);

void cvi_backend_tg_bf16_prelu_kernel(uint32_t layer_id, gaddr_t ga_input,
                                      gaddr_t ga_output, gaddr_t ga_slope,
                                      int n, int c, int h, int w);

void cvi_backend_tg_bf16_softmax_kernel(
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_exponential_table_data_lut,
    gaddr_t ga_exponential_slope_table_data_lut,
    gaddr_t ga_reciprocal_table_data_lut,
    gaddr_t ga_reciprocal_table_mantissa_data_lut, gaddr_t ga_output,
    int64_t *shape, int axis, int dimension, bool do_log);

//////// fixed & bf16 kernel api ////////////////
void cvi_backend_tg_concat_kernel(uint32_t layer_id, int input_num,
                                  gaddr_t input_gaddrs[], gaddr_t output_gaddr,
                                  int axis_dims[], int concat_axis,
                                  int output_dim_size, int *output_dim,
                                  bool do_relu, const int *right_shift_width,
                                  const int *threshold_x_quantized,
                                  cvk_fmt_t fmt);

void cvi_backend_tg_crop_kernel(uint32_t layer_id, gaddr_t ga_input,
                                gaddr_t ga_output, std::vector<int64_t> is_4,
                                std::vector<int64_t> os_4,
                                std::vector<int> offsets,
                                std::vector<int> steps, cvk_fmt_t fmt);

void cvi_backend_tg_pad_kernel(uint32_t layer_id, gaddr_t ga_ifmap,
                               gaddr_t ga_ofmap, int input_n, int input_c,
                               int input_h, int input_w, int *pads,
                               float const_val, const char *mode,
                               cvk_fmt_t fmt);

void cvi_backend_tg_reflectionpad_kernel(uint32_t layer_id, gaddr_t ga_input,
                                         gaddr_t ga_output, gaddr_t ga_left,
                                         gaddr_t ga_right, int outer_size,
                                         int ih, int iw, std::vector<int> &pads,
                                         cvk_fmt_t fmt);

void cvi_backend_tg_relu_kernel(uint32_t layer_id, uint64_t ga_input,
                                uint64_t ga_output, int n, int c, int h, int w,
                                cvk_fmt_t fmt);

void cvi_backend_tg_int8_bcast_add_kernel(uint32_t layer_id, gaddr_t ga_a,
                                          gaddr_t ga_b, gaddr_t ga_output,
                                          int32_t an, int32_t ac, int32_t ah,
                                          int32_t aw, int32_t bn, int32_t bc,
                                          int32_t bh, int32_t bw, bool do_relu,
                                          const int32_t rshift,
                                          const int32_t *multipliers);

void cvi_backend_tg_bf16_bcast_add_kernel(uint32_t layer_id, gaddr_t ga_a,
                                            gaddr_t ga_b, gaddr_t ga_output,
                                            int an, int ac, int ah, int aw,
                                            int bn, int bc, int bh, int bw,
                                            bool do_relu);

void cvi_backend_tg_int8_bcast_mul_kernel(uint32_t layer_id, gaddr_t ga_a,
                                          gaddr_t ga_b, gaddr_t ga_output,
                                          int32_t an, int32_t ac, int32_t ah,
                                          int32_t aw, int32_t bn, int32_t bc,
                                          int32_t bh, int32_t bw, bool do_relu,
                                          const int32_t rshift,
                                          const int32_t *multipliers);

void cvi_backend_tg_bf16_bcast_mul_kernel(uint32_t layer_id, gaddr_t ga_a,
                                          gaddr_t ga_b, gaddr_t ga_output,
                                          int n, int c, int h, int w, int bn,
                                          int bc, int bh, int bw,
                                          bool do_relu);
} // namespace backend
} // namespace tpu_mlir
