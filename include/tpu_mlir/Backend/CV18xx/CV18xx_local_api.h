//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"

namespace tpu_mlir {
namespace backend {
// local tdma
void cvi_backend_tl_load(uint32_t layer_id, laddr_t la_ifmap, gaddr_t ga_ifmap,
                         cvk_fmt_t fmt, uint32_t n, uint32_t ic, uint32_t ih,
                         uint32_t iw);

void cvi_backend_tl_load(uint32_t layer_id, laddr_t la_ifmap, gaddr_t ga_ifmap,
                         uint32_t n, uint32_t ic, uint32_t ih, uint32_t iw,
                         bool doDecompress);

void cvi_backend_tl_store(uint32_t layer_id, laddr_t la_ofmap, gaddr_t ga_ofmap,
                          cvk_fmt_t fmt, uint32_t n, uint32_t oc, uint32_t oh,
                          uint32_t ow);

void cvi_backend_tl_to_tensor(cvk_tl_t *tensor, laddr_t la, uint32_t tensor_n,
                              uint32_t tensor_c, uint32_t tensor_h,
                              uint32_t tensor_w, cvk_fmt_t fmt,
                              uint8_t eu_align);

void cvi_backend_tl_load_stride(uint32_t layer_id, gaddr_t ga_src,
                                laddr_t la_dst, int Local_N, int Local_C,
                                int Local_H, int Local_W, int Global_C,
                                int Global_H, int Global_W, bool DoTranspose,
                                bool DoAligned, bool isNeuron, cvk_fmt_t from,
                                cvk_fmt_t to, bool DoDecompressed);

void cvi_backend_tl_load_stride(uint32_t layer_id, gaddr_t ga_src,
                                laddr_t la_dst, int Local_N, int Local_C,
                                int Local_H, int Local_W, int Global_C,
                                int Global_H, int Global_W, bool DoTranspose,
                                bool DoAligned, bool isNeuron, cvk_fmt_t from,
                                cvk_fmt_t to);

void cvi_backend_tl_load_compressed(uint32_t layer_id, gaddr_t ga_src,
                                    laddr_t la_dst, int Local_N, int Local_C,
                                    int Local_H, int Local_W, int Global_C,
                                    int Global_H, int Global_W,
                                    bool DoTranspose, bool DoAligned,
                                    bool isNeuron, cvk_fmt_t from, cvk_fmt_t to,
                                    int h_step, int step_size, int c_step);

void cvi_backend_tl_load_stride_broadcast(uint32_t layer_id, gaddr_t ga_src,
                                          laddr_t la_dst, int Local_N,
                                          int Local_C, int Local_H, int Local_W,
                                          int Global_C, int Global_H,
                                          int Global_W, bool DoAligned,
                                          bool isNeuron, cvk_fmt_t from,
                                          cvk_fmt_t to, bool DoDecompress);

void cvi_backend_tl_store_stride(uint32_t layer_id, gaddr_t ga_dst,
                                 laddr_t la_src, int Local_N, int Local_C,
                                 int Local_H, int Local_W, int Global_C,
                                 int Global_H, int Global_W, bool DoTranspose,
                                 bool DoAligned, bool isNeuron, cvk_fmt_t from,
                                 cvk_fmt_t to);

void cvi_backend_tl_store_stride(uint32_t layer_id, gaddr_t ga_dst,
                                 laddr_t la_src, int Local_N, int Local_C,
                                 int Local_H, int Local_W, int Global_C,
                                 int Global_H, int Global_W, bool DoTranspose,
                                 bool DoAligned, bool isNeuron, cvk_fmt_t from,
                                 cvk_fmt_t to, bool DoCompress);

void cvi_backend_tl_store_compressed(uint32_t layer_id, gaddr_t ga_dst,
                                     laddr_t la_src, int Local_N, int Local_C,
                                     int Local_H, int Local_W, int Global_C,
                                     int Global_H, int Global_W,
                                     bool DoTranspose, bool DoAligned,
                                     bool isNeuron, cvk_fmt_t from,
                                     cvk_fmt_t to, int h_step, int step_size,
                                     int c_step, bool DoIntraCmdParal = false);

void cvi_backend_tl_copy(uint32_t layer_id, int la_src, int la_dst, int n,
                         int c, int h, int w, bool align, cvk_fmt_t fmt);

void cvi_backend_tl_bf16_ps32_to_fp32(uint32_t layer_id, laddr_t la_addr, int n,
                                      int c, int h, int w);

void cvi_backend_tl_store_fp32(uint32_t layer_id, gaddr_t ga_dst,
                               laddr_t la_src, int n, int c, int h, int w);
void cvi_backend_tl_lut_LA(uint32_t layer_id, laddr_t la_input,
                           laddr_t la_output, laddr_t la_working,
                           gaddr_t ga_input, gaddr_t ga_output,
                           gaddr_t sg_lut_gaddr, int n, int c, int h, int w,
                           bool do_load, bool do_store);

void cvi_backend_int8_tl_lut(uint32_t layer_id, laddr_t la_input,
                             laddr_t la_output, laddr_t la_y_table, int n,
                             int c, int h, int w);

void cvi_backend_bf16_tl_lut(uint32_t layer_id, laddr_t la_input,
                             laddr_t la_output, laddr_t la_working,
                             laddr_t la_y_table, laddr_t la_slope_lut,
                             int thresh_min, int thresh_max, int n, int c,
                             int h, int w, int method);

void cvi_backend_bf16_tl_lut_mantissa_method(
    uint32_t layer_id, laddr_t la_input, laddr_t la_output, laddr_t la_working,
    laddr_t la_exponential_table, laddr_t la_mantissa_lut, int n, int c, int h,
    int w, bool enable_parallel);

void cvi_backend_bf16_tl_log_lut_mantissa_method(
    uint32_t layer_id, laddr_t la_input, laddr_t la_output, laddr_t la_working,
    laddr_t la_exponential_table, laddr_t la_mantissa_lut, int n, int c, int h,
    int w, bool enable_parallel);

void cvi_backend_bf16_tl_lut_slope_method(
    uint32_t layer_id, laddr_t la_input, laddr_t la_output, laddr_t la_working,
    laddr_t la_y_table, laddr_t la_slope_table, int thresh_min, int thresh_max,
    int n, int c, int h, int w, bool enable_parallel);

void cvi_backend_tl_concat(uint32_t layer_id, int *input_dim_c, int input_size,
                           int *output_dim, laddr_t *la_input,
                           laddr_t la_output, bool do_relu, int32_t *r_i8,
                           int32_t *m_i8, cvk_fmt_t fmt);

void cvi_backend_tl_crop(uint32_t layer_id, int64_t *input_dim,
                         int64_t *output_dim, laddr_t la_input,
                         laddr_t la_output, int *offsets, cvk_fmt_t fmt);

void cvi_backend_tl_conv(
    uint32_t layer_id, laddr_t la_ifmap, laddr_t la_ofmap, laddr_t la_weight,
    laddr_t la_working, laddr_t la_perchannel, int input_n, int input_c,
    int input_h, int input_w, int group, int output_c, int output_h,
    int output_w, uint32_t kh, uint32_t kw, uint32_t dilation_h,
    uint32_t dilation_w, uint32_t pad_h_top, uint32_t pad_h_bottom,
    uint32_t pad_w_left, uint32_t pad_w_right, uint32_t stride_h,
    uint32_t stride_w, uint32_t insert_h, uint32_t insert_w,
    uint32_t result_add, uint32_t ctrl, bool do_bias, bool do_relu, float slope,
    int rshift, int rshift_len, int8_t rshift_pos, int8_t rshift_neg,
    int8_t m_i8_pos, int8_t m_i8_neg, bool do_ic_alignment);

void cvi_backend_bf16_tl_conv(
    uint32_t layer_id, laddr_t la_ifmap, laddr_t la_ofmap, laddr_t la_weight,
    laddr_t la_working, laddr_t la_bias, int input_n, int input_c, int input_h,
    int input_w, int group, int output_c, int output_h, int output_w,
    uint32_t kh, uint32_t kw, uint32_t dilation_h, uint32_t dilation_w,
    uint32_t pad_h_top, uint32_t pad_h_bottom, uint32_t pad_w_left,
    uint32_t pad_w_right, uint32_t stride_h, uint32_t stride_w,
    uint32_t insert_h, uint32_t insert_w, bool with_bias, bool do_relu);

void cvi_backend_tl_eltwise(uint32_t layer_id, laddr_t *la_input,
                            laddr_t la_output, laddr_t la_working, int input_n,
                            int input_c, int input_h, int input_w,
                            int input_size, int op, int8_t rshift,
                            const int8_t *m_i8, bool use_default_coeff,
                            bool do_relu, float relu_slope, const int *coeffs,
                            const int i32Multiplier, bool do_early_stride,
                            int stride_h, int stride_w);

void cvi_backend_bf16_tl_eltwise(uint32_t layer_id, laddr_t *la_input,
                                 laddr_t la_output, int input_n, int input_c,
                                 int input_h, int input_w, int input_size,
                                 int op, bool use_default_coeff, bool do_relu,
                                 float relu_slope, const float *coeffs,
                                 bool do_early_stride, int stride_h,
                                 int stride_w);

void cvi_backend_tl_leaky_relu(uint32_t layer_id, laddr_t input_laddr,
                               laddr_t output_laddr, int input_n, int input_c,
                               int input_h, int input_w,
                               int GT_right_shift_width,
                               int LE_right_shift_width, int GT_scale,
                               int LE_scale);

void cvi_backend_tl_bf16_layernorm(uint32_t layer_id, laddr_t la_input,
                                   laddr_t la_output, laddr_t la_table,
                                   laddr_t la_mantissa_table, laddr_t la_scale,
                                   laddr_t la_bias, laddr_t la_working,
                                   bool has_scale, bool has_bias, float eps,
                                   int n, int c, int h, int w);

void cvi_backend_bf16_tl_leaky_relu(uint32_t layer_id, laddr_t input_laddr,
                                    laddr_t output_laddr, laddr_t work_addr,
                                    int input_n, int input_c, int input_h,
                                    int input_w, float neg_slope);

void cvi_backend_tl_prelu(uint32_t layer_id, laddr_t la_input,
                          laddr_t la_output, laddr_t la_slope, int input_n,
                          int input_c, int input_h, int input_w,
                          int8_t r_i8_pos, int8_t m_i8_pos, int8_t r_i8_neg);

void cvi_backend_tl_bf16_prelu(uint32_t layer_id, laddr_t la_input,
                               laddr_t la_output, laddr_t la_slope, int n,
                               int c, int h, int w);

void cvi_backend_tl_pooling(uint32_t layer_id, laddr_t ifmap_laddr,
                            laddr_t ofmap_laddr, int input_n, int input_c,
                            int input_h, int input_w, int output_n,
                            int output_c, int output_h, int output_w,
                            uint32_t kh, uint32_t kw, uint32_t stride_h,
                            uint32_t stride_w, uint32_t pad_h_top,
                            uint32_t pad_h_bottom, uint32_t pad_w_left,
                            uint32_t pad_w_right, bool is_avg_pooling,
                            int8_t rshift, int8_t m_i8);

void cvi_backend_tl_bf16_pooling(uint32_t layer_id, laddr_t ifmap_laddr,
                                 laddr_t ofmap_laddr, int input_n, int input_c,
                                 int input_h, int input_w, int output_n,
                                 int output_c, int output_h, int output_w,
                                 uint32_t kh, uint32_t kw, uint32_t stride_h,
                                 uint32_t stride_w, uint32_t pad_h_top,
                                 uint32_t pad_h_bottom, uint32_t pad_w_left,
                                 uint32_t pad_w_right, bool is_avg_pooling);

void cvi_backend_tl_quant(uint32_t layer_id, laddr_t la_input,
                          laddr_t la_output, laddr_t la_working, cvk_fmt_t from,
                          cvk_fmt_t to, float const_scale, int n, int c, int h,
                          int w, bool bExtraInput);

void cvi_backend_tl_upsample(uint32_t layer_id, laddr_t input_laddr,
                             laddr_t output_laddr, int input_n, int input_c,
                             int input_h, int input_w, int scale_h, int scale_w,
                             cvk_fmt_t fmt);

void cvi_backend_tl_deconv(uint32_t layer_id, laddr_t la_ifmap,
                           laddr_t la_ofmap, laddr_t la_weight,
                           laddr_t la_perchannel, int input_n, int input_c,
                           int input_h, int input_w, int group, int output_c,
                           int output_h, int output_w, int kh, int kw, int dh,
                           int dw, int ins_h, int ins_last_h, int ins_w,
                           int ins_last_w, int pad_h_top, int pad_h_bottom,
                           int pad_w_left, int pad_w_right, int stride_h,
                           int stride_w, bool do_bias, bool do_relu, int rshift,
                           int rshift_len, bool do_ic_alignment);

void cvi_backend_tl_bf16_deconv(
    uint32_t layer_id, laddr_t la_ifmap, laddr_t la_ofmap, laddr_t la_weight,
    laddr_t la_bias, int input_n, int input_c, int input_h, int input_w,
    int group, int output_c, int output_h, int output_w, int kh, int kw, int dh,
    int dw, int ins_h, int ins_last_h, int ins_w, int ins_last_w, int pad_h_top,
    int pad_h_bottom, int pad_w_left, int pad_w_right, int stride_h,
    int stride_w, bool do_bias, bool do_relu);

void cvi_backend_int8_tl_lut(uint32_t layer_id, laddr_t la_input,
                             laddr_t la_output, laddr_t la_y_table, int n,
                             int c, int h, int w);

void cvi_backend_bf16_tl_lut(uint32_t layer_id, laddr_t la_input,
                             laddr_t la_output, laddr_t la_working,
                             laddr_t la_y_table, laddr_t la_slope_table,
                             int thresh_min, int thresh_max, int n, int c,
                             int h, int w, int method);

void cvi_backend_tl_scale_lut(uint32_t layer_id, laddr_t ifmap_laddr,
                              laddr_t ofmap_laddr, laddr_t table_laddr,
                              int input_n, int input_c, int input_h,
                              int input_w);

void cvi_backend_tl_relu(uint32_t layer_id, int n, int c, int h, int w,
                         laddr_t la_input, laddr_t la_output);

void cvi_backend_tl_bf16_relu(uint32_t layer_id, int n, int c, int h, int w,
                              laddr_t la_input, laddr_t la_output);

void cvi_backend_tl_swap_channel(uint32_t layer_id, laddr_t la_input,
                                 laddr_t la_output, int n, int c, int h, int w,
                                 int *order, cvk_fmt_t fmt);

void cvi_backend_tl_pad(uint32_t layer_id, int64_t *input_dim,
                        int64_t *output_dim, laddr_t la_input,
                        laddr_t la_output, float const_val, int32_t *pads);

void cvi_backend_tl_bf16_pad(uint32_t layer_id, int64_t *input_dim,
                             int64_t *output_dim, laddr_t la_input,
                             laddr_t la_output, float const_val, int32_t *pads);

void cvi_backend_bf16_tl_lrn(uint32_t layer_id, laddr_t ifmap_laddr,
                             laddr_t ofmap_laddr, laddr_t power_exp_table,
                             laddr_t power_mantissa_table,
                             laddr_t working_laddr, int input_n, int input_c,
                             int input_h, int input_w, int size, float alpha,
                             float k);
} // namespace backend
} // namespace tpu_mlir
