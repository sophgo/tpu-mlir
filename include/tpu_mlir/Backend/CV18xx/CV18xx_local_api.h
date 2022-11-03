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
// local tdma
void cvi_backend_tl_load(const CviBackendContext &ctx, uint32_t layer_id,
                         laddr_t la_ifmap, gaddr_t ga_ifmap, cvk_fmt_t fmt,
                         uint32_t n, uint32_t ic, uint32_t ih, uint32_t iw);

void cvi_backend_tl_load(const CviBackendContext &ctx, uint32_t layer_id,
                         laddr_t la_ifmap, gaddr_t ga_ifmap, uint32_t n,
                         uint32_t ic, uint32_t ih, uint32_t iw,
                         bool doDecompress);

void cvi_backend_tl_store(const CviBackendContext &ctx, uint32_t layer_id,
                          laddr_t la_ofmap, gaddr_t ga_ofmap, cvk_fmt_t fmt,
                          uint32_t n, uint32_t oc, uint32_t oh, uint32_t ow);

void cvi_backend_tl_to_tensor(const CviBackendContext &ctx, cvk_tl_t *tensor,
                              laddr_t la, uint32_t tensor_n, uint32_t tensor_c,
                              uint32_t tensor_h, uint32_t tensor_w,
                              cvk_fmt_t fmt, uint8_t eu_align);

void cvi_backend_tl_load_stride(const CviBackendContext &ctx, uint32_t layer_id,
                                gaddr_t ga_src, laddr_t la_dst, int Local_N,
                                int Local_C, int Local_H, int Local_W,
                                int Global_C, int Global_H, int Global_W,
                                bool DoTranspose, bool DoAligned, bool isNeuron,
                                cvk_fmt_t from, cvk_fmt_t to,
                                bool DoDecompressed);

void cvi_backend_tl_load_stride(const CviBackendContext &ctx, uint32_t layer_id,
                                gaddr_t ga_src, laddr_t la_dst, int Local_N,
                                int Local_C, int Local_H, int Local_W,
                                int Global_C, int Global_H, int Global_W,
                                bool DoTranspose, bool DoAligned, bool isNeuron,
                                cvk_fmt_t from, cvk_fmt_t to);

void cvi_backend_tl_load_compressed(const CviBackendContext &ctx,
                                    uint32_t layer_id, gaddr_t ga_src,
                                    laddr_t la_dst, int Local_N, int Local_C,
                                    int Local_H, int Local_W, int Global_C,
                                    int Global_H, int Global_W,
                                    bool DoTranspose, bool DoAligned,
                                    bool isNeuron, cvk_fmt_t from, cvk_fmt_t to,
                                    int h_step, int step_size, int c_step);

void cvi_backend_tl_store_stride(const CviBackendContext &ctx,
                                 uint32_t layer_id, gaddr_t ga_dst,
                                 laddr_t la_src, int Local_N, int Local_C,
                                 int Local_H, int Local_W, int Global_C,
                                 int Global_H, int Global_W, bool DoTranspose,
                                 bool DoAligned, bool isNeuron, cvk_fmt_t from,
                                 cvk_fmt_t to);

void cvi_backend_tl_store_stride(const CviBackendContext &ctx,
                                 uint32_t layer_id, gaddr_t ga_dst,
                                 laddr_t la_src, int Local_N, int Local_C,
                                 int Local_H, int Local_W, int Global_C,
                                 int Global_H, int Global_W, bool DoTranspose,
                                 bool DoAligned, bool isNeuron, cvk_fmt_t from,
                                 cvk_fmt_t to, bool DoCompress);

void cvi_backend_tl_store_compressed(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_dst,
    laddr_t la_src, int Local_N, int Local_C, int Local_H, int Local_W,
    int Global_C, int Global_H, int Global_W, bool DoTranspose, bool DoAligned,
    bool isNeuron, cvk_fmt_t from, cvk_fmt_t to, int h_step, int step_size,
    int c_step, bool DoIntraCmdParal = false);

void cvi_backend_tl_copy(const CviBackendContext &ctx, uint32_t layer_id,
                         int la_src, int la_dst, int n, int c, int h, int w,
                         bool align, cvk_fmt_t fmt);

void cvi_backend_tl_bf16_ps32_to_fp32(const CviBackendContext &ctx,
                                      uint32_t layer_id, laddr_t la_addr, int n,
                                      int c, int h, int w);

void cvi_backend_tl_store_fp32(const CviBackendContext &ctx, uint32_t layer_id,
                               gaddr_t ga_dst, laddr_t la_src, int n, int c,
                               int h, int w);
void cvi_backend_tl_lut_LA(const CviBackendContext &ctx, uint32_t layer_id,
                           laddr_t la_input, laddr_t la_output,
                           laddr_t la_working, gaddr_t ga_input,
                           gaddr_t ga_output, gaddr_t sg_lut_gaddr, int n,
                           int c, int h, int w, bool do_load, bool do_store);

void cvi_backend_int8_tl_lut(const CviBackendContext &ctx, uint32_t layer_id,
                             laddr_t la_input, laddr_t la_output,
                             laddr_t la_y_table, int n, int c, int h, int w);

void cvi_backend_bf16_tl_lut(const CviBackendContext &ctx, uint32_t layer_id,
                             laddr_t la_input, laddr_t la_output,
                             laddr_t la_working, laddr_t la_y_table,
                             laddr_t la_slope_lut, int thresh_min,
                             int thresh_max, int n, int c, int h, int w,
                             int method);

void cvi_backend_bf16_tl_lut_mantissa_method(
    const CviBackendContext &ctx, uint32_t layer_id, laddr_t la_input,
    laddr_t la_output, laddr_t la_working, laddr_t la_exponential_table,
    laddr_t la_mantissa_lut, int n, int c, int h, int w);

void cvi_backend_bf16_tl_log_lut_mantissa_method(
    const CviBackendContext &ctx, uint32_t layer_id, laddr_t la_input,
    laddr_t la_output, laddr_t la_working, laddr_t la_exponential_table,
    laddr_t la_mantissa_lut, int n, int c, int h, int w);

void cvi_backend_bf16_tl_lut_slope_method(const CviBackendContext &ctx,
                                          uint32_t layer_id, laddr_t la_input,
                                          laddr_t la_output, laddr_t la_working,
                                          laddr_t la_y_table,
                                          laddr_t la_slope_table,
                                          int thresh_min, int thresh_max, int n,
                                          int c, int h, int w);
} // namespace backend
} // namespace tpu_mlir
