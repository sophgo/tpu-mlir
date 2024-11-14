//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"

#define DEBUG_TYPE "kernel_reflectionpad"

namespace tpu_mlir {
namespace backend {
static void matrix_for_tiu(cvk_ml_t *matrix, cvk_fmt_t fmt) {
  if (matrix->shape.w < CV18xx::tiu_eu_num(fmt) && matrix->shape.c > 1) {
    matrix->shape.w = CV18xx::tiu_eu_num(fmt);
    matrix->stride = CV18xx::ml_default_stride(matrix->shape, fmt, 1);
  }
}

static void matrix_mul(uint32_t layer_id, const cvk_ml_t *ml_res,
                       const cvk_ml_t *ml_left, const cvk_ml_t *ml_right,
                       cvk_fmt_t fmt) {
  cvk_ml_t ml_res_ = *ml_res;
  cvk_ml_t ml_left_ = *ml_left;
  cvk_ml_t ml_right_ = *ml_right;
  matrix_for_tiu(&ml_res_, fmt);
  matrix_for_tiu(&ml_left_, fmt);
  matrix_for_tiu(&ml_right_, fmt);
  cvk_tiu_matrix_multiplication_param_t p = {0};
  p.res = &ml_res_;
  p.left = &ml_left_;
  p.right = &ml_right_;
  p.res_is_int8 = 1; // dont care in bf16
  p.bias = nullptr;
  p.ps32_mode = 0;
  p.layer_id = layer_id;
  CV18xx::tiu_matrix_multiplication(&p);
}

static void do_reflection(uint32_t layer_id, gaddr_t src_addr, gaddr_t dst_addr,
                          gaddr_t weight_addr, cvk_mg_stride_t &src_gstride,
                          cvk_mg_stride_t &dst_gstride, int outer_size, int pad,
                          cvk_fmt_t fmt) {
  if (pad == 0) {
    return;
  }
  auto wshape = CV18xx::ml_default_shape(pad, pad, fmt);
  auto ml_weight = CV18xx::lmem_alloc_matrix(wshape, fmt, 1);
  auto lmem_used = CV18xx::lmem_matrix_to_size(wshape, fmt, 1);
  int step;
  for (step = outer_size; step > 0; step--) {
    auto ishape = CV18xx::ml_default_shape(step, pad, fmt);
    auto lmem_need = CV18xx::lmem_matrix_to_size(ishape, fmt, 1) * 2;
    if (lmem_need + lmem_used <= (uint32_t)CV18xx::LMEM_BYTES) {
      break;
    }
  }
  if (step == 0) {
    llvm_unreachable("pad slice failed");
  }
  CV18xx::tdma_load(ml_weight, weight_addr);
  for (int pos = 0; pos < outer_size; pos += step) {
    int in_offset = pos * src_gstride.row;
    int out_offset = pos * dst_gstride.row;
    int N = std::min(step, outer_size - pos);
    auto ishape = CV18xx::ml_default_shape(N, pad, fmt);
    auto ml_input = CV18xx::lmem_alloc_matrix(ishape, fmt, 1);
    auto ml_output = CV18xx::lmem_alloc_matrix(ishape, fmt, 1);
    CV18xx::tdma_load_stride(ml_input, src_addr + in_offset, src_gstride);
    matrix_mul(layer_id, ml_output, ml_input, ml_weight, fmt);
    CV18xx::tdma_store_stride(ml_output, dst_addr + out_offset, dst_gstride);
    CV18xx::lmem_free_matrix(ml_output);
    CV18xx::lmem_free_matrix(ml_input);
  }
  CV18xx::lmem_free_matrix(ml_weight);
}

void cvi_backend_tg_reflectionpad_kernel(uint32_t layer_id, gaddr_t ga_input,
                                         gaddr_t ga_output, gaddr_t ga_left,
                                         gaddr_t ga_right, int outer_size,
                                         int ih, int iw, std::vector<int> &pads,
                                         cvk_fmt_t fmt) {
  CV18xx::set_layer_id(layer_id);
  auto fmt_size = CV18xx::bytesize_of_fmt(fmt);
  // pad 1D (only pad left and right)
  if (pads[2] == 0 && pads[3] == 0) {
    outer_size *= ih;
    ih = 1;
  }
  int oh = ih + pads[2] + pads[3];
  int ow = iw + pads[0] + pads[1];
  // copy middle first
  auto src_gaddr = ga_input;
  auto dst_gaddr = ga_output + (pads[2] * ow + pads[0]) * fmt_size;
  auto src_shape = CV18xx::tg_shape_t4(1, outer_size, ih, iw);
  auto dst_shape = CV18xx::tg_shape_t4(1, outer_size, oh, ow);
  auto src_stride = CV18xx::tg_default_stride(src_shape, fmt);
  auto dst_stride = CV18xx::tg_default_stride(dst_shape, fmt);
  CV18xx::tdma_g2g_tensor_copy(src_gaddr, src_shape, src_stride, fmt, dst_gaddr,
                               src_shape, dst_stride, fmt);
  // top
  src_shape = CV18xx::tg_shape_t4(1, outer_size, 1, iw);
  for (int i = 0; i < pads[2]; ++i) {
    src_gaddr = ga_input + (i + 1) * iw * fmt_size;
    dst_gaddr = ga_output + ((pads[2] - 1 - i) * ow + pads[0]) * fmt_size;
    CV18xx::tdma_g2g_tensor_copy(src_gaddr, src_shape, src_stride, fmt,
                                 dst_gaddr, src_shape, dst_stride, fmt);
  }
  // bottom
  for (int i = 0; i < pads[3]; ++i) {
    src_gaddr = ga_input + (ih - 2 - i) * iw * fmt_size;
    dst_gaddr = ga_output + ((oh - pads[3] + i) * ow + pads[0]) * fmt_size;
    CV18xx::tdma_g2g_tensor_copy(src_gaddr, src_shape, src_stride, fmt,
                                 dst_gaddr, src_shape, dst_stride, fmt);
  }
  outer_size *= oh;
  // left
  src_gaddr = ga_output + (1 + pads[0]) * fmt_size;
  cvk_mg_stride_t out_gstride = {.row = (uint32_t)ow * fmt_size};
  do_reflection(layer_id, src_gaddr, ga_output, ga_left, out_gstride,
                out_gstride, outer_size, pads[0], fmt);
  // right
  src_gaddr = ga_output + (pads[0] + iw - pads[1] - 1) * fmt_size;
  dst_gaddr = ga_output + (ow - pads[1]) * fmt_size;
  do_reflection(layer_id, src_gaddr, dst_gaddr, ga_right, out_gstride,
                out_gstride, outer_size, pads[1], fmt);
}
} // namespace backend
} // namespace tpu_mlir
