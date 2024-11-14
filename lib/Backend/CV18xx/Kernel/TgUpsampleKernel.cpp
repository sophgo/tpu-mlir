//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <llvm/Support/Debug.h>

#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"

#define DEBUG_TYPE "cvi_backend_upsample_kernel"

namespace tpu_mlir {
namespace backend {
static void upsample_split(uint32_t input_n, uint32_t input_c, uint32_t input_h,
                           uint32_t input_w, uint32_t output_c,
                           uint32_t output_h, uint32_t output_w, cvk_fmt_t fmt,
                           int eu_align, uint32_t &c_step, uint32_t &h_step) {

  h_step = input_h;
  uint32_t h_factor = output_h / input_h;

  for (; h_step > 0; --h_step) {
    uint32_t total_size;
    c_step = std::max(input_c, (uint32_t)CV18xx::NPU_NUM);
    for (; c_step >= (uint32_t)CV18xx::NPU_NUM;
         --c_step) { // at least c = CV18xx::NPU_NUM
      cvk_tl_shape_t tiled_ifmap_shape = {1, c_step, h_step, input_w};
      uint32_t tiled_ifmap_size =
          CV18xx::lmem_tensor_to_size(tiled_ifmap_shape, fmt, eu_align);

      cvk_tl_shape_t tiled_working_shape = {1, c_step, h_step, output_w};
      uint32_t tiled_working_size =
          CV18xx::lmem_tensor_to_size(tiled_working_shape, fmt, eu_align);

      cvk_tl_shape_t tiled_ofmap_shape = {1, c_step, h_step * h_factor,
                                          output_w};
      uint32_t tiled_ofmap_size =
          CV18xx::lmem_tensor_to_size(tiled_ofmap_shape, fmt, eu_align);

      total_size = tiled_ifmap_size + tiled_working_size + tiled_ofmap_size;
      if (total_size <= static_cast<uint32_t>(CV18xx::LMEM_BYTES))
        break;
    }
    if (total_size <= static_cast<uint32_t>(CV18xx::LMEM_BYTES))
      break;
  }

  LLVM_DEBUG(llvm::dbgs() << "  upsample_split: c_step " << c_step
                          << " ,   h_step " << h_step << "\n");

  assert(c_step && h_step && "Expect valid upsample tiling");
}

static void upsample_assign_lmem_layout(
    uint32_t tiling_c, uint32_t tiling_h, uint32_t input_c, uint32_t input_h,
    uint32_t input_w, uint32_t output_c, uint32_t output_h, uint32_t output_w,
    cvk_fmt_t fmt, int eu_align, cvk_tl_t &tl_ifmap, cvk_tl_t &tl_working,
    cvk_tl_t &tl_ofmap) {

  uint32_t tl_offset = 0; // begin of local memory
  CV18xx::lmem_init_tensor(&tl_ifmap, {1, tiling_c, tiling_h, input_w}, fmt,
                           eu_align);
  tl_ifmap.start_address = tl_offset;
  tl_offset += CV18xx::lmem_tensor_to_size(tl_ifmap.shape, tl_ifmap.fmt,
                                           tl_ifmap.eu_align);

  CV18xx::lmem_init_tensor(&tl_working, {1, tiling_c, tiling_h, output_w}, fmt,
                           eu_align);
  tl_working.start_address = tl_offset;
  tl_offset += CV18xx::lmem_tensor_to_size(tl_working.shape, tl_working.fmt,
                                           tl_working.eu_align);

  uint32_t tiling_oh = tiling_h * (output_h / input_h);
  CV18xx::lmem_init_tensor(&tl_ofmap, {1, tiling_c, tiling_oh, output_w}, fmt,
                           eu_align);
  tl_ofmap.start_address = tl_offset;
}

static void upsample_tensor_load(uint32_t layer_id, gaddr_t ga_ifmap,
                                 cvk_tg_stride_t &ifmap_gstride, int n_pos,
                                 int c_pos, int h_pos, cvk_tl_t *tl_ifmap) {
  uint64_t ga_ifmap_offset = ifmap_gstride.n * n_pos + ifmap_gstride.c * c_pos +
                             ifmap_gstride.h * h_pos;

  cvk_tg_t tg_ifmap = {0};
  tg_ifmap.base_reg_index = CV18xx::getTdmaBaseSelectIndexFromGaddr(ga_ifmap);
  tg_ifmap.fmt = tl_ifmap->fmt;
  tg_ifmap.start_address = ga_ifmap + ga_ifmap_offset;
  tg_ifmap.shape = {tl_ifmap->shape.n, tl_ifmap->shape.c, tl_ifmap->shape.h,
                    tl_ifmap->shape.w};
  tg_ifmap.stride = ifmap_gstride;

  cvk_tdma_g2l_tensor_copy_param_t param = {0};
  param.src = &tg_ifmap;
  param.dst = tl_ifmap;
  param.layer_id = layer_id;

  LLVM_DEBUG(llvm::dbgs() << "  upsample_tensor_load\n"
                          << "    tg offset " << ga_ifmap_offset << ", shape("
                          << param.src->shape.n << ", " << param.src->shape.c
                          << ", " << param.src->shape.h << "), stride("
                          << param.src->stride.n << ", " << param.src->stride.c
                          << ", " << param.src->stride.h << ")\n"
                          << "    tl shape(" << param.dst->shape.n << ", "
                          << param.dst->shape.c << ", " << param.dst->shape.h
                          << ", " << param.dst->shape.w << "), stride("
                          << param.dst->stride.n << ", " << param.dst->stride.c
                          << ", " << param.dst->stride.h << ", "
                          << param.dst->stride.w << ")\n");

  CV18xx::tdma_g2l_tensor_copy(&param);
}

/*

  shape (n, c, 2, 2), h_factor 2, w_factor 2

  input (n, c, 2, 2),     output (n, c, 4, 4)
  ________               ________________
  | 1 | 2 |               | 1 | 1 | 2 | 2 |
  ________               ________________
  | 3 | 4 |               | 1 | 1 | 2 | 2 |
  ________               ________________
                          | 3 | 3 | 4 | 4 |
                          ________________
                          | 3 | 3 | 4 | 4 |
                          ________________

  Sub output shape (2, 2) in terms of output stride
  Sub_output_stride = (output_n_stride,
                       output_c_stride,
                       h_factor * output_h_stride,
                       w_factor * output_w_stride)

  | 4 4 3 3 | 4 4 3 3 | 2 2 1 1 | 2 2 1 1 |  C0
                                    -   -
                                    _____
                                    ws = w_factor * w_stride

                    -                   -
                    ---------------------
                    hs = h_factor * h_stride
  | x x x x | x x x x | x x x x | x x x x |  C1

*/
static void upsample_tiu_copy(uint32_t layer_id, uint32_t h_factor,
                              uint32_t w_factor, cvk_tl_t *tl_ifmap,
                              cvk_tl_t *tl_working, cvk_tl_t *tl_ofmap) {

  LLVM_DEBUG(llvm::dbgs()
             << "  upsample_tiu_copy, layer_id " << layer_id << ", h_factor "
             << h_factor << ", w_factor " << w_factor << "\n"
             << "    ifmap addr " << tl_ifmap->start_address << ", shape("
             << tl_ifmap->shape.n << ", " << tl_ifmap->shape.c << ", "
             << tl_ifmap->shape.h << ", " << tl_ifmap->shape.w << "), stride("
             << tl_ifmap->stride.n << ", " << tl_ifmap->stride.c << ", "
             << tl_ifmap->stride.h << ", " << tl_ifmap->stride.w << ")\n"
             << "    working addr " << tl_working->start_address << ", shape("
             << tl_working->shape.n << ", " << tl_working->shape.c << ", "
             << tl_working->shape.h << ", " << tl_working->shape.w
             << "), stride(" << tl_working->stride.n << ", "
             << tl_working->stride.c << ", " << tl_working->stride.h << ", "
             << tl_working->stride.w << ")\n"
             << "    ofmap address " << tl_ofmap->start_address << ", shape("
             << tl_ofmap->shape.n << ", " << tl_ofmap->shape.c << ", "
             << tl_ofmap->shape.h << ", " << tl_ofmap->shape.w << "), stride("
             << tl_ofmap->stride.n << ", " << tl_ofmap->stride.c << ", "
             << tl_ofmap->stride.h << ", " << tl_ofmap->stride.w << ")\n");

  // expand output width

  cvk_tl_stride_t tl_ifmap_fake_stride = {
      0, tl_ifmap->stride.c, tl_ifmap->stride.h, tl_ifmap->stride.w};
  cvk_tl_t tl_ifmap_fake = {0};
  tl_ifmap_fake.start_address = tl_ifmap->start_address;
  tl_ifmap_fake.fmt = tl_ifmap->fmt;
  tl_ifmap_fake.shape = {w_factor, tl_ifmap->shape.c, tl_ifmap->shape.h,
                         tl_ifmap->shape.w};
  tl_ifmap_fake.stride = tl_ifmap_fake_stride;
  tl_ifmap_fake.eu_align = tl_ifmap->eu_align;

  cvk_tl_stride_t tl_working_fake_stride = {
      tl_working->stride.w, tl_working->stride.c, tl_working->stride.h,
      tl_working->stride.w * w_factor};
  cvk_tl_t tl_working_fake = {0};
  tl_working_fake.start_address = tl_working->start_address;
  tl_working_fake.fmt = tl_working->fmt;
  tl_working_fake.shape = {w_factor, tl_ifmap->shape.c, tl_ifmap->shape.h,
                           tl_ifmap->shape.w};
  tl_working_fake.stride = tl_working_fake_stride;
  tl_working_fake.eu_align = tl_working->eu_align;

  cvk_tiu_copy_param_t param = {0};
  param.dst = &tl_working_fake;
  param.src = &tl_ifmap_fake;
  param.layer_id = layer_id;
  CV18xx::tiu_copy(&param);

  cvk_tl_stride_t tl_working_fake2_stride = {
      0, tl_working->stride.c, tl_working->stride.h, tl_working->stride.w};
  cvk_tl_t tl_working_fake2 = {0};
  tl_working_fake2.start_address = tl_working->start_address;
  tl_working_fake2.fmt = tl_working->fmt;
  tl_working_fake2.shape = {h_factor, tl_ofmap->shape.c, tl_ifmap->shape.h,
                            tl_ofmap->shape.w};
  tl_working_fake2.stride = tl_working_fake2_stride;
  tl_working_fake2.eu_align = tl_working->eu_align;

  cvk_tl_stride_t tl_ofmap_fake_stride = {
      tl_ofmap->stride.h, tl_ofmap->stride.c, tl_ofmap->stride.h * h_factor,
      tl_ofmap->stride.w};
  cvk_tl_t tl_ofmap_fake = {0};
  tl_ofmap_fake.start_address = tl_ofmap->start_address;
  tl_ofmap_fake.fmt = tl_ofmap->fmt;
  tl_ofmap_fake.shape = {h_factor, tl_ofmap->shape.c, tl_ifmap->shape.h,
                         tl_ofmap->shape.w};
  tl_ofmap_fake.stride = tl_ofmap_fake_stride;
  tl_ofmap_fake.eu_align = tl_ofmap->eu_align;

  cvk_tiu_copy_param_t param2 = {0};
  param2.dst = &tl_ofmap_fake;
  param2.src = &tl_working_fake2;
  param2.layer_id = layer_id;
  CV18xx::tiu_copy(&param2);
}

static void upsample_tensor_store(uint32_t layer_id, gaddr_t ga_ofmap,
                                  cvk_tl_t *tl_ofmap,
                                  cvk_tg_stride_t &ofmap_gstride, int n_pos,
                                  int c_pos, int oh_pos) {
  uint64_t ga_ofmap_offset = ofmap_gstride.n * n_pos + ofmap_gstride.c * c_pos +
                             ofmap_gstride.h * oh_pos;

  cvk_tg_t tg_ofmap = {0};
  tg_ofmap.base_reg_index = CV18xx::getTdmaBaseSelectIndexFromGaddr(ga_ofmap);
  tg_ofmap.fmt = tl_ofmap->fmt;
  tg_ofmap.start_address = ga_ofmap + ga_ofmap_offset;
  tg_ofmap.shape = {tl_ofmap->shape.n, tl_ofmap->shape.c, tl_ofmap->shape.h,
                    tl_ofmap->shape.w};
  tg_ofmap.stride = ofmap_gstride;

  cvk_tdma_l2g_tensor_copy_param_t param = {0};
  param.src = tl_ofmap;
  param.dst = &tg_ofmap;
  param.layer_id = layer_id;

  LLVM_DEBUG(llvm::dbgs() << "  upsample_tensor_store\n"
                          << "    tl shape(" << param.src->shape.n << ", "
                          << param.src->shape.c << ", " << param.src->shape.h
                          << ", " << param.src->shape.w << "), stride("
                          << param.src->stride.n << ", " << param.src->stride.c
                          << ", " << param.src->stride.h << ", "
                          << param.src->stride.w << ")\n"
                          << "    tg offset " << ga_ofmap_offset << ", shape("
                          << param.dst->shape.n << ", " << param.dst->shape.c
                          << ", " << param.dst->shape.h << ", "
                          << param.dst->shape.w << "), stride("
                          << param.dst->stride.n << ", " << param.dst->stride.c
                          << ", " << param.dst->stride.h << ")\n");

  CV18xx::tdma_l2g_tensor_copy(&param);
}

void cvi_backend_tg_upsample_kernel(uint32_t layer_id, gaddr_t ga_ifmap,
                                    gaddr_t ga_ofmap, uint32_t input_n,
                                    uint32_t input_c, uint32_t input_h,
                                    uint32_t input_w, uint32_t h_factor,
                                    uint32_t w_factor, cvk_fmt_t fmt) {

  LLVM_DEBUG(llvm::dbgs() << llvm::format(
                 "cvi_backend_tg_upsample:\n"
                 "  ga_ifmap 0x%lx, ga_ofmap 0x%lx, shape(%d, %d, %d, %d)\n"
                 "  h_factor %d, w_factor %d\n",
                 ga_ifmap, ga_ofmap, input_n, input_c, input_h, input_w,
                 h_factor, w_factor));

  // For tdma
  CV18xx::set_layer_id(layer_id);

  uint32_t output_c = input_c;
  uint32_t output_h = input_h * h_factor;
  uint32_t output_w = input_w * w_factor;

  int eu_align = 0;    // no need to align eu
  uint32_t n_step = 1; // support only batch = 1 now
  uint32_t c_step = 0;
  uint32_t h_step = 0;
  upsample_split(input_n, input_c, input_h, input_w, output_c, output_h,
                 output_w, fmt, eu_align, c_step, h_step);
  if (!c_step)
    return;

  // Global stride from global shape
  int fmt_size = (fmt == CVK_FMT_BF16) ? 2 : 1;
  cvk_tg_stride_t ifmap_gstride = {input_c * input_h * input_w * fmt_size,
                                   input_h * input_w * fmt_size,
                                   input_w * fmt_size};
  cvk_tg_stride_t ofmap_gstride = {output_c * output_h * output_w * fmt_size,
                                   output_h * output_w * fmt_size,
                                   output_w * fmt_size};

  for (uint32_t n_pos = 0; n_pos < input_n; n_pos += n_step) {
    for (uint32_t c_pos = 0; c_pos < input_c; c_pos += c_step) {
      uint32_t tiling_c = std::min(input_c - c_pos, c_step);
      for (uint32_t h_pos = 0; h_pos < input_h; h_pos += h_step) {
        uint32_t tiling_h = std::min(input_h - h_pos, h_step);
        // 1. Assign local memory layout
        cvk_tl_t tl_ifmap, tl_ofmap, tl_working;
        upsample_assign_lmem_layout(tiling_c, tiling_h, input_c, input_h,
                                    input_w, output_c, output_h, output_w, fmt,
                                    eu_align, tl_ifmap, tl_working, tl_ofmap);
        // 2. tensor load
        upsample_tensor_load(layer_id, ga_ifmap, ifmap_gstride, n_pos, c_pos,
                             h_pos, &tl_ifmap);

        // 3. tiu copy
        upsample_tiu_copy(layer_id, h_factor, w_factor, &tl_ifmap, &tl_working,
                          &tl_ofmap);

        // 4. tensor store
        uint32_t oh_pos = h_pos * h_factor;
        upsample_tensor_store(layer_id, ga_ofmap, &tl_ofmap, ofmap_gstride,
                              n_pos, c_pos, oh_pos);
      }
    } // for (uint32_t c_pos = 0; c_pos < input_c; c_pos += c_step) {
  }   // for (uint32_t n_pos = 0; n_pos < input_n; n_pos += n_step)
}
} // namespace backend
} // namespace tpu_mlir
