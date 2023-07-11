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
#define DEBUG_TYPE "pixelshuffle_kernel"

namespace tpu_mlir {
namespace backend {
static void pixelShuffle_split(uint32_t input_n, uint32_t input_c,
                               uint32_t input_h, uint32_t input_w,
                               uint32_t output_c, uint32_t output_h,
                               uint32_t output_w, uint32_t factor,
                               cvk_fmt_t fmt, int eu_align, uint32_t &h_step,
                               bool isDCR) {
  h_step = input_h;
  uint32_t c = factor * CV18xx::NPU_NUM;
  uint32_t n = factor;

  if (isDCR) {
    // tile for output
    c = input_c / factor / factor;
    n = factor * factor;
  }

  for (; h_step > 0; --h_step) {
    cvk_tl_shape_t tiled_ifmap_shape = {n, c, h_step, input_w};
    uint32_t tiled_ifmap_size =
        CV18xx::lmem_tensor_to_size(tiled_ifmap_shape, fmt, eu_align);

    cvk_tl_shape_t tiled_ofmap_shape = {n, c, h_step, input_w};
    uint32_t tiled_ofmap_size =
        CV18xx::lmem_tensor_to_size(tiled_ofmap_shape, fmt, eu_align);
    uint32_t total_size = tiled_ifmap_size + tiled_ofmap_size;
    if (total_size <= static_cast<uint32_t>(CV18xx::LMEM_BYTES))
      break;
  }

  LLVM_DEBUG(llvm::dbgs() << "  upsample_split:  h_step " << h_step << "\n");

  assert(h_step && "Expect valid upsample tiling");
}

static void pixelShuffle_assign_lmem_layout(
    uint32_t tiling_c, uint32_t tiling_h, uint32_t input_c, uint32_t input_h,
    uint32_t input_w, uint32_t output_c, uint32_t output_h, uint32_t output_w,
    uint32_t factor, cvk_fmt_t fmt, int eu_align, cvk_tl_t &tl_ifmap,
    cvk_tl_t &tl_ofmap, bool isDCR) {

  uint32_t tl_offset = 0; // begin of local memory
  uint32_t n = factor;
  uint32_t c = factor * CV18xx::NPU_NUM;
  if (isDCR) {
    c = input_c / factor / factor;
  }

  CV18xx::lmem_init_tensor(&tl_ifmap, {n, c, tiling_h, input_w}, fmt, eu_align);
  tl_ifmap.start_address = tl_offset;
  auto offset = CV18xx::lmem_tensor_to_size(tl_ifmap.shape, tl_ifmap.fmt,
                                            tl_ifmap.eu_align);
  if (isDCR) {
    cvk_tl_t tl_tmp;
    CV18xx::lmem_init_tensor(&tl_tmp, {factor * factor, c, tiling_h, input_w},
                             fmt, eu_align);
    offset =
        CV18xx::lmem_tensor_to_size(tl_tmp.shape, tl_tmp.fmt, tl_tmp.eu_align);
  }

  tl_offset += offset;

  CV18xx::lmem_init_tensor(&tl_ofmap, tl_ifmap.shape, fmt, eu_align);
  tl_ofmap.start_address = tl_offset;
}

// gap c for take c by gap(oc)
// for dcr mode that layout could be we follow:
// input shape = 1,8,2,2
//    c0          c1         c2        c3          c4        c5         c6 c7
//[[ 0,  1],  [[ 4,  5], [[ 8,  9], [[12, 13], [[16, 17],  [[20, 21], [[24, 25]
//[[28, 29], [ 2,  3]],  [ 6,  7]],  [10, 11]], [14, 15]], [18, 19]], [22, 23]]
//[26, 27]] [30, 31]],,
// it should load c0/c2/c4/c6 as tl channel 0
//                c1/c3/c5/c7 as tl channel 1
static void pixelShuffle_tensor_load_gapc(uint32_t layer_id, gaddr_t ga_ifmap,
                                          cvk_tg_stride_t &ifmap_gstride,
                                          int n_pos, int c_pos, int h_pos,
                                          cvk_fmt_t fmt,
                                          cvk_tl_shape_t tileShape, int oc) {
  uint64_t ga_ifmap_offset = ifmap_gstride.n * n_pos + ifmap_gstride.c * c_pos +
                             ifmap_gstride.h * h_pos;

  cvk_tg_t tg_ifmap = {0};
  tg_ifmap.base_reg_index = CV18xx::getTdmaBaseSelectIndexFromGaddr(ga_ifmap);
  tg_ifmap.fmt = fmt;
  tg_ifmap.start_address = ga_ifmap + ga_ifmap_offset;
  tg_ifmap.shape = {tileShape.c, tileShape.n, tileShape.h, tileShape.w};
  tg_ifmap.stride = ifmap_gstride;
  tg_ifmap.stride.n = tg_ifmap.stride.c * oc;

  cvk_tl_t tl_dst;
  tl_dst.start_address = 0; // Same as ifmap = 0
  tl_dst.fmt = fmt;
  tl_dst.shape = {tileShape.c, tileShape.n, tileShape.h, tileShape.w};
  tl_dst.stride = CV18xx::tl_default_stride(tl_dst.shape, fmt, /*eu_align=*/0);

  CV18xx::tdma_load_stride(&tl_dst, ga_ifmap + ga_ifmap_offset,
                           tg_ifmap.stride);

  LLVM_DEBUG(llvm::dbgs() << "  pixelShuffle_tensor_load_gapc\n"
                          << "    tg offset " << ga_ifmap_offset << ", shape("
                          << tileShape.n << ", " << tileShape.c << ", "
                          << tileShape.h << "), stride(" << tg_ifmap.stride.n
                          << ", " << tg_ifmap.stride.c << ", "
                          << tg_ifmap.stride.h << ")\n");
}

static void pixelShuffle_tensor_load_nc_transpose(
    uint32_t layer_id, gaddr_t ga_ifmap, cvk_tg_stride_t &ifmap_gstride,
    int n_pos, int c_pos, int h_pos, cvk_fmt_t fmt, cvk_tl_shape_t tileShape) {
  uint64_t ga_ifmap_offset = ifmap_gstride.n * n_pos + ifmap_gstride.c * c_pos +
                             ifmap_gstride.h * h_pos;

  cvk_tg_t tg_ifmap = {0};
  tg_ifmap.base_reg_index = CV18xx::getTdmaBaseSelectIndexFromGaddr(ga_ifmap);
  tg_ifmap.fmt = fmt;
  tg_ifmap.start_address = ga_ifmap + ga_ifmap_offset;
  tg_ifmap.shape = {tileShape.n, tileShape.c, tileShape.h, tileShape.w};
  tg_ifmap.stride = ifmap_gstride;
  tg_ifmap.stride.n = tg_ifmap.stride.c * tileShape.c;

  cvk_tl_t tl_dst;
  tl_dst.start_address = 0; // Same as ifmap = 0
  tl_dst.fmt = fmt;
  tl_dst.shape = {tileShape.c, tileShape.n, tileShape.h, tileShape.w};
  tl_dst.stride = CV18xx::tl_default_stride(tl_dst.shape, fmt, /*eu_align=*/0);

  cvk_tdma_g2l_tensor_copy_nc_transposed_param_t param = {0};
  param.src = &tg_ifmap;
  param.dst = &tl_dst;
  param.layer_id = layer_id;

  LLVM_DEBUG(llvm::dbgs() << "  pixelShuffle_tensor_load_nc_transpose\n"
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

  CV18xx::tdma_g2l_tensor_copy_nc_transposed(&param);
}

static void pixelShuffle_tiu_copy(uint32_t layer_id, cvk_tl_t *tl_ifmap,
                                  cvk_tl_t *tl_ofmap, int factor, bool isDCR) {

  cvk_tl_t tl_dst;
  tl_dst.start_address = tl_ofmap->start_address; // start of lmem
  tl_dst.fmt = tl_ofmap->fmt;
  tl_dst.shape = tl_ofmap->shape;
  int bytesize = tl_ofmap->stride.w;
  tl_dst.stride = {(uint32_t)(factor * tl_ofmap->shape.w * bytesize),
                   (uint32_t)bytesize,
                   (uint32_t)(factor * factor * tl_ofmap->shape.w * bytesize),
                   (uint32_t)(factor * bytesize)};

  if (isDCR) {
    // w direction, -1 we calculate c distance that wrap by CV18xx::NPU_NUM
    tl_dst.stride.c =
        factor * factor * tl_ofmap->shape.w * tl_ofmap->shape.h * bytesize;
    tl_dst.stride.n = (factor - 1) * bytesize;
  }

  cvk_tiu_copy_param_t p2 = {0};
  p2.src = tl_ifmap;
  p2.dst = &tl_dst;
  p2.layer_id = layer_id;

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "    [%d] L2L Reshape:\n"
                 "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, "
                 "%d, %d, %d)\n"
                 "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, "
                 "%d, %d, %d)\n",
                 p2.src->start_address, p2.src->shape.n, p2.src->shape.c,
                 p2.src->shape.h, p2.src->shape.w, p2.src->stride.n,
                 p2.src->stride.c, p2.src->stride.h, p2.src->stride.w,
                 p2.dst->start_address, p2.dst->shape.n, p2.dst->shape.c,
                 p2.dst->shape.h, p2.dst->shape.w, p2.dst->stride.n,
                 p2.dst->stride.c, p2.dst->stride.h, p2.dst->stride.w));
  CV18xx::tiu_copy(&p2);

  if (isDCR) {
    // h direction
    assert(factor == 2);
    for (int i = 1; i < factor; i++) {
      cvk_tl_t _tl_ifmap = *tl_ifmap;
      _tl_ifmap.start_address =
          tl_ifmap->start_address + i * factor * tl_ifmap->stride.n;
      tl_dst.start_address =
          tl_ofmap->start_address + factor * tl_ofmap->shape.w * bytesize;

      p2.src = &_tl_ifmap;
      p2.dst = &tl_dst;
      CV18xx::tiu_copy(&p2);
    }
  }
}

static void pixelShuffle_tensor_store(uint32_t layer_id, gaddr_t ga_ofmap,
                                      cvk_tl_t *tl_ofmap,
                                      cvk_tg_stride_t &ofmap_gstride, int n_pos,
                                      int oc_pos, int h_pos,
                                      cvk_tl_shape_t tileShape) {
  uint64_t ga_ofmap_offset = ofmap_gstride.n * n_pos +
                             ofmap_gstride.c * oc_pos + ofmap_gstride.h * h_pos;

  cvk_tg_t tg_ofmap = {0};
  tg_ofmap.base_reg_index = CV18xx::getTdmaBaseSelectIndexFromGaddr(ga_ofmap);
  tg_ofmap.fmt = tl_ofmap->fmt;
  tg_ofmap.start_address = ga_ofmap + ga_ofmap_offset;
  tg_ofmap.shape = {tileShape.n, tileShape.c, tileShape.h, tileShape.w};
  tg_ofmap.stride = ofmap_gstride;

  cvk_tl_t tl_dst;
  tl_dst.start_address = tl_ofmap->start_address; // start of lmem
  tl_dst.fmt = tl_ofmap->fmt;
  tl_dst.shape = tileShape;
  tl_dst.stride =
      CV18xx::tl_default_stride(tl_dst.shape, tl_ofmap->fmt, /*eu_align=*/0);

  cvk_tdma_l2g_tensor_copy_param_t param = {0};
  param.src = &tl_dst;
  param.dst = &tg_ofmap;
  param.layer_id = layer_id;

  LLVM_DEBUG(llvm::dbgs() << "  pixelShuffle_tensor_store\n"
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

static void _pixel_shuffle_fixed_kernel_new(uint32_t layer_id, gaddr_t ga_ifmap,
                                            gaddr_t ga_ofmap, uint32_t input_n,
                                            uint32_t input_c, uint32_t input_h,
                                            uint32_t input_w, uint32_t factor,
                                            bool isDCR, cvk_fmt_t fmt) {

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "pixel_shuffle_fixed_bmkernel:\n"
                 "  ga_ifmap 0x%lx, ga_ofmap 0x%lx, shape(%d, %d, %d, %d)\n"
                 "  h_factor %d, w_factor %d\n",
                 ga_ifmap, ga_ofmap, input_n, input_c, input_h, input_w, factor,
                 factor));

  uint32_t in = input_n;
  uint32_t ic = input_c;
  uint32_t ih = input_h;
  uint32_t iw = input_w;
  uint32_t oc = input_c / (factor * factor);
  uint32_t oh = input_h * factor;
  uint32_t ow = input_w * factor;

  int eu_align = 0; // no need to align eu

  uint32_t n_step = 1;
  uint32_t oc_step = (oc >= (uint32_t)CV18xx::NPU_NUM) ? CV18xx::NPU_NUM : oc;
  uint32_t c_step = oc_step * factor * factor;
  uint32_t h_step = 0;

  if (isDCR) {
    // leverage whole channel
    oc_step = oc;
    c_step = oc_step * factor * factor;
  }

  pixelShuffle_split(in, ic, ih, iw, oc, oh, ow, factor, fmt, eu_align, h_step,
                     isDCR);
  if (!h_step)
    return;

  // TODO: support other fmt
  // 2 means bf16 takes 2 bytes
  int unit_sz = fmt == CVK_FMT_BF16 ? 2 : 1;
  cvk_tg_stride_t ifmap_gstride = {input_c * input_h * input_w * unit_sz,
                                   input_h * input_w * unit_sz,
                                   input_w * unit_sz};
  cvk_tg_stride_t ofmap_gstride = {oc * oh * ow * unit_sz, oh * ow * unit_sz,
                                   ow * unit_sz};

  for (uint32_t n_pos = 0; n_pos < input_n; n_pos += n_step) {
    for (uint32_t c_pos = 0; c_pos < input_c; c_pos += c_step) {
      uint32_t tiling_c = std::min(input_c - c_pos, c_step);
      for (uint32_t h_pos = 0; h_pos < input_h; h_pos += h_step) {
        uint32_t tiling_h = std::min(input_h - h_pos, h_step);
        // 1. Assign local memory layout
        cvk_tl_t tl_ifmap, tl_ofmap;
        pixelShuffle_assign_lmem_layout(tiling_c, tiling_h, ic, ih, iw, oc, oh,
                                        ow, factor, fmt, eu_align, tl_ifmap,
                                        tl_ofmap, isDCR);
        // 2. tensor load
        cvk_tl_shape_t tileShape = {oc_step, factor * factor, tiling_h, iw};
        if (isDCR) {
          pixelShuffle_tensor_load_gapc(layer_id, ga_ifmap, ifmap_gstride,
                                        n_pos, c_pos, h_pos, fmt, tileShape,
                                        oc);
        } else {
          pixelShuffle_tensor_load_nc_transpose(layer_id, ga_ifmap,
                                                ifmap_gstride, n_pos, c_pos,
                                                h_pos, fmt, tileShape);
        }

        // 3. tiu copy
        pixelShuffle_tiu_copy(layer_id, &tl_ifmap, &tl_ofmap, factor, isDCR);

        // 4. tensor store
        uint32_t oc_pos = c_pos / (factor * factor);
        uint32_t oh_pos = h_pos * factor;
        cvk_tl_shape_t outputTileShape = {1, tiling_c / (factor * factor),
                                          tiling_h * factor, ow};
        pixelShuffle_tensor_store(layer_id, ga_ofmap, &tl_ofmap, ofmap_gstride,
                                  n_pos, oc_pos, oh_pos, outputTileShape);
      }
    } // for (uint32_t c_pos = 0; c_pos < input_c; c_pos += c_step) {
  }   // for (uint32_t n_pos = 0; n_pos < input_n; n_pos += n_step)
}

void cvi_backend_tg_fixed_pixel_shuffle_kernel(
    uint32_t layer_id, gaddr_t ga_ifmap, gaddr_t ga_ofmap, int input_n,
    int input_c, int input_h, int input_w, int factor, bool isDCR) {

  // For tdma
  CV18xx::set_layer_id(layer_id);

  _pixel_shuffle_fixed_kernel_new(layer_id, ga_ifmap, ga_ofmap,
                                  (uint32_t)input_n, (uint32_t)input_c,
                                  (uint32_t)input_h, (uint32_t)input_w,
                                  (uint32_t)factor, isDCR, CVK_FMT_I8);
}

void cvi_backend_tg_bf16_pixel_shuffle_kernel(
    uint32_t layer_id, gaddr_t ga_ifmap, gaddr_t ga_ofmap, int input_n,
    int input_c, int input_h, int input_w, int factor, bool isDCR) {
  // For tdma
  CV18xx::set_layer_id(layer_id);

  _pixel_shuffle_fixed_kernel_new(layer_id, ga_ifmap, ga_ofmap,
                                  (uint32_t)input_n, (uint32_t)input_c,
                                  (uint32_t)input_h, (uint32_t)input_w,
                                  (uint32_t)factor, isDCR, CVK_FMT_BF16);
}

} // namespace backend
} // namespace tpu_mlir
