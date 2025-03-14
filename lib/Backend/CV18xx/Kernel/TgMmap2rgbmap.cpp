//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Backend/CV18xx/Kernel/TgMmap2rgbmap.hpp"
#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include "tpu_mlir/Support/LutFunc.h"

#define DEBUG_TYPE "cvi_backend_mmap2rgbmap_kernel"
using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace backend {

void TgMMap2RGBMapKernel::init(uint32_t layer_id, gaddr_t ga_input,
                               gaddr_t ga_output, int n, int c, int h, int w,
                               int block_size, cvk_fmt_t fmt) {
  this->layer_id = layer_id;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->fmt = fmt;
  assert(fmt == CVK_FMT_BF16 || fmt == CVK_FMT_U8);
  this->in_data_size = n * c * h * w;
  assert(in_data_size % CV18xx::NPU_NUM == 0);
  this->n = 1;
  this->c = CV18xx::NPU_NUM;
  this->h = in_data_size / this->c;
  this->w = 1;
  this->block_size = block_size;
  CV18xx::set_layer_id(layer_id);
}

// tile w dims
void TgMMap2RGBMapKernel::selectTilePolicy() {
  CV18xx::tiling_info_t i_tile;
  CV18xx::tiling_info_t o_tile;
  i_tile.n = n;
  i_tile.pos_n = 0;
  i_tile.c = c;
  i_tile.pos_c = 0;
  i_tile.w = w;
  i_tile.pos_w = 0;
  o_tile.n = n;
  o_tile.pos_n = 0;
  o_tile.c = c;
  o_tile.pos_c = 0;
  o_tile.w = block_size;
  o_tile.pos_w = 0;

  int tile_h = h;
  int blob_num = 1; // need only one buffer
  // slice h dim
  for (int step_h = h; step_h >= 1; step_h /= 2) {
    uint32_t lmem_required = 0;
    if (fmt == CVK_FMT_U8) {
      cvk_tl_shape_t max_shape = CV18xx::tl_shape_t4(n, c, step_h, block_size);
      // dst format is always CVK_FMT_U8
      lmem_required =
          blob_num * CV18xx::lmem_tensor_to_size(max_shape, CVK_FMT_U8, 1);
    } else {
      cvk_tl_shape_t buffer_shape = CV18xx::tl_shape_t4(n, c, step_h, w);
      cvk_tl_shape_t cast_shape = CV18xx::tl_shape_t4(n, c, step_h, block_size);
      // dst format is always CVK_FMT_U8
      lmem_required =
          blob_num * (CV18xx::lmem_tensor_to_size(buffer_shape, fmt, 1) +
                      CV18xx::lmem_tensor_to_size(cast_shape, CVK_FMT_U8, 1));
    }
    if (lmem_required <= CV18xx::LMEM_BYTES) {
      tile_h = step_h;
      break;
    } else {
      if (step_h == 1) {
        assert(0 && "Tiling failed");
      }
    }
  }

  int itype_bytesize = (fmt == CVK_FMT_U8) ? 1 : 2;
  int otype_bytesize = 1;
  // store tiles
  for (i_tile.pos_h = 0; i_tile.pos_h < h; i_tile.pos_h += tile_h) {
    i_tile.h = std::min(h - i_tile.pos_h, tile_h);
    i_tile.offset = n * c * i_tile.pos_h * w * itype_bytesize;
    o_tile.h = i_tile.h;
    o_tile.pos_h = i_tile.pos_h;
    o_tile.offset = n * c * o_tile.pos_h * block_size * otype_bytesize;
    input_tiles.emplace_back(i_tile);
    output_tiles.emplace_back(o_tile);
  }
}

void TgMMap2RGBMapKernel::allocLmem() {
  if (fmt == CVK_FMT_U8) {
    cvk_tl_shape_t l_shape =
        CV18xx::tl_shape_t4(n, c, output_tiles[0].h, block_size);
    tl_buffer = CV18xx::lmem_alloc_tensor(l_shape, CVK_FMT_U8, 1);
  } else {
    cvk_tl_shape_t buffer_shape =
        CV18xx::tl_shape_t4(n, c, output_tiles[0].h, w);
    cvk_tl_shape_t cast_shape =
        CV18xx::tl_shape_t4(n, c, output_tiles[0].h, block_size);
    tl_buffer = CV18xx::lmem_alloc_tensor(buffer_shape, fmt, 1);
    tl_cast = CV18xx::lmem_alloc_tensor(cast_shape, CVK_FMT_U8, 1);
  }
}

void TgMMap2RGBMapKernel::deallocLmem() {
  if (fmt == CVK_FMT_U8) {
    CV18xx::lmem_free_tensor(tl_buffer);
  } else {
    CV18xx::lmem_free_tensor(tl_cast);
    CV18xx::lmem_free_tensor(tl_buffer);
  }
}

// bf16/u8 -> u8,
void TgMMap2RGBMapKernel::load(int32_t step_idx) {
  auto tile = input_tiles[step_idx];
  if (fmt == CVK_FMT_U8) {
    // src info
    cvk_tg_shape_t gshape = CV18xx::tg_shape_t4(n, c, tile.h, w);
    cvk_tg_t src;
    src.start_address = ga_input + tile.offset;
    src.base_reg_index =
        CV18xx::getTdmaBaseSelectIndexFromGaddr(src.start_address);
    src.fmt = fmt;
    src.shape = gshape;
    cvk_tg_stride_t gstride = CV18xx::tg_default_stride(gshape, fmt);
    src.stride = gstride;
    src.int8_rnd_mode = 0;
    // dst info
    cvk_tl_shape_t lshape = CV18xx::tl_shape_t4(n, c, tile.h, w);
    cvk_tl_t dst = *tl_buffer;
    dst.shape = lshape;
    cvk_tl_stride_t lstride;
    lstride.w = block_size;
    lstride.h = w * lstride.w;
    lstride.c = align_up(tile.h * lstride.h, CV18xx::tiu_eu_num(CVK_FMT_U8));
    lstride.n = lstride.c * ceiling_func(c, CV18xx::NPU_NUM);
    dst.stride = lstride;
    dst.int8_rnd_mode = 0;
    // cpy
    cvk_tdma_g2l_tensor_copy_param_t p1 = {0};
    p1.src = &src;
    p1.dst = &dst;
    CV18xx::tdma_g2l_tensor_copy(&p1);
  } else {
    // g2l load
    cvk_tg_shape_t gshape = CV18xx::tg_shape_t4(n, c, tile.h, w);
    cvk_tg_stride_t gstride = CV18xx::tg_default_stride(gshape, fmt);
    CV18xx::tdma_load_stride(tl_buffer, ga_input + tile.offset, gstride);
    // l2l bf16->u8
    cvk_tl_shape_t lshape = CV18xx::tl_shape_t4(n, c, tile.h, w);
    cvk_tl_t src = *tl_buffer;
    cvk_tl_t dst = *tl_cast;
    dst.shape = lshape;
    cvk_tl_stride_t lstride;
    lstride.w = block_size;
    lstride.h = w * lstride.w;
    lstride.c = align_up(tile.h * lstride.h, CV18xx::tiu_eu_num(CVK_FMT_U8));
    lstride.n = lstride.c * ceiling_func(c, CV18xx::NPU_NUM);
    dst.stride = lstride;
    dst.int8_rnd_mode = 0;
    cvk_tdma_l2l_tensor_copy_param_t p1 = {0};
    p1.src = &src;
    p1.dst = &dst;
    CV18xx::tdma_l2l_tensor_copy(&p1);
  }
}

void TgMMap2RGBMapKernel::store(int32_t step_idx) {
  auto tile = output_tiles[step_idx];
  // src info
  cvk_tl_t src = (fmt == CVK_FMT_U8) ? (*tl_buffer) : (*tl_cast);
  cvk_tl_shape_t lshape = CV18xx::tl_shape_t4(n, c, tile.h, block_size);
  src.shape = lshape;
  cvk_tl_stride_t lstride = CV18xx::tl_default_stride(lshape, CVK_FMT_U8, 1);
  src.stride = lstride;
  // dst info
  cvk_tg_shape_t gshape = CV18xx::tg_shape_t4(n, c, tile.h, block_size);
  uint64_t dst_gaddr = ga_output + tile.offset;
  cvk_tg_stride_t gstride = CV18xx::tg_default_stride(gshape, CVK_FMT_U8);
  CV18xx::tdma_store_stride(&src, dst_gaddr, gstride);
}

void TgMMap2RGBMapKernel::compute(int32_t step_idx) {
  // no compute, tdma_l2l do data_convert
}

void TgMMap2RGBMapKernel::schedule() {
  allocLmem();
  int times = input_tiles.size();
  // set buffer zero
  cvk_tdma_g2l_tensor_fill_constant_param_t p1 = {0};
  p1.constant = 0;
  p1.dst = (fmt == CVK_FMT_U8) ? tl_buffer : tl_cast;
  p1.layer_id = layer_id;
  CV18xx::parallel_disable();
  CV18xx::tdma_g2l_tensor_fill_constant(&p1);
  for (int i = 0; i < times; i++) {
    load(i);
    store(i);
  }
  deallocLmem();
}

void cvi_backend_tg_mmap2rgbmap_kernel(uint32_t layer_id, gaddr_t ga_input,
                                       gaddr_t ga_output, int n, int c, int h,
                                       int w, int block_size, cvk_fmt_t fmt) {
  TgMMap2RGBMapKernel kernel;
  kernel.init(layer_id, ga_input, ga_output, n, c, h, w, block_size, fmt);
  kernel.selectTilePolicy();
  kernel.schedule();
}

} // namespace backend
} // namespace tpu_mlir
