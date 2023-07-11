//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/Kernel/TgReorgKernel.hpp"
#define DEBUG_TYPE "reorg_kernel"

namespace tpu_mlir {
namespace backend {
// [n, c, oh*r, ow*r] => [n, r*r*c, oh, ow]
// steps:
// 1. use n loops, then [1, c, oh*r, ow*r] load to [r, c*oh, 1, ow*r]
// 2. reshape to [r, c*oh, ow, r], and swap n/h to [ow, c*oh, r, r]
// 3. reshape to [ow, c*oh, 1, r*r], and swap n/w to [r*r, c*oh, 1, ow]

void TgReorgKernel::init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
                         int n, int c, int h, int w, int stride,
                         cvk_fmt_t fmt) {
  this->layer_id = layer_id;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->r = stride;
  this->oh = h / r;
  this->ow = w / r;
  this->n = r;
  this->c = c * oh;
  this->h = 1;
  this->w = ow * r;
  this->fmt = fmt;
  this->fmt_bytes = CV18xx::bytesize_of_fmt(fmt);
  this->n_loop = n;
  this->n_offset = c * h * w * fmt_bytes;
  // src stride
  src_gstride = CV18xx::tg_default_stride(c * oh, r, ow * r, fmt);
  std::swap(src_gstride.n, src_gstride.h);
  // dst stride
  dst_gstride = CV18xx::tg_default_stride(c * oh, 1, ow, fmt);
  CV18xx::set_layer_id(layer_id);
}

void TgReorgKernel::selectTilePolicy() {
  int max_c = std::min(c, MAX_CHANNEL);
  int max_ow = std::min(ow, MAX_CHANNEL);
  uint32_t lmem_required = 0;
  int step_c, step_ow;
  for (step_ow = max_ow; step_ow >= 1; --step_ow) {
    for (step_c = max_c; step_c >= 1;) {
      auto shape0 = CV18xx::tl_shape_t4(r, step_c, 1, step_ow * r);
      auto shape1 = CV18xx::tl_shape_t4(step_ow, step_c, r, r);
      auto shape2 = CV18xx::tl_shape_t4(r * r, step_c, 1, step_ow);
      lmem_required = 2 * CV18xx::lmem_tensor_to_size(shape0, fmt, 1) +
                      CV18xx::lmem_tensor_to_size(shape1, fmt, 1) +
                      2 * CV18xx::lmem_tensor_to_size(shape2, fmt, 1);
      if (lmem_required <= (uint32_t)CV18xx::LMEM_BYTES) {
        goto after_loop;
      }
      if (step_c % CV18xx::NPU_NUM) {
        step_c -= step_c % CV18xx::NPU_NUM;
      } else {
        step_c -= CV18xx::NPU_NUM;
      }
    }
  }
after_loop:
  if (lmem_required > (uint32_t)CV18xx::LMEM_BYTES) {
    llvm::errs() << llvm::format(
        "Reorg tilling failed, shape:(%d,%d,%d,%d), fmt:%d\n", n, c, h, w, fmt);
    assert(0);
  }

  CV18xx::tiling_info_t tile = {0};
  for (int loop = 0; loop < n_loop; loop++) {
    tile.offset = loop * n_offset;
    for (tile.pos_c = 0; tile.pos_c < c; tile.pos_c += step_c) {
      tile.c = std::min(c - tile.pos_c, step_c);
      for (tile.pos_w = 0; tile.pos_w < ow; tile.pos_w += step_ow) {
        tile.w = std::min(ow - tile.pos_w, step_ow);
        tiles.emplace_back(tile);
      }
    }
  }
}

void TgReorgKernel::allocLmem() {
  auto &tile = tiles[0];
  auto shape = CV18xx::tl_shape_t4(r, tile.c, 1, tile.w * r);
  tl_mem[0] = CV18xx::lmem_alloc_tensor(shape, fmt, 1);
  tl_mem[1] = CV18xx::lmem_alloc_tensor(shape, fmt, 1);
  auto shape1 = CV18xx::tl_shape_t4(r * r, tile.c, 1, tile.w);
  tl_mem[2] = CV18xx::lmem_alloc_tensor(shape1, fmt, 1);
  tl_mem[3] = CV18xx::lmem_alloc_tensor(shape1, fmt, 1);
  auto shape2 = CV18xx::tl_shape_t4(tile.w, tile.c, r, r);
  tl_mem[4] = CV18xx::lmem_alloc_tensor(shape2, fmt, 1);
}

void TgReorgKernel::deallocLmem() {
  for (int i = 4; i >= 0; i--) {
    CV18xx::lmem_free_tensor(tl_mem[i]);
  }
}

void TgReorgKernel::refresh(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  tl_ifmap = *tl_mem[step_idx % 2];
  tl_ofmap = *tl_mem[2 + step_idx % 2];
  tl_middle = *tl_mem[4];
  tl_ifmap.shape = CV18xx::tl_shape_t4(r, tile.c, 1, tile.w * r);
  tl_ifmap.stride = CV18xx::tl_default_stride(tl_ifmap.shape, fmt, 1);
  tl_ofmap.shape = CV18xx::tl_shape_t4(r * r, tile.c, 1, tile.w);
  tl_ofmap.stride = CV18xx::tl_default_stride(tl_ofmap.shape, fmt, 1);
  tl_middle.shape = CV18xx::tl_shape_t4(tile.w, tile.c, r, r);
  tl_middle.stride = CV18xx::tl_default_stride(tl_middle.shape, fmt, 1);
}

void TgReorgKernel::load(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  gaddr_t gaddr = ga_input + tile.offset + tile.pos_c * src_gstride.c +
                  tile.pos_w * r * fmt_bytes;
  CV18xx::tdma_load_stride(&tl_ifmap, gaddr, src_gstride);
}

void TgReorgKernel::store(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  gaddr_t gaddr = ga_output + tile.offset + tile.pos_c * dst_gstride.c +
                  tile.pos_w * dst_gstride.w;
  CV18xx::tdma_store_stride(&tl_ofmap, gaddr, dst_gstride);
}

void TgReorgKernel::compute(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  tl_ifmap.shape.h = tile.w;
  tl_ifmap.shape.w = r;
  tl_ifmap.stride = CV18xx::tl_default_stride(tl_ifmap.shape, fmt, 1);
  tl_middle.shape = tl_ifmap.shape;
  std::swap(tl_middle.stride.n, tl_middle.stride.h);
  cvk_tiu_copy_param_t p = {0};
  p.src = &tl_ifmap;
  p.dst = &tl_middle;
  p.layer_id = layer_id;
  CV18xx::tiu_copy(&p);
  tl_middle.shape = CV18xx::tl_shape_t4(tile.w, tile.c, 1, r * r);
  tl_middle.stride = CV18xx::tl_default_stride(tl_middle.shape, fmt, 1);
  tl_ofmap.shape = tl_middle.shape;
  std::swap(tl_ofmap.stride.n, tl_ofmap.stride.w);
  p.src = &tl_middle;
  p.dst = &tl_ofmap;
  CV18xx::tiu_copy(&p);
}

void TgReorgKernel::schedule() {
  allocLmem();
  int32_t total_steps = tiles.size();
  for (int32_t i = 0; i < total_steps + 2; i++) {
    CV18xx::parallel_enable();

    if (i - 1 >= 0 && i - 1 < total_steps) {
      compute(i - 1);
    }
    if (i < total_steps) {
      load(i);
    }
    if (i - 2 >= 0) {
      store(i - 2);
    }
    CV18xx::parallel_disable();
  }
  deallocLmem();
}

void cvi_backend_tg_reorg_kernel(uint32_t layer_id, gaddr_t input_gaddr,
                                 gaddr_t output_gaddr, int n, int c, int h,
                                 int w, int stride, cvk_fmt_t fmt) {
  TgReorgKernel kernel;
  kernel.init(layer_id, input_gaddr, output_gaddr, n, c, h, w, stride, fmt);
  kernel.selectTilePolicy();
  kernel.schedule();
}
} // namespace backend
} // namespace tpu_mlir
