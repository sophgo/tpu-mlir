//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/Kernel/TgPermuteKernel.hpp"

#define DEBUG_TYPE "cvi_backend_permute_kernel"

//
// permute xxx3: (I8 & BF16)
//   TDMA does not has the stride of width(ws).
//   Since the width of destination is unchanged, use tensor store to write one
//   hight to the correct position with ns, cs, hs.
//   It is tricky that destination shape in tensor store is the same as source
//   shape.
//
//
// I8/BF16 support: xxx3,0xxx,2301,3120,3012,3201
//
namespace tpu_mlir {
namespace backend {
void TgPermuteKernel::convert_order() {

  if (order[1] == 1) {
    // if c not swap, then use tiu_copy to do permute
    return;
  }

  if (is_order(0, 2, 3, 1)) {
    if (n < CV18xx::NPU_NUM) {
      n_loop = n;
      reshape(c, h * w);
    } else {
      update_NCHW(1, n, c, h * w);
      update_order(0, 1, 3, 2);
    }
    return;
  }
  if (is_order(0, 3, 2, 1)) {
    n_loop = n;
    n = c;
    reshape(h);
    return;
  }
  if (is_order(0, 3, 1, 2)) {
    if (n < CV18xx::NPU_NUM) {
      n_loop = n;
      reshape(c * h, w);
    } else {
      update_NCHW(1, n, c * h, w);
      update_order(0, 1, 3, 2);
    }
    return;
  }
  if (is_order(2, 3, 0, 1)) {
    reshape(n * c, h * w);
    return;
  }
  if (is_order(3, 0, 1, 2)) {
    reshape(n * c * h, w);
    return;
  }
  if (is_order(3, 2, 0, 1)) {
    n = n * c;
    reshape(h);
    return;
  }
  if (is_order(0, 2, 1, 3) && n < CV18xx::NPU_NUM) {
    n_loop = n;
    if (c >= h) {
      update_NCHW(1, c, h, w);
    } else {
      update_NCHW(c, h, 1, w);
    }
    update_order(2, 1, 0, 3);
    return;
  }
  if (is_order(1, 2, 3, 0)) {
    reshape(n, c * h * w);
  }
  if (is_order(2, 0, 1, 3)) {
    if (n * c > h) {
      update_NCHW(1, n * c, h, w);
    } else {
      update_NCHW(n * c, h, 1, w);
    }
    update_order(2, 1, 0, 3);
    return;
  }
  if (order[3] == 3 || w == 1) {
    // xxx3 use tdma directly
    by_tdma = true;
    return;
  }
  llvm::errs() << llvm::format("Not support permute case, fmt:%d, "
                               "order:(%d,%d,%d,%d), shape:(%d,%d,%d,%d)\n",
                               fmt, order[0], order[1], order[2], order[3], n,
                               c, h, w);
  assert(0);
}

// [n, channel, 1, w] => [n, c, h, w]
void TgPermuteKernel::reshape(int channel) {
  uint32_t lmem_need = 0;
  for (int c_ = channel; c_ >= 1; c_--) {
    if (channel % c_ != 0) {
      continue;
    }
    int h_ = channel / c_;
    auto size =
        CV18xx::lmem_tensor_to_size(CV18xx::tl_shape_t4(n, c_, h_, w), fmt, 1);
    if (lmem_need == 0 || size < lmem_need) {
      lmem_need = size;
      c = c_;
      h = h_;
    }
  }
  update_order(3, 1, 2, 0);
}

// for [dim0, dim1] => [dim1, dim0]
void TgPermuteKernel::reshape(int dim0, int dim1) {
  // case 1, dim0 in channel
  uint32_t lmem_need = 0;
  for (int c_ = dim0; c_ >= 1; c_--) {
    if (dim0 % c_ != 0) {
      continue;
    }
    int h_ = dim0 / c_;
    auto size0 = CV18xx::lmem_tensor_to_size(
        CV18xx::tl_shape_t4(1, c_, h_, dim1), fmt, 1);
    auto size1 = CV18xx::lmem_tensor_to_size(
        CV18xx::tl_shape_t4(dim1, c_, h_, 1), fmt, 1);
    if (lmem_need == 0 || size0 + size1 < lmem_need) {
      lmem_need = size0 + size1;
      n = 1;
      c = c_;
      h = h_;
      w = dim1;
    }
  }

  // case 2, dim1 in channel
  for (int c_ = dim1; c_ >= 1; c_--) {
    if (dim1 % c_ != 0) {
      continue;
    }
    int h_ = dim1 / c_;
    auto size0 = CV18xx::lmem_tensor_to_size(
        CV18xx::tl_shape_t4(dim0, c_, h_, 1), fmt, 1);
    auto size1 = CV18xx::lmem_tensor_to_size(
        CV18xx::tl_shape_t4(1, c_, h_, dim0), fmt, 1);
    if (size0 + size1 < lmem_need) {
      lmem_need = size0 + size1;
      n = dim0;
      c = c_;
      h = h_;
      w = 1;
    }
  }
  update_order(3, 1, 2, 0);
}

bool TgPermuteKernel::is_order(int order_n, int order_c, int order_h,
                               int order_w) const {
  return (order_n == order[0] && order_c == order[1] && order_h == order[2] &&
          order_w == order[3]);
}

void TgPermuteKernel::update_order(int order_n, int order_c, int order_h,
                                   int order_w) {
  order[0] = order_n;
  order[1] = order_c;
  order[2] = order_h;
  order[3] = order_w;
}

void TgPermuteKernel::update_NCHW(int n, int c, int h, int w) {
  this->n = n;
  this->c = c;
  this->h = h;
  this->w = w;
}

void TgPermuteKernel::init(uint32_t layer_id, gaddr_t ga_input,
                           gaddr_t ga_output, int n, int c, int h, int w,
                           int order_n, int order_c, int order_h, int order_w,
                           cvk_fmt_t fmt) {
  this->layer_id = layer_id;
  this->fmt = fmt;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->fmt_bytes = CV18xx::bytesize_of_fmt(fmt);
  this->by_tdma = false;
  this->n_loop = 1;
  this->n_offset = c * h * w * fmt_bytes;
  update_NCHW(n, c, h, w);
  update_order(order_n, order_c, order_h, order_w);
  convert_order();

  int i_s[4] = {this->n, this->c, this->h, this->w};
  int o_s[4] = {i_s[order[0]], i_s[order[1]], i_s[order[2]], i_s[order[3]]};
  src_stride = CV18xx::tg_default_stride(i_s[1], i_s[2], i_s[3], fmt);
  dst_stride = CV18xx::tg_default_stride(o_s[1], o_s[2], o_s[3], fmt);
  uint32_t o_stride[4];
  o_stride[order[0]] = dst_stride.n;
  o_stride[order[1]] = dst_stride.c;
  o_stride[order[2]] = dst_stride.h;
  o_stride[order[3]] = dst_stride.w;
  dst_stride_order = {o_stride[0], o_stride[1], o_stride[2], o_stride[3]};
  CV18xx::set_layer_id(layer_id);
}

uint32_t TgPermuteKernel::tile_offset(const CV18xx::tiling_info_t &tile,
                                      bool is_src) const {
  const cvk_tg_stride_t &s = is_src ? src_stride : dst_stride_order;
  return tile.offset + tile.pos_n * s.n + tile.pos_c * s.c + tile.pos_h * s.h +
         tile.pos_w * s.w;
}

void TgPermuteKernel::selectTilePolicy() {
  if (by_tdma) {
    return;
  }
  assert(order[1] == 1); // c can't be ordered
  int &step_n = step[0];
  int &step_c = step[1];
  int &step_h = step[2];
  int &step_w = step[3];
  uint32_t lmem_need;
  int max_w = std::min(w, MAX_WIDTH);
  int max_h = std::min(h, MAX_HEIGHT);
  int max_c = std::min(c, MAX_CHANNEL);
  int max_n = std::min(n, MAX_CHANNEL);
  for (step_w = max_w; step_w > 0; step_w--) {
    for (step_h = max_h; step_h > 0; step_h--) {
      for (step_n = max_n; step_n > 0; step_n--) {
        for (step_c = max_c; step_c > 0;) {
          auto ishape = CV18xx::tl_shape_t4(step_n, step_c, step_h, step_w);
          auto oshape = CV18xx::tl_shape_t4(step[order[0]], step[order[1]],
                                            step[order[2]], step[order[3]]);
          lmem_need = 2 * CV18xx::lmem_tensor_to_size(ishape, fmt, 1) +
                      2 * CV18xx::lmem_tensor_to_size(oshape, fmt, 1);
          if (lmem_need <= (uint32_t)CV18xx::LMEM_BYTES) {
            goto after_loop;
          }
          if (step_c % CV18xx::NPU_NUM == 0) {
            step_c -= CV18xx::NPU_NUM;
          } else {
            step_c -= step_c % CV18xx::NPU_NUM;
          }
        }
      }
    }
  }
after_loop:
  if (lmem_need > (uint32_t)CV18xx::LMEM_BYTES) {
    llvm::errs() << llvm::format(
        "Permute tiling error, fmt:%d, shape:(%d,%d,%d,%d), "
        "order:(%d,%d,%d,%d)\n",
        fmt, n, c, h, w, order[0], order[1], order[2], order[3]);
    assert(0);
  }
  CV18xx::tiling_info_t tile = {0};
  for (int loop = 0; loop < n_loop; loop++) {
    for (tile.pos_n = 0; tile.pos_n < n; tile.pos_n += step_n) {
      tile.n = std::min(step_n, n - tile.pos_n);
      for (tile.pos_c = 0; tile.pos_c < c; tile.pos_c += step_c) {
        tile.c = std::min(step_c, c - tile.pos_c);
        for (tile.pos_h = 0; tile.pos_h < h; tile.pos_h += step_h) {
          tile.h = std::min(step_h, h - tile.pos_h);
          for (tile.pos_w = 0; tile.pos_w < w; tile.pos_w += step_w) {
            tile.w = std::min(step_w, w - tile.pos_w);
            tile.offset = loop * n_offset;
            tiles.emplace_back(tile);
          }
        }
      }
    }
  }
}

void TgPermuteKernel::load(int step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  CV18xx::tdma_load_stride(&tl_ifmap, ga_input + tile_offset(tile), src_stride);
}

void TgPermuteKernel::store(int step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  CV18xx::tdma_store_stride(&tl_ofmap, ga_output + tile_offset(tile, false),
                            dst_stride);
}

void TgPermuteKernel::allocLmem() {
  auto src_shape = CV18xx::tl_shape_t4(step[0], step[1], step[2], step[3]);
  auto dst_shape = CV18xx::tl_shape_t4(step[order[0]], step[order[1]],
                                       step[order[2]], step[order[3]]);
  tl_mem[0] = CV18xx::lmem_alloc_tensor(src_shape, fmt, 1);
  tl_mem[1] = CV18xx::lmem_alloc_tensor(src_shape, fmt, 1);
  tl_mem[2] = CV18xx::lmem_alloc_tensor(dst_shape, fmt, 1);
  tl_mem[3] = CV18xx::lmem_alloc_tensor(dst_shape, fmt, 1);
}

void TgPermuteKernel::deallocLmem() {
  for (int i = 3; i >= 0; i--) {
    CV18xx::lmem_free_tensor(tl_mem[i]);
  }
}

void TgPermuteKernel::refresh(int step_idx) {
  auto &tile = tiles[step_idx];
  tl_ifmap = *tl_mem[step_idx % 2];
  tl_ofmap = *tl_mem[2 + step_idx % 2];
  int s[4] = {tile.n, tile.c, tile.h, tile.w};
  tl_ifmap.shape = CV18xx::tl_shape_t4(s[0], s[1], s[2], s[3]);
  tl_ifmap.stride = CV18xx::tl_default_stride(tl_ifmap.shape, fmt, 1);
  tl_ofmap.shape =
      CV18xx::tl_shape_t4(s[order[0]], s[order[1]], s[order[2]], s[order[3]]);
  tl_ofmap.stride = CV18xx::tl_default_stride(tl_ofmap.shape, fmt, 1);
}

void TgPermuteKernel::compute(int step_idx) {
  refresh(step_idx);
  assert(order[1] == 1); // c can't be ordered
  tl_ofmap.shape = tl_ifmap.shape;
  auto &stride = tl_ofmap.stride;
  uint32_t s[4];
  s[order[0]] = stride.n;
  s[order[1]] = stride.c;
  s[order[2]] = stride.h;
  s[order[3]] = stride.w;
  tl_ofmap.stride = {s[0], s[1], s[2], s[3]};
  cvk_tiu_copy_param_t p = {0};
  p.src = &tl_ifmap;
  p.dst = &tl_ofmap;
  p.layer_id = layer_id;
  CV18xx::tiu_copy(&p);
}

void TgPermuteKernel::permute_tdma() {
  auto shape = CV18xx::tg_shape_t4(n, c, h, w);
  CV18xx::tdma_g2g_tensor_copy(ga_input, shape, src_stride, fmt, ga_output,
                               shape, dst_stride_order, fmt);
}

void TgPermuteKernel::schedule() {
  if (by_tdma) {
    permute_tdma();
    return;
  }
  allocLmem();
  int total_steps = tiles.size();
  for (int i = 0; i < total_steps + 2; i++) {
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

void cvi_backend_tg_permute_kernel(uint32_t layer_id, gaddr_t ga_input,
                                   gaddr_t ga_output, int n, int c, int h,
                                   int w, int order_n, int order_c, int order_h,
                                   int order_w, cvk_fmt_t fmt) {
  TgPermuteKernel kernel;
  kernel.init(layer_id, ga_input, ga_output, n, c, h, w, order_n, order_c,
              order_h, order_w, fmt);
  kernel.selectTilePolicy();
  kernel.schedule();
}
} // namespace backend
} // namespace tpu_mlir
