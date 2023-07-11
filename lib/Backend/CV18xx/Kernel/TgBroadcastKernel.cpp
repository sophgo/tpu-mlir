//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/Kernel/TgBroadcastKernel.hpp"

#define DEBUG_TYPE "cvi_backend_broadcast_kernel"

namespace tpu_mlir {
namespace backend {
void TgBcastKernel::convert_shape(int an, int ac, int ah, int aw, int bn,
                                  int bc, int bh, int bw) {
  int num_dims = 4;
  int a_s[4] = {an, ac, ah, aw};
  int b_s[4] = {bn, bc, bh, bw};
  index_bcast = -1;
  std::vector<int> a_v;
  std::vector<int> b_v;
  for (int i = 0; i < num_dims;) {
    if (a_s[i] == 1 && b_s[i] == 1) {
      i++;
      continue;
    }
    if (a_s[i] == b_s[i]) {
      int ins = 1;
      do {
        ins *= a_s[i];
        i++;
      } while (i < num_dims && a_s[i] == b_s[i]);
      a_v.push_back(ins);
      b_v.push_back(ins);
    }
    if (i == num_dims) {
      break;
    }
    assert(index_bcast == -1 && "only broadcast continuous axis");
    index_bcast = a_v.size();
    int a_ins = 1, b_ins = 1;
    do {
      assert(b_s[i] == 1 && "only broadcast right operand supported!");
      a_ins *= a_s[i];
      b_ins *= b_s[i];
      i++;
    } while (i < num_dims &&
             (a_s[i] != b_s[i] || (a_s[i] == 1 && b_s[i] == 1)));
    a_v.push_back(a_ins);
    b_v.push_back(b_ins);
  }
  num_dims = a_v.size();
  assert(num_dims <= 3 && num_dims >= 1);
  for (int i = 0; i < 4; i++) {
    shape_a[i] = 1;
    shape_b[i] = 1;
  }
  int h, w;
  if (num_dims == 1) {
    shape_a[0] = a_v[0];
    shape_b[0] = b_v[0];
    mode = BCAST_ALL;
  } else if (num_dims == 2) {
    shape_a[1] = a_v[0];
    shape_b[1] = b_v[0];
    if (index_bcast == 0) {
      CV18xx::size_to_hw(a_v[1], h, w);
      shape_a[2] = h;
      shape_b[2] = h;
      shape_a[3] = w;
      shape_b[3] = w;
      mode = BCAST_C;
    } else {
      CV18xx::size_to_hw(a_v[1], h, w);
      shape_a[2] = h;
      shape_a[3] = w;
      mode = BCAST_HW;
    }
  } else {
    assert(index_bcast == 1);
    shape_a[0] = a_v[0];
    shape_b[0] = b_v[0];
    shape_a[1] = a_v[1];
    shape_b[1] = b_v[1];
    CV18xx::size_to_hw(a_v[2], h, w);
    shape_a[2] = h;
    shape_b[2] = h;
    shape_a[3] = w;
    shape_b[3] = w;
    mode = BCAST_C;
  }
}

void TgBcastKernel::init(uint32_t layer_id, gaddr_t ga_a, gaddr_t ga_b,
                         gaddr_t ga_output, int an, int ac, int ah, int aw,
                         int bn, int bc, int bh, int bw, bool do_relu,
                         int32_t rshift, const int32_t *multipliers,
                         bcast_t type, cvk_fmt_t fmt) {
  this->layer_id = layer_id;
  this->ga_a = ga_a;
  this->ga_b = ga_b;
  this->ga_output = ga_output;
  this->rshift = rshift;
  this->multipliers = multipliers;
  this->do_relu = do_relu;
  this->type = type;
  this->fmt = fmt;
  this->fmt_bytes = CV18xx::bytesize_of_fmt(fmt);
  convert_shape(an, ac, ah, aw, bn, bc, bh, bw);
  num_blobs = (fmt == CVK_FMT_I8 && type != BCAST_MUL) ? 2 : 1;
}

void TgBcastKernel::tile_all() {
  int *o_s = shape_a;
  uint32_t lmem_used = CV18xx::lmem_tensor_to_size(
      CV18xx::tl_shape_t4(1, CV18xx::NPU_NUM, 1, 1), fmt, 1);
  CV18xx::tiling_packing(tiles, o_s[0], o_s[1], o_s[2], o_s[3], fmt, num_blobs,
                         lmem_used, CV18xx::TilingAll);
}

void TgBcastKernel::tile_other() {
  int *o_s = shape_a;
  int step_w, step_h, step_c, step_n;
  int max_c = std::min(o_s[1], MAX_CHANNEL);
  uint32_t lmem_required = 0;
  for (step_w = o_s[3]; step_w >= 1; --step_w) {
    for (step_h = o_s[2]; step_h >= 1; --step_h) {
      for (step_n = o_s[0]; step_n >= 1; --step_n) {
        for (step_c = max_c; step_c >= 1;) {
          auto shape1 = CV18xx::tl_shape_t4(step_n, step_c, step_h, step_w);
          auto shape2 =
              (mode == BCAST_HW ? CV18xx::tl_shape_t4(step_n, step_c, 1, 1)
                                : CV18xx::tl_shape_t4(step_n, CV18xx::NPU_NUM,
                                                      step_h, step_w));
          lmem_required =
              num_blobs * CV18xx::lmem_tensor_to_size(shape1, fmt, 1) +
              CV18xx::lmem_tensor_to_size(shape2, fmt, 1);
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
    }
  }
after_loop:
  if (lmem_required > (uint32_t)CV18xx::LMEM_BYTES) {
    llvm_unreachable("bcast tiling failed\n");
  }
  CV18xx::tiling_info_t tile;
  cvk_tg_stride_t src_stride =
      CV18xx::tg_default_stride(o_s[1], o_s[2], o_s[3], fmt);
  for (tile.pos_n = 0; tile.pos_n < o_s[0]; tile.pos_n += step_n) {
    tile.n = std::min(o_s[0] - tile.pos_n, step_n);
    for (tile.pos_c = 0; tile.pos_c < o_s[1]; tile.pos_c += step_c) {
      tile.c = std::min(o_s[1] - tile.pos_c, step_c);
      for (tile.pos_h = 0; tile.pos_h < o_s[2]; tile.pos_h += step_h) {
        tile.h = std::min(o_s[2] - tile.pos_h, step_h);
        for (tile.pos_w = 0; tile.pos_w < o_s[3]; tile.pos_w += step_w) {
          tile.w = std::min(o_s[3] - tile.pos_w, step_w);
          tile.offset = tile.pos_w * src_stride.w + tile.pos_h * src_stride.h +
                        tile.pos_c * src_stride.c + tile.pos_n * src_stride.n;
          tiles.emplace_back(tile);
        }
      }
    }
  }
}

void TgBcastKernel::selectTilePolicy() {
  switch (mode) {
  case BCAST_ALL:
    tile_all();
    break;
  case BCAST_HW:
  case BCAST_C:
    tile_other();
    break;
  }
}

void TgBcastKernel::schedule() {
  switch (mode) {
  case BCAST_ALL:
    schedule_bcast_all();
    break;
  case BCAST_HW:
    schedule_bcast_hw();
    break;
  case BCAST_C:
    schedule_bcast_c();
    break;
  default:
    assert(0);
  }
}

void TgBcastKernel::schedule_bcast_all() {
  auto b_s = CV18xx::tl_shape_t4(1, CV18xx::NPU_NUM, 1, 1);
  auto tl_b = CV18xx::lmem_alloc_tensor(b_s, fmt, 1);
  cvk_tg_stride_t b_gstride = {.n = 1, .c = 0, .h = 1, .w = 1};
  CV18xx::tdma_load_stride(tl_b, ga_b, b_gstride);
  for (auto &tile : tiles) {
    auto a_s = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    auto tl_a = CV18xx::lmem_alloc_tensor(a_s, fmt, 1);
    cvk_tl_t *tl_buff = nullptr;
    if (num_blobs == 2) {
      tl_buff = CV18xx::lmem_alloc_tensor(a_s, fmt, 1);
    }
    CV18xx::tdma_load(tl_a, ga_a + tile.offset);
    cvk_tl_t tl_b_ = *tl_b;
    tl_b_.shape = a_s;
    tl_b_.stride = {.n = 0, .c = 0, .h = 0, .w = 0};
    tiu_compute(tl_a, tl_a, &tl_b_, tl_buff);
    CV18xx::tdma_store(tl_a, ga_output + tile.offset);
    if (tl_buff != nullptr) {
      CV18xx::lmem_free_tensor(tl_buff);
    }
    CV18xx::lmem_free_tensor(tl_a);
  }
  CV18xx::lmem_free_tensor(tl_b);
}

void TgBcastKernel::schedule_bcast_hw() {
  auto &tile = tiles[0];
  auto b_s = CV18xx::tl_shape_t4(tile.n, tile.c, 1, 1);
  auto tl_b = CV18xx::lmem_alloc_tensor(b_s, fmt, 1);
  cvk_tg_stride_t b_gstride =
      CV18xx::tg_default_stride(shape_b[1], shape_b[2], shape_b[3], fmt);
  cvk_tg_stride_t a_gstride =
      CV18xx::tg_default_stride(shape_a[1], shape_a[2], shape_a[3], fmt);
  int last_n = -1;
  int last_c = -1;
  for (auto &tile : tiles) {
    auto a_s = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    auto tl_a = CV18xx::lmem_alloc_tensor(a_s, fmt, 1);
    cvk_tl_t *tl_buff = nullptr;
    if (num_blobs == 2) {
      tl_buff = CV18xx::lmem_alloc_tensor(a_s, fmt, 1);
    }
    CV18xx::tdma_load_stride(tl_a, ga_a + tile.offset, a_gstride);
    cvk_tl_t tl_b_ = *tl_b;
    tl_b_.shape = CV18xx::tl_shape_t4(tile.n, tile.c, 1, 1);
    tl_b_.stride = CV18xx::tl_default_stride(tl_b_.shape, fmt, 1);
    if (last_n != tile.pos_n || last_c != tile.pos_c) {
      int offset = tile.pos_c * b_gstride.c + tile.pos_n * b_gstride.n;
      CV18xx::tdma_load_stride(&tl_b_, ga_b + offset, b_gstride);
    }
    tl_b_.shape = a_s;
    tl_b_.stride.w = 0;
    tl_b_.stride.h = 0;
    tiu_compute(tl_a, tl_a, &tl_b_, tl_buff);
    CV18xx::tdma_store_stride(tl_a, ga_output + tile.offset, a_gstride);
    if (tl_buff != nullptr) {
      CV18xx::lmem_free_tensor(tl_buff);
    }
    CV18xx::lmem_free_tensor(tl_a);
  }
  CV18xx::lmem_free_tensor(tl_b);
}

void TgBcastKernel::schedule_bcast_c() {
  auto &tile = tiles[0];
  auto b_s = CV18xx::tl_shape_t4(tile.n, CV18xx::NPU_NUM, tile.h, tile.w);
  auto tl_b = CV18xx::lmem_alloc_tensor(b_s, fmt, 1);
  cvk_tg_stride_t b_gstride =
      CV18xx::tg_default_stride(shape_b[1], shape_b[2], shape_b[3], fmt);
  b_gstride.c = 0;
  cvk_tg_stride_t a_gstride =
      CV18xx::tg_default_stride(shape_a[1], shape_a[2], shape_a[3], fmt);
  int last_n = -1;
  int last_h = -1;
  int last_w = -1;
  for (auto &tile : tiles) {
    auto a_s = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    auto tl_a = CV18xx::lmem_alloc_tensor(a_s, fmt, 1);
    cvk_tl_t *tl_buff = nullptr;
    if (num_blobs == 2) {
      tl_buff = CV18xx::lmem_alloc_tensor(a_s, fmt, 1);
    }
    CV18xx::tdma_load_stride(tl_a, ga_a + tile.offset, a_gstride);
    cvk_tl_t tl_b_ = *tl_b;
    tl_b_.shape = CV18xx::tl_shape_t4(tile.n, CV18xx::NPU_NUM, tile.h, tile.w);
    tl_b_.stride = CV18xx::tl_default_stride(tl_b_.shape, fmt, 1);
    if (last_n != tile.pos_n || last_h != tile.pos_h || last_w != tile.pos_w) {
      int offset = tile.pos_w * b_gstride.w + tile.pos_h * b_gstride.h +
                   tile.pos_n * b_gstride.n;
      CV18xx::tdma_load_stride(&tl_b_, ga_b + offset, b_gstride);
    }
    tl_b_.shape = a_s;
    tl_b_.stride.c = 0;
    tiu_compute(tl_a, tl_a, &tl_b_, tl_buff);
    CV18xx::tdma_store_stride(tl_a, ga_output + tile.offset, a_gstride);
    if (tl_buff != nullptr) {
      CV18xx::lmem_free_tensor(tl_buff);
    }
    CV18xx::lmem_free_tensor(tl_a);
  }
  CV18xx::lmem_free_tensor(tl_b);
}

void TgBcastKernel::tiu_compute(cvk_tl_t *tl_result, cvk_tl_t *tl_left,
                                cvk_tl_t *tl_right, cvk_tl_t *tl_buff) {
  if (fmt == CVK_FMT_BF16) {
    switch (type) {
    case BCAST_ADD: {
      cvk_tiu_add_param_t p = {0};
      p.res_high = 0;
      p.res_low = tl_result;
      p.a_high = 0;
      p.a_low = tl_left;
      p.b_is_const = 0;
      p.b.high = 0;
      p.b.low = tl_right;
      p.rshift_bits = 0;
      p.layer_id = layer_id;
      p.relu_enable = do_relu;
      CV18xx::tiu_add(&p);
    } break;
    case BCAST_MUL: {
      cvk_tiu_mul_param_t p = {0};
      p.res_high = nullptr;
      p.res_low = tl_result;
      p.a = tl_left;
      p.b = tl_right;
      p.b_is_const = 0;
      p.rshift_bits = 0;
      p.layer_id = layer_id;
      p.relu_enable = do_relu;
      CV18xx::tiu_mul(&p);
    } break;
    case BCAST_SUB: {
      cvk_tiu_sub_param_t p = {0};
      p.res_high = 0;
      p.res_low = tl_result;
      p.a_high = 0;
      p.a_low = tl_left;
      p.b_high = 0;
      p.b_low = tl_right;
      p.rshift_bits = 0;
      p.layer_id = layer_id;
      CV18xx::tiu_sub(&p);
    } break;
    default:
      assert(0);
      break;
    }
  } else {
    switch (type) {
    case BCAST_ADD:
    case BCAST_SUB: {
      cvk_tiu_mul_param_t p1 = {0};
      p1.res_high = tl_buff;
      p1.res_low = tl_result;
      p1.a = tl_left;
      p1.b_const.val = multipliers[0];
      p1.b_const.is_signed = true;
      p1.b_is_const = true;
      p1.rshift_bits = 0;
      p1.layer_id = layer_id;
      p1.relu_enable = 0;
      CV18xx::tiu_mul(&p1);
      cvk_tiu_mac_param_t p2 = {0};
      p2.res_high = tl_buff;
      p2.res_low = tl_result;
      p2.a = tl_right;
      p2.res_is_int8 = true;
      p2.b_const.val = (type == BCAST_ADD ? 1 : -1) * multipliers[1];
      p2.b_is_const = true;
      p2.b_const.is_signed = true;
      p2.lshift_bits = 0;
      p2.rshift_bits = rshift;
      p2.layer_id = layer_id;
      p2.relu_enable = do_relu;
      CV18xx::tiu_mac(&p2);
    } break;
    case BCAST_MUL: {
      cvk_tiu_mul_qm_param_t p1 = {0};
      p1.res_high = nullptr;
      p1.res_low = tl_result;
      p1.a = tl_left;
      p1.b_is_const = 0;
      p1.b = tl_right;
      p1.rshift_bits = rshift;
      p1.relu_enable = do_relu;
      p1.layer_id = layer_id;
      p1.multiplier = multipliers[0];
      CV18xx::tiu_mul_qm(&p1);
    } break;
    default:
      assert(0);
    }
  }
}

void tg_bcast_kernel(uint32_t layer_id, gaddr_t ga_a, gaddr_t ga_b,
                     gaddr_t ga_output, int an, int ac, int ah, int aw, int bn,
                     int bc, int bh, int bw, bool do_relu, int32_t rshift,
                     const int32_t *multipliers, bcast_t type, cvk_fmt_t fmt) {
  TgBcastKernel kernel;
  kernel.init(layer_id, ga_a, ga_b, ga_output, an, ac, ah, aw, bn, bc, bh, bw,
              do_relu, rshift, multipliers, type, fmt);
  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_int8_bcast_add_kernel(uint32_t layer_id, gaddr_t ga_a,
                                          gaddr_t ga_b, gaddr_t ga_output,
                                          int32_t an, int32_t ac, int32_t ah,
                                          int32_t aw, int32_t bn, int32_t bc,
                                          int32_t bh, int32_t bw, bool do_relu,
                                          const int32_t rshift,
                                          const int32_t *multipliers) {
  tg_bcast_kernel(layer_id, ga_a, ga_b, ga_output, an, ac, ah, aw, bn, bc, bh,
                  bw, do_relu, rshift, multipliers, BCAST_ADD, CVK_FMT_I8);
}

void cvi_backend_tg_int8_bcast_sub_kernel(uint32_t layer_id, gaddr_t ga_a,
                                          gaddr_t ga_b, gaddr_t ga_output,
                                          int32_t an, int32_t ac, int32_t ah,
                                          int32_t aw, int32_t bn, int32_t bc,
                                          int32_t bh, int32_t bw, bool do_relu,
                                          const int32_t rshift,
                                          const int32_t *multipliers) {
  tg_bcast_kernel(layer_id, ga_a, ga_b, ga_output, an, ac, ah, aw, bn, bc, bh,
                  bw, do_relu, rshift, multipliers, BCAST_SUB, CVK_FMT_I8);
}

void cvi_backend_tg_int8_bcast_mul_kernel(uint32_t layer_id, gaddr_t ga_a,
                                          gaddr_t ga_b, gaddr_t ga_output,
                                          int32_t an, int32_t ac, int32_t ah,
                                          int32_t aw, int32_t bn, int32_t bc,
                                          int32_t bh, int32_t bw, bool do_relu,
                                          const int32_t rshift,
                                          const int32_t *multipliers) {
  tg_bcast_kernel(layer_id, ga_a, ga_b, ga_output, an, ac, ah, aw, bn, bc, bh,
                  bw, do_relu, rshift, multipliers, BCAST_MUL, CVK_FMT_I8);
}

void cvi_backend_tg_bf16_bcast_add_kernel(uint32_t layer_id, gaddr_t ga_a,
                                          gaddr_t ga_b, gaddr_t ga_output,
                                          int an, int ac, int ah, int aw,
                                          int bn, int bc, int bh, int bw,
                                          bool do_relu) {
  tg_bcast_kernel(layer_id, ga_a, ga_b, ga_output, an, ac, ah, aw, bn, bc, bh,
                  bw, do_relu, 0, nullptr, BCAST_ADD, CVK_FMT_BF16);
}

void cvi_backend_tg_bf16_bcast_sub_kernel(uint32_t layer_id, gaddr_t ga_a,
                                          gaddr_t ga_b, gaddr_t ga_output,
                                          int an, int ac, int ah, int aw,
                                          int bn, int bc, int bh, int bw,
                                          bool do_relu) {
  tg_bcast_kernel(layer_id, ga_a, ga_b, ga_output, an, ac, ah, aw, bn, bc, bh,
                  bw, do_relu, 0, nullptr, BCAST_SUB, CVK_FMT_BF16);
}

void cvi_backend_tg_bf16_bcast_mul_kernel(uint32_t layer_id, gaddr_t ga_a,
                                          gaddr_t ga_b, gaddr_t ga_output,
                                          int n, int c, int h, int w, int bn,
                                          int bc, int bh, int bw,
                                          bool do_relu) {
  tg_bcast_kernel(layer_id, ga_a, ga_b, ga_output, n, c, h, w, bn, bc, bh, bw,
                  do_relu, 0, nullptr, BCAST_MUL, CVK_FMT_BF16);
}
} // namespace backend
} // namespace tpu_mlir
