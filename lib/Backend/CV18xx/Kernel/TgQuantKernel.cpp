//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/Kernel/TgQuantKernel.hpp"
#include "tpu_mlir/Support/MathUtils.h"

#define DEBUG_TYPE "quant_kernel"

// convert fmt:
//    support I8/BF16/F32 => I8/BF16/F32, && from != to

namespace tpu_mlir {
namespace backend {
void TgQuantKernel::init(uint32_t layer_id, cvk_fmt_t from, cvk_fmt_t to,
                         gaddr_t ga_input, gaddr_t ga_output, int32_t n,
                         int32_t c, int32_t h, int32_t w, float const_scale,
                         int offset) {
  from_byte = CV18xx::bytesize_of_fmt(from);
  to_byte = CV18xx::bytesize_of_fmt(to);
  assert(from_byte != to_byte);

  this->layer_id = layer_id;
  this->n = n;
  this->c = c;
  this->h = h;
  this->w = w;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->from = from;
  this->to = to;
  this->const_scale = const_scale;
  this->offset = offset;
  load_unit = (from_byte + 1) / 2;
  store_unit = (to_byte + 1) / 2;
  CV18xx::set_layer_id(layer_id);
}

void TgQuantKernel::doTileForNormalCase() {
  int blob_num = 2 * (load_unit + store_unit); // 2 to do flip
  CV18xx::tiling_packing(tiles, n, c, h, w, CVK_FMT_BF16, blob_num, 0,
                         CV18xx::TilingAll);
}

void TgQuantKernel::selectTilePolicy() { doTileForNormalCase(); }

cvk_tl_t *TgQuantKernel::alloc_lmem(const cvk_tl_shape_t &shape,
                                    bool clean) const {
  cvk_tl_t *tl_mem = CV18xx::lmem_alloc_tensor(shape, CVK_FMT_BF16, 1);
  if (clean) {
    CV18xx::tiu_zeros(layer_id, tl_mem);
  }
  return tl_mem;
}

void TgQuantKernel::allocLmem() {
  cvk_tl_shape_t input_shape = CV18xx::tl_shape_t4(
      tiles[0].n, tiles[0].c, tiles[0].h, tiles[0].w * load_unit);
  tl_input[0] = alloc_lmem(input_shape, false);
  tl_input[1] = alloc_lmem(input_shape, false);
  cvk_tl_shape_t output_shape = CV18xx::tl_shape_t4(
      tiles[0].n, tiles[0].c, tiles[0].h, tiles[0].w * store_unit);
  tl_output[0] = alloc_lmem(output_shape, to_byte == 4);
  tl_output[1] = alloc_lmem(output_shape, to_byte == 4);
}

void TgQuantKernel::deallocLmem() {
  CV18xx::lmem_free_tensor(tl_output[1]);
  CV18xx::lmem_free_tensor(tl_output[0]);
  CV18xx::lmem_free_tensor(tl_input[1]);
  CV18xx::lmem_free_tensor(tl_input[0]);
}

cvk_tl_stride_t TgQuantKernel::tl_fp32_stride(const cvk_tl_shape_t &shape,
                                              int eu_align) const {
  int fmt_sz = 4; // 4 means fp32 takes 4 bytes
  cvk_tl_stride_t s;
  s.w = fmt_sz;
  s.h = shape.w * fmt_sz;
  s.c = shape.h * shape.w * fmt_sz;
  if (eu_align) {
    s.c = align_up(s.c, CV18xx::EU_BYTES);
  }
  s.n = s.c * ceiling_func(shape.c, CV18xx::NPU_NUM);
  return s;
}

void TgQuantKernel::compute(int32_t step_idx, int32_t flip) {
  auto &tile = tiles[step_idx];
  cvk_tl_t tl_ifmap = *tl_input[flip];
  tl_ifmap.shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  if (from == CVK_FMT_F32) {
    tl_ifmap.start_address += 2; // higher 16bit
    tl_ifmap.stride = tl_fp32_stride(tl_ifmap.shape, 1);
  } else {
    tl_ifmap.stride =
        CV18xx::tl_default_stride(tl_ifmap.shape, CVK_FMT_BF16, 1);
  }
  cvk_tl_t tl_ofmap = *tl_output[flip];
  tl_ofmap.shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  if (to == CVK_FMT_F32) {
    tl_ofmap.start_address += 2; // higher 16bit
    tl_ofmap.stride = tl_fp32_stride(tl_ofmap.shape, 1);
  } else {
    tl_ofmap.stride =
        CV18xx::tl_default_stride(tl_ofmap.shape, CVK_FMT_BF16, 1);
  }
  if (const_scale != 1.0) {
    if (offset != 0) {
      cvk_tiu_add_param_t p_offset = {0};
      p_offset.res_high = NULL;
      p_offset.res_low = &tl_ifmap;
      p_offset.a_high = NULL;
      p_offset.a_low = &tl_ifmap;
      p_offset.b_is_const = 1;
      p_offset.b_const.val = CV18xx::convert_fp32_to_bf16(offset);
      p_offset.relu_enable = 0;
      p_offset.layer_id = layer_id;
      CV18xx::tiu_add(&p_offset);

      cvk_tiu_mul_param_t p = {0};
      p.res_high = NULL;
      p.res_low = &tl_ofmap;
      p.a = &tl_ifmap;
      p.b_is_const = 1;
      p.b_const.val = CV18xx::convert_fp32_to_bf16(const_scale);
      p.relu_enable = 0;
      p.layer_id = layer_id;
      CV18xx::tiu_mul(&p);
    } else {
      cvk_tiu_mul_param_t p = {0};
      p.res_high = NULL;
      p.res_low = &tl_ofmap;
      p.a = &tl_ifmap;
      p.b_is_const = 1;
      p.b_const.val = CV18xx::convert_fp32_to_bf16(const_scale);
      p.relu_enable = 0;
      p.layer_id = layer_id;
      CV18xx::tiu_mul(&p);
    }
  } else {
    cvk_tiu_copy_param_t param = {0};
    param.src = &tl_ifmap;
    param.dst = &tl_ofmap;
    param.layer_id = layer_id;
    CV18xx::tiu_copy(&param);
  }
}

// i8/bf16/f32 => bf16
void TgQuantKernel::load(int32_t step_idx, int32_t flip) {
  auto &tile = tiles[step_idx];
  cvk_tg_t src;
  src.start_address = ga_input + tile.offset * from_byte / 2;
  src.base_reg_index =
      CV18xx::getTdmaBaseSelectIndexFromGaddr(src.start_address);
  src.int8_rnd_mode = 0;
  src.fmt = (load_unit == 2 ? CVK_FMT_BF16 : from);
  src.shape = CV18xx::tg_shape_t4(tile.n, tile.c, tile.h, tile.w * load_unit);
  src.stride = CV18xx::tg_default_stride(src.shape, src.fmt);
  cvk_tl_t tl_ifmap = *tl_input[flip];
  tl_ifmap.shape =
      CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w * load_unit);
  tl_ifmap.stride = CV18xx::tl_default_stride(tl_ifmap.shape, CVK_FMT_BF16, 1);
  cvk_tdma_g2l_tensor_copy_param_t p1 = {0};
  p1.src = &src;
  p1.dst = &tl_ifmap;
  CV18xx::tdma_g2l_tensor_copy(&p1);
}

// bf16 => i8/bf16/f32
void TgQuantKernel::store(int32_t step_idx, int32_t flip) {
  auto &tile = tiles[step_idx];
  cvk_tl_t tl_ofmap = *tl_output[flip];
  tl_ofmap.shape =
      CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w * store_unit);
  tl_ofmap.stride = CV18xx::tl_default_stride(tl_ofmap.shape, tl_ofmap.fmt, 1);

  cvk_tg_t dst = {0};
  dst.start_address = ga_output + tile.offset * to_byte / 2;
  dst.base_reg_index =
      CV18xx::getTdmaBaseSelectIndexFromGaddr(dst.start_address);
  dst.fmt = (store_unit == 2 ? CVK_FMT_BF16 : to);
  dst.shape = CV18xx::tg_shape_t4(tile.n, tile.c, tile.h, tile.w * store_unit);
  dst.stride = CV18xx::tg_default_stride(dst.shape, dst.fmt);

  cvk_tdma_l2g_tensor_copy_param_t p1 = {0};
  p1.src = &tl_ofmap;
  p1.dst = &dst;
  CV18xx::tdma_l2g_tensor_copy(&p1);
}

void TgQuantKernel::schedule() {
  allocLmem();
  int32_t total_steps = tiles.size();
  for (int32_t i = 0; i < total_steps + 2; i++) {
    CV18xx::parallel_enable();

    if (i - 1 >= 0 && i - 1 < total_steps) {
      compute(i - 1, 1 - flip);
    }
    if (i < total_steps) {
      load(i, flip);
    }
    if (i - 2 >= 0) {
      store(i - 2, flip);
    }
    flip = 1 - flip;
    CV18xx::parallel_disable();
  }
  deallocLmem();
}

void cvi_backend_tg_quant_kernel(uint32_t layer_id, cvk_fmt_t from,
                                 cvk_fmt_t to, gaddr_t bottom_gaddr,
                                 gaddr_t top_gaddr, int input_n, int input_c,
                                 int input_h, int input_w, float const_scale,
                                 int offset) {
  TgQuantKernel kernel;
  kernel.init(layer_id, from, to, bottom_gaddr, top_gaddr, input_n, input_c,
              input_h, input_w, const_scale, offset);

  kernel.selectTilePolicy();
  kernel.schedule();
}
} // namespace backend
} // namespace tpu_mlir
