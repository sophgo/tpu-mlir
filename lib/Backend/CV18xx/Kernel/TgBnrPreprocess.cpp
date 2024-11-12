//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Backend/CV18xx/Kernel/TgBnrPreprocess.hpp"
#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include "tpu_mlir/Support/LutFunc.h"

#define DEBUG_TYPE "cvi_backend_bnr_preprocess_kernel"

using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace backend {
void TgBnrPreprocessKernel::init(uint32_t layer_id, gaddr_t ga_input,
                                 gaddr_t ga_output, gaddr_t ga_table_high,
                                 gaddr_t ga_table_low, int n, int c, int h,
                                 int w, int start_h, int start_w,
                                 int channel_order[4], cvk_fmt_t fmt) {
  this->layer_id = layer_id;
  this->n = n;
  this->c = c;
  this->h = h;
  this->w = w;
  this->ga_input = ga_input;
  this->ga_table_high = ga_table_high;
  this->ga_table_low = ga_table_low;
  this->start_h = start_h;
  this->start_w = start_w;
  this->channel_order[0] = channel_order[0];
  this->channel_order[1] = channel_order[1];
  this->channel_order[2] = channel_order[2];
  this->channel_order[3] = channel_order[3];

  this->fmt = fmt;
  // assert(fmt == CVK_FMT_BF16);

  this->ih = h - start_h * 2;
  this->iw = w - start_w * 2;
  this->oh = ih / 2;
  this->ow = iw / 3;

  int od_size = 2;
  int channel_offset = oh * ow * od_size;
  out_gstride = CV18xx::tg_default_stride(oh, 1, ow, CVK_FMT_BF16);
  out_gstride.n *= 4;
  gaddr_t out_channel_addr[4];
  for (int i = 0; i < 4; ++i) {
    out_channel_addr[i] = ga_output + channel_offset * i;
  }
  for (int i = 0; i < 4; ++i) {
    ga_out[channel_order[i]] = out_channel_addr[i];
  }

  in_gstride = CV18xx::tg_default_stride(h, 1, w, CVK_FMT_I8);
  in_gstride.c *= 2;

  lut_shape = CV18xx::lut_table_shape(CVK_FMT_I8);
  CV18xx::set_layer_id(layer_id);
}

void TgBnrPreprocessKernel::selectTilePolicy() {
  // table size
  uint32_t lmem_used =
      2 * CV18xx::lmem_tensor_to_size(lut_shape, CVK_FMT_I8, 1);
  uint32_t w_step = ow;
  uint32_t c_step = oh;
  uint32_t n_step = n;
  auto tiling = [&]() {
    for (w_step = ow; w_step > 0; w_step /= 2) {
      for (c_step = oh; c_step > 0; c_step -= CV18xx::NPU_NUM) {
        for (n_step = n; n_step > 0; n_step /= 2) {
          cvk_tl_shape_t in_shape = {n_step, c_step, 1, w_step * 3};
          cvk_tl_shape_t out_shape = {n_step, c_step, 1, w_step};
          uint32_t in_size =
              CV18xx::lmem_tensor_to_size(in_shape, CVK_FMT_I8, 1) * 2;
          uint32_t tmp_i8_size =
              CV18xx::lmem_tensor_to_size(out_shape, CVK_FMT_I8, 1);
          uint32_t tmp_bf16_size =
              CV18xx::lmem_tensor_to_size(out_shape, CVK_FMT_BF16, 1);
          uint32_t mid_size = tmp_i8_size * 6 + tmp_bf16_size;
          uint32_t out_size = tmp_bf16_size * 4;
          if (in_size + out_size + mid_size + lmem_used <= CV18xx::LMEM_BYTES) {
            return;
          }
        }
      }
    }
  };
  tiling();
  if (w_step > 0) {
    for (uint32_t loc_w = 0; loc_w < ow; loc_w += w_step) {
      for (uint32_t loc_c = 0; loc_c < oh; loc_c += c_step) {
        for (uint32_t loc_n = 0; loc_n < n; loc_n += n_step) {
          CV18xx::tiling_info_t tile;
          tile.n = std::min(n_step, n - loc_n);
          tile.c = std::min(c_step, oh - loc_c);
          tile.h = 1;
          tile.w = std::min(w_step, ow - loc_w);
          tile.pos_n = loc_n;
          tile.pos_c = loc_c;
          tile.pos_h = 0;
          tile.pos_w = loc_w * 3;
          tile.offset = tile.pos_w * 3 * in_gstride.w +
                        tile.pos_c * in_gstride.c + tile.pos_n * in_gstride.n;
          // printf("begin[%d, %d, %d, %d] shape[%d, %d, %d, %d]\n", tile.pos_n,
          //        tile.pos_c, tile.pos_h, tile.pos_w, tile.n, tile.c, tile.h,
          //        tile.w);
          tiles.emplace_back(tile);
        }
      }
    }
  } else {
    assert(0);
  }
}

void TgBnrPreprocessKernel::allocLmem() {
  cvk_tl_shape_t gshape =
      CV18xx::tl_shape_t4(tiles[0].n, tiles[0].c, tiles[0].h, tiles[0].w * 3);
  cvk_tl_shape_t pixel_shape =
      CV18xx::tl_shape_t4(tiles[0].n, tiles[0].c, tiles[0].h,
                          align_up(tiles[0].w, CV18xx::tiu_eu_num(CVK_FMT_I8)));
  // printf("tile[%d, %d, %d, %d]\n", tiles[0].n, tiles[0].c, tiles[0].h,
  //        tiles[0].w);
  // alloc for and
  tl_and_0f_shift = CV18xx::lmem_alloc_tensor(pixel_shape, CVK_FMT_I8, 1);
  tl_and_f0_shift = CV18xx::lmem_alloc_tensor(pixel_shape, CVK_FMT_I8, 1);
  auto tl_and_f0_high = CV18xx::lmem_alloc_tensor(pixel_shape, CVK_FMT_I8, 1);

  CV18xx::tiu_zeros(layer_id, tl_and_0f_shift);
  CV18xx::tiu_zeros(layer_id, tl_and_f0_shift);
  CV18xx::tiu_zeros(layer_id, tl_and_f0_high);

  // create 0x0F
  cvk_tiu_add_param_t p01 = {0};
  p01.res_high = nullptr;
  p01.res_low = tl_and_0f_shift;
  p01.a_high = tl_and_f0_shift;
  p01.a_low = tl_and_0f_shift;
  p01.b_is_const = true;
  p01.b_const.val = 0x0F;
  p01.b_const.is_signed = 0;
  p01.rshift_bits = 0;
  p01.layer_id = layer_id;
  p01.relu_enable = 0;
  CV18xx::tiu_add(&p01);

  // create 0xF0
  cvk_tiu_mul_param_t p_mul = {0};
  p_mul.res_high = tl_and_f0_high;
  p_mul.res_low = tl_and_f0_shift;
  p_mul.a = tl_and_0f_shift;
  p_mul.b_const.val = 16;
  p_mul.b_const.is_signed = false;
  p_mul.b_is_const = true;
  p_mul.rshift_bits = 0;
  p_mul.layer_id = layer_id;
  p_mul.relu_enable = 0;
  CV18xx::tiu_mul(&p_mul);

  CV18xx::lmem_free_tensor(tl_and_f0_high);

  for (int i = 0; i < 2; ++i) {
    tl_table[i] = CV18xx::lmem_alloc_tensor(lut_shape, CVK_FMT_I8, 1);
  }
  for (int i = 0; i < 2; i++) {
    tl_in[i] = CV18xx::lmem_alloc_tensor(gshape, CVK_FMT_I8, 1);
  }
  for (int i = 0; i < 2; i++) {
    tl_out_A[i] = CV18xx::lmem_alloc_tensor(pixel_shape, CVK_FMT_BF16, 1);
    tl_out_B[i] = CV18xx::lmem_alloc_tensor(pixel_shape, CVK_FMT_BF16, 1);
  }
  tl_out_tmp_bf16 = CV18xx::lmem_alloc_tensor(pixel_shape, CVK_FMT_BF16, 1);
  for (int i = 0; i < 4; i++) {
    tl_out_tmp_int8[i] = CV18xx::lmem_alloc_tensor(pixel_shape, CVK_FMT_I8, 1);
  }

  // load table
  CV18xx::tdma_load_table(tl_table[0], ga_table_high);
  CV18xx::tdma_load_table(tl_table[1], ga_table_low);
}

void TgBnrPreprocessKernel::deallocLmem() {
  for (int i = 3; i >= 0; i--) {
    CV18xx::lmem_free_tensor(tl_out_tmp_int8[i]);
  }
  for (int i = 1; i >= 0; i--) {
    CV18xx::lmem_free_tensor(tl_out_B[i]);
    CV18xx::lmem_free_tensor(tl_out_A[i]);
  }
  for (int i = 1; i >= 0; i--) {
    CV18xx::lmem_free_tensor(tl_in[i]);
  }
  for (int i = 1; i >= 0; i--) {
    CV18xx::lmem_free_tensor(tl_table[i]);
  }
  CV18xx::lmem_free_tensor(tl_and_f0_shift);
  CV18xx::lmem_free_tensor(tl_and_0f_shift);
}

void TgBnrPreprocessKernel::schedule() {
  allocLmem();
  int32_t total_steps = tiles.size();
  for (int iter = 0; iter < 2; ++iter) {
    for (int32_t i = 0; i < total_steps + 2; i++) {
      CV18xx::parallel_enable();

      if (i - 1 >= 0 && i - 1 < total_steps) {
        compute(i - 1);
      }
      if (i < total_steps) {
        load(i, iter);
      }
      if (i - 2 >= 0) {
        store(i - 2, iter);
      }
      CV18xx::parallel_disable();
    }
  }
  deallocLmem();
}

void TgBnrPreprocessKernel::refresh(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  tl_ifmap = *tl_in[step_idx % 2];
  tl_ofmap_A = *tl_out_A[step_idx % 2];
  tl_ofmap_B = *tl_out_B[step_idx % 2];
  auto in_shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w * 3);
  auto out_shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  auto in_stride = CV18xx::tl_default_stride(in_shape, CVK_FMT_I8, 1);
  auto out_stride = CV18xx::tl_default_stride(out_shape, CVK_FMT_BF16, 1);
  auto out_stride_i8 = CV18xx::tl_default_stride(out_shape, CVK_FMT_I8, 1);
  tl_ifmap.shape = in_shape;
  tl_ifmap.stride = in_stride;
  tl_ofmap_A.shape = out_shape;
  tl_ofmap_A.stride = out_stride;
  tl_ofmap_B.shape = out_shape;
  tl_ofmap_B.stride = out_stride;

  tl_out_tmp_bf16->shape = out_shape;
  tl_out_tmp_bf16->stride = out_stride;
  for (int i = 0; i < 4; ++i) {
    tl_out_tmp_int8[i]->shape = out_shape;
    tl_out_tmp_int8[i]->stride = out_stride_i8;
  }
  tl_and_0f_shift->shape = out_shape;
  tl_and_0f_shift->stride = out_stride_i8;
  tl_and_f0_shift->shape = out_shape;
  tl_and_f0_shift->stride = out_stride_i8;
}

void TgBnrPreprocessKernel::load(int32_t step_idx, int iter) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  auto base_offset = in_gstride.c / 2 * (start_h + iter) + start_w;
  CV18xx::tdma_load_stride(&tl_ifmap, ga_input + tile.offset + base_offset,
                           in_gstride);
}

void TgBnrPreprocessKernel::store(int32_t step_idx, int iter) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  auto base_addr_A = ga_out[iter * 2];
  auto base_addr_B = ga_out[iter * 2 + 1];
  auto base_offset = tile.pos_w * out_gstride.w + tile.pos_h * out_gstride.h +
                     tile.pos_c * out_gstride.c + tile.pos_n * out_gstride.n;

  CV18xx::tdma_store_stride(&tl_ofmap_A, base_addr_A + base_offset,
                            out_gstride);
  CV18xx::tdma_store_stride(&tl_ofmap_B, base_addr_B + base_offset,
                            out_gstride);
}

void TgBnrPreprocessKernel::split_pixel(cvk_tl_t *in_pixel, cvk_tl_t *in_AB,
                                        cvk_tl_t *out_high, cvk_tl_t *out_low,
                                        cvk_tl_t *tl_shift_0f,
                                        cvk_tl_t *tl_shift_f0, bool is_A) {
  CV18xx::tiu_zeros(layer_id, out_high);
  // deal AB
  // AB & 0x0F

  // copy AB[A or B] & 0x0F to A
  cvk_tiu_and_int8_param_t p_and = {0};
  p_and.res = out_low;
  p_and.a = in_AB;
  p_and.b = is_A ? tl_shift_0f : tl_shift_f0;
  p_and.layer_id = layer_id;
  CV18xx::tiu_and_int8(&p_and);

  if (!is_A) {
    cvk_tiu_add_param_t p01 = {0};
    p01.res_high = out_high;
    p01.res_low = out_low;
    p01.a_high = out_high;
    p01.a_low = out_low;
    p01.b_is_const = true;
    p01.b_const.val = 0;
    p01.b_const.is_signed = false;
    p01.rshift_bits = 4;
    p01.layer_id = layer_id;
    p01.relu_enable = 0;
    CV18xx::tiu_add(&p01);
  }

  // (A << 4) + A_low
  cvk_tiu_mac_param_t p_mac = {0};
  p_mac.res_high = out_high;
  p_mac.res_low = out_low;
  p_mac.a = in_pixel;
  p_mac.res_is_int8 = false;
  p_mac.b_const.val = 16;
  p_mac.b_is_const = 1;
  p_mac.b_const.is_signed = false;
  p_mac.lshift_bits = 0;
  p_mac.rshift_bits = 0;
  p_mac.layer_id = layer_id;
  p_mac.relu_enable = 0;
  CV18xx::tiu_mac(&p_mac);

  // A_high & 0x0F
  p_and.res = out_high;
  p_and.a = out_high;
  p_and.b = tl_shift_0f;
  p_and.layer_id = layer_id;
  CV18xx::tiu_and_int8(&p_and);
}

void TgBnrPreprocessKernel::int16_to_bf16(cvk_tl_t *in_high, cvk_tl_t *in_low,
                                          cvk_tl_t *out, cvk_tl_t *tmp,
                                          cvk_tl_t *bf_high, cvk_tl_t *bf_low,
                                          cvk_tl_t *table_high,
                                          cvk_tl_t *table_low) {

  cvk_tl_t out_low = *out;
  out_low.fmt = CVK_FMT_I8;
  out_low.stride.w = 2;
  cvk_tl_t out_high = *out;
  out_high.fmt = CVK_FMT_I8;
  out_high.stride.w = 2;
  out_high.start_address += 1;

  cvk_tl_t tmp_low = *tmp;
  tmp_low.fmt = CVK_FMT_I8;
  tmp_low.stride.w = 2;
  cvk_tl_t tmp_high = *tmp;
  tmp_high.fmt = CVK_FMT_I8;
  tmp_high.stride.w = 2;
  tmp_high.start_address += 1;

  // low 8bit -> bf16
  cvk_tiu_lookup_table_param_t p = {0};
  p.ofmap = bf_low;
  p.ifmap = in_low;
  p.table = table_low;
  p.layer_id = layer_id;
  CV18xx::tiu_lookup_table(&p);

  p.ofmap = bf_high;
  p.ifmap = in_low;
  p.table = table_high;
  p.layer_id = layer_id;
  CV18xx::tiu_lookup_table(&p);

  // // copy two 8 bit -> bf16
  cvk_tiu_copy_param_t param;
  param.src = bf_low;
  param.dst = &out_low;
  CV18xx::tiu_copy(&param);

  param.src = bf_high;
  param.dst = &out_high;
  CV18xx::tiu_copy(&param);

  // high 8bit -> bf16
  p.ofmap = bf_low;
  p.ifmap = in_high;
  p.table = table_low;
  p.layer_id = layer_id;
  CV18xx::tiu_lookup_table(&p);

  p.ofmap = bf_high;
  p.ifmap = in_high;
  p.table = table_high;
  p.layer_id = layer_id;
  CV18xx::tiu_lookup_table(&p);

  param.src = bf_low;
  param.dst = &tmp_low;
  CV18xx::tiu_copy(&param);

  param.src = bf_high;
  param.dst = &tmp_high;
  CV18xx::tiu_copy(&param);

  cvk_tiu_mac_param_t p_mac = {0};
  p_mac.res_high = nullptr;
  p_mac.res_low = out;
  p_mac.a = tmp;
  p_mac.res_is_int8 = false;
  p_mac.b_const.val = CV18xx::convert_fp32_to_bf16(256);
  p_mac.b_is_const = 1;
  p_mac.b_const.is_signed = true;
  p_mac.lshift_bits = 0;
  p_mac.rshift_bits = 0;
  p_mac.layer_id = layer_id;
  p_mac.relu_enable = 0;
  CV18xx::tiu_mac(&p_mac);
}

void TgBnrPreprocessKernel::compute(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  cvk_tl_t tl_in_A = tl_ifmap;
  tl_in_A.shape.w = tile.w;
  tl_in_A.stride.w = tl_ifmap.stride.w * 3;

  cvk_tl_t tl_in_AB = tl_in_A;
  tl_in_AB.start_address = tl_in_A.start_address + 2;

  cvk_tl_t tl_in_B = tl_in_A;
  tl_in_B.start_address = tl_in_A.start_address + 1;
  split_pixel(&tl_in_A, &tl_in_AB, tl_out_tmp_int8[1], tl_out_tmp_int8[0],
              tl_and_0f_shift, tl_and_f0_shift, true);
  int16_to_bf16(tl_out_tmp_int8[1], tl_out_tmp_int8[0], &tl_ofmap_A,
                tl_out_tmp_bf16, tl_out_tmp_int8[2], tl_out_tmp_int8[3],
                tl_table[0], tl_table[1]);
  split_pixel(&tl_in_B, &tl_in_AB, tl_out_tmp_int8[1], tl_out_tmp_int8[0],
              tl_and_0f_shift, tl_and_f0_shift, false);
  int16_to_bf16(tl_out_tmp_int8[1], tl_out_tmp_int8[0], &tl_ofmap_B,
                tl_out_tmp_bf16, tl_out_tmp_int8[2], tl_out_tmp_int8[3],
                tl_table[0], tl_table[1]);
}

void cvi_backend_tg_bnr_preprocess_kernel(
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
    gaddr_t ga_table_high, gaddr_t ga_table_low, int n, int c, int h, int w,
    int start_h, int start_w, int channel_order[4], cvk_fmt_t fmt) {
  TgBnrPreprocessKernel kernel;
  kernel.init(layer_id, ga_input, ga_output, ga_table_high, ga_table_low, n, c,
              h, w, start_h, start_w, channel_order, fmt);
  kernel.selectTilePolicy();
  kernel.schedule();
}

} // namespace backend
} // namespace tpu_mlir
