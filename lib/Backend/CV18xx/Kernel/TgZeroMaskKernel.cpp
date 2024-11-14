//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/Kernel/TgZeroMaskKernel.hpp"

#define DEBUG_TYPE "cvi_backend_zero_mask_kernel"

namespace tpu_mlir {
namespace backend {
void TgZeroMaskKernel::init(uint32_t layer_id, gaddr_t ga_input,
                            gaddr_t ga_output, int n, int c, int h, int w,
                            bool positive, cvk_fmt_t fmt) {
  this->layer_id = layer_id;
  this->n = n;
  this->c = c;
  this->h = h;
  this->w = w;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->positive = positive;
  this->fmt = fmt;
  this->blob_num = (fmt == CVK_FMT_BF16 ? 5 : 4);
  CV18xx::set_layer_id(layer_id);
}

void TgZeroMaskKernel::selectTilePolicy() {
  CV18xx::tiling_packing(tiles, n, c, h, w, fmt, blob_num, 0,
                         CV18xx::TilingAll);
}

void TgZeroMaskKernel::allocLmem() {
  auto &tile = tiles[0];
  auto shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  if (fmt == CVK_FMT_BF16) {
    shape.w *= 2;
  }
  for (int i = 0; i < blob_num; i++) {
    tl_mem[i] = CV18xx::lmem_alloc_tensor(shape, CVK_FMT_U8, 1);
  }
}

void TgZeroMaskKernel::deallocLmem() {
  for (int i = blob_num - 1; i >= 0; i--) {
    CV18xx::lmem_free_tensor(tl_mem[i]);
  }
}

void TgZeroMaskKernel::refresh(int32_t step_idx) {
  auto &tile = tiles[step_idx];

  auto shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  int offset = 0;
  if (fmt == CVK_FMT_BF16) {
    auto stride = CV18xx::tl_default_stride(shape, CVK_FMT_U8, 1);
    tl_buffer = *tl_mem[0];
    tl_buffer.shape = shape;
    tl_buffer.stride = stride;
    offset = 1;
    shape.w *= 2;
  }
  auto stride = CV18xx::tl_default_stride(shape, CVK_FMT_U8, 1);
  tl_ifmap = *tl_mem[step_idx % 2 + offset];
  tl_ofmap = *tl_mem[step_idx % 2 + 2 + offset];
  tl_ifmap.shape = shape;
  tl_ifmap.stride = stride;
  tl_ofmap.shape = shape;
  tl_ofmap.stride = stride;
}

void TgZeroMaskKernel::load(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  CV18xx::tdma_load(&tl_ifmap, ga_input + tile.offset);
}

void TgZeroMaskKernel::store(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  CV18xx::tdma_store(&tl_ofmap, ga_output + tile.offset);
}

void TgZeroMaskKernel::compute_bf16() {
  cvk_tl_t tl_ifmap_low = tl_ifmap;
  tl_ifmap_low.shape.w /= 2;
  tl_ifmap_low.stride.w = 2;
  cvk_tl_t tl_ifmap_high = tl_ifmap_low;
  tl_ifmap_high.start_address += 1;

  cvk_tiu_or_int8_param_t p0 = {0};
  p0.res = &tl_ifmap_low;
  p0.a = &tl_ifmap_low;
  p0.b = &tl_ifmap_high;
  p0.layer_id = layer_id;
  CV18xx::tiu_or_int8(&p0);

  cvk_tiu_copy_param_t p = {0};
  p.dst = &tl_buffer;
  p.src = &tl_ifmap_low;
  p.layer_id = layer_id;
  CV18xx::tiu_copy(&p);

  cvk_tl_t tl_high = tl_buffer;
  tl_high.start_address = tl_ofmap.start_address;
  CV18xx::tiu_zeros(layer_id, &tl_high);

  cvk_tiu_add_param_t p01 = {0};
  p01.res_high = &tl_high;
  p01.res_low = &tl_buffer;
  p01.a_high = &tl_high;
  p01.a_low = &tl_buffer;
  p01.b_is_const = true;
  p01.b_const.val = 255;
  p01.b_const.is_signed = 0;
  p01.rshift_bits = 0;
  p01.layer_id = layer_id;
  p01.relu_enable = 0;
  CV18xx::tiu_add(&p01);

  p.dst = &tl_buffer;
  p.src = &tl_high;
  p.layer_id = layer_id;
  CV18xx::tiu_copy(&p);

  if (positive == false) {
    tl_high.fmt = CVK_FMT_I8;
    tl_buffer.fmt = CVK_FMT_I8;
    cvk_tiu_mul_param_t p2 = {0};
    p2.res_high = &tl_high;
    p2.res_low = &tl_buffer;
    p2.a = &tl_buffer;
    p2.b_const.val = -1;
    p2.b_const.is_signed = 1;
    p2.b_is_const = 1;
    p2.rshift_bits = 0;
    p2.layer_id = layer_id;
    p2.relu_enable = 0;
    CV18xx::tiu_mul(&p2);
    cvk_tiu_add_param_t p02 = {0};
    p02.res_high = &tl_high;
    p02.res_low = &tl_buffer;
    p02.a_high = &tl_high;
    p02.a_low = &tl_buffer;
    p02.b_is_const = true;
    p02.b_const.is_signed = 1;
    p02.b_const.val = 1;
    p02.rshift_bits = 0;
    p02.layer_id = layer_id;
    p02.relu_enable = 0;
    CV18xx::tiu_add(&p02);
    tl_buffer.fmt = CVK_FMT_U8;
    tl_high.fmt = CVK_FMT_U8;
  }

  cvk_tl_t tl_ofmap_low = tl_ofmap;
  tl_ofmap_low.shape.w /= 2;
  tl_ofmap_low.stride.w = 2;
  cvk_tl_t tl_ofmap_high = tl_ofmap_low;
  tl_ofmap_high.start_address += 1;

  cvk_tl_t tl_low = tl_buffer;
  tl_low.start_address = tl_ifmap.start_address;

  cvk_tiu_mul_param_t p4 = {0};
  p4.res_high = &tl_high;
  p4.res_low = &tl_low;
  p4.a = &tl_buffer;
  p4.b_const.val = 0x80;
  p4.b_const.is_signed = 0;
  p4.b_is_const = 1;
  p4.rshift_bits = 0;
  p4.layer_id = layer_id;
  p4.relu_enable = 0;
  CV18xx::tiu_mul(&p4);

  cvk_tiu_mul_param_t p3 = {0};
  p3.res_high = &tl_high;
  p3.res_low = &tl_buffer;
  p3.a = &tl_buffer;
  p3.b_const.val = 0x3F;
  p3.b_const.is_signed = 0;
  p3.b_is_const = 1;
  p3.rshift_bits = 0;
  p3.layer_id = layer_id;
  p3.relu_enable = 0;
  CV18xx::tiu_mul(&p3);

  p.dst = &tl_ofmap_high;
  p.src = &tl_buffer;
  p.layer_id = layer_id;
  CV18xx::tiu_copy(&p);
  p.dst = &tl_ofmap_low;
  p.src = &tl_low;
  p.layer_id = layer_id;
  CV18xx::tiu_copy(&p);
}

void TgZeroMaskKernel::compute_int8() {
  CV18xx::tiu_zeros(layer_id, &tl_ofmap);
  cvk_tiu_add_param_t p1 = {0};
  p1.res_high = &tl_ofmap;
  p1.res_low = &tl_ifmap;
  p1.a_high = &tl_ofmap;
  p1.a_low = &tl_ifmap;
  p1.b_is_const = true;
  p1.b_const.val = 1;
  p1.rshift_bits = 0;
  p1.layer_id = layer_id;
  p1.relu_enable = 0;
  CV18xx::tiu_add(&p1);

  cvk_tiu_max_param_t param_relu = {0};
  param_relu.max = &tl_ifmap;
  param_relu.a = &tl_ifmap;
  param_relu.b_is_const = true;
  param_relu.b_const.val = (0);
  param_relu.b_const.is_signed = 1;
  param_relu.layer_id = layer_id;
  CV18xx::tiu_max(&param_relu);

  cvk_tiu_mul_param_t p3 = {0};
  p3.res_high = &tl_ofmap;
  p3.res_low = &tl_ifmap;
  p3.a = &tl_ifmap;
  p3.b_const.val = -1;
  p3.b_const.is_signed = 1;
  p3.b_is_const = 1;
  p3.rshift_bits = 0;
  p3.layer_id = layer_id;
  p3.relu_enable = 0;
  CV18xx::tiu_mul(&p3);

  cvk_tiu_add_param_t p2 = {0};
  p2.res_high = &tl_ofmap;
  p2.res_low = &tl_ifmap;
  p2.a_high = &tl_ofmap;
  p2.a_low = &tl_ifmap;
  p2.b_is_const = true;
  p2.b_const.val = 2;
  p2.rshift_bits = 0;
  p2.layer_id = layer_id;
  p2.relu_enable = 0;
  CV18xx::tiu_add(&p2);

  CV18xx::tiu_max(&param_relu);

  if (positive) {
    cvk_tiu_mul_param_t p4 = {0};
    p4.res_high = &tl_ofmap;
    p4.res_low = &tl_ifmap;
    p4.a = &tl_ifmap;
    p4.b_const.val = -1;
    p4.b_const.is_signed = 1;
    p4.b_is_const = 1;
    p4.rshift_bits = 0;
    p4.layer_id = layer_id;
    p4.relu_enable = 0;
    CV18xx::tiu_mul(&p4);

    cvk_tiu_add_param_t p5 = {0};
    p5.res_high = &tl_ofmap;
    p5.res_low = &tl_ifmap;
    p5.a_high = &tl_ofmap;
    p5.a_low = &tl_ifmap;
    p5.b_is_const = true;
    p5.b_const.val = 2;
    p5.rshift_bits = 0;
    p5.layer_id = layer_id;
    p5.relu_enable = 0;
    CV18xx::tiu_add(&p5);
    CV18xx::tiu_max(&param_relu);
  }
  cvk_tiu_mul_param_t p6 = {0};
  p6.res_high = &tl_ofmap;
  p6.res_low = &tl_ifmap;
  p6.a = &tl_ifmap;
  p6.b_const.val = 127;
  p6.b_const.is_signed = 1;
  p6.b_is_const = 1;
  p6.rshift_bits = 0;
  p6.layer_id = layer_id;
  p6.relu_enable = 0;
  CV18xx::tiu_mul(&p6);
  cvk_tiu_copy_param_t p = {0};
  p.src = &tl_ifmap;
  p.dst = &tl_ofmap;
  p.layer_id = layer_id;
  CV18xx::tiu_copy(&p);
}

void TgZeroMaskKernel::compute(int32_t step_idx) {
  refresh(step_idx);
  if (fmt == CVK_FMT_BF16) {
    compute_bf16();
  } else {
    compute_int8();
  }
}

void TgZeroMaskKernel::schedule() {
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

void cvi_backend_zero_mask_kernel(uint32_t layer_id, gaddr_t ga_input,
                                  gaddr_t ga_output, int n, int c, int h, int w,
                                  bool positive, cvk_fmt_t fmt) {
  TgZeroMaskKernel kernel;
  kernel.init(layer_id, ga_input, ga_output, n, c, h, w, positive, fmt);
  kernel.selectTilePolicy();
  kernel.schedule();
}
} // namespace backend
} // namespace tpu_mlir
