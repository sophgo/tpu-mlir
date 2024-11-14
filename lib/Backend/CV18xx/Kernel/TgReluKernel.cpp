//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/Kernel/TgReluKernel.hpp"
#define DEBUG_TYPE "cvi_backend_relu_kernel"

namespace tpu_mlir {
namespace backend {
void TgReluKernel::init(uint32_t layer_id, int32_t n, int32_t c, int32_t h,
                        int32_t w, gaddr_t ga_input, gaddr_t ga_output,
                        gaddr_t ga_slope, float negative_slope, int GT_rshift,
                        int GT_scale, int LE_rshift, int LE_scale,
                        cvk_fmt_t fmt, mode_t mode) {
  this->layer_id = layer_id;
  this->n = n;
  this->c = c;
  this->h = h;
  this->w = w;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->ga_slope = ga_slope;
  this->negative_slope = negative_slope;
  this->GT_rshift = GT_rshift;
  this->GT_scale = GT_scale;
  this->LE_rshift = LE_rshift;
  this->LE_scale = LE_scale;
  this->fmt = fmt;
  this->mode = mode;
  gstride = CV18xx::tg_default_stride(c, h, w, fmt);

  CV18xx::set_layer_id(layer_id);
}

void TgReluKernel::selectTilePolicy() {
  // blob_num = 4, input/output with flip
  int blob_num = 4;
  if (mode == PRELU) {
    auto slope_size = CV18xx::lmem_tensor_to_size(1, c, 1, 1, fmt, 1);
    CV18xx::tiling_packing(tiles, n, c, h, w, fmt, blob_num, slope_size,
                           CV18xx::TilingNHW);

  } else {
    CV18xx::tiling_packing(tiles, n, c, h, w, fmt, blob_num, 0,
                           CV18xx::TilingAll);
  }
}

void TgReluKernel::allocLmem() {
  if (mode == PRELU) {
    cvk_tl_shape_t slope_shape = CV18xx::tl_shape_t4(1, c, 1, 1);
    tl_slope = CV18xx::lmem_alloc_tensor(slope_shape, fmt, 1);
    CV18xx::tdma_load(tl_slope, ga_slope);
  }

  cvk_tl_shape_t tile_shape =
      CV18xx::tl_shape_t4(tiles[0].n, tiles[0].c, tiles[0].h, tiles[0].w);

  tl_input[0] = CV18xx::lmem_alloc_tensor(tile_shape, fmt, 1);
  tl_input[1] = CV18xx::lmem_alloc_tensor(tile_shape, fmt, 1);
  tl_output[0] = CV18xx::lmem_alloc_tensor(tile_shape, fmt, 1);
  tl_output[1] = CV18xx::lmem_alloc_tensor(tile_shape, fmt, 1);
}

void TgReluKernel::deallocLmem() {
  CV18xx::lmem_free_tensor(tl_output[1]);
  CV18xx::lmem_free_tensor(tl_output[0]);
  CV18xx::lmem_free_tensor(tl_input[1]);
  CV18xx::lmem_free_tensor(tl_input[0]);
  if (mode == PRELU) {
    CV18xx::lmem_free_tensor(tl_slope);
  }
}

cvk_tl_t TgReluKernel::get_input(int32_t step_idx, int32_t flip) {
  auto &tile = tiles[step_idx];
  auto tl_ifmap = *tl_input[flip];
  tl_ifmap.shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  tl_ifmap.stride = CV18xx::tl_default_stride(tl_ifmap.shape, fmt, 1);
  return tl_ifmap;
}

cvk_tl_t TgReluKernel::get_output(int32_t step_idx, int32_t flip) {
  auto &tile = tiles[step_idx];
  auto tl_ofmap = *tl_output[flip];
  tl_ofmap.shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  tl_ofmap.stride = CV18xx::tl_default_stride(tl_ofmap.shape, fmt, 1);
  return tl_ofmap;
}

void TgReluKernel::change_workspace_size(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  for (size_t i = 0; i < 2; i++) {
    tl_working[i]->shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    tl_working[i]->stride =
        CV18xx::tl_default_stride(tl_working[i]->shape, fmt, 1);
  }
  tl_pos_neg_map->shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  tl_pos_neg_map->stride =
      CV18xx::tl_default_stride(tl_pos_neg_map->shape, fmt, 1);
}

void TgReluKernel::load(int32_t step_idx, int32_t flip) {
  auto &tile = tiles[step_idx];
  auto tl_ifmap = get_input(step_idx, flip);
  if (mode == PRELU) {
    CV18xx::tdma_load_stride(&tl_ifmap, ga_input + tile.offset, gstride);
  } else {
    CV18xx::tdma_load(&tl_ifmap, ga_input + tile.offset);
  }
}

void TgReluKernel::compute_relu(int32_t step_idx, int32_t flip) {
  auto tl_ifmap = get_input(step_idx, flip);
  auto tl_ofmap = get_output(step_idx, flip);

  cvk_tiu_max_param_t p = {0};
  p.max = &tl_ofmap;
  p.a = &tl_ifmap;
  p.b_is_const = 1;
  p.layer_id = layer_id;

  if (fmt == CVK_FMT_BF16) {
    p.b_const.val = CV18xx::convert_fp32_to_bf16(0);
  } else {
    p.b_const.val = (0);
    if (fmt == CVK_FMT_I8) {
      p.b_const.is_signed = 1;
    } else if (fmt == CVK_FMT_U8) {
      p.b_const.is_signed = 0;
    } else {
      assert(0 && "fmt not supported");
    }
  }
  CV18xx::tiu_max(&p);
}

/*
  LeakyRelu in asymmetric quantization
  There are two cases in leaky relu, postive and negative
  postive cases:
    Qy = Sx/Sy(Qx - Zx) + Qy
  negative cases:
    Qy = alpha * Sx/Sy * (Qx - Zx) + Qy

  in scale Sx/Sy, we make it to 2^rshift * mutlipiler(8bit)
  therefore,
    Qy = 2^rshift * multiplier * (Qx - Zx) + Qy

  output = ((Qx+offset)) * multiplier) >> rshift
*/
void TgReluKernel::compute_leaky_relu_fixed_sym(int32_t step_idx,
                                                int32_t flip) {
  auto tl_ifmap = get_input(step_idx, flip);
  auto tl_ofmap = get_output(step_idx, flip);

  bool isIgnorePosPart = (GT_scale == 0 || (GT_scale == 1 && GT_rshift == 0));
  bool isSlopeSmallerThanOne = ((LE_scale >> LE_rshift) == 0);

  if (isIgnorePosPart && LE_scale >= 0) {
    cvk_tiu_mul_param_t p1 = {0};
    p1.res_high = nullptr;
    p1.res_low = &tl_ofmap;
    p1.a = &tl_ifmap;
    p1.b_const.val = LE_scale;
    p1.b_const.is_signed = true;
    p1.b_is_const = 1;
    p1.rshift_bits = LE_rshift;
    p1.layer_id = layer_id;
    p1.relu_enable = 0;
    CV18xx::tiu_mul(&p1);

    if (isSlopeSmallerThanOne) {
      cvk_tiu_max_param_t p2 = {0};
      p2.max = &tl_ofmap;
      p2.a = &tl_ofmap;
      p2.b = &tl_ifmap;
      p2.b_is_const = 0;
      p2.layer_id = layer_id;
      CV18xx::tiu_max(&p2);
    } else {
      cvk_tiu_min_param_t p2 = {0};
      p2.min = &tl_ofmap;
      p2.a = &tl_ofmap;
      p2.b = &tl_ifmap;
      p2.b_is_const = 0;
      p2.layer_id = layer_id;
      CV18xx::tiu_min(&p2);
    }
  } else {
    cvk_tiu_max_param_t p3 = {0};
    p3.max = &tl_ofmap;
    p3.a = &tl_ifmap;
    p3.b_is_const = 1;
    p3.b_const.is_signed = 1;
    p3.b_const.val = 0;
    p3.layer_id = layer_id;
    CV18xx::tiu_max(&p3);
    if (!isIgnorePosPart) {
      // tl_ofmap = (tl_ofmap * GT_scale) >> GT_rshift
      cvk_tiu_mul_param_t p4 = {0};
      p4.res_high = nullptr;
      p4.res_low = &tl_ofmap;
      p4.a = &tl_ofmap;
      p4.b_const.val = GT_scale;
      p4.b_const.is_signed = true;
      p4.b_is_const = 1;
      p4.rshift_bits = GT_rshift;
      p4.layer_id = layer_id;
      p4.relu_enable = 0;
      CV18xx::tiu_mul(&p4);
    }
    // tl_ifmap = min(0, tl_ifmap) select meg part
    cvk_tiu_min_param_t p5 = {0};
    p5.min = &tl_ifmap;
    p5.a = &tl_ifmap;
    p5.b_is_const = 1;
    p5.b_const.val = 0;
    p5.b_const.is_signed = 1;
    p5.layer_id = layer_id;
    CV18xx::tiu_min(&p5);

    // tl_ifmap = (tl_ifmap * slope) >> LE_rshift
    cvk_tiu_mul_param_t p6 = {0};
    p6.res_high = nullptr;
    p6.res_low = &tl_ifmap;
    p6.a = &tl_ifmap;
    p6.b_const.val = LE_scale;
    p6.b_const.is_signed = true;
    p6.b_is_const = 1;
    p6.rshift_bits = LE_rshift;
    p6.layer_id = layer_id;
    p6.relu_enable = 0;
    CV18xx::tiu_mul(&p6);

    // tl_ofmap = or(tl_ofmap, tl_ifmap)
    cvk_tiu_or_int8_param_t p7 = {0};
    p7.res = &tl_ofmap;
    p7.a = &tl_ofmap;
    p7.b = &tl_ifmap;
    p7.layer_id = layer_id;
    CV18xx::tiu_or_int8(&p7);
  }
}

void TgReluKernel::compute_leaky_relu_bf16(int32_t step_idx, int32_t flip) {
  auto tl_ifmap = get_input(step_idx, flip);
  auto tl_ofmap = get_output(step_idx, flip);
  cvk_tiu_mul_param_t p1 = {0};
  p1.res_high = nullptr; // useless
  p1.res_low = &tl_ofmap;
  p1.a = &tl_ifmap;
  p1.b_const.val = CV18xx::convert_fp32_to_bf16(negative_slope);
  p1.b_const.is_signed = true;
  p1.b_is_const = true;
  p1.rshift_bits = 0;
  p1.layer_id = layer_id;
  p1.relu_enable = 0;
  CV18xx::tiu_mul(&p1);

  // 1. relu = max(tl_ifmap, relu)
  if (negative_slope <= 1) {
    cvk_tiu_max_param_t p2 = {0};
    p2.max = &tl_ofmap;
    p2.a = &tl_ifmap;
    p2.b_is_const = 0;
    p2.b_const.is_signed = 1;
    p2.b = &tl_ofmap;
    p2.layer_id = layer_id;
    CV18xx::tiu_max(&p2);
  } else {
    cvk_tiu_min_param_t p3 = {0};
    p3.min = &tl_ofmap;
    p3.a = &tl_ifmap;
    p3.b_is_const = 0;
    p3.b_const.is_signed = 1;
    p3.b = &tl_ofmap;
    p3.layer_id = layer_id;
    CV18xx::tiu_min(&p3);
  }
}

void TgReluKernel::compute_prelu_fixed(int32_t step_idx, int32_t flip) {
  auto tl_ifmap = get_input(step_idx, flip);
  auto tl_ofmap = get_output(step_idx, flip);
  cvk_tiu_max_param_t p1 = {0};
  p1.max = &tl_ofmap;
  p1.a = &tl_ifmap;
  p1.b_is_const = 1;
  p1.b_const.is_signed = 1;
  p1.b_const.val = 0;
  p1.layer_id = layer_id;
  CV18xx::tiu_max(&p1);

  // 1. relu = (relu * GT_scale) >> GT_rshift
  if (GT_scale != 0 && (GT_scale != 1 || GT_rshift != 0)) {
    cvk_tiu_mul_param_t p2 = {0};
    p2.res_high = nullptr;
    p2.res_low = &tl_ofmap;
    p2.a = &tl_ofmap;
    p2.b_const.val = GT_scale;
    p2.b_const.is_signed = true;
    p2.b_is_const = 1;
    p2.rshift_bits = GT_rshift;
    p2.layer_id = layer_id;
    p2.relu_enable = 0;
    CV18xx::tiu_mul(&p2);
  }

  // 2. neg = neg(0, botom)
  cvk_tiu_min_param_t p3 = {0};
  p3.min = &tl_ifmap;
  p3.a = &tl_ifmap;
  p3.b_is_const = 1;
  p3.b_const.val = 0;
  p3.b_const.is_signed = 1;
  p3.layer_id = layer_id;
  CV18xx::tiu_min(&p3);

  // 3. neg (n,c,h,w) = (neg(n,c,h,w) * slope(1,c,1,1)) >>
  // LE_rshift
  cvk_tiu_depthwise_pt_convolution_param_t p4 = {0};
  p4.ins_h = 0;
  p4.ins_last_h = 0;
  p4.ins_w = 0;
  p4.ins_last_w = 0;
  p4.pad_top = 0;
  p4.pad_bottom = 0;
  p4.pad_left = 0;
  p4.pad_right = 0;
  p4.stride_h = 1;
  p4.stride_w = 1;
  p4.dilation_h = 1;
  p4.dilation_w = 1;
  p4.ofmap = &tl_ifmap;
  p4.ifmap = &tl_ifmap;
  p4.weight = tl_slope;
  p4.bias = nullptr;
  p4.rshift_bits = LE_rshift;
  p4.relu_enable = 0;
  p4.layer_id = layer_id;
  p4.ins_val = 0;                                // symmetric quantization
  p4.ins_fp = CV18xx::convert_fp32_to_bf16(0.0); // symmetric quantization
  CV18xx::tiu_pt_depthwise_convolution(&p4);

  // 4. tl_ifmap = or relu, neg
  cvk_tiu_or_int8_param_t p5 = {0};
  p5.res = &tl_ofmap;
  p5.a = &tl_ofmap;
  p5.b = &tl_ifmap;
  p5.layer_id = layer_id;
  CV18xx::tiu_or_int8(&p5);
}

void TgReluKernel::compute_prelu_bf16(int32_t step_idx, int32_t flip) {
  auto tl_ifmap = get_input(step_idx, flip);
  auto tl_ofmap = get_output(step_idx, flip);
  cvk_tiu_min_param_t p1 = {0};
  p1.min = &tl_ofmap;
  p1.a = &tl_ifmap;
  p1.b_is_const = 1;
  p1.b_const.val = CV18xx::convert_fp32_to_bf16(0.0);
  p1.b_const.is_signed = 1;
  p1.layer_id = layer_id;
  CV18xx::tiu_min(&p1);

  // 2. neg (n,c,h,w) = (neg(n,c,h,w) * slope(1,c,1,1)) >> LE_rshift
  cvk_tiu_depthwise_pt_convolution_param_t p2 = {0};
  p2.ins_h = 0;
  p2.ins_last_h = 0;
  p2.ins_w = 0;
  p2.ins_last_w = 0;
  p2.pad_top = 0;
  p2.pad_bottom = 0;
  p2.pad_left = 0;
  p2.pad_right = 0;
  p2.stride_h = 1;
  p2.stride_w = 1;
  p2.dilation_h = 1;
  p2.dilation_w = 1;
  p2.ofmap = &tl_ofmap;
  p2.ifmap = &tl_ofmap;
  p2.weight = tl_slope;
  p2.bias = nullptr;
  p2.rshift_bits = 0;
  p2.relu_enable = 0;
  p2.layer_id = layer_id;
  p2.ins_val = 0;                                // symmetric quantization
  p2.ins_fp = CV18xx::convert_fp32_to_bf16(0.0); // symmetric quantization
  CV18xx::tiu_pt_depthwise_convolution(&p2);

  // 3. relu = relu(tl_ifmap), dirty it
  cvk_tiu_max_param_t p3 = {0};
  p3.max = &tl_ifmap;
  p3.a = &tl_ifmap;
  p3.b_is_const = 1;
  p3.b_const.is_signed = 1;
  p3.b_const.val = CV18xx::convert_fp32_to_bf16(0.0);
  p3.layer_id = layer_id;
  CV18xx::tiu_max(&p3);

  cvk_tiu_add_param_t p4 = {0};
  p4.res_high = nullptr;
  p4.res_low = &tl_ofmap;
  p4.a_high = nullptr;
  p4.a_low = &tl_ifmap;
  p4.b_is_const = false;
  p4.b.high = nullptr;
  p4.b.low = &tl_ofmap;
  p4.rshift_bits = 0;
  p4.layer_id = layer_id;
  p4.relu_enable = 0;
  CV18xx::tiu_add(&p4);
}

void TgReluKernel::compute(int32_t step_idx, int32_t flip) {
  switch (mode) {
  case RELU:
    compute_relu(step_idx, flip);
    break;
  case LEAKY_RELU:
    if (fmt == CVK_FMT_BF16) {
      compute_leaky_relu_bf16(step_idx, flip);
    } else {
      compute_leaky_relu_fixed_sym(step_idx, flip);
    }
    break;
  case PRELU:
    if (fmt == CVK_FMT_BF16) {
      compute_prelu_bf16(step_idx, flip);
    } else {
      compute_prelu_fixed(step_idx, flip);
    }
    break;
  default:
    break;
  }
}

void TgReluKernel::store(int32_t step_idx, int32_t flip) {
  auto &tile = tiles[step_idx];
  auto tl_ofmap = get_output(step_idx, flip);
  if (mode == PRELU) {
    CV18xx::tdma_store_stride(&tl_ofmap, ga_output + tile.offset, gstride);
  } else {
    CV18xx::tdma_store(&tl_ofmap, ga_output + tile.offset);
  }
}

void TgReluKernel::schedule() {
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

// i8/bf16 relu
void cvi_backend_tg_relu_kernel(uint32_t layer_id, uint64_t ga_input,
                                uint64_t ga_output, int n, int c, int h, int w,
                                cvk_fmt_t fmt) {
  TgReluKernel kernel;
  kernel.init(layer_id, n, c, h, w, ga_input, ga_output, 0, 0, 0, 0, 0, 0, fmt,
              TgReluKernel::RELU);
  kernel.selectTilePolicy();
  kernel.schedule();
}

// i8 leakyrelu
void cvi_backend_tg_fixed_leakyrelu_kernel(uint32_t layer_id, uint64_t ga_input,
                                           uint64_t ga_output, int n, int c,
                                           int h, int w, int GT_rshift,
                                           int LE_rshift, int GT_scale,
                                           int LE_scale) {
  TgReluKernel kernel;
  kernel.init(layer_id, n, c, h, w, ga_input, ga_output, 0, 0, GT_rshift,
              GT_scale, LE_rshift, LE_scale, CVK_FMT_I8,
              TgReluKernel::LEAKY_RELU);
  kernel.selectTilePolicy();
  kernel.schedule();
}

// bf16 leakyrelu
void cvi_backend_tg_bf16_leakyrelu_kernel(uint32_t layer_id, gaddr_t ga_input,
                                          gaddr_t ga_output,
                                          float negative_slope, int n, int c,
                                          int h, int w) {
  TgReluKernel kernel;
  kernel.init(layer_id, n, c, h, w, ga_input, ga_output, 0, negative_slope, 0,
              0, 0, 0, CVK_FMT_BF16, TgReluKernel::LEAKY_RELU);
  kernel.selectTilePolicy();
  kernel.schedule();
}

// i8 prelu
void cvi_backend_tg_fixed_prelu_kernel(uint32_t layer_id, uint64_t ga_input,
                                       uint64_t ga_output,
                                       uint64_t negative_scope_gaddr, int n,
                                       int c, int h, int w, int GT_rshift,
                                       int GT_scale, int LE_rshift) {
  TgReluKernel kernel;
  kernel.init(layer_id, n, c, h, w, ga_input, ga_output, negative_scope_gaddr,
              0, GT_rshift, GT_scale, LE_rshift, 0, CVK_FMT_I8,
              TgReluKernel::PRELU);
  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_bf16_prelu_kernel(uint32_t layer_id, gaddr_t ga_input,
                                      gaddr_t ga_output, gaddr_t ga_slope,
                                      int n, int c, int h, int w) {
  TgReluKernel kernel;
  kernel.init(layer_id, n, c, h, w, ga_input, ga_output, ga_slope, 0, 0, 0, 0,
              0, CVK_FMT_BF16, TgReluKernel::PRELU);
  kernel.selectTilePolicy();
  kernel.schedule();
}
} // namespace backend
} // namespace tpu_mlir
