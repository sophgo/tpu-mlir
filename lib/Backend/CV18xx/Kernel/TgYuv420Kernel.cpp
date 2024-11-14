//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/Kernel/TgYuv420Kernel.hpp"
#include "tpu_mlir/Support/MathUtils.h"

#define DEBUG_TYPE "cvi_backend_tgYuv420Kernel_kernel"
// yuv_type 1--yuv420_planar  2--yuv_nv12  3--yuv_nv21

// yuv420_planar
// data from vpss, y = h * w, u = h/2 * w/2, v = h/2 * w/2.
// format example:
//     y y y y y y y y
//     y y y y y y y y
//     u u u u
//     v v v v
// 183x y aligned by 64 bytes, w aligned by 32 bytes, channel aligned by 4K
// bytes 182x y aligned by 128 bytes, w aligned by 64 bytes, channel aligned by
// 64 bytes

// YUV => RGB :
//     R = 1.164(Y - 16) + 1.596(V - 128)
//     G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
//     B = 1.164(Y - 16)                  + 2.018(U - 128)
//

// tiling policy :
//   y  = [n, h/2, 2, w]
//   u  = [n, h/2, 1, w/2]
//   v  = [n, h/2, 1, w/2]
//   4u = [n, h/2, 2, w], upsample from u
//   4v = [n, h/2, 2, w], upsample from v
//   b  = [n, h/2, 2, w]
//   g  = [n, h/2, 2, w]
//   r  = [n, h/2, 2, w]
//
// h/2 sliced by NPU_NUM, and put on each lane
//

// yuv_nv12
// data from vpss, y = h * w, uv = h/2 * w
// format example:
//     y y y y y y y y
//     y y y y y y y y
//     u v u v u v u v
// 183x y aligned by 32 bytes, w aligned by 32 bytes, channel aligned by 4K
// bytes 182x y aligned by 64 bytes, w aligned by 64 bytes, channel aligned by
// 64 bytes

// YUV => RGB :
//     R = 1.164(Y - 16) + 1.596(V - 128)
//     G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
//     B = 1.164(Y - 16)                  + 2.018(U - 128)
//

// tiling policy :
//   y  = [n, h/2, 2, w]
//   uv  = [n, h/2, 1, w]
//   uv_cache = [n, h/2, 1, w/2]
//   4u = [n, h/2, 2, w], upsample from u
//   4v = [n, h/2, 2, w], upsample from v
//   b  = [n, h/2, 2, w]
//   g  = [n, h/2, 2, w]
//   r  = [n, h/2, 2, w]
//
// h/2 sliced by NPU_NUM, and put on each lane
//

// yuv_nv21
// data from vpss, y = h * w, uv = h/2 * w
// format example:
//     y y y y y y y y
//     y y y y y y y y
//     v u v u v u v u
// 183x y aligned by 32 bytes, w aligned by 32 bytes, channel aligned by 4K
// bytes 182x y aligned by 64 bytes, w aligned by 64 bytes, channel aligned by
// 64 bytes

// YUV => RGB :
//     R = 1.164(Y - 16) + 1.596(V - 128)
//     G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
//     B = 1.164(Y - 16)                  + 2.018(U - 128)
//

// tiling policy :
//   y  = [n, h/2, 2, w]
//   uv  = [n, h/2, 1, w]
//   uv_cache = [n, h/2, 1, w/2]
//   4u = [n, h/2, 2, w], upsample from u
//   4v = [n, h/2, 2, w], upsample from v
//   b  = [n, h/2, 2, w]
//   g  = [n, h/2, 2, w]
//   r  = [n, h/2, 2, w]
//
// h/2 sliced by NPU_NUM, and put on each lane
//
namespace tpu_mlir {
namespace backend {
void TgYuv420Kernel::init(uint32_t layer_id, gaddr_t ga_input,
                          gaddr_t ga_output, int n, int c, int h, int w,
                          const std::vector<int> &order, int32_t pixel_type,
                          int32_t channel_align, int32_t y_align,
                          int32_t w_align, cvk_fmt_t fmt) {
  assert(c == 3 && "rgb channel must be 3");
  // convert to output shape
  this->layer_id = layer_id;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->n = n;
  this->c = c;
  this->h = h;
  this->w = w;
  this->yuv_type = pixel_type; // 1--i420  2--nv12  3--nv21
  if (order.empty()) {
    this->order.push_back(0);
    this->order.push_back(1);
    this->order.push_back(2);
  } else {
    this->order = order;
    assert(order.size() == 3);
    for (auto &channel : order) {
      assert(channel < 3 && channel >= 0);
    }
  }
  this->fmt = fmt;
  if (fmt == CVK_FMT_I8) {
    // only support u8, regard i8 as u8
    fmt = CVK_FMT_U8;
  }
  assert(fmt == CVK_FMT_U8);
  y_w_aligned = align_up(w, y_align);
  int y_offset = 0;
  int u_offset = 0;
  int v_offset = 0;
  if (this->yuv_type == 1) {
    uv_w_aligned = align_up(w / 2, w_align);
    u_offset = align_up(y_offset + y_w_aligned * h, channel_align);
    v_offset = align_up(u_offset + uv_w_aligned * h / 2, channel_align);
    BLOB_NUM = 14;
  } else {
    uv_w_aligned = align_up(w, w_align);
    u_offset = align_up(y_offset + y_w_aligned * h, channel_align);
    v_offset = u_offset;
    BLOB_NUM = 12;
  }
  n_stride = align_up(v_offset + uv_w_aligned * h / 2, channel_align);
  ga_y = ga_input + y_offset;
  ga_u = ga_input + u_offset;
  ga_v = ga_input + v_offset;
  kernel_shape = CV18xx::tl_shape_t4(1, CV18xx::NPU_NUM, 2, 2);

  y_gstride = {static_cast<uint32_t>(n_stride),
               static_cast<uint32_t>(2 * y_w_aligned),
               static_cast<uint32_t>(y_w_aligned), static_cast<uint32_t>(1)};
  uv_gstride = {static_cast<uint32_t>(n_stride),
                static_cast<uint32_t>(uv_w_aligned),
                static_cast<uint32_t>(uv_w_aligned), static_cast<uint32_t>(1)};
  rgb_gstride = {static_cast<uint32_t>(h * w * 3), static_cast<uint32_t>(2 * w),
                 static_cast<uint32_t>(w), static_cast<uint32_t>(1)};

  // tiling step
  step_c = CV18xx::NPU_NUM;
  step_h = 2;
}

void TgYuv420Kernel::allocLmem() {
  if (tl_mem.size() != BLOB_NUM) {
    tl_mem.resize(BLOB_NUM, nullptr);
  }
  // for depthwise conv kernel
  tl_mem_kernel = CV18xx::lmem_alloc_tensor(kernel_shape, CVK_FMT_BF16, 1);
  cvk_tdma_g2l_tensor_fill_constant_param_t p = {0};
  p.constant = CV18xx::convert_fp32_to_bf16(1.0f);
  p.layer_id = layer_id;
  p.dst = tl_mem_kernel;
  CV18xx::tdma_g2l_tensor_fill_constant(&p);
  // for yuv,rgb,4u,4v
  auto y_shape = CV18xx::tl_shape_t4(step_n, step_c, step_h, step_w);
  for (int i = 0; i < 10; i++) {
    tl_mem[i] = CV18xx::lmem_alloc_tensor(y_shape, CVK_FMT_BF16, 1);
  }
  cvk_tl_shape_t uv_shape;
  if (yuv_type == 1) {
    uv_shape = CV18xx::tl_shape_t4(step_n, step_c, step_h / 2, step_w / 2);
    for (uint32_t i = 10; i < BLOB_NUM; i++) {
      tl_mem[i] = CV18xx::lmem_alloc_tensor(uv_shape, CVK_FMT_BF16, 1);
    }
  } else {
    uv_shape = CV18xx::tl_shape_t4(step_n, step_c, step_h / 2, step_w);
    for (uint32_t i = 10; i < BLOB_NUM; i++) {
      tl_mem[i] = CV18xx::lmem_alloc_tensor(uv_shape, CVK_FMT_BF16, 1);
    }
  }
}

void TgYuv420Kernel::deallocLmem() {
  for (int i = BLOB_NUM - 1; i >= 0; i--) {
    CV18xx::lmem_free_tensor(tl_mem[i]);
  }
  CV18xx::lmem_free_tensor(tl_mem_kernel);
}

void TgYuv420Kernel::selectTilePolicy() {
  uint32_t lmem_size =
      (uint32_t)CV18xx::LMEM_BYTES -
      CV18xx::lmem_tensor_to_size(kernel_shape, CVK_FMT_BF16, 1);
  uint32_t lmem_required = 0;
  int max_w = std::min(w, MAX_WIDTH);
  for (step_w = max_w; step_w > 0; step_w -= 2) {
    for (step_n = n; step_n > 0; --step_n) {
      cvk_tl_shape_t uv_shape;
      if (yuv_type == 1) {
        uv_shape = CV18xx::tl_shape_t4(step_n, step_c, step_h / 2, step_w / 2);
      } else {
        uv_shape = CV18xx::tl_shape_t4(step_n, step_c, step_h / 2, step_w);
      }
      auto y_shape = CV18xx::tl_shape_t4(step_n, step_c, step_h, step_w);
      // u,v
      uint32_t uv_need = 0;
      if (yuv_type == 1) {
        uv_need = 4 * CV18xx::lmem_tensor_to_size(uv_shape, CVK_FMT_BF16, 1);
      } else {
        uv_need = 2 * CV18xx::lmem_tensor_to_size(uv_shape, CVK_FMT_BF16, 1);
      }
      // 4u,4v, (y,r,g,b * 2)
      uint32_t y_need =
          10 * CV18xx::lmem_tensor_to_size(y_shape, CVK_FMT_BF16, 1);
      lmem_required = uv_need + y_need;
      if (lmem_required <= lmem_size) {
        goto after_loop;
      }
    }
  }
after_loop:
  if (lmem_required > lmem_size) {
    llvm::errs() << llvm::format(
        "Tilling failed, src shape:(%d,%d,%d,%d), fmt:%d\n", n, c, h, w, fmt);
    assert(0);
  }
  CV18xx::tiling_info_t tile;
  for (tile.pos_n = 0; tile.pos_n < n; tile.pos_n += step_n) {
    tile.n = std::min(n - tile.pos_n, step_n);
    for (tile.pos_c = 0; tile.pos_c < h / 2; tile.pos_c += step_c) {
      tile.c = std::min(h / 2 - tile.pos_c, step_c);
      tile.h = 2;
      for (tile.pos_w = 0; tile.pos_w < w; tile.pos_w += step_w) {
        tile.w = std::min(w - tile.pos_w, step_w);
        tiles.emplace_back(tile);
      }
    }
  }
}

void TgYuv420Kernel::refresh(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  auto y_shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  auto y_stride = CV18xx::tl_default_stride(y_shape, CVK_FMT_BF16, 1);
  cvk_tl_shape_t uv_shape;
  if (yuv_type == 1) {
    uv_shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h / 2, tile.w / 2);
  } else {
    uv_shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h / 2, tile.w);
  }
  auto uv_stride = CV18xx::tl_default_stride(uv_shape, CVK_FMT_BF16, 1);
  for (int i = 0; i < 10; i++) {
    tl_mem[i]->shape = y_shape;
    tl_mem[i]->stride = y_stride;
  }
  for (uint32_t i = 10; i < BLOB_NUM; i++) {
    tl_mem[i]->shape = uv_shape;
    tl_mem[i]->stride = uv_stride;
  }
  tl_y = *tl_mem[0 + (step_idx % 2)];
  tl_r = *tl_mem[2 + (step_idx % 2)];
  tl_g = *tl_mem[4 + (step_idx % 2)];
  tl_b = *tl_mem[6 + (step_idx % 2)];
  tl_4u = *tl_mem[8];
  tl_4v = *tl_mem[9];
  if (yuv_type == 1) {
    tl_u = *tl_mem[10 + (step_idx % 2)];
    tl_v = *tl_mem[12 + (step_idx % 2)];
  } else {
    tl_uv = *tl_mem[10 + (step_idx % 2)];
  }
  tl_kernel = *tl_mem_kernel;
  if (tile.c != CV18xx::NPU_NUM) {
    tl_kernel.shape.c = tile.c;
    tl_kernel.stride =
        CV18xx::tl_default_stride(tl_kernel.shape, tl_kernel.fmt, 1);
  }
}

void TgYuv420Kernel::load_u8_to_bf16(cvk_tl_t *dst, uint64_t src_gaddr,
                                     cvk_tg_stride_t stride) {
  cvk_tg_t src;
  src.start_address = src_gaddr;
  src.base_reg_index =
      CV18xx::getTdmaBaseSelectIndexFromGaddr(src.start_address);
  src.int8_rnd_mode = 0;
  src.fmt = fmt; // CVK_FMT_U8
  src.shape = {dst->shape.n, dst->shape.c, dst->shape.h, dst->shape.w};
  src.stride = stride;

  cvk_tdma_g2l_tensor_copy_param_t p = {0};
  p.src = &src;
  p.dst = dst;
  p.layer_id = layer_id;
  CV18xx::tdma_g2l_tensor_copy(&p);
}

void TgYuv420Kernel::store_bf16_to_u8(cvk_tl_t *src, uint64_t dst_gaddr,
                                      cvk_tg_stride_t stride) {

  cvk_tg_t dst = {0};
  dst.start_address = dst_gaddr;
  dst.base_reg_index =
      CV18xx::getTdmaBaseSelectIndexFromGaddr(dst.start_address);
  dst.int8_rnd_mode = 0;
  dst.fmt = fmt; // CVK_FMT_U8
  dst.shape = {src->shape.n, src->shape.c, src->shape.h, src->shape.w};
  dst.stride = stride;

  cvk_tdma_l2g_tensor_copy_param_t p1 = {0};
  p1.src = src;
  p1.dst = &dst;
  p1.layer_id = layer_id;
  CV18xx::tdma_l2g_tensor_copy(&p1);
}

void TgYuv420Kernel::load(int32_t step_idx) {
  refresh(step_idx);
  auto &tile = tiles[step_idx];
  if (yuv_type == 1) {
    uint64_t y_gaddr = ga_y + tile.pos_n * n_stride +
                       tile.pos_c * 2 * y_w_aligned + tile.pos_w;
    uint64_t u_gaddr = ga_u + tile.pos_n * n_stride +
                       tile.pos_c * uv_w_aligned + tile.pos_w / 2;
    uint64_t v_gaddr = ga_v + tile.pos_n * n_stride +
                       tile.pos_c * uv_w_aligned + tile.pos_w / 2;
    load_u8_to_bf16(&tl_y, y_gaddr, y_gstride);
    load_u8_to_bf16(&tl_u, u_gaddr, uv_gstride);
    load_u8_to_bf16(&tl_v, v_gaddr, uv_gstride);
  } else {
    uint64_t y_gaddr = ga_y + tile.pos_n * n_stride +
                       tile.pos_c * 2 * y_w_aligned + tile.pos_w;
    uint64_t uv_gaddr =
        ga_u + tile.pos_n * n_stride + tile.pos_c * uv_w_aligned + tile.pos_w;
    load_u8_to_bf16(&tl_y, y_gaddr, y_gstride);
    load_u8_to_bf16(&tl_uv, uv_gaddr, uv_gstride);
  }
}

void TgYuv420Kernel::store(int32_t step_idx) {
  refresh(step_idx);
  auto &tile = tiles[step_idx];
  cvk_tl_t *bgr[3] = {&tl_b, &tl_g, &tl_r};
  uint64_t rgb_offset =
      tile.pos_n * 3 * h * w + tile.pos_c * 2 * w + tile.pos_w;

  for (int order_idx = 0; order_idx < 3; order_idx++) {
    int c = order[order_idx];
    uint64_t rgb_gaddr = ga_output + rgb_offset + order_idx * h * w;
    store_bf16_to_u8(bgr[c], rgb_gaddr, rgb_gstride);
  }
}

void TgYuv420Kernel::compute(int32_t step_idx) {
  refresh(step_idx);
  // u -= 128, v-=128, y = (y - 16) * 1.164
  if (yuv_type != 1) {
    tl_u = tl_r;
    tl_u.shape = tl_uv.shape;
    tl_u.shape.w /= 2;
    tl_v = tl_g;
    tl_v.shape = tl_uv.shape;
    tl_v.shape.w /= 2;

    auto tl_u_src = tl_uv;
    tl_u_src.shape.w /= 2;
    tl_u_src.stride.w *= 2;
    if (yuv_type == 3) { // nv21
      tl_u_src.start_address += 2;
    }
    auto tl_v_src = tl_uv;
    tl_v_src.shape.w /= 2;
    tl_v_src.stride.w *= 2;
    if (yuv_type == 2) { // nv21
      tl_v_src.start_address += 2;
    }

    // Separate u from uv
    cvk_tiu_copy_param_t p = {0};
    p.dst = &tl_u;
    p.src = &tl_u_src;
    p.layer_id = layer_id;
    CV18xx::tiu_copy(&p);

    // Separate v from uv
    p = {0};
    p.dst = &tl_v;
    p.src = &tl_v_src;
    p.layer_id = layer_id;
    CV18xx::tiu_copy(&p);
  }

  cvk_tiu_add_param_t p1 = {0};
  p1.res_high = nullptr;
  p1.res_low = &tl_u;
  p1.a_high = nullptr;
  p1.a_low = &tl_u;
  p1.b_is_const = true;
  p1.b_const.val = CV18xx::convert_fp32_to_bf16(-128.0f);
  p1.b_const.is_signed = 1;
  p1.rshift_bits = 0;
  p1.layer_id = layer_id;
  p1.relu_enable = false;
  CV18xx::tiu_add(&p1);
  cvk_tiu_add_param_t p2 = {0};
  p2.res_high = nullptr;
  p2.res_low = &tl_v;
  p2.a_high = nullptr;
  p2.a_low = &tl_v;
  p2.b_is_const = true;
  p2.b_const.val = CV18xx::convert_fp32_to_bf16(-128.0f);
  p2.b_const.is_signed = 1;
  p2.rshift_bits = 0;
  p2.layer_id = layer_id;
  p2.relu_enable = false;
  CV18xx::tiu_add(&p2);
  cvk_tiu_add_param_t p21 = {0};
  p21.res_high = nullptr;
  p21.res_low = &tl_y;
  p21.a_high = nullptr;
  p21.a_low = &tl_y;
  p21.b_is_const = true;
  p21.b_const.val = CV18xx::convert_fp32_to_bf16(-16.0f);
  p21.b_const.is_signed = 1;
  p21.rshift_bits = 0;
  p21.layer_id = layer_id;
  p21.relu_enable = 0;
  CV18xx::tiu_add(&p21);
  cvk_tiu_mul_param_t p22 = {0};
  p22.res_high = nullptr;
  p22.res_low = &tl_y;
  p22.a = &tl_y;
  p22.b_const.val = CV18xx::convert_fp32_to_bf16(1.164f);
  p22.b_is_const = 1;
  p22.b_const.is_signed = 1;
  p22.rshift_bits = 0;
  p22.relu_enable = 0;
  p22.layer_id = layer_id;
  CV18xx::tiu_mul(&p22);

  // upsampe u, upsampe v
  cvk_tiu_depthwise_pt_convolution_param_t p3 = {0};
  p3.ofmap = &tl_4u;
  p3.ifmap = &tl_u;
  p3.weight = &tl_kernel;
  p3.bias = nullptr;
  p3.ins_h = 1;
  p3.ins_last_h = 0;
  p3.ins_w = 1;
  p3.ins_last_w = 0;
  p3.pad_top = 1;
  p3.pad_bottom = 1;
  p3.pad_left = 1;
  p3.pad_right = 1;
  p3.stride_h = 1;
  p3.stride_w = 1;
  p3.dilation_h = 1;
  p3.dilation_w = 1;
  p3.relu_enable = 0;
  p3.layer_id = layer_id;
  p3.ins_val = 0;                                // symmetric quantization
  p3.ins_fp = CV18xx::convert_fp32_to_bf16(0.0); // symmetric quantization
  CV18xx::tiu_pt_depthwise_convolution(&p3);

  cvk_tiu_depthwise_pt_convolution_param_t p4 = {0};
  p4.ofmap = &tl_4v;
  p4.ifmap = &tl_v;
  p4.weight = &tl_kernel;
  p4.bias = nullptr;
  p4.ins_h = 1;
  p4.ins_last_h = 0;
  p4.ins_w = 1;
  p4.ins_last_w = 0;
  p4.pad_top = 1;
  p4.pad_bottom = 1;
  p4.pad_left = 1;
  p4.pad_right = 1;
  p4.stride_h = 1;
  p4.stride_w = 1;
  p4.dilation_h = 1;
  p4.dilation_w = 1;
  p4.relu_enable = 0;
  p4.layer_id = layer_id;
  p4.ins_val = 0;                                // symmetric quantization
  p4.ins_fp = CV18xx::convert_fp32_to_bf16(0.0); // symmetric quantization
  CV18xx::tiu_pt_depthwise_convolution(&p4);

  // => tl_r
  cvk_tiu_copy_param_t p5 = {0};
  p5.src = &tl_y;
  p5.dst = &tl_r;
  p5.layer_id = layer_id;
  CV18xx::tiu_copy(&p5);
  cvk_tiu_mac_param_t p6 = {0};
  p6.res_high = nullptr;
  p6.res_low = &tl_r;
  p6.res_is_int8 = 0;
  p6.a = &tl_4v;
  p6.b_const.val = CV18xx::convert_fp32_to_bf16(1.596f);
  p6.b_is_const = 1;
  p6.b_const.is_signed = 1;
  p6.lshift_bits = 0;
  p6.rshift_bits = 0;
  p6.relu_enable = 1;
  p6.layer_id = layer_id;
  CV18xx::tiu_mac(&p6);
  cvk_tiu_min_param_t pmin = {0};
  pmin.min = &tl_r;
  pmin.a = &tl_r;
  pmin.b_is_const = 1;
  pmin.b_const.val = CV18xx::convert_fp32_to_bf16(255.0f);
  pmin.b_const.is_signed = 1;
  pmin.layer_id = layer_id;
  CV18xx::tiu_min(&pmin);

  // => tl_b
  cvk_tiu_copy_param_t p7 = {0};
  p7.src = &tl_y;
  p7.dst = &tl_b;
  p7.layer_id = layer_id;
  CV18xx::tiu_copy(&p7);
  cvk_tiu_mac_param_t p8 = {0};
  p8.res_high = nullptr;
  p8.res_low = &tl_b;
  p8.res_is_int8 = 0;
  p8.a = &tl_4u;
  p8.b_const.val = CV18xx::convert_fp32_to_bf16(2.018f);
  p8.b_is_const = 1;
  p8.b_const.is_signed = 1;
  p8.lshift_bits = 0;
  p8.rshift_bits = 0;
  p8.relu_enable = 1;
  p8.layer_id = layer_id;
  CV18xx::tiu_mac(&p8);
  pmin.min = &tl_b;
  pmin.a = &tl_b;
  pmin.b_is_const = 1;
  pmin.b_const.val = CV18xx::convert_fp32_to_bf16(255.0f);
  pmin.b_const.is_signed = 1;
  pmin.layer_id = layer_id;
  CV18xx::tiu_min(&pmin);

  // => tl_g
  cvk_tiu_copy_param_t p9 = {0};
  p9.src = &tl_y;
  p9.dst = &tl_g;
  p9.layer_id = layer_id;
  CV18xx::tiu_copy(&p9);
  cvk_tiu_mac_param_t p10 = {0};
  p10.res_high = nullptr;
  p10.res_low = &tl_g;
  p10.res_is_int8 = 0;
  p10.a = &tl_4v;
  p10.b_const.val = CV18xx::convert_fp32_to_bf16(-0.813f);
  p10.b_is_const = 1;
  p10.b_const.is_signed = 1;
  p10.lshift_bits = 0;
  p10.rshift_bits = 0;
  p10.relu_enable = 0;
  p10.layer_id = layer_id;
  CV18xx::tiu_mac(&p10);
  cvk_tiu_mac_param_t p11 = {0};
  p11.res_high = nullptr;
  p11.res_low = &tl_g;
  p11.res_is_int8 = 0;
  p11.a = &tl_4u;
  p11.b_const.val = CV18xx::convert_fp32_to_bf16(-0.391f);
  p11.b_is_const = 1;
  p11.b_const.is_signed = 1;
  p11.lshift_bits = 0;
  p11.rshift_bits = 0;
  p11.relu_enable = 1;
  p11.layer_id = layer_id;
  CV18xx::tiu_mac(&p11);
  pmin.min = &tl_g;
  pmin.a = &tl_g;
  pmin.b_is_const = 1;
  pmin.b_const.val = CV18xx::convert_fp32_to_bf16(255.0f);
  pmin.b_const.is_signed = 1;
  pmin.layer_id = layer_id;
  CV18xx::tiu_min(&pmin);
}

void TgYuv420Kernel::schedule() {
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

void cvi_backend_tg_yuv420_csc_kernel(uint32_t layer_id, gaddr_t ga_input,
                                      gaddr_t ga_output, int n, int c, int h,
                                      int w, const std::vector<int> &order,
                                      cvk_fmt_t fmt, int32_t pixel_type,
                                      int32_t y_align, int32_t w_align,
                                      int32_t channel_align) {
  TgYuv420Kernel kernel;
  // yuv channel align is 4KB, w_align is 32B
  kernel.init(layer_id, ga_input, ga_output, n, c, h, w, order, pixel_type,
              channel_align, y_align, w_align, fmt);
  kernel.selectTilePolicy();
  kernel.schedule();
}
} // namespace backend
} // namespace tpu_mlir
