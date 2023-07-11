//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/Kernel/TgPoolMaskKernel.hpp"
#include "tpu_mlir/Support/MathUtils.h"

#define DEBUG_TYPE "TgPoolMaskKernel"

namespace tpu_mlir {
namespace backend {
void TgPoolMaskKernel::init(uint32_t layer_id, gaddr_t input_gaddr,
                            gaddr_t output_gaddr, int n, int c, int h, int w,
                            int scale, cvk_fmt_t fmt) {
  CV18xx::assert_support_fmt(fmt);
  this->layer_id = layer_id;
  this->fmt = fmt;
  this->n = 1;
  this->c = n * c;
  this->h = h;
  this->w = w;
  this->scale = scale;
  this->h_ex = align_up(h, scale);
  this->w_ex = align_up(w, scale);
  this->ga_input = input_gaddr;
  this->ga_output = output_gaddr;
  this->src_stride = CV18xx::tg_default_stride(this->c, this->h, this->w, fmt);
  this->dst_stride =
      CV18xx::tg_default_stride(this->c, this->h_ex, this->w_ex, fmt);
}

void TgPoolMaskKernel::selectTilePolicy() {
  uint32_t lmem_required = 0;
  step_c = CV18xx::NPU_NUM;
  step_n = 1;
  blob_num = (fmt == CVK_FMT_BF16 ? 4 : 6);
  for (step_w = w_ex; step_w > 0; step_w -= scale) {
    for (step_h = h_ex; step_h > 0; step_h -= scale) {
      for (step_c = c; step_c > 0;) {
        // input, upsample
        uint32_t input_required =
            CV18xx::lmem_tensor_to_size(step_n, step_c, step_h, step_w, fmt, 1);
        uint32_t pool_required = CV18xx::lmem_tensor_to_size(
            step_n, step_c, step_h / scale, step_w / scale, fmt, 1);
        uint32_t kernel_required =
            CV18xx::lmem_tensor_to_size(1, step_c, scale, scale, fmt, 1);
        // two input
        lmem_required =
            blob_num * input_required + pool_required + kernel_required;
        if (lmem_required <= (uint32_t)CV18xx::LMEM_BYTES) {
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
after_loop:
  if (lmem_required > (uint32_t)CV18xx::LMEM_BYTES) {
    llvm::errs() << llvm::format(
        "Tilling failed, src shape:(%d,%d,%d,%d), scale:%d, fmt:%d\n", n, c, h,
        w, scale, fmt);
    assert(0);
  }
  TileInfo tile = {0};
  for (tile.pos_n = 0; tile.pos_n < n; tile.pos_n += step_n) {
    tile.n = std::min(n - tile.pos_n, step_n);
    for (tile.pos_c = 0; tile.pos_c < c; tile.pos_c += step_c) {
      tile.c = std::min(c - tile.pos_c, step_c);
      for (tile.pos_h = 0; tile.pos_h < h_ex; tile.pos_h += step_h) {
        tile.h = std::min(h_ex - tile.pos_h, step_h);
        tile.pad_h = std::max(tile.h + tile.pos_h - h, 0);
        for (tile.pos_w = 0; tile.pos_w < w_ex; tile.pos_w += step_w) {
          tile.w = std::min(w_ex - tile.pos_w, step_w);
          tile.pad_w = std::max(tile.w + tile.pos_w - w, 0);
          tile.src_offset =
              tile.pos_w * src_stride.w + tile.pos_h * src_stride.h +
              tile.pos_c * src_stride.c + tile.pos_n * src_stride.n;
          tile.dst_offset =
              tile.pos_w * dst_stride.w + tile.pos_h * dst_stride.h +
              tile.pos_c * dst_stride.c + tile.pos_n * dst_stride.n;
          tiles.emplace_back(tile);
        }
      }
    }
  }
}

void TgPoolMaskKernel::allocLmem() {
  for (int i = 0; i < blob_num; i++) {
    tl_mem[i] = CV18xx::lmem_alloc_tensor(
        CV18xx::tl_shape_t4(step_n, step_c, step_h, step_w), fmt, 1);
  }
  tl_pooling = CV18xx::lmem_alloc_tensor(
      CV18xx::tl_shape_t4(step_n, step_c, step_h / scale, step_w / scale), fmt,
      1);
  kernel_shape = CV18xx::tl_shape_t4(1, step_c, scale, scale);
  tl_kernel = CV18xx::lmem_alloc_tensor(kernel_shape, fmt, 1);
  // load kernel
  cvk_tdma_g2l_tensor_fill_constant_param_t p2 = {0};
  if (fmt == CVK_FMT_BF16) {
    p2.constant = CV18xx::convert_fp32_to_bf16(1.0f);
  } else {
    p2.constant = 1;
  }
  p2.layer_id = layer_id;
  p2.dst = tl_kernel;
  CV18xx::tdma_g2l_tensor_fill_constant(&p2);
}

void TgPoolMaskKernel::deallocLmem() {
  CV18xx::lmem_free_tensor(tl_kernel);
  CV18xx::lmem_free_tensor(tl_pooling);
  for (int i = blob_num - 1; i >= 0; i--) {
    CV18xx::lmem_free_tensor(tl_mem[i]);
  }
}

void TgPoolMaskKernel::refresh(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  tl_input = *tl_mem[step_idx % 2 + 2];
  tl_input.shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  tl_input.stride = CV18xx::tl_default_stride(tl_input.shape, fmt, 1);
  tl_output = tl_input;
  tl_output.start_address = tl_mem[step_idx % 2]->start_address;
  if (fmt != CVK_FMT_BF16) {
    tl_low = tl_input;
    tl_high = tl_input;
    tl_low.start_address = tl_mem[4]->start_address;
    tl_high.start_address = tl_mem[5]->start_address;
  }

  tl_pooling->shape =
      CV18xx::tl_shape_t4(tile.n, tile.c, tile.h / scale, tile.w / scale);
  tl_pooling->stride = CV18xx::tl_default_stride(tl_pooling->shape, fmt, 1);
  if (tile.c != step_c) {
    tl_kernel->shape.c = tile.c;
    tl_kernel->stride = CV18xx::tl_default_stride(tl_kernel->shape, fmt, 1);
  }
}

void TgPoolMaskKernel::load(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  if (tile.pad_h != 0 || tile.pad_w != 0) {
    cvk_tdma_g2l_tensor_fill_constant_param_t p = {0};
    p.dst = &tl_input;
    p.layer_id = layer_id;
    if (fmt == CVK_FMT_I8) {
      p.constant = static_cast<uint16_t>(-128);
    } else {
      p.constant = 0xFF7F;
    }
    CV18xx::tdma_g2l_tensor_fill_constant(&p);
    tl_input.shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h - tile.pad_h,
                                         tile.w - tile.pad_w);
    CV18xx::tdma_load_stride(&tl_input, ga_input + tile.src_offset, src_stride);
    tl_input.shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  } else {
    CV18xx::tdma_load_stride(&tl_input, ga_input + tile.src_offset, src_stride);
  }
}

void TgPoolMaskKernel::store(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  refresh(step_idx);
  CV18xx::tdma_store_stride(&tl_output, ga_output + tile.dst_offset,
                            dst_stride);
}

void TgPoolMaskKernel::compute_bf16() {
  // pooling max first
  cvk_tiu_max_pooling_param_t p1 = {0};
  p1.ofmap = tl_pooling;
  p1.ifmap = &tl_input;
  p1.kh = scale;
  p1.kw = scale;
  p1.stride_h = scale;
  p1.stride_w = scale;
  p1.layer_id = layer_id;
  p1.ins_val = -128;
  p1.ins_fp = 0xff7f;
  CV18xx::tiu_max_pooling(&p1);
  // upsample then
  cvk_tiu_depthwise_pt_convolution_param_t p2 = {0};
  p2.ofmap = &tl_output;
  p2.ifmap = tl_pooling;
  p2.weight = tl_kernel;
  p2.bias = nullptr;
  p2.ins_h = scale - 1;
  p2.ins_last_h = 0;
  p2.ins_w = scale - 1;
  p2.ins_last_w = 0;
  p2.pad_top = scale - 1;
  p2.pad_bottom = scale - 1;
  p2.pad_left = scale - 1;
  p2.pad_right = scale - 1;
  p2.stride_h = 1;
  p2.stride_w = 1;
  p2.dilation_h = 1;
  p2.dilation_w = 1;
  p2.relu_enable = 0;
  p2.layer_id = layer_id;
  p2.ins_val = 0;                                // symmetric quantization
  p2.ins_fp = CV18xx::convert_fp32_to_bf16(0.0); // symmetric quantization
  CV18xx::tiu_pt_depthwise_convolution(&p2);

  cvk_tiu_sub_param_t p3 = {0};
  p3.res_high = 0;
  p3.res_low = &tl_input;
  p3.a_high = 0;
  p3.a_low = &tl_input;
  p3.b_high = 0;
  p3.b_low = &tl_output;
  p3.rshift_bits = 0;
  p3.layer_id = layer_id;
  CV18xx::tiu_sub(&p3);
  cvk_tiu_add_param_t p4 = {0};
  p4.res_high = nullptr;
  p4.res_low = &tl_output;
  p4.a_high = nullptr;
  p4.a_low = &tl_input;
  p4.b_is_const = true;
  p4.b_const.val = CV18xx::convert_fp32_to_bf16(0.000005f);
  p4.rshift_bits = 0;
  p4.layer_id = layer_id;
  p4.relu_enable = 1;
  CV18xx::tiu_add(&p4);
  cvk_tiu_mul_param_t p5 = {0};
  p5.res_high = nullptr;
  p5.res_low = &tl_output;
  p5.a = &tl_output; // rt
  p5.b_is_const = 1;
  p5.b_const.val = CV18xx::convert_fp32_to_bf16(200000.0f);
  p5.rshift_bits = 0;
  p5.layer_id = layer_id;
  p5.relu_enable = 0;
  CV18xx::tiu_mul(&p5);

  // mul index
  cvk_tl_t tl_index_src = tl_output;
  tl_index_src.shape.w /= scale;
  tl_index_src.shape.h /= scale;
  tl_index_src.stride.w *= scale;
  tl_index_src.stride.h *= scale;
  cvk_tl_t tl_index_dst = tl_index_src;
  for (int ih = 0; ih < scale; ih++) {
    for (int iw = 0; iw < scale; iw++) {
      uint32_t offset = ih * tl_output.stride.h + iw * tl_output.stride.w;
      float val = scale * scale - ih * scale - iw;
      tl_index_src.start_address = tl_output.start_address + offset;
      tl_index_dst.start_address = tl_input.start_address + offset;
      cvk_tiu_mul_param_t p6 = {0};
      p6.res_high = nullptr;
      p6.res_low = &tl_index_dst;
      p6.a = &tl_index_src;
      p6.b_is_const = 1;
      p6.b_const.val = CV18xx::convert_fp32_to_bf16(val);
      p6.rshift_bits = 0;
      p6.layer_id = layer_id;
      p6.relu_enable = 0;
      CV18xx::tiu_mul(&p6);
    }
  }
  // again
  CV18xx::tiu_max_pooling(&p1);
  CV18xx::tiu_pt_depthwise_convolution(&p2);
  CV18xx::tiu_sub(&p3);
  p4.b_const.val = CV18xx::convert_fp32_to_bf16(1.0f);
  CV18xx::tiu_add(&p4);
}

void TgPoolMaskKernel::compute_int8() {
  // pooling max first
  cvk_tiu_max_pooling_param_t p1 = {0};
  p1.ofmap = tl_pooling;
  p1.ifmap = &tl_input;
  p1.kh = scale;
  p1.kw = scale;
  p1.stride_h = scale;
  p1.stride_w = scale;
  p1.layer_id = layer_id;
  p1.ins_val = -128;
  p1.ins_fp = 0xff7f;
  CV18xx::tiu_max_pooling(&p1);
  // upsample then
  cvk_tiu_depthwise_pt_convolution_param_t p2 = {0};
  p2.ofmap = &tl_output;
  p2.ifmap = tl_pooling;
  p2.weight = tl_kernel;
  p2.bias = nullptr;
  p2.ins_h = scale - 1;
  p2.ins_last_h = 0;
  p2.ins_w = scale - 1;
  p2.ins_last_w = 0;
  p2.pad_top = scale - 1;
  p2.pad_bottom = scale - 1;
  p2.pad_left = scale - 1;
  p2.pad_right = scale - 1;
  p2.stride_h = 1;
  p2.stride_w = 1;
  p2.dilation_h = 1;
  p2.dilation_w = 1;
  p2.relu_enable = 0;
  p2.layer_id = layer_id;
  p2.ins_val = 0;                                // symmetric quantization
  p2.ins_fp = CV18xx::convert_fp32_to_bf16(0.0); // symmetric quantization
  CV18xx::tiu_pt_depthwise_convolution(&p2);

  CV18xx::tiu_zeros(layer_id, &tl_low);
  CV18xx::tiu_zeros(layer_id, &tl_high);
  cvk_tiu_mac_param_t p3 = {0};
  p3.res_high = &tl_high;
  p3.res_low = &tl_low;
  p3.res_is_int8 = 0;
  p3.a = &tl_output;
  p3.b_const.val = -1;
  p3.b_is_const = 1;
  p3.b_const.is_signed = 1;
  p3.lshift_bits = 0;
  p3.rshift_bits = 0;
  p3.layer_id = layer_id;
  p3.relu_enable = 0;
  CV18xx::tiu_mac(&p3);

  cvk_tiu_mac_param_t p4 = {0};
  p4.res_high = &tl_high;
  p4.res_low = &tl_low;
  p4.res_is_int8 = 0;
  p4.a = &tl_input;
  p4.b_const.val = 1;
  p4.b_is_const = 1;
  p4.b_const.is_signed = 1;
  p4.lshift_bits = 0;
  p4.rshift_bits = 0;
  p4.layer_id = layer_id;
  p4.relu_enable = 0;
  CV18xx::tiu_mac(&p4);

  cvk_tiu_add_param_t p5 = {0};
  p5.res_high = nullptr;
  p5.res_low = &tl_output;
  p5.a_high = &tl_high;
  p5.a_low = &tl_low;
  p5.b_is_const = true;
  p5.b_const.val = 1;
  p5.b_const.is_signed = 1;
  p5.rshift_bits = 0;
  p5.layer_id = layer_id;
  p5.relu_enable = 1;
  CV18xx::tiu_add(&p5);

  // mul index
  cvk_tl_t tl_index_src = tl_output;
  tl_index_src.shape.w /= scale;
  tl_index_src.shape.h /= scale;
  tl_index_src.stride.w *= scale;
  tl_index_src.stride.h *= scale;
  cvk_tl_t tl_index_dst = tl_index_src;
  for (int ih = 0; ih < scale; ih++) {
    for (int iw = 0; iw < scale; iw++) {
      uint32_t offset = ih * tl_output.stride.h + iw * tl_output.stride.w;
      int val = scale * scale - ih * scale - iw;
      tl_index_src.start_address = tl_output.start_address + offset;
      tl_index_dst.start_address = tl_input.start_address + offset;
      cvk_tiu_mul_param_t p6 = {0};
      p6.res_high = nullptr;
      p6.res_low = &tl_index_dst;
      p6.a = &tl_index_src;
      p6.b_is_const = 1;
      p6.b_const.val = val;
      p6.b_const.is_signed = 1;
      p6.rshift_bits = 0;
      p6.layer_id = layer_id;
      p6.relu_enable = 0;
      CV18xx::tiu_mul(&p6);
    }
  }
  // again
  CV18xx::tiu_max_pooling(&p1);
  CV18xx::tiu_pt_depthwise_convolution(&p2);
  CV18xx::tiu_zeros(layer_id, &tl_low);
  CV18xx::tiu_zeros(layer_id, &tl_high);
  CV18xx::tiu_mac(&p3);
  CV18xx::tiu_mac(&p4);
  CV18xx::tiu_add(&p5);
}

void TgPoolMaskKernel::compute(int32_t step_idx) {
  refresh(step_idx);
  if (fmt == CVK_FMT_BF16) {
    compute_bf16();
  } else {
    compute_int8();
  }
}

void TgPoolMaskKernel::schedule() {
  allocLmem();
  int total_tiles = tiles.size();
  for (int step = 0; step < total_tiles + 2; step++) {
    CV18xx::parallel_enable();
    if (step > 0 && step - 1 < total_tiles) {
      compute(step - 1);
    }
    if (step < total_tiles) {
      load(step);
    }
    if (step > 1) {
      store(step - 2);
    }
    CV18xx::parallel_disable();
  }
  deallocLmem();
}

void cvi_backend_tg_pool_mask_kernel(uint32_t layer_id, gaddr_t input_gaddr,
                                     gaddr_t output_gaddr, int n, int c, int h,
                                     int w, int scale, cvk_fmt_t fmt) {
  TgPoolMaskKernel kernel;
  kernel.init(layer_id, input_gaddr, output_gaddr, n, c, h, w, scale, fmt);
  kernel.selectTilePolicy();
  kernel.schedule();
}
} // namespace backend
} // namespace tpu_mlir
