//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/Kernel/TgFixedPoolingKernel.hpp"
#include "tpu_mlir/Support/MathUtils.h"

#define DEBUG_TYPE "TgFixedPoolingKernel"

namespace tpu_mlir {
namespace backend {
void TgInt8PoolingKernel::adjustPadding() {
  if ((oh - 1) * stride_h > h + pad_t + pad_b - kh) {
    pad_b = (oh - 1) * stride_h - (h + pad_t - kh);
  }
  if ((ow - 1) * stride_w > w + pad_l + pad_r - kw) {
    pad_r = (ow - 1) * stride_w - (w + pad_l - kw);
  }
}

void TgInt8PoolingKernel::init(uint32_t layer_id, gaddr_t ga_input,
                               gaddr_t ga_output, int32_t n, int32_t c,
                               int32_t h, int32_t w, int32_t pad_t,
                               int32_t pad_b, int32_t pad_l, int32_t pad_r,
                               int32_t kh, int32_t kw, int32_t stride_h,
                               int32_t stride_w, bool is_avg_pooling,
                               bool do_relu, int32_t rshift, int32_t multiplier,
                               bool ceil_mode) {

  this->layer_id = layer_id;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->n = 1; // fuse n and c
  this->c = n * c;
  this->h = h;
  this->w = w;
  this->oh = (h + pad_t + pad_b - kh) / stride_h + 1;
  this->ow = (w + pad_l + pad_r - kw) / stride_w + 1;
  this->pad_t = pad_t;
  this->pad_b = pad_b;
  this->pad_l = pad_l;
  this->pad_r = pad_r;
  this->kh = kh;
  this->kw = kw;
  this->stride_h = stride_h;
  this->stride_w = stride_w;
  this->is_avg_pooling = is_avg_pooling;
  this->do_relu = do_relu;
  assert(!do_relu); // TODO
  this->rshift = rshift;
  this->multiplier = multiplier;
  if (ceil_mode) {
    adjustPadding();
  }
}

void TgInt8PoolingKernel::allocLmem(cvk_tl_shape_t &input_shape,
                                    cvk_tl_shape_t &output_shape) {
  tl_input[0] = CV18xx::lmem_alloc_tensor(input_shape, CVK_FMT_I8, 1);
  tl_input[1] = CV18xx::lmem_alloc_tensor(input_shape, CVK_FMT_I8, 1);
  tl_output[0] = CV18xx::lmem_alloc_tensor(output_shape, CVK_FMT_I8, 1);
  tl_output[1] = CV18xx::lmem_alloc_tensor(output_shape, CVK_FMT_I8, 1);
  assert(tl_input[0] && tl_input[1]);
  assert(tl_output[0] && tl_output[1]);
}

void TgInt8PoolingKernel::deallocLmem() {
  CV18xx::lmem_free_tensor(tl_output[1]);
  CV18xx::lmem_free_tensor(tl_output[0]);
  CV18xx::lmem_free_tensor(tl_input[1]);
  CV18xx::lmem_free_tensor(tl_input[0]);
}

void TgInt8PoolingKernel::selectTilePolicy() { doTileForNormalCase(); }

void TgInt8PoolingKernel::doTileForNormalCase() {
  int32_t step_c, step_oh, step_ow;

  // determin the shape of tile.
  for (step_ow = stride_w > 15 ? 1 : ow; step_ow > 0; step_ow--) {
    for (step_oh = stride_h > 15 ? 1 : oh; step_oh > 0; step_oh--) {
      for (step_c = std::min(c, MAX_CHANNEL); step_c > 0;
           step_c -= CV18xx::NPU_NUM) {
        if (step_c != c) {
          step_c = align_up(step_c, CV18xx::NPU_NUM);
        }
        auto step_ih = (step_oh - 1) * stride_h + kh;
        auto step_iw = (step_ow - 1) * stride_w + kw;
        if (step_ih > h) {
          step_ih = h;
        }
        if (step_iw > w) {
          step_iw = w;
        }
        if (step_iw > MAX_WIDTH || step_ih > MAX_HEIGHT) {
          continue;
        }
        cvk_tl_shape_t input_shape =
            CV18xx::tl_shape_t4(1, step_c, step_ih, step_iw);
        cvk_tl_shape_t output_shape =
            CV18xx::tl_shape_t4(1, step_c, step_oh, step_ow);
        auto total_lmem =
            2 * (CV18xx::lmem_tensor_to_size(input_shape, CVK_FMT_I8, 1) +
                 CV18xx::lmem_tensor_to_size(output_shape, CVK_FMT_I8, 1));
        LLVM_DEBUG(llvm::errs()
                   << llvm::format("try input shape:%d,%d,%d,%d output "
                                   "shape:%d,%d,%d,%d total lmem:%d\n",
                                   1, step_c, step_ih, step_iw, 1, step_c,
                                   step_oh, step_ow, total_lmem));
        if (total_lmem <= (uint32_t)CV18xx::LMEM_BYTES) {
          LLVM_DEBUG(
              llvm::errs() << llvm::format(
                  "input_shape(%d,%d,%d,%d), output_shape(%d,%d,%d,%d)\n"
                  "kh:%d, kw:%d, sh:%d, sw:%d, pad(%d,%d,%d,%d), "
                  "avg_pooling:%d, do_relu:%d\n"
                  "step_c:%d, step_ih:%d, step_iw:%d, step_oh:%d, step_ow:%d\n",
                  n, c, h, w, n, c, oh, ow, kh, kw, stride_h, stride_w, pad_t,
                  pad_b, pad_l, pad_r, is_avg_pooling, do_relu, step_c, step_ih,
                  step_iw, step_oh, step_ow));

          allocLmem(input_shape, output_shape);
          goto do_tile;
        }
      }
    }
  }
  assert(0 && "failed to split");

do_tile:

  PoolingTile tile;
  for (int c_pos = 0; c_pos < c; c_pos += step_c) {
    auto cur_c = std::min(c - c_pos, step_c);
    for (int oh_pos = 0; oh_pos < oh; oh_pos += step_oh) {
      int32_t cur_oh = std::min(oh - oh_pos, step_oh);
      int32_t oh_top = oh_pos;
      int32_t oh_bot = oh_pos + cur_oh;
      int32_t ih_top = std::max(oh_top * stride_h - pad_t, 0);
      int32_t ih_bot = std::min((oh_bot - 1) * stride_h + kh - pad_t, h);
      int32_t cur_ih = ih_bot - ih_top;
      int32_t cur_pad_t = (ih_top == 0) ? (pad_t - oh_top * stride_h) : 0;
      int32_t cur_pad_b =
          (ih_bot == h) ? ((oh_bot - 1) * stride_h + kh - pad_t - h) : 0;

      for (int ow_pos = 0; ow_pos < ow; ow_pos += step_ow) {
        int32_t cur_ow = std::min(ow - ow_pos, step_ow);
        int32_t ow_left = ow_pos;
        int32_t ow_right = ow_pos + cur_ow;
        int32_t iw_left = std::max(ow_left * stride_w - pad_l, 0);
        int32_t iw_right = std::min((ow_right - 1) * stride_w + kw - pad_l, w);
        int32_t cur_iw = iw_right - iw_left;
        int32_t cur_pad_l = (iw_left == 0) ? (pad_l - ow_left * stride_w) : 0;
        int32_t cur_pad_r =
            (iw_right == w) ? ((ow_right - 1) * stride_w + kw - pad_l - w) : 0;
        tile.n = 1;
        tile.c = cur_c;
        tile.h = cur_ih;
        tile.w = cur_iw;
        tile.oh = cur_oh;
        tile.ow = cur_ow;
        tile.c_pos = c_pos;
        tile.ih_pos = ih_top;
        tile.oh_pos = oh_pos;
        tile.pad[0] = cur_pad_t;
        tile.pad[1] = cur_pad_b;
        tile.pad[2] = cur_pad_l;
        tile.pad[3] = cur_pad_r;
        tile.input_offset =
            (c_pos * h * w + ih_top * w + iw_left) * sizeof(int8_t);
        tile.output_offset =
            (c_pos * oh * ow + oh_pos * ow + ow_pos) * sizeof(int8_t);
        tiles.push_back(tile);
      }
    }
  }
}

void TgInt8PoolingKernel::schedule() {
  int32_t total_steps = tiles.size();
  for (int32_t i = 0; i < total_steps + 2; i++) {
    CV18xx::parallel_enable();

    if (i - 1 >= 0 && i - 1 < total_steps) {
      compute(i - 1, flip);
    }
    if (i < total_steps) {
      load(i, flip);
    }
    if (i - 2 >= 0) {
      store(i - 2, flip);
    }
    flip = 1 - flip;
    CV18xx::parallel_disable();

    LLVM_DEBUG(llvm::errs() << "########\n");
  }
  deallocLmem();
}

void TgInt8PoolingKernel::load(int32_t step_idx, int32_t flip) {
  cvk_tl_t operand;
  auto tile = tiles[step_idx];
  cvk_tl_shape_t shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  operand.start_address = tl_input[1 - flip]->start_address;
  operand.shape = shape;
  operand.stride = CV18xx::tl_default_stride(shape, CVK_FMT_I8, 1);
  operand.fmt = CVK_FMT_I8;
  cvk_tg_stride_t stride = {(uint32_t)(c * h * w), (uint32_t)(h * w),
                            (uint32_t)w, 1};
  CV18xx::tdma_load_stride(&operand, ga_input + tile.input_offset, stride);

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "load[%d], flip[%d], addr:%d, shape<%d,%d,%d,%d:%d,%d,%d,%d>, "
                 "offset:%d\n",
                 step_idx, 1 - flip, operand.start_address, shape.n, shape.c,
                 shape.h, shape.w, stride.n, stride.c, stride.h, stride.w,
                 tile.input_offset));
}

void TgInt8PoolingKernel::store(int32_t step_idx, int32_t flip) {
  cvk_tl_t result;
  auto tile = tiles[step_idx];
  cvk_tl_shape_t shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.oh, tile.ow);
  result.start_address = tl_output[1 - flip]->start_address;
  result.shape = shape;
  result.stride = CV18xx::tl_default_stride(shape, CVK_FMT_I8, 1);
  result.fmt = CVK_FMT_I8;
  cvk_tg_stride_t stride = {(uint32_t)(c * oh * ow), (uint32_t)(oh * ow),
                            (uint32_t)ow, 1};
  CV18xx::tdma_store_stride(&result, ga_output + tile.output_offset, stride);

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "store[%d], flip[%d], addr:%d, "
                 "shape<%d,%d,%d,%d:%d,%d,%d,%d>, offset:%d\n",
                 step_idx, 1 - flip, result.start_address, shape.n, shape.c,
                 shape.h, shape.w, stride.n, stride.c, stride.h, stride.w,
                 tile.output_offset));
}

void TgInt8PoolingKernel::compute(int32_t step_idx, int32_t flip) {
  auto tile = tiles[step_idx];
  cvk_tl_shape_t input_shape;
  cvk_tl_shape_t output_shape;
  cvk_tl_t input;
  cvk_tl_t output;

  input_shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  input.start_address = tl_input[flip]->start_address;
  input.shape = input_shape;
  input.fmt = CVK_FMT_I8;
  input.stride = CV18xx::tl_default_stride(input_shape, CVK_FMT_I8, 1);

  output_shape = CV18xx::tl_shape_t4(tile.n, tile.c, tile.oh, tile.ow);
  output.start_address = tl_output[flip]->start_address;
  output.shape = output_shape;
  output.fmt = CVK_FMT_I8;
  output.stride = CV18xx::tl_default_stride(output_shape, CVK_FMT_I8, 1);

  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "compute[%d], flip[%d, %d], addr:%d, %d,input<%d,%d,%d,%d>, "
                 "output<%d,%d,%d,%d>, pad<%d,%d,%d,%d>, "
                 "kh:%d, kw:%d, sh:%d, sw:%d\n",
                 step_idx, flip, flip, input.start_address,
                 output.start_address, input.shape.n, input.shape.c,
                 input.shape.h, input.shape.w, output.shape.n, output.shape.c,
                 output.shape.h, output.shape.w, tile.pad[0], tile.pad[1],
                 tile.pad[2], tile.pad[3], kh, kw, stride_h, stride_w));

  if ((tile.oh > 1 && stride_h > 15) || (tile.ow > 1 && stride_w > 15)) {
    // not support stride > 15
    llvm_unreachable("Not support now.");
  }
  // max pooling
  if (!is_avg_pooling) {
    cvk_tiu_max_pooling_param_t param = {0};
    param.ofmap = &output;
    param.ifmap = &input;
    param.kh = kh;
    param.kw = kw;
    param.pad_top = tile.pad[0];
    param.pad_bottom = tile.pad[1];
    param.pad_left = tile.pad[2];
    param.pad_right = tile.pad[3];
    param.stride_h = stride_h;
    param.stride_w = stride_w;
    param.layer_id = layer_id;
    param.ins_val = -128;
    param.ins_fp = 0xff7f;
    CV18xx::tiu_max_pooling(&param);
  } else {
    cvk_tiu_average_pooling_param_t param = {0};
    param.ofmap = &output;
    param.ifmap = &input;
    param.kh = kh;
    param.kw = kw;
    param.ins_h = 0;
    param.ins_last_h = 0;
    param.ins_w = 0;
    param.ins_last_w = 0;
    param.pad_top = tile.pad[0];
    param.pad_bottom = tile.pad[1];
    param.pad_left = tile.pad[2];
    param.pad_right = tile.pad[3];
    param.stride_h = stride_h;
    param.stride_w = stride_w;
    param.avg_pooling_const = multiplier;
    param.rshift_bits = rshift;
    param.layer_id = layer_id;
    param.ins_val = 0;
    param.ins_fp = CV18xx::convert_fp32_to_bf16(0.0);
    CV18xx::tiu_average_pooling(&param);
  }
}

void cvi_backend_tg_fixed_max_pooling_kernel(
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output, int n, int c, int h,
    int w, int kh, int kw, int pad_top, int pad_bot, int pad_left,
    int pad_right, int stride_h, int stride_w, bool do_relu, bool ceil_mode) {

  assert(!do_relu);
  TgInt8PoolingKernel kernel;
  kernel.init(layer_id, ga_input, ga_output, n, c, h, w, pad_top, pad_bot,
              pad_left, pad_right, kh, kw, stride_h, stride_w, false, do_relu,
              0, 1, ceil_mode);

  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_fixed_avg_pooling_kernel(
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output, int n, int c, int h,
    int w, int kh, int kw, int pad_top, int pad_bot, int pad_left,
    int pad_right, int stride_h, int stride_w, bool do_relu, int rshift,
    int multiplier, bool ceil_mode) {

  assert(!do_relu);
  TgInt8PoolingKernel kernel;
  kernel.init(layer_id, ga_input, ga_output, n, c, h, w, pad_top, pad_bot,
              pad_left, pad_right, kh, kw, stride_h, stride_w, true, do_relu,
              rshift, multiplier, ceil_mode);

  kernel.selectTilePolicy();
  kernel.schedule();
}
} // namespace backend
} // namespace tpu_mlir
