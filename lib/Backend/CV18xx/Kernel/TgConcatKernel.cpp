//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/Kernel/TgConcatKernel.hpp"

#define DEBUG_TYPE "cvi_backend_concat_kernel"

namespace tpu_mlir {
namespace backend {
uint32_t &TgConcatKernel::axis_dim(cvk_tg_shape_t &shape) {
  switch (axis) {
  case 0:
    return shape.n;
  case 1:
    return shape.c;
  case 2:
    return shape.h;
  case 3:
    return shape.w;
  default:
    assert(0);
    return shape.w;
  }
}

void TgConcatKernel::update_output(int output_dim[], int dim_size,
                                   int concat_axis) {
  axis = concat_axis;
  axis_before =
      std::accumulate(output_dim, output_dim + axis, 1, std::multiplies<int>());
  axis_after = std::accumulate(output_dim + axis + 1, output_dim + dim_size, 1,
                               std::multiplies<int>());
  int axis_dim = output_dim[axis];
  int h, w;
  bool ret = CV18xx::size_to_hw(axis_after, h, w);
  assert(ret && "axis after size too large");
  if (axis_before == 1) {
    tiling_mode = CV18xx::TilingAll;
    output_shape = CV18xx::tg_shape_t4(axis_before, axis_dim, h, w);
    axis = 1;
  } else if (h == 1 && (axis_dim < axis_before || w == 1)) {
    output_shape = CV18xx::tg_shape_t4(1, axis_before, axis_dim, w);
    axis = 2;
  } else {
    output_shape = CV18xx::tg_shape_t4(axis_before, axis_dim, h, w);
    axis = 1;
  }

  output_stride = CV18xx::tg_default_stride(output_shape, fmt);
}

void TgConcatKernel::init(uint32_t layer_id, int input_num, int dim_size,
                          int concat_axis, gaddr_t input_gaddrs[],
                          gaddr_t output_gaddr, int axis_dims[],
                          int output_dim[], bool do_relu,
                          const int right_shift_width[],
                          const int threshold_x_quantized[], cvk_fmt_t fmt) {
  CV18xx::assert_support_fmt(fmt);
  assert(dim_size >= 2);
  assert(concat_axis < dim_size);
  this->layer_id = layer_id;
  this->fmt = fmt;
  this->do_relu = do_relu;
  this->input_num = input_num;
  this->tiling_mode = CV18xx::TilingNCHW;

  update_output(output_dim, dim_size, concat_axis);
  uint64_t axis_addr_offset = 0;
  do_parallel = false;
  for (int i = 0; i < input_num; i++) {
    input_info_t info;
    memset((void *)&info, 0, sizeof(input_info_t));
    if (right_shift_width != nullptr) {
      info.rshift_width = right_shift_width[i];
    }
    if (threshold_x_quantized != nullptr && threshold_x_quantized[i] != 0) {
      info.data_quantized = threshold_x_quantized[i];
    } else {
      info.data_quantized = 1;
    }
    info.shape = output_shape;
    axis_dim(info.shape) = axis_dims[i];
    info.stride = CV18xx::tg_default_stride(info.shape, fmt);
    info.do_quantize = true;
    if (info.rshift_width == 0 && info.data_quantized == 1 &&
        do_relu == false) {
      info.do_quantize = false;
    }
    if (info.do_quantize && false == do_parallel) {
      do_parallel = true;
    }
    info.ga_input = input_gaddrs[i];
    info.ga_output = output_gaddr + axis_addr_offset;
    axis_addr_offset +=
        CV18xx::bytesize_of_fmt(fmt) * axis_dim(info.shape) * axis_after;
    inputs.emplace_back(info);
  }
}

uint64_t TgConcatKernel::dst_offset(const CV18xx::tiling_info_t &tile) const {
  return tile.pos_w * output_stride.w + tile.pos_h * output_stride.h +
         tile.pos_c * output_stride.c + tile.pos_n * output_stride.n;
}

void TgConcatKernel::selectTilePolicy() {
  total_tiles = 0;
  if (do_parallel) {
    for (auto &input : inputs) {
      input.tile_idx = total_tiles;
      CV18xx::tiling_packing(input.tiles, input.shape, fmt, 4, 0, tiling_mode);
      total_tiles += input.tiles.size();
    }
    // half the lmem
    int lsize = ALIGN(CV18xx::LMEM_BYTES / 4, CV18xx::EU_BYTES);
    base_addr[0] = 0;
    base_addr[1] = lsize;
    base_addr[2] = base_addr[1] + lsize;
    base_addr[3] = base_addr[2] + lsize;
  } else {
    for (auto &input : inputs) {
      input.tile_idx = total_tiles;
      CV18xx::tiling_packing(input.tiles, input.shape, fmt, 1, 0, tiling_mode);
      total_tiles += input.tiles.size();
    }
    memset(base_addr, 0, sizeof(base_addr));
  }
}

void TgConcatKernel::prepare(int32_t step_idx,
                             TgConcatKernel::input_info_t *&input,
                             CV18xx::tiling_info_t *&tile) {
  for (int idx = input_num - 1; idx >= 0; idx--) {
    input = &inputs[idx];
    if (input->tile_idx <= step_idx) {
      tile = &input->tiles[step_idx - input->tile_idx];
      CV18xx::lmem_init_tensor(
          &tl_input, CV18xx::tl_shape_t4(tile->n, tile->c, tile->h, tile->w),
          fmt, 1);
      CV18xx::lmem_init_tensor(
          &tl_output, CV18xx::tl_shape_t4(tile->n, tile->c, tile->h, tile->w),
          fmt, 1);
      tl_input.start_address = base_addr[step_idx % 2];
      tl_output.start_address = base_addr[step_idx % 2 + 2];
      return;
    }
  }
  assert(0 && "tile incorrect");
}

void TgConcatKernel::load(int32_t step_idx) {
  TgConcatKernel::input_info_t *input;
  CV18xx::tiling_info_t *tile;
  prepare(step_idx, input, tile);
  if (tiling_mode == CV18xx::TilingNCHW) {
    CV18xx::tdma_load_stride(&tl_input, input->ga_input + tile->offset,
                             input->stride);
  } else {
    CV18xx::tdma_load(&tl_input, input->ga_input + tile->offset);
  }
}

void TgConcatKernel::store(int32_t step_idx) {
  TgConcatKernel::input_info_t *input;
  CV18xx::tiling_info_t *tile;
  prepare(step_idx, input, tile);
  if (tiling_mode == CV18xx::TilingNCHW) {
    CV18xx::tdma_store_stride(&tl_output, input->ga_output + dst_offset(*tile),
                              output_stride);
  } else {
    CV18xx::tdma_store(&tl_output, input->ga_output + tile->offset);
  }
}

void TgConcatKernel::compute(int32_t step_idx) {
  TgConcatKernel::input_info_t *input;
  CV18xx::tiling_info_t *tile;
  prepare(step_idx, input, tile);
  // do quantize
  if (input->do_quantize) {
    cvk_tiu_mul_param_t p = {0};
    p.res_high = nullptr;
    p.res_low = &tl_output;
    p.a = &tl_input;
    if (fmt == CVK_FMT_BF16) {
      // bf16 no quant now
      p.b_const.val = CV18xx::convert_fp32_to_bf16(1.0);
      p.rshift_bits = 0;
    } else {
      p.b_const.val = static_cast<int16_t>(input->data_quantized);
      p.rshift_bits = static_cast<uint8_t>(input->rshift_width);
    }
    p.b_const.is_signed = false;
    p.b_is_const = 1;
    p.layer_id = layer_id;
    p.relu_enable = do_relu ? 1 : 0;
    CV18xx::tiu_mul(&p);
  } else {
    cvk_tiu_copy_param_t p;
    p.layer_id = layer_id;
    p.src = &tl_input;
    p.dst = &tl_output;
    CV18xx::tiu_copy(&p);
  }
}

void TgConcatKernel::schedule() {
  if (do_parallel) {
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
  } else {
    for (int step = 0; step < total_tiles; step++) {
      load(step);
      store(step);
    }
  }
}

void cvi_backend_tg_concat_kernel(uint32_t layer_id, int input_num,
                                  gaddr_t input_gaddrs[], gaddr_t output_gaddr,
                                  int axis_dims[], int concat_axis,
                                  int output_dim_size, int output_dim[],
                                  bool do_relu, const int *right_shift_width,
                                  const int *threshold_x_quantized,
                                  cvk_fmt_t fmt) {
  TgConcatKernel kernel;
  kernel.init(layer_id, input_num, output_dim_size, concat_axis, input_gaddrs,
              output_gaddr, axis_dims, output_dim, do_relu, right_shift_width,
              threshold_x_quantized, fmt);
  kernel.selectTilePolicy();
  kernel.schedule();
}
} // namespace backend
} // namespace tpu_mlir
