//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include <llvm/Support/Debug.h>

namespace tpu_mlir {
namespace backend {
class TgConcatKernel {
public:
  TgConcatKernel() {}

  void init(uint32_t layer_id, int input_num, int dim_size, int concat_axis,
            gaddr_t input_gaddrs[], gaddr_t output_gaddr, int axis_dims[],
            int output_dim[], bool do_relu, const int right_shift_width[],
            const int threshold_x_quantized[], cvk_fmt_t fmt);
  void selectTilePolicy();
  void schedule();

protected:
  void compute(int32_t step_idx);
  void load(int32_t step_idx);
  void store(int32_t step_idx);
  void update_output(int output_dim[], int dim_size, int concat_axis);
  uint32_t &axis_dim(cvk_tg_shape_t &shape);
  uint64_t dst_offset(const CV18xx::tiling_info_t &tile) const;

protected:
  gaddr_t ga_output;
  cvk_tg_shape_t output_shape;
  cvk_tg_stride_t output_stride;
  bool do_relu;
  int dim_size;
  int axis;
  int axis_before;
  int axis_after;
  int input_num;
  cvk_fmt_t fmt;
  int32_t layer_id;
  uint32_t base_addr[4]; // do flip, 2 input, 2 output
  cvk_tl_t tl_input;
  cvk_tl_t tl_output;
  bool do_parallel;

  typedef struct {
    bool do_quantize;
    gaddr_t ga_input;
    gaddr_t ga_output;
    cvk_tg_shape_t shape;
    cvk_tg_stride_t stride;
    int rshift_width;
    int data_quantized;
    int tile_idx;
    std::vector<CV18xx::tiling_info_t> tiles;
  } input_info_t;
  std::vector<input_info_t> inputs;
  CV18xx::tiling_mode_t tiling_mode;
  int total_tiles;
  void prepare(int32_t step_idx, input_info_t *&input,
               CV18xx::tiling_info_t *&tile);
};
} // namespace backend
} // namespace tpu_mlir
