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
class TgReluKernel {
public:
  TgReluKernel() {}

  typedef enum { RELU, LEAKY_RELU, PRELU } mode_t;

  void init(uint32_t layer_id, int32_t n, int32_t c, int32_t h, int32_t w,
            gaddr_t ga_input, gaddr_t ga_output, gaddr_t ga_negative_slope,
            float negative_slope, int GT_rshift, int GT_scale, int LE_rshift,
            int LE_scale, cvk_fmt_t fmt, mode_t mode);

  void selectTilePolicy();
  void schedule();

protected:
  void compute(int32_t step_idx, int32_t flip);
  void load(int32_t step_idx, int32_t flip);
  void store(int32_t step_idx, int32_t flip);
  void allocLmem();
  void deallocLmem();
  void compute_relu(int32_t step_idx, int32_t flip);
  void compute_leaky_relu_fixed_sym(int32_t step_idx, int32_t flip);
  void compute_leaky_relu_bf16(int32_t step_idx, int32_t flip);
  void compute_prelu_fixed(int32_t step_idx, int32_t flip);
  void compute_prelu_bf16(int32_t step_idx, int32_t flip);
  cvk_tl_t get_input(int32_t step_idx, int32_t flip);
  cvk_tl_t get_output(int32_t step_idx, int32_t flip);
  void change_workspace_size(int32_t step_idx);

protected:

  gaddr_t ga_input;
  gaddr_t ga_output;

  cvk_tl_t *tl_input[2];
  cvk_tl_t *tl_output[2];
  cvk_tl_t *tl_slope; // for prelu
  cvk_tl_t *tl_working[2];
  cvk_tl_t *tl_pos_neg_map;
  cvk_tg_stride_t gstride;

  int32_t n, c, h, w;
  int32_t layer_id;
  cvk_fmt_t fmt;
  mode_t mode;

  gaddr_t ga_slope; // for prelu
  int GT_rshift;    // for i8
  int GT_scale;     // for i8
  int LE_rshift;    // for i8
  int LE_scale;     // for i8
  float negative_slope;

  int32_t flip = 0;
  std::vector<CV18xx::tiling_info_t> tiles;
};
}
}
