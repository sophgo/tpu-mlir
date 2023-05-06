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
typedef struct {
  int32_t n;
  int32_t c;
  int32_t h;
  int32_t w;
  int32_t c_pos;
  int32_t oh;
  int32_t ow;
  int32_t ih_pos;
  int32_t oh_pos;
  int32_t pad[4];
  uint64_t input_offset;
  uint64_t output_offset;
} PoolingTile;

class TgInt8PoolingKernel {
public:
  TgInt8PoolingKernel() {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output, int32_t n,
            int32_t c, int32_t h, int32_t w, int32_t pad_t, int32_t pad_b,
            int32_t pad_l, int32_t pad_r, int32_t kh, int32_t kw,
            int32_t stride_h, int32_t stride_w, bool is_avg_pooling,
            bool do_relu, int32_t rshift, int32_t multipliers, bool ceil_mode);

  void selectTilePolicy();
  void schedule();

protected:
  void allocLmem(cvk_tl_shape_t &input_shape, cvk_tl_shape_t &output_shape);
  void deallocLmem();
  void doTileForNormalCase();
  void compute(int32_t step_idx, int32_t flip);
  void load(int32_t step_idx, int32_t flip);
  void store(int32_t step_idx, int32_t flip);
  void adjustPadding();

  gaddr_t ga_input;
  gaddr_t ga_output;

  cvk_tl_t *tl_input[2];
  cvk_tl_t *tl_output[2];

  int32_t n;
  int32_t c;
  int32_t h;
  int32_t w;
  int32_t oh;
  int32_t ow;
  int32_t pad_t;
  int32_t pad_b;
  int32_t pad_l;
  int32_t pad_r;
  int32_t kh;
  int32_t kw;
  int32_t stride_h;
  int32_t stride_w;
  int32_t rshift;
  int32_t multiplier;
  int32_t layer_id;
  bool do_relu;
  bool is_avg_pooling;
  bool ceil_mode;

  int32_t flip = 0;

  std::vector<PoolingTile> tiles;
};

} // namespace backend
} // namespace tpu_mlir
