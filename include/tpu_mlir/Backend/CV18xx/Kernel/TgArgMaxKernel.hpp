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
  int32_t iw_pos;
  int32_t oh_pos;
  uint64_t input_offset;
  uint64_t output_offset;
} ArgMaxTile;

class TgArgMaxKernel {
public:
  TgArgMaxKernel() {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
            int32_t outer, int32_t inner, int32_t w_tile_size, cvk_fmt_t fmt);
  void selectTilePolicy();
  void schedule();

protected:
  void allocLmem(cvk_tl_shape_t &input_shape, cvk_tl_shape_t &output_shape);
  void deallocLmem();
  void doTileForNormalCase();
  void compute(int32_t step_idx, int32_t flip);
  void load(int32_t step_idx, int32_t flip);
  void store(int32_t step_idx, int32_t flip);

  gaddr_t ga_input;
  gaddr_t ga_output;

  cvk_tl_t *tl_input[2];
  cvk_tl_t *tl_output[2];

  cvk_fmt_t fmt;
  int32_t n;
  int32_t c;
  int32_t h;
  int32_t w;
  int32_t oh;
  int32_t ow;
  int32_t w_tile_size;
  int32_t layer_id;
  int32_t flip = 0;
  std::vector<ArgMaxTile> tiles;
};
} // namespace backend
} // namespace tpu_mlir
