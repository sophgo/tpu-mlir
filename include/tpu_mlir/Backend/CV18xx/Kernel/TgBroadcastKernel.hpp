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
typedef enum {
  BCAST_HW,
  BCAST_C,
  BCAST_ALL,
} bcast_mode_t;
typedef enum _bcast_t {
  BCAST_ADD,
  BCAST_SUB,
  BCAST_MUL,
} bcast_t;
bcast_mode_t mode;

class TgBcastKernel {
public:
  TgBcastKernel() {}

  void init(uint32_t layer_id, gaddr_t ga_a, gaddr_t ga_b, gaddr_t ga_output,
            int an, int ac, int ah, int aw, int bn, int bc, int bh, int bw,
            bool do_relu, int32_t rshift, const int32_t *multipliers,
            bcast_t type, cvk_fmt_t fmt);

  void selectTilePolicy();
  void schedule();

protected:
  void compute(int32_t step_idx);
  void load(int32_t step_idx);
  void store(int32_t step_idx);
  void convert_shape(int an, int ac, int ah, int aw, int bn, int bc, int bh,
                     int bw);
  void schedule_bcast_all();
  void schedule_bcast_c();
  void schedule_bcast_hw();
  void tile_all();
  void tile_other();
  void tiu_compute(cvk_tl_t *tl_result, cvk_tl_t *tl_left, cvk_tl_t *tl_right,
                   cvk_tl_t *tl_buff = nullptr);

protected:
  uint32_t layer_id;
  gaddr_t ga_a;
  gaddr_t ga_b;
  gaddr_t ga_output;
  int shape_a[4];
  int shape_b[4];
  bool do_relu;
  int32_t rshift;
  const int32_t *multipliers;
  bcast_mode_t mode;
  bcast_t type;
  cvk_fmt_t fmt;
  int fmt_bytes;
  int index_bcast;
  int num_blobs;
  std::vector<CV18xx::tiling_info_t> tiles;
};
} // namespace backend
} // namespace tpu_mlir
