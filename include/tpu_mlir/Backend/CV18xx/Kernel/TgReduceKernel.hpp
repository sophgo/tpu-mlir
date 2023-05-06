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

typedef enum reduce_type {
  REDUCE_MEAN,
  REDUCE_MAX,
  REDUCE_MIN,
  REDUCE_SUM,
  REDUCE_L2,
} reduce_type_t;

class TgReduceKernel {
public:
  TgReduceKernel() {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
            std::vector<int64_t> shape, std::vector<int32_t> axes,
            int multiplier, int rshift, reduce_type_t type, cvk_fmt_t fmt,
            gaddr_t ga_table = 0, gaddr_t ga_mantissa_table = 0);

  void selectTilePolicy();
  void schedule();

protected:
  void compute(int32_t step_idx);
  void load(int32_t step_idx);
  void store(int32_t step_idx);
  void refresh(int32_t step_idx);
  void allocLmem();
  void deallocLmem();
  void reshape(std::vector<int64_t> shape, std::vector<int32_t> axes);
  void reduce_max();
  void reduce_min();
  void reduce_mean();
  void reduce_sum();
  void reduce_l2();

protected:
  uint32_t layer_id;
  gaddr_t ga_input;
  gaddr_t ga_output;
  gaddr_t ga_table;
  gaddr_t ga_mantissa_table;
  int n, c, h, w;
  int kh, kw;
  reduce_type_t type;
  cvk_fmt_t fmt;
  int fmt_size;
  bool end_reduce;
  cvk_tl_t *tl_mem[4];
  cvk_tl_t *tl_lut;
  cvk_tl_t *tl_lut_mantissa;
  cvk_tl_t *tl_sum; // for reduce_l2
  cvk_tg_stride_t in_gstride;
  cvk_tg_stride_t out_gstride;

  cvk_tl_t tl_ifmap;
  cvk_tl_t tl_ofmap;
  int multiplier;
  int rshift;
  std::vector<int32_t> axes;
  std::vector<CV18xx::tiling_info_t> tiles;
};
} // namespace backend
} // namespace tpu_mlir
