//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/Support/Debug.h>
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"

namespace tpu_mlir {
namespace backend {
class TgScaleLutKernel {
public:
  TgScaleLutKernel()  {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
            gaddr_t table_gaddr, int n, int c, int h, int w, cvk_fmt_t fmt);

  void selectTilePolicy();
  void schedule();

protected:
  void compute(int32_t step_idx);
  void load(int32_t step_idx);
  void store(int32_t step_idx);
  void refresh(int32_t step_idx);
  void allocLmem();
  void deallocLmem();
  void reshape();

protected:
  uint32_t layer_id;
  gaddr_t ga_input;
  gaddr_t ga_output;
  gaddr_t ga_lut;
  cvk_fmt_t fmt;
  int n, c, h, w, c_times, c_ori;

  cvk_tl_t tl_ifmap;
  cvk_tl_t tl_ofmap;
  cvk_tg_stride_t gstride;
  cvk_tl_shape_t lut_shape;
  static const int BLOB_NUM = 4;
  cvk_tl_t *tl_mem[BLOB_NUM];
  std::vector<cvk_tl_t *> tl_lut;
  std::vector<CV18xx::tiling_info_t> tiles;
};
}
}
