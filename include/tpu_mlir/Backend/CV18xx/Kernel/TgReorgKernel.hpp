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
class TgReorgKernel {
public:
  TgReorgKernel() {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output, int n,
            int c, int h, int w, int stride, cvk_fmt_t fmt);

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
  cvk_fmt_t fmt;
  int fmt_bytes;
  int n_loop, n_offset;
  int n, c, h, w, oh, ow, r;
  cvk_tg_stride_t src_gstride;
  cvk_tg_stride_t dst_gstride;
  cvk_tl_t *tl_mem[5];
  cvk_tl_t tl_ifmap;
  cvk_tl_t tl_ofmap;
  cvk_tl_t tl_middle;
  std::vector<CV18xx::tiling_info_t> tiles;
};
} // namespace backend
} // namespace tpu_mlir
