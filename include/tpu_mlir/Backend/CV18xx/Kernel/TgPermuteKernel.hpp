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
class TgPermuteKernel {
public:
  TgPermuteKernel() {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output, int n,
            int c, int h, int w, int order_n, int order_c, int order_h,
            int order_w, cvk_fmt_t fmt);

  void selectTilePolicy();
  void schedule();

protected:
  bool is_order(int order_n, int order_c, int order_h, int order_w) const;
  void update_order(int order_n, int order_c, int order_h, int order_w);
  void update_NCHW(int n, int c, int h, int w);
  void convert_order();
  void allocLmem();
  void deallocLmem();
  void refresh(int step_idx);
  void compute(int step_idx);
  void load(int step_idx);
  void store(int step_idx);
  void reshape(int dim0, int dim1);
  void reshape(int channel);
  void permute_tdma();
  uint32_t tile_offset(const CV18xx::tiling_info_t &tile,
                       bool is_src = true) const;


  gaddr_t ga_input;
  gaddr_t ga_output;
  cvk_tl_t *tl_mem[4];
  cvk_tl_t tl_ifmap;
  cvk_tl_t tl_ofmap;
  int n, c, h, w; // input_shape
  int step[4];
  int n_loop;
  uint64_t n_offset;
  cvk_tg_stride_t src_stride;
  cvk_tg_stride_t dst_stride;
  cvk_tg_stride_t dst_stride_order;
  int order[4];
  cvk_fmt_t fmt;
  int fmt_bytes;
  int layer_id;
  bool by_tdma;
  std::vector<CV18xx::tiling_info_t> tiles;
};
}
}
