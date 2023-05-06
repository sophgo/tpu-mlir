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
class TgPoolMaskKernel {
public:
  TgPoolMaskKernel() {}

  void init(uint32_t layer_id, gaddr_t input_gaddr, gaddr_t output_gaddr, int n,
            int c, int h, int w, int scale, cvk_fmt_t fmt);
  void selectTilePolicy();
  void schedule();

protected:
  void compute(int32_t step_idx);
  void load(int32_t step_idx);
  void store(int32_t step_idx);
  void refresh(int32_t step_idx);
  void allocLmem();
  void deallocLmem();
  void compute_bf16();
  void compute_int8();

protected:
  typedef struct {
    int32_t n;
    int32_t c;
    int32_t h;
    int32_t w;
    int32_t pos_n;
    int32_t pos_c;
    int32_t pos_h;
    int32_t pos_w;
    int32_t pad_h, pad_w;
    uint64_t src_offset;
    uint64_t dst_offset;
  } TileInfo;
  std::vector<TileInfo> tiles;
  gaddr_t ga_input;
  gaddr_t ga_output;
  cvk_tg_stride_t src_stride;
  cvk_tg_stride_t dst_stride;
  int n, c, h, w, h_ex, w_ex;
  int step_n, step_c, step_h, step_w;
  int scale;
  int blob_num;
  int32_t layer_id;
  cvk_fmt_t fmt;
  cvk_tl_shape_t kernel_shape;
  cvk_tl_t *tl_kernel;
  cvk_tl_t *tl_pooling;
  cvk_tl_t *tl_mem[6];
  cvk_tl_t tl_input;
  cvk_tl_t tl_output;
  cvk_tl_t tl_high;
  cvk_tl_t tl_low;
};
} // namespace backend
} // namespace tpu_mlir
