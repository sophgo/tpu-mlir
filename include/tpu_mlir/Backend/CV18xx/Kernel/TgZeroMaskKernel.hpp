/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * refined 2020-11-12
 */
#pragma once

#include <llvm/Support/Debug.h>
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"

namespace tpu_mlir {
namespace backend {
class TgZeroMaskKernel {
public:
  TgZeroMaskKernel() {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output, int n,
            int c, int h, int w, bool positive, cvk_fmt_t fmt);

  void selectTilePolicy();
  void schedule();

protected:
  void compute(int32_t step_idx);
  void compute_bf16();
  void compute_int8();
  void load(int32_t step_idx);
  void store(int32_t step_idx);
  void refresh(int32_t step_idx);
  void allocLmem();
  void deallocLmem();

protected:
  uint32_t layer_id;
  gaddr_t ga_input;
  gaddr_t ga_output;
  cvk_fmt_t fmt;
  int n, c, h, w;
  bool positive;
  int blob_num;

  cvk_tl_t tl_ifmap;
  cvk_tl_t tl_ofmap;
  cvk_tl_t tl_buffer;
  cvk_tg_stride_t gstride;
  cvk_tl_t *tl_mem[5];
  std::vector<CV18xx::tiling_info_t> tiles;
};
} // namespace backend
} // namespace tpu_mlir
