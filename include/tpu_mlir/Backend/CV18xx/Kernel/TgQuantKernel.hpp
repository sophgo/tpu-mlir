//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TG_QUANT_KERNEL_HPP
#define TG_QUANT_KERNEL_HPP

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include <llvm/Support/Debug.h>

namespace tpu_mlir {
namespace backend {
class TgQuantKernel {
public:
  TgQuantKernel() {}

  void init(uint32_t layer_id, cvk_fmt_t from, cvk_fmt_t to, gaddr_t ga_input,
            gaddr_t ga_output, int32_t n, int32_t c, int32_t h, int32_t w,
            float const_scale, int offset);

  void selectTilePolicy();
  void doTileForNormalCase();
  void schedule();

protected:
  void compute(int32_t step_idx, int32_t flip);
  void load(int32_t step_idx, int32_t flip);
  void store(int32_t step_idx, int32_t flip);
  void allocLmem();
  void deallocLmem();
  cvk_tl_t *alloc_lmem(const cvk_tl_shape_t &shape, bool clean) const;
  cvk_tl_stride_t tl_fp32_stride(const cvk_tl_shape_t &shape,
                                 int eu_align) const;

protected:

  gaddr_t ga_input;
  gaddr_t ga_output;

  cvk_tl_t *tl_input[2];
  cvk_tl_t *tl_output[2];

  int32_t n, c, h, w;
  cvk_fmt_t from;
  cvk_fmt_t to;
  int32_t from_byte;
  int32_t to_byte;
  int32_t load_unit;
  int32_t store_unit;
  int32_t layer_id;
  int32_t flip = 0;
  float const_scale;
  int offset;
  std::vector<CV18xx::tiling_info_t> tiles;
};
} // namespace backend
} // namespace tpu_mlir
#endif
