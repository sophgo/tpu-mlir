#pragma once

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include <llvm/Support/Debug.h>

namespace tpu_mlir {
namespace backend {
class TgMMap2RGBMapKernel {
public:
  TgMMap2RGBMapKernel() {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output, int n,
            int c, int h, int w, int block_size, cvk_fmt_t fmt);

  void selectTilePolicy();
  void schedule();

protected:
  void compute(int32_t step_idx);
  void load(int32_t step_idx);
  void store(int32_t step_idx);
  void allocLmem();
  void deallocLmem();

protected:
  uint32_t layer_id;
  gaddr_t ga_input;
  gaddr_t ga_output;
  cvk_fmt_t fmt;
  int n, c, h, w;
  int block_size;
  int in_data_size;

  cvk_tl_t *tl_buffer;
  cvk_tl_t *tl_cast;

  std::vector<CV18xx::tiling_info_t> input_tiles;
  std::vector<CV18xx::tiling_info_t> output_tiles;
};
} // namespace backend
} // namespace tpu_mlir
