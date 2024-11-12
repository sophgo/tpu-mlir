#pragma once

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include <llvm/Support/Debug.h>

namespace tpu_mlir {
namespace backend {

class TgBnrPreprocessKernel {
public:
  TgBnrPreprocessKernel() {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_output,
            gaddr_t ga_table_high, gaddr_t ga_table_low, int n, int c, int h,
            int w, int start_h, int start_w, int channel_order[4],
            cvk_fmt_t fmt);

  void selectTilePolicy();
  void schedule();

protected:
  void compute(int32_t step_idx);
  void load(int32_t step_idx, int iter);
  void store(int32_t step_idx, int iter);
  void refresh(int32_t step_idx);
  void allocLmem();
  void deallocLmem();
  void reshape();
  void int16_to_bf16(cvk_tl_t *in_high, cvk_tl_t *in_low, cvk_tl_t *out,
                     cvk_tl_t *tmp, cvk_tl_t *bf_high, cvk_tl_t *bf_low,
                     cvk_tl_t *table_high, cvk_tl_t *table_low);
  void split_pixel(cvk_tl_t *in_pixel, cvk_tl_t *in_AB, cvk_tl_t *out_low,
                   cvk_tl_t *out_high, cvk_tl_t *tl_shift_0f,
                   cvk_tl_t *tl_shift_f0, bool is_A);

protected:
  uint32_t layer_id;
  gaddr_t ga_input;
  gaddr_t ga_out[4];
  gaddr_t ga_table_high;
  gaddr_t ga_table_low;
  cvk_fmt_t fmt;
  int n, c, h, w;
  int ih, iw, oh, ow;
  int start_h, start_w;
  int channel_order[4];

  cvk_tl_t tl_ifmap;
  cvk_tl_t tl_ofmap_A;
  cvk_tl_t tl_ofmap_B;
  cvk_tg_stride_t in_gstride;
  cvk_tg_stride_t out_gstride;
  cvk_tl_shape_t lut_shape;

  cvk_tl_t *tl_and_0f_shift;
  cvk_tl_t *tl_and_f0_shift;
  cvk_tl_t *tl_table[2];
  cvk_tl_t *tl_in[2];
  cvk_tl_t *tl_out_A[2];
  cvk_tl_t *tl_out_B[2];
  cvk_tl_t *tl_out_tmp_bf16;
  cvk_tl_t *tl_out_tmp_int8[4];
  std::vector<CV18xx::tiling_info_t> tiles;
};

} // namespace backend
} // namespace tpu_mlir
