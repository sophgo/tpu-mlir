//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#define DEBUG_TYPE "reverse_kernel"

namespace tpu_mlir {
namespace backend {
void cvi_backend_tg_reverse_kernel(uint32_t layer_id, gaddr_t ga_input,
                                   gaddr_t ga_output, int n, int c, int h,
                                   int w, int axis, cvk_fmt_t fmt) {
  assert((axis == 0 || axis == 3) && "only support axis = 0, 3 now");
  if (axis == 0) {
    std::vector<CV18xx::tiling_info_t> tiles;
    CV18xx::tiling_packing(tiles, 1, c, h, w, fmt, 1, 0, CV18xx::TilingAll);
    uint64_t n_stride = c * h * w * CV18xx::bytesize_of_fmt(fmt);
    for (int in = 0; in < n; in++) {
      for (auto &tile : tiles) {
        cvk_tl_t *tl_input = CV18xx::lmem_alloc_tensor(
            CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w), fmt, 1);
        gaddr_t ga_ifmap = ga_input + in * n_stride + tile.offset;
        gaddr_t ga_ofmap = ga_output + (n - 1 - in) * n_stride + tile.offset;
        CV18xx::tdma_load(tl_input, ga_ifmap);
        CV18xx::tdma_store(tl_input, ga_ofmap);
        CV18xx::lmem_free_tensor(tl_input);
      }
    }
  }

  if (axis == 3) {
    std::vector<CV18xx::tiling_info_t> tiles;
    cvk_tg_stride_t gstride =
        CV18xx::tg_default_stride(CV18xx::tg_shape_t4(n, c, h, w), fmt);
    CV18xx::tiling_packing(tiles, n, c, h, 1, fmt, 1, 0);
    uint64_t w_stride = CV18xx::bytesize_of_fmt(fmt);
    for (int iw = 0; iw < w; iw++) {
      for (auto &tile : tiles) {
        cvk_tl_t *tl_input = CV18xx::lmem_alloc_tensor(
            CV18xx::tl_shape_t4(tile.n, tile.c, tile.h, tile.w), fmt, 1);
        gaddr_t ga_ifmap = ga_input + tile.offset * w + iw * w_stride;
        gaddr_t ga_ofmap =
            ga_output + tile.offset * w + (w - 1 - iw) * w_stride;

        CV18xx::tdma_load_stride(tl_input, ga_ifmap, gstride);
        CV18xx::tdma_store_stride(tl_input, ga_ofmap, gstride);
        CV18xx::lmem_free_tensor(tl_input);
      }
    }
  }
}
} // namespace backend
} // namespace tpu_mlir
