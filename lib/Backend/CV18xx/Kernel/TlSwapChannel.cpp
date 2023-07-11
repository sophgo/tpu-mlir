//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"

#define DEBUG_TYPE "tl_swapchannel"

namespace tpu_mlir {
namespace backend {
void cvi_backend_tl_swap_channel(uint32_t layer_id, laddr_t la_input,
                                 laddr_t la_output, int n, int c, int h, int w,
                                 int *order, cvk_fmt_t fmt) {
  cvk_tl_t tl_input = {0};
  tl_input.fmt = fmt;
  tl_input.shape = CV18xx::tl_shape_t4(n, 1, h, w);
  tl_input.stride =
      CV18xx::tl_default_stride(CV18xx::tl_shape_t4(n, 3, h, w), fmt, 1);
  cvk_tl_t tl_output = tl_input;
  cvk_tdma_l2l_tensor_copy_param_t p = {0};
  p.dst = &tl_output;
  p.src = &tl_input;
  p.layer_id = layer_id;
  CV18xx::parallel_disable();
  for (int i = 0; i < 3; i++) {
    tl_input.start_address = la_input + order[i] * CV18xx::LMEM_BYTES;
    tl_output.start_address = la_output + i * CV18xx::LMEM_BYTES;
    CV18xx::tdma_l2l_tensor_copy(&p);
  }
  CV18xx::parallel_enable();
}
} // namespace backend
} // namespace tpu_mlir
