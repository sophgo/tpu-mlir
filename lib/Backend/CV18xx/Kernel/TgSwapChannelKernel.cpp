//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"

#define DEBUG_TYPE "cvi_backend_swapchannel_kernel"
namespace tpu_mlir {
namespace backend {
void cvi_backend_tg_swap_channel_kernel(uint32_t layer_id, gaddr_t input_gaddr,
                                        gaddr_t output_gaddr,
                                        int input_dim_size, int *input_dim,
                                        int *channel_order, cvk_fmt_t fmt) {
  assert(input_dim_size == 4 && input_dim[1] == 3 && "parameter error");
  cvk_tg_shape_t shape =
      CV18xx::tg_shape_t4(input_dim[0], 1, input_dim[2], input_dim[3]);
  cvk_tg_stride_t stride =
      CV18xx::tg_default_stride(input_dim[1], input_dim[2], input_dim[3], fmt);
  uint64_t frame_size =
      input_dim[2] * input_dim[3] * CV18xx::bytesize_of_fmt(fmt);
  for (uint32_t i = 0; i < 3; i++) {
    assert((uint32_t)channel_order[i] < 3 && "channel_order is illegal");
    gaddr_t s_gaddr = input_gaddr + frame_size * channel_order[i];
    gaddr_t d_gaddr = output_gaddr + frame_size * i;
    CV18xx::tdma_g2g_tensor_copy(s_gaddr, shape, stride, fmt, d_gaddr, shape,
                                 stride, fmt);
  }
}
} // namespace backend
} // namespace tpu_mlir
