//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"

#define DEBUG_TYPE "cvi_backend_scatterND_kernel"
namespace tpu_mlir {
namespace backend {
void cvi_backend_tg_scatterND_kernel(gaddr_t ga_input, gaddr_t ga_updates,
                                     gaddr_t ga_output,
                                     const std::vector<int> &ishape,
                                     const std::vector<int> &ushape,
                                     const std::vector<int> &o_stride,
                                     const uint32_t offset, cvk_fmt_t fmt) {
  assert(ishape.size() == 4 && ushape.size() == 4);
  int fmt_size = CV18xx::bytesize_of_fmt(fmt);
  // first copy inp to out
  // todo i_stride and u_stride is unnessery
  cvk_tg_shape_t input_shape =
      CV18xx::tg_shape_t4(ishape[0], ishape[1], ishape[2], ishape[3]);
  cvk_tg_stride_t input_gstride = CV18xx::tg_default_stride(input_shape, fmt);
  ;
  cvk_tg_stride_t output_gstride = input_gstride;
  CV18xx::tdma_g2g_tensor_copy(ga_input, input_shape, input_gstride, fmt,
                               ga_output, input_shape, output_gstride, fmt);
  // second copy update to out
  cvk_tg_shape_t update_shape =
      CV18xx::tg_shape_t4(ushape[0], ushape[1], ushape[2], ushape[3]);
  input_gstride = CV18xx::tg_default_stride(update_shape, fmt);
  output_gstride.n = o_stride[0] * fmt_size;
  output_gstride.c = o_stride[1] * fmt_size;
  output_gstride.h = o_stride[2] * fmt_size;
  output_gstride.w = o_stride[3] * fmt_size;
  CV18xx::tdma_g2g_tensor_copy(ga_updates, update_shape, input_gstride, fmt,
                               ga_output + offset * fmt_size, update_shape,
                               output_gstride, fmt);
}
} // namespace backend
} // namespace tpu_mlir
