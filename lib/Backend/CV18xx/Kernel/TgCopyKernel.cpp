//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"

#define DEBUG_TYPE "cvi_backend_copy_kernel"
namespace tpu_mlir {
namespace backend {
void cvi_backend_tg_copy_kernel(gaddr_t ga_input, gaddr_t ga_output,
                                const std::vector<int> &shape,
                                const std::vector<int> &i_stride,
                                const std::vector<int> &o_stride,
                                cvk_fmt_t fmt) {
  assert(shape.size() == 4);
  int fmt_size = CV18xx::bytesize_of_fmt(fmt);
  cvk_tg_shape_t output_shape =
      CV18xx::tg_shape_t4(shape[0], shape[1], shape[2], shape[3]);
  cvk_tg_shape_t input_shape = output_shape;
  cvk_tg_stride_t input_gstride;
  input_gstride.n = i_stride[0] * fmt_size;
  input_gstride.c = i_stride[1] * fmt_size;
  input_gstride.h = i_stride[2] * fmt_size;
  input_gstride.w = i_stride[3] * fmt_size;

  cvk_tg_stride_t output_gstride;
  output_gstride.n = o_stride[0] * fmt_size;
  output_gstride.c = o_stride[1] * fmt_size;
  output_gstride.h = o_stride[2] * fmt_size;
  output_gstride.w = o_stride[3] * fmt_size;

  CV18xx::tdma_g2g_tensor_copy(ga_input, input_shape, input_gstride, fmt,
                               ga_output, output_shape, output_gstride, fmt);
}
} // namespace backend
} // namespace tpu_mlir
