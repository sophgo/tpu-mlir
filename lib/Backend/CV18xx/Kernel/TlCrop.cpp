//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "tl_crop"

namespace tpu_mlir {
namespace backend {
void cvi_backend_tl_crop(uint32_t layer_id, int64_t *input_dim,
                         int64_t *output_dim, laddr_t la_input,
                         laddr_t la_output, int *offsets, cvk_fmt_t fmt) {

  LLVM_DEBUG(llvm::errs() << llvm::format("cvi_backend_tl_crop:\n"
                                          "  layer_id %d\n",
                                          layer_id));

  LLVM_DEBUG(llvm::errs() << llvm::format("la_input:%d\n", la_input));

  LLVM_DEBUG(llvm::errs() << llvm::format("la_output:%d\n", la_output));

  uint32_t in = input_dim[0];
  uint32_t ic = input_dim[1];
  uint32_t ih = input_dim[2];
  uint32_t iw = input_dim[3];

  uint32_t on = output_dim[0];
  uint32_t oc = output_dim[1];
  uint32_t oh = output_dim[2];
  uint32_t ow = output_dim[3];

  uint32_t offset_n = offsets[0];
  uint32_t offset_c = offsets[1];
  uint32_t offset_h = offsets[2];
  uint32_t offset_w = offsets[3];

  cvk_tl_shape_t input_shape = {in, ic, ih, iw};
  cvk_tl_shape_t output_shape = {on, oc, oh, ow};
  cvk_tl_t tl_input = {};
  tl_input.start_address = la_input;
  tl_input.fmt = fmt;
  tl_input.shape = input_shape;
  tl_input.stride = CV18xx::tl_default_stride(input_shape, fmt, 1);

  cvk_tl_t tl_output = {};
  tl_output.start_address = la_output;
  tl_output.fmt = fmt;
  tl_output.shape = output_shape;
  tl_output.stride = CV18xx::tl_default_stride(output_shape, fmt, 1);

  auto input_offset =
      offset_n * tl_input.stride.n +
      ceiling_func(offset_c, CV18xx::NPU_NUM) * tl_input.stride.c +
      offset_h * tl_input.stride.h + offset_w * tl_input.stride.w;

  uint32_t input_addr = la_input + input_offset;
  tl_input.start_address = input_addr;
  tl_input.shape = output_shape;
  if (in == on && ic == oc & ih == oh && iw == ow && in < 4096 && ic < 4096 &&
      ih < 4096 - 32 && iw < 4096 - 32) {
    cvk_tiu_copy_param_t p = {0};
    p.src = &tl_input;
    p.dst = &tl_output;
    p.layer_id = layer_id;
    CV18xx::tiu_copy(&p);
  } else {
    cvk_tdma_l2l_tensor_copy_param_t p1 = {0};
    p1.dst = &tl_output;
    p1.src = &tl_input;
    p1.layer_id = layer_id;
    CV18xx::parallel_disable();
    CV18xx::tdma_l2l_tensor_copy(&p1);
    CV18xx::parallel_enable();
  }
}

} // namespace backend
} // namespace tpu_mlir
