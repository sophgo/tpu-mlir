//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "tl_relu"

#define METHOD_MANTISSA 0
#define METHOD_LOG 1
#define METHOD_SLOPE 2

namespace tpu_mlir {
namespace backend {
void cvi_backend_tl_relu(uint32_t layer_id, int n, int c, int h, int w,
                         laddr_t la_input, laddr_t la_output) {

  LLVM_DEBUG(llvm::errs() << llvm::format("cvi_backend_tl_relu:\n"
                                          "  layer_id %d\n",
                                          layer_id));
  LLVM_DEBUG(llvm::errs() << llvm::format("la_input:%d\n", la_input));

  LLVM_DEBUG(llvm::errs() << llvm::format("la_output:%d\n", la_output));

  cvk_tl_shape_t shape = CV18xx::tl_shape_t4(n, c, h, w);
  cvk_tl_t tl_input;
  tl_input.start_address = la_input;
  tl_input.fmt = CVK_FMT_I8;
  tl_input.shape = shape;
  tl_input.stride = CV18xx::tl_default_stride(shape, CVK_FMT_I8, 1);

  cvk_tl_t tl_output;
  tl_output.start_address = la_output;
  tl_output.fmt = CVK_FMT_I8;
  tl_output.shape = shape;
  tl_output.stride = CV18xx::tl_default_stride(shape, CVK_FMT_I8, 1);

  cvk_tiu_max_param_t p1 = {0};
  p1.max = &tl_output;
  p1.a = &tl_input;
  p1.b_is_const = 1;
  p1.b_const.is_signed = 1;
  p1.b_const.val = 0;
  p1.layer_id = layer_id;
  CV18xx::tiu_max(&p1);
}

void cvi_backend_tl_bf16_relu(uint32_t layer_id, int n, int c, int h, int w,
                              laddr_t la_input, laddr_t la_output) {

  LLVM_DEBUG(llvm::errs() << llvm::format("cvi_backend_tl_bf16_relu:\n"
                                          "  layer_id %d\n",
                                          layer_id));
  LLVM_DEBUG(llvm::errs() << llvm::format("la_input:%d\n", la_input));

  LLVM_DEBUG(llvm::errs() << llvm::format("la_output:%d\n", la_output));

  cvk_tl_shape_t shape = CV18xx::tl_shape_t4(n, c, h, w);
  cvk_tl_t tl_input;
  tl_input.start_address = la_input;
  tl_input.fmt = CVK_FMT_BF16;
  tl_input.shape = shape;
  tl_input.stride = CV18xx::tl_default_stride(shape, CVK_FMT_BF16, 1);

  cvk_tl_t tl_output;
  tl_output.start_address = la_output;
  tl_output.fmt = CVK_FMT_BF16;
  tl_output.shape = shape;
  tl_output.stride = CV18xx::tl_default_stride(shape, CVK_FMT_BF16, 1);

  cvk_tiu_max_param_t p1 = {0};
  p1.max = &tl_output;
  p1.a = &tl_input;
  p1.b_is_const = 1;
  p1.b_const.is_signed = 1;
  p1.b_const.val = 0;
  p1.layer_id = layer_id;
  CV18xx::tiu_max(&p1);
}
} // namespace backend
} // namespace tpu_mlir
