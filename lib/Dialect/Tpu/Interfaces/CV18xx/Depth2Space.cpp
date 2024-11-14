//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

using namespace tpu_mlir::backend;

void tpu::Depth2SpaceOp::codegen_global_cv18xx(int64_t layer_id) {
  assert(getInIs_NCHW() && getOutIs_NCHW() && !getSwapCr());
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  int64_t n, c, h, w;
  module::getNCHW(this->getInput(), n, c, h, w);
  int64_t scale_h = getBlockH();
  int64_t scale_w = getBlockW();
  assert(scale_h == scale_w);
  bool isDCR = !getIs_CRD();
  bool isInversed = getIsInversed();
  if (module::isUniformQuantized(getOutput())) {
    if (isInversed) {
      cvi_backend_tg_reorg_kernel(layer_id, ga_input, ga_output, n, c, h, w,
                                  scale_h, CVK_FMT_I8);
    } else {
      cvi_backend_tg_fixed_pixel_shuffle_kernel(layer_id, ga_input, ga_output,
                                                n, c, h, w, scale_h, isDCR);
    }
  } else {
    if (isInversed) {
      cvi_backend_tg_reorg_kernel(layer_id, ga_input, ga_output, n, c, h, w,
                                  scale_h, CVK_FMT_BF16);

    } else {
      cvi_backend_tg_bf16_pixel_shuffle_kernel(layer_id, ga_input, ga_output, n,
                                               c, h, w, scale_h, isDCR);
    }
  }
}
