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
// =========================================
// GlobalGenInterface
// =========================================

void tpu::CscOp::codegen_global_cv18xx(int64_t layer_id) {
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w, false);
  std::vector<int32_t> order;
  if (this->getChannelOrder().has_value()) {
    order = *(module::getI32Array(this->getChannelOrderAttr()));
  }
  gaddr_t input_gaddr = module::getAddress(this->getInput());
  gaddr_t output_gaddr = module::getAddress(this->getOutput());
  cvi_backend_tg_yuv420_csc_kernel(layer_id, input_gaddr, output_gaddr, n, c, h,
                                   w, order, CVK_FMT_U8, this->getPixelType(),
                                   this->getYAlign(), this->getWAlign(),
                                   this->getChannelAlign());
}
