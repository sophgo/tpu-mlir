//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::Mmap2RgbmapOp::codegen_global_cv18xx(int64_t layer_id) {
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  int64_t in, ic, ih, iw;
  module::getNCHW(getInput(), in, ic, ih, iw);
  cvk_fmt_t fmt = CV18xx::getDataType(getInput());
  int64_t on, oc, oh, ow;
  module::getNCHW(getOutput(), on, oc, oh, ow);
  assert(ow % iw == 0);
  cvi_backend_tg_mmap2rgbmap_kernel(layer_id, ga_input, ga_output, in, ic, ih,
                                    iw, ow / iw, fmt);
}
