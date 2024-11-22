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

void tpu::PackRawOp::codegen_global_cv18xx(int64_t layer_id) {
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  auto channel_order = module::getI64Array(getChannelOrder());
  gaddr_t ga_table_high = module::getAddress(getHighTable());
  gaddr_t ga_table_low = module::getAddress(getLowTable());
  cvk_fmt_t fmt =
      module::isUniformQuantized(getOutput()) ? CVK_FMT_I8 : CVK_FMT_BF16;
  int chennel_order_cvi[4];
  int start_h = 0;
  int start_w = 0;
  for (int i = 0; i < 4; i++) {
    chennel_order_cvi[i] = channel_order->at(i);
  }
  cvi_backend_tg_bnr_preprocess_kernel(
      layer_id, ga_input, ga_output, ga_table_high, ga_table_low, n, c, h, w,
      start_h, start_w, chennel_order_cvi, fmt);
}