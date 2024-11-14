//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================
void tpu::UpsampleOp::codegen_global_cv18xx(int64_t layer_id) {

  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  auto scale_h = this->getScaleH();
  auto scale_w = this->getScaleW();
  if (module::isUniformQuantized(getOutput())) {
    cvi_backend_tg_upsample_kernel(layer_id, ga_input, ga_output, n, c, h, w,
                                   scale_h, scale_w, CVK_FMT_I8);
  } else {
    cvi_backend_tg_upsample_kernel(layer_id, ga_input, ga_output, n, c, h, w,
                                   scale_h, scale_w, CVK_FMT_BF16);
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::UpsampleOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::UpsampleOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                           int64_t d_step, int64_t w_step,
                                           group_type_t group_type,
                                           local_sec_info_t &sec_info,
                                           int64_t layer_id) {
  int64_t n, c, ih, iw;
  module::getNCHW(getInput(), n, c, ih, iw);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto out_gi = LocalGenInterface::getGroupInfo(getOutput(), n_step, h_step);
  n = sec_info.n_slice;
  ih = sec_info.h_slice;
  laddr_t la_input = in_gi.out_addr;
  laddr_t la_output = out_gi.out_addr;
  auto ifmt = CV18xx::getDataType(getInput());
  auto scale_h = getScaleH();
  auto scale_w = getScaleW();

  cvi_backend_tl_upsample(layer_id, // layer_id,
                          la_input, la_output, n, c, ih, iw, scale_h, scale_w,
                          ifmt);
}
