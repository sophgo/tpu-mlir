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

void tpu::SliceOp::codegen_global_cv18xx(int64_t layer_id) {
  auto p = parseParam();
  CVIKERNEL_FMT_E fmt;
  if (module::isUniformQuantized(getOutput())) {
    fmt = CVK_FMT_I8;
  } else {
    fmt = CVK_FMT_BF16;
  }
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  if (p.fusible == false) {
    cvi_backend_tg_crop_kernel(layer_id, ga_input, ga_output, p.is_4, p.os_4,
                               p.offset_4, p.step_4, fmt);
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::SliceOp::getBufferSize_cv18xx(int64_t in_lmem_bytes,
                                           int64_t out_lmem_bytes,
                                           int64_t in_nslice, int64_t in_hslice,
                                           int64_t out_nslice,
                                           int64_t out_hslice) {
  return 0;
}

void tpu::SliceOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                        int64_t d_step, int64_t w_step,
                                        group_type_t group_type,
                                        local_sec_info_t &sec_info,
                                        int64_t layer_id) {
  int64_t in, ic, ih, iw, on, oc, oh, ow;
  module::getNCHW(getOutput(), on, oc, oh, ow);
  module::getNCHW(getInput(), in, ic, ih, iw);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto out_gi = LocalGenInterface::getGroupInfo(getOutput(), n_step, h_step);
  auto crop_offset = module::getI64Array(getOffset());
  laddr_t la_input = in_gi.out_addr;
  laddr_t la_output = out_gi.out_addr;

  std::vector<int64_t> input_shape = {sec_info.n_slice, ic, sec_info.h_slice,
                                      iw};
  std::vector<int64_t> output_shape = {sec_info.out_n_slice, oc,
                                       sec_info.out_h_slice, ow};
  std::vector<int32_t> crop_offset_v;
  crop_offset_v.assign(crop_offset->begin(), crop_offset->end());
  auto fmt = CV18xx::getDataType(getOutput());

  cvi_backend_tl_crop(layer_id, input_shape.data(), output_shape.data(),
                      la_input, la_output, crop_offset_v.data(), fmt);
}
