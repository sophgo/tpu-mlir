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

// int8
void tpu::CastOp::codegen_global_cv18xx(int64_t layer_id) {
  int64_t n, c, h, w;
  int64_t offset = 0;
  float_t scale = 1.;
  module::getNCHW(getInput(), n, c, h, w);
  cvk_fmt_t from = CV18xx::getDataType(getInput());
  cvk_fmt_t to = CV18xx::getDataType(getOutput());
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());

  bool qInput = module::isUniformQuantized(getInput());
  bool qOutput = module::isUniformQuantized(getOutput());
  if (qInput || qOutput) {
    if (!qInput && qOutput) {
      auto qtype = module::getUniformQuantizedType(getOutput());
      scale = 1. / qtype.getScale();
    } else {
      auto qtype = module::getUniformQuantizedType(getInput());
      scale = qtype.getScale();
    }
  }
  //  quant to int8
  cvi_backend_tg_quant_kernel(layer_id, from, to, ga_input, ga_output, n, c, h,
                              w, scale, offset);
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::CastOp::getBufferSize_cv18xx(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  auto in_type = module::getStorageType(getInput());
  if (in_type.isBF16() && !getInput().hasOneUse()) {
    // to avoid quant input been override
    // check if quant's input has multi-usage
    int64_t n, c, h, w;
    auto vIn = getInput();
    auto fmt = CV18xx::getDataType(vIn);
    module::getNCHW(vIn, n, c, h, w);
    n = in_nslice;
    h = in_hslice;
    auto op = getOperation();
    OpBuilder builder(op->getContext());
    getOperation()->setAttr("extra_input", builder.getBoolAttr(true));
    return CV18xx::lmem_woring_size({n, c, h, w}, 1, true, fmt);
  }
  return 0;
}

void tpu::CastOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                       int64_t d_step, int64_t w_step,
                                       group_type_t group_type,
                                       local_sec_info_t &sec_info,
                                       int64_t layer_id) {
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto out_gi = LocalGenInterface::getGroupInfo(getOutput(), n_step, h_step);

  n = sec_info.n_slice;
  h = sec_info.h_slice;

  laddr_t la_input = in_gi.out_addr;
  laddr_t la_output = out_gi.out_addr;
  laddr_t la_working = gi.buffer_addr;
  bool bExtraInput = false;
  if (getExtraInput().has_value() && getExtraInput().value()) {
    bExtraInput = true;
  }
  auto ifmt = CV18xx::getDataType(getInput());
  auto ofmt = CV18xx::getDataType(getOutput());

  float_t scale = 1.;
  if (module::isUniformQuantized(getInput())) {
    auto qtype = module::getUniformQuantizedType(getInput());
    scale = qtype.getScale();
  } else {
    auto qtype = module::getUniformQuantizedType(getOutput());
    scale = 1. / qtype.getScale();
  }

  // FIXME: support F16 -> U8
  cvi_backend_tl_quant(layer_id, la_input, la_output, la_working, ifmt, ofmt,
                       scale, n, c, h, w, bExtraInput);
}
