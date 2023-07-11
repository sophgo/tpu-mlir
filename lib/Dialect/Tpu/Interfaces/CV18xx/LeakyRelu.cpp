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

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::LeakyReluOp::codegen_global_cv18xx(int64_t layer_id) {

  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  if (module::isUniformQuantized(getOutput())) {
    auto pos_rshift = this->getRshift().value();
    auto pos_m = this->getMultiplier().value();
    auto neg_rshift = this->getRshiftNeg().value();
    auto neg_m = this->getMultiplierNeg().value();
    cvi_backend_tg_fixed_leakyrelu_kernel(layer_id, ga_input, ga_output, n, c,
                                          h, w, pos_rshift, neg_rshift, pos_m,
                                          neg_m);
  } else {
    float negative_slope =
        static_cast<float>(getAlpha().value().convertToDouble());
    cvi_backend_tg_bf16_leakyrelu_kernel(layer_id, ga_input, ga_output,
                                         negative_slope, n, c, h, w);
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::LeakyReluOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  if (module::isUniformQuantized(getOutput())) {
    return 0;
  }
  int64_t n, c, h, w;
  auto vIn = getInput();
  module::getNCHW(vIn, n, c, h, w);
  n = in_nslice;
  h = in_hslice;
  auto fmt = CV18xx::getDataType(vIn);
  return CV18xx::lmem_woring_size({n, c, h, w}, 1, true, fmt);
}

void tpu::LeakyReluOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
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

  if (module::isUniformQuantized(getOutput())) {
    int8_t pos_rshift = getRshift().value();
    int8_t pos_m_i8 = getMultiplier().value();
    int8_t neg_rshift = getRshiftNeg().value();
    int8_t neg_m_i8 = getMultiplierNeg().value();
    cvi_backend_tl_leaky_relu(layer_id, // layer_id,
                              la_input, la_output, n, c, h, w, pos_rshift,
                              neg_rshift, pos_m_i8, neg_m_i8);
  } else {
    float neg_slope = static_cast<float>(getAlpha().value().convertToDouble());
    cvi_backend_bf16_tl_leaky_relu(layer_id, // layer_id,
                                   la_input, la_output, la_working, n, c, h, w,
                                   neg_slope);
  }
}
