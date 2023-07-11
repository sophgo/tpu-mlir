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
void tpu::LRNOp::codegen_global_cv18xx(int64_t layer_id) {
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t exp_gaddr = module::getAddress(getTable());
  gaddr_t mantissa_gaddr = module::getAddress(getMantissa());
  gaddr_t ga_output = module::getAddress(getOutput());
  int64_t n, c, h, w;
  int64_t local_size = getSize();
  double alpha = this->getAlpha().convertToDouble();
  double k = this->getBias().convertToDouble();

  module::getNCHW(this->getInput(), n, c, h, w);
  if (module::isUniformQuantized(getOutput())) {
    llvm_unreachable("Not supported now");
  } else {
    cvi_backend_tg_bf16_lrn_kernel(layer_id, ga_input, ga_output, exp_gaddr,
                                   mantissa_gaddr, n, c, h, w, local_size,
                                   alpha, k);
  }
}

int64_t tpu::LRNOp::getBufferSize_cv18xx(int64_t in_lmem_bytes,
                                         int64_t out_lmem_bytes,
                                         int64_t in_nslice, int64_t in_hslice,
                                         int64_t out_nslice,
                                         int64_t out_hslice) {
  int64_t n, c, h, w;
  auto vIn = getInput();
  module::getNCHW(vIn, n, c, h, w);
  n = in_nslice;
  h = in_hslice;
  auto fmt = CV18xx::getDataType(vIn);
  return CV18xx::lmem_woring_size({n, c, h, w}, 2, true, fmt);
}

void tpu::LRNOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                      int64_t d_step, int64_t w_step,
                                      group_type_t group_type,
                                      local_sec_info_t &sec_info,
                                      int64_t layer_id) {
  if (module::isUniformQuantized(getOutput())) {
    llvm_unreachable("Not supported now");
  }
  int64_t n, c, h, w;
  auto shape = module::getShape(getInput());
  module::getNCHW(shape, n, c, h, w);

  auto gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto out_gi = LocalGenInterface::getGroupInfo(getOutput(), n_step, h_step);
  auto table_gi = LocalGenInterface::getGroupInfo(getTable(), n_step, h_step);
  auto mantissa_gi =
      LocalGenInterface::getGroupInfo(getMantissa(), n_step, h_step);
  laddr_t la_input = in_gi.out_addr;
  laddr_t la_output = out_gi.out_addr;
  laddr_t la_working = gi.buffer_addr;
  laddr_t la_table = table_gi.out_addr;
  laddr_t la_mantissa = mantissa_gi.out_addr;

  n = sec_info.n_slice;
  h = sec_info.h_slice;

  int local_size = getSize();
  float alpha = getAlpha().convertToDouble();
  float k = getBias().convertToDouble();

  cvi_backend_bf16_tl_lrn(layer_id, la_input, la_output, la_table, la_mantissa,
                          la_working, n, c, h, w, local_size, alpha, k);
  return;
}
