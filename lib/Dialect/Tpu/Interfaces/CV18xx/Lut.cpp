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

void tpu::LutOp::codegen_global_cv18xx(int64_t layer_id) {

  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  gaddr_t ga_table = module::getAddress(getTable());
  cvi_backend_tg_lut_kernel(layer_id, ga_input, ga_output, ga_table, n, c, h, w,
                            CVK_FMT_I8);
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::LutOp::getBufferSize_cv18xx(int64_t in_lmem_bytes,
                                         int64_t out_lmem_bytes,
                                         int64_t in_nslice, int64_t in_hslice,
                                         int64_t out_nslice,
                                         int64_t out_hslice) {
  return 0;
}

void tpu::LutOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                      int64_t d_step, int64_t w_step,
                                      group_type_t group_type,
                                      local_sec_info_t &sec_info,
                                      int64_t layer_id) {
  int64_t n, c, h, w;
  auto shape = module::getShape(getInput());
  module::getNCHW(shape, n, c, h, w);

  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto out_gi = LocalGenInterface::getGroupInfo(getOutput(), n_step, h_step);
  auto table_gi = LocalGenInterface::getGroupInfo(getTable(), n_step, h_step);
  laddr_t la_input = in_gi.out_addr;
  laddr_t la_output = out_gi.out_addr;
  laddr_t la_y_table = table_gi.out_addr;

  n = sec_info.n_slice;
  h = sec_info.h_slice;

  cvi_backend_int8_tl_lut(layer_id, la_input, la_output, la_y_table, n, c, h,
                          w);
  return;
}
