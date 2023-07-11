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

void tpu::LutBF16Op::codegen_global_cv18xx(int64_t layer_id) {

  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  gaddr_t ga_table = module::getAddress(getTable());

  gaddr_t ga_mantissa = module::getAddress(getMantissa());
  auto _lut_mode = getLutMode();
  if (_lut_mode == LutBF16Mode::Slope) {
    cvi_backend_tg_bf16_lut_slope_kernel(
        layer_id, ga_input, ga_output, ga_table, ga_mantissa, n, c, h, w,
        getMinRange().convertToDouble(), getMaxRange().convertToDouble());
  } else if (_lut_mode == LutBF16Mode::Mantissa) {
    cvi_backend_tg_bf16_lut_mantissa_kernel(
        layer_id, ga_input, ga_output, ga_table, ga_mantissa, n, c, h, w, 0);
  } else if (_lut_mode == LutBF16Mode::Log) {
    cvi_backend_tg_bf16_lut_mantissa_kernel(
        layer_id, ga_input, ga_output, ga_table, ga_mantissa, n, c, h, w, 1);
  } else {
    llvm_unreachable("Not supported now!");
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::LutBF16Op::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  int64_t n, c, h, w;
  auto vIn = getInput();
  module::getNCHW(getInput(), n, c, h, w);
  n = in_nslice;
  h = in_hslice;
  auto fmt = CV18xx::getDataType(vIn);
  return CV18xx::lmem_woring_size({n, c, h, w}, 2, true, fmt);
}

void tpu::LutBF16Op::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                          int64_t d_step, int64_t w_step,
                                          group_type_t group_type,
                                          local_sec_info_t &sec_info,
                                          int64_t layer_id) {
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
  laddr_t la_y_table = table_gi.out_addr;
  laddr_t la_y_mantissa = mantissa_gi.out_addr;
  laddr_t la_working = gi.buffer_addr;

  n = sec_info.n_slice;
  h = sec_info.h_slice;

  int method = 0;
  if (getLutMode() == LutBF16Mode::Mantissa) {
    method = 0;
  } else if (getLutMode() == LutBF16Mode::Log) {
    method = 1;
  } else if (getLutMode() == LutBF16Mode::Slope) {
    method = 2;
  }

  float table_thresh_min = getMinRange().convertToDouble();
  float table_thresh_max = getMaxRange().convertToDouble();

  cvi_backend_bf16_tl_lut(layer_id, la_input, la_output, la_working, la_y_table,
                          la_y_mantissa, table_thresh_min, table_thresh_max, n,
                          c, h, w, method);
  return;
}
