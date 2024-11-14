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
void tpu::ActiveOp::codegen_global_cv18xx(int64_t layer_id) {
  int input_num = 1;
  gaddr_t input = module::getAddress(this->getInput());
  gaddr_t ga_inputs[] = {input};
  int64_t n, c, h, w;
  module::getNCHW(this->getInput(), n, c, h, w);
  gaddr_t ga_output = module::getAddress(getOutput());
  bool do_relu = false;
  bool do_early_stride = false;
  int early_stride_h = 0;
  int early_stride_w = 0;
  switch (getMode()) {
  case ActiveMode::ABSVAL: {
    if (module::isUniformQuantized(getOutput())) {
      cvi_backend_tg_eltwise_abs_kernel(
          layer_id, ga_inputs, ga_output, input_num, n, c, h, w, do_relu,
          do_early_stride, early_stride_h, early_stride_w, 0, NULL, NULL,
          CVK_FMT_I8);
    } else {
      cvi_backend_tg_eltwise_abs_kernel(
          layer_id, ga_inputs, ga_output, input_num, n, c, h, w, do_relu,
          do_early_stride, early_stride_h, early_stride_w, 0, NULL, NULL,
          CVK_FMT_BF16);
    }
  } break;
  default:
    break;
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::ActiveOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::ActiveOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                         int64_t d_step, int64_t w_step,
                                         group_type_t group_type,
                                         local_sec_info_t &sec_info,
                                         int64_t layer_id) {
  if (getMode() != ActiveMode::ABSVAL) {
    return;
  }
  int64_t n, c, h, w;
  auto shape = module::getShape(getInput());
  module::getNCHW(shape, n, c, h, w);

  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto out_gi = LocalGenInterface::getGroupInfo(getOutput(), n_step, h_step);
  std::vector<laddr_t> la_input(1);
  la_input[0] = in_gi.out_addr;
  laddr_t la_output = out_gi.out_addr;
  n = sec_info.n_slice;
  h = sec_info.h_slice;
  int op_code = 3; // abs
  int nInputs = 1;
  if (module::isUniformQuantized(getOutput())) {
    cvi_backend_tl_eltwise(
        layer_id, la_input.data(), la_output, -1, /*la_working*/
        n, c, h, w, nInputs, op_code, 0 /*rshift*/, 0 /*m_i8_input*/,
        0 /*use_default_coeff*/, 0 /*do_relu*/, 0 /*relu_slope*/,
        NULL /*coeffs*/, 0, 0, 0,
        0 /*do_early_stride, early_stride_h, early_stride_w*/);
  } else {
    cvi_backend_bf16_tl_eltwise(layer_id, la_input.data(), la_output, n, c, h,
                                w, nInputs, op_code, 0 /*use_default_coeff*/,
                                0 /*do_relu*/, 0 /*relu_slope*/,
                                NULL /*coeffs*/, 0 /*do_early_stride*/, 0,
                                0 /*early_stride_h, early_stride_w*/);
  }
  // llvm_unreachable("Not supported now");
}
