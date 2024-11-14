//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================
void tpu::GRUOp::codegen_global_cv18xx(int64_t layer_id) {
  auto attr = parseParam();

  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output_y = attr.output_y ? module::getAddress(getY()) : 0;
  gaddr_t ga_output_yh = attr.output_yh ? module::getAddress(getYH()) : 0;
  gaddr_t ga_bias = attr.have_bias ? module::getAddress(getBias()) : 0;
  gaddr_t ga_initial_h = attr.have_h0 ? module::getAddress(getInitialH()) : 0;
  gaddr_t ga_recurrence = module::getAddress(getRecurrence());
  gaddr_t ga_sigmoid_table = module::getAddress(getSigmoidTable());
  gaddr_t ga_sigmoid_slope = module::getAddress(getSigmoidSlopeTable());
  gaddr_t ga_tanh_table = module::getAddress(getTanhTable());
  gaddr_t ga_tanh_slope = module::getAddress(getTanhSlopeTable());
  auto is_torch = module::isPlatform(module::Platform::TORCH);

  bool is_linear_before_reset = getLinearBeforeReset();
  bool is_bidirectional = getBidirectional();

  if (module::isUniformQuantized(getInput())) {
    llvm_unreachable("Not supported now");
  } else {
    cvi_backend_tg_bf16_gru_kernel(
        layer_id, ga_input, ga_recurrence, ga_bias, ga_initial_h,
        ga_sigmoid_table, ga_sigmoid_slope, ga_tanh_table, ga_tanh_slope,
        ga_output_y, ga_output_yh, attr.seq_len, attr.num_direction,
        attr.batch_size, attr.hidden_size, attr.have_bias, attr.have_h0,
        is_linear_before_reset, is_bidirectional, attr.output_y, attr.output_yh,
        is_torch);
  }
}
