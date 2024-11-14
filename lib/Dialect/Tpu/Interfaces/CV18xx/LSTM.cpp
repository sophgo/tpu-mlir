//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================
void tpu::LSTMOp::codegen_global_cv18xx(int64_t layer_id) {
  auto attr = parseParam();
  gaddr_t ga_bias = GA_INVALID;
  gaddr_t ga_initial_h = GA_INVALID;
  gaddr_t ga_initial_c = GA_INVALID;
  gaddr_t ga_cont = GA_INVALID;
  gaddr_t ga_last_h = GA_INVALID;
  gaddr_t ga_last_c = GA_INVALID;
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_recurrence = module::getAddress(getRecurrence());
  gaddr_t ga_sigmoid_table = module::getAddress(getSigmoidTable());
  gaddr_t ga_sigmoid_slope = module::getAddress(getSigmoidSlopeTable());
  gaddr_t ga_tanh_table = module::getAddress(getTanhTable());
  gaddr_t ga_tanh_slope = module::getAddress(getTanhSlopeTable());
  gaddr_t ga_output = module::getAddress(getResults()[0]);
  auto is_torch = module::isPlatform(module::Platform::TORCH);
  bool output_y = true;
  if (getResults()[0].getType().isa<mlir::NoneType>()) {
    output_y = false;
  }

  if (attr.have_bias) {
    ga_bias = module::getAddress(getBias());
  }
  if (attr.have_h0) {
    ga_initial_h = module::getAddress(getInitialH());
  }
  if (attr.have_c0) {
    ga_initial_c = module::getAddress(getInitialC());
  }
  if (attr.output_yh) {
    ga_last_h = module::getAddress(getResults()[1]);
  }
  if (attr.output_yc) {
    ga_last_c = module::getAddress(getResults()[2]);
  }
  if (attr.have_cont) {
    ga_cont = module::getAddress(getCont());
  }
  cvi_backend_tg_bf16_lstm_kernel(
      layer_id, ga_input, ga_recurrence, ga_bias, ga_initial_h, ga_initial_c,
      ga_cont, ga_sigmoid_table, ga_sigmoid_slope, ga_tanh_table, ga_tanh_slope,
      ga_output, ga_last_h, ga_last_c, attr.seq_len, attr.num_direction,
      attr.batch_size, attr.hidden_size, attr.have_bias, attr.have_h0,
      attr.have_c0, attr.have_cont, getBidirectional(), attr.output_yh,
      attr.output_yc, output_y, is_torch);
}
