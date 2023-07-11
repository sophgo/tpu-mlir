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
void tpu::MatchTemplateOp::codegen_global_cv18xx(int64_t layer_id) {
  auto input_shape = module::getShape(getInput());
  auto template_shape = module::getShape(getMatch());
  auto mode = getMode().str();
  assert(input_shape.size() == 2);
  assert(template_shape.size() == 2);
  int ih = input_shape[0];
  int iw = input_shape[1];
  int th = template_shape[0];
  int tw = template_shape[1];
  assert(ih >= th);
  assert(iw >= tw);

  gaddr_t input_gaddr = module::getAddress(getInput());
  gaddr_t template_gaddr = module::getAddress(getMatch());
  gaddr_t output_gaddr = module::getAddress(getOutput());
  gaddr_t ga_table = module::getAddress(getTable());
  gaddr_t ga_mantissa_table = module::getAddress(getMantissaTable());
  cvi_backend_tg_bf16_match_template_kernel(
      layer_id, input_gaddr, template_gaddr, ga_table, ga_mantissa_table,
      output_gaddr, ih, iw, th, tw, mode.c_str());
}
