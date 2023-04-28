//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::GridSamplerOp::codegen_global_bm1684x() {
  grid_sample_global_param_t param = {0};
  memset(&param, 0, sizeof(param));
  param.input_addr = module::getAddress(getInput());
  param.grid_addr = module::getAddress(getGrid());
  param.output_addr = module::getAddress(getOutput());

  param.align_corners = getAlignCorners();
  param.interp_mode = GridSampleNearest;
  int padding_mode_int = getPaddingMode();
  GridSamplePaddingMode padding_mode;
  switch (padding_mode_int) {
  case 0:
    padding_mode = GridSampleZeros;
    break;
  case 1:
    padding_mode = GridSampleBorder;
    break;
  case 2:
    padding_mode = GridSampleReflection;
    break;
  default:
    llvm_unreachable("not implemented.");
    break;
  }

  param.padding_mode = padding_mode;

  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);

  param.input_n = n;
  param.input_c = c;
  param.input_h = h;
  param.input_w = w;
  auto out_shape = module::getShape(getOutput());
  param.output_h = out_shape[2];
  param.output_w = out_shape[3];
  param.dtype = BM168x::getDataType(getInput());
  BM168x::call_global_func("backend_api_grid_sample_global", &param,
                           sizeof(grid_sample_global_param_t));
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::GridSamplerOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}

// ======================================
// Dynamic LocalGenInterface
// ======================================

int64_t tpu::GridSamplerOp::get_fw_type_bm1684x() { return FW_LAYER_UNKNOWN; }
