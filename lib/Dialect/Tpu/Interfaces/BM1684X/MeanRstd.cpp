//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::MeanRstdOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  mean_rstd_param_t param = {0};
  param.eps = getEps().convertToDouble();
  param.momentum = getMomentum().convertToDouble();
  llvm::errs() << "call backend_api_mean_rstd_global\n";
  BM168x::call_global_func("backend_api_mean_rstd_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

void tpu::MeanRstdOp::codegen_global_bm1684() {}

void tpu::MeanRstdOp::codegen_global_cv18xx(int64_t layer_id) {}
