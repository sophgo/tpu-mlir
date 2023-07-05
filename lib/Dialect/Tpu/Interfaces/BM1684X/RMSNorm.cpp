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

void tpu::RMSNormOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  rms_norm_global_spec_t param = {0};
  const bool have_gamma = !getGamma().getType().isa<NoneType>();

  param.common.eps = getEps().convertToDouble();
  param.common.affine = have_gamma;
  BM168x::call_global_func("backend_api_rms_norm_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::RMSNormOp::dyn_codegen_global_bm1684x(void *buffer) {
  llvm_unreachable("to be implemented");
}

int64_t tpu::RMSNormOp::get_fw_type_bm1684x() { return FW_LAYER_UNKNOWN; }
