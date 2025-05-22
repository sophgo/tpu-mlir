//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::FusedActiveCastOp::codegen_global_bm1684x() {
  active_requant_spec_t spec = {0};
  spec.active_spec.active_type = (int)getMode();
  if (getCoeffs().has_value()) {
    const auto coeffs_ = module::getF64Array(getCoeffs().value());
    for (int i = 0; i < coeffs_->size(); ++i) {
      spec.active_spec.coeffs[i] = (float)coeffs_->at(i);
    }
  }
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  bool qOutput = module::isUniformQuantized(getOutput());
  auto round_mode = round_mode_convert(getRoundMode());
  auto in_type = module::getStorageType(getInput());
  bool fInput = in_type.isIntOrIndex() == false;
  if (fInput && qOutput) {
    auto qtype = module::getUniformQuantizedType(getOutput());
    spec.requant_spec.is_perchannel = false;
    spec.requant_spec.scale_value = 1.0 / qtype.getScale();
    spec.requant_spec.offset_value = qtype.getZeroPoint();
    spec.requant_spec.output_dtype = BM168x::getDataType(getOutput());
    spec.requant_spec.mode = 0;
    spec.requant_spec.round_mode = round_mode;
    BM168x::call_global_func("backend_api_active_requant_global", &spec,
                             sizeof(spec), input_spec->data(),
                             output_spec->data());
  } else {
    UNREACHABLE_THIS("to be implemented");
  }
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::FusedActiveCastOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}

int64_t tpu::FusedActiveCastOp::get_fw_type_bm1684x() {
  return FW_LAYER_UNKNOWN;
}
