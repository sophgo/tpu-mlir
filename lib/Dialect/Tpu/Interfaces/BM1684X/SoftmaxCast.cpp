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
// GloballGenInterface
// =========================================
void tpu::SoftmaxCastOp::codegen_global_bm1684x() {
  auto in_type = module::getStorageType(getInput());

  // Softmax + Requant: bf16->i8
  bool fInput = in_type.isIntOrIndex() == false;
  bool qOutput = module::isUniformQuantized(getOutput());

  auto round_mode = round_mode_convert(getRoundMode());

  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  if (fInput && qOutput) {
    softmax_cast_global_spec_t param = {0};
    // Softmax
    param.common_softmax.begin_axis = getAxis();
    param.common_softmax.end_axis = getAxis();
    param.common_softmax.scale_val = 1.0;
    param.common_softmax.log = getLog();

    // requant
    auto qtype = module::getUniformQuantizedType(getOutput());
    param.common_dequant.scale_value = 1.0 / qtype.getScale();
    param.common_dequant.offset_value = qtype.getZeroPoint();
    param.common_dequant.round_mode = round_mode;
    BM168x::call_global_func("backend_api_softmax_cast_global", &param,
                             sizeof(param), input_spec->data(),
                             output_spec->data());
  } else {
    UNREACHABLE_THIS("Not Implemented");
  }
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::SoftmaxCastOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}

int64_t tpu::SoftmaxCastOp::get_fw_type_bm1684x() { return FW_LAYER_UNKNOWN; }
