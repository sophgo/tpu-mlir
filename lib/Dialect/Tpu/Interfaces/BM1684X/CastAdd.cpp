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

// int8
void tpu::CastAddOp::codegen_global_bm1684x() {
  int qInput_0 = module::isUniformQuantized(getInputs()[0]);
  int qInput_1 = module::isUniformQuantized(getInputs()[1]);
  assert(qInput_0 + qInput_1 == 1);
  bool qInput = qInput_0 || qInput_1;

  auto out_type = module::getStorageType(getOutput());
  bool fOutput = out_type.isIntOrIndex() == false;

  auto round_mode = round_mode_convert(getRoundMode());

  if (qInput && fOutput) {
    auto qtype = qInput_0 ? module::getUniformQuantizedType(getInputs()[0])
                          : module::getUniformQuantizedType(getInputs()[1]);

    cast_add_param_t param = {0};
    param.is_perchannel = false;
    param.scale_value = qtype.getScale();
    param.offset_value = qtype.getZeroPoint();
    param.round_mode = round_mode;
    // Add
    param.binary_type = BINARY_ADD;
    param.if_relu = getDoRelu();
    param.relu_upper_limit = getReluLimit().convertToDouble();
    auto op = getOperation();
    auto input_spec = BM168x::get_input_spec(op);
    auto output_spec = BM168x::get_output_spec(op);
    BM168x::call_global_func("backend_api_cast_add_global", &param,
                             sizeof(param), input_spec->data(),
                             output_spec->data());
  } else {
    assert(0);
  }
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::CastAddOp::dyn_codegen_global_bm1684x(void *buffer) { return 0; }

int64_t tpu::CastAddOp::get_fw_type_bm1684x() { return FW_LAYER_UNKNOWN; }
