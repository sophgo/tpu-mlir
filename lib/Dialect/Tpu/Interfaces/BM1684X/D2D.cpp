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

// int8
void tpu::D2DOp::codegen_global_bm1684x() {
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  auto input_type = module::getStorageType(getInput());
  constbinary_global_spec_t param = {0};
  param.common.binary_type = BINARY_ADD;
  param.common.B_const_val = 0.0f;
  param.common.inversed = 0;
  param.common.scale_A = 1;
  param.common.rshift_A = 0;
  param.common.B_dtype = input_type.isa<FloatType>() ? DTYPE_FP32 : DTYPE_INT32;

  BM168x::call_global_func("backend_api_constbinary_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::D2DOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(constbinary_global_spec_t);
  auto input_type = module::getStorageType(getInput());
  constbinary_global_spec_t param = {0};
  param.common.binary_type = BINARY_ADD;
  param.common.B_const_val = 0.0f;
  param.common.inversed = 0;
  param.common.scale_A = 1;
  param.common.rshift_A = 0;
  param.common.B_dtype = input_type.isa<FloatType>() ? DTYPE_FP32 : DTYPE_INT32;

  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::D2DOp::get_fw_type_bm1684x() { return FW_BMNET_CONST_BINARY; }
