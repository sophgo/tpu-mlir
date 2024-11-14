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

void tpu::GatherNDOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  auto batch_dims = getBatchDims();
  if (batch_dims != 0) {
    UNREACHABLE_THIS("Not Implemented");
  }
  gather_nd_global_param_t param;
  param.batch_dims = batch_dims;
  param.const_val = 0; // no use temporary
  BM168x::call_global_func("backend_api_gather_nd_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::GatherNDOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) {
    return sizeof(gather_nd_global_param_t);
  }
  auto batch_dims = getBatchDims();
  if (batch_dims != 0) {
    UNREACHABLE_THIS("Not Implemented");
  }
  gather_nd_global_param_t param;
  param.batch_dims = batch_dims;
  param.const_val = 0; // no use temporary
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::GatherNDOp::get_fw_type_bm1684x() { return FW_BMNET_GATHERND; }
