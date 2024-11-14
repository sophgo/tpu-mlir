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
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynCompileCommon.hpp"
using namespace tpu_mlir::backend;

// ======================================
// GlobalGenInterface
// ======================================
void tpu::SortOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  sort_per_dim_param_t param = {0};
  param.buffer_addr = module::getAddress(getBuffer());
  param.axis = getAxis();
  param.descending = getDescending();
  param.is_argsort = module::isNone(getValues());
  BM168x::call_global_func("backend_api_sort_per_dim_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::SortOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(sort_per_dim_param_t);
  sort_per_dim_param_t param = {0};
  param.buffer_addr = module::getAddress(getBuffer());
  param.axis = getAxis();
  param.descending = getDescending();
  param.is_argsort = module::isNone(getValues());
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::SortOp::get_fw_type_bm1684x() { return FW_BMNET_SORT_PER_DIM; }
