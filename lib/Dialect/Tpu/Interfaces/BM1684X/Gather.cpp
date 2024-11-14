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

void tpu::GatherOp::codegen_global_bm1684x() {
  auto op = getOperation();
  index_select_global_spec_t param{0};
  param.common.axis = getAxis();
  param.common.index_is_coeff = false;
  param.common.if_neg_index = getIfNegIndex();
  // assert(module::getStorageType(getIndices()).isInteger(32));
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_index_select_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::GatherOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(index_select_global_spec_t);
  index_select_global_spec_t param{0};
  param.common.axis = getAxis();
  param.common.index_is_coeff = false;
  param.common.if_neg_index = getIfNegIndex();
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::GatherOp::get_fw_type_bm1684x() { return FW_BMNET_INDEX_SELECT; }
