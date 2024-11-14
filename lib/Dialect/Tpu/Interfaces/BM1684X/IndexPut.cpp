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

void tpu::IndexPutOp::codegen_global_bm1684x() {
  auto op = getOperation();
  index_put_spec_t param{0};
  param.mode = 1;
  param.accumulate = getAccumulate();
  param.buffer_addr = module::getAddress(getBuffer());
  // assert(module::getStorageType(getIndices()).isInteger(32));
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_index_put_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::IndexPutOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(index_put_spec_t);
  index_put_spec_t param{0};
  param.mode = 1;
  param.accumulate = getAccumulate();
  param.buffer_addr = module::getAddress(getBuffer());
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::IndexPutOp::get_fw_type_bm1684x() { return FW_BMNET_INDEX_PUT; }
