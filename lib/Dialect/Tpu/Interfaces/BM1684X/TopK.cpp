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
void tpu::TopKOp::codegen_global_bm1684x() {
  // auto op = getOperation();
  // auto input_spec = BM168x::get_input_spec(op);
  // auto output_spec = BM168x::get_output_spec(op);
  // topk_spec_t spec = {0};
  // spec.k = getK();
  // spec.dim = getAxis();
  // spec.descending = getLargest();
  // BM168x::call_global_func("backend_api_topk_global", &spec, sizeof(spec),
  //                          input_spec->data(), output_spec->data());
  llvm_unreachable("Only support dynamic codegen");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::TopKOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) {
    return sizeof(topk_spec_t);
  }
  topk_spec_t spec = {0};
  spec.k = getK();
  spec.dim = getAxis();
  spec.descending = getLargest();
  spec.buffer_val_addr = module::getAddress(getBufferVal());
  spec.buffer_idx_addr = module::getAddress(getBufferIdx());
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::TopKOp::get_fw_type_bm1684x() { return FW_BMNET_TOPK; }
