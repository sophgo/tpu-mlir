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

void tpu::ScatterElementsOp::codegen_global_bm1684x() {
  // auto op = getOperation();
  // auto input_spec = BM168x::get_input_spec(op);
  // auto output_spec = BM168x::get_output_spec(op);
  // scatter_elements_global_spec_t param = {0};
  // param.common.axis = getAxis();
  // BM168x::call_global_func("backend_api_scatter_elements_global", &param,
  //                          sizeof(param), input_spec->data(),
  //                          output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ScatterElementsOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}

int64_t tpu::ScatterElementsOp::get_fw_type_bm1684x() {
  llvm_unreachable("Not Implemented");
  return FW_LAYER_UNKNOWN;
}
