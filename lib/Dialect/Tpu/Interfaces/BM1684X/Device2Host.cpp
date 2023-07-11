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

void tpu::Device2HostOp::codegen_global_bm1684x() {
  llvm_unreachable("Not supported now");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::Device2HostOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (buffer) {
    int flag = 0;
    return BM168x::dynamic_spec_to_buffer(buffer, flag);
  }
  return sizeof(int);
}

int64_t tpu::Device2HostOp::get_fw_type_bm1684x() {
  return FW_BMNET_DEVICE2HOST;
}
