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

void tpu::NonZeroOp::codegen_global_bm1684x() {
  llvm_unreachable("Only support dynamic codegen");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::NonZeroOp::dyn_codegen_global_bm1684x(void *buffer) {
  assert(getOrder().str() == "ColMajor");
  if (!buffer)
    return sizeof(where_spec_t);
  where_spec_t spec = {0};
  spec.buffer_addr = module::getAddress(getBuffer());
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::NonZeroOp::get_fw_type_bm1684x() { return FW_BMNET_WHERE; }
