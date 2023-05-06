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
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::ConstantFillOp::codegen_global_bm1684x() {
  llvm_unreachable("Not supported now");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================

int64_t tpu::ConstantFillOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) return sizeof(constantfill_common_spec_t);
  constantfill_common_spec_t spec = {0};
  float value = getValue().convertToDouble();
  spec.filled_value = *(uint32_t*)&value;
  spec.dtype = DTYPE_FP32;
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::ConstantFillOp::get_fw_type_bm1684x() {
  return FW_BMNET_CONSTANT_FILL;
}
