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

void tpu::CopyOp::codegen_global_bm1684x() {
  UNREACHABLE_THIS("Not Implemented");
}

int64_t tpu::CopyOp::dyn_codegen_global_bm1684x(void *buffer) { return 0; }

int64_t tpu::CopyOp::get_fw_type_bm1684x() { return FW_LAYER_UNKNOWN; }
