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

using namespace tpu_mlir::backend;

void tpu::Batch2SpaceOp::codegen_global_bm1684x() {
  UNREACHABLE_THIS("Batch2SpaceOp type error");
}

int64_t tpu::Batch2SpaceOp::dyn_codegen_global_bm1684x(void *buffer) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
int64_t tpu::Batch2SpaceOp::get_fw_type_bm1684x() { return -1; }
