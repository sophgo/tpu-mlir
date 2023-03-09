//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/DynCompileCommon.hpp"
using namespace tpu_mlir::backend;

void tpu::ShuffleChannelOp::codegen_global_bm1684x() {
  llvm_unreachable("Not Implemented");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ShuffleChannelOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}

int64_t tpu::ShuffleChannelOp::get_layer_type() {
  return FW_LAYER_UNKNOWN;
}
