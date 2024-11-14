//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

void tpu::ShapeUnsqueezeOp::codegen_global_bm1684() {
  llvm_unreachable("Not supported now");
}

uint32_t tpu::ShapeUnsqueezeOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::ShapeUnsqueezeOp::get_fw_type_bm1684() { return -1; }
