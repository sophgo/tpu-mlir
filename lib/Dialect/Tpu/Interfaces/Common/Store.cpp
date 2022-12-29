//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"

#include "tpu_mlir/Support/Module.h"



LogicalResult tpu::StoreOp::init(InferenceParameter &p) { return success(); }
void tpu::StoreOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::StoreOp::inference(InferenceParameter &p) {
  llvm_unreachable("Not Implemented");
  return success();
}
