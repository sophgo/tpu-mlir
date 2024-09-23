//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"

namespace tpu_mlir {
namespace tpu {

void doCoreParallelPattern(ModuleOp module);

void doSpecificPattern(ModuleOp module);

} // namespace tpu
} // namespace tpu_mlir
