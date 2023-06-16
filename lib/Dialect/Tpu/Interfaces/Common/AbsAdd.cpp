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
#include "tpu_mlir/Support/MathUtils.h"

// LogicalResult tpu::AbsAddOp::init(InferenceParameter &p) { return success(); }
// void tpu::AbsAddOp::deinit(InferenceParameter &p) {}

// LogicalResult tpu::AbsAddOp::inference(InferenceParameter &p) {
//   return success();
// }
