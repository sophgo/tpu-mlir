//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"



int64_t top::CastOp::getFLOPs() { return module::getNumElements(output()); }

LogicalResult top::CastOp::init(InferenceParameter &p) { return success(); }
void top::CastOp::deinit(InferenceParameter &p) {}

LogicalResult top::CastOp::inference(InferenceParameter &p) {
  llvm_unreachable("Not Implemented");
}
