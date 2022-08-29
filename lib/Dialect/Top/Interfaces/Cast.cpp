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
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

int64_t top::CastOp::getFLOPs() { return Module::getNumElements(output()); }

LogicalResult top::CastOp::init(InferenceParameter &p) { return success(); }
void top::CastOp::deinit(InferenceParameter &p) {}

LogicalResult top::CastOp::inference(InferenceParameter &p) {
  llvm_unreachable("Not support now");
}
