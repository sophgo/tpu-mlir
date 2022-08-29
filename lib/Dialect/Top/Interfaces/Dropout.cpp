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

int64_t top::DropoutOp::getFLOPs() {
  return Module::getNumElements(output());
}

LogicalResult top::DropoutOp::init(InferenceParameter &p) {
  return success();
}
void top::DropoutOp::deinit(InferenceParameter &p) {}

LogicalResult top::DropoutOp::inference(InferenceParameter &p) {
  const float *src = p.inputs[0];
  float *dst = p.outputs[0];
  int64_t num_elements = Module::getNumElements(input());
  memcpy(dst, src, sizeof(float) * num_elements);
  return success();
}
